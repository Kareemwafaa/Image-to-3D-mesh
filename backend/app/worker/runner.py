"""The GPU worker: consumes job messages and runs the pipeline.

Per message:
  1. Load the job row. Missing or already final -> ack and skip (a duplicate).
  2. Guarded transition to `running`, attempts += 1. Lost the race -> skip.
  3. Run the pipeline in a thread while an asyncio task renews the lease.
     Progress ticks update the row (throttled) and publish events; each tick
     also checks the job's deadline (cooperative timeout).
  4. Upload outputs, mark `succeeded`, ack.
  5. On failure, classify: permanent -> `failed`, ack. Transient with attempts
     left -> back to `queued`, retry with exponential backoff. Out of attempts
     -> `failed` + dead letter.

Background loops on every worker: promote due retries, reclaim expired leases
(dead workers), reconcile jobs stuck in `queued` (lost enqueue), heartbeats,
queue-depth metrics, and expiry of old jobs.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import structlog

from ..backends.base import GenerateOptions, MeshBackend
from ..container import Container
from ..core import metrics
from ..core.errors import JobError, JobTimeout, classify
from ..core.logging import get_logger
from ..core.tracing import StageSpans, continued_span, span
from ..jobqueue.base import Message, backoff_delay
from .pipeline import RESULT_FILE, VIEWS_FILE, run_pipeline

log = get_logger(__name__)


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:4]}"


class Worker:
    def __init__(self, c: Container, backend: MeshBackend, worker_id: Optional[str] = None) -> None:
        self.c = c
        self.s = c.settings
        self.backend = backend
        self.worker_id = worker_id or self.s.worker_id or default_worker_id()
        self.model_loaded = False
        self.model_error: Optional[str] = None
        self.current_job: Optional[str] = None
        self.jobs_done = 0
        self._stop = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._idle = asyncio.Event()
        self._idle.set()

    # --- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        """Finish the current job, then return from run(). Safe to call from any thread."""
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                same_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                same_loop = False
            if not same_loop:
                # asyncio.Event is not thread-safe: setting it from another
                # thread would not wake the tasks waiting on it.
                loop.call_soon_threadsafe(self._stop.set)
                return
        self._stop.set()

    async def run(self) -> None:
        """Run until stop(); then finish the current job and return."""
        self._loop = asyncio.get_running_loop()
        log.info("worker_starting", worker_id=self.worker_id, backend=getattr(self.backend, "name", "?"))
        maintenance = asyncio.create_task(self._maintenance_loop(), name="worker-maintenance")
        heartbeats = asyncio.create_task(self._heartbeat_loop(), name="worker-heartbeat")
        try:
            await self._load_model()
            await self._consume_loop()
        finally:
            for t in (maintenance, heartbeats):
                t.cancel()
            await asyncio.gather(maintenance, heartbeats, return_exceptions=True)
            log.info("worker_stopped", worker_id=self.worker_id, jobs_done=self.jobs_done)

    async def _load_model(self) -> None:
        delay = 5.0
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.backend.load)
                self.model_loaded, self.model_error = True, None
                log.info("model_loaded", worker_id=self.worker_id)
                await self._beat()
                return
            except Exception as exc:  # keep reporting the error, retry with backoff
                self.model_error = f"{type(exc).__name__}: {exc}"
                log.exception("model_load_failed", retry_in_s=delay)
                await self._beat()
                try:
                    await asyncio.wait_for(self._stop.wait(), delay)
                except TimeoutError:
                    pass
                delay = min(delay * 2, 300)

    async def _consume_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = await self.c.queue.receive(self.worker_id, block_s=2.0)
            except Exception:
                log.exception("receive_failed")
                await asyncio.sleep(1.0)
                continue
            if msg is None:
                continue
            try:
                await self.handle(msg)
            except Exception:
                # Never let one message take the worker down. It stays unacked,
                # so its lease expires and it is redelivered (and dead-lettered
                # by the poison check below if it keeps failing like this).
                log.exception("handle_failed", job_id=msg.job_id)

    # --- one message -------------------------------------------------------
    async def handle(self, msg: Message) -> None:
        q, repo = self.c.queue, self.c.jobs
        try:
            job_id = uuid.UUID(msg.job_id)
        except ValueError:
            log.warning("malformed_message", delivery_id=msg.delivery_id)
            await q.ack(msg)
            return
        job = await repo.get(job_id)
        if job is None or job.status in ("succeeded", "failed"):
            log.info("duplicate_or_orphan_message_skipped", job_id=msg.job_id, status=getattr(job, "status", None))
            await q.ack(msg)
            return
        if job.attempts >= job.max_attempts:
            # Every allowed attempt was started and none recorded an outcome: the
            # job keeps killing its worker (a poison message). Stop redelivering it.
            await self._dead_letter_poison(msg, job_id, job.attempts)
            return
        attempt = await repo.start_attempt(job_id, self.worker_id)
        if attempt is None:
            await q.ack(msg)
            return

        structlog.contextvars.bind_contextvars(job_id=msg.job_id, attempt=attempt, worker_id=self.worker_id)
        self.current_job, self._idle = msg.job_id, asyncio.Event()
        metrics.JOBS_IN_PROGRESS.inc()
        lease = asyncio.create_task(self._renew_lease(msg))
        started = time.monotonic()
        status = "failed"
        try:
            with continued_span("job.run", msg.traceparent, job_id=msg.job_id, attempt=attempt):
                await self.c.bus.publish(msg.job_id, {"type": "status", "status": "running", "attempt": attempt})
                options = GenerateOptions.from_dict(job.options or {})
                result_key, views_key = await self._execute(msg.job_id, job.input_key, options)
                if await repo.succeed(job_id, result_key, views_key):
                    status = "succeeded"
                    await self.c.bus.publish(msg.job_id, {"type": "done", "status": "succeeded"})
                    metrics.JOBS_FINISHED.labels(status="succeeded", error_code="").inc()
                    log.info("job_succeeded", duration_s=round(time.monotonic() - started, 2))
                await q.ack(msg)
        except Exception as exc:
            await self._on_failure(msg, job_id, attempt, job.max_attempts, exc)
        finally:
            lease.cancel()
            metrics.JOBS_IN_PROGRESS.dec()
            metrics.JOB_DURATION.labels(status=status).observe(time.monotonic() - started)
            self.jobs_done += 1
            self.current_job = None
            self._idle.set()
            structlog.contextvars.unbind_contextvars("job_id", "attempt", "worker_id")
            self._record_gpu_memory()

    async def _execute(self, job_id: str, input_key: str, options: GenerateOptions) -> tuple[str, Optional[str]]:
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + self.s.job_timeout_s
        state = {"stage": None, "stage_started": time.monotonic(), "db_at": 0.0, "progress": 0.0}
        stage_spans = StageSpans()  # one child span of job.run per pipeline stage

        async def report(progress: float, stage: Optional[str], force: bool) -> None:
            try:
                await self.c.bus.publish(job_id, {"type": "progress", "progress": round(progress, 3), "stage": stage})
                now = time.monotonic()
                if force or now - state["db_at"] >= self.s.progress_db_interval_s:
                    state["db_at"] = now
                    await self.c.jobs.set_progress(uuid.UUID(job_id), progress, stage)
            except Exception:  # progress is best effort; never fail the job over it
                log.warning("progress_report_failed", exc_info=True)

        def on_progress(value: float, stage: Optional[str] = None) -> None:
            # Runs in the pipeline thread.
            if time.monotonic() > deadline:
                raise JobTimeout(f"Job exceeded {self.s.job_timeout_s:.0f}s")
            stage_changed = bool(stage and stage != state["stage"])
            if stage_changed:
                if state["stage"]:
                    metrics.STAGE_DURATION.labels(stage=state["stage"]).observe(
                        time.monotonic() - state["stage_started"]
                    )
                state["stage"], state["stage_started"] = stage, time.monotonic()
                stage_spans.enter(stage)
            state["progress"] = max(state["progress"], min(float(value), 0.99))
            asyncio.run_coroutine_threadsafe(report(state["progress"], state["stage"], stage_changed), loop)

        with tempfile.TemporaryDirectory(prefix=f"job-{job_id[:8]}-") as tmp:
            work = Path(tmp)
            input_path = work / "input.png"
            with span("storage.download", key=input_key):
                await self.c.storage.download(input_key, input_path)
            backstop = self.s.job_timeout_s + 60  # if the pipeline ignores progress ticks
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(run_pipeline, self.backend, input_path, work, options, on_progress),
                    timeout=backstop,
                )
            except TimeoutError as exc:
                raise JobTimeout(f"Job exceeded {backstop:.0f}s and stopped reporting progress") from exc
            finally:
                stage_spans.close()
            if state["stage"]:
                metrics.STAGE_DURATION.labels(stage=state["stage"]).observe(time.monotonic() - state["stage_started"])
            prefix = f"jobs/{job_id}"
            result_key = f"{prefix}/{RESULT_FILE}"
            views_key = None
            with span("storage.upload", prefix=prefix):
                await self.c.storage.put_file(result_key, work / RESULT_FILE, "model/gltf-binary")
                if (work / VIEWS_FILE).is_file():
                    views_key = f"{prefix}/{VIEWS_FILE}"
                    await self.c.storage.put_file(views_key, work / VIEWS_FILE, "image/png")
            return result_key, views_key

    async def _on_failure(
        self, msg: Message, job_id: uuid.UUID, attempt: int, max_attempts: int, exc: Exception
    ) -> None:
        q, repo, bus = self.c.queue, self.c.jobs, self.c.bus
        try:
            err: JobError = classify(exc)
            detail = str(err)
            if err.retryable and attempt < max_attempts:
                delay = backoff_delay(attempt, self.s.retry_base_delay_s, self.s.retry_max_delay_s)
                log.warning("job_retry_scheduled", error_code=err.code, error=detail, delay_s=round(delay, 1))
                await repo.requeue_for_retry(job_id, detail, err.code)
                await q.retry(msg, delay)
                metrics.JOB_RETRIES.labels(error_code=err.code).inc()
                await bus.publish(
                    msg.job_id,
                    {"type": "retrying", "attempt": attempt, "delay_s": round(delay, 1), "error_code": err.code},
                )
            elif err.retryable:
                log.error("job_dead_lettered", error_code=err.code, error=detail)
                await repo.fail(job_id, detail, err.code, dead_lettered=True)
                await q.dead_letter(msg, detail, err.code)
                metrics.JOBS_DEAD_LETTERED.inc()
                metrics.JOBS_FINISHED.labels(status="failed", error_code=err.code).inc()
                await bus.publish(
                    msg.job_id, {"type": "done", "status": "failed", "error_code": err.code, "error": detail}
                )
            else:
                log.warning("job_failed", error_code=err.code, error=detail)
                await repo.fail(job_id, detail, err.code)
                await q.ack(msg)
                metrics.JOBS_FINISHED.labels(status="failed", error_code=err.code).inc()
                await bus.publish(
                    msg.job_id, {"type": "done", "status": "failed", "error_code": err.code, "error": detail}
                )
        except Exception:
            # Could not record the failure (DB/Redis down). Leave the message
            # unacked: its lease expires and another worker retries it.
            log.exception("failure_handling_failed")

    async def _dead_letter_poison(self, msg: Message, job_id: uuid.UUID, attempts: int) -> None:
        detail = f"The worker stopped without recording a result on all {attempts} attempts."
        log.error("job_poison_dead_lettered", job_id=msg.job_id, attempts=attempts)
        await self.c.jobs.fail(job_id, detail, "worker_lost", dead_lettered=True)
        await self.c.queue.dead_letter(msg, detail, "worker_lost")
        metrics.JOBS_DEAD_LETTERED.inc()
        metrics.JOBS_FINISHED.labels(status="failed", error_code="worker_lost").inc()
        await self.c.bus.publish(
            msg.job_id, {"type": "done", "status": "failed", "error_code": "worker_lost", "error": detail}
        )

    async def _renew_lease(self, msg: Message) -> None:
        while True:
            await asyncio.sleep(self.s.heartbeat_interval_s)
            try:
                await self.c.queue.extend(msg, self.worker_id)
            except Exception:
                log.warning("lease_renewal_failed", exc_info=True)

    # --- background loops ----------------------------------------------------
    async def _heartbeat_loop(self) -> None:
        while True:
            await self._beat()
            await asyncio.sleep(self.s.heartbeat_interval_s)

    async def _beat(self) -> None:
        info = {
            "worker_id": self.worker_id,
            "backend": getattr(self.backend, "name", "?"),
            "model_loaded": self.model_loaded,
            "model_error": self.model_error,
            "current_job": self.current_job,
            "jobs_done": self.jobs_done,
            "at": time.time(),
        }
        try:
            await self.c.bus.heartbeat(self.worker_id, info, ttl_s=self.s.heartbeat_interval_s * 3)
        except Exception:
            log.warning("heartbeat_failed", exc_info=True)

    async def _maintenance_loop(self) -> None:
        last_reclaim = last_reconcile = last_expire = 0.0
        while True:
            now = time.monotonic()
            try:
                await self.c.queue.promote_due()
                if now - last_reclaim >= max(self.s.visibility_timeout_s / 2, 1):
                    last_reclaim = now
                    await self._reclaim()
                if now - last_reconcile >= 60:
                    last_reconcile = now
                    await self._reconcile()
                if self.s.job_ttl_hours > 0 and now - last_expire >= 3600:
                    last_expire = now
                    await self._expire_old_jobs()
                for state, n in (await self.c.queue.depth()).items():
                    metrics.QUEUE_DEPTH.labels(state=state).set(n)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("maintenance_failed", exc_info=True)
            await asyncio.sleep(1.0)

    async def _reclaim(self) -> None:
        """Take over messages whose worker stopped renewing its lease. They go
        back into the local handling path only when this worker is idle;
        otherwise they are re-enqueued for whoever is free."""
        msgs = await self.c.queue.reclaim(self.worker_id, self.s.visibility_timeout_s, count=10)
        for msg in msgs:
            metrics.JOBS_RECLAIMED.inc()
            log.warning("message_reclaimed", job_id=msg.job_id, delivery_id=msg.delivery_id)
            await self.c.queue.retry(msg, delay_s=0.001)

    async def _reconcile(self) -> None:
        for job in await self.c.jobs.stale_queued(self.s.stale_queued_after_s):
            log.warning("reconcile_requeue", job_id=str(job.id))
            await self.c.queue.enqueue(str(job.id), job.traceparent)
            await self.c.jobs.touch(job.id)

    async def _expire_old_jobs(self) -> None:
        old = await self.c.jobs.expired(self.s.job_ttl_hours * 3600)
        for job in old:
            await self.c.storage.delete_prefix(f"jobs/{job.id}/")
        if old:
            await self.c.jobs.delete([j.id for j in old])
            log.info("jobs_expired", count=len(old))

    def _record_gpu_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                metrics.GPU_MEMORY.labels(kind="allocated").set(torch.cuda.memory_allocated())
                metrics.GPU_MEMORY.labels(kind="peak").set(torch.cuda.max_memory_allocated())
        except Exception:
            pass
