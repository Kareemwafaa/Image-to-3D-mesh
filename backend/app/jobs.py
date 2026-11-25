"""Job records, the code that runs one job, and the local (in-process) queue.

Two job managers share the same interface:
  * LocalJobManager (here): one worker thread in the API process. For
    development and for a single GPU box.
  * ModalJobManager (modal_jobs.py): the API enqueues jobs onto Modal GPU
    containers, job state lives in a modal.Dict, files in a modal.Volume.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from PIL import Image

from .backends import GenerateOptions, MeshBackend
from .pipeline.segmentation import prepare

log = logging.getLogger(__name__)

RESULT_FILE = "result.glb"
VIEWS_FILE = "views.png"
INPUT_FILE = "input.png"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    options: dict
    status: JobStatus = JobStatus.queued
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    has_views: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobRecord":
        d = dict(d)
        d["status"] = JobStatus(d["status"])
        return cls(**d)

    @property
    def finished(self) -> bool:
        return self.status in (JobStatus.succeeded, JobStatus.failed)


class JobManager(Protocol):
    def submit(self, image: Image.Image, options: GenerateOptions) -> JobRecord: ...
    def get(self, job_id: str) -> Optional[JobRecord]: ...
    def read_file(self, job_id: str, name: str) -> Optional[Iterator[bytes]]:
        """Chunks of a job's file, or None if it does not exist."""
    def health(self) -> dict: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


def new_job_id() -> str:
    return uuid.uuid4().hex


def execute_job(
    backend: MeshBackend,
    job_dir: Path,
    options: GenerateOptions,
    update: Callable[..., None],
) -> None:
    """Run the whole pipeline for one job. `update(**fields)` records progress.
    Never raises: failures are recorded on the job."""
    started = time.monotonic()
    try:
        update(status=JobStatus.running, stage="segmenting", progress=0.01)
        with Image.open(job_dir / INPUT_FILE) as img:
            image = prepare(img, options.remove_background, options.foreground_ratio)
        image.save(job_dir / "segmented.png")
        last = {"p": 0.0}

        def progress(value: float, stage: Optional[str] = None) -> None:
            last["p"] = max(last["p"], min(value, 0.99))
            fields = {"progress": round(last["p"], 3)}
            if stage:
                fields["stage"] = stage
            if (job_dir / VIEWS_FILE).exists():
                fields["has_views"] = True
            update(**fields)

        backend.generate(image, options, job_dir, progress)
        if not (job_dir / RESULT_FILE).exists():
            raise RuntimeError("Backend finished without writing a mesh")
        update(
            status=JobStatus.succeeded,
            stage="done",
            progress=1.0,
            has_views=(job_dir / VIEWS_FILE).exists(),
        )
        log.info("Job %s done in %.1fs", job_dir.name, time.monotonic() - started)
    except Exception as exc:
        log.exception("Job %s failed", job_dir.name)
        update(status=JobStatus.failed, error=f"{type(exc).__name__}: {exc}")


class LocalJobManager:
    """Single worker thread: one GPU runs one job at a time. Job records are in
    memory (lost on restart); files stay under DATA_DIR until the TTL."""

    def __init__(self, backend: MeshBackend, data_dir: Path, ttl_hours: float = 24.0, preload: bool = True) -> None:
        self.backend = backend
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.preload = preload
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._model_ready = threading.Event()
        self._model_error: Optional[str] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mesh-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)

    def submit(self, image: Image.Image, options: GenerateOptions) -> JobRecord:
        job = JobRecord(id=new_job_id(), options=options.to_dict())
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True)
        image.save(job_dir / INPUT_FILE, format="PNG")
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            return JobRecord.from_dict(job.to_dict()) if job else None

    def read_file(self, job_id: str, name: str) -> Optional[Iterator[bytes]]:
        path = self.jobs_dir / job_id / name
        if not path.is_file():
            return None

        def chunks() -> Iterator[bytes]:
            with path.open("rb") as f:
                while chunk := f.read(1 << 20):
                    yield chunk

        return chunks()

    def health(self) -> dict:
        return {
            "model_loaded": self._model_ready.is_set(),
            "model_error": self._model_error,
            "queue_size": self._queue.qsize(),
        }

    # -- worker ----------------------------------------------------------
    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def _load_model(self) -> None:
        try:
            self.backend.load()
            self._model_error = None
            self._model_ready.set()
        except Exception as exc:  # keep the API up so /health can report it
            log.exception("Model failed to load")
            self._model_error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        if self.preload:
            self._load_model()
        while True:
            try:
                job_id = self._queue.get(timeout=60)
            except queue.Empty:
                self._cleanup()
                continue
            if job_id is None:
                return
            if not self._model_ready.is_set():
                self._load_model()
            if self._model_error:
                self._update(job_id, status=JobStatus.failed, error=f"Model is not available: {self._model_error}")
                continue
            job = self.get(job_id)
            if job is not None:
                execute_job(
                    self.backend,
                    self.jobs_dir / job_id,
                    GenerateOptions.from_dict(job.options),
                    lambda **kw: self._update(job_id, **kw),
                )
            self._cleanup()

    def _cleanup(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired = [
                j.id
                for j in self._jobs.values()
                if j.finished and datetime.fromisoformat(j.updated_at).timestamp() < cutoff
            ]
            for job_id in expired:
                del self._jobs[job_id]
        for job_id in expired:
            shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)
