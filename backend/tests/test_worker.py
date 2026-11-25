"""Worker reliability: retries, dead letters, timeouts, duplicates, crash recovery."""

from __future__ import annotations

import time
import uuid

from app.backends.mock import MockBackend
from app.core.errors import TransientJobError
from app.db.models import Job

from .conftest import ADMIN_KEY, client_for, make_settings, submit, wait_for


class FlakyBackend(MockBackend):
    """Fails with a transient error `failures` times, then works."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def generate(self, image, options, out_dir, progress):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientJobError("storage blip", code="transient")
        return super().generate(image, options, out_dir, progress)


class SlowBackend(MockBackend):
    """Reports progress every 50 ms for `seconds`, then writes the mesh."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def generate(self, image, options, out_dir, progress):
        end = time.monotonic() + self.seconds
        i = 0
        while time.monotonic() < end:
            i += 1
            progress(min(0.9, i / 100), "reconstructing")
            time.sleep(0.05)
        return super().generate(image, options, out_dir, progress)


class CrashBackend(MockBackend):
    def generate(self, image, options, out_dir, progress):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


def container(client):
    return client.app.state.container


def test_transient_failures_are_retried(tmp_path):
    backend = FlakyBackend(failures=2)
    with client_for(make_settings(tmp_path, max_attempts=3), backend=backend) as client:
        job_id = submit(client).json()["job_id"]
        body = wait_for(client, job_id)
        assert body["status"] == "succeeded" and body["attempts"] == 3
        assert backend.calls == 3


def test_exhausted_retries_go_to_dead_letter_and_can_be_redriven(tmp_path):
    backend = FlakyBackend(failures=2)
    with client_for(make_settings(tmp_path, max_attempts=2), backend=backend) as client:
        job_id = submit(client).json()["job_id"]
        body = wait_for(client, job_id)
        assert body["status"] == "failed" and body["dead_lettered"] is True and body["attempts"] == 2

        admin = {"X-API-Key": ADMIN_KEY}
        dead = client.get("/v1/admin/dead-letters", headers=admin).json()
        assert [j["job_id"] for j in dead["jobs"]] == [job_id]
        assert dead["messages"][0]["job_id"] == job_id

        # The backend now works (its failures are used up): re-drive succeeds.
        assert client.post(f"/v1/admin/jobs/{job_id}/redrive", headers=admin).status_code == 200
        body = wait_for(client, job_id, states=("succeeded",))
        assert body["dead_lettered"] is False
        assert client.post(f"/v1/admin/jobs/{job_id}/redrive", headers=admin).status_code == 409


def test_unknown_errors_are_retried_then_dead_lettered(tmp_path):
    with client_for(make_settings(tmp_path, max_attempts=2), backend=CrashBackend()) as client:
        body = wait_for(client, submit(client).json()["job_id"])
        assert body["status"] == "failed" and body["error_code"] == "internal" and body["dead_lettered"]


def test_timeout_is_permanent(tmp_path):
    with client_for(make_settings(tmp_path, job_timeout_s=0.3), backend=SlowBackend(seconds=5)) as client:
        body = wait_for(client, submit(client).json()["job_id"])
        assert body["status"] == "failed" and body["error_code"] == "timeout" and body["attempts"] == 1


def test_progress_is_visible_while_running(tmp_path):
    with client_for(make_settings(tmp_path), backend=SlowBackend(seconds=1.0)) as client:
        job_id = submit(client).json()["job_id"]
        running = wait_for(client, job_id, states=("running",))
        deadline = time.time() + 3
        while running["progress"] == 0 and time.time() < deadline:
            running = client.get(f"/v1/jobs/{job_id}").json()
        assert running["stage"] in ("reconstructing", "segmenting") and running["progress"] > 0
        assert wait_for(client, job_id)["status"] == "succeeded"


def test_duplicate_messages_do_not_rerun_jobs(tmp_path):
    with client_for(make_settings(tmp_path)) as client:
        c = container(client)
        job_id = submit(client).json()["job_id"]
        wait_for(client, job_id)
        client.portal.call(c.queue.enqueue, job_id, None)  # a redelivery of a finished job
        time.sleep(0.5)
        body = client.get(f"/v1/jobs/{job_id}").json()
        assert body["attempts"] == 1
        depth = client.portal.call(c.queue.depth)
        assert depth["ready"] == 0 and depth["inflight"] == 0


def test_crashed_worker_lease_is_reclaimed(tmp_path):
    """A message taken by a worker that died is redelivered after the visibility timeout."""
    with client_for(make_settings(tmp_path, visibility_timeout_s=0.5), backend=SlowBackend(seconds=0.1)) as client:
        c = container(client)
        client.app.state.worker.stop()  # pause the real worker so the ghost gets the message
        time.sleep(2.2)
        job_id = submit(client).json()["job_id"]
        msg = client.portal.call(c.queue.receive, "ghost-worker", 1.0)
        assert msg is not None and msg.job_id == job_id

        from app.worker.runner import Worker

        worker = Worker(c, SlowBackend(seconds=0.1), worker_id="rescuer")
        running = client.portal.start_task_soon(worker.run)
        body = wait_for(client, job_id, timeout=10)
        worker.stop()
        running.result(timeout=15)
        assert body["status"] == "succeeded"
        assert client.portal.call(c.queue.depth)["inflight"] == 0


def test_reconciler_enqueues_orphaned_jobs(tmp_path):
    """A job row whose enqueue was lost (crash between insert and XADD) still runs."""
    with client_for(make_settings(tmp_path, stale_queued_after_s=0)) as client:
        c = container(client)
        submit(client)  # stores the input image under its hash
        first = client.get("/v1/jobs").json()["items"][0]
        wait_for(client, first["job_id"])
        src = client.portal.call(c.jobs.get, uuid.UUID(first["job_id"]))
        orphan = Job(
            owner=src.owner,
            backend="mock",
            options=src.options,
            input_key=src.input_key,
            input_sha256=src.input_sha256,
            max_attempts=3,
        )
        client.portal.call(c.jobs.create, orphan)
        client.portal.call(client.app.state.worker._reconcile)
        assert wait_for(client, str(orphan.id))["status"] == "succeeded"


def test_poison_message_is_dead_lettered(tmp_path):
    """A job whose every attempt killed its worker is dead-lettered, not redelivered forever."""
    with client_for(make_settings(tmp_path)) as client:
        c = container(client)
        first = submit(client).json()["job_id"]
        src = client.portal.call(c.jobs.get, uuid.UUID(wait_for(client, first)["job_id"]))
        job = Job(
            owner=src.owner,
            backend="mock",
            options=src.options,
            input_key=src.input_key,
            input_sha256=src.input_sha256,
            max_attempts=2,
        )
        client.portal.call(c.jobs.create, job)
        for _ in range(2):  # two workers took it and died before recording anything
            client.portal.call(c.jobs.start_attempt, job.id, "crashed-worker")

        client.portal.call(c.queue.enqueue, str(job.id), None)  # the lease expired: redelivery
        body = wait_for(client, str(job.id))
        assert body["status"] == "failed" and body["error_code"] == "worker_lost"
        assert body["dead_lettered"] and body["attempts"] == 2
        dead = client.get("/v1/admin/dead-letters", headers={"X-API-Key": ADMIN_KEY}).json()
        assert [m["job_id"] for m in dead["messages"]] == [str(job.id)]


def test_failure_classification_does_not_need_torch():
    """CPU workers (mock model) have no torch; classifying a failure must not import it."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from app.core.errors import classify\n"
        "from app.pipeline.errors import MeshingError, PreprocessError\n"
        "assert classify(PreprocessError('x')).code == 'no_foreground'\n"
        "assert classify(MeshingError('x')).code == 'empty_mesh'\n"
        "assert classify(RuntimeError('x')).code == 'internal'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__("pathlib").Path(__file__).parents[1]))
