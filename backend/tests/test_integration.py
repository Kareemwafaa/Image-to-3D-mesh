"""Full mode against real infrastructure: Postgres, Redis Streams, an S3 server.

These run in CI with service containers. Locally, start the stack and point the
tests at it:

    docker compose up -d postgres redis s3
    I3D_IT_DATABASE_URL=postgresql+asyncpg://i3d:i3d@localhost:5432/i3d \
    I3D_IT_REDIS_URL=redis://localhost:6379/15 \
    I3D_IT_S3_ENDPOINT=http://localhost:8333 \
    pytest -m integration

Without those variables the whole module is skipped.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import sqlalchemy as sa

from .conftest import ADMIN_KEY, make_settings, submit, wait_for
from .test_worker import FlakyBackend, SlowBackend

DB = os.environ.get("I3D_IT_DATABASE_URL")
REDIS = os.environ.get("I3D_IT_REDIS_URL")
S3 = os.environ.get("I3D_IT_S3_ENDPOINT")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DB and REDIS and S3), reason="set I3D_IT_* to run against real services"),
]


@pytest.fixture(scope="module", autouse=True)
def migrated():
    from app.db.migrate import upgrade

    upgrade(DB)


def run_sql(fn):
    """Run `await fn(conn)` on a short-lived engine (tests run outside the app's loop)."""
    from sqlalchemy.ext.asyncio import create_async_engine

    async def go():
        engine = create_async_engine(DB)
        try:
            async with engine.begin() as conn:
                return await fn(conn)
        finally:
            await engine.dispose()

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def clean_state():
    """Each test starts with empty tables and an empty Redis database."""
    from redis import Redis

    run_sql(lambda conn: conn.execute(sa.text("TRUNCATE jobs, api_keys CASCADE")))
    r = Redis.from_url(REDIS)
    r.flushdb()
    r.close()


def full_settings(tmp_path, **overrides):
    base = dict(
        mode="full",
        database_url=DB,
        redis_url=REDIS,
        storage_backend="s3",
        s3_endpoint_url=S3,
        s3_bucket="i3d-integration",  # one bucket, like production; job ids keep keys apart
        s3_access_key=os.environ.get("I3D_IT_S3_ACCESS_KEY", "i3d-access-key"),
        s3_secret_key=os.environ.get("I3D_IT_S3_SECRET_KEY", "i3d-secret-key-change-me"),
    )
    base.update(overrides)
    return make_settings(tmp_path, **base)


def gateway(settings, backend=None, run_worker=True):
    from fastapi.testclient import TestClient

    from app.main import create_app

    from .conftest import wait_ready

    class _Client:
        def __enter__(self):
            self.cm = TestClient(create_app(settings, run_worker=run_worker, backend=backend))
            client = self.cm.__enter__()
            wait_ready(client)
            return client

        def __exit__(self, *exc):
            return self.cm.__exit__(*exc)

    return _Client()


def test_migrations_on_postgres_match_models():
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    from app.db.models import Base

    def diff(conn):
        return compare_metadata(MigrationContext.configure(conn), Base.metadata)

    assert run_sql(lambda conn: conn.run_sync(diff)) == []


def test_end_to_end_on_real_services(tmp_path):
    with gateway(full_settings(tmp_path)) as client:
        r = submit(client, headers={"Idempotency-Key": "e2e-1"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        with client.stream("GET", f"/v1/jobs/{job_id}/events") as stream:
            text = "".join(stream.iter_text())  # events arrive over Redis pub/sub
        assert "event: done" in text

        body = wait_for(client, job_id)
        assert body["status"] == "succeeded" and body["attempts"] == 1
        mesh = client.get(body["result_url"])  # streamed back from S3
        assert mesh.status_code == 200 and mesh.content[:4] == b"glTF"

        ready = client.get("/readyz").json()
        assert all(v == "ok" for k, v in ready["checks"].items() if k in ("database", "queue", "storage", "bus"))
        depth = client.get("/v1/admin/queue", headers={"X-API-Key": ADMIN_KEY}).json()["depth"]
        assert depth["inflight"] == 0 and depth["ready"] == 0


def test_concurrent_idempotent_submissions_create_one_job(tmp_path):
    """Eight clients race with one Idempotency-Key; the unique index picks a winner."""
    with gateway(full_settings(tmp_path), run_worker=False) as client:
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: submit(client, headers={"Idempotency-Key": "race"}), range(8)))
        assert {r.status_code for r in results} <= {200, 202}
        assert sum(r.status_code == 202 for r in results) == 1
        assert len({r.json()["job_id"] for r in results}) == 1
        rows = run_sql(lambda conn: conn.execute(sa.text("SELECT count(*) FROM jobs")))
        assert rows.scalar_one() == 1


def test_retries_dead_letter_and_redrive_through_redis(tmp_path):
    backend = FlakyBackend(failures=2)
    with gateway(full_settings(tmp_path, max_attempts=2), backend=backend) as client:
        job_id = submit(client).json()["job_id"]
        body = wait_for(client, job_id)
        assert body["status"] == "failed" and body["dead_lettered"] and body["attempts"] == 2

        admin = {"X-API-Key": ADMIN_KEY}
        dead = client.get("/v1/admin/dead-letters", headers=admin).json()
        assert dead["messages"][0]["job_id"] == job_id  # read from the Redis dead-letter stream
        assert client.post(f"/v1/admin/jobs/{job_id}/redrive", headers=admin).status_code == 200
        assert wait_for(client, job_id, states=("succeeded",))["status"] == "succeeded"


def test_competing_workers_share_the_stream(tmp_path):
    """Two workers in one consumer group: every job runs once, both do work."""
    from app.worker.runner import Worker

    with gateway(full_settings(tmp_path), backend=SlowBackend(seconds=0.3)) as client:
        c = client.app.state.container
        second = Worker(c, SlowBackend(seconds=0.3), worker_id="worker-b")
        running = client.portal.start_task_soon(second.run)
        time.sleep(0.5)

        ids = [submit(client).json()["job_id"] for _ in range(8)]
        bodies = [wait_for(client, j, timeout=30) for j in ids]
        second.stop()
        running.result(timeout=15)
        assert all(b["status"] == "succeeded" and b["attempts"] == 1 for b in bodies)
        workers = {client.portal.call(c.jobs.get, uuid.UUID(j)).worker_id for j in ids}
        assert len(workers) == 2, workers


def test_crashed_consumer_is_reclaimed_with_xautoclaim(tmp_path):
    from app.worker.runner import Worker

    s = full_settings(tmp_path, visibility_timeout_s=0.5)
    with gateway(s, run_worker=False) as client:
        c = client.app.state.container
        job_id = submit(client).json()["job_id"]
        msg = client.portal.call(c.queue.receive, "ghost", 1.0)  # taken, then the ghost "dies"
        assert msg is not None and msg.job_id == job_id

        rescuer = Worker(c, SlowBackend(seconds=0.1), worker_id="rescuer")
        running = client.portal.start_task_soon(rescuer.run)
        body = wait_for(client, job_id, timeout=15)
        rescuer.stop()
        running.result(timeout=15)
        assert body["status"] == "succeeded"
        assert client.portal.call(c.queue.depth)["inflight"] == 0


def test_rate_limit_is_shared_between_gateway_replicas(tmp_path):
    """The token bucket lives in Redis, so two gateways enforce one limit."""
    s = full_settings(tmp_path, anon_submit_per_minute=1, anon_submit_burst=2)
    with gateway(s, run_worker=False) as a, gateway(s, run_worker=False) as b:
        assert submit(a).status_code == 202
        assert submit(b).status_code == 202
        assert submit(a).status_code == 429
        assert submit(b).status_code == 429
