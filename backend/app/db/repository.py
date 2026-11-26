"""Data access for jobs and API keys.

Job state changes are *guarded* updates (`UPDATE ... WHERE id = ? AND status IN
(...)`), so concurrent or duplicate deliveries of the same message cannot move a
job backwards or finish it twice. Each method returns whether the transition
happened; callers treat False as "someone else already did it".
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from .models import ApiKey, Job, JobStatus, utcnow

ACTIVE = (JobStatus.queued.value, JobStatus.running.value)


class DuplicateIdempotencyKey(Exception):
    pass


class JobRepository:
    def __init__(self, sessions: async_sessionmaker) -> None:
        self.sessions = sessions

    # --- reads ---------------------------------------------------------------
    async def get(self, job_id: uuid.UUID) -> Optional[Job]:
        async with self.sessions() as s:
            return await s.get(Job, job_id)

    async def get_by_idempotency_key(self, owner: str, key: str) -> Optional[Job]:
        async with self.sessions() as s:
            res = await s.execute(select(Job).where(Job.owner == owner, Job.idempotency_key == key))
            return res.scalar_one_or_none()

    async def count_active(self, owner: str) -> int:
        async with self.sessions() as s:
            res = await s.execute(
                select(func.count()).select_from(Job).where(Job.owner == owner, Job.status.in_(ACTIVE))
            )
            return int(res.scalar_one())

    async def list_for_owner(self, owner: str, limit: int = 20, before: Optional[datetime] = None) -> Sequence[Job]:
        q = select(Job).where(Job.owner == owner)
        if before is not None:
            q = q.where(Job.created_at < before)
        q = q.order_by(Job.created_at.desc()).limit(limit)
        async with self.sessions() as s:
            return (await s.execute(q)).scalars().all()

    async def list_dead_lettered(self, limit: int = 50) -> Sequence[Job]:
        q = select(Job).where(Job.dead_lettered.is_(True)).order_by(Job.updated_at.desc()).limit(limit)
        async with self.sessions() as s:
            return (await s.execute(q)).scalars().all()

    async def stale_queued(self, older_than_s: float, limit: int = 100) -> Sequence[Job]:
        cutoff = utcnow() - timedelta(seconds=older_than_s)
        q = (
            select(Job)
            .where(Job.status == JobStatus.queued.value, Job.updated_at < cutoff)
            .order_by(Job.updated_at)
            .limit(limit)
        )
        async with self.sessions() as s:
            return (await s.execute(q)).scalars().all()

    async def expired(self, older_than_s: float, limit: int = 500) -> Sequence[Job]:
        cutoff = utcnow() - timedelta(seconds=older_than_s)
        q = select(Job).where(Job.status.in_(("succeeded", "failed")), Job.updated_at < cutoff).limit(limit)
        async with self.sessions() as s:
            return (await s.execute(q)).scalars().all()

    # --- writes ----------------------------------------------------------------
    async def create(self, job: Job) -> Job:
        async with self.sessions() as s:
            s.add(job)
            try:
                await s.commit()
            except IntegrityError as exc:
                await s.rollback()
                if job.idempotency_key:
                    raise DuplicateIdempotencyKey from exc
                raise
            return job

    async def _guarded(self, job_id: uuid.UUID, allowed: Iterable[str], **values) -> bool:
        values.setdefault("updated_at", utcnow())
        async with self.sessions() as s:
            res = await s.execute(update(Job).where(Job.id == job_id, Job.status.in_(tuple(allowed))).values(**values))
            await s.commit()
            return res.rowcount == 1

    async def start_attempt(self, job_id: uuid.UUID, worker_id: str) -> Optional[int]:
        """queued|running -> running, attempts += 1. Returns the attempt number,
        or None if the job is already final. `running` is allowed because a
        reclaimed message means the previous worker died mid-job."""
        async with self.sessions() as s:
            res = await s.execute(
                update(Job)
                .where(Job.id == job_id, Job.status.in_(ACTIVE))
                .values(
                    status=JobStatus.running.value,
                    attempts=Job.attempts + 1,
                    worker_id=worker_id,
                    stage="starting",
                    progress=0.0,
                    started_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
            await s.commit()
            if res.rowcount != 1:
                return None
            job = await s.get(Job, job_id, populate_existing=True)
            return job.attempts if job else None

    async def set_progress(self, job_id: uuid.UUID, progress: float, stage: Optional[str]) -> bool:
        values = {"progress": progress}
        if stage:
            values["stage"] = stage
        return await self._guarded(job_id, (JobStatus.running.value,), **values)

    async def set_outputs(self, job_id: uuid.UUID, **keys) -> bool:
        return await self._guarded(job_id, (JobStatus.running.value,), **keys)

    async def succeed(self, job_id: uuid.UUID, result_key: str, views_key: Optional[str]) -> bool:
        return await self._guarded(
            job_id,
            (JobStatus.running.value,),
            status=JobStatus.succeeded.value,
            stage="done",
            progress=1.0,
            result_key=result_key,
            views_key=views_key,
            error=None,
            error_code=None,
            finished_at=utcnow(),
        )

    async def requeue_for_retry(self, job_id: uuid.UUID, error: str, code: str) -> bool:
        return await self._guarded(
            job_id,
            (JobStatus.running.value,),
            status=JobStatus.queued.value,
            stage="retrying",
            error=error,
            error_code=code,
        )

    async def fail(self, job_id: uuid.UUID, error: str, code: str, dead_lettered: bool = False) -> bool:
        return await self._guarded(
            job_id,
            ACTIVE,
            status=JobStatus.failed.value,
            error=error[:2000],
            error_code=code,
            dead_lettered=dead_lettered,
            finished_at=utcnow(),
        )

    async def reset_for_redrive(self, job_id: uuid.UUID) -> bool:
        """Admin re-drive of a dead-lettered job: failed -> queued, attempts reset."""
        async with self.sessions() as s:
            res = await s.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == JobStatus.failed.value, Job.dead_lettered.is_(True))
                .values(
                    status=JobStatus.queued.value,
                    attempts=0,
                    dead_lettered=False,
                    stage=None,
                    progress=0.0,
                    error=None,
                    error_code=None,
                    finished_at=None,
                    updated_at=utcnow(),
                )
            )
            await s.commit()
            return res.rowcount == 1

    async def touch(self, job_id: uuid.UUID) -> None:
        async with self.sessions() as s:
            await s.execute(update(Job).where(Job.id == job_id).values(updated_at=utcnow()))
            await s.commit()

    async def delete(self, job_ids: Sequence[uuid.UUID]) -> None:
        from sqlalchemy import delete

        async with self.sessions() as s:
            await s.execute(delete(Job).where(Job.id.in_(job_ids)))
            await s.commit()

    async def ping(self) -> None:
        async with self.sessions() as s:
            await s.execute(select(1))


class ApiKeyRepository:
    def __init__(self, sessions: async_sessionmaker) -> None:
        self.sessions = sessions

    async def by_prefix(self, prefix: str) -> Optional[ApiKey]:
        async with self.sessions() as s:
            res = await s.execute(select(ApiKey).where(ApiKey.prefix == prefix))
            return res.scalar_one_or_none()

    async def create(self, key: ApiKey) -> ApiKey:
        async with self.sessions() as s:
            s.add(key)
            await s.commit()
            return key

    async def list(self) -> Sequence[ApiKey]:
        async with self.sessions() as s:
            return (await s.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))).scalars().all()

    async def revoke(self, key_id: uuid.UUID) -> bool:
        async with self.sessions() as s:
            res = await s.execute(
                update(ApiKey).where(ApiKey.id == key_id, ApiKey.revoked_at.is_(None)).values(revoked_at=utcnow())
            )
            await s.commit()
            return res.rowcount == 1

    async def mark_used(self, key_id: uuid.UUID) -> None:
        async with self.sessions() as s:
            await s.execute(update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=utcnow()))
            await s.commit()
