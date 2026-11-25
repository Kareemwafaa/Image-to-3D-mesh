"""Composition root: builds the adapters for the configured mode.

Everything above this file depends on interfaces (JobQueue, EventBus,
ObjectStorage, RateLimiter, repositories); only this file knows whether they
are backed by Redis/Postgres/S3 (full) or memory/SQLite/disk (lite).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .bus import EventBus, InMemoryBus, RedisBus
from .core.config import Settings
from .core.ratelimit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from .db.repository import ApiKeyRepository, JobRepository
from .db.session import make_engine, make_sessionmaker
from .jobqueue.base import JobQueue
from .jobqueue.memory import InMemoryQueue
from .jobqueue.redis_streams import RedisStreamQueue
from .storage.base import ObjectStorage
from .storage.local import LocalStorage


@dataclass
class Container:
    settings: Settings
    engine: object
    jobs: JobRepository
    api_keys: ApiKeyRepository
    storage: ObjectStorage
    queue: JobQueue
    bus: EventBus
    limiter: RateLimiter
    redis: Optional[object] = None
    extras: dict = field(default_factory=dict)

    async def start(self, migrate: bool = False) -> None:
        if migrate:
            from .db.migrate import upgrade_async

            await upgrade_async(self.settings.database_url)
        ensure = getattr(self.storage, "ensure_bucket", None)
        if ensure is not None:
            await ensure()
        await self.queue.setup()

    async def close(self) -> None:
        await self.queue.close()
        if self.redis is not None:
            await self.redis.aclose()
        await self.engine.dispose()


def build_storage(s: Settings) -> ObjectStorage:
    if s.storage_backend == "local":
        return LocalStorage(s.storage_dir)
    from .storage.s3 import S3Storage

    return S3Storage(s.s3_bucket, s.s3_endpoint_url, s.s3_access_key, s.s3_secret_key, s.s3_region)


def build_container(s: Settings, redis=None) -> Container:
    engine = make_engine(s.database_url)
    sessions = make_sessionmaker(engine)
    if s.is_lite:
        queue, bus, limiter = InMemoryQueue(), InMemoryBus(), InMemoryRateLimiter()
    else:
        if redis is None:
            from redis.asyncio import Redis

            redis = Redis.from_url(s.redis_url, health_check_interval=30)
        queue, bus, limiter = RedisStreamQueue(redis), RedisBus(redis), RedisRateLimiter(redis)
    return Container(
        settings=s,
        engine=engine,
        jobs=JobRepository(sessions),
        api_keys=ApiKeyRepository(sessions),
        storage=build_storage(s),
        queue=queue,
        bus=bus,
        limiter=limiter,
        redis=redis,
    )
