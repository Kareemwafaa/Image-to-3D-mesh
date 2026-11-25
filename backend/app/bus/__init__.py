"""Progress events (pub/sub) and worker heartbeats.

Events are fire-and-forget hints for live UIs; the database row stays the
source of truth, which is why an SSE stream always starts with a snapshot.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional, Protocol

from redis.asyncio import Redis


class EventBus(Protocol):
    async def publish(self, job_id: str, event: dict) -> None: ...
    def subscribe(self, job_id: str): ...  # async context manager -> Subscription
    async def heartbeat(self, worker_id: str, info: dict, ttl_s: float) -> None: ...
    async def workers(self) -> list[dict]: ...
    async def ping(self) -> None: ...


class Subscription(Protocol):
    async def get(self, timeout: float) -> Optional[dict]: ...


# --- in-memory ---------------------------------------------------------------


class _MemorySub:
    def __init__(self) -> None:
        self.q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)

    async def get(self, timeout: float) -> Optional[dict]:
        try:
            return await asyncio.wait_for(self.q.get(), timeout)
        except TimeoutError:
            return None


class InMemoryBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[_MemorySub]] = {}
        self._workers: dict[str, tuple[float, dict]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def publish(self, job_id: str, event: dict) -> None:
        for sub in list(self._subs.get(job_id, ())):
            try:
                sub.q.put_nowait(event)
            except asyncio.QueueFull:  # slow consumer: drop, the snapshot will catch up
                pass

    @asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[_MemorySub]:
        sub = _MemorySub()
        self._subs.setdefault(job_id, set()).add(sub)
        try:
            yield sub
        finally:
            self._subs[job_id].discard(sub)
            if not self._subs[job_id]:
                del self._subs[job_id]

    async def heartbeat(self, worker_id: str, info: dict, ttl_s: float) -> None:
        self._workers[worker_id] = (time.time() + ttl_s, info)

    async def workers(self) -> list[dict]:
        now = time.time()
        return [info for exp, info in self._workers.values() if exp > now]

    async def ping(self) -> None:
        pass


# --- Redis -----------------------------------------------------------------------


class _RedisSub:
    def __init__(self, pubsub) -> None:
        self.pubsub = pubsub

    async def get(self, timeout: float) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            msg = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining)
            if msg and msg.get("type") == "message":
                return json.loads(msg["data"])


class RedisBus:
    def __init__(self, redis: Redis, prefix: str = "i3d") -> None:
        self.r = redis
        self.prefix = prefix

    def _channel(self, job_id: str) -> str:
        return f"{self.prefix}:events:{job_id}"

    async def publish(self, job_id: str, event: dict) -> None:
        await self.r.publish(self._channel(job_id), json.dumps(event, default=str))

    @asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[_RedisSub]:
        pubsub = self.r.pubsub()
        await pubsub.subscribe(self._channel(job_id))
        try:
            yield _RedisSub(pubsub)
        finally:
            await pubsub.unsubscribe(self._channel(job_id))
            await pubsub.aclose()

    async def heartbeat(self, worker_id: str, info: dict, ttl_s: float) -> None:
        await self.r.set(f"{self.prefix}:workers:{worker_id}", json.dumps(info), ex=max(int(ttl_s), 1))

    async def workers(self) -> list[dict]:
        out = []
        async for key in self.r.scan_iter(match=f"{self.prefix}:workers:*", count=100):
            raw = await self.r.get(key)
            if raw:
                out.append(json.loads(raw))
        return out

    async def ping(self) -> None:
        await self.r.ping()
