"""In-process queue for lite mode and tests, with the same lease semantics as Redis."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from typing import Optional

from .base import Message


class InMemoryQueue:
    def __init__(self) -> None:
        self._ready: deque[Message] = deque()
        self._inflight: dict[str, tuple[Message, str, float]] = {}  # id -> (msg, consumer, lease start)
        self._delayed: list[tuple[float, int, str, Optional[str]]] = []
        self._dead: list[dict] = []
        self._ids = itertools.count(1)
        self._cond = asyncio.Condition()

    async def setup(self) -> None:
        pass

    def _new(self, job_id: str, traceparent: Optional[str]) -> Message:
        return Message(
            delivery_id=f"{int(time.time() * 1000)}-{next(self._ids)}",
            job_id=job_id,
            traceparent=traceparent,
            enqueued_at=time.time(),
        )

    async def enqueue(self, job_id: str, traceparent: Optional[str] = None, delay_s: float = 0.0) -> None:
        async with self._cond:
            if delay_s > 0:
                heapq.heappush(self._delayed, (time.time() + delay_s, next(self._ids), job_id, traceparent))
            else:
                self._ready.append(self._new(job_id, traceparent))
                self._cond.notify_all()

    async def receive(self, consumer: str, block_s: float = 5.0) -> Optional[Message]:
        deadline = time.monotonic() + block_s
        async with self._cond:
            while not self._ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    return None
            msg = self._ready.popleft()
            self._inflight[msg.delivery_id] = (msg, consumer, time.monotonic())
            return msg

    async def ack(self, msg: Message) -> None:
        self._inflight.pop(msg.delivery_id, None)

    async def extend(self, msg: Message, consumer: str) -> None:
        if msg.delivery_id in self._inflight:
            self._inflight[msg.delivery_id] = (msg, consumer, time.monotonic())

    async def retry(self, msg: Message, delay_s: float) -> None:
        await self.enqueue(msg.job_id, msg.traceparent, delay_s=max(delay_s, 0.001))
        await self.ack(msg)

    async def dead_letter(self, msg: Message, reason: str, error_code: str) -> None:
        self._dead.append(
            {
                "id": msg.delivery_id,
                "job_id": msg.job_id,
                "reason": reason,
                "error_code": error_code,
                "failed_at": f"{time.time():.3f}",
            }
        )
        await self.ack(msg)

    async def reclaim(self, consumer: str, min_idle_s: float, count: int = 10) -> list[Message]:
        now = time.monotonic()
        taken = []
        for did, (msg, _, started) in list(self._inflight.items()):
            if now - started >= min_idle_s and len(taken) < count:
                self._inflight[did] = (msg, consumer, now)
                taken.append(msg)
        return taken

    async def promote_due(self) -> int:
        moved = 0
        async with self._cond:
            now = time.time()
            while self._delayed and self._delayed[0][0] <= now:
                _, _, job_id, tp = heapq.heappop(self._delayed)
                self._ready.append(self._new(job_id, tp))
                moved += 1
            if moved:
                self._cond.notify_all()
        return moved

    async def depth(self) -> dict[str, int]:
        return {
            "ready": len(self._ready),
            "inflight": len(self._inflight),
            "delayed": len(self._delayed),
            "dead": len(self._dead),
        }

    async def list_dead(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._dead))[:limit]

    async def ping(self) -> None:
        pass

    async def close(self) -> None:
        pass
