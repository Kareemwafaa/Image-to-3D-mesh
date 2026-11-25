"""Redis Streams queue.

    i3d:jobs           stream; consumer group "workers" gives leases (the PEL)
    i3d:jobs:delayed   sorted set of retries, scored by the time they are due
    i3d:jobs:dead      stream of dead-lettered messages (capped)

XREADGROUP delivers a message to one consumer and records it as pending.
XCLAIM with min-idle 0 resets its idle time (the lease heartbeat), and
XAUTOCLAIM takes over messages idle past the visibility timeout.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from .base import Message

STREAM = "i3d:jobs"
GROUP = "workers"
DELAYED = "i3d:jobs:delayed"
DEAD = "i3d:jobs:dead"
DEAD_MAXLEN = 10_000


def _decode(d: dict) -> dict:
    return {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in d.items()
    }


def _message(entry_id, fields) -> Message:
    f = _decode(fields)
    return Message(
        delivery_id=entry_id.decode() if isinstance(entry_id, bytes) else entry_id,
        job_id=f["job_id"],
        traceparent=f.get("traceparent") or None,
        enqueued_at=float(f.get("enqueued_at", 0)),
        fields=f,
    )


class RedisStreamQueue:
    def __init__(self, redis: Redis, stream: str = STREAM, group: str = GROUP) -> None:
        self.r = redis
        self.stream = stream
        self.group = group
        self.delayed = f"{stream}:delayed" if stream != STREAM else DELAYED
        self.dead = f"{stream}:dead" if stream != STREAM else DEAD

    async def setup(self) -> None:
        try:
            await self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _fields(self, job_id: str, traceparent: Optional[str]) -> dict:
        return {"job_id": job_id, "traceparent": traceparent or "", "enqueued_at": f"{time.time():.3f}"}

    async def enqueue(self, job_id: str, traceparent: Optional[str] = None, delay_s: float = 0.0) -> None:
        if delay_s > 0:
            member = json.dumps({**self._fields(job_id, traceparent), "nonce": uuid.uuid4().hex})
            await self.r.zadd(self.delayed, {member: time.time() + delay_s})
        else:
            await self.r.xadd(self.stream, self._fields(job_id, traceparent))

    async def receive(self, consumer: str, block_s: float = 5.0) -> Optional[Message]:
        res = await self.r.xreadgroup(
            self.group, consumer, {self.stream: ">"}, count=1, block=max(int(block_s * 1000), 1)
        )
        if not res:
            return None
        _, entries = res[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        return _message(entry_id, fields)

    async def ack(self, msg: Message) -> None:
        async with self.r.pipeline(transaction=True) as p:
            p.xack(self.stream, self.group, msg.delivery_id)
            p.xdel(self.stream, msg.delivery_id)
            await p.execute()

    async def extend(self, msg: Message, consumer: str) -> None:
        await self.r.xclaim(self.stream, self.group, consumer, 0, [msg.delivery_id], justid=True)

    async def retry(self, msg: Message, delay_s: float) -> None:
        await self.enqueue(msg.job_id, msg.traceparent, delay_s=max(delay_s, 0.001))
        await self.ack(msg)

    async def dead_letter(self, msg: Message, reason: str, error_code: str) -> None:
        await self.r.xadd(
            self.dead,
            {
                "job_id": msg.job_id,
                "reason": reason[:1000],
                "error_code": error_code,
                "failed_at": f"{time.time():.3f}",
            },
            maxlen=DEAD_MAXLEN,
            approximate=True,
        )
        await self.ack(msg)

    async def reclaim(self, consumer: str, min_idle_s: float, count: int = 10) -> list[Message]:
        res = await self.r.xautoclaim(self.stream, self.group, consumer, int(min_idle_s * 1000), "0-0", count=count)
        entries = res[1] if len(res) > 1 else []
        return [_message(eid, f) for eid, f in entries if f]

    async def promote_due(self) -> int:
        due = await self.r.zrangebyscore(self.delayed, "-inf", time.time(), start=0, num=100)
        moved = 0
        for member in due:
            # ZREM is the claim: when several workers promote at once, one wins.
            if await self.r.zrem(self.delayed, member):
                f = json.loads(member)
                await self.r.xadd(self.stream, self._fields(f["job_id"], f.get("traceparent") or None))
                moved += 1
        return moved

    async def depth(self) -> dict[str, int]:
        length = await self.r.xlen(self.stream)
        try:
            pending = (await self.r.xpending(self.stream, self.group))["pending"]
        except ResponseError:
            pending = 0
        return {
            "ready": max(length - pending, 0),
            "inflight": pending,
            "delayed": await self.r.zcard(self.delayed),
            "dead": await self.r.xlen(self.dead),
        }

    async def list_dead(self, limit: int = 50) -> list[dict]:
        entries = await self.r.xrevrange(self.dead, count=limit)
        return [{"id": (eid.decode() if isinstance(eid, bytes) else eid), **_decode(f)} for eid, f in entries]

    async def ping(self) -> None:
        await self.r.ping()

    async def close(self) -> None:
        pass
