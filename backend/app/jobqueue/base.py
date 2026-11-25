"""Job queue interface: at-least-once delivery with leases.

Semantics both implementations follow (and the tests check):
  * `receive` hands a message to one consumer and starts a lease
    (visibility timeout). Until it is acked, nobody else gets it.
  * `extend` renews the lease; workers call it as a heartbeat during long jobs.
  * `reclaim` hands messages whose lease expired (the worker died) to a live
    consumer. That is what makes delivery at-least-once, so consumers must be
    idempotent (they are: see worker/runner.py and the guarded DB updates).
  * `retry` acks the message and schedules a new one after a delay;
    `promote_due` moves due delayed messages back into the ready queue.
  * `dead_letter` acks the message and records it for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Message:
    delivery_id: str
    job_id: str
    traceparent: Optional[str] = None
    enqueued_at: float = 0.0
    fields: dict = field(default_factory=dict)


class JobQueue(Protocol):
    async def setup(self) -> None: ...
    async def enqueue(self, job_id: str, traceparent: Optional[str] = None, delay_s: float = 0.0) -> None: ...
    async def receive(self, consumer: str, block_s: float = 5.0) -> Optional[Message]: ...
    async def ack(self, msg: Message) -> None: ...
    async def extend(self, msg: Message, consumer: str) -> None: ...
    async def retry(self, msg: Message, delay_s: float) -> None: ...
    async def dead_letter(self, msg: Message, reason: str, error_code: str) -> None: ...
    async def reclaim(self, consumer: str, min_idle_s: float, count: int = 10) -> list[Message]: ...
    async def promote_due(self) -> int: ...
    async def depth(self) -> dict[str, int]: ...
    async def list_dead(self, limit: int = 50) -> list[dict]: ...
    async def ping(self) -> None: ...
    async def close(self) -> None: ...


def backoff_delay(attempt: int, base_s: float, max_s: float, jitter: float = 0.2) -> float:
    """Exponential backoff with +-jitter: base * 2^(attempt-1), capped."""
    import random

    delay = min(max_s, base_s * (2 ** max(attempt - 1, 0)))
    return max(0.0, delay * (1 + random.uniform(-jitter, jitter)))
