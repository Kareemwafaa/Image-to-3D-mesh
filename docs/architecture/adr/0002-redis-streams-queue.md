# ADR 0002: Redis Streams with a small hand-written consumer

- Status: accepted
- Date: 2026-11-25

## Context

Jobs need durable delivery to exactly one worker at a time, recovery when a
worker dies mid-job, retries with backoff, and a dead-letter queue. The same
semantics must also run in-process for lite mode (ADR 0008).

## Options

1. **Celery.** Mature and featureful, but its prefork pool fights CUDA (forked
   children cannot reuse the parent's CUDA context), visibility-timeout
   behaviour on Redis is subtle, and it would hide the reliability logic this
   project is meant to show.
2. **RQ / arq.** Simpler, but built on lists; leases and reclaim are weaker,
   and there is no in-memory twin for lite mode.
3. **RabbitMQ / SQS.** Excellent brokers, but one more service to run, and SQS
   is not free or local.
4. **Redis Streams consumer groups, with a ~300-line consumer.**

## Decision

Option 4 (`app/jobqueue/redis_streams.py`):

| Need | Mechanism |
| --- | --- |
| Deliver to one worker | `XREADGROUP` on stream `i3d:jobs`, group `workers` |
| Lease while running | The pending entry list; the worker renews with `XCLAIM … JUSTID` on each heartbeat |
| Recover from a dead worker | `XAUTOCLAIM` of entries idle longer than `VISIBILITY_TIMEOUT_S` |
| Retry with backoff | `ZADD` to `i3d:jobs:delayed` scored by due time; a promoter moves due entries back with `ZREM` (the claim) then `XADD` |
| Dead letters | `XADD` to `i3d:jobs:dead` (capped with `MAXLEN`), then `XACK` |
| Acknowledge | `XACK` + `XDEL` |

`app/jobqueue/memory.py` implements the same interface with an asyncio
condition, a deque, an in-flight map and a heap. Both pass the same test suite
(`tests/test_queue.py`, run against fakeredis and in CI against real Redis).

## Consequences

- Delivery is at-least-once, so every consumer step must be idempotent
  (ADR 0003 explains how).
- A message that kills its worker every time would be reclaimed forever; the
  worker dead-letters a job whose every allowed attempt started but never
  recorded an outcome (`error_code = worker_lost`).
- We own and must test this code. In exchange the semantics are explicit,
  small enough to read in one sitting, and identical in lite mode.
