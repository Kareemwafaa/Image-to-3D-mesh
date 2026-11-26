"""The same contract tests run against both queue implementations."""

from __future__ import annotations

import asyncio

import fakeredis
import pytest

from app.jobqueue.base import backoff_delay
from app.jobqueue.memory import InMemoryQueue
from app.jobqueue.redis_streams import RedisStreamQueue


@pytest.fixture(params=["memory", "redis"])
async def queue(request):
    if request.param == "memory":
        q = InMemoryQueue()
    else:
        q = RedisStreamQueue(fakeredis.FakeAsyncRedis())
    await q.setup()
    yield q
    await q.close()


async def test_deliver_ack(queue):
    await queue.enqueue("job-1", "00-abc-def-01")
    msg = await queue.receive("w1", block_s=0.2)
    assert msg.job_id == "job-1" and msg.traceparent == "00-abc-def-01"
    assert (await queue.depth())["inflight"] == 1
    await queue.ack(msg)
    assert await queue.depth() == {"ready": 0, "inflight": 0, "delayed": 0, "dead": 0}


async def test_receive_times_out_when_empty(queue):
    assert await queue.receive("w1", block_s=0.05) is None


async def test_message_goes_to_one_consumer(queue):
    await queue.enqueue("job-1")
    first = await queue.receive("w1", block_s=0.2)
    second = await queue.receive("w2", block_s=0.05)
    assert first is not None and second is None


async def test_expired_lease_is_reclaimed(queue):
    await queue.enqueue("job-1")
    msg = await queue.receive("dead-worker", block_s=0.2)
    assert await queue.reclaim("w2", min_idle_s=10) == []  # lease still fresh
    await asyncio.sleep(0.15)
    reclaimed = await queue.reclaim("w2", min_idle_s=0.1)
    assert [m.job_id for m in reclaimed] == ["job-1"]
    assert reclaimed[0].delivery_id == msg.delivery_id


async def test_extend_keeps_lease_alive(queue):
    await queue.enqueue("job-1")
    msg = await queue.receive("w1", block_s=0.2)
    await asyncio.sleep(0.15)
    await queue.extend(msg, "w1")
    assert await queue.reclaim("w2", min_idle_s=0.1) == []


async def test_retry_is_delayed_then_promoted(queue):
    await queue.enqueue("job-1")
    msg = await queue.receive("w1", block_s=0.2)
    await queue.retry(msg, delay_s=0.1)
    assert (await queue.depth())["delayed"] == 1
    assert await queue.promote_due() == 0
    await asyncio.sleep(0.15)
    assert await queue.promote_due() == 1
    again = await queue.receive("w1", block_s=0.2)
    assert again.job_id == "job-1" and again.delivery_id != msg.delivery_id


async def test_dead_letter(queue):
    await queue.enqueue("job-1")
    msg = await queue.receive("w1", block_s=0.2)
    await queue.dead_letter(msg, "boom", "internal")
    depth = await queue.depth()
    assert depth["dead"] == 1 and depth["inflight"] == 0
    dead = await queue.list_dead()
    assert dead[0]["job_id"] == "job-1" and dead[0]["error_code"] == "internal"


async def test_concurrent_consumers_get_distinct_messages(queue):
    for i in range(20):
        await queue.enqueue(f"job-{i}")
    got = await asyncio.gather(*(queue.receive(f"w{i}", block_s=0.2) for i in range(20)))
    assert sorted(m.job_id for m in got) == sorted(f"job-{i}" for i in range(20))


def test_backoff_grows_and_caps():
    delays = [backoff_delay(a, base_s=1, max_s=10, jitter=0) for a in range(1, 7)]
    assert delays == [1, 2, 4, 8, 10, 10]
    assert 0.8 <= backoff_delay(1, 1, 10, jitter=0.2) <= 1.2
