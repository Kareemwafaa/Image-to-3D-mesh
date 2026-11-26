from __future__ import annotations

import asyncio

import fakeredis
import pytest

from app.core.ratelimit import InMemoryRateLimiter, RedisRateLimiter


@pytest.fixture(params=["memory", "redis"])
def limiter(request):
    if request.param == "memory":
        return InMemoryRateLimiter()
    return RedisRateLimiter(fakeredis.FakeAsyncRedis())


async def test_burst_then_refill(limiter):
    rate = 20.0  # tokens per second
    results = [await limiter.hit("k", rate, burst=3) for _ in range(4)]
    assert [d.allowed for d in results] == [True, True, True, False]
    assert results[-1].retry_after_s > 0 and results[-1].headers()["Retry-After"] == "1"
    await asyncio.sleep(0.1)  # 2 tokens back
    assert (await limiter.hit("k", rate, burst=3)).allowed


async def test_buckets_are_independent(limiter):
    assert (await limiter.hit("a", 0.01, burst=1)).allowed
    assert not (await limiter.hit("a", 0.01, burst=1)).allowed
    assert (await limiter.hit("b", 0.01, burst=1)).allowed
