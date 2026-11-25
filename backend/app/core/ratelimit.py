"""Token-bucket rate limiting.

A bucket holds up to `burst` tokens and refills at `rate` tokens per second.
Each request takes one; an empty bucket means 429 with Retry-After = time
until the next token. The Redis version runs as one Lua script so concurrent
gateway replicas share a bucket without races.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis


@dataclass
class Decision:
    allowed: bool
    remaining: int
    retry_after_s: float
    limit: int

    def headers(self) -> dict[str, str]:
        h = {"RateLimit-Limit": str(self.limit), "RateLimit-Remaining": str(self.remaining)}
        if not self.allowed:
            h["Retry-After"] = str(max(1, math.ceil(self.retry_after_s)))
        return h


class RateLimiter(Protocol):
    async def hit(self, key: str, rate_per_s: float, burst: int) -> Decision: ...


def _take(tokens: float, last: float, now: float, rate: float, burst: int) -> tuple[bool, float, float]:
    tokens = min(float(burst), tokens + max(0.0, now - last) * rate)
    if tokens >= 1.0:
        return True, tokens - 1.0, 0.0
    return False, tokens, (1.0 - tokens) / rate if rate > 0 else math.inf


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def hit(self, key: str, rate_per_s: float, burst: int) -> Decision:
        async with self._lock:
            now = time.monotonic()
            tokens, last = self._buckets.get(key, (float(burst), now))
            allowed, tokens, wait = _take(tokens, last, now, rate_per_s, burst)
            self._buckets[key] = (tokens, now)
            return Decision(allowed, int(tokens), wait, burst)


TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then tokens = burst; ts = now end
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local wait = 0
if tokens >= 1 then
  allowed = 1
  tokens = tokens - 1
else
  wait = (1 - tokens) / rate
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / rate) + 60)
return {allowed, tostring(tokens), tostring(wait)}
"""


class RedisRateLimiter:
    def __init__(self, redis: Redis, prefix: str = "i3d:rl") -> None:
        self.r = redis
        self.prefix = prefix
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    async def hit(self, key: str, rate_per_s: float, burst: int) -> Decision:
        allowed, tokens, wait = await self._script(keys=[f"{self.prefix}:{key}"], args=[rate_per_s, burst, time.time()])
        return Decision(bool(int(allowed)), int(float(tokens)), float(wait), burst)
