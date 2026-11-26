# ADR 0008: One code base with swappable adapters (full and lite)

- Status: accepted
- Date: 2026-11-25

## Context

The project must run for free: on a single Colab or Kaggle GPU, where there is
no Docker, no Postgres and no Redis. It must also show a production-shaped
system. Two separate apps would drift apart, and the notebook would exercise
code that production never runs.

## Decision

Every piece of infrastructure is an interface with two implementations, and
`app/container.py` (the composition root) is the only place that chooses:

| Interface | Full | Lite |
| --- | --- | --- |
| `JobQueue` | Redis Streams | asyncio, in memory |
| `EventBus` | Redis pub/sub | in memory |
| `ObjectStorage` | S3 | local disk |
| `RateLimiter` | Redis + Lua | in memory |
| Job store | Postgres | SQLite (same models and migrations) |
| Worker | separate process | task in the API process |

`python -m app.lite` sets `MODE=lite` and runs the gateway with an in-process
worker.

## Consequences

- The gateway, the worker loop, the retry and dead-letter logic and the
  pipeline are the same code in both modes; only the adapters differ.
- The API test suite runs in lite mode in seconds, while the integration suite
  runs the full adapters against real services.
- Each adapter pair must keep identical semantics, which the shared queue and
  rate-limiter tests enforce.
