# ADR 0003: Postgres is the source of truth; Redis is transport

- Status: accepted
- Date: 2026-11-25

## Context

Job state (status, progress, attempts, errors, output keys) could live in Redis
next to the queue, which is fast and simple, or in a relational database.

## Options

1. **Job state in Redis hashes.** One fewer dependency. But no ad-hoc queries
   (list a user's jobs, find stuck jobs, analytics), memory-bound retention,
   and idempotency would need hand-rolled Lua.
2. **Postgres for state, Redis only for messages and events.**

## Decision

Option 2. `jobs` and `api_keys` live in Postgres (SQLAlchemy 2 async, Alembic
migrations). Queue messages carry only a job id and trace context.

Every state change is a guarded update, so the row is a state machine that a
duplicate or late message cannot corrupt:

```sql
UPDATE jobs SET status = 'running', attempts = attempts + 1, ...
WHERE id = :id AND status IN ('queued', 'running');
```

Idempotency is a unique constraint on `(owner, idempotency_key)` plus a
SHA-256 fingerprint of the request; concurrent submits with one key race on
the index and exactly one wins (tested against real Postgres in CI).

## Consequences

- If Redis loses data, the database still knows every job. A reconciler
  re-enqueues jobs stuck in `queued`, which also covers a gateway crash between
  the insert and the enqueue.
- Progress writes are throttled (`PROGRESS_DB_INTERVAL_S`) because live
  progress goes over pub/sub; the row only needs to be roughly current for polling.
- Lite mode uses SQLite with the same models and migrations (ADR 0008).
