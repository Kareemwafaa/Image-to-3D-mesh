# ADR 0006: Cooperative timeouts instead of killing processes

- Status: accepted
- Date: 2026-11-25

## Context

A job must not run forever (a diverging NeuS fit, a stuck download). The usual
fix is to run each job in a subprocess and kill it on timeout, but the worker
keeps about 5 GB of weights loaded, and a fresh process would reload them
(30 to 60 s) for every job.

## Decision

The pipeline already reports progress many times a second. The progress
callback checks a per-job deadline (`JOB_TIMEOUT_S`, 15 min by default) and
raises `JobTimeout`, which unwinds the pipeline thread normally. Timeouts are
permanent failures (a retry would time out again). As a backstop,
`asyncio.wait_for` gives up on a pipeline that stops reporting progress at all
(deadline + 60 s), and the queue's visibility timeout covers a worker that
hangs completely (ADR 0002).

## Consequences

- Models stay loaded; a timeout costs nothing but the job.
- A stage that blocks inside one native call without reporting progress can
  only be stopped by the backstop; it cannot be interrupted mid-call.
