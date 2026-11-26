# ADR 0004: Store uploads once, addressed by SHA-256

- Status: accepted
- Date: 2026-11-25

## Context

The gateway must persist the input image before enqueueing, so a worker on
another machine can fetch it. Users often resubmit the same photo with
different settings, and clients retry uploads.

## Decision

The gateway normalises the upload (verify, downscale to at most 2048 px,
re-encode as PNG), hashes the result, and stores it at `inputs/<sha256>.png`.
Outputs go to `jobs/<job_id>/result.glb` and `jobs/<job_id>/views.png`.

## Consequences

- Re-uploads cost nothing, and writes are naturally idempotent: the same bytes
  always land on the same key.
- The image hash is part of the idempotency fingerprint (ADR 0003), so "same
  key, different image" is detected cheaply.
- Output keys are deterministic per job, so a redelivered job overwrites its
  own outputs rather than creating duplicates.
- Deleting an input needs a reference check, because several jobs can share
  it. Expiry (`JOB_TTL_HOURS`) therefore only removes per-job outputs.
