# ADR 0001: Separate the API gateway from the GPU workers

- Status: accepted
- Date: 2026-11-25

## Context

The first version ran the model inside the FastAPI process, on a background
thread. That is fine for a demo but couples two very different resources: HTTP
handling is cheap, I/O-bound and wants many replicas; inference is expensive,
holds ~5 GB of weights in GPU memory and wants exactly one job per GPU. A
CUDA fault or an out-of-memory error in the model also took the API down with it.

## Options

1. **One process, model on a thread** (the old design).
2. **Gateway plus workers behind a queue.** The gateway validates and enqueues;
   workers own the GPUs and pull work.
3. **Model server (Triton, TorchServe) called synchronously.** Good for
   millisecond inference, but a job here takes minutes, so a synchronous call
   would hold connections open and lose work on any disconnect.

## Decision

Option 2. The gateway (`app/api`, `app/services`) never imports torch. Workers
(`app/worker`) are separate processes that each load the model once and pull
jobs from the queue (ADR 0002).

## Consequences

- The two tiers scale independently: gateway replicas on CPU, worker replicas
  one per GPU (`docker compose up --scale worker-gpu=N`).
- A model crash restarts one worker; queued jobs wait, and the job it was
  running is redelivered (ADR 0002).
- The gateway image is about 500 MB (no torch) instead of several GB.
- More moving parts: a queue, a shared job store (ADR 0003) and object storage
  (ADR 0004) are now required. Lite mode (ADR 0008) keeps a one-process option.
