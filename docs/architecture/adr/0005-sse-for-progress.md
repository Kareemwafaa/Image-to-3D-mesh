# ADR 0005: Server-Sent Events for progress, polling kept

- Status: accepted
- Date: 2026-11-25

## Context

A job takes minutes and moves through stages (segmentation, multi-view
diffusion, NeuS fitting, meshing). The web app polled every second.

## Options

1. **Polling only.** Simple and robust, but laggy and wasteful at scale.
2. **WebSockets.** Bidirectional, which we do not need; harder to proxy, and
   some tunnels used by the notebooks handle upgrades poorly.
3. **Server-Sent Events.** One-way, plain HTTP, reconnects by itself.

## Decision

`GET /v1/jobs/{id}/events` streams SSE. It sends a `snapshot` event first (the
same body as `GET /v1/jobs/{id}`), then `progress`, `status` and `retrying`
events relayed from Redis pub/sub, and finally `done`. It re-sends a snapshot
every 10 s (so a missed pub/sub message heals itself) and a comment heartbeat
every 15 s (so proxies keep the connection open). Polling stays fully supported.

## Consequences

- Any gateway replica can serve any stream, because events come from pub/sub,
  not from local memory.
- Pub/sub is fire-and-forget; correctness comes from the snapshot and resync,
  not from the event stream.
- Each open stream holds a connection, which is counted in `sse_connections`.
