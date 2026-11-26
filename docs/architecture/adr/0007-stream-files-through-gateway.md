# ADR 0007: Stream files through the gateway

- Status: accepted
- Date: 2026-11-25

## Context

Results live in object storage. Clients can get them either through the
gateway or straight from storage with a presigned URL.

## Options

1. **Presigned-URL redirects.** No bytes through the gateway; the standard
   answer at scale. But it requires the storage endpoint to be reachable by
   the client, which is not true for an S3 server on a private compose
   network, or behind the Cloudflare tunnel the notebooks use.
2. **Stream through the gateway.**

## Decision

Option 2. `GET /v1/jobs/{id}/result` and `/views` stream the object in chunks,
with the same auth and ownership checks as the job itself.

## Consequences

- Works identically in every environment, including lite mode and tunnels.
- Meshes are small (a few MB), so gateway bandwidth is not a concern at this
  scale.
- At scale, the change is local: return a `302` to a presigned URL (and put a
  CDN in front of the bucket) when storage is public.
