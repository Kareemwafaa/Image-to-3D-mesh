# ADR 0009: SeaweedFS as the local S3 server

- Status: accepted
- Date: 2026-11-25

## Context

The full stack needs an S3-compatible object store that runs in Docker
Compose and in CI. MinIO was the obvious choice, but its community Docker
images are no longer published (`minio/minio` is gone from Docker Hub), and
pinning an old, unmaintained image is a poor default for a public repo.

## Options

1. **Old MinIO images** from a mirror. Work today, receive no fixes.
2. **Build MinIO from source** in the compose file. Slow, and the community
   edition's future is unclear.
3. **SeaweedFS** (Apache 2.0, actively maintained, `weed server -s3`).
4. **Moto server.** Great for tests, but it is a mock, not storage.

## Decision

Option 3: `chrislusf/seaweedfs` with S3 credentials from
`deploy/seaweedfs/s3.json`. The application code only speaks S3 through
boto3, so nothing outside `docker-compose.yml` and CI names the server.

## Consequences

- Switching to AWS S3, Cloudflare R2, Backblaze B2 or a self-built MinIO is a
  change of `S3_ENDPOINT_URL` and credentials only.
- The integration tests run against SeaweedFS in CI and pass against moto and
  MinIO-compatible servers too.
