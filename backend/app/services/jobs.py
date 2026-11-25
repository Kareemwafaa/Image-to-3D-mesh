"""Job submission: validation, deduplication, storage, persistence, enqueue.

Order matters for crash safety:
  1. store the input (content-addressed, so re-uploads are no-ops)
  2. insert the job row as `queued` (the durable record)
  3. enqueue the message
If the process dies between 2 and 3, the worker's reconciler finds the job
still `queued` after STALE_QUEUED_AFTER_S and enqueues it again. Duplicate
messages are harmless because workers check the row before running.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Optional

from PIL import Image, UnidentifiedImageError

from ..api.auth import Principal
from ..backends.base import GenerateOptions
from ..container import Container
from ..core import metrics
from ..core.errors import AppError
from ..core.logging import get_logger
from ..core.tracing import inject_context
from ..db.models import Job
from ..db.repository import DuplicateIdempotencyKey

log = get_logger(__name__)

MAX_SIDE = 2048


def normalise_image(data: bytes) -> bytes:
    """Verify the upload is an image, shrink huge ones, re-encode as PNG."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise AppError(400, "invalid_image", "Could not read the file as an image.") from exc
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def request_fingerprint(image_sha: str, options: GenerateOptions) -> str:
    payload = json.dumps({"image": image_sha, "options": options.to_dict()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


async def submit_job(
    c: Container,
    principal: Principal,
    data: bytes,
    options: GenerateOptions,
    idempotency_key: Optional[str],
) -> tuple[Job, bool]:
    """Returns (job, replayed). `replayed` is True when an earlier request with
    the same Idempotency-Key already created this job."""
    s = c.settings
    if len(data) > s.max_upload_bytes:
        raise AppError(413, "payload_too_large", f"Image is larger than {s.max_upload_mb:g} MB.")
    if idempotency_key is not None and not (1 <= len(idempotency_key) <= 255):
        raise AppError(400, "invalid_idempotency_key", "Idempotency-Key must be 1-255 characters.")

    sha = hashlib.sha256(data).hexdigest()
    fingerprint = request_fingerprint(sha, options)

    if idempotency_key:
        existing = await c.jobs.get_by_idempotency_key(principal.owner, idempotency_key)
        if existing is not None:
            return _replay(existing, fingerprint)

    cap = principal.inflight_cap(s)
    if cap is not None and await c.jobs.count_active(principal.owner) >= cap:
        raise AppError(
            429,
            "too_many_inflight_jobs",
            f"At most {cap} jobs may be queued or running at once.",
            {"Retry-After": "10"},
        )

    png = normalise_image(data)
    input_key = f"inputs/{sha}.png"
    if await c.storage.size(input_key) is None:
        await c.storage.put_bytes(input_key, png, "image/png")

    job = Job(
        owner=principal.owner,
        api_key_id=principal.api_key_id,
        status="queued",
        backend=s.model_backend,
        options=options.to_dict(),
        input_key=input_key,
        input_sha256=sha,
        idempotency_key=idempotency_key,
        request_hash=fingerprint,
        max_attempts=s.max_attempts,
        traceparent=inject_context(),
    )
    try:
        job = await c.jobs.create(job)
    except DuplicateIdempotencyKey:
        # Lost a race with a concurrent request carrying the same key.
        existing = await c.jobs.get_by_idempotency_key(principal.owner, idempotency_key or "")
        if existing is None:
            raise
        return _replay(existing, fingerprint)

    try:
        await c.queue.enqueue(str(job.id), job.traceparent)
    except Exception:
        # The row exists and is `queued`; the reconciler will enqueue it later.
        log.exception("enqueue_failed_will_reconcile", job_id=str(job.id))
    metrics.JOBS_SUBMITTED.labels(idempotent_replay="false").inc()
    log.info("job_submitted", job_id=str(job.id), owner=principal.owner, input_sha256=sha[:12])
    return job, False


def _replay(existing: Job, fingerprint: str) -> tuple[Job, bool]:
    if existing.request_hash != fingerprint:
        raise AppError(
            422,
            "idempotency_key_reused",
            "This Idempotency-Key was already used with a different image or options.",
        )
    metrics.JOBS_SUBMITTED.labels(idempotent_replay="true").inc()
    return existing, True
