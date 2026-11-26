from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status

from ..container import Container
from ..core.errors import AppError
from ..core.logging import get_logger
from ..db.models import ApiKey
from .auth import generate_key
from .deps import admin, container
from .schemas import ApiKeyCreate, ApiKeyCreated, ApiKeyOut, JobOut

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(admin)])
log = get_logger(__name__)


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(body: ApiKeyCreate, c: Container = Depends(container)) -> ApiKeyCreated:
    plaintext, prefix, digest = generate_key()
    key = await c.api_keys.create(
        ApiKey(
            name=body.name,
            prefix=prefix,
            key_hash=digest,
            is_admin=body.is_admin,
            submit_per_minute=body.submit_per_minute,
            max_inflight=body.max_inflight,
        )
    )
    log.info("api_key_created", key_id=str(key.id), prefix=prefix, is_admin=body.is_admin)
    return ApiKeyCreated(**ApiKeyOut.from_key(key).model_dump(), api_key=plaintext)


@router.get("/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(c: Container = Depends(container)) -> list[ApiKeyOut]:
    return [ApiKeyOut.from_key(k) for k in await c.api_keys.list()]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(key_id: uuid.UUID, c: Container = Depends(container)) -> None:
    if not await c.api_keys.revoke(key_id):
        raise AppError(404, "api_key_not_found", "No active API key with that id.")
    log.info("api_key_revoked", key_id=str(key_id))


@router.get("/dead-letters")
async def list_dead_letters(limit: int = Query(50, ge=1, le=500), c: Container = Depends(container)) -> dict:
    jobs = await c.jobs.list_dead_lettered(limit)
    return {
        "jobs": [JobOut.from_job(j).model_dump(mode="json") for j in jobs],
        "messages": await c.queue.list_dead(limit),
    }


@router.post("/jobs/{job_id}/redrive", response_model=JobOut)
async def redrive(job_id: uuid.UUID, c: Container = Depends(container)) -> JobOut:
    """Send a dead-lettered job through the queue again with fresh attempts."""
    if not await c.jobs.reset_for_redrive(job_id):
        raise AppError(409, "not_dead_lettered", "Only dead-lettered jobs can be re-driven.")
    job = await c.jobs.get(job_id)
    await c.queue.enqueue(str(job_id), job.traceparent if job else None)
    log.info("job_redriven", job_id=str(job_id))
    return JobOut.from_job(job)


@router.get("/queue")
async def queue_stats(c: Container = Depends(container)) -> dict:
    return {"depth": await c.queue.depth(), "workers": await c.bus.workers()}
