"""Job endpoints. Every route is served under /v1 and, for the existing web
app, at its original unversioned path."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from ..backends.base import GenerateOptions
from ..container import Container
from ..core import metrics
from ..core.errors import AppError
from ..services.jobs import submit_job
from .auth import Principal
from .deps import container, principal, rate_limited, visible_job
from .schemas import JobCreated, JobOut, JobPage

router = APIRouter(tags=["jobs"])

SSE_HEARTBEAT_S = 15.0
SSE_RESYNC_S = 10.0  # pub/sub can drop messages; re-read the row this often


@router.post(
    "/v1/jobs",
    response_model=JobCreated,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limited("submit"))],
    summary="Submit an image; returns a job id",
)
@router.post(
    "/generate",
    response_model=JobCreated,
    status_code=202,
    include_in_schema=False,
    dependencies=[Depends(rate_limited("submit"))],
)
async def create_job(
    response: Response,
    image: UploadFile = File(..., description="PNG, JPEG or WebP image of one object"),
    remove_background: bool = Form(True),
    foreground_ratio: Optional[float] = Form(None, ge=0.5, le=1.0),
    mc_resolution: Optional[int] = Form(None, ge=32, le=512),
    diffusion_steps: Optional[int] = Form(None, ge=10, le=150),
    recon_iters: Optional[int] = Form(None, ge=100, le=20000),
    reconstructor: Optional[Literal["instantmesh", "neus"]] = Form(
        None, description="instantmesh (fast, default) or neus (classic per-object fit)"
    ),
    texture: bool = Form(True, description="Bake a colour texture from the generated views"),
    texture_size: Optional[int] = Form(None, ge=256, le=2048),
    color_match: Optional[float] = Form(None, ge=0.0, le=1.0, description="Match texture colours to the photo"),
    target_faces: Optional[int] = Form(None, ge=0, le=500_000, description="0 keeps every triangle"),
    seed: Optional[int] = Form(None, ge=0, le=2**31 - 1),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    p: Principal = Depends(principal),
    c: Container = Depends(container),
) -> JobCreated:
    s = c.settings
    data = await image.read(s.max_upload_bytes + 1)
    options = GenerateOptions(
        remove_background=remove_background,
        foreground_ratio=foreground_ratio or s.default_foreground_ratio,
        mc_resolution=mc_resolution or s.default_mc_resolution,
        diffusion_steps=diffusion_steps or s.default_diffusion_steps,
        recon_iters=recon_iters or s.default_recon_iters,
        reconstructor=reconstructor or s.default_reconstructor,
        texture=texture,
        texture_size=texture_size or s.default_texture_size,
        color_match=s.default_color_match if color_match is None else color_match,
        target_faces=s.default_target_faces if target_faces is None else target_faces,
        seed=seed,
    )
    job, replayed = await submit_job(c, p, data, options, idempotency_key)
    jid = str(job.id)
    response.headers["Location"] = f"/v1/jobs/{jid}"
    if replayed:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replayed"] = "true"
    return JobCreated(job_id=jid, status=job.status, status_url=f"/v1/jobs/{jid}", events_url=f"/v1/jobs/{jid}/events")


@router.get(
    "/v1/jobs",
    response_model=JobPage,
    dependencies=[Depends(rate_limited("read"))],
    summary="List the caller's jobs, newest first",
)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None, description="next_cursor from the previous page"),
    p: Principal = Depends(principal),
    c: Container = Depends(container),
) -> JobPage:
    before = None
    if cursor:
        try:
            before = datetime.fromisoformat(cursor)
        except ValueError:
            raise AppError(400, "invalid_cursor", "Malformed cursor.") from None
    jobs = await c.jobs.list_for_owner(p.owner, limit=limit, before=before)
    nxt = jobs[-1].created_at.isoformat() if len(jobs) == limit else None
    return JobPage(items=[JobOut.from_job(j) for j in jobs], next_cursor=nxt)


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobOut,
    dependencies=[Depends(rate_limited("read"))],
    summary="Job status (poll this, or use /events)",
)
@router.get(
    "/jobs/{job_id}", response_model=JobOut, include_in_schema=False, dependencies=[Depends(rate_limited("read"))]
)
async def get_job(job_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> JobOut:
    return JobOut.from_job(await visible_job(job_id, p, c))


def _stream_file(c: Container, key: str, media_type: str, filename: Optional[str] = None) -> StreamingResponse:
    headers = {"Cache-Control": "private, max-age=3600"}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return StreamingResponse(c.storage.iter_chunks(key), media_type=media_type, headers=headers)


@router.get(
    "/v1/jobs/{job_id}/result",
    dependencies=[Depends(rate_limited("read"))],
    response_class=StreamingResponse,
    summary="Download the mesh (.glb)",
)
@router.get("/jobs/{job_id}/result", include_in_schema=False, dependencies=[Depends(rate_limited("read"))])
async def get_result(job_id: str, p: Principal = Depends(principal), c: Container = Depends(container)):
    job = await visible_job(job_id, p, c)
    if job.status == "failed":
        raise AppError(409, "job_failed", f"Job failed: {job.error}")
    if job.status != "succeeded" or not job.result_key:
        raise AppError(404, "not_ready", "The mesh is not ready yet.")
    if await c.storage.size(job.result_key) is None:
        raise AppError(410, "result_expired", "The mesh is no longer stored.")
    return _stream_file(c, job.result_key, "model/gltf-binary", f"{job_id}.glb")


@router.get(
    "/v1/jobs/{job_id}/views",
    dependencies=[Depends(rate_limited("read"))],
    response_class=StreamingResponse,
    summary="The six generated views as one PNG",
)
@router.get("/jobs/{job_id}/views", include_in_schema=False, dependencies=[Depends(rate_limited("read"))])
async def get_views(job_id: str, p: Principal = Depends(principal), c: Container = Depends(container)):
    job = await visible_job(job_id, p, c)
    if not job.views_key or await c.storage.size(job.views_key) is None:
        raise AppError(404, "not_ready", "No views for this job yet.")
    return _stream_file(c, job.views_key, "image/png")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get(
    "/v1/jobs/{job_id}/events",
    dependencies=[Depends(rate_limited("read"))],
    summary="Live progress as Server-Sent Events",
)
async def job_events(
    job_id: str, request: Request, p: Principal = Depends(principal), c: Container = Depends(container)
):
    """Starts with a `snapshot` event (the same body as GET /v1/jobs/{id}), then
    `progress`, `status`, `retrying` and finally `done` followed by a last
    snapshot. A comment line is sent every 15 s to keep proxies from closing."""
    job = await visible_job(job_id, p, c)

    async def stream():
        metrics.SSE_CONNECTIONS.inc()
        try:
            async with c.bus.subscribe(str(job.id)) as sub:
                current = await c.jobs.get(job.id)
                yield _sse("snapshot", JobOut.from_job(current).model_dump(mode="json"))
                if current.status in ("succeeded", "failed"):
                    yield _sse("done", {"status": current.status})
                    return
                last_beat = last_sync = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        return
                    event = await sub.get(timeout=1.0)
                    now = time.monotonic()
                    if event is not None:
                        if event.get("type") == "done":
                            final = await c.jobs.get(job.id)
                            yield _sse("snapshot", JobOut.from_job(final).model_dump(mode="json"))
                            yield _sse("done", event)
                            return
                        yield _sse(event.get("type", "message"), event)
                    if now - last_sync >= SSE_RESYNC_S:
                        last_sync = now
                        fresh = await c.jobs.get(job.id)
                        if fresh.status in ("succeeded", "failed"):
                            yield _sse("snapshot", JobOut.from_job(fresh).model_dump(mode="json"))
                            yield _sse("done", {"status": fresh.status})
                            return
                    if now - last_beat >= SSE_HEARTBEAT_S:
                        last_beat = now
                        yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            metrics.SSE_CONNECTIONS.dec()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
