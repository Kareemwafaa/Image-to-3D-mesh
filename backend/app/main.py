"""HTTP API: upload an image, poll the job, download the .glb."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .backends import GenerateOptions
from .config import Settings, get_settings
from .jobs import RESULT_FILE, VIEWS_FILE, JobManager, JobRecord, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Inputs larger than this on either side are shrunk before saving; the models
# work at 320-512 px, so bigger images only cost upload and decode time.
MAX_SIDE = 2048


class GenerateResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: Optional[str]
    progress: float
    result_url: Optional[str]
    views_url: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    model_error: Optional[str]
    queue_size: Optional[int]


def local_manager_factory(settings: Settings) -> JobManager:
    from .backends import create_backend
    from .jobs import LocalJobManager

    return LocalJobManager(
        create_backend(settings), settings.data_dir, settings.job_ttl_hours, preload=settings.preload_model
    )


def create_app(
    settings: Optional[Settings] = None,
    manager_factory: Callable[[Settings], JobManager] = local_manager_factory,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = manager_factory(settings)
        manager.start()
        app.state.jobs = manager
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="Image to 3D mesh", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    def jobs(request: Request) -> JobManager:
        return request.app.state.jobs

    def find(request: Request, job_id: str) -> JobRecord:
        job = jobs(request).get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
        return job

    def to_response(job: JobRecord) -> JobResponse:
        return JobResponse(
            job_id=job.id,
            status=job.status,
            stage=job.stage,
            progress=round(job.progress, 3),
            result_url=f"/jobs/{job.id}/result" if job.status == JobStatus.succeeded else None,
            views_url=f"/jobs/{job.id}/views" if job.has_views else None,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        info = jobs(request).health()
        return HealthResponse(
            status="ok" if not info.get("model_error") else "degraded",
            backend=settings.backend,
            model_loaded=bool(info.get("model_loaded")),
            model_error=info.get("model_error"),
            queue_size=info.get("queue_size"),
        )

    @app.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
    async def generate(
        request: Request,
        image: UploadFile = File(..., description="PNG, JPEG or WebP image of one object"),
        remove_background: bool = Form(True),
        foreground_ratio: Optional[float] = Form(None, ge=0.5, le=1.0),
        mc_resolution: Optional[int] = Form(None, ge=32, le=512),
        diffusion_steps: Optional[int] = Form(None, ge=10, le=150),
        recon_iters: Optional[int] = Form(None, ge=100, le=20000),
        seed: Optional[int] = Form(None, ge=0, le=2**31 - 1),
    ) -> GenerateResponse:
        data = await image.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"Image is larger than {settings.max_upload_mb:g} MB")
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
            pil = Image.open(io.BytesIO(data))
            pil.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not read the file as an image") from exc
        if max(pil.size) > MAX_SIDE:
            pil.thumbnail((MAX_SIDE, MAX_SIDE))

        options = GenerateOptions(
            remove_background=remove_background,
            foreground_ratio=foreground_ratio or settings.default_foreground_ratio,
            mc_resolution=mc_resolution or settings.default_mc_resolution,
            diffusion_steps=diffusion_steps or settings.default_diffusion_steps,
            recon_iters=recon_iters or settings.default_recon_iters,
            seed=seed,
        )
        job = jobs(request).submit(pil, options)
        return GenerateResponse(job_id=job.id)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, request: Request) -> JobResponse:
        return to_response(find(request, job_id))

    @app.get("/jobs/{job_id}/result", response_class=StreamingResponse)
    def get_result(job_id: str, request: Request) -> StreamingResponse:
        job = find(request, job_id)
        if job.status == JobStatus.failed:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Job failed: {job.error}")
        data = jobs(request).read_file(job_id, RESULT_FILE) if job.status == JobStatus.succeeded else None
        if data is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Mesh is not ready yet")
        return StreamingResponse(
            data,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.glb"'},
        )

    @app.get("/jobs/{job_id}/views", response_class=StreamingResponse)
    def get_views(job_id: str, request: Request) -> StreamingResponse:
        """The six generated views as one 3x2 PNG (multiview backend only)."""
        find(request, job_id)
        data = jobs(request).read_file(job_id, VIEWS_FILE)
        if data is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No views for this job (yet)")
        return StreamingResponse(data, media_type="image/png")

    return app


app = create_app()
