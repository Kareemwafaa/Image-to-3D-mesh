"""The API gateway (FastAPI app factory).

`uvicorn app.main:app` serves the gateway only (full mode; workers run
separately). `python -m app.lite` serves it with an in-process worker.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .backends.base import MeshBackend
from .container import Container, build_container
from .core import metrics
from .core.config import Settings, get_settings
from .core.errors import AppError
from .core.logging import get_logger, setup_logging
from .core.tracing import instrument_app, instrument_engine, setup_tracing

log = get_logger("app.gateway")

OPENAPI_TAGS = [
    {"name": "jobs", "description": "Submit images, follow progress, download meshes."},
    {"name": "admin", "description": "API keys, dead letters, queue state. Admin key required."},
    {"name": "health", "description": "Liveness, readiness, metrics."},
]


def problem(status: int, code: str, detail: str, request: Request, headers: Optional[dict] = None) -> JSONResponse:
    """RFC 9457 problem details."""
    body = {
        "type": f"https://errors.image-to-3d.dev/{code}",
        "title": code.replace("_", " ").capitalize(),
        "status": status,
        "code": code,
        "detail": detail,
        "instance": request.url.path,
        "request_id": getattr(request.state, "request_id", None),
    }
    return JSONResponse(body, status_code=status, headers=headers, media_type="application/problem+json")


def create_app(
    settings: Optional[Settings] = None,
    container: Optional[Container] = None,
    run_worker: Optional[bool] = None,
    backend: Optional[MeshBackend] = None,
) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level, s.log_json)
    tracing_on = setup_tracing(f"{s.service_name}-api", s.otel_exporter_otlp_endpoint)
    run_worker = s.is_lite if run_worker is None else run_worker

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(s)
        await c.start(migrate=s.is_lite)
        instrument_engine(c.engine)
        app.state.container = c
        worker_task = None
        if run_worker:
            from .worker.pipeline import create_backend
            from .worker.runner import Worker

            worker = Worker(c, backend or create_backend(s))
            app.state.worker = worker
            worker_task = asyncio.create_task(worker.run(), name="in-process-worker")
        log.info(
            "gateway_started", mode=s.mode, backend=s.model_backend, in_process_worker=run_worker, tracing=tracing_on
        )
        try:
            yield
        finally:
            if worker_task is not None:
                app.state.worker.stop()
                try:
                    await asyncio.wait_for(worker_task, timeout=30)
                except TimeoutError:
                    worker_task.cancel()
            if container is None:
                await c.close()

    app = FastAPI(
        title="Image to 3D mesh API",
        version="1.0.0",
        description="Single image to textured 3D mesh: Zero123++ multi-view diffusion + NeuS reconstruction.",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.settings = s

    allow_all = s.cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else s.cors_origins,
        allow_origin_regex=s.cors_origin_regex,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "Location",
            "Retry-After",
            "RateLimit-Limit",
            "RateLimit-Remaining",
            "X-Request-ID",
            "Idempotent-Replayed",
        ],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - started
            if s.metrics_enabled and path not in ("/metrics",):
                metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                metrics.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            if path not in ("/metrics", "/healthz", "/readyz", "/health"):
                log.info("http_request", method=request.method, path=path, status=status, ms=round(elapsed * 1000, 1))

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return problem(exc.status, exc.code, exc.detail, request, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return problem(exc.status_code, code, str(exc.detail), request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        detail = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())
        return problem(422, "validation_error", detail, request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error")
        return problem(500, "internal_error", "Internal server error.", request)

    from .api import routes_admin, routes_health, routes_jobs

    app.include_router(routes_health.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_admin.router)
    instrument_app(app)
    return app


def __getattr__(name: str):
    # `uvicorn app.main:app` builds the app lazily, so importing this module
    # (tests, tools) has no side effects.
    if name == "app":
        return create_app()
    raise AttributeError(name)
