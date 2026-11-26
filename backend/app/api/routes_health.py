from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..container import Container
from .deps import container
from .schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness: the process is up")
async def healthz() -> dict:
    return {"status": "ok"}


async def _check(name: str, coro, timeout: float = 2.0) -> tuple[str, str]:
    try:
        await asyncio.wait_for(coro, timeout)
        return name, "ok"
    except Exception as exc:  # report, don't raise
        return name, f"error: {type(exc).__name__}: {exc}"[:200]


@router.get("/readyz", summary="Readiness: dependencies reachable (and, in lite mode, model loaded)")
async def readyz(c: Container = Depends(container)) -> JSONResponse:
    results = dict(
        await asyncio.gather(
            _check("database", c.jobs.ping()),
            _check("queue", c.queue.ping()),
            _check("storage", c.storage.ping()),
            _check("bus", c.bus.ping()),
        )
    )
    workers = await c.bus.workers()
    results["workers"] = str(len(workers))
    ok = all(v == "ok" for k, v in results.items() if k != "workers")
    if c.settings.is_lite:
        loaded = any(w.get("model_loaded") for w in workers)
        results["model"] = "ok" if loaded else "loading"
        ok = ok and loaded
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": results}, status_code=200 if ok else 503)


@router.get("/health", response_model=HealthOut, summary="Status summary for the web app")
async def health(c: Container = Depends(container)) -> HealthOut:
    workers = await c.bus.workers()
    errors = [w.get("model_error") for w in workers if w.get("model_error")]
    try:
        depth = await c.queue.depth()
        queue_size = depth.get("ready", 0) + depth.get("delayed", 0)
    except Exception:
        queue_size = None
    loaded = any(w.get("model_loaded") for w in workers)
    return HealthOut(
        status="ok" if loaded else ("degraded" if errors else "starting"),
        mode=c.settings.mode,
        backend=c.settings.model_backend,
        model_loaded=loaded,
        model_error=errors[0] if errors else None,
        queue_size=queue_size,
        workers=len(workers),
    )


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
