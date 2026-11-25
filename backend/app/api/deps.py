from __future__ import annotations

import uuid

from fastapi import Depends, Request, Response

from ..container import Container
from ..core import metrics
from ..core.errors import AppError
from ..db.models import Job
from .auth import Principal, resolve_principal


def container(request: Request) -> Container:
    return request.app.state.container


async def principal(request: Request) -> Principal:
    p = getattr(request.state, "principal", None)
    if p is None:
        p = await resolve_principal(request)
        request.state.principal = p
    return p


async def admin(p: Principal = Depends(principal)) -> Principal:
    if not p.is_admin:
        raise AppError(403, "forbidden", "Admin API key required.")
    return p


def rate_limited(bucket: str):
    """Dependency factory: one token from the caller's bucket (`submit` or `read`)."""

    async def dep(response: Response, p: Principal = Depends(principal), c: Container = Depends(container)) -> None:
        s = c.settings
        if p.is_admin:
            return
        if bucket == "submit":
            per_min, burst = p.submit_limits(s)
        else:
            per_min, burst = s.read_per_minute, s.read_burst
        decision = await c.limiter.hit(f"{bucket}:{p.owner}", per_min / 60.0, burst)
        for k, v in decision.headers().items():
            response.headers[k] = v
        if not decision.allowed:
            metrics.RATE_LIMITED.labels(bucket=bucket).inc()
            raise AppError(
                429,
                "rate_limited",
                f"Too many requests; retry in {decision.headers()['Retry-After']}s.",
                decision.headers(),
            )

    return dep


async def visible_job(job_id: str, p: Principal, c: Container) -> Job:
    """Anonymous jobs are capability URLs (anyone with the id may read them);
    jobs created with an API key are visible to that key and to admins."""
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise AppError(404, "job_not_found", "Job not found.") from None
    job = await c.jobs.get(jid)
    if job is None:
        raise AppError(404, "job_not_found", "Job not found.") from None
    if job.owner.startswith("key:") and not p.is_admin and job.owner != p.owner:
        raise AppError(404, "job_not_found", "Job not found.") from None
    return job
