from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from ..db.models import ApiKey, Job


class JobCreated(BaseModel):
    job_id: str
    status: str
    status_url: str
    events_url: str


class JobOut(BaseModel):
    job_id: str
    status: str = Field(description="queued | running | succeeded | failed")
    stage: Optional[str] = None
    progress: float
    result_url: Optional[str] = None
    views_url: Optional[str] = None
    events_url: str
    error: Optional[str] = None
    error_code: Optional[str] = None
    attempts: int
    dead_lettered: bool = False
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None

    @classmethod
    def from_job(cls, job: Job) -> JobOut:
        jid = str(job.id)
        return cls(
            job_id=jid,
            status=job.status,
            stage=job.stage,
            progress=round(job.progress or 0.0, 3),
            result_url=f"/v1/jobs/{jid}/result" if job.status == "succeeded" else None,
            views_url=f"/v1/jobs/{jid}/views" if job.views_key else None,
            events_url=f"/v1/jobs/{jid}/events",
            error=job.error if job.status == "failed" else None,
            error_code=job.error_code if job.status == "failed" else None,
            attempts=job.attempts,
            dead_lettered=job.dead_lettered,
            created_at=job.created_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
        )


class JobPage(BaseModel):
    items: list[JobOut]
    next_cursor: Optional[str] = None


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    is_admin: bool = False
    submit_per_minute: Optional[float] = Field(default=None, gt=0, le=10_000)
    max_inflight: Optional[int] = Field(default=None, ge=1, le=1000)


class ApiKeyOut(BaseModel):
    id: str
    name: str
    prefix: str
    is_admin: bool
    submit_per_minute: Optional[float]
    max_inflight: Optional[int]
    created_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]

    @classmethod
    def from_key(cls, k: ApiKey) -> ApiKeyOut:
        return cls(
            id=str(k.id),
            name=k.name,
            prefix=k.prefix,
            is_admin=k.is_admin,
            submit_per_minute=k.submit_per_minute,
            max_inflight=k.max_inflight,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            revoked_at=k.revoked_at,
        )


class ApiKeyCreated(ApiKeyOut):
    api_key: str = Field(description="Shown once. Store it now; only a hash is kept.")


class HealthOut(BaseModel):
    """Legacy status endpoint used by the web app."""

    status: str
    mode: str
    backend: str
    model_loaded: bool
    model_error: Optional[str]
    queue_size: Optional[int]
    workers: int
