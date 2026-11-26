"""Relational schema: jobs are the source of truth; Redis only carries messages."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every database (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.succeeded, JobStatus.failed)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    key_hash: Mapped[str] = mapped_column(String(64))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    submit_per_minute: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_inflight: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("owner", "idempotency_key", name="uq_jobs_owner_idempotency_key"),
        Index("ix_jobs_status_updated_at", "status", "updated_at"),
        Index("ix_jobs_owner_created_at", "owner", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # "key:<api key id>" or "ip:<address>": who rate limits and idempotency keys belong to
    owner: Mapped[str] = mapped_column(String(80))
    api_key_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.queued.value)
    stage: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    dead_lettered: Mapped[bool] = mapped_column(Boolean, default=False)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    backend: Mapped[str] = mapped_column(String(32))
    input_key: Mapped[str] = mapped_column(String(255))
    input_sha256: Mapped[str] = mapped_column(String(64))
    result_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    views_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    request_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    traceparent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime(), nullable=True)

    @property
    def status_enum(self) -> JobStatus:
        return JobStatus(self.status)
