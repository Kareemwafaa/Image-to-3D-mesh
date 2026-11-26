"""All runtime settings, from environment variables (12-factor).

Two presets share one code base:
  * MODE=full  - Postgres, Redis, any S3 store, separate worker processes (docker compose)
  * MODE=lite  - SQLite, local disk, in-process queue and worker (one notebook GPU)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    mode: Literal["full", "lite"] = "full"
    service_name: str = "image-to-3d"
    environment: str = "dev"

    # --- HTTP ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:4200"]
    )
    cors_origin_regex: Optional[str] = None
    max_upload_mb: float = 10.0

    # --- Infrastructure ------------------------------------------------------
    database_url: str = "postgresql+asyncpg://i3d:i3d@localhost:5432/i3d"
    redis_url: str = "redis://localhost:6379/0"
    storage_backend: Literal["s3", "local"] = "s3"
    storage_dir: str = "./data/files"
    s3_endpoint_url: Optional[str] = "http://localhost:8333"
    s3_bucket: str = "image-to-3d"
    s3_access_key: str = "i3d-access-key"  # matches deploy/seaweedfs/s3.json
    s3_secret_key: str = "i3d-secret-key-change-me"
    s3_region: str = "us-east-1"

    # --- Auth and limits -----------------------------------------------------
    # required: every call needs a key; optional: anonymous allowed with per-IP limits
    auth_mode: Literal["required", "optional"] = "optional"
    admin_api_key: Optional[str] = None  # bootstrap admin key, i3d_<prefix>_<secret>
    trust_proxy_headers: bool = False  # use X-Forwarded-For for client IPs
    anon_submit_per_minute: float = 5
    anon_submit_burst: int = 5
    key_submit_per_minute: float = 30
    key_submit_burst: int = 10
    read_per_minute: float = 600
    read_burst: int = 120
    anon_max_inflight: int = 2
    key_max_inflight: int = 10

    # --- Jobs ------------------------------------------------------------------
    job_timeout_s: float = 15 * 60
    max_attempts: int = 3
    retry_base_delay_s: float = 5.0
    retry_max_delay_s: float = 120.0
    visibility_timeout_s: float = 60.0  # lease; renewed by heartbeats while a job runs
    heartbeat_interval_s: float = 15.0
    stale_queued_after_s: float = 600.0  # reconciler re-enqueues jobs stuck in queued
    progress_db_interval_s: float = 1.0  # throttle progress writes to the database
    job_ttl_hours: float = 0  # 0 = keep forever

    # --- Model -----------------------------------------------------------------
    model_backend: Literal["multiview", "mock"] = "multiview"
    device: str = "auto"
    preload_model: bool = True
    zero123pp_model: str = "sudo-ai/zero123plus-v1.2"
    zero123pp_pipeline: str = "sudo-ai/zero123plus-pipeline"
    # "instantmesh": the white-background UNet InstantMesh was trained with (best for both reconstructors)
    zero123pp_unet: Literal["instantmesh", "original"] = "instantmesh"
    instantmesh_variant: Literal["large", "base"] = "large"
    # Load InstantMesh at start-up. Set false to save ~2 GB of GPU memory if you only use NeuS.
    preload_instantmesh: bool = True
    default_reconstructor: Literal["instantmesh", "neus"] = Field(
        "instantmesh", validation_alias=AliasChoices("RECONSTRUCTOR", "DEFAULT_RECONSTRUCTOR")
    )
    default_texture_size: int = Field(1024, validation_alias=AliasChoices("TEXTURE_SIZE", "DEFAULT_TEXTURE_SIZE"))
    default_target_faces: int = Field(40_000, validation_alias=AliasChoices("TARGET_FACES", "DEFAULT_TARGET_FACES"))
    # Defaults for the per-request form fields (short env names kept for the notebooks).
    default_mc_resolution: int = Field(256, validation_alias=AliasChoices("MC_RESOLUTION", "DEFAULT_MC_RESOLUTION"))
    default_foreground_ratio: float = Field(
        0.85, validation_alias=AliasChoices("FOREGROUND_RATIO", "DEFAULT_FOREGROUND_RATIO")
    )
    default_diffusion_steps: int = Field(
        75, validation_alias=AliasChoices("DIFFUSION_STEPS", "DEFAULT_DIFFUSION_STEPS")
    )
    default_recon_iters: int = Field(2500, validation_alias=AliasChoices("RECON_ITERS", "DEFAULT_RECON_ITERS"))

    # --- Worker ------------------------------------------------------------------
    worker_id: Optional[str] = None
    worker_metrics_port: int = 9100

    # --- Observability -------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter_otlp_endpoint: Optional[str] = None
    metrics_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def is_lite(self) -> bool:
        return self.mode == "lite"


def lite_defaults() -> dict:
    """Overrides applied by `python -m app.lite` unless set in the environment."""
    return {
        "MODE": "lite",
        "DATABASE_URL": "sqlite+aiosqlite:///./data/lite.db",
        "STORAGE_BACKEND": "local",
        "AUTH_MODE": "optional",
        "LOG_JSON": "false",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
