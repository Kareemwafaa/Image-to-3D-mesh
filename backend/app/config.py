"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Which pipeline turns the image into a mesh:
    #   "multiview" (default) - Zero123++ multi-view diffusion + NeuS reconstruction
    #   "triposr"             - feed-forward baseline (one network pass)
    #   "mock"                - no GPU, returns a coloured cube; for development and tests
    backend: str = field(default_factory=lambda: os.getenv("MODEL_BACKEND", "multiview").lower())
    # "cuda", "cpu" or "auto" (cuda when available).
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    max_upload_mb: float = field(default_factory=lambda: float(os.getenv("MAX_UPLOAD_MB", "10")))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if o.strip()
        ]
    )
    # Load the model at startup instead of on the first job.
    preload_model: bool = field(default_factory=lambda: _bool("PRELOAD_MODEL", True))
    # Finished jobs and their files are deleted after this many hours (0 = keep).
    job_ttl_hours: float = field(default_factory=lambda: float(os.getenv("JOB_TTL_HOURS", "24")))

    # TripoSR knobs (defaults match TripoSR's run.py).
    triposr_model: str = field(default_factory=lambda: os.getenv("TRIPOSR_MODEL", "stabilityai/TripoSR"))
    triposr_chunk_size: int = field(default_factory=lambda: int(os.getenv("TRIPOSR_CHUNK_SIZE", "8192")))
    default_mc_resolution: int = field(default_factory=lambda: int(os.getenv("MC_RESOLUTION", "256")))
    default_foreground_ratio: float = field(
        default_factory=lambda: float(os.getenv("FOREGROUND_RATIO", "0.85"))
    )

    # Multiview pipeline knobs.
    zero123pp_model: str = field(default_factory=lambda: os.getenv("ZERO123PP_MODEL", "sudo-ai/zero123plus-v1.2"))
    zero123pp_pipeline: str = field(
        default_factory=lambda: os.getenv("ZERO123PP_PIPELINE", "sudo-ai/zero123plus-pipeline")
    )
    default_diffusion_steps: int = field(default_factory=lambda: int(os.getenv("DIFFUSION_STEPS", "75")))
    default_recon_iters: int = field(default_factory=lambda: int(os.getenv("RECON_ITERS", "2500")))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def get_settings() -> Settings:
    return Settings()
