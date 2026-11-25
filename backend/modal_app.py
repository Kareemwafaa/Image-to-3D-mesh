"""Optional: serve the backend on a Modal GPU (https://modal.com), billed per second.

    pip install modal && modal setup          # once
    modal deploy modal_app.py                 # prints https://<workspace>--image-to-3d-web.modal.run
    modal serve modal_app.py                  # dev loop with live reload

This runs lite mode (ADR 0008): one GPU container holds the gateway, the
in-process queue and the worker, with SQLite and files on the container's
disk. The container scales to zero after SCALEDOWN_S idle seconds, and job
history goes with it, so download results when a job finishes. For a
persistent multi-worker deployment, use the docker-compose stack instead.

Settings come from the same environment variables as everywhere else, e.g.
    CORS_ORIGINS=https://kareemwafaa.github.io MODAL_GPU=L4 modal deploy modal_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "image-to-3d"
GPU = os.environ.get("MODAL_GPU", "A10G")  # 24 GB; L4 also works, a little slower
SCALEDOWN_S = int(os.environ.get("MODAL_SCALEDOWN_S", "300"))

here = Path(__file__).parent

# Passed through to the container at deploy time (pydantic-settings reads them).
PASSTHROUGH = (
    "CORS_ORIGINS",
    "CORS_ORIGIN_REGEX",
    "AUTH_MODE",
    "ADMIN_API_KEY",
    "DIFFUSION_STEPS",
    "RECON_ITERS",
    "MC_RESOLUTION",
    "ANON_SUBMIT_PER_MINUTE",
    "ANON_MAX_INFLIGHT",
)
runtime_env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
runtime_env.setdefault("CORS_ORIGINS", "*")


def _download_weights() -> None:
    """Runs once at image build time, so containers start without downloads."""
    from huggingface_hub import snapshot_download

    snapshot_download("sudo-ai/zero123plus-v1.2")
    snapshot_download("sudo-ai/zero123plus-pipeline")
    import rembg

    rembg.new_session("u2net")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(here / "requirements" / "lite-gpu.txt"))
    .env({"HF_HOME": "/models/hf", "U2NET_HOME": "/models/u2net"})
    .run_function(_download_weights)
    .env(
        {
            "MODE": "lite",
            "DATABASE_URL": "sqlite+aiosqlite:////tmp/i3d/lite.db",
            "STORAGE_BACKEND": "local",
            "STORAGE_DIR": "/tmp/i3d/files",
            "MODEL_BACKEND": "multiview",
            "TRUST_PROXY_HEADERS": "true",  # client IPs for per-IP limits
            "LOG_JSON": "true",
            **runtime_env,
        }
    )
    .add_local_dir(str(here / "app"), "/root/app")
    .add_local_file(str(here / "alembic.ini"), "/root/alembic.ini")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=30 * 60,
    scaledown_window=SCALEDOWN_S,
    max_containers=1,  # lite mode keeps its queue in memory: exactly one container
)
@modal.concurrent(max_inputs=64)  # many HTTP requests (polls, SSE) share the one GPU worker
@modal.asgi_app()
def web():
    from app.core.config import Settings
    from app.main import create_app

    return create_app(Settings())
