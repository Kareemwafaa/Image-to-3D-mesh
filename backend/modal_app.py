"""Deploy the API and the GPU pipeline on Modal (https://modal.com).

    pip install modal && modal setup          # once
    CORS_ORIGINS=https://your-frontend.example modal deploy modal_app.py

`modal deploy` prints the API URL (https://<workspace>--image-to-3d-api.modal.run).
For a dev loop with live reload, use `modal serve modal_app.py`.

Layout:
  * `api`        - CPU container running the FastAPI app (cheap, scales freely)
  * `MeshWorker` - GPU container running the multiview pipeline, scales to
                   zero when idle; weights are baked into its image
  * `cleanup`    - deletes jobs older than JOB_TTL_HOURS every 6 hours
State: job records in a modal.Dict, files in a modal.Volume.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

APP_NAME = "image-to-3d"
GPU = os.environ.get("MODAL_GPU", "A10G")  # GUESS: 24 GB is plenty; L4 also works, slower
VOLUME_MOUNT = "/vol"
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", "24"))
ZERO123PP_MODEL = "sudo-ai/zero123plus-v1.2"
ZERO123PP_PIPELINE = "sudo-ai/zero123plus-pipeline"

app = modal.App(APP_NAME)
jobs_dict = modal.Dict.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
volume = modal.Volume.from_name(f"{APP_NAME}-files", create_if_missing=True)

here = Path(__file__).parent


def _download_weights() -> None:
    """Runs once at image build time, so containers start without downloads."""
    from huggingface_hub import snapshot_download

    snapshot_download(ZERO123PP_MODEL)
    snapshot_download(ZERO123PP_PIPELINE)
    import rembg

    rembg.new_session("u2net")


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(here / "requirements-gpu.txt"))
    .env({"HF_HOME": "/models/hf", "U2NET_HOME": "/models/u2net"})
    .run_function(_download_weights)
    .add_local_python_source("app")
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.115", "python-multipart>=0.0.9", "pillow>=10.1", "numpy>=1.26,<2")
    .env({"CORS_ORIGINS": os.environ.get("CORS_ORIGINS", "http://localhost:3000"), "MODEL_BACKEND": "multiview"})
    .add_local_python_source("app")
)


@app.cls(
    image=gpu_image,
    gpu=GPU,
    volumes={VOLUME_MOUNT: volume},
    timeout=20 * 60,
    scaledown_window=5 * 60,  # keep a warm GPU for 5 minutes after the last job
    max_containers=int(os.environ.get("MODAL_MAX_GPUS", "2")),  # cost cap
)
class MeshWorker:
    @modal.enter()
    def load(self) -> None:
        from app.backends.multiview import MultiViewNeuSBackend
        from app.pipeline import segmentation

        self.backend = MultiViewNeuSBackend(ZERO123PP_MODEL, ZERO123PP_PIPELINE, "cuda")
        self.backend.load()
        segmentation._rembg_session()

    @modal.method()
    def run(self, job_id: str, image_png: bytes, options: dict) -> None:
        from app.backends import GenerateOptions
        from app.jobs import INPUT_FILE, JobRecord, _now, execute_job

        job_dir = Path(VOLUME_MOUNT) / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / INPUT_FILE).write_bytes(image_png)
        record = JobRecord.from_dict(jobs_dict[job_id])

        def update(**fields) -> None:
            # Files must be committed before the record says they exist,
            # or the API could look for them too early.
            newly_has_views = fields.get("has_views") and not record.has_views
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = _now()
            if newly_has_views or record.finished:
                volume.commit()
            jobs_dict[job_id] = record.to_dict()

        execute_job(self.backend, job_dir, GenerateOptions.from_dict(options), update)


@app.function(image=api_image)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    from app.config import Settings
    from app.main import create_app
    from app.modal_jobs import ModalJobManager

    worker = MeshWorker()

    def spawn(job_id: str, image_png: bytes, options: dict) -> None:
        worker.run.spawn(job_id, image_png, options)

    return create_app(Settings(), lambda _settings: ModalJobManager(jobs_dict, volume, spawn))


@app.function(image=api_image, volumes={VOLUME_MOUNT: volume}, schedule=modal.Period(hours=6))
def cleanup() -> None:
    import shutil
    from datetime import datetime

    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for job_id, data in list(jobs_dict.items()):
        if data.get("status") in ("succeeded", "failed") and datetime.fromisoformat(data["updated_at"]).timestamp() < cutoff:
            shutil.rmtree(Path(VOLUME_MOUNT) / "jobs" / job_id, ignore_errors=True)
            jobs_dict.pop(job_id)
    volume.commit()
