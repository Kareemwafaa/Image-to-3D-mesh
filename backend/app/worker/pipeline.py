"""Glue between a job and the ML pipeline: build the backend, run one image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..backends.base import GenerateOptions, MeshBackend, ProgressFn
from ..core.config import Settings
from ..core.errors import JobError
from ..pipeline.segmentation import prepare

RESULT_FILE = "result.glb"
VIEWS_FILE = "views.png"


def create_backend(s: Settings) -> MeshBackend:
    if s.model_backend == "multiview":
        from ..backends.multiview import MultiViewBackend

        return MultiViewBackend(
            s.zero123pp_model,
            s.zero123pp_pipeline,
            s.device,
            unet=s.zero123pp_unet,
            instantmesh_variant=s.instantmesh_variant,
            preload_instantmesh=s.preload_instantmesh,
            low_vram=s.low_vram,
        )
    from ..backends.mock import MockBackend

    return MockBackend()


def run_pipeline(
    backend: MeshBackend,
    input_path: Path,
    out_dir: Path,
    options: GenerateOptions,
    progress: ProgressFn,
) -> None:
    """Blocking; runs in a worker thread. Raises on failure."""
    progress(0.01, "segmenting")
    with Image.open(input_path) as img:
        image = prepare(img, options.remove_background, options.foreground_ratio)
    image.save(out_dir / "segmented.png")
    progress(0.04, "segmenting")
    backend.generate(image, options, out_dir, progress)
    if not (out_dir / RESULT_FILE).is_file():
        raise JobError("The backend finished without writing a mesh", code="no_output")
