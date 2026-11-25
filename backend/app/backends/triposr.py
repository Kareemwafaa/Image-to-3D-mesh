"""Feed-forward baseline: TripoSR (https://github.com/VAST-AI-Research/TripoSR), MIT licence.

TripoSR is not on PyPI. The Dockerfile clones the repo and puts it on
PYTHONPATH so that `import tsr` works.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ..pipeline.segmentation import composite_on_gray
from .base import GenerateOptions, ProgressFn, resolve_device

log = logging.getLogger(__name__)


class TripoSRBackend:
    name = "triposr"

    def __init__(self, model_id: str, device: str, chunk_size: int) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.chunk_size = chunk_size
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from tsr.system import TSR

        log.info("Loading TripoSR (%s) on %s", self.model_id, self.device)
        model = TSR.from_pretrained(self.model_id, config_name="config.yaml", weight_name="model.ckpt")
        model.renderer.set_chunk_size(self.chunk_size)
        model.to(self.device)
        self.model = model

    def generate(
        self,
        image: Image.Image,
        options: GenerateOptions,
        out_dir: Path,
        progress: ProgressFn,
    ) -> None:
        import torch

        self.load()
        assert self.model is not None
        rgb = composite_on_gray(image)
        progress(0.3, "inference")
        with torch.no_grad():
            scene_codes = self.model([rgb], device=self.device)
        progress(0.6, "meshing")
        meshes = self.model.extract_mesh(scene_codes, True, resolution=options.mc_resolution)
        meshes[0].export(str(out_dir / "result.glb"), file_type="glb")
