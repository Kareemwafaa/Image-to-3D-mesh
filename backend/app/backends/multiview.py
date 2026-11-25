"""Default backend: Zero123++ multi-view diffusion + NeuS reconstruction.

    segmented image --(Zero123++)--> 6 views --(U^2-Net)--> 6 masks
        --(NeuS fit, known cameras)--> SDF --(marching cubes)--> .glb
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..pipeline import cameras, meshing, multiview, reconstruct, segmentation
from .base import GenerateOptions, ProgressFn, resolve_device

log = logging.getLogger(__name__)

# Share of the progress bar for each stage (rough timings on a GPU).
_MV = (0.05, 0.30)
_RECON = (0.35, 0.90)


class MultiViewNeuSBackend:
    name = "multiview"

    def __init__(self, model_id: str, pipeline_id: str, device: str) -> None:
        self.device = resolve_device(device)
        self.generator = multiview.Zero123PlusGenerator(model_id, pipeline_id, self.device)

    def load(self) -> None:
        self.generator.load()

    def generate(
        self,
        image: Image.Image,
        options: GenerateOptions,
        out_dir: Path,
        progress: ProgressFn,
    ) -> None:
        t0 = time.monotonic()
        progress(_MV[0], "multiview")
        views = self.generator.generate(image, steps=options.diffusion_steps, seed=options.seed)
        multiview.make_grid(views).save(out_dir / "views.png")
        t1 = time.monotonic()

        progress(_MV[1], "segmenting_views")
        masks = [segmentation.view_mask(v) for v in views]
        view_arr = multiview.views_to_arrays(views)
        mask_arr = np.stack(masks)

        progress(_RECON[0], "reconstructing")
        cfg = reconstruct.ReconConfig(iters=options.recon_iters)
        model = reconstruct.reconstruct(
            view_arr,
            mask_arr,
            cameras.zero123pp_cameras(view_arr.shape[1]),
            cfg,
            device=self.device,
            progress=lambda f: progress(_RECON[0] + f * (_RECON[1] - _RECON[0]), "reconstructing"),
            seed=options.seed or 0,
        )
        t2 = time.monotonic()

        progress(_RECON[1], "meshing")
        mesh = meshing.extract_mesh(model, options.mc_resolution, device=self.device)
        mesh.export(out_dir / "result.glb", file_type="glb")
        log.info(
            "multiview %.1fs, reconstruction %.1fs, meshing %.1fs, %d faces",
            t1 - t0, t2 - t1, time.monotonic() - t2, len(mesh.faces),
        )
