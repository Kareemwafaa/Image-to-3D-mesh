"""A backend that needs no GPU or model weights.

It returns a unit cube coloured with the image's average colour, so the whole
API (upload, queue, status, download) can be run and tested on any machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .base import GenerateOptions, ProgressFn


class MockBackend:
    name = "mock"

    def load(self) -> None:
        pass

    def generate(
        self,
        image: Image.Image,
        options: GenerateOptions,
        out_dir: Path,
        progress: ProgressFn,
    ) -> None:
        # Stand-in for the multiview backend's intermediate views.
        image.convert("RGB").save(out_dir / "views.png")
        progress(0.3, "mock")
        rgb = np.asarray(image.convert("RGB").resize((8, 8))).reshape(-1, 3).mean(axis=0)
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.visual.vertex_colors = np.tile(np.append(rgb, 255).astype(np.uint8), (len(mesh.vertices), 1))
        progress(0.8)
        mesh.export(out_dir / "result.glb", file_type="glb")
