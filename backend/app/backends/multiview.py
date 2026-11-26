"""Default backend: Zero123++ multi-view diffusion, then one of two reconstructors.

segmented image --(Zero123++)--> 6 views --(U^2-Net)--> 6 masks
    --> reconstruction, one of:
          instantmesh  feed-forward LRM + FlexiCubes, ~10 s (default)
          neus         per-object NeuS fit + marching cubes, minutes ("classic")
    --> cleanup (floaters, holes, Taubin smoothing, quadric decimation)
    --> texture baked from the 6 views --> .glb
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..pipeline import cameras, multiview, postprocess, segmentation
from .base import RECONSTRUCTORS, GenerateOptions, ProgressFn, resolve_device

log = logging.getLogger(__name__)

# Share of the progress bar for each stage (rough timings on a GPU).
_MV = (0.05, 0.30)
_RECON = (0.35, 0.80)
_CLEAN = 0.82
_TEXTURE = 0.88

# Taubin iterations when the request does not say. NeuS surfaces are noisier.
_SMOOTHING = {"instantmesh": 3, "neus": 10}


class MultiViewBackend:
    name = "multiview"

    def __init__(
        self,
        model_id: str,
        pipeline_id: str,
        device: str,
        unet: str = "instantmesh",
        instantmesh_variant: str = "large",
        preload_instantmesh: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.generator = multiview.Zero123PlusGenerator(model_id, pipeline_id, self.device, unet=unet)
        self.instantmesh_variant = instantmesh_variant
        self.preload_instantmesh = preload_instantmesh
        self._lrm = None

    @property
    def lrm(self):
        if self._lrm is None:
            from ..pipeline.instantmesh import InstantMeshReconstructor

            self._lrm = InstantMeshReconstructor(self.instantmesh_variant, self.device)
        return self._lrm

    def load(self) -> None:
        self.generator.load()
        if self.preload_instantmesh:
            self.lrm.load()

    # --- reconstructors -------------------------------------------------------------
    def _instantmesh(self, view_arr: np.ndarray, progress) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        progress(_RECON[0], "reconstructing")
        _free_cuda_cache()
        out = self.lrm.reconstruct(view_arr)
        return out.vertices, out.faces, out.vertex_colors

    def _neus(self, view_arr, mask_arr, options: GenerateOptions, progress) -> tuple[np.ndarray, ...]:
        from ..pipeline import meshing, reconstruct

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
        progress(_RECON[1], "meshing")
        mesh = meshing.extract_mesh(model, options.mc_resolution, device=self.device)
        return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.vertex_colors)

    # --- the job ------------------------------------------------------------------------------
    def generate(
        self,
        image: Image.Image,
        options: GenerateOptions,
        out_dir: Path,
        progress: ProgressFn,
    ) -> None:
        from ..pipeline import meshing

        method = options.reconstructor
        if method not in RECONSTRUCTORS:
            raise ValueError(f"Unknown reconstructor {method!r}")
        t0 = time.monotonic()
        progress(_MV[0], "multiview")
        views = self.generator.generate(image, steps=options.diffusion_steps, seed=options.seed)
        multiview.make_grid(views).save(out_dir / "views.png")
        t1 = time.monotonic()

        progress(_MV[1], "segmenting_views")
        mask_arr = np.stack([segmentation.view_mask(v) for v in views])
        view_arr = multiview.views_to_arrays(views)

        if method == "instantmesh":
            verts, faces, colors = self._instantmesh(view_arr, progress)
        else:
            verts, faces, colors = self._neus(view_arr, mask_arr, options, progress)
        t2 = time.monotonic()

        progress(_CLEAN, "meshing")
        smooth = options.smooth_iters if options.smooth_iters is not None else _SMOOTHING[method]
        mesh, colors = postprocess.clean_mesh(
            verts,
            faces,
            colors,
            postprocess.CleanupConfig(smooth_iterations=smooth, target_faces=options.target_faces),
        )
        t3 = time.monotonic()

        if options.texture:
            from ..pipeline.texturing import bake_texture

            progress(_TEXTURE, "texturing")
            mesh = bake_texture(
                np.asarray(mesh.vertices),
                np.asarray(mesh.faces),
                view_arr,
                cameras.zero123pp_cameras(view_arr.shape[1]),
                masks=mask_arr,
                vertex_colors=colors,
                resolution=options.texture_size,
            )
        elif colors is not None:
            mesh.visual.vertex_colors = colors
        progress(0.97, "export")
        meshing.export_glb(mesh, out_dir / "result.glb")
        log.info(
            "%s: multiview %.1fs, reconstruction %.1fs, cleanup %.1fs, texture %.1fs, %d faces",
            method,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            time.monotonic() - t3,
            len(mesh.faces),
        )


# Old name, kept so existing imports keep working.
MultiViewNeuSBackend = MultiViewBackend


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
