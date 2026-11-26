"""Stage 2: multi-view diffusion with Zero123++ (Shi et al., 2023).

A single image does not determine the back of an object, so we let a
diffusion model imagine it. Zero123++ is Stable Diffusion 2 fine-tuned to
denoise one 960x640 canvas that tiles six 320x320 views of the object at
fixed poses. Generating all six in one image lets self-attention share
information across views, which is what makes them consistent with each other
(earlier Zero-1-to-3 sampled each view independently and they disagreed).
The input image conditions the model twice: through CLIP image embeddings
(global semantics, via cross-attention) and through "reference attention",
where the UNet's own features for the noised input image are concatenated
into the self-attention keys/values (fine appearance detail).

InstantMesh's authors fine-tuned the Zero123++ UNet to draw the views on a
white background (the background its reconstruction model was trained on).
`unet="instantmesh"` swaps in those weights; "original" keeps sudo-ai's gray.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

VIEW_SIZE = 320
GRID_ROWS, GRID_COLS = 3, 2


def split_grid(grid: Image.Image) -> list[Image.Image]:
    """Cut Zero123++'s 3x2 canvas into six views, row-major (matches cameras.zero123pp_cameras)."""
    w, h = grid.size
    vw, vh = w // GRID_COLS, h // GRID_ROWS
    return [grid.crop((c * vw, r * vh, (c + 1) * vw, (r + 1) * vh)) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def make_grid(views: list[Image.Image]) -> Image.Image:
    vw, vh = views[0].size
    grid = Image.new("RGB", (vw * GRID_COLS, vh * GRID_ROWS))
    for i, view in enumerate(views):
        r, c = divmod(i, GRID_COLS)
        grid.paste(view.convert("RGB"), (c * vw, r * vh))
    return grid


class Zero123PlusGenerator:
    def __init__(
        self,
        model_id: str = "sudo-ai/zero123plus-v1.2",
        pipeline_id: str = "sudo-ai/zero123plus-pipeline",
        device: str = "cuda",
        unet: str = "original",
    ) -> None:
        if unet not in ("original", "instantmesh"):
            raise ValueError(f"unet must be 'original' or 'instantmesh', not {unet!r}")
        self.model_id = model_id
        self.pipeline_id = pipeline_id
        self.device = device
        self.unet = unet
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        log.info("Loading %s on %s (%s)", self.model_id, self.device, dtype)
        pipe = DiffusionPipeline.from_pretrained(self.model_id, custom_pipeline=self.pipeline_id, torch_dtype=dtype)
        # Settings recommended by the Zero123++ authors.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        if self.unet == "instantmesh":
            from huggingface_hub import hf_hub_download

            from .instantmesh import INSTANTMESH_REPO, WHITE_BG_UNET_FILE

            path = hf_hub_download(INSTANTMESH_REPO, WHITE_BG_UNET_FILE)
            log.info("Using InstantMesh's white-background UNet (%s)", path)
            pipe.unet.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def generate(self, image: Image.Image, steps: int = 75, seed: int | None = None) -> list[Image.Image]:
        """`image` is the segmented, centred RGBA input. Returns six 320x320 RGB views."""
        import torch

        self.load()
        assert self.pipe is not None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        # The pipeline composites RGBA onto gray itself; the input must be square.
        cond = image.convert("RGBA").resize((VIEW_SIZE, VIEW_SIZE), Image.LANCZOS)
        grid = self.pipe(cond, num_inference_steps=steps, generator=generator).images[0]
        return split_grid(grid)


def views_to_arrays(views: list[Image.Image]) -> np.ndarray:
    """(V, H, W, 3) float32 in [0, 1]."""
    return np.stack([np.asarray(v.convert("RGB"), dtype=np.float32) / 255.0 for v in views])
