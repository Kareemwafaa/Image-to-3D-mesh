"""Common interface for image-to-mesh backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

from PIL import Image

# How the 6 views become a mesh: a feed-forward model (default) or a per-object fit.
RECONSTRUCTORS = ("instantmesh", "neus")


class ProgressFn(Protocol):
    def __call__(self, value: float, stage: Optional[str] = None) -> None: ...


@dataclass
class GenerateOptions:
    remove_background: bool = True
    foreground_ratio: float = 0.85
    mc_resolution: int = 256
    # multiview backend only
    diffusion_steps: int = 75
    reconstructor: str = "instantmesh"  # one of RECONSTRUCTORS
    recon_iters: int = 2500  # neus only
    texture: bool = True  # bake a texture from the views (else vertex colours)
    texture_size: int = 1024
    target_faces: int = 40_000  # quadric decimation target; 0 keeps every face
    smooth_iters: Optional[int] = None  # Taubin iterations; None = per-reconstructor default
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GenerateOptions:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MeshBackend(Protocol):
    name: str

    def load(self) -> None:
        """Load weights. Called once, from the worker."""

    def generate(
        self,
        image: Image.Image,
        options: GenerateOptions,
        out_dir: Path,
        progress: ProgressFn,
    ) -> None:
        """Turn one segmented RGBA image into `out_dir/result.glb`.
        May also write `out_dir/views.png` (intermediate views, for display)."""


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
