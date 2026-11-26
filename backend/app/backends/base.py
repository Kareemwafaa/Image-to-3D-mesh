"""Common interface for image-to-mesh backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

from PIL import Image


class ProgressFn(Protocol):
    def __call__(self, value: float, stage: Optional[str] = None) -> None: ...


@dataclass
class GenerateOptions:
    remove_background: bool = True
    foreground_ratio: float = 0.85
    mc_resolution: int = 256
    # multiview backend only
    diffusion_steps: int = 75
    recon_iters: int = 2500
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
