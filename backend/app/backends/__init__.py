from __future__ import annotations

from ..config import Settings
from .base import GenerateOptions, MeshBackend


def create_backend(settings: Settings) -> MeshBackend:
    if settings.backend == "multiview":
        from .multiview import MultiViewNeuSBackend

        return MultiViewNeuSBackend(settings.zero123pp_model, settings.zero123pp_pipeline, settings.device)
    if settings.backend == "triposr":
        from .triposr import TripoSRBackend

        return TripoSRBackend(settings.triposr_model, settings.device, settings.triposr_chunk_size)
    if settings.backend == "mock":
        from .mock import MockBackend

        return MockBackend()
    raise ValueError(f"Unknown MODEL_BACKEND {settings.backend!r}; use multiview, triposr or mock")


__all__ = ["GenerateOptions", "MeshBackend", "create_backend"]
