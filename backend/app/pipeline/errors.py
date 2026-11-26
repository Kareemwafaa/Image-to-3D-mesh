"""Pipeline exceptions, in a module with no heavy imports.

The worker classifies failures without importing torch, so a CPU-only worker
(mock model) can handle every error path.
"""


class PreprocessError(Exception):
    """The input has no usable foreground (empty or unreadable cut-out)."""


class MeshingError(Exception):
    """Marching cubes found no surface."""
