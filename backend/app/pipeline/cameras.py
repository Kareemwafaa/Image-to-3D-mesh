"""Stage 3: camera geometry for the generated views.

Zero123++ v1.2 always renders the same six poses relative to the input view:
azimuths 30, 90, 150, 210, 270, 330 degrees, alternating absolute elevations
of +20 and -10 degrees, and a 30 degree field of view. Because the poses are
known, the reconstruction does not need structure-from-motion: we can write
down every pixel's ray directly.

Conventions (same as InstantMesh, which reconstructs from Zero123++ views):
  * world is z-up, the input view's camera sits on the +x axis;
  * camera-to-world matrices use the OpenGL convention: the camera looks down
    its local -z axis, +x is right and +y is up in the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ZERO123PP_AZIMUTHS = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
ZERO123PP_ELEVATIONS = (20.0, -10.0, 20.0, -10.0, 20.0, -10.0)
ZERO123PP_FOV_DEG = 30.0
# GUESS: Zero123++ only says it assumes a "normalized object size". InstantMesh
# uses radius 4.0 for these views, which puts an object of half-size ~1 in frame.
# A wrong radius only rescales the reconstructed object uniformly.
ZERO123PP_RADIUS = 4.0


@dataclass
class Camera:
    c2w: torch.Tensor  # (4, 4) camera-to-world
    fov_deg: float
    width: int
    height: int

    @property
    def focal(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(self.fov_deg) / 2)

    def rays(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Return (origins, directions), each (H*W, 3), one ray per pixel centre."""
        v, u = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing="ij",
        )
        x = (u + 0.5 - self.width / 2) / self.focal
        y = -(v + 0.5 - self.height / 2) / self.focal  # image rows go down, camera +y goes up
        dirs_cam = torch.stack([x, y, -torch.ones_like(x)], dim=-1).reshape(-1, 3)
        c2w = self.c2w.float()
        dirs = dirs_cam @ c2w[:3, :3].T
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        origins = c2w[:3, 3].expand_as(dirs)
        return origins.to(device), dirs.to(device)


def look_at(position: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """OpenGL camera-to-world matrix for a camera at `position` looking at `target`, z-up world."""
    target = torch.zeros(3) if target is None else target
    up = torch.tensor([0.0, 0.0, 1.0])
    z = position - target
    z = z / z.norm()
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    c2w = torch.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, position
    return c2w


def orbit_camera(azimuth_deg: float, elevation_deg: float, radius: float, fov_deg: float, size: int) -> Camera:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    position = torch.tensor(
        [radius * math.cos(el) * math.cos(az), radius * math.cos(el) * math.sin(az), radius * math.sin(el)]
    )
    return Camera(c2w=look_at(position), fov_deg=fov_deg, width=size, height=size)


def zero123pp_cameras(size: int = 320, radius: float = ZERO123PP_RADIUS) -> list[Camera]:
    """The six output cameras of Zero123++ v1.2, in the order of its 3x2 output grid."""
    return [
        orbit_camera(az, el, radius, ZERO123PP_FOV_DEG, size)
        for az, el in zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS, strict=True)
    ]
