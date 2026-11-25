"""Stage 6: turn the SDF into a coloured triangle mesh.

Marching cubes (Lorensen & Cline, 1987) samples f on a regular N^3 grid and,
in every cell whose corners change sign, places triangle vertices on the edges
by linear interpolation of f (one of 256 corner sign patterns, reduced to a
small table of cases). Because f is a signed *distance*, the interpolation is
accurate and the mesh is watertight.

After extraction: drop small disconnected floaters, make the
triangles face outward (agree with grad f), and colour each vertex by querying
the colour field looking straight at the surface (view direction = -normal).
glTF is y-up with the front facing +z, so the z-up reconstruction is rotated.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage import measure

from .neus import NeuSModel


class MeshingError(Exception):
    pass


@torch.no_grad()
def sample_sdf_grid(model: NeuSModel, resolution: int, device, chunk: int = 2**18) -> np.ndarray:
    b = model.bound
    axis = torch.linspace(-b, b, resolution, device=device)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    out = torch.cat([model.sdf_field.sdf(pts[i : i + chunk]) for i in range(0, pts.shape[0], chunk)])
    return out.view(resolution, resolution, resolution).cpu().numpy()


@torch.no_grad()
def _vertex_colors(model: NeuSModel, verts: np.ndarray, device, chunk: int = 2**16) -> tuple[np.ndarray, np.ndarray]:
    colors, normals = [], []
    v_all = torch.from_numpy(verts).float().to(device)
    for i in range(0, v_all.shape[0], chunk):
        v = v_all[i : i + chunk]
        _, feats = model.sdf_field(v)
        n = F.normalize(model.sdf_field.gradient(v, model.fd_eps()), dim=-1)
        colors.append(model.color_field(v, n, -n, feats).cpu())
        normals.append(n.cpu())
    return torch.cat(colors).numpy(), torch.cat(normals).numpy()


def zup_to_gltf(verts: np.ndarray) -> np.ndarray:
    """(x, y, z) z-up, input view on +x  ->  glTF y-up, input view on +z.
    A cyclic axis permutation, so it is a proper rotation (no mirroring)."""
    return verts[:, [1, 2, 0]]


def drop_floaters(mesh: trimesh.Trimesh, min_fraction: float = 0.05) -> trimesh.Trimesh:
    """Remove disconnected pieces smaller than `min_fraction` of the largest one.
    (Keeping only the largest would also delete genuinely separate parts.)"""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest = max(p.area for p in parts)
    return trimesh.util.concatenate([p for p in parts if p.area >= min_fraction * largest])


def extract_mesh(model: NeuSModel, resolution: int = 256, device: str = "cuda") -> trimesh.Trimesh:
    sdf = sample_sdf_grid(model, resolution, device)
    if sdf.min() > 0 or sdf.max() < 0:
        raise MeshingError("The reconstructed SDF has no surface (no zero crossing).")
    b = model.bound
    spacing = 2 * b / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(spacing,) * 3)
    verts = verts - b
    mesh = trimesh.Trimesh(verts, faces, process=True)

    mesh = drop_floaters(mesh)

    colors, normals = _vertex_colors(model, np.asarray(mesh.vertices), device)
    if (np.asarray(mesh.vertex_normals) * normals).sum(-1).mean() < 0:
        mesh.invert()

    rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    out = trimesh.Trimesh(
        vertices=zup_to_gltf(np.asarray(mesh.vertices)),
        faces=np.asarray(mesh.faces),
        vertex_colors=(rgba * 255).clip(0, 255).astype(np.uint8),
        process=False,
    )
    return out
