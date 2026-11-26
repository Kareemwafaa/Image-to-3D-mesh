"""Stage 6: clean up the raw mesh before texturing and export.

An iso-surface straight out of marching cubes or FlexiCubes has three
problems that hurt how it looks and how big the file is:

* Floaters and holes. Small disconnected blobs (where the field briefly
  crossed zero in empty space) are removed; small boundary loops are closed
  by fan triangulation around their centroid, oriented to match the
  neighbouring faces.
* High-frequency noise ("lumpy" surfaces, grid stair-steps). Taubin
  smoothing (1995) alternates a shrinking Laplacian step (lambda > 0) with an
  inflating one (mu < -lambda). Together they form a low-pass filter on the
  mesh: bumps of a few triangles are removed, but unlike plain Laplacian
  smoothing the object does not shrink.
* Far more triangles than detail. Quadric error metric decimation (Garland &
  Heckbert, 1997) repeatedly collapses the edge whose collapse moves the
  surface least, measured as the sum of squared distances to the planes of
  the original faces around it (a 4x4 quadric per vertex). Flat regions lose
  most triangles, curved ones keep them. The GLB shrinks ~10x.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

log = logging.getLogger(__name__)


@dataclass
class CleanupConfig:
    min_component_fraction: float = 0.05  # drop parts smaller than this share of the largest (by area)
    max_hole_edges: int = 200  # close boundary loops up to this many edges
    smooth_iterations: int = 10  # Taubin iterations; 0 to skip
    target_faces: int = 40_000  # 0 to skip decimation


def drop_floaters(mesh: trimesh.Trimesh, min_fraction: float = 0.05) -> trimesh.Trimesh:
    """Remove disconnected pieces smaller than `min_fraction` of the largest one.
    (Keeping only the largest would also delete genuinely separate parts.)"""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest = max(p.area for p in parts)
    return trimesh.util.concatenate([p for p in parts if p.area >= min_fraction * largest])


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Closed loops of boundary edges, each as vertex ids in the faces' winding direction."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse.reshape(-1)] == 1]
    nxt: dict[int, int] = {}
    for a, b in boundary.tolist():
        if a in nxt:  # a non-manifold vertex: two boundary edges leave it; skip such loops
            nxt[a] = -1
        else:
            nxt[a] = b
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen or nxt[start] == -1:
            continue
        loop, v = [], start
        while v not in seen and v in nxt and nxt[v] != -1:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_holes(mesh: trimesh.Trimesh, max_edges: int = 200) -> trimesh.Trimesh:
    """Close small holes with a fan of triangles around each hole's centroid."""
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    new_v, new_f = [], []
    for loop in boundary_loops(faces):
        if len(loop) > max_edges:
            continue
        if len(loop) == 3:
            new_f.append([loop[2], loop[1], loop[0]])
            continue
        c = len(verts) + len(new_v)
        new_v.append(verts[loop].mean(axis=0))
        # Boundary edge a->b belongs to a face winding a->b, so the patch uses b->a.
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            new_f.append([b, a, c])
    if not new_f:
        return mesh
    out = trimesh.Trimesh(
        np.vstack([verts, np.asarray(new_v).reshape(-1, 3)]),
        np.vstack([faces, np.asarray(new_f)]),
        process=False,
    )
    log.info("filled %d triangles' worth of holes", len(new_f))
    return out


def _transfer_colors(src: trimesh.Trimesh, colors: np.ndarray, dst_vertices: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, idx = cKDTree(np.asarray(src.vertices)).query(dst_vertices)
    return colors[idx]


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
    except (ImportError, ModuleNotFoundError, TypeError) as exc:
        log.warning("decimation skipped (%s); install fast-simplification", exc)
        return mesh
    return trimesh.Trimesh(np.asarray(out.vertices), np.asarray(out.faces), process=True)


def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    cfg: CleanupConfig | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray | None]:
    """Return (mesh, vertex_colors) after floater removal, hole filling, smoothing and
    decimation. Colours (N, 3|4 uint8) are carried over to the new vertices."""
    cfg = cfg or CleanupConfig()
    original = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=True)
    colors = None
    if vertex_colors is not None:
        # process=True merged duplicate vertices; map colours by position.
        colors = _transfer_colors(
            trimesh.Trimesh(np.asarray(vertices), np.zeros((0, 3), int), process=False),
            np.asarray(vertex_colors),
            np.asarray(original.vertices),
        )
    n_in = len(original.faces)

    mesh = drop_floaters(original, cfg.min_component_fraction)
    if cfg.max_hole_edges > 0:
        mesh = fill_holes(mesh, cfg.max_hole_edges)
    if cfg.smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=cfg.smooth_iterations)
    mesh = decimate(mesh, cfg.target_faces)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()  # consistent winding, outward-facing
    if colors is not None:
        colors = _transfer_colors(original, colors, np.asarray(mesh.vertices))
    log.info("cleanup: %d -> %d faces, watertight=%s", n_in, len(mesh.faces), mesh.is_watertight)
    return mesh, colors
