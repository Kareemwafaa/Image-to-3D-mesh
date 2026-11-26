"""A small vectorised triangle rasteriser in NumPy (no GPU, no OpenGL).

Used twice by texture baking:
  * in UV space, to find which triangle (and where in it) every texel lies;
  * in each camera's image, as a z-buffer, to know which surface point is
    visible from that camera (a point hidden behind another part of the
    object must not take that view's colour).

Each triangle's integer bounding box is expanded into candidate pixels with
one `np.repeat`, and barycentric coordinates select the pixels whose centre
lies inside. Work is O(sum of bounding-box areas), done in chunks to bound
memory. Good enough for ~10^5 triangles at 1024^2 in a few seconds on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fragments:
    """Covered pixels: index into the flattened (H*W) image, triangle id, barycentrics."""

    pixel: np.ndarray  # (K,) int64
    face: np.ndarray  # (K,) int64
    bary: np.ndarray  # (K, 3) float32


def rasterize(
    tri_xy: np.ndarray,
    width: int,
    height: int,
    depth: np.ndarray | None = None,
    chunk_pixels: int = 4_000_000,
) -> Fragments:
    """Rasterise triangles given in pixel coordinates (x right, y down, pixel centres at +0.5).

    tri_xy: (F, 3, 2). depth: optional (F, 3) per-vertex depth; when given,
    only the nearest fragment per pixel is kept (a z-buffer).
    """
    tri_xy = np.asarray(tri_xy, dtype=np.float64)
    n_faces = tri_xy.shape[0]
    empty = Fragments(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), np.float32))
    if n_faces == 0:
        return empty

    lo = np.floor(tri_xy.min(axis=1) - 0.5).astype(np.int64) + 1  # first pixel centre >= min
    hi = np.floor(tri_xy.max(axis=1) - 0.5).astype(np.int64)  # last pixel centre <= max
    lo = np.maximum(lo, 0)
    hi[:, 0] = np.minimum(hi[:, 0], width - 1)
    hi[:, 1] = np.minimum(hi[:, 1], height - 1)
    span = np.maximum(hi - lo + 1, 0)
    count = span[:, 0] * span[:, 1]

    # Signed area; degenerate triangles cover nothing.
    a, b, c = tri_xy[:, 0], tri_xy[:, 1], tri_xy[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    count[np.abs(area) < 1e-12] = 0

    pix_out, face_out, bary_out, z_out = [], [], [], []
    order = np.flatnonzero(count)
    start = 0
    while start < len(order):
        # take as many faces as fit in the chunk budget (at least one)
        csum = np.cumsum(count[order[start:]])
        stop = start + max(1, int(np.searchsorted(csum, chunk_pixels, side="right")))
        faces = order[start:stop]
        start = stop

        n = count[faces]
        f = np.repeat(faces, n)
        offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        w = span[f, 0]
        px = lo[f, 0] + offsets % w
        py = lo[f, 1] + offsets // w
        cx, cy = px + 0.5, py + 0.5

        ax, ay = a[f, 0], a[f, 1]
        bx, by = b[f, 0], b[f, 1]
        qx, qy = c[f, 0], c[f, 1]
        inv = 1.0 / area[f]
        w0 = ((bx - cx) * (qy - cy) - (by - cy) * (qx - cx)) * inv
        w1 = ((qx - cx) * (ay - cy) - (qy - cy) * (ax - cx)) * inv
        w2 = 1.0 - w0 - w1
        eps = -1e-7
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)

        bary = np.stack([w0, w1, w2], axis=-1)[inside]
        pix_out.append((py * width + px)[inside])
        face_out.append(f[inside])
        bary_out.append(bary.astype(np.float32))
        if depth is not None:
            z_out.append((bary * depth[f[inside]]).sum(-1))

    if not pix_out:
        return empty
    pixel = np.concatenate(pix_out)
    face = np.concatenate(face_out)
    bary = np.concatenate(bary_out)
    if depth is None:
        return Fragments(pixel, face, bary)

    # z-buffer: keep the nearest fragment of each pixel
    z = np.concatenate(z_out)
    order = np.lexsort((z, pixel))
    pixel, face, bary = pixel[order], face[order], bary[order]
    first = np.ones(len(pixel), dtype=bool)
    first[1:] = pixel[1:] != pixel[:-1]
    return Fragments(pixel[first], face[first], bary[first])


def depth_buffer(tri_xy: np.ndarray, tri_depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H, W) depth of the nearest surface; +inf where nothing is drawn."""
    frags = rasterize(tri_xy, width, height, depth=tri_depth)
    zbuf = np.full(width * height, np.inf, dtype=np.float32)
    zbuf[frags.pixel] = (frags.bary * tri_depth[frags.face]).sum(-1)
    return zbuf.reshape(height, width)
