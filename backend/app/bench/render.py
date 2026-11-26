"""Render a (textured) mesh from a pinhole camera, on the CPU.

Used to make the benchmark's inputs (one view of each ground-truth object,
plus optionally the six Zero123++ views for the "oracle views" run) and to
compare renders of the reconstruction with renders of the ground truth.
It reuses the pipeline's NumPy z-buffer rasteriser, so it needs no OpenGL,
EGL or GPU and runs the same on a laptop, in CI and on Kaggle.

Shading is deliberately simple: albedo times a soft headlight term
(0.45 + 0.55 |n . v|), which gives the object readable shape cues like a
studio render without baking in a light direction that could favour one
method. Images are rendered at `ssaa` times the size and box-filtered down,
so edges are anti-aliased and the alpha channel has soft borders.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from ..pipeline.cameras import Camera
from ..pipeline.raster import rasterize
from ..pipeline.texturing import _bilinear, _project

DEFAULT_GRAY = 0.7


def _texture_image(mesh: trimesh.Trimesh) -> np.ndarray | None:
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    img = getattr(material, "baseColorTexture", None)
    if img is None:
        img = getattr(material, "image", None)
    if img is None:
        return None
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def _albedo(mesh: trimesh.Trimesh, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """(K, 3) base colour in [0, 1] at each fragment."""
    faces = np.asarray(mesh.faces)
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals):
        uv = getattr(visual, "uv", None)
        tex = _texture_image(mesh)
        if uv is not None and tex is not None and len(uv) == len(mesh.vertices):
            uvf = (np.asarray(uv, dtype=np.float64)[faces[face]] * bary[..., None]).sum(1)
            h, w = tex.shape[:2]
            u = np.mod(uvf[:, 0], 1.0)
            v = np.mod(uvf[:, 1], 1.0)
            return _bilinear(tex, u * w, (1.0 - v) * h)  # uv origin is bottom-left
        material = getattr(visual, "material", None)
        base = getattr(material, "baseColorFactor", None) if material is not None else None
        if base is None and material is not None:
            base = getattr(material, "diffuse", None)
        if base is not None:
            rgb = np.asarray(base, dtype=np.float64)[:3]
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            return np.broadcast_to(rgb, (len(face), 3)).copy()
    elif visual.kind == "vertex":
        vc = np.asarray(visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        return (vc[faces[face]] * bary[..., None]).sum(1)
    elif visual.kind == "face":
        return np.asarray(visual.face_colors, dtype=np.float64)[face, :3] / 255.0
    return np.full((len(face), 3), DEFAULT_GRAY)


def render(
    mesh: trimesh.Trimesh,
    cam: Camera,
    background: float = 1.0,
    ssaa: int = 2,
    shading: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb (H, W, 3) float in [0, 1] composited on `background`, alpha (H, W) in [0, 1])."""
    big = Camera(cam.c2w, cam.fov_deg, cam.width * ssaa, cam.height * ssaa)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    x, y, depth = _project(verts, big)
    tri_xy = np.stack([x, y], axis=-1)[faces]
    tri_depth = depth[faces]
    in_front = (tri_depth > 1e-4).all(axis=1)
    keep = np.flatnonzero(in_front)
    frags = rasterize(tri_xy[keep], big.width, big.height, depth=tri_depth[keep])
    face = keep[frags.face]
    bary = frags.bary.astype(np.float64)

    color = _albedo(mesh, face, bary)
    if shading:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        n = (normals[faces[face]] * bary[..., None]).sum(1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        c2w = cam.c2w.detach().cpu().numpy().astype(np.float64)
        p = (verts[faces[face]] * bary[..., None]).sum(1)
        view = c2w[:3, 3] - p
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        color = color * (0.45 + 0.55 * np.abs((n * view).sum(1)))[:, None]

    h, w = big.height, big.width
    rgb = np.zeros((h * w, 3))
    alpha = np.zeros(h * w)
    rgb[frags.pixel] = color
    alpha[frags.pixel] = 1.0
    rgb = rgb.reshape(h, w, 3)
    alpha = alpha.reshape(h, w)
    if ssaa > 1:  # box filter: average each ssaa x ssaa block (premultiplied colour)
        rgb = rgb.reshape(cam.height, ssaa, cam.width, ssaa, 3).mean((1, 3))
        alpha = alpha.reshape(cam.height, ssaa, cam.width, ssaa).mean((1, 3))
    rgb = rgb + (1.0 - alpha[..., None]) * background
    return np.clip(rgb, 0, 1).astype(np.float32), alpha.astype(np.float32)


def to_rgba_image(rgb: np.ndarray, alpha: np.ndarray, background: float = 1.0) -> Image.Image:
    """Un-composite a render into an RGBA image (straight alpha), as a segmenter would output."""
    a = alpha[..., None]
    straight = np.where(a > 1e-6, (rgb - (1 - a) * background) / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([np.clip(straight, 0, 1), a], axis=-1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), mode="RGBA")
