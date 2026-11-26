"""Stage 7: bake a colour texture from the generated views onto the mesh.

Both reconstructors predict a colour per vertex, but a colour field is
smooth by construction (an MLP of position), and per-vertex colour can only
be as sharp as the mesh is dense. The six generated views are much sharper,
and their cameras are known exactly, so we project them back onto the
surface instead ("projective texturing"):

1. UV unwrapping. xatlas cuts the surface into charts, flattens each one with
   a conformal (angle-preserving) map and packs them into one square atlas.
   Every texel then corresponds to exactly one surface point.
2. For every texel we know its 3D point p and normal n (rasterise the mesh in
   UV space and interpolate with barycentric coordinates).
3. For every view i: project p with the pinhole model, look up a z-buffer of
   the mesh rendered from that camera to test visibility, and sample the
   view's colour. The weight is
       w_i = visible_i * inside_silhouette_i * max(0, n . d_i)^k
   where d_i is the direction to camera i. The cosine term prefers views that
   see the surface head-on, where one pixel covers the least area and
   resampling blur is lowest; k sharpens the choice (k -> inf picks the single
   best view, k = 0 averages everything).
4. colour = sum_i w_i c_i / sum_i w_i. Texels that no view sees (e.g. the
   underside, which no Zero123++ camera looks at) fall back to the
   reconstructor's own colour, interpolated from the vertices.
   The six views are separate diffusion samples and disagree a little in
   colour; averaging them mixes those disagreements into blotches. So k is
   high (12: at 30 degrees off-axis a view's weight is already a fifth of the
   head-on one), which makes each texel come almost entirely from its best
   view and keeps blending to narrow seams. Silhouettes are eroded by a few
   pixels, because edge pixels mix object and background colour.
4b. Colour matching (Reinhard et al., 2001, in YCbCr instead of Lab): the views
   come out with their own exposure and tint (a matte black mug turned light
   grey). We shift the texture's luma mean and spread to the input photo's,
   and pull its chroma towards the photo's, shrinking the chroma spread
   when the photo is less colourful. That also mutes leftover colour blotches.
5. Charts are padded by dilating colours into empty texels, so bilinear and
   mip-mapped lookups at chart borders do not bleed background into seams.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from PIL import Image

from .cameras import Camera
from .raster import depth_buffer, rasterize

log = logging.getLogger(__name__)


def unwrap(vertices: np.ndarray, faces: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xatlas UV atlas. Returns (vmapping, new_faces, uvs): new vertex i is old vertex vmapping[i]."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = resolution
    pack.padding = 2
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmapping, new_faces, uvs = atlas[0]
    return vmapping.astype(np.int64), new_faces.astype(np.int64), uvs.astype(np.float32)


def _project(points: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (x, y) in pixels (x right, y down) and depth along the view axis."""
    c2w = cam.c2w.detach().cpu().numpy().astype(np.float64)
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]  # rows: p_cam = R^T (p - t)
    depth = -local[:, 2]  # OpenGL: the camera looks down -z
    safe = np.where(depth > 1e-6, depth, 1e-6)
    x = cam.focal * local[:, 0] / safe + cam.width / 2
    y = cam.height / 2 - cam.focal * local[:, 1] / safe
    return x, y, depth


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample (H, W, C) at pixel coordinates with pixel centres at +0.5, clamped at borders."""
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1)
    y = np.clip(y - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    y = rgb @ np.array([0.299, 0.587, 0.114])
    return np.stack([y, (rgb[:, 2] - y) * 0.564, (rgb[:, 0] - y) * 0.713], axis=-1)


def _from_ycbcr(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[:, 0], ycc[:, 1], ycc[:, 2]
    return np.stack([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], axis=-1)


def match_colors(colors: np.ndarray, reference: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Statistics transfer of (K, 3) RGB colours towards (M, 3) reference pixels, both in [0, 1].
    Luma: mean and spread matched (spread ratio clamped to [0.5, 2]). Chroma: mean matched,
    spread only ever reduced, so a colourful texture can calm down but noise is never amplified."""
    if len(colors) == 0 or len(reference) == 0 or strength <= 0:
        return colors
    src, ref = _to_ycbcr(colors), _to_ycbcr(reference)
    mu_s, sd_s = src.mean(0), src.std(0) + 1e-6
    mu_r, sd_r = ref.mean(0), ref.std(0) + 1e-6
    scale = sd_r / sd_s
    scale[0] = np.clip(scale[0], 0.5, 2.0)
    scale[1:] = np.minimum(scale[1:], 1.0)
    out = _from_ycbcr((src - mu_s) * scale + mu_r)
    return np.clip(colors + strength * (out - colors), 0.0, 1.0)


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink a (H, W) soft mask by taking the minimum over a (2p+1)^2 window."""
    if pixels <= 0:
        return mask
    out = mask.copy()
    padded = np.pad(mask, pixels, mode="edge")
    h, w = mask.shape
    for dy in range(2 * pixels + 1):
        for dx in range(2 * pixels + 1):
            np.minimum(out, padded[dy : dy + h, dx : dx + w], out=out)
    return out


def dilate(color: np.ndarray, known: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow known texels into unknown ones by averaging known 8-neighbours."""
    color = color.copy()
    known = known.copy()
    for _ in range(iterations):
        if known.all():
            break
        acc = np.zeros_like(color)
        cnt = np.zeros(known.shape, dtype=np.float32)
        padded_c = np.pad(color * known[..., None], ((1, 1), (1, 1), (0, 0)))
        padded_k = np.pad(known.astype(np.float32), 1)
        h, w = known.shape
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                acc += padded_c[dy : dy + h, dx : dx + w]
                cnt += padded_k[dy : dy + h, dx : dx + w]
        grow = (~known) & (cnt > 0)
        if not grow.any():
            break
        color[grow] = acc[grow] / cnt[grow][:, None]
        known |= grow
    return color, known


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: np.ndarray,
    cameras: list[Camera],
    masks: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    resolution: int = 1024,
    sharpness: float = 12.0,
    mask_threshold: float = 0.8,
    mask_erosion: int = 2,
    depth_tolerance: float = 0.02,
    reference: np.ndarray | None = None,
    color_match: float = 1.0,
) -> trimesh.Trimesh:
    """Return a z-up mesh with UVs and a baked baseColor texture.

    vertices/faces: the (cleaned) mesh in the cameras' world frame.
    views: (V, H, W, 3) in [0, 1]; masks: (V, H, W) in [0, 1] or None.
    vertex_colors: (N, 3|4) uint8 fallback for texels that no view sees.
    reference: (M, 3) RGB pixels of the object in the input photo, in [0, 1]; when
    given, the texture's colours are matched to it (`color_match` = strength).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.asarray(trimesh.Trimesh(vertices, faces, process=False).vertex_normals, dtype=np.float64)

    vmap, uv_faces, uvs = unwrap(vertices, faces, resolution)
    res = resolution

    # 1-2. texel -> (face, barycentrics) -> 3D point and normal
    tri_uv = uvs[uv_faces] * res
    tri_uv[..., 1] = res - tri_uv[..., 1]  # v points up in UV space, rows go down in the image
    frags = rasterize(tri_uv, res, res)
    corner = vmap[uv_faces[frags.face]]  # (K, 3) original vertex ids
    bary = frags.bary.astype(np.float64)[..., None]
    pos = (vertices[corner] * bary).sum(1)
    nrm = (normals[corner] * bary).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12)

    # 3-4. weighted projection from every view
    acc = np.zeros((len(pos), 3))
    wsum = np.zeros(len(pos))
    for i, cam in enumerate(cameras):
        vx, vy, vz = _project(vertices, cam)
        zbuf = depth_buffer(np.stack([vx, vy], -1)[faces], vz[faces], cam.width, cam.height)
        x, y, d = _project(pos, cam)
        ix = np.floor(x).astype(np.int64)
        iy = np.floor(y).astype(np.int64)
        inb = (ix >= 0) & (ix < cam.width) & (iy >= 0) & (iy < cam.height) & (d > 0)
        visible = np.zeros(len(pos), dtype=bool)
        visible[inb] = d[inb] <= zbuf[iy[inb], ix[inb]] + depth_tolerance

        to_cam = cam.c2w[:3, 3].detach().cpu().numpy() - pos
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True).clip(1e-12)
        cos = (nrm * to_cam).sum(1)
        w = np.where(visible & (cos > 0.05), np.clip(cos, 0, 1) ** sharpness, 0.0)
        if masks is not None:
            m = erode(masks[i].astype(np.float64), mask_erosion)
            inside = _bilinear(m[..., None], x, y)[:, 0]
            w = np.where(inside >= mask_threshold, w, 0.0)
        sel = w > 0
        if sel.any():
            acc[sel] += w[sel, None] * _bilinear(views[i].astype(np.float64), x[sel], y[sel])
            wsum[sel] += w[sel]

    seen = wsum > 1e-8
    color = np.zeros((len(pos), 3))
    color[seen] = acc[seen] / wsum[seen, None]
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors, dtype=np.float64)[:, :3] / 255.0
        color[~seen] = (vc[corner[~seen]] * bary[~seen]).sum(1)
        seen[:] = True
    if reference is not None and color_match > 0:
        color[seen] = match_colors(color[seen], np.asarray(reference, dtype=np.float64), color_match)
    log.info("texture: %d texels, %.1f%% seen by a view", len(pos), 100.0 * (wsum > 1e-8).mean() if len(pos) else 0)

    # 5. write the atlas and pad the charts
    image = np.zeros((res * res, 3))
    known = np.zeros(res * res, dtype=bool)
    image[frags.pixel[seen]] = color[seen]
    known[frags.pixel[seen]] = True
    image, known = dilate(image.reshape(res, res, 3), known.reshape(res, res), iterations=16)
    if not known.all():
        image[~known] = image[known].mean(axis=0) if known.any() else 0.5

    texture = Image.fromarray((image.clip(0, 1) * 255 + 0.5).astype(np.uint8), mode="RGB")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=0.85, name="baked"
    )
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return trimesh.Trimesh(vertices=vertices[vmap], faces=uv_faces, visual=visual, process=False)
