"""Tests for the mesh-quality stages: rasteriser, cleanup, texture baking, the
InstantMesh source loader, and the multiview backend wired end to end with
stand-ins for the two neural networks (no weights or GPU needed)."""

from __future__ import annotations

import io
import os
import sys
import tarfile

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from app.backends.base import GenerateOptions  # noqa: E402
from app.pipeline import cameras, instantmesh  # noqa: E402
from app.pipeline.postprocess import CleanupConfig, boundary_loops, clean_mesh, fill_holes  # noqa: E402
from app.pipeline.raster import depth_buffer, rasterize  # noqa: E402

# ---------------------------------------------------------------------------- rasteriser


def test_rasterize_covers_the_triangle_area():
    tri = np.array([[[0.0, 0.0], [40.0, 0.0], [0.0, 40.0]]])
    frags = rasterize(tri, 64, 64)
    assert abs(len(frags.pixel) - 800) < 45  # area 800, give or take the diagonal edge
    np.testing.assert_allclose(frags.bary.sum(-1), 1.0, atol=1e-5)
    assert (frags.bary >= -1e-6).all()


def test_depth_buffer_keeps_the_nearest_triangle():
    square = np.array([[[0, 0], [32, 0], [0, 32]], [[32, 0], [32, 32], [0, 32]]], dtype=float)
    tris = np.concatenate([square, square])
    depth = np.array([[5.0] * 3, [5.0] * 3, [2.0] * 3, [2.0] * 3])
    zbuf = depth_buffer(tris, depth, 32, 32)
    assert np.isfinite(zbuf).all() and np.allclose(zbuf, 2.0)
    assert np.isinf(depth_buffer(tris, depth, 64, 64)[40:, 40:]).all()


# ---------------------------------------------------------------------------- cleanup


def _holey_noisy_sphere(seed=0):
    rng = np.random.default_rng(seed)
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.6)  # 20480 faces
    v = np.asarray(sphere.vertices) * (1 + 0.01 * rng.standard_normal((len(sphere.vertices), 1)))
    faces = np.asarray(sphere.faces)
    top = np.argsort(-v[faces].mean(1)[:, 2])[:6]  # cut a small hole at the north pole
    faces = np.delete(faces, top, axis=0)
    floater = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
    fv = np.asarray(floater.vertices) + [0.9, 0, 0]
    verts = np.vstack([v, fv])
    faces = np.vstack([faces, np.asarray(floater.faces) + len(v)])
    return verts, faces


def test_fill_holes_closes_small_loops():
    verts, faces = _holey_noisy_sphere()
    sphere_faces = faces[: len(faces) - 80]
    mesh = trimesh.Trimesh(verts, sphere_faces, process=True)
    assert len(boundary_loops(np.asarray(mesh.faces))) == 1 and not mesh.is_watertight
    filled = fill_holes(mesh)
    assert filled.is_watertight and filled.is_winding_consistent


def test_clean_mesh_removes_floaters_fills_holes_and_decimates():
    verts, faces = _holey_noisy_sphere()
    colors = np.tile([200, 50, 50, 255], (len(verts), 1)).astype(np.uint8)
    mesh, out_colors = clean_mesh(verts, faces, colors, CleanupConfig(smooth_iterations=5, target_faces=4000))
    assert len(mesh.faces) <= 4400
    assert mesh.is_watertight and mesh.volume > 0  # outward-facing
    assert np.linalg.norm(mesh.vertices, axis=1).max() < 0.7  # the floater at x=0.9 is gone
    assert abs(mesh.volume / (4 / 3 * np.pi * 0.6**3) - 1) < 0.05  # Taubin does not shrink it
    assert out_colors.shape[0] == len(mesh.vertices) and (out_colors[:, 0] == 200).all()


def test_taubin_smoothing_reduces_noise():
    verts, faces = _holey_noisy_sphere(seed=1)
    cfg = dict(target_faces=0, min_component_fraction=0.05)
    rough, _ = clean_mesh(verts, faces, None, CleanupConfig(smooth_iterations=0, **cfg))
    smooth, _ = clean_mesh(verts, faces, None, CleanupConfig(smooth_iterations=10, **cfg))

    def roughness(m):
        r = np.linalg.norm(m.vertices, axis=1)
        return r.std()

    assert roughness(smooth) < 0.6 * roughness(rough)


# ---------------------------------------------------------------------------- texture baking

RED, BLUE = (0.9, 0.1, 0.1), (0.1, 0.2, 0.9)


def _render_two_tone_sphere(cam, radius=0.6):
    """Analytic render: red above the equator (z > 0), blue below, white background."""
    o, d = cam.rays()
    b = (o * d).sum(-1)
    disc = b * b - ((o * o).sum(-1) - radius**2)
    hit = disc > 0
    t = -b - disc.clamp(min=0).sqrt()
    p = o + t[:, None] * d
    rgb = torch.ones(o.shape[0], 3)
    top = hit & (p[:, 2] > 0)
    rgb[top] = torch.tensor(RED)
    rgb[hit & ~top] = torch.tensor(BLUE)
    n = cam.height
    return rgb.view(n, n, 3).numpy(), hit.float().view(n, n).numpy()


def _scene(size=96):
    cams = cameras.zero123pp_cameras(size)
    rendered = [_render_two_tone_sphere(c) for c in cams]
    return cams, np.stack([v for v, _ in rendered]), np.stack([m for _, m in rendered])


def _texel_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64) / 255
    uv = np.asarray(mesh.visual.uv)
    h, w = img.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return img[y, x]


def test_bake_texture_projects_the_views():
    pytest.importorskip("xatlas")
    from app.pipeline.texturing import bake_texture

    cams, views, masks = _scene()
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
    gray = np.full((len(sphere.vertices), 3), 128, np.uint8)
    baked = bake_texture(sphere.vertices, sphere.faces, views, cams, masks=masks, vertex_colors=gray, resolution=256)

    colors = _texel_colors(baked)
    z = np.asarray(baked.vertices)[:, 2]
    # Every camera is within 20 degrees of the equator, so the poles are seen
    # obliquely; check the band the views cover well.
    upper = (z > 0.1) & (z < 0.45)
    lower = (z < -0.1) & (z > -0.45)
    assert (np.abs(colors[upper] - RED).max(1) < 0.15).mean() > 0.9
    assert (np.abs(colors[lower] - BLUE).max(1) < 0.15).mean() > 0.9

    # Survives a glTF round trip with its texture.
    from app.pipeline.meshing import export_glb

    buf = io.BytesIO()
    export_glb(baked, buf)
    loaded = trimesh.load(io.BytesIO(buf.getvalue()), file_type="glb", force="mesh")
    assert loaded.visual.kind == "texture" and loaded.visual.material.baseColorTexture is not None


def test_bake_texture_falls_back_to_vertex_colors_where_unseen():
    pytest.importorskip("xatlas")
    from app.pipeline.texturing import bake_texture

    cams, views, masks = _scene(64)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    green = np.tile([0, 255, 0], (len(sphere.vertices), 1)).astype(np.uint8)
    no_masks = np.zeros_like(masks)  # every view says "background": nothing is seen
    baked = bake_texture(
        sphere.vertices, sphere.faces, views, cams, masks=no_masks, vertex_colors=green, resolution=128
    )
    assert np.allclose(_texel_colors(baked), [0, 1, 0], atol=0.02)


# ---------------------------------------------------------------------------- InstantMesh source


def _fake_upstream_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        files = {
            "InstantMesh-abc/src/__init__.py": b"",
            "InstantMesh-abc/src/models/__init__.py": b"",
            "InstantMesh-abc/src/models/lrm_mesh.py": b"X = 1\n",
            "InstantMesh-abc/LICENSE": b"Apache License 2.0\n",
            "InstantMesh-abc/app.py": b"raise SystemExit('should not be extracted')\n",
        }
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_upstream_source_is_extracted_under_its_own_package(tmp_path, monkeypatch):
    monkeypatch.delenv("INSTANTMESH_SRC", raising=False)
    dest = instantmesh.source_dir(tmp_path)
    instantmesh._extract_src(_fake_upstream_tarball(), dest)
    assert (dest / "instantmesh_src" / "models" / "lrm_mesh.py").read_text() == "X = 1\n"
    assert (dest / "instantmesh_src" / "LICENSE").is_file()
    assert not (dest / "app.py").exists()

    # ensure_source finds the cached copy without touching the network.
    def no_network(*_a, **_k):
        raise AssertionError("downloaded although the source was cached")

    monkeypatch.setattr(instantmesh.urllib.request, "urlopen", no_network)
    assert instantmesh.ensure_source(tmp_path) == dest


def test_nvdiffrast_stub_imports_but_refuses_to_run(monkeypatch):
    try:
        import nvdiffrast  # noqa: F401

        pytest.skip("the real nvdiffrast is installed")
    except ImportError:
        pass
    for name in ("nvdiffrast", "nvdiffrast.torch"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    instantmesh._install_nvdiffrast_stub()
    import nvdiffrast.torch as dr

    with pytest.raises(RuntimeError, match="nvdiffrast is not installed"):
        dr.RasterizeCudaContext(device="cuda")
    with pytest.raises(AttributeError):
        _ = dr.__file__


# ---------------------------------------------------------------------------- backend, end to end


class _FakeGenerator:
    """Stands in for Zero123++: returns the analytic renders of the two-tone sphere."""

    def __init__(self, views):
        from PIL import Image

        self.views = [Image.fromarray((v * 255).astype(np.uint8)) for v in views]

    def load(self):
        pass

    def generate(self, image, steps, seed):
        return self.views


class _FakeLRM:
    """Stands in for InstantMesh: a slightly bumpy sphere with gray vertex colours."""

    def __init__(self, background=None):
        self.calls = 0
        self.background = background

    def reconstruct(self, views):
        self.calls += 1
        assert views.shape == (6, 96, 96, 3)
        assert (views[self.background] == 1.0).all()  # pure white, as InstantMesh was trained
        s = trimesh.creation.icosphere(subdivisions=5, radius=0.6)
        return instantmesh.LRMOutput(
            np.asarray(s.vertices, np.float32),
            np.asarray(s.faces),
            np.full((len(s.vertices), 3), 128, np.uint8),
        )


def test_multiview_backend_produces_a_small_textured_glb(tmp_path, monkeypatch):
    pytest.importorskip("xatlas")
    from PIL import Image

    from app.backends import multiview as mv
    from app.pipeline import segmentation

    cams, views, masks = _scene(96)
    mask_by_id = {}
    backend = mv.MultiViewBackend("m", "p", "cpu", unet="original", preload_instantmesh=False)
    backend.generator = _FakeGenerator(views)
    for img, m in zip(backend.generator.views, masks, strict=True):
        mask_by_id[id(img)] = m
    monkeypatch.setattr(segmentation, "view_mask", lambda v: mask_by_id[id(v)])
    backend._lrm = _FakeLRM(background=masks == 0)

    stages = []
    opts = GenerateOptions(texture_size=256, target_faces=3000)
    backend.generate(Image.new("RGBA", (64, 64)), opts, tmp_path, lambda p, s=None: stages.append(s))

    assert backend._lrm.calls == 1
    assert {"multiview", "reconstructing", "meshing", "texturing", "export"} <= set(stages)
    glb = tmp_path / "result.glb"
    assert glb.is_file() and glb.stat().st_size < 1_000_000
    mesh = trimesh.load(glb, file_type="glb", force="mesh")
    assert len(mesh.faces) <= 3300 and mesh.visual.kind == "texture"
    # exported y-up: the red half (z-up +z) is now on +y
    colors = _texel_colors(mesh)
    y = np.asarray(mesh.vertices)[:, 1]
    band = (y > 0.1) & (y < 0.45)
    assert (np.abs(colors[band] - RED).max(1) < 0.2).mean() > 0.85


def test_unknown_reconstructor_is_rejected(tmp_path):
    from PIL import Image

    from app.backends import multiview as mv

    backend = mv.MultiViewBackend("m", "p", "cpu", unet="original", preload_instantmesh=False)
    with pytest.raises(ValueError, match="reconstructor"):
        backend.generate(Image.new("RGBA", (8, 8)), GenerateOptions(reconstructor="magic"), tmp_path, lambda *a: None)


# ---------------------------------------------------------------------------- InstantMesh, real upstream code


def test_instantmesh_upstream_runs_without_nvdiffrast(tmp_path, monkeypatch):
    """Download the pinned upstream source and run a tiny, randomly initialised
    InstantMesh on CPU: checks that our wrapper matches upstream's code (imports,
    camera format, FlexiCubes extraction) without the 1.7 GB of real weights."""
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    pytest.importorskip("cv2")
    try:
        instantmesh.import_upstream()  # downloads the source and puts it on sys.path
    except OSError as exc:  # no network
        if os.environ.get("I3D_REQUIRE_NETWORK") == "1":
            raise
        pytest.skip(f"could not download the InstantMesh source: {exc}")

    import importlib

    from transformers import ViTConfig, ViTImageProcessor

    dino = importlib.import_module("instantmesh_src.models.encoder.dino")
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, patch_size=16)
    dino.ViTModel(cfg, add_pooling_layer=False).save_pretrained(tmp_path)
    ViTImageProcessor(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]).save_pretrained(tmp_path)

    torch.manual_seed(0)
    rec = instantmesh.InstantMeshReconstructor("large", device="cpu")
    rec.build(
        params=dict(
            encoder_model_name=str(tmp_path),
            encoder_feat_dim=32,
            transformer_dim=32,
            transformer_layers=1,
            transformer_heads=2,
            triplane_dim=8,
            grid_res=24,
        )
    )
    # Untrained weights predict an SDF without a surface, so upstream falls back to
    # the grid-sized box. That is exactly what the first real T4 run produced, and
    # it must now fail loudly instead of being exported.
    from app.pipeline.errors import MeshingError

    with pytest.raises(MeshingError, match="empty-shape fallback"):
        rec.reconstruct(np.ones((6, 160, 160, 3), np.float32))  # resized to 320 inside

    # The raw upstream output has the shapes our wrapper relies on.
    planes = torch.randn(1, 3, 8, 64, 64)
    with torch.no_grad():
        v, f, c = rec.model.extract_mesh(planes, use_texture_map=False)
    assert v.ndim == 2 and v.shape[1] == 3 and len(f) > 0
    assert c.shape == (len(v), 3) and c.dtype == np.uint8
    assert np.abs(v).max() <= 1.05 + 1e-3  # inside the FlexiCubes grid (scale 2.1)


def test_chunked_geometry_matches_upstream(tmp_path):
    """Our memory-bounded FlexiCubes head must give exactly upstream's numbers."""
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    pytest.importorskip("cv2")
    try:
        up = instantmesh.import_upstream()
    except OSError as exc:
        if os.environ.get("I3D_REQUIRE_NETWORK") == "1":
            raise
        pytest.skip(f"could not download the InstantMesh source: {exc}")
    import importlib

    synth_mod = importlib.import_module("instantmesh_src.models.renderer.synthesizer_mesh")
    torch.manual_seed(0)
    synth = synth_mod.TriplaneSynthesizer(triplane_dim=8, samples_per_ray=16)
    planes = torch.randn(1, 3, 8, 16, 16)
    geo = up.FlexiCubesGeometry(grid_res=10, scale=2.1, renderer=None, device="cpu")
    coords = geo.verts.unsqueeze(0)
    with torch.no_grad():
        expected = synth.get_geometry_prediction(planes, coords, geo.indices)
        chunked = instantmesh.chunked_geometry_prediction(synth, up.sample_from_planes, chunk=64)
        got = chunked(planes, coords, geo.indices)
    for e, g in zip(expected, got, strict=True):
        assert e.shape == g.shape
        torch.testing.assert_close(g, e, rtol=1e-5, atol=1e-6)


def test_grid_box_detector():
    box = trimesh.creation.box(extents=(2.1, 2.1, 2.1))
    sphere = trimesh.creation.icosphere(radius=0.8)
    assert instantmesh.looks_like_the_grid_box(np.asarray(box.vertices), 1.05)
    assert not instantmesh.looks_like_the_grid_box(np.asarray(sphere.vertices), 1.05)
    tall = np.asarray(sphere.vertices) * [0.5, 0.5, 1.3]  # touches top and bottom only
    assert not instantmesh.looks_like_the_grid_box(tall, 1.05)


def test_whiten_background_and_white_conditioning():
    from PIL import Image

    from app.pipeline.multiview import composite_on, whiten_background

    views = np.full((1, 4, 4, 3), 0.8, np.float32)  # off-white background, as diffusion produces
    views[0, 1:3, 1:3] = 0.1
    masks = np.zeros((1, 4, 4), np.float32)
    masks[0, 1:3, 1:3] = 1.0
    out = whiten_background(views, masks)
    assert (out[0][masks[0] == 0] == 1.0).all() and np.allclose(out[0, 1:3, 1:3], 0.1)

    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba.putpixel((1, 1), (10, 20, 30, 255))
    rgb = composite_on(rgba, (255, 255, 255))
    assert rgb.mode == "RGB" and rgb.getpixel((0, 0)) == (255, 255, 255) and rgb.getpixel((1, 1)) == (10, 20, 30)


def test_color_match_pulls_texture_towards_the_photo():
    from app.pipeline.texturing import _to_ycbcr, match_colors

    rng = np.random.default_rng(0)
    # A light-grey texture with purple and red blotches; the photo is a matte black mug.
    texture = np.clip(0.7 + 0.05 * rng.standard_normal((5000, 3)), 0, 1)
    texture[:500] = [0.6, 0.4, 0.75]
    texture[500:800] = [0.75, 0.35, 0.35]
    photo = np.clip(0.12 + 0.04 * rng.standard_normal((5000, 1)) * [1, 1, 1], 0, 1)

    out = match_colors(texture, photo)
    y_out, y_photo = _to_ycbcr(out)[:, 0], _to_ycbcr(photo)[:, 0]
    assert abs(y_out.mean() - y_photo.mean()) < 0.02  # exposure matches the photo
    chroma = lambda c: np.linalg.norm(_to_ycbcr(c)[:, 1:], axis=1).mean()  # noqa: E731
    assert chroma(out) < 0.35 * chroma(texture)  # blotches are muted
    assert np.allclose(match_colors(texture, photo, strength=0.0), texture)


def test_color_match_removes_colours_the_photo_does_not_have():
    from app.pipeline.texturing import _to_ycbcr, match_colors

    rng = np.random.default_rng(1)
    texture = np.clip(0.5 + 0.03 * rng.standard_normal((4000, 3)), 0, 1)
    texture[:300] = [0.55, 0.42, 0.7]  # purple blotch
    grey_photo = np.clip(0.2 + 0.05 * rng.standard_normal((4000, 1)) * [1, 1, 1], 0, 1)

    out = _to_ycbcr(match_colors(texture, grey_photo))
    assert np.abs(out[:300, 1:]).max() < 0.02  # the blotch is grey now

    # Half the photo's brightness when asked to, measured on the median.
    y = _to_ycbcr(match_colors(texture, grey_photo, albedo_scale=0.5))[:, 0]
    assert abs(np.median(y) - 0.5 * np.median(_to_ycbcr(grey_photo)[:, 0])) < 0.01

    # A colourful photo keeps its colours: a red object stays red.
    red_photo = np.clip([0.7, 0.1, 0.1] + 0.03 * rng.standard_normal((4000, 3)), 0, 1)
    red_tex = np.clip([0.6, 0.2, 0.2] + 0.03 * rng.standard_normal((4000, 3)), 0, 1)
    out = match_colors(red_tex, red_photo)
    assert (out[:, 0] > out[:, 1] + 0.3).mean() > 0.95


def test_erode_shrinks_masks():
    from app.pipeline.texturing import erode

    m = np.zeros((9, 9))
    m[2:7, 2:7] = 1.0
    e = erode(m, 1)
    assert e.sum() == 9 and e[4, 4] == 1.0 and e[2, 2] == 0.0
