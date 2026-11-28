"""Tests for the accuracy benchmark (app/bench), on CPU with tiny synthetic objects.

The neural networks are replaced by a fake backend whose "reconstructions" are
known transforms of the ground truth, so the expected scores are known."""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

torch = pytest.importorskip("torch")

from app.bench import align, gso, metrics, render, report, runner  # noqa: E402
from app.bench.__main__ import main as cli_main  # noqa: E402
from app.pipeline import cameras  # noqa: E402


def ellipsoid(radii=(1.0, 0.6, 0.8), subdivisions=3) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions)
    m.vertices = np.asarray(m.vertices) * np.asarray(radii)
    return m


def random_rotation(deg: float, axis=(0.3, 0.5, 1.0)) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(math.radians(deg), axis)[:3, :3]


# ---------------------------------------------------------------------------- metrics


def test_identical_meshes_score_perfectly():
    m = ellipsoid()
    a = metrics.sample_surface(m, 20_000, seed=0)
    b = metrics.sample_surface(m, 20_000, seed=1)
    out = metrics.geometry_metrics(a, b)
    assert out["chamfer"] < 0.02
    assert out["fscore@0.05"] > 0.99
    assert out["normal_consistency"] > 0.97


def test_chamfer_matches_the_gap_between_concentric_spheres():
    inner = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    outer = trimesh.creation.icosphere(subdivisions=4, radius=1.1)
    out = metrics.geometry_metrics(metrics.sample_surface(inner, 20_000), metrics.sample_surface(outer, 20_000, seed=1))
    assert out["chamfer"] == pytest.approx(0.1, abs=0.01)
    assert out["fscore@0.05"] < 0.05  # every point is ~0.1 away
    assert out["fscore@0.2"] > 0.99
    assert out["normal_consistency"] > 0.97  # same orientation everywhere


def test_fscore_precision_and_recall_are_asymmetric():
    gt = metrics.sample_surface(ellipsoid(), 10_000)
    half = gt.points[:, 0] > 0
    pred = metrics.SurfaceSample(gt.points[half], gt.normals[half])  # only half the object
    out = metrics.geometry_metrics(pred, gt)
    assert out["precision@0.05"] == pytest.approx(1.0)
    assert 0.45 < out["recall@0.05"] < 0.7
    assert out["accuracy"] < out["completeness"]


def test_image_metrics():
    a = np.full((8, 8, 3), 0.5)
    m = np.zeros((8, 8), bool)
    m[2:6, 2:6] = True
    assert metrics.image_metrics(a, a, m, m) == {"psnr": float("inf"), "mask_iou": 1.0}
    out = metrics.image_metrics(a, a + 0.1, m, np.zeros_like(m))
    assert out["psnr"] == pytest.approx(20.0, abs=1e-6)
    assert out["mask_iou"] == 0.0


# ---------------------------------------------------------------------------- alignment


def test_normalize_to_unit_cube():
    m = ellipsoid()
    m.vertices = np.asarray(m.vertices) * 7 + [3, -2, 5]
    n, tf = align.normalize_to_unit_cube(m)
    np.testing.assert_allclose(n.bounds, [[-1, -0.6, -0.8], [1, 0.6, 0.8]], atol=1e-6)
    np.testing.assert_allclose(tf.apply(np.asarray(m.vertices)), n.vertices)


def test_umeyama_recovers_a_similarity_exactly():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3))
    true = align.Similarity(random_rotation(40), np.array([0.3, -1.0, 2.0]), 1.7)
    est = align.umeyama(src, true.apply(src))
    np.testing.assert_allclose(est.rotation, true.rotation, atol=1e-9)
    np.testing.assert_allclose(est.translation, true.translation, atol=1e-9)
    assert est.scale == pytest.approx(1.7)


def test_similarity_composition():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(10, 3))
    a = align.Similarity(random_rotation(20), np.array([1.0, 0, 0]), 2.0)
    b = align.Similarity(random_rotation(-35, (1, 0, 0)), np.array([0, 2.0, 0]), 0.5)
    np.testing.assert_allclose(a.then(b).apply(p), b.apply(a.apply(p)))
    np.testing.assert_allclose(a.matrix()[:3, :3] @ p.T + a.matrix()[:3, 3:], a.apply(p).T)


def test_align_undoes_a_small_misalignment():
    gt, _ = align.normalize_to_unit_cube(ellipsoid())
    gt_s = metrics.sample_surface(gt, 20_000, seed=0)
    pred = gt.copy()
    pred.vertices = align.Similarity(random_rotation(12), np.array([0.05, -0.03, 0.02]), 1.0).apply(
        np.asarray(gt.vertices)
    )
    pred, _ = align.normalize_to_unit_cube(pred)
    sample = metrics.sample_surface(pred, 20_000, seed=2)
    before = metrics.geometry_metrics(sample, gt_s)
    tf = align.align(sample.points, gt_s.points)
    after = metrics.geometry_metrics(
        metrics.SurfaceSample(tf.apply(sample.points), sample.normals @ tf.rotation.T), gt_s
    )
    assert after["chamfer"] < 0.02 < before["chamfer"]
    assert after["fscore@0.05"] > 0.99
    assert after["normal_consistency"] > 0.97


def test_align_does_not_shrink_a_partial_prediction_to_nothing():
    gt, _ = align.normalize_to_unit_cube(ellipsoid())
    gt_s = metrics.sample_surface(gt, 10_000)
    keep = gt_s.points[:, 0] > -0.2  # a prediction missing the back
    tf = align.align(gt_s.points[keep], gt_s.points)
    assert 0.85 < tf.scale < 1.15


# ---------------------------------------------------------------------------- rendering


def textured_quad_mesh() -> trimesh.Trimesh:
    """A square in the x=0 plane; its texture is red for u < 0.5 (world y < 0) and blue for u > 0.5."""
    verts = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    tex = np.zeros((4, 4, 3), np.uint8)
    tex[:, :2] = (255, 0, 0)
    tex[:, 2:] = (0, 0, 255)
    visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(image=Image.fromarray(tex))
    )
    return trimesh.Trimesh(verts, faces, visual=visual, process=False)


def test_render_samples_the_texture_with_the_right_orientation():
    cam = cameras.orbit_camera(0.0, 0.0, 6.0, 30.0, 64)  # on +x looking at -x, z up: image right is world +y
    rgb, alpha = render.render(textured_quad_mesh(), cam, shading=False, ssaa=1)
    assert alpha[32, 32] == 1.0 and alpha[0, 0] == 0.0
    left, right = rgb[32, 26], rgb[32, 38]
    # u grows with world y, so u=0 (red) is on the image's left and u=1 (blue) on its right.
    assert left[0] > 0.9 and left[2] < 0.1
    assert right[2] > 0.9 and right[0] < 0.1
    np.testing.assert_allclose(rgb[0, 0], 1.0)  # white background


def test_render_silhouette_and_soft_edges():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    cam = cameras.orbit_camera(30.0, 20.0, 5.0, 30.0, 96)
    rgb, alpha = render.render(sphere, cam)
    # Projected disc radius: f * r / sqrt(d^2 - r^2), f = 48 / tan(15 deg)
    expected = (48 / math.tan(math.radians(15))) / math.sqrt(24)
    assert alpha.sum() == pytest.approx(math.pi * expected**2, rel=0.05)
    assert ((alpha > 0) & (alpha < 1)).any()  # anti-aliased border
    img = render.to_rgba_image(rgb, alpha)
    assert img.mode == "RGBA" and img.size == (96, 96)


# ---------------------------------------------------------------------------- GSO helpers


def test_subset_is_deterministic_order_free_and_nested():
    names = [f"Object_{i:04d}" for i in range(500)]
    a = gso.select_subset(names, 30)
    assert a == gso.select_subset(list(reversed(names)), 30)
    assert gso.select_subset(names, 50)[:30] == a
    assert gso.select_subset(names, 30, salt="other") != a
    assert len(set(a)) == 30


def test_list_models_pages_through_the_catalogue(monkeypatch):
    pages = {1: [{"name": f"m{i}"} for i in range(3)], 2: [{"name": "m3"}]}

    def fake_get(url, timeout=0):
        page = int(url.split("page=")[1].split("&")[0])
        return json.dumps(pages.get(page, [])).encode()

    monkeypatch.setattr(gso, "_get", fake_get)
    assert gso.list_models(per_page=3) == ["m0", "m1", "m2", "m3"]


def make_gso_zip(mesh: trimesh.Trimesh) -> bytes:
    """A model zip laid out like Gazebo Fuel's: meshes/model.obj + .mtl, texture under materials/."""
    uv = np.c_[(mesh.vertices[:, 0] + 1) / 2, (mesh.vertices[:, 2] + 1) / 2]
    lines = ["mtllib model.mtl", "usemtl material_0"]
    lines += [f"v {x} {y} {z}" for x, y, z in mesh.vertices]
    lines += [f"vt {u} {v}" for u, v in uv]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in mesh.faces]
    tex = np.zeros((8, 8, 3), np.uint8)
    tex[..., 1] = 200
    png = io.BytesIO()
    Image.fromarray(tex).save(png, format="PNG")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.sdf", "<sdf/>")
        zf.writestr("meshes/model.obj", "\n".join(lines) + "\n")
        zf.writestr("meshes/model.mtl", "newmtl material_0\nKd 1 1 1\nmap_Kd texture.png\n")
        zf.writestr("materials/textures/texture.png", png.getvalue())
        zf.writestr("thumbnails/0.jpg", b"")
    return buf.getvalue()


def test_download_and_load_a_fuel_style_zip(tmp_path, monkeypatch):
    payload = make_gso_zip(ellipsoid(subdivisions=2))
    urls = []

    def fake_get(url, timeout=0):
        urls.append(url)
        return payload

    monkeypatch.setattr(gso, "_get", fake_get)
    model_dir = gso.download("Toy Box", tmp_path)
    assert urls == ["https://fuel.gazebosim.org/1.0/GoogleResearch/models/Toy%20Box.zip"]
    gso.download("Toy Box", tmp_path)  # cached: no second request
    assert len(urls) == 1
    mesh = gso.load_mesh(model_dir)
    assert len(mesh.faces) == 320
    cam = cameras.orbit_camera(0.0, 0.0, 6.0, 30.0, 32)
    rgb, alpha = render.render(mesh, cam, shading=False, ssaa=1)
    centre = rgb[16, 16]
    assert alpha[16, 16] == 1.0 and centre[1] > 0.7 and centre[0] < 0.1  # the green texture came through


def test_extract_refuses_paths_outside_the_folder(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError, match="outside"):
        gso._safe_extract(buf.getvalue(), tmp_path / "m")
    assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------------------- the whole loop


class FakeBackend:
    """Stand-in for MultiViewBackend. 'good' returns the ground truth moved, rotated a
    little and rescaled (what a perfect single-view method could return); 'blob' returns
    a sphere; 'broken' raises."""

    def __init__(self, gt_by_call):
        self.gt = gt_by_call
        self.view_calls = 0

    def make_views(self, image, options, progress=None):
        self.view_calls += 1
        assert image.mode == "RGBA" and options.seed is not None
        views = np.ones((6, 32, 32, 3), np.float32)
        masks = np.zeros((6, 32, 32), np.float32)
        return None, views, masks

    def reconstruct_views(self, views, masks, options, progress=None):
        if options.reconstructor == "broken":
            raise RuntimeError("CUDA out of memory (pretend)")
        if options.reconstructor == "blob":
            return trimesh.creation.icosphere(subdivisions=3, radius=3.0), None
        m = self.gt.copy()
        tf = align.Similarity(random_rotation(8), np.array([10.0, 0, 0]), 2.5)
        m.vertices = tf.apply(np.asarray(m.vertices))
        colors = np.full((len(m.vertices), 4), 200, np.uint8)
        return m, colors


def test_benchmark_end_to_end_with_a_fake_backend(tmp_path):
    gt_mesh = ellipsoid(radii=(3.0, 1.8, 2.4), subdivisions=3)  # GSO units are arbitrary
    payload = make_gso_zip(ellipsoid(subdivisions=3))

    def fetch(name, data_dir):
        d = data_dir / name
        if gso.find_obj(d) is None:
            d.mkdir(parents=True, exist_ok=True)
            gso._safe_extract(payload, d)
        return d

    gt_norm, _ = align.normalize_to_unit_cube(gso.load_mesh(fetch("A", tmp_path / "data")))
    backend = FakeBackend(gt_norm)
    del gt_mesh
    cfg = runner.BenchConfig(
        methods=("good", "blob", "broken"),
        n_objects=2,
        input_size=64,
        n_points=5000,
        eval_views=2,
        oracle=True,
    )
    out = tmp_path / "out"
    catalogue = ["A", "B", "C"]
    runner.run(cfg, backend, out, tmp_path / "data", fetch=fetch, list_models=lambda: catalogue)

    rows = report.load_rows(out / runner.RESULTS_FILE)
    assert len(rows) == 2 * 3 * 2  # objects x methods x view sources
    assert backend.view_calls == 2  # generated views made once per object, shared by every method
    subset = json.loads((out / runner.SUBSET_FILE).read_text())
    assert subset["names"] == gso.select_subset(catalogue, 2) and subset["license"] == "CC BY 4.0"

    good = [r for r in rows if r["method"] == "good"]
    blob = [r for r in rows if r["method"] == "blob"]
    broken = [r for r in rows if r["method"] == "broken"]
    assert all(r["status"] == "ok" for r in good + blob)
    assert all(r["status"] == "failed" and "out of memory" in r["error"] for r in broken)
    for r in good:
        assert r["aligned"]["chamfer"] < 0.03
        assert r["aligned"]["fscore@0.05"] > 0.97
        assert r["render"]["mask_iou"] > 0.9
        assert r["time_recon_s"] >= 0 and r["faces"] > 0
    for r in blob:
        assert r["aligned"]["chamfer"] > 2 * max(g["aligned"]["chamfer"] for g in good)
    obj = out / "objects" / subset["names"][0]
    assert (obj / "input.png").is_file() and (obj / "views_oracle.png").is_file()
    assert (obj / "good_generated.ply").is_file()
    oracle_grid = np.asarray(Image.open(obj / "views_oracle.png"))
    assert oracle_grid.shape == (960, 640, 3) and oracle_grid.min() < 100  # the object is drawn

    # Resuming does nothing new; retry_failed only re-runs the failures.
    runner.run(cfg, backend, out, tmp_path / "data", fetch=fetch, list_models=lambda: [])
    assert len(report.load_rows(out / runner.RESULTS_FILE)) == 12
    assert backend.view_calls == 2
    cfg.retry_failed = True
    runner.run(cfg, backend, out, tmp_path / "data", fetch=fetch, list_models=lambda: [])
    assert backend.view_calls == 4

    paths = report.write_report(out)
    md = paths["markdown"].read_text()
    assert "| good | 2/2 |" in md and "| broken | 0/2 |" in md
    assert "good, ground-truth views" in md
    svg = paths["chart"].read_text()
    assert svg.startswith("<svg") and "F-score @ 0.1" in svg
    assert paths["csv"].read_text().count("\n") == 13


def test_report_paired_statistics():
    rows = []
    for i in range(12):
        rows.append(
            {
                "object": f"o{i}",
                "method": "instantmesh",
                "views": "generated",
                "status": "ok",
                "aligned": {"chamfer": 0.05 + 0.001 * i},
                "time_recon_s": 10.0,
            }
        )
        rows.append(
            {
                "object": f"o{i}",
                "method": "neus",
                "views": "generated",
                "status": "ok",
                "aligned": {"chamfer": 0.08 + 0.001 * i},
                "time_recon_s": 300.0,
            }
        )
    pr = report.paired(rows, "instantmesh", "neus", "generated", "aligned.chamfer", higher_better=False)
    assert pr["wins"] == 12 and pr["n"] == 12 and pr["p"] < 0.01
    md = report.markdown(rows)
    assert "InstantMesh better on 12/12 objects" in md
    assert "| InstantMesh | 12/12 |" in md and "| NeuS (classic) | 12/12 |" in md
    lo, hi = report.bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert 1.0 <= lo < 2.5 < hi <= 4.0


def test_cli_report(tmp_path, capsys):
    rows = [
        {
            "object": "o",
            "method": "neus",
            "views": "generated",
            "status": "ok",
            "aligned": {"chamfer": 0.1, "fscore@0.1": 0.5},
            "time_recon_s": 3.0,
        }
    ]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert cli_main(["report", "--out", str(tmp_path)]) == 0
    assert "NeuS (classic)" in capsys.readouterr().out
    assert Path(tmp_path / "chart.svg").is_file()
