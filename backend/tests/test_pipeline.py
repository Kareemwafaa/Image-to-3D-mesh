"""Tests for the pipeline stages that run without model weights.

The reconstruction test renders a synthetic scene analytically from the six
Zero123++ cameras and checks that NeuS + marching cubes recovers it. torch and
scikit-image are needed; the test is skipped without them.
"""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from app.pipeline import cameras, multiview  # noqa: E402
from app.pipeline.meshing import extract_mesh, zup_to_gltf  # noqa: E402
from app.pipeline.neus import FieldConfig, NeuSModel, ray_box, sample_pdf  # noqa: E402
from app.pipeline.reconstruct import ReconConfig, reconstruct  # noqa: E402


def test_zero123pp_cameras_look_at_origin():
    cams = cameras.zero123pp_cameras(32)
    assert len(cams) == 6
    for cam, el in zip(cams, cameras.ZERO123PP_ELEVATIONS, strict=True):
        pos = cam.c2w[:3, 3]
        assert torch.isclose(pos.norm(), torch.tensor(cameras.ZERO123PP_RADIUS))
        assert np.isclose(np.degrees(np.arcsin(pos[2].item() / pos.norm().item())), el, atol=1e-4)
        o, d = cam.rays()
        # The rays through the 4 centre pixels point (almost) at the origin.
        centre = d.view(32, 32, 3)[15:17, 15:17].mean(dim=(0, 1))
        assert torch.allclose(centre / centre.norm(), -pos / pos.norm(), atol=1e-3)
        # Image "up" has a positive world z component (no upside-down cameras).
        assert cam.c2w[2, 1] > 0


def test_grid_roundtrip():
    views = [Image.new("RGB", (320, 320), (i * 40, 0, 0)) for i in range(6)]
    grid = multiview.make_grid(views)
    assert grid.size == (640, 960)
    back = multiview.split_grid(grid)
    assert [v.getpixel((5, 5))[0] for v in back] == [i * 40 for i in range(6)]


def test_ray_box_and_pdf():
    o = torch.tensor([[3.0, 0.0, 0.0], [3.0, 5.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    near, far, hit = ray_box(o, d, 1.0)
    assert hit.tolist() == [True, False]
    assert torch.allclose(near[0], torch.tensor(2.0)) and torch.allclose(far[0], torch.tensor(4.0))
    bins = torch.linspace(0, 1, 11)[None]
    weights = torch.zeros(1, 10)
    weights[0, 7] = 1.0
    samples = sample_pdf(bins, weights, 8)
    assert ((samples >= 0.69) & (samples <= 0.81)).float().mean() > 0.9


def test_initial_field_is_a_sphere():
    model = NeuSModel(FieldConfig(levels=2, max_res=16))
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sdf = model.sdf_field.sdf(pts).squeeze(-1)
    assert torch.allclose(sdf, torch.tensor([-0.5, 0.0, 0.5]), atol=0.02)
    grad = model.sdf_field.gradient(pts[1:], 1e-3)
    assert torch.allclose(grad.norm(dim=-1), torch.ones(2), atol=0.05)


def test_zup_to_gltf_is_a_rotation():
    m = np.eye(3)[[1, 2, 0]]
    assert np.isclose(np.linalg.det(m), 1.0)
    assert np.allclose(zup_to_gltf(np.array([[0.0, 0.0, 1.0]])), [[0.0, 1.0, 0.0]])  # z-up -> y-up
    assert np.allclose(zup_to_gltf(np.array([[1.0, 0.0, 0.0]])), [[0.0, 0.0, 1.0]])  # input view faces +z


def _render_sphere(cam, centre, radius, colour):
    o, d = cam.rays()
    oc = o - torch.tensor(centre)
    b = (oc * d).sum(-1)
    disc = b * b - ((oc * oc).sum(-1) - radius**2)
    hit = disc > 0
    rgb = torch.ones(o.shape[0], 3)
    rgb[hit] = torch.tensor(colour)
    n = cam.height
    return rgb.view(n, n, 3).numpy(), hit.float().view(n, n).numpy()


@pytest.mark.slow
def test_neus_recovers_a_sphere():
    """Fit an off-centre sphere (the field starts as a centred one) from the six views."""
    centre, radius = (0.15, -0.1, 0.05), 0.6
    cams = cameras.zero123pp_cameras(32)
    rendered = [_render_sphere(c, centre, radius, (0.8, 0.3, 0.2)) for c in cams]
    views = np.stack([v for v, _ in rendered])
    masks = np.stack([m for _, m in rendered])
    cfg = ReconConfig(
        iters=150,
        batch_rays=256,
        n_samples=24,
        n_importance=16,
        log_every=1000,
        field=FieldConfig(levels=3, base_res=8, max_res=32),
    )
    model = reconstruct(views, masks, cams, cfg, device="cpu")
    mesh = extract_mesh(model, resolution=48, device="cpu")
    verts_zup = np.asarray(mesh.vertices)[:, [2, 0, 1]]
    dist = np.abs(np.linalg.norm(verts_zup - np.array(centre), axis=-1) - radius)
    assert dist.mean() < 0.08, dist.mean()
    colours = mesh.visual.vertex_colors[:, :3].mean(axis=0) / 255
    assert colours[0] > colours[2] + 0.2  # reddish, learned from the views
