import io
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.pipeline import segmentation as preprocess
from app.config import Settings
from app.main import create_app


def png_bytes(size=(64, 64), color=(200, 30, 30, 255), mode="RGBA") -> bytes:
    img = Image.new(mode, size, (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255))
    # draw a square "object" in the middle
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), color if mode == "RGBA" else color[:3])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # rembg downloads a model on first use; tests fake it with a white-is-background mask.
    def fake_remove_background(image):
        rgba = np.asarray(image.convert("RGBA")).copy()
        rgba[..., 3] = np.where(rgba[..., :3].min(axis=-1) > 240, 0, 255)
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(preprocess, "remove_background", fake_remove_background)
    settings = Settings(backend="mock", data_dir=tmp_path, max_upload_mb=1, cors_origins=["http://localhost:3000"])
    with TestClient(create_app(settings)) as c:
        yield c


def wait_for(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "mock"


@pytest.mark.parametrize("mode", ["RGBA", "RGB"])
def test_generate_and_download(client, mode):
    r = client.post("/generate", files={"image": ("obj.png", png_bytes(mode=mode), "image/png")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    body = wait_for(client, job_id)
    assert body["status"] == "succeeded", body
    assert body["progress"] == 1.0
    assert body["error"] is None
    assert body["result_url"] == f"/jobs/{job_id}/result"
    assert body["stage"] == "done"
    assert body["views_url"] == f"/jobs/{job_id}/views"
    views = client.get(body["views_url"])
    assert views.status_code == 200 and views.headers["content-type"] == "image/png"

    mesh = client.get(body["result_url"])
    assert mesh.status_code == 200
    assert mesh.headers["content-type"] == "model/gltf-binary"
    scene = trimesh.load(io.BytesIO(mesh.content), file_type="glb")
    assert len(scene.geometry) == 1


def test_without_background_removal(client):
    r = client.post(
        "/generate",
        files={"image": ("obj.png", png_bytes(mode="RGB"), "image/png")},
        data={"remove_background": "false", "mc_resolution": "128"},
    )
    assert r.status_code == 202
    assert wait_for(client, r.json()["job_id"])["status"] == "succeeded"


def test_empty_foreground_fails_cleanly(client):
    blank = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    r = client.post(
        "/generate",
        files={"image": ("blank.png", buf.getvalue(), "image/png")},
        data={"remove_background": "false"},
    )
    body = wait_for(client, r.json()["job_id"])
    assert body["status"] == "failed"
    assert "foreground" in body["error"]
    assert client.get(f"/jobs/{body['job_id']}/result").status_code == 409


def test_rejects_non_image(client):
    r = client.post("/generate", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_rejects_too_large(client):
    r = client.post("/generate", files={"image": ("big.png", b"0" * (1024 * 1024 + 10), "image/png")})
    assert r.status_code == 413


def test_rejects_bad_params(client):
    r = client.post(
        "/generate",
        files={"image": ("obj.png", png_bytes(), "image/png")},
        data={"mc_resolution": "4096"},
    )
    assert r.status_code == 422


def test_unknown_job(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/result").status_code == 404


def test_cors(client):
    r = client.options(
        "/generate",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_resize_foreground_centres_object():
    img = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    for x in range(10, 30):
        for y in range(5, 15):
            img.putpixel((x, y), (255, 0, 0, 255))
    out = preprocess.resize_foreground(img, 0.5)
    assert out.size == (40, 40)
    alpha = np.asarray(out)[..., 3]
    ys, xs = np.nonzero(alpha)
    assert (xs.min(), xs.max()) == (10, 29)
    assert (ys.min(), ys.max()) == (15, 24)
