from __future__ import annotations

import io
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.pipeline import segmentation

ADMIN_KEY = "i3d_adm1n000_test-admin-secret-0123456789"


@pytest.fixture(autouse=True)
def fake_background_removal(monkeypatch):
    """rembg downloads a model on first use; tests treat near-white as background."""

    def fake(image):
        rgba = np.asarray(image.convert("RGBA")).copy()
        rgba[..., 3] = np.where(rgba[..., :3].min(axis=-1) > 240, 0, 255)
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(segmentation, "remove_background", fake)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        mode="lite",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        storage_backend="local",
        storage_dir=str(tmp_path / "files"),
        model_backend="mock",
        log_json=False,
        log_level="WARNING",
        admin_api_key=ADMIN_KEY,
        heartbeat_interval_s=0.2,
        visibility_timeout_s=1.0,
        retry_base_delay_s=0.05,
        retry_max_delay_s=0.1,
        progress_db_interval_s=0.0,
        anon_submit_per_minute=600,
        anon_submit_burst=50,
        anon_max_inflight=50,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@contextmanager
def client_for(settings: Settings, backend=None):
    with TestClient(create_app(settings, backend=backend)) as client:
        wait_ready(client)
        yield client


@pytest.fixture
def client(settings):
    with client_for(settings) as c:
        yield c


def wait_ready(client: TestClient, timeout: float = 10) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError(f"not ready: {client.get('/readyz').json()}")


def png_bytes(size=(64, 64), color=(200, 30, 30, 255), mode="RGBA") -> bytes:
    bg = (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)
    img = Image.new(mode, size, bg)
    img.paste(color if mode == "RGBA" else color[:3], (16, 16, 48, 48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def submit(client: TestClient, data: bytes | None = None, headers=None, path="/v1/jobs", **form):
    form.setdefault("remove_background", "false")
    return client.post(
        path, files={"image": ("obj.png", data or png_bytes(), "image/png")}, data=form, headers=headers or {}
    )


def wait_for(
    client: TestClient, job_id: str, headers=None, timeout: float = 10, states=("succeeded", "failed")
) -> dict:
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get(f"/v1/jobs/{job_id}", headers=headers or {}).json()
        if body.get("status") in states:
            return body
        time.sleep(0.03)
    raise AssertionError(f"job did not reach {states}: {body}")
