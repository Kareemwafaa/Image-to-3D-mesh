"""Gateway behaviour, run against lite mode with the mock model."""

from __future__ import annotations

import io

import trimesh

from .conftest import ADMIN_KEY, client_for, make_settings, png_bytes, submit, wait_for


def test_submit_poll_download(client):
    r = submit(client)
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    assert r.headers["Location"] == f"/v1/jobs/{job_id}"
    assert body["events_url"] == f"/v1/jobs/{job_id}/events"

    done = wait_for(client, job_id)
    assert done["status"] == "succeeded", done
    assert done["stage"] == "done" and done["progress"] == 1.0 and done["attempts"] == 1
    mesh = client.get(done["result_url"])
    assert mesh.status_code == 200 and mesh.headers["content-type"] == "model/gltf-binary"
    assert len(trimesh.load(io.BytesIO(mesh.content), file_type="glb").geometry) == 1
    views = client.get(done["views_url"])
    assert views.status_code == 200 and views.headers["content-type"] == "image/png"


def test_legacy_routes_still_work(client):
    """The web app's original contract: /generate, /jobs/{id}, /jobs/{id}/result, /health."""
    r = submit(client, path="/generate")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    wait_for(client, job_id)
    legacy = client.get(f"/jobs/{job_id}").json()
    assert {"job_id", "status", "stage", "progress", "result_url", "views_url", "error", "created_at"} <= set(legacy)
    assert client.get(f"/jobs/{job_id}/result").status_code == 200
    assert client.get(f"/jobs/{job_id}/views").status_code == 200
    health = client.get("/health").json()
    assert health["model_loaded"] is True and health["status"] == "ok"


def test_idempotency_replay_and_conflict(client):
    h = {"Idempotency-Key": "order-42"}
    first = submit(client, headers=h)
    again = submit(client, headers=h)
    assert first.status_code == 202 and again.status_code == 200
    assert again.headers["Idempotent-Replayed"] == "true"
    assert again.json()["job_id"] == first.json()["job_id"]

    other_image = submit(client, data=png_bytes(color=(0, 0, 255, 255)), headers=h)
    assert other_image.status_code == 422
    assert other_image.json()["code"] == "idempotency_key_reused"

    other_options = submit(client, headers=h, mc_resolution="128")
    assert other_options.status_code == 422


def test_validation_errors_are_problem_json(client):
    bad = client.post("/v1/jobs", files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    assert bad.headers["content-type"].startswith("application/problem+json")
    assert bad.json()["code"] == "invalid_image" and bad.json()["request_id"]

    assert submit(client, mc_resolution="4096").json()["code"] == "validation_error"
    assert client.get("/v1/jobs/not-a-uuid").status_code == 404
    assert client.get("/v1/jobs/00000000-0000-0000-0000-000000000000").json()["code"] == "job_not_found"


def test_upload_size_limit(tmp_path):
    with client_for(make_settings(tmp_path, max_upload_mb=0.01)) as client:
        r = submit(client, data=b"0" * 20_000)
        assert r.status_code == 413 and r.json()["code"] == "payload_too_large"


def test_failed_job_reports_error(client):
    blank = io.BytesIO()
    from PIL import Image

    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(blank, format="PNG")
    job_id = submit(client, data=blank.getvalue()).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "failed" and body["error_code"] == "no_foreground"
    assert body["attempts"] == 1  # permanent errors are not retried
    assert client.get(f"/v1/jobs/{job_id}/result").status_code == 409


def test_sse_stream(client):
    job_id = submit(client).json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert text.startswith("event: snapshot")
    assert "event: done" in text
    assert '"status": "succeeded"' in text


def test_auth_required_mode_and_api_keys(tmp_path):
    with client_for(make_settings(tmp_path, auth_mode="required")) as client:
        assert submit(client).status_code == 401
        assert client.get("/v1/admin/api-keys").status_code == 401
        assert client.get("/v1/admin/api-keys", headers={"X-API-Key": "i3d_deadbeef_" + "x" * 30}).status_code == 401

        admin = {"X-API-Key": ADMIN_KEY}
        created = client.post("/v1/admin/api-keys", json={"name": "alice"}, headers=admin)
        assert created.status_code == 201
        key = created.json()["api_key"]
        assert key.startswith("i3d_" + created.json()["prefix"])
        alice = {"Authorization": f"Bearer {key}"}
        listed = client.get("/v1/admin/api-keys", headers=admin).json()
        assert all("api_key" not in k for k in listed)  # plaintext is never shown again
        assert client.get("/v1/admin/api-keys", headers=alice).status_code == 403

        job_id = submit(client, headers=alice).json()["job_id"]
        wait_for(client, job_id, headers=alice)
        assert [j["job_id"] for j in client.get("/v1/jobs", headers=alice).json()["items"]] == [job_id]

        bob_key = client.post("/v1/admin/api-keys", json={"name": "bob"}, headers=admin).json()["api_key"]
        assert client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": bob_key}).status_code == 404
        assert client.get(f"/v1/jobs/{job_id}", headers=admin).status_code == 200

        key_id = created.json()["id"]
        assert client.delete(f"/v1/admin/api-keys/{key_id}", headers=admin).status_code == 204
        assert client.get("/v1/jobs", headers=alice).status_code == 401


def test_rate_limit(tmp_path):
    with client_for(make_settings(tmp_path, anon_submit_per_minute=1, anon_submit_burst=2)) as client:
        assert submit(client).status_code == 202
        second = submit(client)
        assert second.status_code == 202 and second.headers["RateLimit-Remaining"] == "0"
        third = submit(client)
        assert third.status_code == 429
        assert int(third.headers["Retry-After"]) >= 1
        assert third.json()["code"] == "rate_limited"
        # admins are exempt
        assert submit(client, headers={"X-API-Key": ADMIN_KEY}).status_code == 202


def test_inflight_cap(tmp_path):
    from .test_worker import SlowBackend

    s = make_settings(tmp_path, anon_max_inflight=1)
    with client_for(s, backend=SlowBackend(seconds=1.0)) as client:
        assert submit(client).status_code == 202
        r = submit(client)
        assert r.status_code == 429 and r.json()["code"] == "too_many_inflight_jobs"


def test_health_readiness_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["checks"]["database"] == "ok"
    submit(client)
    text = client.get("/metrics").text
    assert "jobs_submitted_total" in text and "http_request_duration_seconds" in text
    assert client.get("/healthz").headers["X-Request-ID"]


def test_cors(tmp_path):
    with client_for(make_settings(tmp_path, cors_origin_regex=r"https://.*\.github\.io")) as client:

        def preflight(origin):
            return client.options(
                "/v1/jobs", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
            ).headers.get("access-control-allow-origin")

        assert preflight("http://localhost:4200") == "http://localhost:4200"
        assert preflight("https://kareemwafaa.github.io") == "https://kareemwafaa.github.io"
        assert preflight("https://evil.example") is None


def test_cors_wildcard_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    from app.core.config import Settings

    assert Settings().cors_origins == ["*"]
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    assert Settings().cors_origins == ["https://a.example", "https://b.example"]


def test_short_env_names_for_defaults(monkeypatch):
    monkeypatch.setenv("DIFFUSION_STEPS", "28")
    monkeypatch.setenv("RECON_ITERS", "800")
    from app.core.config import Settings

    s = Settings()
    assert s.default_diffusion_steps == 28 and s.default_recon_iters == 800
    assert Settings(default_diffusion_steps=40).default_diffusion_steps == 40


def test_reconstructor_and_quality_fields(client):
    assert submit(client, reconstructor="magic").json()["code"] == "validation_error"
    assert submit(client, texture_size="64").json()["code"] == "validation_error"
    for method in ("instantmesh", "neus"):
        r = submit(client, reconstructor=method, texture="false", target_faces="0")
        assert r.status_code == 202
        assert wait_for(client, r.json()["job_id"])["status"] == "succeeded"


def test_quality_defaults_come_from_the_environment(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("RECONSTRUCTOR", "neus")
    monkeypatch.setenv("TARGET_FACES", "20000")
    s = Settings()
    assert s.default_reconstructor == "neus" and s.default_target_faces == 20000
    assert s.zero123pp_unet == "instantmesh" and s.default_texture_size == 1024
