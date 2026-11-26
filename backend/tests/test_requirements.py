"""The GPU lock must hold every package the Zero123++ custom pipeline imports.

diffusers loads `sudo-ai/zero123plus-pipeline` at runtime and refuses to run it
if any of its top-level imports is missing, so a gap here only shows up on a
GPU notebook after a 5 GB download. Catch it in CI instead.
"""

from __future__ import annotations

import re
from pathlib import Path

REQS = Path(__file__).resolve().parents[1] / "requirements"

# Third-party imports of the pipeline's pipeline.py (diffusers-support branch).
ZERO123PLUS_IMPORTS = {"torch", "torchvision", "diffusers", "transformers", "numpy", "pillow"}


def pinned(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#;]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_lite_gpu_lock_has_every_zero123plus_import():
    lock = pinned(REQS / "lite-gpu.txt")
    assert ZERO123PLUS_IMPORTS <= lock.keys(), ZERO123PLUS_IMPORTS - lock.keys()


# Everything the InstantMesh source, cleanup and texture baking import (see app/doctor.py).
QUALITY_IMPORTS = {"einops", "opencv-python-headless", "scipy", "fast-simplification", "xatlas", "trimesh"}


def test_lite_gpu_lock_has_the_quality_stage_packages():
    lock = pinned(REQS / "lite-gpu.txt")
    assert QUALITY_IMPORTS <= lock.keys(), QUALITY_IMPORTS - lock.keys()


def test_doctor_checks_every_locked_runtime_package():
    """The notebook's dependency check must cover each package the stages import."""
    from app.doctor import REQUIRED

    checked = {pkg for _, pkg, _ in REQUIRED}
    assert (ZERO123PLUS_IMPORTS - {"numpy", "pillow"}) | QUALITY_IMPORTS <= checked


def test_torchvision_matches_torch():
    """torchvision 0.N+15 is built for torch 2.N; a mismatch fails at import time."""
    lock = pinned(REQS / "lite-gpu.txt")
    torch_minor = int(lock["torch"].split(".")[1])
    vision_minor = int(lock["torchvision"].split(".")[1])
    pair = (lock["torch"], lock["torchvision"])
    assert lock["torchvision"].startswith("0.") and vision_minor == torch_minor + 15, pair


def test_doctor_reports_missing_packages(capsys, monkeypatch):
    from app import doctor

    checks = doctor.check_imports((("json", "json", "x"), ("surely_not_installed_pkg", "nope", "the test")))
    assert [c.ok for c in checks] == [True, False]
    assert "pip install nope" in checks[1].detail

    monkeypatch.setattr(doctor, "run", lambda **_: checks)
    assert doctor.main(["--no-gpu"]) == 1
    assert "FAIL" in capsys.readouterr().out
