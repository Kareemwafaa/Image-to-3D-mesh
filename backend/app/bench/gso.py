"""Google Scanned Objects (GSO): download, pick a fixed subset, load meshes.

GSO (Downs et al., ICRA 2022) is 1,030 household objects scanned with a
structured-light rig into textured meshes. It is the standard evaluation set
for single-image 3D (Zero123, SyncDreamer, Wonder3D, InstantMesh all report
on it) because the objects are real, everyday and were never part of the
Objaverse training data these models learn from.

License: CC BY 4.0 (Creative Commons Attribution 4.0 International). We do
not redistribute any meshes; they are downloaded at run time from Gazebo Fuel,
where Google Research publishes them. Credit: "Google Scanned Objects" by
Google Research, https://goo.gle/scanned-objects, CC BY 4.0.

Subset. Rather than hand-pick objects (which invites picking easy ones), the
subset is chosen by hashing: every model name is hashed together with a fixed
salt, and the N names with the smallest hashes are taken. That is a uniform
random sample, it is reproducible by anyone from the name list alone, it does
not depend on the order the server lists models in, and growing N keeps the
smaller subset (the first 30 of a 50-object run are the 30-object run).
The chosen names are written to `subset.json` next to the results.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

log = logging.getLogger(__name__)

FUEL_API = "https://fuel.gazebosim.org/1.0"
OWNER = "GoogleResearch"
DEFAULT_SALT = "facet-gso-v1"
LICENSE = "CC BY 4.0"
CREDIT = '"Google Scanned Objects" by Google Research (https://goo.gle/scanned-objects), CC BY 4.0'
_UA = {"User-Agent": "facet-benchmark/1.0"}


def _get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        return resp.read()


def list_models(per_page: int = 100, max_pages: int = 50) -> list[str]:
    """Every model name in the GoogleResearch collection on Gazebo Fuel."""
    names: list[str] = []
    for page in range(1, max_pages + 1):
        url = f"{FUEL_API}/{OWNER}/models?page={page}&per_page={per_page}"
        try:
            batch = json.loads(_get(url))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # past the last page
                break
            raise
        if not batch:
            break
        names.extend(item["name"] for item in batch)
        if len(batch) < per_page:
            break
    return sorted(set(names))


def select_subset(names: list[str], count: int, salt: str = DEFAULT_SALT) -> list[str]:
    """The `count` names with the smallest sha256(salt:name). See the module docstring."""

    def key(name: str) -> str:
        return hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()

    return sorted(sorted(set(names), key=key)[:count], key=key)


def download(name: str, dest_root: Path) -> Path:
    """Fetch and unzip one model into dest_root/name (skipped if already there)."""
    dest = dest_root / name
    if find_obj(dest) is not None:
        return dest
    quoted = urllib.parse.quote(name)
    urls = [
        f"{FUEL_API}/{OWNER}/models/{quoted}.zip",
        f"{FUEL_API}/{OWNER}/models/{quoted}/tip/{quoted}.zip",
        f"{FUEL_API}/{OWNER}/models/{quoted}/1/{quoted}.zip",
    ]
    last: Exception | None = None
    for url in urls:
        try:
            data = _get(url)
            break
        except Exception as exc:  # try the next URL form
            last = exc
    else:
        raise RuntimeError(f"Could not download GSO model {name!r}: {last}")
    dest.mkdir(parents=True, exist_ok=True)
    _safe_extract(data, dest)
    if find_obj(dest) is None:
        raise RuntimeError(f"GSO model {name!r} has no meshes/model.obj")
    return dest


def _safe_extract(data: bytes, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Refusing to extract {member.filename!r} outside {dest}")
        zf.extractall(dest)


def find_obj(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    preferred = model_dir / "meshes" / "model.obj"
    if preferred.is_file():
        return preferred
    found = sorted(model_dir.rglob("*.obj"))
    return found[0] if found else None


def load_mesh(model_dir: Path) -> trimesh.Trimesh:
    """The model's textured mesh, z-up, in its own units (normalise it with align.py)."""
    obj = find_obj(model_dir)
    if obj is None:
        raise FileNotFoundError(f"No .obj under {model_dir}")
    mesh = trimesh.load(obj, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{obj} did not load as a triangle mesh")
    _attach_texture(mesh, model_dir)
    return mesh


def _attach_texture(mesh: trimesh.Trimesh, model_dir: Path) -> None:
    """GSO's .mtl names `texture.png` but the file lives in materials/textures/, which
    trimesh's loader does not look in. Attach it by hand when the UVs came through."""
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    if uv is None:
        return
    material = getattr(visual, "material", None)
    if material is not None and (getattr(material, "image", None) is not None):
        return
    candidates = sorted(model_dir.rglob("texture.png")) or sorted(model_dir.rglob("*.png"))
    candidates = [p for p in candidates if "thumbnail" not in p.parts]
    if not candidates:
        log.warning("%s: UVs but no texture image; rendering it untextured", model_dir.name)
        return
    image = Image.open(candidates[0]).convert("RGB")
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(uv), material=trimesh.visual.material.SimpleMaterial(image=image)
    )
