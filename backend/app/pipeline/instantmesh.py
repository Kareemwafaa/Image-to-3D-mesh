"""Stage 4 (default): feed-forward reconstruction with InstantMesh (Xu et al., 2024).

NeuS fits a new neural field to every object by gradient descent, which takes
minutes and has to cope with views that are not perfectly consistent.
InstantMesh is a Large Reconstruction Model (LRM): a network trained on
Objaverse to map exactly these six Zero123++ views to a 3D shape in a single
forward pass (seconds).

    6 views + camera poses
      -> DINO ViT encoder (each view's patch tokens, modulated by its camera
         through adaptive LayerNorm)
      -> transformer decoder: learned triplane tokens cross-attend to all
         view tokens, giving three axis-aligned feature planes (a "triplane")
      -> small MLPs read features at any 3D point (sum of the three bilinear
         plane samples) and predict SDF, a vertex offset, and colour
      -> FlexiCubes (Shen et al., 2023) turns the SDF on a 128^3 grid into a
         mesh. Unlike marching cubes it has learned per-cube weights and
         vertex offsets, so the extracted surface follows sharp features and
         was supervised directly in training (rendered depth/normals/masks).

The model code is Apache-2.0 and is not on PyPI, so we download the upstream
source at a pinned commit (see `ensure_source`) instead of copying ~3k lines
into this repo. Upstream imports nvdiffrast at module level, but only uses it
for rendering during training and for its own texture-map export; mesh
extraction is pure PyTorch. When nvdiffrast is not installed (it needs a CUDA
compiler) we register a stub module so the imports succeed; anything that
really calls it fails with a clear message.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Pinned upstream commit (2025-01-03, the latest on main when this was written).
INSTANTMESH_COMMIT = "08822c52fdc399b93ea00e4fa9e596344ed52ccc"
INSTANTMESH_REPO = "TencentARC/InstantMesh"  # Hugging Face repo with the weights
PACKAGE = "instantmesh_src"  # upstream's `src/` package, renamed so it cannot clash with anything

# Hyper-parameters of the released checkpoints (configs/instant-mesh-{large,base}.yaml upstream).
VARIANTS: dict[str, dict] = {
    "large": dict(
        encoder_feat_dim=768,
        encoder_freeze=False,
        encoder_model_name="facebook/dino-vitb16",
        transformer_dim=1024,
        transformer_layers=16,
        transformer_heads=16,
        triplane_low_res=32,
        triplane_high_res=64,
        triplane_dim=80,
        rendering_samples_per_ray=128,
        grid_res=128,
        grid_scale=2.1,
    ),
    "base": dict(
        encoder_feat_dim=768,
        encoder_freeze=False,
        encoder_model_name="facebook/dino-vitb16",
        transformer_dim=1024,
        transformer_layers=12,
        transformer_heads=16,
        triplane_low_res=32,
        triplane_high_res=64,
        triplane_dim=40,
        rendering_samples_per_ray=96,
        grid_res=128,
        grid_scale=2.1,
    ),
}

# The white-background Zero123++ UNet that InstantMesh was trained with.
WHITE_BG_UNET_FILE = "diffusion_pytorch_model.bin"

_import_lock = threading.Lock()


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "facet"


def source_dir(cache_dir: Path | None = None) -> Path:
    """Directory that contains the `instantmesh_src` package."""
    override = os.environ.get("INSTANTMESH_SRC")
    if override:
        return Path(override)
    return (cache_dir or default_cache_dir()) / f"instantmesh-{INSTANTMESH_COMMIT[:12]}"


def _extract_src(archive: bytes, dest: Path) -> None:
    """Copy `<top>/src/**` and `<top>/LICENSE` from the GitHub tarball into dest/instantmesh_src."""
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) >= 2 and (parts[1] == "src" or parts[1:] == ("LICENSE",)) and (m.isfile() or m.isdir()):
                members.append(m)
        if not any(Path(m.name).parts[1:3] == ("src", "models") for m in members):
            raise RuntimeError("The InstantMesh archive has no src/models directory")
        try:
            tar.extractall(tmp, members=members, filter="data")
        except TypeError:  # pragma: no cover - Python without tarfile filters
            tar.extractall(tmp, members=members)  # noqa: S202 - members are filtered above
        top = Path(tmp) / Path(members[0].name).parts[0]
        staging = dest.with_name(dest.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copytree(top / "src", staging / PACKAGE)
        if (top / "LICENSE").is_file():
            shutil.copy(top / "LICENSE", staging / PACKAGE / "LICENSE")
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)


def ensure_source(cache_dir: Path | None = None, timeout: float = 60.0) -> Path:
    """Make sure the pinned InstantMesh source is on disk; return the directory to put on sys.path."""
    dest = source_dir(cache_dir)
    if (dest / PACKAGE / "models" / "lrm_mesh.py").is_file():
        return dest
    if os.environ.get("INSTANTMESH_SRC"):
        raise RuntimeError(f"INSTANTMESH_SRC={dest} does not contain {PACKAGE}/models/lrm_mesh.py")
    url = f"https://codeload.github.com/TencentARC/InstantMesh/tar.gz/{INSTANTMESH_COMMIT}"
    log.info("Downloading InstantMesh source (%s)", INSTANTMESH_COMMIT[:12])
    req = urllib.request.Request(url, headers={"User-Agent": "facet-image-to-3d"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        archive = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _extract_src(archive, dest)
    return dest


def _install_nvdiffrast_stub() -> None:
    try:
        import nvdiffrast.torch  # noqa: F401

        return
    except ImportError:
        pass

    def _missing(*_args, **_kwargs):
        raise RuntimeError(
            "nvdiffrast is not installed. Facet only needs it for InstantMesh's own "
            "texture export, which it does not use; mesh extraction works without it."
        )

    def _getattr(name: str):
        if name.startswith("__"):  # keep introspection (inspect, pickle, ...) working
            raise AttributeError(name)
        return _missing

    pkg = types.ModuleType("nvdiffrast")
    mod = types.ModuleType("nvdiffrast.torch")
    mod.__getattr__ = _getattr  # type: ignore[method-assign]
    pkg.torch = mod  # type: ignore[attr-defined]
    sys.modules.setdefault("nvdiffrast", pkg)
    sys.modules.setdefault("nvdiffrast.torch", mod)


def import_upstream(cache_dir: Path | None = None) -> types.SimpleNamespace:
    """Import the upstream modules we use, downloading them first if needed."""
    with _import_lock:
        root = ensure_source(cache_dir)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _install_nvdiffrast_stub()
        import importlib

        lrm_mesh = importlib.import_module(f"{PACKAGE}.models.lrm_mesh")
        fc_geometry = importlib.import_module(f"{PACKAGE}.models.geometry.rep_3d.flexicubes_geometry")
        camera_util = importlib.import_module(f"{PACKAGE}.utils.camera_util")
        plane_utils = importlib.import_module(f"{PACKAGE}.models.renderer.utils.renderer")
        return types.SimpleNamespace(
            sample_from_planes=plane_utils.sample_from_planes,
            InstantMesh=lrm_mesh.InstantMesh,
            FlexiCubesGeometry=fc_geometry.FlexiCubesGeometry,
            input_cameras=camera_util.get_zero123plus_input_cameras,
        )


def chunked_geometry_prediction(synthesizer, sample_from_planes, chunk: int = 2**18):
    """A drop-in for upstream's TriplaneSynthesizer.get_geometry_prediction that
    never materialises a full-grid tensor.

    Upstream predicts the FlexiCubes weights from the features of each cube's 8
    corners, gathered for all 128^3 cubes at once: 2.1M x 8 x 240 floats, a single
    15 GiB tensor (fine on the A100s it was trained on, an out-of-memory error on a
    16 GB T4). Here points and cubes are processed in chunks, and each chunk's
    corner features are re-sampled from the triplane instead of gathered from a
    stored full-grid copy. Same maths, same outputs, peak memory ~chunk-sized.
    """

    def predict(planes, sample_coordinates, flexicubes_indices):
        import torch

        axes = synthesizer.plane_axes.to(planes.device)
        warp = synthesizer.rendering_kwargs["box_warp"]
        dec = synthesizer.decoder

        def features(points):  # (N, K, 3) -> (N, K, 3 * C)
            f = sample_from_planes(axes, planes, points, padding_mode="zeros", box_warp=warp)
            n, p, k, c = f.shape
            return f.permute(0, 2, 1, 3).reshape(n, k, p * c)

        n_points = sample_coordinates.shape[1]
        sdf, deformation = [], []
        for i in range(0, n_points, chunk):
            f = features(sample_coordinates[:, i : i + chunk])
            sdf.append(dec.net_sdf(f))
            deformation.append(dec.net_deformation(f))
            del f

        weight = []
        cubes_per_chunk = max(1, chunk // flexicubes_indices.shape[1])
        for j in range(0, flexicubes_indices.shape[0], cubes_per_chunk):
            idx = flexicubes_indices[j : j + cubes_per_chunk]
            f = features(sample_coordinates[:, idx.reshape(-1)])
            weight.append(dec.net_weight(f.reshape(f.shape[0], idx.shape[0], -1)) * 0.1)
            del f
        return torch.cat(sdf, 1), torch.cat(deformation, 1), torch.cat(weight, 1)

    return predict


def looks_like_the_grid_box(vertices: np.ndarray, half_extent: float, tol: float = 0.05) -> bool:
    """True if the mesh reaches the FlexiCubes grid's boundary on all six sides.

    When the predicted SDF has no zero crossing, upstream forces the grid's
    boundary to be "outside" so that extraction still returns something: the
    result is a cube the size of the whole grid. A real object is normalised to
    fit well inside the grid, so touching every face means the reconstruction
    failed (usually because the views' background was not pure white).
    """
    if len(vertices) == 0:
        return True
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    edge = half_extent * (1 - tol)
    return bool((lo <= -edge).all() and (hi >= edge).all())


@dataclass
class LRMOutput:
    vertices: np.ndarray  # (N, 3) float32, z-up world frame of cameras.zero123pp_cameras
    faces: np.ndarray  # (F, 3) int64
    vertex_colors: np.ndarray  # (N, 3) uint8, from the model's colour field


class InstantMeshReconstructor:
    """Wraps the upstream model: load once, then views -> mesh in one forward pass."""

    def __init__(
        self,
        variant: str = "large",
        device: str = "cuda",
        hf_repo: str = INSTANTMESH_REPO,
        grid_res: int | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown InstantMesh variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = variant
        self.device = device
        self.hf_repo = hf_repo
        # FlexiCubes grid. The checkpoints were trained at 128; the triplane is continuous,
        # so a finer grid (e.g. 192) resolves thin walls and cavities better. GUESS:
        # quality at other resolutions has not been measured.
        self.grid_res = grid_res
        self.model = None
        self._upstream = None
        self._lock = threading.Lock()

    def build(self, params: dict | None = None, state_dict: dict | None = None):
        """Construct the network (and its FlexiCubes grid) on self.device.
        `params`/`state_dict` override the released config/weights (used by tests)."""
        up = self._upstream = import_upstream()
        cfg = dict(VARIANTS[self.variant])
        if self.grid_res:
            cfg["grid_res"] = self.grid_res
        cfg.update(params or {})
        model = up.InstantMesh(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device).eval()
        model.synthesizer.get_geometry_prediction = chunked_geometry_prediction(
            model.synthesizer, up.sample_from_planes
        )
        # Upstream's init_flexicubes_geometry() also builds an nvdiffrast renderer,
        # which inference does not need; build the geometry alone.
        model.geometry = up.FlexiCubesGeometry(
            grid_res=model.grid_res,
            scale=model.grid_scale,
            renderer=None,
            render_type="neural_render",
            device=self.device,
        )
        self.model = model
        return model

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            import torch
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(self.hf_repo, f"instant_mesh_{self.variant}.ckpt")
            log.info("Loading InstantMesh-%s from %s", self.variant, ckpt)
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
            prefix = "lrm_generator."
            state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
            self.build(state_dict=state)

    def reconstruct(self, views: np.ndarray, radius: float = 4.0) -> LRMOutput:
        """views: (6, H, W, 3) float in [0, 1], Zero123++ order, white background."""
        import torch
        import torch.nn.functional as F

        if self.model is None:
            self.load()
        assert self.model is not None and self._upstream is not None
        if views.shape[0] != 6:
            raise ValueError(f"InstantMesh needs the 6 Zero123++ views, got {views.shape[0]}")
        images = torch.from_numpy(np.ascontiguousarray(views)).float().permute(0, 3, 1, 2)
        if images.shape[-1] != 320:
            images = F.interpolate(images, size=(320, 320), mode="bicubic", antialias=True)
        images = images.clamp(0, 1).unsqueeze(0).to(self.device)
        cams = self._upstream.input_cameras(batch_size=1, radius=radius).to(self.device)
        with torch.no_grad():
            planes = self.model.forward_planes(images, cams)
            verts, faces, colors = self.model.extract_mesh(planes, use_texture_map=False)
        verts = np.asarray(verts, dtype=np.float32)
        if looks_like_the_grid_box(verts, self.model.grid_scale / 2):
            from .errors import MeshingError

            raise MeshingError(
                "InstantMesh returned the empty-shape fallback (a box filling the whole grid). "
                "Check the generated views: the object must be on a pure white background."
            )
        return LRMOutput(
            vertices=verts,
            faces=np.asarray(faces, dtype=np.int64),
            vertex_colors=np.asarray(colors, dtype=np.uint8),
        )

    def unload(self) -> None:
        self.model = None
