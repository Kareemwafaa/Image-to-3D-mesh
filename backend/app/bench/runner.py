"""The benchmark loop: for each object, render an input photo, run each method, score it.

Per object:
  1. Load the ground-truth mesh and normalise it into [-1, 1]^3 (align.py).
  2. Render one input view from the +x axis at a small elevation (render.py) and
     prepare it exactly as the app prepares an upload (crop, centre, pad), but
     with the rendered alpha instead of U^2-Net: segmentation errors are not
     what this benchmark measures.
  3. Generate the six Zero123++ views once and give the *same* views to every
     reconstructor, so the comparison isolates reconstruction (a paired design:
     each object is its own control). With --oracle, also feed the six views
     rendered from the ground truth itself, which measures what each
     reconstructor can do with perfect views; the gap between the two runs is
     the error the diffusion stage adds.
  4. Reconstruct, clean up (the app's own postprocess), align, score; record
     wall time and peak GPU memory of each stage.

Results are appended to results.jsonl one line per (object, method, views),
so a run that hits Kaggle's time limit resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from ..backends.base import GenerateOptions
from ..pipeline import cameras, multiview, segmentation
from . import align, gso, metrics, render

log = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"
SUBSET_FILE = "subset.json"
CONFIG_FILE = "config.json"


@dataclass
class BenchConfig:
    methods: tuple[str, ...] = ("instantmesh", "neus")
    n_objects: int = 30
    names: tuple[str, ...] | None = None  # explicit object list instead of the hashed subset
    salt: str = gso.DEFAULT_SALT
    oracle: bool = False  # also reconstruct from ground-truth renders of the six views
    generated: bool = True  # reconstruct from Zero123++ views (the real pipeline)
    input_size: int = 512
    input_elevation: float = 10.0  # degrees above the horizon
    fov_deg: float = cameras.ZERO123PP_FOV_DEG
    oracle_scale: float = 0.8  # object half-size in the oracle views (the views' radius is 4)
    seed: int = 42
    diffusion_steps: int = 75
    recon_iters: int = 2500  # NeuS
    foreground_ratio: float = 0.85
    n_points: int = 100_000
    thresholds: tuple[float, ...] = metrics.DEFAULT_THRESHOLDS
    eval_views: int = 4  # novel views for mask IoU / PSNR (0 to skip)
    save_meshes: bool = True
    retry_failed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Measure:
    seconds: float = 0.0
    peak_gpu_mib: float | None = None


@contextmanager
def measure() -> Iterator[Measure]:
    """Wall time and peak GPU memory (bytes allocated by PyTorch) of a block."""
    m = Measure()
    cuda = _cuda()
    if cuda:
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield m
    finally:
        if cuda:
            cuda.synchronize()
            m.peak_gpu_mib = cuda.max_memory_allocated() / 2**20
        m.seconds = time.perf_counter() - t0


def _cuda():
    try:
        import torch

        return torch.cuda if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------- inputs


def input_camera(cfg: BenchConfig, gt: trimesh.Trimesh) -> cameras.Camera:
    """Camera on the +x side, far enough that the whole object is in frame."""
    radius_bs = float(np.linalg.norm(np.asarray(gt.vertices), axis=1).max())
    dist = radius_bs / math.sin(math.radians(cfg.fov_deg) / 2) * 1.05
    return cameras.orbit_camera(0.0, cfg.input_elevation, dist, cfg.fov_deg, cfg.input_size)


def eval_cameras(cfg: BenchConfig, gt: trimesh.Trimesh) -> list[cameras.Camera]:
    """Novel views spread around the object (offset from Zero123++'s azimuths), for render metrics."""
    radius_bs = float(np.linalg.norm(np.asarray(gt.vertices), axis=1).max())
    dist = radius_bs / math.sin(math.radians(cfg.fov_deg) / 2) * 1.05
    n = cfg.eval_views
    return [cameras.orbit_camera(360.0 * i / n + 45.0, 15.0, dist, cfg.fov_deg, 256) for i in range(n)]


def oracle_views(gt: trimesh.Trimesh, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The ground truth rendered at the six Zero123++ poses on white: (6, 320, 320, 3), (6, 320, 320)."""
    scaled = gt.copy()
    scaled.vertices = np.asarray(gt.vertices) * scale
    views, masks = [], []
    for cam in cameras.zero123pp_cameras(multiview.VIEW_SIZE):
        rgb, alpha = render.render(scaled, cam, background=1.0)
        views.append(rgb)
        masks.append(alpha)
    return np.stack(views), np.stack(masks)


# ---------------------------------------------------------------------------- scoring


def score(
    pred: trimesh.Trimesh,
    gt: trimesh.Trimesh,
    gt_sample: metrics.SurfaceSample,
    cfg: BenchConfig,
    eval_cams: list[cameras.Camera],
    gt_renders: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], trimesh.Trimesh]:
    """Normalise, align and score one prediction. Returns (metrics, aligned mesh)."""
    pred_n, _ = align.normalize_to_unit_cube(pred)
    sample = metrics.sample_surface(pred_n, cfg.n_points, seed=cfg.seed)
    out: dict[str, Any] = {"unaligned": metrics.geometry_metrics(sample, gt_sample, cfg.thresholds)}

    tf = align.align(sample.points, gt_sample.points, seed=cfg.seed)
    aligned = pred_n.copy()
    aligned.vertices = tf.apply(np.asarray(pred_n.vertices, dtype=np.float64))
    sample_a = metrics.SurfaceSample(tf.apply(sample.points), sample.normals @ tf.rotation.T)
    out["aligned"] = metrics.geometry_metrics(sample_a, gt_sample, cfg.thresholds)
    out["alignment"] = {
        "scale": tf.scale,
        "rotation_deg": _angle_deg(tf.rotation),
        "translation": tf.translation.tolist(),
    }

    if eval_cams:
        per_view = []
        for cam, (gt_rgb, gt_alpha) in zip(eval_cams, gt_renders, strict=True):
            # The prediction's colours come from images (lighting already in them), so no extra shading.
            rgb, alpha = render.render(aligned, cam, background=1.0, shading=False)
            per_view.append(metrics.image_metrics(rgb, gt_rgb, alpha > 0.5, gt_alpha > 0.5))
        out["render"] = {k: float(np.mean([v[k] for v in per_view])) for k in per_view[0]}
    return out, aligned


def _angle_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))


# ---------------------------------------------------------------------------- the loop


def load_done(path: Path, retry_failed: bool) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" or not retry_failed:
            done.add((row["object"], row["method"], row["views"]))
    return done


def resolve_subset(
    cfg: BenchConfig, out_dir: Path, list_models: Callable[[], list[str]] = gso.list_models
) -> list[str]:
    """Explicit names, else a subset saved by an earlier run, else the hashed subset of the catalogue."""
    if cfg.names:
        return list(cfg.names)
    saved = out_dir / SUBSET_FILE
    if saved.is_file():
        data = json.loads(saved.read_text())
        if data.get("salt") == cfg.salt and len(data["names"]) >= cfg.n_objects:
            return data["names"][: cfg.n_objects]
    catalogue = list_models()
    names = gso.select_subset(catalogue, cfg.n_objects, cfg.salt)
    saved.write_text(
        json.dumps(
            {"salt": cfg.salt, "catalogue_size": len(catalogue), "names": names, "license": gso.LICENSE}, indent=2
        )
        + "\n"
    )
    return names


def run(
    cfg: BenchConfig,
    backend,
    out_dir: Path,
    data_dir: Path,
    fetch: Callable[[str, Path], Path] = gso.download,
    list_models: Callable[[], list[str]] = gso.list_models,
    limit_minutes: float | None = None,
) -> Path:
    """Run (or resume) the benchmark. `backend` needs make_views() and reconstruct_views()
    (MultiViewBackend). Returns the results.jsonl path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILE).write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    results = out_dir / RESULTS_FILE
    done = load_done(results, cfg.retry_failed)
    names = resolve_subset(cfg, out_dir, list_models)
    sources = (["generated"] if cfg.generated else []) + (["oracle"] if cfg.oracle else [])
    started = time.monotonic()

    for idx, name in enumerate(names):
        todo = [(m, s) for s in sources for m in cfg.methods if (name, m, s) not in done]
        if not todo:
            continue
        if limit_minutes is not None and time.monotonic() - started > limit_minutes * 60:
            log.warning("Time budget reached; stopping before %s (rerun to resume)", name)
            break
        log.info("[%d/%d] %s", idx + 1, len(names), name)
        try:
            ctx = _prepare_object(cfg, name, data_dir, out_dir, fetch)
        except Exception as exc:
            log.exception("%s: could not prepare the ground truth", name)
            for method, src in todo:
                _append(results, _row(name, method, src, status="failed", error=f"prepare: {exc}"))
            continue

        view_sets: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
        for src in sorted({s for _, s in todo}):
            try:
                view_sets[src] = _views(cfg, backend, ctx, src, idx, out_dir / "objects" / name)
            except Exception as exc:
                log.exception("%s: %s views failed", name, src)
                for method in [m for m, s in todo if s == src]:
                    _append(results, _row(name, method, src, status="failed", error=f"views: {exc}"))

        for method, src in todo:
            if src not in view_sets:
                continue
            view_arr, mask_arr, view_stats = view_sets[src]
            _append(
                results,
                _reconstruct_and_score(cfg, backend, ctx, name, method, src, view_arr, mask_arr, view_stats, out_dir),
            )
    return results


@dataclass
class _ObjectContext:
    gt: trimesh.Trimesh
    gt_sample: metrics.SurfaceSample
    image: Image.Image
    eval_cams: list[cameras.Camera] = field(default_factory=list)
    gt_renders: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _prepare_object(cfg: BenchConfig, name: str, data_dir: Path, out_dir: Path, fetch) -> _ObjectContext:
    model_dir = fetch(name, data_dir)
    gt, _ = align.normalize_to_unit_cube(gso.load_mesh(model_dir))
    gt_sample = metrics.sample_surface(gt, cfg.n_points, seed=cfg.seed + 1)
    rgb, alpha = render.render(gt, input_camera(cfg, gt), background=1.0)
    rgba = render.to_rgba_image(rgb, alpha)
    image = segmentation.prepare(rgba, remove_bg=False, foreground_ratio=cfg.foreground_ratio)
    obj_dir = out_dir / "objects" / name
    obj_dir.mkdir(parents=True, exist_ok=True)
    image.save(obj_dir / "input.png")
    eval_cams = eval_cameras(cfg, gt) if cfg.eval_views > 0 else []
    gt_renders = [render.render(gt, cam, background=1.0) for cam in eval_cams]
    return _ObjectContext(gt, gt_sample, image, eval_cams, gt_renders)


def _views(cfg: BenchConfig, backend, ctx: _ObjectContext, src: str, idx: int, obj_dir: Path):
    if src == "oracle":
        with measure() as m:
            view_arr, mask_arr = oracle_views(ctx.gt, cfg.oracle_scale)
        stats = {"seconds": m.seconds, "peak_gpu_mib": None}
    else:
        opts = GenerateOptions(remove_background=False, diffusion_steps=cfg.diffusion_steps, seed=cfg.seed + idx)
        with measure() as m:
            _, view_arr, mask_arr = backend.make_views(ctx.image, opts)
        stats = {"seconds": m.seconds, "peak_gpu_mib": m.peak_gpu_mib}
    grid = [Image.fromarray((v * 255).astype(np.uint8)) for v in view_arr]
    multiview.make_grid(grid).save(obj_dir / f"views_{src}.png")
    return view_arr, mask_arr, stats


def _reconstruct_and_score(cfg, backend, ctx, name, method, src, view_arr, mask_arr, view_stats, out_dir) -> dict:
    opts = GenerateOptions(
        remove_background=False,
        reconstructor=method,
        recon_iters=cfg.recon_iters,
        diffusion_steps=cfg.diffusion_steps,
        texture=False,
        seed=cfg.seed,
    )
    try:
        with measure() as m:
            mesh, colors = backend.reconstruct_views(view_arr, mask_arr, opts)
        if colors is not None and len(colors) == len(mesh.vertices):
            mesh.visual.vertex_colors = colors
        t0 = time.perf_counter()
        scores, aligned = score(mesh, ctx.gt, ctx.gt_sample, cfg, ctx.eval_cams, ctx.gt_renders)
        score_s = time.perf_counter() - t0
    except Exception as exc:
        log.exception("%s / %s / %s failed", name, method, src)
        return _row(
            name,
            method,
            src,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            trace=traceback.format_exc(limit=5),
        )

    if cfg.save_meshes:
        aligned.export(out_dir / "objects" / name / f"{method}_{src}.ply")
    return _row(
        name,
        method,
        src,
        status="ok",
        faces=int(len(mesh.faces)),
        time_views_s=view_stats["seconds"],
        time_recon_s=m.seconds,
        time_total_s=view_stats["seconds"] + m.seconds,
        peak_gpu_views_mib=view_stats["peak_gpu_mib"],
        peak_gpu_recon_mib=m.peak_gpu_mib,
        time_scoring_s=score_s,
        **scores,
    )


def _row(name: str, method: str, src: str, **extra) -> dict:
    return {"object": name, "method": method, "views": src, **extra}


def _append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)}")
