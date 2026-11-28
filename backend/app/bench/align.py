"""Putting the reconstruction and the ground truth in the same frame.

A single photo does not fix an object's absolute size or distance, so a
reconstruction from one image is only defined up to a similarity transform
(rotation, translation, uniform scale). Comparing meshes without removing
that ambiguity would mostly measure how well the scale was guessed. The
protocol, applied identically to every method:

1. Known orientation. The input view is rendered from a camera on the ground
   truth's +x axis (azimuth 0), and the pipeline reconstructs in a frame
   where the input camera sits on +x, z up (cameras.py). So the two frames
   already agree on orientation, up to how well Zero123++ guesses the input's
   elevation.
2. Normalisation. Each mesh is translated so its bounding-box centre is at the
   origin and scaled so its largest half-extent is 1: it fits [-1, 1]^3 and
   touches the cube. This fixes scale and translation from the shapes alone.
3. Refinement. Similarity ICP (Besl and McKay, 1992, with Umeyama's 1991
   closed-form least-squares similarity in each step) moves the prediction
   onto the ground truth. Correspondences go both ways (each predicted point
   to its nearest ground-truth point and vice versa), so shrinking the
   prediction into the ground truth is not rewarded; the worst 10% of pairs
   are dropped each step so a floater cannot drag the fit. ICP only finds a
   local optimum, so it starts from a few small rotations around the
   identity (yaw and pitch, +-15 degrees) and keeps the best fit.

The ground truth is never moved after step 2, so every method is measured in
exactly the same frame. Step 3 can only improve a method's score, and it does
so for all methods alike; the "unaligned" scores (after step 2 only) are kept
too, see the report.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class Similarity:
    """x -> scale * R @ x + t"""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    scale: float

    @classmethod
    def identity(cls) -> Similarity:
        return cls(np.eye(3), np.zeros(3), 1.0)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation

    def then(self, other: Similarity) -> Similarity:
        """The transform that applies self first, then other."""
        return Similarity(
            other.rotation @ self.rotation,
            other.scale * other.rotation @ self.translation + other.translation,
            other.scale * self.scale,
        )

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m


def normalize_to_unit_cube(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, Similarity]:
    """Centre the bounding box on the origin and scale the largest half-extent to 1."""
    lo, hi = mesh.bounds
    half = float((hi - lo).max()) / 2
    if half <= 0:
        raise ValueError("Degenerate mesh: zero extent")
    tf = Similarity(np.eye(3), -(lo + hi) / 2 / half, 1.0 / half)
    out = mesh.copy()
    out.vertices = tf.apply(np.asarray(mesh.vertices, dtype=np.float64))
    return out, tf


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """Least-squares similarity mapping src onto dst (Umeyama, 1991). (N, 3) each."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1  # a reflection would fit better; force a proper rotation
    rot = u @ s @ vt
    var_s = (xs**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ s) / var_s) if with_scale and var_s > 0 else 1.0
    return Similarity(rot, mu_d - scale * rot @ mu_s, scale)


def icp(
    src: np.ndarray,
    dst: np.ndarray,
    iterations: int = 40,
    trim: float = 0.1,
    with_scale: bool = True,
    init: Similarity | None = None,
    tol: float = 1e-7,
) -> tuple[Similarity, float]:
    """Similarity ICP with two-way nearest-neighbour pairs. Returns (transform, final Chamfer)."""
    tf = init or Similarity.identity()
    dst_tree = cKDTree(dst)
    prev = math.inf
    err = math.inf
    for _ in range(iterations):
        cur = tf.apply(src)
        d_fwd, i_fwd = dst_tree.query(cur, workers=-1)
        d_bwd, i_bwd = cKDTree(cur).query(dst, workers=-1)
        err = 0.5 * (d_fwd.mean() + d_bwd.mean())
        if prev - err < tol:
            break
        prev = err
        a = np.concatenate([cur, cur[i_bwd]])
        b = np.concatenate([dst[i_fwd], dst])
        d = np.concatenate([d_fwd, d_bwd])
        if trim > 0:
            keep = d <= np.quantile(d, 1 - trim)
            a, b = a[keep], b[keep]
        tf = tf.then(umeyama(a, b, with_scale=with_scale))
    return tf, float(err)


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation about z (yaw), then about y (pitch: tilts the input camera's view up or down)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    return ry @ rz


def align(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    yaw_starts: tuple[float, ...] = (-15.0, 0.0, 15.0),
    pitch_starts: tuple[float, ...] = (-15.0, 0.0, 15.0),
    iterations: int = 40,
    max_points: int = 8000,
    seed: int = 0,
) -> Similarity:
    """Best similarity ICP over a few starting rotations, fitted on a subsample of points.
    Both point sets must already be normalised (normalize_to_unit_cube)."""
    rng = np.random.default_rng(seed)

    def sub(p: np.ndarray) -> np.ndarray:
        return p if len(p) <= max_points else p[rng.choice(len(p), max_points, replace=False)]

    src, dst = sub(pred_points), sub(gt_points)
    best: tuple[float, Similarity] | None = None
    for yaw, pitch in itertools.product(yaw_starts, pitch_starts):
        init = Similarity(_rotation(yaw, pitch), np.zeros(3), 1.0)
        tf, err = icp(src, dst, iterations=iterations, init=init)
        if best is None or err < best[0]:
            best = (err, tf)
    assert best is not None
    return best[1]
