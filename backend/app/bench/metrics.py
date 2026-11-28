"""Geometry metrics between a reconstructed mesh and the ground truth.

Both meshes are compared as point sets sampled uniformly on their surfaces
(area-weighted, so a densely triangulated region does not count more than a
coarse one). With P sampled on the prediction and G on the ground truth, and
d(x, S) = min_{y in S} ||x - y|| the distance to the nearest sample:

* accuracy      = mean_{p in P} d(p, G)   (is what we built really there?)
* completeness  = mean_{g in G} d(g, P)   (did we build everything that is there?)
* Chamfer (CD)  = (accuracy + completeness) / 2, unsquared L2 distances ("L1
  Chamfer"). Squared Chamfer is also reported because some papers use it; it
  punishes a few far-off points much more.
* F-score@tau   = harmonic mean of precision (share of P within tau of G) and
  recall (share of G within tau of P) (Knapitsch et al., 2017, Tanks and
  Temples). Unlike Chamfer it is bounded in [0, 1] and is not dominated by
  outliers, which makes it easier to read: "F@0.05 = 0.8" means 80% of the
  surface is right to within 5% of the object's half-size.
* normal consistency = mean over both directions of |n_p . n_nearest|
  (Mescheder et al., 2019, Occupancy Networks): 1 when the surface orientation
  matches everywhere, ~0.5 for random normals. It sees bumps and flat-shaded
  stair-steps that distance metrics barely notice. The absolute value makes it
  ignore inverted faces.

All distances are in the normalised frame of `align.py` (the ground truth fits
the cube [-1, 1]^3), so thresholds are fractions of the object's half-size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# F-score thresholds, as fractions of the ground truth's half-size. 0.2 matches the
# setting InstantMesh reports on GSO (objects in [-1, 1]); 0.05 and 0.1 are stricter.
DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2)


@dataclass
class SurfaceSample:
    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3) unit


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> SurfaceSample:
    """Area-weighted uniform samples with the normal of the face each one lies on."""
    if len(mesh.faces) == 0:
        raise ValueError("Cannot sample an empty mesh")
    points, face_idx = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals)[face_idx]
    return SurfaceSample(np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64))


def geometry_metrics(
    pred: SurfaceSample,
    gt: SurfaceSample,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    d_pred, i_pred = cKDTree(gt.points).query(pred.points, workers=-1)  # P -> G
    d_gt, i_gt = cKDTree(pred.points).query(gt.points, workers=-1)  # G -> P

    out: dict[str, float] = {
        "accuracy": float(d_pred.mean()),
        "completeness": float(d_gt.mean()),
        "chamfer": float(0.5 * (d_pred.mean() + d_gt.mean())),
        "chamfer_sq": float(0.5 * ((d_pred**2).mean() + (d_gt**2).mean())),
    }
    for tau in thresholds:
        precision = float((d_pred < tau).mean())
        recall = float((d_gt < tau).mean())
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[f"fscore@{tau:g}"] = f
        out[f"precision@{tau:g}"] = precision
        out[f"recall@{tau:g}"] = recall

    nc_pred = np.abs((pred.normals * gt.normals[i_pred]).sum(-1)).mean()
    nc_gt = np.abs((gt.normals * pred.normals[i_gt]).sum(-1)).mean()
    out["normal_consistency"] = float(0.5 * (nc_pred + nc_gt))
    return out


def image_metrics(pred: np.ndarray, gt: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Appearance and silhouette agreement of two renders from the same camera.

    pred, gt: (H, W, 3) in [0, 1] composited on the same background;
    masks: (H, W) bool. PSNR is over the whole image (the usual convention for
    novel-view synthesis); mask IoU compares silhouettes only.
    """
    mse = float(((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = 1.0 if union == 0 else float(np.logical_and(pred_mask, gt_mask).sum() / union)
    return {"psnr": psnr, "mask_iou": iou}
