"""Stage 5: fit the NeuS field to the generated views.

Per-object optimisation ("test-time training"): starting from a sphere, we
render random batches of pixels from the six known cameras and minimise

    L = L_rgb + w_mask * L_mask + w_eik * L_eikonal

  * L_rgb: L1 between rendered and generated colour (views composited on white
    using their masks, so the background carries no geometry).
  * L_mask: binary cross-entropy between rendered opacity and the silhouette.
    With only six views, silhouettes are the strongest geometric constraint
    (this is a soft, differentiable version of visual-hull carving).
  * L_eikonal: (|grad f| - 1)^2 at ray samples and random points, which keeps
    f a signed distance function, so its zero set is a clean surface.

Two schedules help with sparse, slightly inconsistent views:
  * coarse-to-fine (Neuralangelo): fine grid levels are switched on gradually,
    so large-scale shape settles before detail;
  * cos-annealing (NeuS): the alpha estimate goes from smoothed to exact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .cameras import Camera
from .neus import FieldConfig, NeuSModel, cosine_lr, eikonal_loss, render_rays

log = logging.getLogger(__name__)


@dataclass
class ReconConfig:
    # GUESS: tuned by reasoning, not on a GPU yet. On an A10G we expect ~1-3 min.
    iters: int = 2500
    batch_rays: int = 1024
    n_samples: int = 64
    n_importance: int = 64
    up_sample_steps: int = 4
    lr_grid: float = 1e-2
    lr_net: float = 1e-3
    mask_weight: float = 0.5
    eikonal_weight: float = 0.1
    eikonal_random_points: int = 1024
    anneal_end: float = 0.2  # fraction of iters for cos annealing
    coarse_to_fine_end: float = 0.5  # fraction of iters by which all grid levels are on
    background: float = 1.0
    log_every: int = 100
    field: FieldConfig = field(default_factory=FieldConfig)


def build_rays(cameras: list[Camera], views: np.ndarray, masks: np.ndarray, background: float, device):
    """Flatten every pixel of every view into one table of rays and targets."""
    origins, dirs = zip(*(cam.rays(device) for cam in cameras))
    rays_o = torch.cat(origins)
    rays_d = torch.cat(dirs)
    rgb = torch.from_numpy(views).float().reshape(-1, 3).to(device)
    alpha = torch.from_numpy(masks).float().reshape(-1, 1).to(device)
    target = rgb * alpha + background * (1.0 - alpha)
    return rays_o, rays_d, target, alpha.squeeze(-1)


def reconstruct(
    views: np.ndarray,
    masks: np.ndarray,
    cameras: list[Camera],
    cfg: ReconConfig | None = None,
    device: str = "cuda",
    progress: Callable[[float], None] | None = None,
    seed: int = 0,
) -> NeuSModel:
    """views: (V, H, W, 3) in [0,1]; masks: (V, H, W) in [0,1]; one camera per view."""
    cfg = cfg or ReconConfig()
    torch.manual_seed(seed)
    assert len(views) == len(masks) == len(cameras)
    rays_o, rays_d, target, target_mask = build_rays(cameras, views, masks, cfg.background, device)
    n_rays = rays_o.shape[0]

    model = NeuSModel(cfg.field).to(device)
    grid_params = list(model.sdf_field.encoding.parameters())
    grid_ids = {id(p) for p in grid_params}
    other = [p for p in model.parameters() if id(p) not in grid_ids]
    opt = torch.optim.Adam(
        [{"params": grid_params, "lr": cfg.lr_grid}, {"params": other, "lr": cfg.lr_net}],
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    encoding = model.sdf_field.encoding
    n_levels = len(encoding.resolutions)

    started = time.monotonic()
    for step in range(cfg.iters):
        frac = step / max(cfg.iters - 1, 1)
        # coarse-to-fine: start with 2 levels, unlock the rest linearly
        encoding.active_levels = min(
            n_levels, 2 + int((n_levels - 2) * min(frac / max(cfg.coarse_to_fine_end, 1e-6), 1.0))
        )
        cos_anneal = min(frac / max(cfg.anneal_end, 1e-6), 1.0)
        scale = cosine_lr(step, cfg.iters)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale

        idx = torch.randint(0, n_rays, (cfg.batch_rays,), device=device)
        out = render_rays(
            model,
            rays_o[idx],
            rays_d[idx],
            background=cfg.background,
            n_samples=cfg.n_samples,
            n_importance=cfg.n_importance,
            up_sample_steps=cfg.up_sample_steps,
            cos_anneal_ratio=cos_anneal,
        )
        loss_rgb = (out.rgb - target[idx]).abs().mean()
        opacity = out.opacity.clamp(1e-3, 1.0 - 1e-3)
        loss_mask = F.binary_cross_entropy(opacity, target_mask[idx])

        rand_pts = (torch.rand(cfg.eikonal_random_points, 3, device=device) * 2 - 1) * model.bound
        rand_grads = model.sdf_field.gradient(rand_pts, model.fd_eps())
        loss_eik = eikonal_loss(torch.cat([out.gradients[out.inside], rand_grads]))

        loss = loss_rgb + cfg.mask_weight * loss_mask + cfg.eikonal_weight * loss_eik
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.iters - 1:
            psnr = -10 * torch.log10(((out.rgb - target[idx]) ** 2).mean().clamp(min=1e-10))
            log.info(
                "recon %d/%d loss=%.4f rgb=%.4f mask=%.4f eik=%.4f psnr=%.2f s=%.1f levels=%d (%.0fs)",
                step, cfg.iters, loss.item(), loss_rgb.item(), loss_mask.item(), loss_eik.item(),
                psnr.item(), model.inv_s().item(), encoding.active_levels, time.monotonic() - started,
            )
            if progress:
                progress(frac)
    model.eval()
    return model
