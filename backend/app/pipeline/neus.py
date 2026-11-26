"""Stage 4: the neural surface model, after NeuS (Wang et al., NeurIPS 2021).

The object's surface is the zero level set of a signed distance function
f: R^3 -> R, negative inside and positive outside, represented by a neural
field. To fit f to images we need a differentiable renderer, and NeuS gives
one that is *unbiased* for surfaces:

  * Map SDF to opacity with the logistic CDF Phi_s(x) = sigmoid(s * x).
    Along a ray p(t) = o + t*d, NeuS defines the opaque density
        rho(t) = max( -(d/dt) Phi_s(f(p(t))) / Phi_s(f(p(t))), 0 )
    so that the rendering weight w(t) = T(t) * rho(t) peaks exactly where
    the ray crosses the surface (plain "SDF -> NeRF density" conversions
    peak slightly in front of it).
  * Discretised over sections [t_i, t_i+1] this gives the alpha used below:
        alpha_i = max( (Phi_s(f_i) - Phi_s(f_i+1)) / Phi_s(f_i), 0 )
  * s is learned. Its inverse 1/s is the "thickness" of the surface: it
    starts blurry so gradients reach far, and shrinks as the fit sharpens.

The field itself is a multi-resolution dense feature grid (the idea behind
Instant-NGP, without the hash so it is pure PyTorch) feeding a small MLP.
Surface normals are the SDF gradient. We take it with central finite
differences, as Neuralangelo does: with grid encodings, the numerical
gradient smooths across cells and gives better surfaces than the analytic
one, and it avoids double-backward through grid_sample (unsupported). The
Eikonal term (|grad f| = 1) keeps f a true distance function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FieldConfig:
    bound: float = 1.1  # the scene is the cube [-bound, bound]^3
    init_radius: float = 0.5  # the SDF starts as a sphere of this radius
    levels: int = 6
    base_res: int = 16
    max_res: int = 128
    features: int = 4
    hidden: int = 64
    geo_features: int = 15


class MultiResGrid(nn.Module):
    """L dense 3D feature grids at geometrically spaced resolutions, trilinearly
    interpolated and concatenated. Coarse levels capture shape, fine levels detail."""

    def __init__(self, bound: float, levels: int, base_res: int, max_res: int, features: int) -> None:
        super().__init__()
        self.bound = bound
        growth = (max_res / base_res) ** (1 / max(levels - 1, 1))
        self.resolutions = [int(round(base_res * growth**i)) for i in range(levels)]
        self.grids = nn.ParameterList(
            [nn.Parameter(torch.randn(1, features, r, r, r) * 1e-4) for r in self.resolutions]
        )
        self.features = features
        self.active_levels = levels  # coarse-to-fine schedule sets this

    @property
    def out_dim(self) -> int:
        return len(self.grids) * self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        coords = (x / self.bound).clamp(-1, 1).view(1, n, 1, 1, 3)
        feats = []
        for level, grid in enumerate(self.grids):
            f = F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=True)
            f = f.view(self.features, n).T
            if level >= self.active_levels:
                f = torch.zeros_like(f)
            feats.append(f)
        return torch.cat(feats, dim=-1)

    def finest_cell(self) -> float:
        res = self.resolutions[min(self.active_levels, len(self.resolutions)) - 1]
        return 2 * self.bound / (res - 1)


class SDFField(nn.Module):
    def __init__(self, cfg: FieldConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoding = MultiResGrid(cfg.bound, cfg.levels, cfg.base_res, cfg.max_res, cfg.features)
        self.mlp = nn.Sequential(
            nn.Linear(3 + self.encoding.out_dim, cfg.hidden),
            nn.Softplus(beta=100),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.Softplus(beta=100),
            nn.Linear(cfg.hidden, 1 + cfg.geo_features),
        )
        # Start from a sphere: the network only learns a residual on top of it
        # ("geometric initialisation", simplified).
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(torch.cat([x, self.encoding(x)], dim=-1))
        sdf = h[:, :1] + (x.norm(dim=-1, keepdim=True) - self.cfg.init_radius)
        return sdf, h[:, 1:]

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]

    def gradient(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Central finite-difference gradient of the SDF, (N, 3)."""
        offsets = torch.eye(3, device=x.device, dtype=x.dtype) * eps
        pts = torch.cat([x + o for o in offsets] + [x - o for o in offsets], dim=0)
        s = self.sdf(pts).view(6, -1)
        return torch.stack([(s[i] - s[i + 3]) / (2 * eps) for i in range(3)], dim=-1)


class ColorField(nn.Module):
    """Radiance c(x, n, v, z): position, normal, view direction, geometry feature."""

    def __init__(self, geo_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9 + geo_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, x, normals, dirs, feats) -> torch.Tensor:
        return torch.sigmoid(self.mlp(torch.cat([x, normals, dirs, feats], dim=-1)))


class NeuSModel(nn.Module):
    def __init__(self, cfg: FieldConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FieldConfig()
        self.sdf_field = SDFField(self.cfg)
        self.color_field = ColorField(self.cfg.geo_features, self.cfg.hidden)
        # s = exp(10 * variance); NeuS initialises variance to 0.3 (s ~ 20).
        self.variance = nn.Parameter(torch.tensor(0.3))

    @property
    def bound(self) -> float:
        return self.cfg.bound

    def inv_s(self) -> torch.Tensor:
        return torch.exp(self.variance * 10.0).clamp(1e-6, 1e6)

    def fd_eps(self) -> float:
        # Neuralangelo: step size tied to the finest active grid cell, so it
        # shrinks as finer levels switch on.
        return 0.5 * self.sdf_field.encoding.finest_cell()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def ray_box(rays_o: torch.Tensor, rays_d: torch.Tensor, bound: float):
    """Slab test against the cube [-bound, bound]^3. Returns near, far, hit (B,)."""
    inv = 1.0 / torch.where(rays_d.abs() < 1e-9, torch.full_like(rays_d, 1e-9), rays_d)
    t0 = (-bound - rays_o) * inv
    t1 = (bound - rays_o) * inv
    near = torch.minimum(t0, t1).amax(dim=-1).clamp(min=0.0)
    far = torch.maximum(t0, t1).amin(dim=-1)
    hit = far > near + 1e-4
    far = torch.where(hit, far, near + 1e-3)
    return near, far, hit


def _section_alpha(sdf_prev, sdf_next, inv_s):
    prev_cdf = torch.sigmoid(sdf_prev * inv_s)
    next_cdf = torch.sigmoid(sdf_next * inv_s)
    return ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)


def _weights(alpha: torch.Tensor) -> torch.Tensor:
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-7], dim=-1), dim=-1)
    return alpha * trans[:, :-1]


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse-transform sampling of n depths from the piecewise-constant pdf
    given by `weights` (B, N-1) over `bins` (B, N). Deterministic (mid-quantiles)."""
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, dim=-1)], dim=-1)
    u = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=bins.device).expand(cdf.shape[0], n).contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_b, cdf_a = cdf.gather(1, below), cdf.gather(1, above)
    bin_b, bin_a = bins.gather(1, below), bins.gather(1, above)
    denom = torch.where(cdf_a - cdf_b < 1e-5, torch.ones_like(cdf_a), cdf_a - cdf_b)
    return bin_b + (u - cdf_b) / denom * (bin_a - bin_b)


@torch.no_grad()
def _up_sample(model: NeuSModel, rays_o, rays_d, z, n_importance: int, steps: int) -> torch.Tensor:
    """NeuS hierarchical sampling: add samples where a sharpened (fixed large s)
    version of the current SDF says the surface is, so the final pass spends
    its samples near the surface."""
    b = z.shape[0]
    pts = rays_o[:, None] + rays_d[:, None] * z[..., None]
    sdf = model.sdf_field.sdf(pts.reshape(-1, 3)).view(b, -1)
    for i in range(steps):
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        dist = z[:, 1:] - z[:, :-1]
        mid_sdf = 0.5 * (prev_sdf + next_sdf)
        cos = (next_sdf - prev_sdf) / (dist + 1e-5)
        cos = torch.minimum(torch.cat([torch.zeros_like(cos[:, :1]), cos[:, :-1]], dim=-1), cos).clamp(-1e3, 0.0)
        inv_s = 64.0 * 2**i
        alpha = _section_alpha(mid_sdf - cos * dist * 0.5, mid_sdf + cos * dist * 0.5, inv_s)
        new_z = sample_pdf(z, _weights(alpha), n_importance // steps)
        new_pts = rays_o[:, None] + rays_d[:, None] * new_z[..., None]
        new_sdf = model.sdf_field.sdf(new_pts.reshape(-1, 3)).view(b, -1)
        z, order = torch.sort(torch.cat([z, new_z], dim=-1), dim=-1)
        sdf = torch.cat([sdf, new_sdf], dim=-1).gather(1, order)
    return z


@dataclass
class RenderOutput:
    rgb: torch.Tensor  # (B, 3) composited over the background
    opacity: torch.Tensor  # (B,)
    gradients: torch.Tensor  # (B*S, 3) SDF gradients at the samples, for the Eikonal loss
    inside: torch.Tensor  # (B*S,) which samples lie inside the scene cube
    depth: torch.Tensor  # (B,)


def render_rays(
    model: NeuSModel,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    background: float = 1.0,
    n_samples: int = 64,
    n_importance: int = 64,
    up_sample_steps: int = 4,
    cos_anneal_ratio: float = 1.0,
    perturb: bool = True,
) -> RenderOutput:
    b = rays_o.shape[0]
    near, far, hit = ray_box(rays_o, rays_d, model.bound)
    t = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    z = near[:, None] + (far - near)[:, None] * t[None]
    if perturb:
        step = ((far - near) / (n_samples - 1))[:, None]
        z = z + (torch.rand_like(z) - 0.5) * step
        z = torch.minimum(torch.maximum(z, near[:, None]), far[:, None]).sort(dim=-1).values
    if n_importance > 0:
        z = _up_sample(model, rays_o, rays_d, z, n_importance, up_sample_steps)

    # Evaluate at section mid-points.
    dists = z[:, 1:] - z[:, :-1]
    mid_z = z[:, :-1] + 0.5 * dists
    s = mid_z.shape[1]
    pts = (rays_o[:, None] + rays_d[:, None] * mid_z[..., None]).reshape(-1, 3)
    dirs = rays_d[:, None].expand(b, s, 3).reshape(-1, 3)

    sdf, feats = model.sdf_field(pts)
    grads = model.sdf_field.gradient(pts, model.fd_eps())
    normals = F.normalize(grads, dim=-1)
    colors = model.color_field(pts, normals, dirs, feats).view(b, s, 3)

    # NeuS alpha. The SDF at the section ends is estimated from the mid-point
    # value and the directional derivative; the "cos anneal" blends from a
    # smoothed estimate (helps early training) to the exact one.
    true_cos = (dirs * grads).sum(-1, keepdim=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio)
    d = dists.reshape(-1, 1)
    # iter_cos <= 0 when entering a surface, so prev > next and alpha > 0.
    alpha = _section_alpha(sdf - iter_cos * d * 0.5, sdf + iter_cos * d * 0.5, model.inv_s())
    alpha = alpha.view(b, s) * hit[:, None]
    weights = _weights(alpha)

    opacity = weights.sum(dim=-1)
    rgb = (weights[..., None] * colors).sum(dim=1) + (1.0 - opacity[:, None]) * background
    depth = (weights * mid_z).sum(dim=-1)
    inside = (pts.abs() <= model.bound).all(dim=-1)
    return RenderOutput(rgb=rgb, opacity=opacity, gradients=grads, inside=inside, depth=depth)


def eikonal_loss(gradients: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (gradients.norm(dim=-1) - 1.0) ** 2
    if mask is not None and mask.any():
        err = err[mask]
    return err.mean()


def cosine_lr(step: int, total: int, warmup: int = 100, floor: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
