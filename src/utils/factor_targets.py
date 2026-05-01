"""Targets for weakly-supervised disentanglement on dSprites.

Used by `scripts/train_vae.py` when `--supervise-target-factors` is set.
The latent-index convention assumed by the supervision is:
    z[Z_SCALE_IDX]      = scale         (regressed to a normalized scalar)
    z[Z_ORIENT_IDX]     = orientation   (regressed to (sin(k*theta), cos(k*theta)))
"""

from __future__ import annotations

import math

import torch

# Latent index convention.
Z_SCALE_IDX: int = 0
Z_ORIENT_IDX: tuple[int, int] = (1, 2)

# Rotational symmetry order per dSprites shape index.
# shape: 0 = square, 1 = ellipse, 2 = heart.
# Square has 4-fold rotational symmetry, ellipse 2-fold, heart 1.
_SYMMETRY_ORDER_LIST: list[int] = [4, 2, 1]

# dSprites factor structure (see src/datasets/dsprites.py:FACTOR_SIZES).
_NUM_SCALES: int = 6   # scale indices 0..5
_NUM_ORIENTS: int = 40  # orient indices 0..39, full circle


def _symmetry_table(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(_SYMMETRY_ORDER_LIST, device=device, dtype=dtype)


def scale_target(scale_idx: torch.Tensor) -> torch.Tensor:
    """Map scale class index 0..5 to a roughly unit-variance scalar.

    Returns a 1-D tensor of shape (B,) with values in approximately [-1, 1].
    """
    return (scale_idx.to(torch.float32) - (_NUM_SCALES - 1) / 2.0) / ((_NUM_SCALES - 1) / 2.0)


def orient_target(orient_idx: torch.Tensor, shape_idx: torch.Tensor) -> torch.Tensor:
    """Symmetry-aware orientation target on the unit circle.

    Returns (B, 2) tensor of (sin(k*theta), cos(k*theta)) where k is the rotational
    symmetry order of the shape. theta = 2*pi * orient_idx / 40.

    Mapping the angle through k folds the fundamental domain of the rotational
    symmetry group onto a single revolution of the circle, removing the
    many-to-one collision that otherwise makes orientation unidentifiable for
    the square (k=4) and ellipse (k=2).
    """
    theta = (2.0 * math.pi / _NUM_ORIENTS) * orient_idx.to(torch.float32)
    k = _symmetry_table(orient_idx.device)[shape_idx.to(torch.long)]
    angle = k * theta
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


# ---------------------------------------------------------------------------
# Eval-time metrics (used by the supervised trainer's val loop).
# ---------------------------------------------------------------------------

def scale_r2(mu_scale: torch.Tensor, scale_idx: torch.Tensor) -> float:
    """Coefficient of determination R^2 between mu[:,0] and the normalized scale target.

    Robust to a constant shift, scale, and the (rare) zero-variance batch.
    Returned as a python float, computed in fp32.
    """
    pred   = mu_scale.detach().float().reshape(-1)
    target = scale_target(scale_idx).reshape(-1)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    if float(ss_tot) <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def orient_angular_error_deg(
    mu_orient: torch.Tensor,
    orient_idx: torch.Tensor,
    shape_idx:  torch.Tensor,
) -> float:
    """Mean physical-orientation error in degrees, modulo each shape's symmetry.

    Decoder picks an angle from `mu_orient` ≈ (sin(k·θ), cos(k·θ)) by
    `k·θ̂ = atan2(z2, z1)`. The error in *physical* orientation is therefore
    `(k·θ̂ − k·θ_true) / k`, wrapped to (−π, π] and divided by k. Reported in
    degrees so values are scale-free and easy to read against a thresholds
    bar (≤10° strong, 10–30° moderate, ≥30° no signal).
    """
    mu = mu_orient.detach().float()                       # (B, 2)
    pred_angle   = torch.atan2(mu[:, 0], mu[:, 1])        # (B,) in (−π, π]
    theta_true   = (2.0 * math.pi / _NUM_ORIENTS) * orient_idx.to(torch.float32)
    k = _symmetry_table(orient_idx.device)[shape_idx.to(torch.long)]
    target_angle = k * theta_true
    diff = pred_angle - target_angle
    # Wrap to (−π, π] then divide by k to recover physical-angle error.
    wrapped = (diff + math.pi) % (2.0 * math.pi) - math.pi
    err_rad = wrapped.abs() / k
    return float(err_rad.mean() * (180.0 / math.pi))
