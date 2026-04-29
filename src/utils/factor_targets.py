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
