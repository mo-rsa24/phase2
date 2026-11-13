"""Ada-GVAE loss (Locatello et al., ICML 2020).

Implements the weakly-supervised disentanglement objective for paired
observations (x_a, x_b) that differ in exactly k generative factors.

The key idea: compare q(z|x_a) and q(z|x_b) per latent dimension.
Dimensions where the two posteriors are similar → the underlying factor
did not change ("shared").  Dimensions where they diverge → the changed
factor is encoded there.

Two variants are supported:

  Ada-GVAE   (adaptive, n_changed=None):
      Infer the changed set C per sample as the dims whose symmetric
      KL exceeds the per-sample median.  No hyperparameter.

  Ada-GVAE-k (fixed, n_changed=k):
      Treat exactly the top-k dims (by symmetric KL) as changed.
      Use n_changed=1 when paired data has exactly one varying factor.

KL applied per dimension:
    k ∈ shared  →  KL( q(z_k|x_a) ‖ q(z_k|x_b) )
                   drives the two posteriors to agree on shared dims
    k ∈ changed →  KL( q(z_k|x_a) ‖ N(0,1) )
                 + KL( q(z_k|x_b) ‖ N(0,1) )
                   regularises the changed dims to the prior independently

Reference:
    Locatello et al., "Weakly-Supervised Disentanglement Without
    Compromises", ICML 2020.  https://arxiv.org/abs/2002.02886
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gaussian KL primitives  (all per-dimension, return shape (B, D))
# ---------------------------------------------------------------------------

def _kl_to_prior(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL( N(μ, σ²) ‖ N(0, 1) ) per dimension."""
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())


def _kl_between(
    mu_p: Tensor, lv_p: Tensor,
    mu_q: Tensor, lv_q: Tensor,
) -> Tensor:
    """KL( N(μ_p, σ_p²) ‖ N(μ_q, σ_q²) ) per dimension."""
    return 0.5 * (
        lv_q - lv_p
        + (lv_p.exp() + (mu_p - mu_q).pow(2)) / lv_q.exp()
        - 1.0
    )


def _symmetric_kl(
    mu_a: Tensor, lv_a: Tensor,
    mu_b: Tensor, lv_b: Tensor,
) -> Tensor:
    """Symmetric per-dim KL: 0.5 · (KL(a‖b) + KL(b‖a)). Shape (B, D)."""
    return 0.5 * (
        _kl_between(mu_a, lv_a, mu_b, lv_b)
        + _kl_between(mu_b, lv_b, mu_a, lv_a)
    )


# ---------------------------------------------------------------------------
# Shared-dimension inference
# ---------------------------------------------------------------------------

def _infer_shared(d_sym: Tensor, n_changed: int | None) -> Tensor:
    """Return a (B, D) bool mask: True = shared dimension.

    Ada-GVAE   (n_changed=None): threshold = per-sample median of d_sym.
    Ada-GVAE-k (n_changed=k):   mark only the top-k dims as changed.
    """
    if n_changed is None:
        threshold = d_sym.median(dim=1, keepdim=True).values
        return d_sym <= threshold
    # top-k dims (highest pairwise KL) → changed; rest → shared
    _, changed_idx = d_sym.topk(n_changed, dim=1)          # (B, k)
    is_shared = torch.ones_like(d_sym, dtype=torch.bool)
    is_shared.scatter_(1, changed_idx, False)
    return is_shared


# ---------------------------------------------------------------------------
# Public loss function
# ---------------------------------------------------------------------------

def adagvae_loss(
    x_a: Tensor,
    x_hat_a: Tensor,
    mu_a: Tensor,
    log_var_a: Tensor,
    x_b: Tensor,
    x_hat_b: Tensor,
    mu_b: Tensor,
    log_var_b: Tensor,
    beta: float = 1.0,
    n_changed: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Ada-GVAE objective for a batch of paired observations.

    Args:
        x_a, x_hat_a   : original and reconstruction for sample a  (B, *)
        mu_a, log_var_a : encoder output for x_a  (B, D)
        x_b, x_hat_b   : original and reconstruction for sample b  (B, *)
        mu_b, log_var_b : encoder output for x_b  (B, D)
        beta            : KL weight  (default 1.0)
        n_changed       : None → adaptive median threshold (Ada-GVAE);
                          int → fix exactly n_changed dims as changed
                          (Ada-GVAE-k; use 1 when pairs differ in 1 factor)

    Returns:
        loss    : scalar Tensor
        metrics : dict of float scalars for logging
    """
    B = x_a.shape[0]

    # --- Reconstruction ---------------------------------------------------
    recon_a = F.mse_loss(x_hat_a, x_a, reduction="sum") / B
    recon_b = F.mse_loss(x_hat_b, x_b, reduction="sum") / B

    # --- Adaptive shared/changed inference --------------------------------
    d_sym     = _symmetric_kl(mu_a, log_var_a, mu_b, log_var_b)   # (B, D)
    is_shared = _infer_shared(d_sym, n_changed)                    # (B, D)

    # --- Selective KL -----------------------------------------------------
    # shared  → inter-posterior: drives q(z_k|x_a) ≈ q(z_k|x_b)
    # changed → both posteriors regularised to N(0,1) independently
    kl_inter  = _kl_between(mu_a, log_var_a, mu_b, log_var_b)     # (B, D)
    kl_prior  = _kl_to_prior(mu_a, log_var_a) + _kl_to_prior(mu_b, log_var_b)  # (B, D)

    kl = torch.where(is_shared, kl_inter, kl_prior)               # (B, D)

    kl_mean = kl.sum(dim=1).mean()
    loss    = recon_a + recon_b + beta * kl_mean

    metrics: dict[str, float] = {
        "loss":          loss.item(),
        "recon_a":       recon_a.item(),
        "recon_b":       recon_b.item(),
        "kl":            kl_mean.item(),
        "n_shared_dims": is_shared.float().mean().item(),
    }
    return loss, metrics
