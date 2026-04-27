"""Disentanglement metrics: KL activity, Spearman correlation, MIG."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import entropy as scipy_entropy

from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES


def kl_per_dim(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """Mean KL contribution per latent dimension.

    Args:
        mu:     (N, latent_dim)
        logvar: (N, latent_dim)

    Returns:
        (latent_dim,) array of mean KL values (nats).
    """
    lv = np.clip(logvar, -10, 10)
    kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv))
    return kl.mean(axis=0)


def factor_latent_correlation(
    mu: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Absolute Spearman correlation between latent dims and ground-truth factors.

    Args:
        mu:      (N, latent_dim) — posterior means
        factors: (N, 6)         — integer class indices per dSprites factor

    Returns:
        (latent_dim, 6) array of |ρ| values; 0 for constant factors (color).
    """
    latent_dim = mu.shape[1]
    n_factors  = len(FACTOR_NAMES)
    corr       = np.zeros((latent_dim, n_factors))
    for i in range(latent_dim):
        for j in range(n_factors):
            if FACTOR_SIZES[j] <= 1:
                continue
            r, _ = spearmanr(mu[:, i], factors[:, j])
            corr[i, j] = float(abs(r)) if not np.isnan(r) else 0.0
    return corr


def compute_mig(
    mu: np.ndarray,
    factors: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Mutual Information Gap (Chen et al. 2018).

    Uses sklearn's k-NN MI estimator for continuous latents vs discrete factors.

    Args:
        mu:      (N, latent_dim) — posterior means
        factors: (N, 6)         — integer class indices

    Returns:
        (mig_score, per_factor_dict)  where mig_score ∈ [0, 1].
    """
    from sklearn.feature_selection import mutual_info_classif

    per_factor: dict[str, float] = {}
    mig_vals: list[float] = []

    for j, name in enumerate(FACTOR_NAMES):
        if FACTOR_SIZES[j] <= 1:
            continue
        factor_vals = factors[:, j].astype(int)

        # H(v_k) for normalisation
        counts = np.bincount(factor_vals, minlength=FACTOR_SIZES[j])
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        h      = float(scipy_entropy(probs, base=np.e))
        if h < 1e-8:
            continue

        # MI(z_i, v_k) for all latent dims in one vectorised call
        mi         = mutual_info_classif(mu, factor_vals, discrete_features=False, random_state=0)
        sorted_mi  = np.sort(mi)[::-1]
        gap        = float(sorted_mi[0] - sorted_mi[1]) / h if len(sorted_mi) > 1 else float(sorted_mi[0]) / h

        per_factor[name] = max(float(gap), 0.0)
        mig_vals.append(per_factor[name])

    mig_score = float(np.mean(mig_vals)) if mig_vals else 0.0
    return mig_score, per_factor
