"""Product-of-Experts (PoE) posterior fusion for Gaussian experts.

All functions are pure numpy — no side effects, no I/O.
"""

from __future__ import annotations

import numpy as np


def compute_poe(
    lambda1: np.ndarray,
    mu1: np.ndarray,
    lambda2: np.ndarray,
    mu2: np.ndarray,
    lambda_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse two Gaussian experts via the Product-of-Experts rule.

    Λ_PoE = Λ₁ + Λ₂ − Λ_prior
    μ_PoE = Λ_PoE⁻¹ · (Λ₁μ₁ + Λ₂μ₂)

    Parameters
    ----------
    lambda1, lambda2 : (d, d) ndarray
        Precision matrices of the two experts.
    mu1, mu2 : (d,) ndarray
        Mean vectors of the two experts.
    lambda_prior : (d, d) ndarray
        Prior precision matrix (typically np.eye(d) for a standard normal).

    Returns
    -------
    lambda_poe : (d, d) ndarray
        Fused precision matrix.
    mu_poe : (d,) ndarray
        Fused mean vector.
    """
    lambda_poe = lambda1 + lambda2 - lambda_prior
    mu_poe = np.linalg.solve(lambda_poe, lambda1 @ mu1 + lambda2 @ mu2)
    return lambda_poe, mu_poe


def poe_error(mu_poe: np.ndarray, alpha: float, beta: float) -> float:
    """Squared distance from the PoE mean to the target factor values.

    error = |μ_PoE[0] − α|² + |μ_PoE[1] − β|²

    Parameters
    ----------
    mu_poe : (2,) ndarray
        PoE posterior mean.
    alpha : float
        Target value for the first factor (ĉ₁).
    beta : float
        Target value for the second factor (ĉ₂).

    Returns
    -------
    float
        Squared reconstruction error in factor space.
    """
    return float((mu_poe[0] - alpha) ** 2 + (mu_poe[1] - beta) ** 2)


def encode_reference_set(
    model,
    data_x: np.ndarray,
    data_c: np.ndarray,
    factor_id: int,
    factor_val: float,
    n: int = 200,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean posterior parameters over a reference set.

    Selects up to `n` samples where data_c[:, factor_id] == factor_val
    (within floating-point tolerance), encodes them, and returns the mean
    posterior mean and mean diagonal precision.

    Parameters
    ----------
    model : LinearVAE
        Trained VAE; encode(x) returns (mu, log_var).
    data_x : (N, 2) ndarray
        Full data matrix.
    data_c : (N, K) ndarray
        Factor labels; column `factor_id` is used for selection.
    factor_id : int
        Which factor column to condition on.
    factor_val : float
        Target factor value to select.
    n : int
        Maximum number of reference samples to average over.
    device : str
        PyTorch device string.

    Returns
    -------
    mu_bar : (latent_dim,) ndarray
        Mean of posterior means over the reference set.
    lambda_bar : (latent_dim,) ndarray
        Mean diagonal precision (1 / exp(log_var)) over the reference set.
    """
    import torch
    mask = np.abs(data_c[:, factor_id] - factor_val) < 0.15
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No samples with factor_id={factor_id} ≈ {factor_val:.3f} (atol=0.15). "
            "Pass factor_val within [−0.85, 0.85] or generate more data."
        )
    if len(idx) > n:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(idx, size=n, replace=False)
    x_sub = torch.tensor(data_x[idx], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(x_sub)
    mu_bar     = mu.cpu().numpy().mean(axis=0)
    lambda_bar = np.exp(-log_var.cpu().numpy()).mean(axis=0)
    return mu_bar, lambda_bar
