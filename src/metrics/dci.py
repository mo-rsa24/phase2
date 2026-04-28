"""DCI metric — Disentanglement / Completeness / Informativeness.

Eastwood & Williams (2018), "A Framework for the Quantitative Evaluation of
Disentangled Representations", ICLR.

Procedure
---------
1. For each ground-truth factor v_k, train a regressor (Random Forest by
   default) to predict v_k from the latent representation μ.
2. Read off feature importances R[i, k] = importance of latent dim i for
   predicting factor k.
3. Compute three scores from R:

   - **Disentanglement (D)**, per latent dim i:
       P_i  = R[i, :] / R[i, :].sum()
       D_i  = 1 - H(P_i) / log(K)
     A single-factor latent → P_i is one-hot → H = 0 → D = 1.
     A latent that splits its importance evenly across all K factors →
     H = log K → D = 0.

   - **Completeness (C)**, per factor k:
       P_k = R[:, k] / R[:, k].sum()
       C_k = 1 - H(P_k) / log(d)
     A factor concentrated in one latent dim → C = 1.

   - **Informativeness (I)**, per factor k:
       I_k = max(0, R^2_k)
     Held-out R² of the regressor that predicts v_k from μ.

4. Aggregate:
       D_overall = importance-weighted mean of D_i over latent dims
       C_overall = mean of C_k
       I_overall = mean of I_k

Notes
-----
- Constant factors (FACTOR_SIZES <= 1, e.g. dSprites' `color`) are dropped
  before fitting; this matches the convention used by `compute_mig` and
  `factor_latent_correlation` in the same module.
- Orientation is cyclic. Treating it as a real-valued regression target
  imports the same caveat as |Spearman ρ| — the metric will underestimate
  cyclic-encoded factors. We surface the score regardless; the explorer's
  conditional-histogram modal is the right cross-check.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def _entropy(p: np.ndarray, *, eps: float = 1e-12) -> float:
    """Shannon entropy in nats (natural log)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def compute_dci(
    mu: np.ndarray,
    factors: np.ndarray,
    factor_names: Sequence[str],
    factor_sizes: Sequence[int],
    *,
    test_frac: float = 0.2,
    n_estimators: int = 50,
    max_depth: int | None = None,
    seed: int = 0,
) -> dict:
    """Compute DCI scores and the underlying importance matrix.

    Parameters
    ----------
    mu : (N, d) array
        Latent representations (typically encoder posterior means).
    factors : (N, K_total) array
        Integer factor classes for each sample.
    factor_names : sequence of str, length K_total
        Names matching `factors` columns.
    factor_sizes : sequence of int, length K_total
        Number of distinct values per factor; columns with size <= 1
        are skipped.
    test_frac : float
        Held-out fraction for the informativeness R² estimate.
    n_estimators, max_depth :
        Random-forest hyperparameters.
    seed : int
        Reproducibility seed for split + RF.

    Returns
    -------
    dict with keys:
      'importance'   : (d, K_active) row-and-column-described importance
      'D_per_latent' : (d,)       per-latent disentanglement
      'D'            : float      importance-weighted overall D
      'C_per_factor' : (K_active,) per-factor completeness
      'C'            : float
      'I_per_factor' : (K_active,) per-factor informativeness (R²)
      'I'            : float
      'factor_names' : list[str]  active factor names (in column order)
    """
    mu      = np.asarray(mu, dtype=np.float64)
    factors = np.asarray(factors)
    if mu.ndim != 2 or factors.ndim != 2 or mu.shape[0] != factors.shape[0]:
        raise ValueError(
            f"shape mismatch: mu {mu.shape}, factors {factors.shape}"
        )

    # Filter out constant factors.
    active = [(name, idx) for idx, (name, size)
              in enumerate(zip(factor_names, factor_sizes)) if size > 1]
    if not active:
        raise ValueError("No active factors (all factor_sizes <= 1).")
    active_names = [n for n, _ in active]
    active_idx   = [i for _, i in active]
    factors_active = factors[:, active_idx]

    n_samples, d_latent = mu.shape
    K = len(active_idx)

    # Train one RF per factor; collect feature importances + R² on held-out.
    importance = np.zeros((d_latent, K), dtype=np.float64)
    informativeness = np.zeros(K, dtype=np.float64)

    for k, name in enumerate(active_names):
        y = factors_active[:, k].astype(np.float64)
        X_tr, X_te, y_tr, y_te = train_test_split(
            mu, y, test_size=test_frac, random_state=seed,
        )
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(X_tr, y_tr)
        importance[:, k] = rf.feature_importances_
        # R² on the held-out split. Cap at 0 — a worse-than-mean predictor
        # carries no information about the factor.
        r2 = rf.score(X_te, y_te)
        informativeness[k] = max(0.0, float(r2))

    # ---------------- Disentanglement (per latent) ----------------
    log_K = np.log(K) if K > 1 else 1.0  # divisor; if K=1 there is nothing to disentangle
    row_sums = importance.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero for completely-uninformative latent dims.
    row_safe = np.where(row_sums > 0, row_sums, 1.0)
    P_rows   = importance / row_safe
    D_per_latent = np.array([
        (1.0 - _entropy(P_rows[i]) / log_K) if row_sums[i, 0] > 0 else 0.0
        for i in range(d_latent)
    ])
    # Importance-weighted overall D — uninformative dims don't dilute the score.
    total_importance = importance.sum()
    if total_importance > 0:
        weights = row_sums.flatten() / total_importance
        D_overall = float((weights * D_per_latent).sum())
    else:
        D_overall = 0.0

    # ---------------- Completeness (per factor) -------------------
    log_d = np.log(d_latent) if d_latent > 1 else 1.0
    col_sums = importance.sum(axis=0, keepdims=True)
    col_safe = np.where(col_sums > 0, col_sums, 1.0)
    P_cols   = importance / col_safe
    C_per_factor = np.array([
        (1.0 - _entropy(P_cols[:, k]) / log_d) if col_sums[0, k] > 0 else 0.0
        for k in range(K)
    ])
    C_overall = float(C_per_factor.mean()) if K > 0 else 0.0

    I_overall = float(informativeness.mean()) if K > 0 else 0.0

    return {
        "importance":   importance,
        "D_per_latent": D_per_latent,
        "D":            D_overall,
        "C_per_factor": C_per_factor,
        "C":            C_overall,
        "I_per_factor": informativeness,
        "I":            I_overall,
        "factor_names": list(active_names),
    }
