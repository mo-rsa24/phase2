# Plan 01 — Gaussian Analytic Proof

> **Scope:** disentangled-vae-poe | **Stage:** 1 | **Objective:** O1
> **Goals covered:** G1.1 G1.2 G1.3 G1.4 G1.5 G1.6
> **W&B project:** disentangled-vae-poe | **tag:** stage1

## Objective

Derive and computationally verify the analytic PoE failure result: that
composition error for a linear Gaussian VAE scales as |sin(2θ)| with θ
encoding the rotation-entanglement of the posterior.

## Context

All code is standalone Python scripts. No notebooks. Figures saved to
`figures/` and logged to W&B. `src/metrics/poe.py` produced here is reused
in Plans 02 and 03.

## Tasks

- [ ] **T1** Data generator (`src/data/gaussian_rotation.py`)
  - `make_rotation_data(n, theta, seed) -> (x, c1, c2)` where x ∈ ℝ²
  - R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]; noise ε ~ N(0, 0.05·I)
  - Expose θ as a float argument; callers pass 0, π/8, π/4

- [ ] **T2** Linear VAE (`src/models/linear_vae.py`)
  - Encoder: single `nn.Linear` → (μ, log σ²), diagonal Gaussian posterior
  - Decoder: single `nn.Linear` → x̂; no activation functions
  - Keeps the posterior geometry tractable and analytically interpretable
  - ELBO: MSE reconstruction + β·KL, β=1

- [ ] **T3** PoE analytic module (`src/metrics/poe.py`)
  - `compute_poe(lambda1, mu1, lambda2, mu2, lambda_prior) -> (lambda_poe, mu_poe)`
    where Λ_PoE = Λ₁ + Λ₂ − Λ_prior and μ_PoE = Λ_PoE⁻¹·(Λ₁μ₁ + Λ₂μ₂)
  - `poe_error(mu_poe, alpha, beta) -> scalar` = |ĉ₁−α|² + |ĉ₂−β|²
  - NumPy only — no torch dependency

- [ ] **T4** Training script (`scripts/fit_linear_vae.py`)
  - `--theta` and `--seed` flags; trains one VAE per call
  - Saves checkpoint to `checkpoints/gaussian/linear_vae_theta{value}_seed{s}.pt`
  - Logs training loss curve to W&B

- [ ] **T5** Posterior geometry figure (`scripts/plot_posterior_geometry.py`)
  - For each θ ∈ {0, π/8, π/4}: extract q(z|x) for two reference points
    (c₁=1, c₂=0) and (c₁=0, c₂=1); compute PoE posterior analytically;
    compute true joint p(z|c₁=1, c₂=1)
  - Plot all four as 2D Gaussian ellipses (1σ and 2σ contours)
  - Three-panel figure, one subplot per θ
  - Saves `figures/stage1_posterior_geometry_theta{0,8,4}.png`
  - Logs W&B artifact: `viz/posterior_geometry`

- [ ] **T6** PoE error curve (`scripts/plot_poe_error_curve.py`)
  - For each θ: mean `poe_error` over a 10×10 grid of (α, β) pairs
  - Plot measured error vs θ; overlay fitted curve c·|sin(2θ)|; compute and
    print R²
  - Saves `figures/stage1_poe_error_vs_theta.png`
  - Logs W&B artifact: `viz/poe_error_vs_theta`

- [ ] **T7** Results documentation (`docs/stage1_results.md`)
  - Describes what each figure shows
  - States the R² value of the |sin(2θ)| fit
  - Explicit confirmation or denial of O1

## Engagement Instructions

```bash
for theta in 0 0.3927 0.7854; do
  python scripts/fit_linear_vae.py --theta $theta --seed 42
done
python scripts/plot_posterior_geometry.py
python scripts/plot_poe_error_curve.py
```

Confirm:
- θ=0: PoE ellipse overlaps the true joint ellipse; `poe_error` < 0.05
- θ=π/4: PoE ellipse diverges from the true joint; `poe_error` is ≥ 3×
  the θ=0 error
- R² of |sin(2θ)| fit exceeds 0.95
- W&B run under `disentangled-vae-poe`, tag `stage1`, has
  `viz/posterior_geometry` and `viz/poe_error_vs_theta` logged

## DoD Cross-Reference

| MASTER_PLAN DoD item | Task |
|---|---|
| `src/data/gaussian_rotation.py` | T1 |
| `src/models/linear_vae.py` | T2 |
| `src/metrics/poe.py` | T3 |
| `figures/stage1_posterior_geometry_theta{0,8,4}.png` | T5 |
| `figures/stage1_poe_error_vs_theta.png`; R² > 0.95 | T6 |
| W&B `viz/posterior_geometry` + `viz/poe_error_vs_theta` | T5, T6 |
| `docs/stage1_results.md` | T7 |
