# 01 · Gaussian Analytic Proof

[⌂ Index](00-INDEX.md) · [next → 02](02-gaussian-numerical.md)

## Reference while you do it
- 📄 Plan: plans/01-gaussian-analytic.md
- 💻 Code: src/data/gaussian_rotation.py · src/models/linear_vae.py · src/metrics/poe.py
- 💻 Scripts: scripts/fit_linear_vae.py · scripts/plot_posterior_geometry.py · scripts/plot_poe_error_curve.py
- 📊 W&B: project `disentangled-vae-poe`, tag `stage1`

## Section context (paste into the Todoist section)
**Description:** Derive and computationally verify that PoE composition error for a linear Gaussian VAE scales as |sin(2θ)|. All code is standalone Python scripts; figures saved to `figures/` and logged to W&B. `src/metrics/poe.py` produced here is reused in Plans 02 and 03.
**Objective:** Prove that the PoE failure is structural — any non-zero off-diagonal posterior covariance corrupts the PoE product mean, making the error exactly zero at θ=0 and proportional to |sin(2θ)| otherwise.
**Goal:** G1.1–G1.6: data generator → linear VAE → PoE module → posterior geometry figure → PoE error curve → results doc.
**Verify (whole leaf):**
```
for theta in 0 0.3927 0.7854; do
  python scripts/fit_linear_vae.py --theta $theta --seed 42
done
python scripts/plot_posterior_geometry.py
python scripts/plot_poe_error_curve.py
```
Confirm: θ=0 `poe_error` < 0.05; θ=π/4 `poe_error` ≥ 3× θ=0 error; R² of |sin(2θ)| fit > 0.95; W&B run tagged `stage1` with `viz/posterior_geometry` and `viz/poe_error_vs_theta` logged.

## Tasks (one at a time)
- T1 — Data generator (`src/data/gaussian_rotation.py`)
  - **Goal:** G1.1 — `make_rotation_data(n, theta, seed) -> (x, c1, c2)` where x ∈ ℝ²; θ ∈ {0, π/8, π/4}
  - **Verify:** Tested implicitly when `python scripts/fit_linear_vae.py --theta 0 --seed 42` runs without ImportError
  - `make_rotation_data(n, theta, seed) -> (x, c1, c2)` where x ∈ ℝ²
  - R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]; noise ε ~ N(0, 0.05·I)
  - Expose θ as a float argument; callers pass 0, π/8, π/4
- T2 — Linear VAE (`src/models/linear_vae.py`)
  - **Goal:** G1.2 — single `nn.Linear` encoder → (μ, log σ²); single `nn.Linear` decoder → x̂; ELBO: MSE + β·KL, β=1
  - **Verify:** Tested implicitly when `python scripts/fit_linear_vae.py --theta 0 --seed 42` completes and saves checkpoint to `checkpoints/gaussian/`
  - Encoder: single `nn.Linear` → (μ, log σ²), diagonal Gaussian posterior
  - Decoder: single `nn.Linear` → x̂; no activation functions
  - Keeps the posterior geometry tractable and analytically interpretable
  - ELBO: MSE reconstruction + β·KL, β=1
- T3 — PoE analytic module (`src/metrics/poe.py`)
  - **Goal:** G1.3 — `compute_poe(lambda1, mu1, lambda2, mu2, lambda_prior) -> (lambda_poe, mu_poe)`; `poe_error(mu_poe, alpha, beta) -> scalar`; NumPy only; `encode_reference_set` stub
  - **Verify:** Tested implicitly when `python scripts/plot_poe_error_curve.py` generates the error curve without exception
  - `compute_poe(lambda1, mu1, lambda2, mu2, lambda_prior) -> (lambda_poe, mu_poe)` where Λ_PoE = Λ₁ + Λ₂ − Λ_prior and μ_PoE = Λ_PoE⁻¹·(Λ₁μ₁ + Λ₂μ₂)
  - `poe_error(mu_poe, alpha, beta) -> scalar` = |ĉ₁−α|² + |ĉ₂−β|²
  - NumPy only — no torch dependency
- T4 — Training script (`scripts/fit_linear_vae.py`)
  - **Goal:** G1.2 — `--theta`, `--seed`, `--epochs`, `--lr`, `--latent-dim`, `--wandb-project` flags; checkpoint to `checkpoints/gaussian/linear_vae_theta{value}_seed{s}.pt`; training loss logged to W&B
  - **Verify:** `for theta in 0 0.3927 0.7854; do python scripts/fit_linear_vae.py --theta $theta --seed 42; done` — all complete; W&B run appears under `disentangled-vae-poe`, tag `stage1`
  - `--theta` and `--seed` flags; trains one VAE per call
  - Saves checkpoint to `checkpoints/gaussian/linear_vae_theta{value}_seed{s}.pt`
  - Logs training loss curve to W&B
- T5 — Posterior geometry figure (`scripts/plot_posterior_geometry.py`)
  - **Goal:** G1.4 — three-panel figure (one subplot per θ), 4 ellipses per panel; saves `figures/stage1_posterior_geometry_theta{0,8,4}.png`; logs W&B artifact `viz/posterior_geometry`
  - **Verify:** `python scripts/plot_posterior_geometry.py` — θ=0 panel: PoE ellipse overlaps true joint ellipse; θ=π/4 panel: PoE ellipse visibly diverges; `viz/posterior_geometry` logged in W&B under `stage1`
  - For each θ ∈ {0, π/8, π/4}: extract q(z|x) for reference points (c₁=1, c₂=0) and (c₁=0, c₂=1); compute PoE posterior analytically; compute true joint p(z|c₁=1, c₂=1)
  - Plot all four as 2D Gaussian ellipses (1σ and 2σ contours)
  - Three-panel figure, one subplot per θ
  - Saves `figures/stage1_posterior_geometry_theta{0,8,4}.png`
  - Logs W&B artifact: `viz/posterior_geometry`
- T6 — PoE error curve (`scripts/plot_poe_error_curve.py`)
  - **Goal:** G1.5 — 10×10 (α, β) grid per θ; fit c·|sin(2θ)|; R² > 0.95; saves `figures/stage1_poe_error_vs_theta.png`; logs W&B artifact `viz/poe_error_vs_theta`
  - **Verify:** `python scripts/plot_poe_error_curve.py` — R² > 0.95 printed to stdout; figure saved; `viz/poe_error_vs_theta` logged in W&B
  - For each θ: mean `poe_error` over a 10×10 grid of (α, β) pairs
  - Plot measured error vs θ; overlay fitted curve c·|sin(2θ)|; compute and print R²
  - Saves `figures/stage1_poe_error_vs_theta.png`
  - Logs W&B artifact: `viz/poe_error_vs_theta`
- T7 — Results documentation (`docs/stage1_results.md`)
  - **Goal:** G1.6 — describes what each figure shows; states R² value; explicit O1 verdict
  - **Verify:** ⚠️ No CLI check — confirm `docs/stage1_results.md` exists, contains the R² value (from T6), and states "O1 confirmed" or "O1 denied" with a reason
  - Describes what each figure shows
  - States the R² value of the |sin(2θ)| fit
  - Explicit confirmation or denial of O1
