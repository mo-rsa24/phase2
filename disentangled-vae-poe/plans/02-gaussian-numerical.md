# Plan 02 — Gaussian Numerical Verification

> **Scope:** disentangled-vae-poe | **Stage:** 2 | **Objective:** O2
> **Goals covered:** G2.1 G2.2 G2.3 G2.4 G2.5 G2.6 G2.7
> **Prerequisite:** Plan 01 complete (`src/metrics/poe.py` available)
> **W&B project:** disentangled-vae-poe | **tag:** stage2

## Objective

Train three model variants on rotated-Gaussian data at θ=π/4 and confirm
that the Disentangled VAE achieves significantly higher PoE `joint_acc` than
the Entangled VAE across 5 seeds (bootstrap 95% CI non-overlapping).

## Context

All variants use the same data (`make_rotation_data(n, theta=π/4, seed=seed)`).
The PoE composition step is identical across variants — only the latent
representation differs. `src/metrics/poe.py` from Plan 01 is reused directly.

## Tasks

- [ ] **T1** Oracle variant (`src/models/oracle.py`)
  - Bypasses the encoder; returns true [c₁, c₂] as z with near-certain
    precision Λ_oracle = I / σ² where σ = 0.01
  - No training required; Oracle runs are pure eval

- [ ] **T2** Configs for trained variants (`configs/gaussian/`)
  - `entangled_vae.yml`: linear VAE β=1, trained on R(π/4)·c + ε
  - `disentangled_vae.yml`: linear VAE β=8, same rotated data

- [ ] **T3** Training script (`scripts/train_gaussian_variants.py`)
  - `--variant {entangled,disentangled}` and `--seed {0..4}` flags
  - 5 seeds × 2 variants = 10 W&B training runs
  - Saves checkpoints to `checkpoints/gaussian/{variant}_seed{s}.pt`

- [ ] **T4** Reference-image averaging (extend `src/metrics/poe.py`)
  - `encode_reference_set(model, data, factor_id, factor_val, n=200) -> (mu_bar, lambda_bar)`
  - Encodes 200 samples with the target factor fixed, all others varying
  - Returns mean μ and diagonal precision

- [ ] **T5** Linear probe (`src/metrics/probes.py`)
  - `fit_linear_probe(model, dataset) -> probe` mapping z → c
  - Used to recover (ĉ₁, ĉ₂) from decoded images; fit on train split

- [ ] **T6** PoE composition evaluation (`scripts/eval_poe_gaussian.py`)
  - `--variant` and `--seed` flags
  - For each (α, β) pair in a 10×10 grid: compute PoE z* from reference
    posteriors; decode; apply linear probe; compute
    `joint_acc = P(|ĉ₁−α| < 0.1 and |ĉ₂−β| < 0.1)`
  - Saves `results/gaussian/joint_acc_{variant}_seed{s}.json`

- [ ] **T7** Posterior geometry plots for all three variants
  (`scripts/plot_posterior_geometry.py --stage 2`)
  - Reuses T5 from Plan 01; adds `--variant` flag
  - Saves `figures/stage2_posterior_geometry_{variant}.png`
  - Logs W&B artifact: `viz/posterior_geometry_stage2`

- [ ] **T8** Composition error vs θ curve (`scripts/plot_composition_error.py`)
  - Sweeps θ ∈ {0, π/8, π/4, 3π/8, π/2} for Entangled and Disentangled;
    mean ± std across 5 seeds on the same axes
  - Saves `figures/stage2_composition_error_vs_theta.png`
  - Logs W&B artifact: `viz/composition_error_vs_theta`

- [ ] **T9** Statistical significance (`scripts/compute_significance.py`)
  - Bootstrap 95% CI on `joint_acc(Disentangled) − joint_acc(Entangled)`
    across 5 seeds (10,000 bootstrap resamples); primary pass criterion
  - Also reports two-sided t-test p-value (secondary)
  - Pass criterion: CI lower bound > 0

- [ ] **T10** Results table + documentation (`docs/stage2_results.md`)
  - Table: variant × mean joint_acc ± std × 95% CI lower/upper × p-value
  - States whether O2 is confirmed; if not, applies contingency from
    MASTER_PLAN §Expected Outcomes

## Engagement Instructions

```bash
# Train
for seed in 0 1 2 3 4; do
  python scripts/train_gaussian_variants.py --variant entangled --seed $seed
  python scripts/train_gaussian_variants.py --variant disentangled --seed $seed
done

# Evaluate all three (Oracle needs no checkpoint)
for variant in oracle entangled disentangled; do
  for seed in 0 1 2 3 4; do
    python scripts/eval_poe_gaussian.py --variant $variant --seed $seed
  done
done

# Plots and stats
python scripts/plot_composition_error.py
python scripts/compute_significance.py
```

Confirm:
- Oracle > Disentangled > Entangled on mean joint_acc
- Bootstrap 95% CI lower bound on (Disentangled − Entangled) > 0
- Composition error vs θ curve is monotone increasing for Entangled VAE
- 10 W&B training runs exist under `disentangled-vae-poe`, tag `stage2`

## DoD Cross-Reference

| MASTER_PLAN DoD item | Task |
|---|---|
| All three variants trained across 5 seeds | T3 (Oracle: no training) |
| `figures/stage2_composition_error_vs_theta.png` | T8 |
| Bootstrap 95% CI non-overlapping | T9 |
| W&B `viz/composition_error_vs_theta` | T8 |
| `docs/stage2_results.md` | T10 |
