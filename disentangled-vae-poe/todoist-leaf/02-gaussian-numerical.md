# 02 · Gaussian Numerical Verification

[⌂ Index](00-INDEX.md) · [← 01](01-gaussian-analytic.md) · [next → 03](03-3dshapes-ldm.md)

## Reference while you do it
- 📄 Plan: plans/02-gaussian-numerical.md
- ⚠️ Prerequisite: Plan 01 complete (`src/metrics/poe.py` must exist)
- 💻 Code: src/models/oracle.py · src/metrics/poe.py (extend) · src/metrics/probes.py
- 💻 Configs: configs/gaussian/entangled_vae.yml · configs/gaussian/disentangled_vae.yml
- 💻 Scripts: scripts/train_gaussian_variants.py · scripts/eval_poe_gaussian.py · scripts/plot_posterior_geometry.py · scripts/plot_composition_error.py · scripts/compute_significance.py
- 📊 W&B: project `disentangled-vae-poe`, tag `stage2`

## Section context (paste into the Todoist section)
**Description:** Train three VAE variants (Oracle, Entangled, Disentangled) on rotated-Gaussian data at θ=π/4 and measure PoE `joint_acc` across 5 seeds. All variants use the same data and the same PoE composition step from Plan 01 — only the latent representation differs.
**Objective:** Confirm numerically that disentanglement improves PoE composition accuracy (O2), with bootstrap 95% CI non-overlapping as the primary pass criterion.
**Goal:** G2.1–G2.7: Oracle module + YAML configs → training script → reference encoder + linear probe → eval script → posterior geometry plots → composition error curve → significance test → results doc.
**Verify (whole leaf):**
```
for seed in 0 1 2 3 4; do
  python scripts/train_gaussian_variants.py --variant entangled --seed $seed
  python scripts/train_gaussian_variants.py --variant disentangled --seed $seed
done
for variant in oracle entangled disentangled; do
  for seed in 0 1 2 3 4; do
    python scripts/eval_poe_gaussian.py --variant $variant --seed $seed
  done
done
python scripts/plot_composition_error.py
python scripts/compute_significance.py
```
Confirm: Oracle > Disentangled > Entangled on mean `joint_acc`; bootstrap 95% CI lower bound on (Disentangled − Entangled) > 0; composition error vs θ is monotone increasing for Entangled VAE; 10 W&B training runs under `disentangled-vae-poe`, tag `stage2`.

## Tasks (one at a time)
- T1 — Oracle variant (`src/models/oracle.py`)
  - **Goal:** Oracle that bypasses the encoder and returns true [c₁, c₂] as z with Λ_oracle = I / σ² (σ = 0.01); no training required; serves as the upper-bound baseline
  - **Verify:** Tested implicitly when `python scripts/eval_poe_gaussian.py --variant oracle --seed 0` runs without ImportError and produces `results/gaussian/joint_acc_oracle_seed0.json`
  - Bypasses the encoder; returns true [c₁, c₂] as z with near-certain precision Λ_oracle = I / σ² where σ = 0.01
  - No training required; Oracle runs are pure eval
- T2 — Configs for trained variants (`configs/gaussian/`)
  - **Goal:** G2.1 — `configs/gaussian/entangled_vae.yml` (linear VAE β=1, trained on R(π/4)·c + ε) and `configs/gaussian/disentangled_vae.yml` (linear VAE β=8, same rotated data)
  - **Verify:** Tested implicitly when `python scripts/train_gaussian_variants.py --variant entangled --seed 0` loads the config without KeyError
  - `entangled_vae.yml`: linear VAE β=1, trained on R(π/4)·c + ε
  - `disentangled_vae.yml`: linear VAE β=8, same rotated data
- T3 — Training script (`scripts/train_gaussian_variants.py`)
  - **Goal:** G2.2 — `--variant {entangled,disentangled}` and `--seed {0..4}` flags; 5 seeds × 2 variants = 10 W&B training runs; checkpoints to `checkpoints/gaussian/{variant}_seed{s}.pt`
  - **Verify:** `for seed in 0 1 2 3 4; do python scripts/train_gaussian_variants.py --variant entangled --seed $seed && python scripts/train_gaussian_variants.py --variant disentangled --seed $seed; done` — 10 checkpoints saved; 10 W&B runs appear under `disentangled-vae-poe`, tag `stage2`
  - `--variant {entangled,disentangled}` and `--seed {0..4}` flags
  - 5 seeds × 2 variants = 10 W&B training runs
  - Saves checkpoints to `checkpoints/gaussian/{variant}_seed{s}.pt`
- T4 — Reference-image averaging (extend `src/metrics/poe.py`)
  - **Goal:** `encode_reference_set(model, data, factor_id, factor_val, n=200) -> (mu_bar, lambda_bar)` — encodes 200 samples with target factor fixed, all others varying; returns mean μ and diagonal precision
  - **Verify:** Tested implicitly when `python scripts/eval_poe_gaussian.py --variant entangled --seed 0` runs without error (calls `encode_reference_set` internally)
  - `encode_reference_set(model, data, factor_id, factor_val, n=200) -> (mu_bar, lambda_bar)`
  - Encodes 200 samples with the target factor fixed, all others varying
  - Returns mean μ and diagonal precision
- T5 — Linear probe (`src/metrics/probes.py`)
  - **Goal:** `fit_linear_probe(model, dataset) -> probe` mapping z → c; fit on train split; used to recover (ĉ₁, ĉ₂) from decoded images for `joint_acc` computation
  - **Verify:** Tested implicitly when `python scripts/eval_poe_gaussian.py --variant entangled --seed 0` produces a `joint_acc` value (probe is called internally)
  - `fit_linear_probe(model, dataset) -> probe` mapping z → c
  - Used to recover (ĉ₁, ĉ₂) from decoded images; fit on train split
- T6 — PoE composition evaluation (`scripts/eval_poe_gaussian.py`)
  - **Goal:** G2.3 — `--variant` and `--seed` flags; 10×10 (α, β) grid; `joint_acc = P(|ĉ₁−α| < 0.1 and |ĉ₂−β| < 0.1)`; saves `results/gaussian/joint_acc_{variant}_seed{s}.json`
  - **Verify:** Full eval loop: `for variant in oracle entangled disentangled; do for seed in 0 1 2 3 4; do python scripts/eval_poe_gaussian.py --variant $variant --seed $seed; done; done` — Oracle > Disentangled > Entangled on mean `joint_acc`
  - `--variant` and `--seed` flags
  - For each (α, β) pair in a 10×10 grid: compute PoE z* from reference posteriors; decode; apply linear probe; compute `joint_acc = P(|ĉ₁−α| < 0.1 and |ĉ₂−β| < 0.1)`
  - Saves `results/gaussian/joint_acc_{variant}_seed{s}.json`
- T7 — Posterior geometry plots for all three variants (`scripts/plot_posterior_geometry.py --stage 2`)
  - **Goal:** `figures/stage2_posterior_geometry_{variant}.png` for Oracle, Entangled, Disentangled; W&B artifact `viz/posterior_geometry_stage2`
  - **Verify:** `python scripts/plot_posterior_geometry.py --stage 2` saves three PNG files (one per variant) and logs `viz/posterior_geometry_stage2` to W&B
  - Reuses T5 from Plan 01; adds `--variant` flag
  - Saves `figures/stage2_posterior_geometry_{variant}.png`
  - Logs W&B artifact: `viz/posterior_geometry_stage2`
- T8 — Composition error vs θ curve (`scripts/plot_composition_error.py`)
  - **Goal:** G2.5 — sweeps θ ∈ {0, π/8, π/4, 3π/8, π/2}; mean ± std across 5 seeds on same axes; saves `figures/stage2_composition_error_vs_theta.png`; logs W&B artifact `viz/composition_error_vs_theta`
  - **Verify:** `python scripts/plot_composition_error.py` — figure saved; Entangled VAE curve is monotone increasing with θ; `viz/composition_error_vs_theta` logged in W&B
  - Sweeps θ ∈ {0, π/8, π/4, 3π/8, π/2} for Entangled and Disentangled; mean ± std across 5 seeds on the same axes
  - Saves `figures/stage2_composition_error_vs_theta.png`
  - Logs W&B artifact: `viz/composition_error_vs_theta`
- T9 — Statistical significance (`scripts/compute_significance.py`)
  - **Goal:** G2.6 — bootstrap 95% CI on `joint_acc(Disentangled) − joint_acc(Entangled)` across 5 seeds (10,000 bootstrap resamples); CI lower bound > 0 is the primary pass criterion; two-sided t-test p-value is secondary
  - **Verify:** `python scripts/compute_significance.py` — bootstrap 95% CI lower bound on (Disentangled − Entangled) > 0 printed to stdout
  - Bootstrap 95% CI on `joint_acc(Disentangled) − joint_acc(Entangled)` across 5 seeds (10,000 bootstrap resamples); primary pass criterion
  - Also reports two-sided t-test p-value (secondary)
  - Pass criterion: CI lower bound > 0
- T10 — Results table + documentation (`docs/stage2_results.md`)
  - **Goal:** G2.7 — table: variant × mean `joint_acc` ± std × 95% CI lower/upper × p-value; explicit O2 verdict; contingency from MASTER_PLAN §Expected Outcomes applied if O2 not confirmed
  - **Verify:** ⚠️ No CLI check — confirm `docs/stage2_results.md` exists, has the results table, and states "O2 confirmed" or "O2 denied" with the contingency note
  - Table: variant × mean joint_acc ± std × 95% CI lower/upper × p-value
  - States whether O2 is confirmed; if not, applies contingency from MASTER_PLAN §Expected Outcomes
