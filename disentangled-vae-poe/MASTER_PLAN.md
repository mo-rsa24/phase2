# MASTER PLAN — disentangled-vae-poe

> **Scope root:** `/home/molef/PhD/Paper2/disentangled-vae-poe/`
> **Experiment matrix:** [../EXPERIMENTS.md](../EXPERIMENTS.md)
> **W&B project:** `disentangled-vae-poe`
> **Created:** 2026-06-14

---

## 1. Mission

Demonstrate that Product of Experts (PoE) composition for conditional latent
diffusion models fails when generative factors are entangled in the VAE's latent
space, and that a disentangled latent space enables correct and tractable PoE
composition — **proved** analytically on a Gaussian rotation-mixing toy,
**verified** numerically on controlled synthetic data, and **demonstrated**
end-to-end on a FiLM-conditioned DDPM trained on 3D Shapes.

---

## 2. Objectives

### O1 — Analytic Proof

Derive the closed-form PoE composition error for a linear VAE with
rotation-entangled Gaussian posteriors, using the Wu & Goodman (2018)
precision-weighted PoE form:

```
Λ_PoE = Λ(z|c₁) + Λ(z|c₂) − Λ_prior
μ_PoE = Λ_PoE⁻¹ · (Λ(z|c₁)μ₁ + Λ(z|c₂)μ₂)
```

Show that composition error is **exactly zero** at θ=0 (axis-aligned posteriors)
and scales as **|sin(2θ)|** as the rotation angle — and hence the posterior
entanglement — increases. This makes the failure structural, not a matter of
degree: any non-zero off-diagonal covariance corrupts the PoE product mean.

*Experiment rows:* S1.θ0, S1.θ8, S1.θ4 — see [../EXPERIMENTS.md](../EXPERIMENTS.md).

---

### O2 — Numerical Verification

Train three VAE variants (Oracle, Vanilla VAE, Disentangled VAE) on the same
rotated-Gaussian synthetic data at θ=π/4 and measure PoE `joint_acc` — the
fraction of (c₁=α, c₂=β) combinations recovered within tolerance ε — across
**5 independent seeds**. Confirm that the gap between Disentangled and Vanilla is
statistically significant (bootstrap 95% confidence intervals non-overlapping).

*Experiment rows:* S2.oracle, S2.entangled, S2.disent.

---

### O3 — End-to-End Pipeline on 3D Shapes

Train a VAE family on 3D Shapes, train one FiLM-conditioned DDPM per VAE variant,
and show that **Spearman ρ(MIG, joint_acc) > 0.7** across all variants, where
`joint_acc` measures wall_hue × floor_hue PoE composition accuracy.

**Why ρ, not just Ada-GVAE > Vanilla:** The rank-correlation claim forces the
mechanism to be systematic across all variants. A two-endpoint comparison would
be satisfied by supervision alone; the correlation forces the ranking to hold at
every intermediate point. Ada-GVAE > Vanilla remains the **hard falsification
anchor** — if the weakly supervised upper bound cannot beat the unsupervised
baseline, the disentanglement → PoE connection is not demonstrated.

*Experiment rows:* S3.V-*, S3.D-* — see [../EXPERIMENTS.md](../EXPERIMENTS.md).

---

## 3. Goals

### Stage 1 — Gaussian Analytic Proof

- [ ] **G1.1** Data generator: `src/data/gaussian_rotation.py`
  - `make_rotation_data(n, theta, seed)` → `(x, c1, c2)` arrays
  - θ ∈ {0, π/8, π/4}

- [ ] **G1.2** Linear VAE fit + posterior extraction: `src/models/linear_vae.py`,
  `scripts/fit_linear_vae.py`
  - Fit diagonal Gaussian posterior at each θ; extract Λ(z|c₁), Λ(z|c₂), μ₁, μ₂

- [ ] **G1.3** PoE analytic computation + error metric: `src/metrics/poe.py`
  - `compute_poe(lambda1, mu1, lambda2, mu2, lambda_prior)` → `(lambda_poe, mu_poe)`
  - `poe_error(mu_poe, alpha, beta)` → scalar

- [ ] **G1.4** Posterior geometry figure: `scripts/plot_posterior_geometry.py`
  - Plots reference posteriors, PoE posterior, and true joint as Gaussian ellipses
    in 2D latent space, for each θ
  - Saves `figures/stage1_posterior_geometry_theta{0,8,4}.png`
  - Logs W&B artifact: `viz/posterior_geometry`

- [ ] **G1.5** PoE error vs θ curve: `scripts/plot_poe_error_curve.py`
  - Plots measured `poe_error` at {0, π/8, π/4} overlaid with |sin(2θ)|
  - Saves `figures/stage1_poe_error_vs_theta.png`
  - Logs W&B artifact: `viz/poe_error_vs_theta`

- [ ] **G1.6** Results documentation: `docs/stage1_results.md`
  - Explains what each figure shows; states whether R² of |sin(2θ)| fit exceeds
    0.95; interprets the result in light of O1

---

### Stage 2 — Gaussian Numerical Verification

- [ ] **G2.1** Configs for three variants: `configs/gaussian/oracle.yml`,
  `vanilla_vae.yml`, `disentangled_vae.yml`

- [ ] **G2.2** Training script: `scripts/train_gaussian_variants.py`
  - All three variants; `--seed` flag; W&B run per variant per seed

- [ ] **G2.3** PoE composition evaluation: `scripts/eval_poe_gaussian.py`
  - Encode reference images sharing one factor value → average posteriors →
    PoE analytically → decode → check if (c₁, c₂) recovered within ε
  - Outputs `joint_acc` per variant per seed

- [ ] **G2.4** 5-seed sweep: all three variants × 5 seeds = 15 runs

- [ ] **G2.5** Composition error vs θ curve, all variants overlaid:
  `scripts/plot_composition_error.py`
  - Saves `figures/stage2_composition_error_vs_theta.png`
  - Logs W&B artifact: `viz/composition_error_vs_theta`

- [ ] **G2.6** Statistical significance: bootstrap 95% CI on
  `joint_acc(Disentangled) − joint_acc(Vanilla)` across 5 seeds; CIs must be
  non-overlapping

- [ ] **G2.7** Results documentation: `docs/stage2_results.md`
  - Reports `joint_acc` mean ± std per variant; CI plot; states whether gap is
    significant; interprets in light of O2

---

### Stage 3 — 3D Shapes + FiLM-DDPM

#### 3a — VAE Family

- [ ] **G3.1** 3D Shapes loader: `src/data/shapes3d.py`
  - Auto-downloads 460,800 64×64 RGB images; returns `(img, latents)` pairs

- [ ] **G3.2** Train five VAE variants (configs in `configs/shapes3d/vae/`):
  - Vanilla VAE (`vanilla.yml`)
  - β-VAE at β ∈ {4, 8, 16} (`beta4.yml`, `beta8.yml`, `beta16.yml`)
  - FactorVAE (`factorvae.yml`)
  - Ada-GVAE (`adagvae.yml`) — weakly supervised, pairs differing in 1 factor
  - SepVAE — **[TENTATIVE — see architecture design plan below]**

- [ ] **G3.3** Per-run W&B artifacts for every VAE variant:
  - `viz/latent_traversals` — vary one dim, fix others, reconstruct; logged
    every N epochs
  - `viz/dci_importance_heatmap` — (latent_dims × factors) importance matrix
  - `viz/factor_lda_histograms` — 1D LDA projection per factor

- [ ] **G3.4** Disentanglement eval script: `scripts/eval_disentanglement.py`
  - Computes and logs `metrics/mig` and `metrics/dci_d` per variant

#### 3b — FiLM-DDPM Composition

- [ ] **G3.5** FiLM conditioning module: `src/models/film.py`
  - `h' = (1 + γ(z)) · h + β(z)`, where γ/β are linear projections of z

- [ ] **G3.6** FiLM-conditioned DDPM: `src/models/ddpm_film.py`

- [ ] **G3.7** Training script: `scripts/train_ddpm_film.py`
  - One run per VAE variant; `--vae-checkpoint` flag

- [ ] **G3.8** PoE composition eval on 3D Shapes: `scripts/eval_poe_shapes3d.py`
  - Encodes references sharing one hue → PoE → conditions DDPM → generates
  - Measures `joint_acc`, `wall_acc`, `floor_acc` per variant
  - Logs W&B artifact: `viz/poe_posterior_geometry`

- [ ] **G3.9** Rank-correlation analysis: `scripts/compute_rank_correlation.py`
  - Computes Spearman ρ(MIG, joint_acc) and ρ(DCI-D, joint_acc) across variants

- [ ] **G3.10** Results documentation: `docs/stage3_results.md`
  - `joint_acc` table across all variants; ρ values; interpretation in light of O3

---

## 4. SepVAE Architecture Design — Pending

SepVAE is included in Stage 3a as a **tentative target**. The architecture borrows
GRL wrong-head suppression from the adversarial disentanglement literature but must
be redesigned for this setting: **no contrastive loss, Gaussian posteriors,
PoE composition at inference**. The objective function has not been finalized.

The original SepVAE is trained with a supervised contrastive signal; we are
applying the partitioned-latent idea in an unsupervised or weakly-supervised
setting where the primary constraint is that PoE must remain tractable. These are
different operating conditions and the loss derivation cannot be borrowed directly.

**Before implementing SepVAE, run this skill sequence in order:**

```
/init-master-plan
"Create a focused architecture design plan for a disentangled VAE with GRL
wrong-head suppression adapted for PoE composition on 3D Shapes. No contrastive
loss. Gaussian posteriors. Latent partitioned into [z_common, z_wall_hue,
z_floor_hue]. Objective function is unknown — the plan must derive it from two
constraints: (1) each partition should be informative about its assigned factor,
(2) PoE over partitions must remain tractable (requires near-diagonal posterior
covariance). Deliverable: an objective function, an ablation plan, and a
falsification condition."
```

```
/unpack-paper
"Locatello et al. 2019 — Challenging Common Assumptions in the Unsupervised
Learning of Disentangled Representations. Focus on: why unsupervised
disentanglement is unidentifiable; what inductive biases are required; what
Ada-GVAE does differently and why weak supervision changes the picture."
```

```
/unpack-paper
"Wu & Goodman 2018 — Multimodal Generative Models for Scalable Weakly-Supervised
Learning. Focus on: the PoE precision-weighting closed form; the independence
assumption each expert must satisfy; what breaks structurally when experts are
not independent (connect explicitly to Stage 1 derivation)."
```

```
/socratic
"I want to partition the VAE latent space into [z_common, z_wall_hue, z_floor_hue]
and suppress cross-factor information using GRL adversarial heads. Walk me through:
(1) What objective function is sufficient to guarantee z_wall_hue ⊥ z_floor_hue
in the posterior? (2) Is GRL the right suppression mechanism here, or is there a
cleaner option given Gaussian posteriors? (3) Does PoE tractability require true
posterior independence or only low off-diagonal covariance? (4) How does the
absence of contrastive loss change what the model can learn?"
```

```
/experiment-planner
"Design a minimal ablation for SepVAE in Stage 3a. Key design choices:
partition sizes [k_common, k_wall, k_floor], GRL λ schedule (fixed vs. annealed),
whether to include a swap consistency loss, and whether to use per-partition β
or shared β. What is the smallest set of configurations that isolates the
contribution of each design choice without adding more than 3 extra training runs?"
```

Promote SepVAE from tentative to committed only after all five prompts above
have been run and the objective function is written down in `docs/sepvae_design.md`.

---

## 5. Expected Outcomes

### If the full chain holds

> PoE composition error in the Gaussian setting is exactly proportional to
> |sin(2θ)|, making the failure structural: any non-zero off-diagonal covariance
> corrupts the PoE product mean. Numerically, the disentangled VAE recovers joint
> factor combinations with `joint_acc` significantly above the vanilla baseline
> (5-seed 95% CI non-overlapping). On 3D Shapes, Spearman ρ(MIG, joint_acc)
> exceeds 0.7 across the five VAE variants, Ada-GVAE tops both the MIG and
> joint_acc rankings, and vanilla VAE is at the bottom.

### Partial-failure contingencies

| Failure mode | What it means | How to handle |
|---|---|---|
| Stage 2 gap is not significant (5 seeds) | Numerical support is absent | Scope retains Stage 1 as analytic contribution; downgrade O2 to "directional" |
| Stage 3 ρ < 0.7 but Ada-GVAE > Vanilla | Mechanism works at extremes but middle variants break the ordering | Report two-endpoint result; discuss why ρ is low (training variance? metric mismatch?) |
| Ada-GVAE ≤ Vanilla in Stage 3 | C3b falsified entirely | Scope ends at Stage 2; report the null result |
| SepVAE architecture unresolvable | Tentative target cannot be promoted | Drop SepVAE; run Stage 3a with five variants; report it as a limitation |

---

## 6. Definition of Done

- [x] `EXPERIMENTS.md` locked and committed (at [`../EXPERIMENTS.md`](../EXPERIMENTS.md))

- [ ] **Stage 1**
  - [ ] `src/data/gaussian_rotation.py`, `src/models/linear_vae.py`,
    `src/metrics/poe.py` written and tested
  - [ ] `figures/stage1_posterior_geometry_theta{0,8,4}.png` saved
  - [ ] `figures/stage1_poe_error_vs_theta.png` saved; |sin(2θ)| fit R² > 0.95
  - [ ] W&B run under project `disentangled-vae-poe` with
    `viz/posterior_geometry` and `viz/poe_error_vs_theta` logged
  - [ ] `docs/stage1_results.md` written

- [ ] **Stage 2**
  - [ ] All three variants trained across 5 seeds (15 W&B runs)
  - [ ] `figures/stage2_composition_error_vs_theta.png` saved
  - [ ] Bootstrap 95% CI on `joint_acc(Disentangled) − joint_acc(Vanilla)`:
    CIs non-overlapping
  - [ ] W&B runs with `viz/composition_error_vs_theta` logged
  - [ ] `docs/stage2_results.md` written

- [ ] **Stage 3**
  - [ ] 3D Shapes loader (`src/data/shapes3d.py`) complete
  - [ ] All five committed VAE variants trained; W&B runs under
    `disentangled-vae-poe`
  - [ ] Every VAE run has `viz/latent_traversals`, `viz/dci_importance_heatmap`,
    `viz/factor_lda_histograms` logged
  - [ ] FiLM-DDPM (`src/models/ddpm_film.py`) implemented and trained per variant
  - [ ] `joint_acc` table produced across all variants
  - [ ] Ada-GVAE achieves highest `joint_acc`; vanilla VAE achieves lowest
  - [ ] Spearman ρ(MIG, joint_acc) > 0.7 computed and logged
  - [ ] W&B runs with `viz/poe_posterior_geometry` logged per DDPM variant
  - [ ] `docs/stage3_results.md` written

- [ ] All W&B runs tagged under project `disentangled-vae-poe`
