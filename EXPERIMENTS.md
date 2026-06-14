# Experiments — Compositional Latent Diffusion via Product of Experts

> **Linked to:** [MASTER_PLAN.md](MASTER_PLAN.md)
> **Last updated:** 2026-06-14

---

## Core Hypothesis

PoE composition for conditional latent diffusion models fails when generative
factors are tangled in the VAE's latent space. A disentangled latent space —
where each dimension corresponds to one factor of variation — makes PoE
composition tractable and correct.

**PoE closed form (Wu & Goodman 2018):**

```
Λ_PoE = Λ(z|c₁) + Λ(z|c₂) − Λ_prior
μ_PoE = Λ_PoE⁻¹ · (Λ(z|c₁)μ₁ + Λ(z|c₂)μ₂)
```

---

## Claims

| claim_id | statement |
|---|---|
| C1 | PoE composition error scales as \|sin(2θ)\| when the linear VAE has rotation-entangled posteriors |
| C2 | Disentanglement improves PoE `joint_acc` over the entangled vanilla-VAE baseline on rotated-Gaussian data |
| C3a | On 3D Shapes, more disentangled VAEs (higher MIG / DCI-D) achieve higher PoE composition accuracy |
| C3b | A FiLM-DDPM conditioned on a disentangled VAE generates more accurate joint compositions than one conditioned on an entangled VAE |

---

## Stage 1 — Gaussian Analytic Proof

**Data:** `x = R(θ) · [c₁, c₂]ᵀ + ε`, `c₁, c₂ ~ N(0,1)`, `ε ~ N(0, σ²I)`
**Model:** linear VAE — single linear encoder, diagonal Gaussian posterior
**θ sweep:** 0 (identity / disentangled), π/8 (partial), π/4 (maximal entanglement)

| config_id | claim_id | ablation_var | baseline | condition | primary_metric | falsification | wandb_artifact |
|---|---|---|---|---|---|---|---|
| S1.θ0 | C1 | rotation angle θ | — *(this is the baseline)* | θ = 0, identity rotation — posteriors axis-aligned | `poe_error` = \|ĉ₁ − α\|² + \|ĉ₂ − β\|² | — | `viz/posterior_geometry` at θ=0: reference ellipses + PoE ellipse + true joint in 2D latent |
| S1.θ8 | C1 | rotation angle θ | S1.θ0 (`poe_error` ≈ 0) | θ = π/8, partial entanglement | `poe_error` | `poe_error(S1.θ8) ≤ poe_error(S1.θ0)` — error should strictly increase | `viz/posterior_geometry` at θ=π/8 |
| S1.θ4 | C1 | rotation angle θ | S1.θ0 (`poe_error` ≈ 0) | θ = π/4, maximal entanglement | `poe_error` | `poe_error(S1.θ4)` not significantly larger than `poe_error(S1.θ0)` — the analytic claim is wrong | `viz/posterior_geometry` at θ=π/4; `viz/poe_error_vs_theta` curve fitted against \|sin(2θ)\| |

**Derived check:** fit `poe_error` vs `{0, π/8, π/4}` against `|sin(2θ)|`; R² should exceed 0.95.

---

## Stage 2 — Gaussian Numerical Verification

**Data:** θ = π/4 (maximally entangled), same rotation-mixing process
**Composition:** encode reference images sharing one factor value → average posteriors → PoE `q̄(z|c₁=α)` → decode
**All variants trained on the same rotated-Gaussian data**

| config_id | claim_id | ablation_var | baseline | condition | primary_metric | falsification | wandb_artifact |
|---|---|---|---|---|---|---|---|
| S2.oracle | C2 | model type | S2.entangled | True [c₁, c₂] used directly as z — upper bound, no encoder | `joint_acc` = % of (c₁=α, c₂=β) pairs recovered within tolerance ε | — | `viz/posterior_geometry`; `viz/composition_error_vs_theta` |
| S2.entangled | C2 | model type | — *(this is the baseline)* | Vanilla VAE (β=1) — no disentanglement pressure | `joint_acc` | — | `viz/posterior_geometry`; `metrics/mig`; `metrics/dci_d` |
| S2.disent | C2 | model type | S2.entangled | β-VAE (β=8) *or* SepVAE with GRL wrong-head probes | `joint_acc` | `joint_acc(S2.disent) ≤ joint_acc(S2.entangled)` — disentanglement does not help PoE | `viz/posterior_geometry`; `viz/mig_vs_joint_acc` scatter; `metrics/mig`; `metrics/dci_d` |

**Secondary metrics (all variants):** MIG, DCI-D.
**Expected ordering:** `S2.oracle > S2.disent > S2.entangled`.

---

## Stage 3 — 3D Shapes + FiLM-Conditioned DDPM

**Dataset:** 3D Shapes — 460,800 64×64 RGB images, 6 independent factors:
`floor_hue` (10), `wall_hue` (10), `object_hue` (10), `scale` (8), `shape` (4), `orientation` (15)

**PoE target:** `wall_hue × floor_hue` composition — encode references sharing one hue, compose
via PoE analytically, condition DDPM on the composed `z*`, generate.

**FiLM conditioning:** `h' = (1 + γ(z)) · h + β(z)`, where γ/β are linear projections of `z`.
One FiLM-DDPM trained per VAE variant.

---

### 3a — VAE Disentanglement Ablation

Primary claim: C3a. Goal is to rank VAE variants by disentanglement quality before conditioning the DDPM.

| config_id | claim_id | ablation_var | baseline | condition | primary_metric | falsification | wandb_artifact |
|---|---|---|---|---|---|---|---|
| S3.V-vanilla | C3a | VAE architecture / β | — *(this is the baseline)* | Vanilla VAE (β=1) | MIG on 3D Shapes factors | — | `viz/latent_traversals` (every N epochs); `viz/dci_importance_heatmap`; `viz/factor_lda_histograms` |
| S3.V-b4 | C3a | β | S3.V-vanilla | β-VAE, β=4 | MIG | — | same as S3.V-vanilla |
| S3.V-b8 | C3a | β | S3.V-vanilla | β-VAE, β=8 | MIG | — | same as S3.V-vanilla |
| S3.V-b16 | C3a | β | S3.V-vanilla | β-VAE, β=16 | MIG | — | same as S3.V-vanilla |
| S3.V-fvae | C3a | VAE architecture | S3.V-vanilla | FactorVAE | MIG | — | same as S3.V-vanilla |
| S3.V-sepvae | C3a | VAE architecture | S3.V-vanilla | SepVAE — partitioned latent space + GRL wrong-head probes | MIG | — | same + `viz/wrong_head_probe_acc` curve (should fall to chance) |
| S3.V-adagvae | C3a | architecture + supervision | S3.V-vanilla | Ada-GVAE — weakly supervised, pairs differing in exactly 1 factor | MIG | `MIG(S3.V-adagvae) ≤ MIG(S3.V-vanilla)` — label signal not helping | same as S3.V-vanilla |

**Secondary metric (all 3a rows):** DCI-D. Used to rank variants entering 3b.

---

### 3b — FiLM-DDPM Composition Evaluation

Primary claim: C3b. Ablation variable is the conditioned VAE variant; all else held equal
(same DDPM architecture, same training data, same PoE composition procedure).

| config_id | claim_id | ablation_var | baseline | condition | primary_metric | falsification | wandb_artifact |
|---|---|---|---|---|---|---|---|
| S3.D-vanilla | C3b | conditioned VAE | — *(this is the baseline)* | FiLM-DDPM conditioned on S3.V-vanilla z | `joint_acc` (wall_hue × floor_hue, via DCI linear probe on generated images) | — | `viz/poe_posterior_geometry` (ref posteriors + PoE + true joint, projected onto top-2 factor dims) |
| S3.D-b4 | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-b4 z | `joint_acc` | — | same as S3.D-vanilla |
| S3.D-b8 | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-b8 z | `joint_acc` | — | same as S3.D-vanilla |
| S3.D-b16 | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-b16 z | `joint_acc` | — | same as S3.D-vanilla |
| S3.D-fvae | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-fvae z | `joint_acc` | — | same as S3.D-vanilla |
| S3.D-sepvae | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-sepvae z | `joint_acc` | — | same as S3.D-vanilla |
| S3.D-adagvae | C3b | conditioned VAE | S3.D-vanilla | FiLM-DDPM conditioned on S3.V-adagvae z — weakly supervised upper bound | `joint_acc` | `joint_acc(S3.D-adagvae) ≤ joint_acc(S3.D-vanilla)` — **this falsifies C3b entirely**: disentanglement provides no benefit | same as S3.D-vanilla |

**Secondary metrics (all 3b rows):** `wall_acc`, `floor_acc` (marginal factor accuracies for PoE composition).
**Expected ordering:** `S3.D-adagvae ≥ S3.D-sepvae > S3.D-fvae ≥ S3.D-b8/b16 > S3.D-b4 > S3.D-vanilla`.

---

## Open Grafts

Not part of the primary falsification chain. Do not run until main ablation matrix is complete.

| graft_id | description | connection to main matrix | when to pursue |
|---|---|---|---|
| G-A | **Score-space PoE:** `ε̂ = ε(xₜ, z_wall) + ε(xₜ, z_floor) − ε(xₜ, ∅)`. Composes in diffusion score space rather than VAE latent space; does not require Gaussian posteriors or factored latents. | Alternative composition mechanism to all S3.D-* rows; compare `joint_acc` directly against S3.D-adagvae | After S3b completes; run on the single best-performing VAE variant only |
| G-B | **GECO reconstruction constraint** as an alternative to fixed β in S3.V-b*. Adaptively tunes β to maintain a target reconstruction loss, potentially finding a better operating point without a manual sweep. | Replacement for the β ∈ {4,8,16} grid in Stage 3a | If no clear winner emerges from S3.V-b4/b8/b16 or if all under-perform SepVAE |

---

## W&B Artifact Index

| artifact key | produced by | what it shows |
|---|---|---|
| `viz/posterior_geometry` | S1.θ*, S2.*, S3.D-* | Reference posteriors + PoE posterior + true joint as Gaussian ellipses (Stage 1/2) or 2D scatter projected onto top-2 factor dims (Stage 3) |
| `viz/poe_error_vs_theta` | S1.θ* | PoE error vs θ ∈ {0, π/8, π/4}; overlaid curve `|sin(2θ)|` for fit check |
| `viz/composition_error_vs_theta` | S2.* | Same curve per model variant; all three variants overlaid |
| `viz/latent_traversals` | S3.V-* | Vary one latent dim, fix all others, reconstruct — logged every N epochs |
| `viz/dci_importance_heatmap` | S3.V-* | (latent\_dims × factors) importance matrix as a logged image |
| `viz/factor_lda_histograms` | S3.V-* | 1D LDA projection per factor, coloured by factor class |
| `viz/wrong_head_probe_acc` | S3.V-sepvae | Correct-head vs wrong-head probe accuracy over training; wrong-head should fall to chance |
| `viz/poe_posterior_geometry` | S3.D-* | Reference + PoE + true joint in the 2D latent subspace most predictive of wall\_hue and floor\_hue |
| `metrics/mig` | S2.*, S3.V-* | Mutual Information Gap scalar |
| `metrics/dci_d` | S2.*, S3.V-* | DCI Disentanglement scalar |
| `metrics/joint_acc` | S2.*, S3.D-* | % of (c₁=α, c₂=β) pairs recovered within tolerance ε |
| `metrics/wall_acc` | S3.D-* | Marginal wall\_hue accuracy in PoE composition |
| `metrics/floor_acc` | S3.D-* | Marginal floor\_hue accuracy in PoE composition |
