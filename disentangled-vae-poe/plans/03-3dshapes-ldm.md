# Plan 03 — 3D Shapes Latent Diffusion Pipeline

> **Scope:** disentangled-vae-poe | **Stage:** 3 | **Objective:** O3
> **Goals covered:** G3.1–G3.10
> **Prerequisites:** Plans 01 and 02 complete;
> `src/metrics/poe.py`, `src/metrics/probes.py` available
> **W&B project:** disentangled-vae-poe | **tag:** stage3

## Objective

Train a VAE family on 3D Shapes, train one FiLM-conditioned DDPM per VAE
variant, and demonstrate that Spearman ρ(MIG, joint_acc) > 0.7 across
variants, with Ada-GVAE achieving the highest joint_acc and vanilla VAE
the lowest.

## Context

Execute Stage 3a fully and verify disentanglement quality before beginning
Stage 3b — the DDPM training is expensive and should not start until MIG
and DCI-D rankings confirm the VAE ordering. SepVAE is a tentative target:
do not begin its training until the architecture design plan is resolved
(MASTER_PLAN §SepVAE Architecture Design).

---

## Tasks

### 3a — Dataset and VAE Family

- [ ] **T1** 3D Shapes loader (`src/datasets/shapes3d.py`)
  - Downloads and caches the DeepMind HDF5 (480 MB)
  - Returns `(img [3,64,64] float32, latents [6] int)` per sample
  - Factors: floor_hue(10), wall_hue(10), object_hue(10),
    scale(8), shape(4), orientation(15)
  - Standard 70/15/15 IID split for all variants except Ada-GVAE

- [ ] **T2** Paired iterator for Ada-GVAE (`src/datasets/shapes3d.py`, extend)
  - `PairedShapes3DDataset`: returns (x_a, x_b, shared_factors_mask)
    where exactly one factor differs between x_a and x_b
  - `shared_factors_mask` is a boolean vector indicating which of the 6
    factors are held constant in the pair

- [ ] **T3** Ada-GVAE loss (`src/losses/adagvae_loss.py`)
  - Implements the Ada-GVAE objective: KL loss zeroed on shared factors,
    full KL on the changed factor
  - Reference: Locatello et al. 2020 (Weakly-Supervised Disentanglement
    Without Compromises)

- [ ] **T4** VAE training script (`scripts/train_vae_shapes3d.py`)
  - `--variant {vanilla,beta4,beta8,beta16,factorvae,adagvae,sepvae}` flag
  - Logs every 5 epochs: `viz/latent_traversals`, `viz/dci_importance_heatmap`,
    `viz/factor_lda_histograms`, `viz/kl_spectrum`
  - Checkpoint to `checkpoints/shapes3d/vae/{variant}/best.pt`

- [ ] **T5** Disentanglement eval (`scripts/eval_disentanglement.py`)
  - Computes MIG and DCI-D per variant via `compute_dci()` from
    `src/metrics/dci.py` (reuse from parent scope)
  - Logs `metrics/mig` and `metrics/dci_d` to W&B
  - Saves `results/shapes3d/disentanglement_table.csv`
  - **Gate:** confirm MIG ranking matches expectation before proceeding to 3b.
    Expected order: Ada-GVAE ≥ SepVAE > FactorVAE > β-VAE variants > vanilla.
    If the ranking is inverted at the top or bottom, investigate before
    continuing.

---

### 3b — FiLM-Conditioned DDPM

- [ ] **T6** FiLM conditioning module (`src/models/film.py`)
  - `FiLMBlock(z_dim, h_dim)`: `γ, β = Linear(z)` split; no activation
  - `h' = (1 + γ(z)) · h + β(z)` applied at each resolution block

- [ ] **T7** FiLM-conditioned DDPM (`src/models/ddpm_film.py`)
  - U-Net backbone; FiLM blocks at every resolution replacing AdaGN
  - Accepts frozen VAE z vector as conditioning input

- [ ] **T8** DDPM training script (`scripts/train_ddpm_film.py`)
  - `--vae-checkpoint <path>` flag; VAE encoder frozen during DDPM training
  - One run per VAE variant (6 committed + SepVAE if promoted)
  - Checkpoint to `checkpoints/shapes3d/ddpm/{variant}/best.pt`
  - Logs reconstruction MSE every N epochs to W&B

---

### 3c — Composition Evaluation

- [ ] **T9** PoE composition for 3D Shapes (`scripts/eval_poe_shapes3d.py`)
  - `encode_reference_set` (from `src/metrics/poe.py`) encodes N=200
    images with fixed wall_hue=α, then N=200 with fixed floor_hue=β
  - Computes PoE z* analytically; conditions DDPM on z*; generates images
  - Factor values recovered via DCI linear probe on encoder output
  - `joint_acc`, `wall_acc`, `floor_acc` for all 10×10 wall×floor pairs
  - Saves `results/shapes3d/poe_eval_{variant}.json`
  - Logs W&B artifact: `viz/poe_posterior_geometry`

- [ ] **T10** Rank-correlation analysis (`scripts/compute_rank_correlation.py`)
  - Spearman ρ(MIG, joint_acc) and ρ(DCI-D, joint_acc) across all variants
  - Saves `results/shapes3d/rank_correlation.json`

- [ ] **T11** Results table + documentation (`docs/stage3_results.md`)
  - Table: variant × MIG × DCI-D × joint_acc × wall_acc × floor_acc
  - States ρ values; confirms or denies O3; applies contingency from
    MASTER_PLAN §Expected Outcomes if needed

---

## Engagement Instructions

```bash
# 3a: VAE family (run in parallel where compute allows)
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/train_vae_shapes3d.py --variant $variant
done

# Gate check — inspect MIG ranking before starting DDPMs
python scripts/eval_disentanglement.py

# 3b: FiLM-DDPMs
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/train_ddpm_film.py \
    --vae-checkpoint checkpoints/shapes3d/vae/$variant/best.pt
done

# 3c: Composition evaluation
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/eval_poe_shapes3d.py --variant $variant
done
python scripts/compute_rank_correlation.py
```

Confirm:
- Ada-GVAE achieves highest joint_acc; vanilla VAE achieves lowest
- DCI importance matrix for Ada-GVAE shows near-permutation-matrix pattern
  (one dominant latent dim per factor)
- Spearman ρ(MIG, joint_acc) > 0.7
- 12 W&B runs minimum (6 VAE + 6 DDPM) under `disentangled-vae-poe`,
  tag `stage3`; every VAE run has `viz/latent_traversals`,
  `viz/dci_importance_heatmap`, `viz/factor_lda_histograms` logged

## DoD Cross-Reference

| MASTER_PLAN DoD item | Task |
|---|---|
| `src/datasets/shapes3d.py` complete | T1 |
| All five VAE variants trained; W&B runs | T4 |
| Every VAE run: traversals, DCI heatmap, LDA histograms | T4 |
| FiLM-DDPM implemented and trained per variant | T6, T7, T8 |
| `joint_acc` table produced | T9, T11 |
| Ada-GVAE highest joint_acc; vanilla lowest | T9 (measured) |
| Spearman ρ(MIG, joint_acc) > 0.7 computed | T10 |
| W&B `viz/poe_posterior_geometry` per DDPM variant | T9 |
| `docs/stage3_results.md` | T11 |
| All runs tagged `disentangled-vae-poe` | T4, T8 |
