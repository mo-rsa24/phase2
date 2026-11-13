# 03 · 3D Shapes Latent Diffusion Pipeline

[⌂ Index](00-INDEX.md) · [← 02](02-gaussian-numerical.md)

## Reference while you do it
- 📄 Plan: plans/03-3dshapes-ldm.md
- ⚠️ Prerequisite: Plans 01 + 02 complete (`src/metrics/poe.py` and `src/metrics/probes.py` must exist)
- ⚠️ SepVAE is TENTATIVE — do not begin training until architecture design plan is resolved (see MASTER_PLAN §SepVAE Architecture Design)
- 💻 Code: src/datasets/shapes3d.py · src/losses/adagvae_loss.py · src/models/film.py · src/models/ddpm_film.py
- 💻 Scripts: scripts/train_vae_shapes3d.py · scripts/eval_disentanglement.py · scripts/train_ddpm_film.py · scripts/eval_poe_shapes3d.py · scripts/compute_rank_correlation.py
- 📊 W&B: project `disentangled-vae-poe`, tag `stage3`

## Section context (paste into the Todoist section)
**Description:** Train a VAE family on 3D Shapes (460,800 images, 6 factors), train one FiLM-conditioned DDPM per VAE variant (frozen encoder), and show Spearman ρ(MIG, joint_acc) > 0.7 across variants. Execute Stage 3a fully and verify MIG ranking before starting Stage 3b — DDPM training is expensive and should not begin until the VAE ordering is confirmed.
**Objective:** Demonstrate that the disentanglement→PoE link is systematic across a family of VAEs (O3): not just Ada-GVAE > Vanilla, but a monotone rank correlation across all variants.
**Goal:** G3.1–G3.10: 3D Shapes loader + paired iterator + Ada-GVAE loss → VAE training script → disentanglement eval (GATE) → FiLM module + DDPM model → DDPM training → PoE composition eval → rank correlation → results doc.
**Verify (whole leaf):**
```
# 3a — VAE family
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/train_vae_shapes3d.py --variant $variant
done
# Gate check — inspect MIG ranking before starting DDPMs
python scripts/eval_disentanglement.py
# 3b — FiLM-DDPMs
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/train_ddpm_film.py \
    --vae-checkpoint checkpoints/shapes3d/vae/$variant/best.pt
done
# 3c — Composition evaluation
for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do
  python scripts/eval_poe_shapes3d.py --variant $variant
done
python scripts/compute_rank_correlation.py
```
Confirm: Ada-GVAE achieves highest `joint_acc`; vanilla VAE achieves lowest; DCI importance matrix for Ada-GVAE shows near-permutation-matrix pattern; Spearman ρ(MIG, joint_acc) > 0.7; 12 W&B runs minimum (6 VAE + 6 DDPM) under `disentangled-vae-poe`, tag `stage3`; every VAE run has `viz/latent_traversals`, `viz/dci_importance_heatmap`, `viz/factor_lda_histograms` logged.

## Tasks (one at a time)

**Sub-stage 3a — Dataset and VAE Family**
- T1 — 3D Shapes loader (`src/datasets/shapes3d.py`)
  - **Goal:** G3.1 — downloads and caches the DeepMind HDF5 (480 MB); returns `(img [3,64,64] float32, latents [6] int)` per sample; standard 70/15/15 IID split for all non-Ada-GVAE variants
  - **Verify:** Tested implicitly when `python scripts/train_vae_shapes3d.py --variant vanilla` starts without HDF5 import error and begins iterating batches
  - Downloads and caches the DeepMind HDF5 (480 MB)
  - Returns `(img [3,64,64] float32, latents [6] int)` per sample
  - Factors: floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)
  - Standard 70/15/15 IID split for all variants except Ada-GVAE
- T2 — Paired iterator for Ada-GVAE (`src/datasets/shapes3d.py`, extend)
  - **Goal:** G3.2 (Ada-GVAE variant) — `PairedShapes3DDataset` returning `(x_a, x_b, shared_factors_mask)` where exactly one factor differs between x_a and x_b
  - **Verify:** Tested implicitly when `python scripts/train_vae_shapes3d.py --variant adagvae` runs without AttributeError and begins training
  - `PairedShapes3DDataset`: returns (x_a, x_b, shared_factors_mask) where exactly one factor differs between x_a and x_b
  - `shared_factors_mask` is a boolean vector indicating which of the 6 factors are held constant in the pair
- T3 — Ada-GVAE loss (`src/losses/adagvae_loss.py`)
  - **Goal:** G3.2 (Ada-GVAE variant) — KL loss zeroed on shared factors, full KL on changed factor; reference: Locatello et al. 2020
  - **Verify:** Tested implicitly when `python scripts/train_vae_shapes3d.py --variant adagvae` completes a training step without shape mismatch error
  - Implements the Ada-GVAE objective: KL loss zeroed on shared factors, full KL on the changed factor
  - Reference: Locatello et al. 2020 (Weakly-Supervised Disentanglement Without Compromises)
- T4 — VAE training script (`scripts/train_vae_shapes3d.py`)
  - **Goal:** G3.2 + G3.3 — `--variant {vanilla,beta4,beta8,beta16,factorvae,adagvae,sepvae}` flag; logs every 5 epochs: `viz/latent_traversals`, `viz/dci_importance_heatmap`, `viz/factor_lda_histograms`, `viz/kl_spectrum`; checkpoint to `checkpoints/shapes3d/vae/{variant}/best.pt`
  - **Verify:** `for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do python scripts/train_vae_shapes3d.py --variant $variant; done` — 6 W&B runs appear; every run has `viz/latent_traversals`, `viz/dci_importance_heatmap`, `viz/factor_lda_histograms` logged by epoch 5
  - `--variant {vanilla,beta4,beta8,beta16,factorvae,adagvae,sepvae}` flag
  - Logs every 5 epochs: `viz/latent_traversals`, `viz/dci_importance_heatmap`, `viz/factor_lda_histograms`, `viz/kl_spectrum`
  - Checkpoint to `checkpoints/shapes3d/vae/{variant}/best.pt`
- T5 — Disentanglement eval (`scripts/eval_disentanglement.py`) ← GATE before 3b
  - **Goal:** G3.4 — computes MIG and DCI-D per variant via `compute_dci()` from `src/metrics/dci.py`; logs `metrics/mig` and `metrics/dci_d` to W&B; saves `results/shapes3d/disentanglement_table.csv`
  - **Verify:** `python scripts/eval_disentanglement.py` — MIG ranking matches expectation: Ada-GVAE ≥ FactorVAE > β-VAE variants > vanilla. If ranking is inverted at top or bottom, stop and investigate before proceeding to 3b.
  - Computes MIG and DCI-D per variant via `compute_dci()` from `src/metrics/dci.py` (reuse from parent scope)
  - Logs `metrics/mig` and `metrics/dci_d` to W&B
  - Saves `results/shapes3d/disentanglement_table.csv`
  - **Gate:** confirm MIG ranking matches expectation before proceeding to 3b. Expected order: Ada-GVAE ≥ SepVAE > FactorVAE > β-VAE variants > vanilla. If the ranking is inverted at the top or bottom, investigate before continuing.

**Sub-stage 3b — FiLM-Conditioned DDPM (requires 3a Gate passed)**
- T6 — FiLM conditioning module (`src/models/film.py`)
  - **Goal:** G3.5 — `FiLMBlock(z_dim, h_dim)`: `γ, β = Linear(z)` split; no activation; `h' = (1 + γ(z)) · h + β(z)` applied at each resolution block
  - **Verify:** Tested implicitly when `python scripts/train_ddpm_film.py --vae-checkpoint checkpoints/shapes3d/vae/vanilla/best.pt` runs without AttributeError on the FiLM module
  - `FiLMBlock(z_dim, h_dim)`: `γ, β = Linear(z)` split; no activation
  - `h' = (1 + γ(z)) · h + β(z)` applied at each resolution block
- T7 — FiLM-conditioned DDPM (`src/models/ddpm_film.py`)
  - **Goal:** G3.6 — U-Net backbone; FiLM blocks at every resolution replacing AdaGN; accepts frozen VAE z vector as conditioning input
  - **Verify:** Tested implicitly when `python scripts/train_ddpm_film.py --vae-checkpoint checkpoints/shapes3d/vae/vanilla/best.pt` completes a forward pass without shape mismatch
  - U-Net backbone; FiLM blocks at every resolution replacing AdaGN
  - Accepts frozen VAE z vector as conditioning input
- T8 — DDPM training script (`scripts/train_ddpm_film.py`)
  - **Goal:** G3.7 — `--vae-checkpoint <path>` flag; VAE encoder frozen; one run per VAE variant (6 committed + SepVAE if promoted); checkpoint to `checkpoints/shapes3d/ddpm/{variant}/best.pt`; logs reconstruction MSE every N epochs
  - **Verify:** `for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do python scripts/train_ddpm_film.py --vae-checkpoint checkpoints/shapes3d/vae/$variant/best.pt; done` — 6 DDPM W&B runs appear under `disentangled-vae-poe`, tag `stage3`
  - `--vae-checkpoint <path>` flag; VAE encoder frozen during DDPM training
  - One run per VAE variant (6 committed + SepVAE if promoted)
  - Checkpoint to `checkpoints/shapes3d/ddpm/{variant}/best.pt`
  - Logs reconstruction MSE every N epochs to W&B

**Sub-stage 3c — Composition Evaluation**
- T9 — PoE composition for 3D Shapes (`scripts/eval_poe_shapes3d.py`)
  - **Goal:** G3.8 — `encode_reference_set` with N=200 per hue value; PoE z* analytically; DDPM generation; `joint_acc`, `wall_acc`, `floor_acc` for all 10×10 wall×floor pairs; saves `results/shapes3d/poe_eval_{variant}.json`; logs W&B artifact `viz/poe_posterior_geometry`
  - **Verify:** `for variant in vanilla beta4 beta8 beta16 factorvae adagvae; do python scripts/eval_poe_shapes3d.py --variant $variant; done` — Ada-GVAE achieves highest `joint_acc`; vanilla achieves lowest; `viz/poe_posterior_geometry` logged per variant
  - `encode_reference_set` (from `src/metrics/poe.py`) encodes N=200 images with fixed wall_hue=α, then N=200 with fixed floor_hue=β
  - Computes PoE z* analytically; conditions DDPM on z*; generates images
  - Factor values recovered via DCI linear probe on encoder output
  - `joint_acc`, `wall_acc`, `floor_acc` for all 10×10 wall×floor pairs
  - Saves `results/shapes3d/poe_eval_{variant}.json`
  - Logs W&B artifact: `viz/poe_posterior_geometry`
- T10 — Rank-correlation analysis (`scripts/compute_rank_correlation.py`)
  - **Goal:** G3.9 — Spearman ρ(MIG, joint_acc) and ρ(DCI-D, joint_acc) across all variants; saves `results/shapes3d/rank_correlation.json`
  - **Verify:** `python scripts/compute_rank_correlation.py` — Spearman ρ(MIG, joint_acc) > 0.7 printed to stdout; `results/shapes3d/rank_correlation.json` saved
  - Spearman ρ(MIG, joint_acc) and ρ(DCI-D, joint_acc) across all variants
  - Saves `results/shapes3d/rank_correlation.json`
- T11 — Results table + documentation (`docs/stage3_results.md`)
  - **Goal:** G3.10 — table: variant × MIG × DCI-D × joint_acc × wall_acc × floor_acc; ρ values stated; explicit O3 verdict; contingency from MASTER_PLAN §Expected Outcomes applied if needed
  - **Verify:** ⚠️ No CLI check — confirm `docs/stage3_results.md` exists, has the full results table, states ρ values, and gives an explicit "O3 confirmed" or "O3 denied" verdict with contingency note
  - Table: variant × MIG × DCI-D × joint_acc × wall_acc × floor_acc
  - States ρ values; confirms or denies O3; applies contingency from MASTER_PLAN §Expected Outcomes if needed
