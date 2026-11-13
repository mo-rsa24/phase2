# 02 — SP-VAE Loss Function and Schedulers

## Section context

**Description:** Implement the full multi-objective loss for SP-VAE and the β / DANN-λ annealing schedulers. Six named terms on top of reconstruction: per-slot KL (×3), correct-head CE, wrong-head GRL, cross-partition orth penalty, Gaussian MI lower bound, and per-slot VICReg variance guard.
**Objective:** O2 — Training Stability: ensure convergence without posterior collapse via controlled annealing and cross-slot suppression.
**Goal:** `SPVAELoss` passes a forward-pass test returning all 12 named keys; schedulers return correct scalar values at step 0 / mid-warmup / post-warmup; `configs/shapes3d/sp_vae/default.yml` present and valid.
**Verify:** Unit test asserting `EXPECTED == set(terms.keys())` for `{"recon","kl_c","kl_d1","kl_d2","ce_wall","ce_floor","grl_wall","grl_floor","orth","mi","vicreg","total"}`, scheduler values at steps 0/100/200 out of 1000, and yaml config loads with `z_c_dim` and `beta_d_max` present.

**Reference (open beside you):** plan `disentangled-vae-poe/plans/sp-vae/plans/02-loss-function.md` · code `disentangled-vae-poe/src/models/sp_vae/loss.py` · `disentangled-vae-poe/src/training/schedulers.py`

## Tasks

- T1: Write `src/training/schedulers.py` — BetaAnneal and DannLambda classes, each with `get(step, total_steps)` → linear ramp to max over warmup_frac; create `src/training/__init__.py`
- T2: GRL layer in `src/models/sp_vae/loss.py` — GRLFunction (autograd.Function: forward=identity, backward multiplies by −λ), GRLLayer (nn.Module wrapping it, λ as buffer updated each call)
- T3: SlotClassifier in same file — small MLP (Linear(z_d_dim,64) → ReLU → Linear(64,n_classes)); shared class used by both CE and GRL heads
- T4: SPVAELoss (nn.Module) — constructor takes n_wall/floor classes + loss weights; forward takes fwd_out + labels + beta_c/d + dann_lambda; computes all 6 term groups; returns dict of 12 named scalars including "total"
- T5: Write `configs/shapes3d/sp_vae/default.yml` — z_c_dim:8, z_d_dim:2, beta_c_max:1.0, beta_d_max:4.0, dann_lambda_max:1.0, warmup_frac:0.2, orth_weight:1.0, mi_weight:0.1, ce_weight:1.0, grl_weight:1.0, vicreg_eps:1.0, lr:1e-4, batch_size:64, epochs:100, seed:42
