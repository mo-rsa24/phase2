# 03 — SP-VAE Training Run

## Section context

**Description:** Wire SP-VAE into `train_vae_shapes3d.py` as a first-class variant and run the full 100-epoch training on 3D Shapes. Adds `spvae` to VARIANTS, a `_step_spvae()` function, β/λ scheduler stepping per batch, W&B logging of all named loss terms, and a convergence guard at epoch 100.
**Objective:** O2 — Training Stability: checkpoint at `checkpoints/shapes3d/vae/spvae/best.pt`; W&B run tagged `sp-vae`; per-slot KL > 0.05 for all dimensions at epoch 100.
**Goal:** Smoke test (5 epochs, no W&B) exits cleanly and writes checkpoint; full run completes without collapse warnings.
**Verify:** `python scripts/train_vae_shapes3d.py --variant spvae --epochs 5 --no-wandb --seed 42` — prints `[spvae] epoch 5/5 …` and writes `checkpoints/shapes3d/vae/spvae/best.pt`. Checkpoint sanity check: `ckpt["config"]["variant"] == "spvae"`. In W&B: `val/kl_d1_mean` and `val/kl_d2_mean` both > 0.05 at epoch 100.

**Reference (open beside you):** plan `disentangled-vae-poe/plans/sp-vae/plans/03-training.md` · code `disentangled-vae-poe/scripts/train_vae_shapes3d.py`

## Tasks

- T1: Add `"spvae"` to `VARIANTS` tuple in `scripts/train_vae_shapes3d.py`
- T2: Import SPVAE, SPVAELoss, BetaAnneal, DannLambda at top of script
- T3: Add `_step_spvae(model, loss_fn, batch, opt, beta_c, beta_d, dann_lambda, device)` — unpacks wall_labels=labels[:,2].long(), floor_labels=labels[:,3].long(); calls loss_fn; returns metric dict with `train/` prefix
- T4: Add spvae branch in `main()` — load default.yml config, instantiate SPVAE + SPVAELoss + schedulers, compute total_steps, dispatch in per-batch step block, add "sp-vae" to W&B tags
- T5: Convergence guard at epoch 100 — log val/kl_c_mean, val/kl_d1_mean, val/kl_d2_mean; print WARNING if any < 0.05
- T6: Run smoke test — 5 epochs, no W&B, seed 42
- T7: Run full training — 100 epochs, seed 42
