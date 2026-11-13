# Plan 03 — SP-VAE Training Run

> **Scope:** sp-vae | **Stage:** 2 | **Objective:** O2
> **Goals covered:** G2.2 G2.4
> **W&B project:** disentangled-vae-poe | **tag:** sp-vae

## Objective

Wire SP-VAE into `train_vae_shapes3d.py` as a first-class variant and run the full
100-epoch training on 3D Shapes. Deliverable: checkpoint at
`checkpoints/shapes3d/vae/spvae/best.pt`; W&B run tagged `sp-vae`; per-slot KL > 0.05
for all dimensions at epoch 100.

## Context

`scripts/train_vae_shapes3d.py` already has a `sepvae` branch that uses β-TC-VAE as
a placeholder. Add `spvae` as a separate variant — do not rename or replace `sepvae`,
which is still needed as a baseline in the parent scope. The spvae variant needs:

1. A new `_step_spvae()` function analogous to `_step_adagvae()`, calling `SPVAELoss`
2. β and λ schedulers instantiated in `main()` and stepped each batch
3. W&B logging of all named loss terms returned by `SPVAELoss`
4. A convergence guard that warns if any slot KL < 0.05 at epoch 100

The 3D Shapes loader (`src/datasets/shapes3d.py`) returns `(img, labels)` where
`labels` is a 6-vector of factor values. Factor index mapping: `wall_hue = labels[:, 2]`,
`floor_hue = labels[:, 3]` per `FACTOR_NAMES`.

## Tasks

- [ ] **T1** Add `"spvae"` to the `VARIANTS` tuple in `scripts/train_vae_shapes3d.py`

- [ ] **T2** Import new modules at the top of the script:
  ```python
  from src.models.sp_vae.model import SPVAE
  from src.models.sp_vae.loss import SPVAELoss
  from src.training.schedulers import BetaAnneal, DannLambda
  ```

- [ ] **T3** Add `_step_spvae(model, loss_fn, batch, opt, beta_c, beta_d, dann_lambda, device)`:
  - Unpack `x, labels = batch`
  - Derive `wall_labels = labels[:, 2].long()`, `floor_labels = labels[:, 3].long()`
  - `fwd = model(x.to(device))`
  - Call `loss_fn(fwd, x, wall_labels, floor_labels, beta_c=beta_c, beta_d=beta_d,
    dann_lambda=dann_lambda)`
  - `terms["total"].backward()`, `opt.step()`
  - Return metric dict with all term keys prefixed `train/` for epoch accumulation

- [ ] **T4** In `main()`, add the spvae model branch:
  - Load `configs/shapes3d/sp_vae/default.yml` when `--variant spvae`
  - Instantiate `SPVAE(z_c_dim=cfg["z_c_dim"], z_d_dim=cfg["z_d_dim"])`
  - Instantiate `SPVAELoss(...)`, `BetaAnneal(...)`, `DannLambda(...)`
  - Compute `total_steps = epochs * len(train_loader)`; pass to scheduler `.get()`
    at each batch step
  - Add `spvae` dispatch in the per-batch step block
  - Add `"sp-vae"` to the W&B `tags` list when variant is `spvae`

- [ ] **T5** Convergence guard at end of epoch 100:
  - Compute mean KL per slot from one pass over val loader
  - Log as `val/kl_c_mean`, `val/kl_d1_mean`, `val/kl_d2_mean`
  - Print `WARNING: posterior collapse in slot X (KL={v:.4f})` for any value < 0.05

- [ ] **T6** Smoke test — 5 epochs, no W&B:
  ```bash
  python scripts/train_vae_shapes3d.py --variant spvae --epochs 5 --no-wandb --seed 42
  ```

- [ ] **T7** Full training run:
  ```bash
  python scripts/train_vae_shapes3d.py --variant spvae --epochs 100 --seed 42
  ```

## Engagement Instructions

```bash
cd /home/molef/PhD/Paper2/disentangled-vae-poe

# Smoke test (should complete without errors)
python scripts/train_vae_shapes3d.py --variant spvae --epochs 5 --no-wandb --seed 42
# Expected output line: "[spvae] epoch   5/5  loss=... val=... best=..."
# Expected file: checkpoints/shapes3d/vae/spvae/best.pt

# Checkpoint sanity check after full run
python - <<'EOF'
import torch
ckpt = torch.load("checkpoints/shapes3d/vae/spvae/best.pt", map_location="cpu")
print("epoch:", ckpt["epoch"])
print("val_loss:", round(ckpt["val_loss"], 4))
assert ckpt["config"]["variant"] == "spvae"
print("PASS — checkpoint valid")
EOF
# In W&B: confirm val/kl_d1_mean and val/kl_d2_mean both > 0.05 at epoch 100
```

## Recommended skill

— custom; no skill fits (project-specific training integration).
