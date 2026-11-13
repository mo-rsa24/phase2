# Plan 02 — SP-VAE Loss Function and Schedulers

> **Scope:** sp-vae | **Stage:** 1 | **Objective:** O2
> **Goals covered:** G1.4 G2.1 G2.3
> **W&B project:** disentangled-vae-poe | **tag:** sp-vae

## Objective

Implement the full multi-objective loss for SP-VAE and the β / DANN-λ schedulers that
control its annealing schedule. Deliverable: `SPVAELoss` passes a forward-pass unit
test returning a dict of all named loss terms; schedulers return correct scalar values
at step 0, mid-warmup, and post-warmup; config file is present and valid YAML.

## Context

The loss has six named terms on top of reconstruction:

| Term | Purpose |
|---|---|
| KL per slot (×3) | ELBO regularisation, β-weighted |
| Correct-head CE | Push μ_d1 → wall_hue, μ_d2 → floor_hue |
| Wrong-head GRL | Suppress μ_d1 from predicting floor_hue and vice versa |
| Orth penalty | ‖Cov(z_d1, z_d2)‖²_F — batch-level cross-slot decorrelation |
| MI penalty | Gaussian closed-form Î(z_d1; z_d2) — lower bound on MI |
| VICReg guard | max(0, ε − std(z_di))² per slot dimension, summed over batch |

β_max per slot is annealed linearly to β_max over the first 20% of steps; DANN λ
uses the same ramp.

## Tasks

- [ ] **T1** Schedulers: `src/training/schedulers.py`
  - `class BetaAnneal` — `get(step, total_steps)` → `beta_max * min(1, step /
    (warmup_frac * total_steps))`
  - `class DannLambda` — same ramp but for λ; separate class for clarity
  - Create `src/training/__init__.py` if it does not exist

- [ ] **T2** GRL layer: top of `src/models/sp_vae/loss.py`
  - `class GRLFunction(torch.autograd.Function)` — forward returns identity,
    backward multiplies grad by −λ
  - `class GRLLayer(nn.Module)` — wraps GRLFunction, stores λ as a buffer
    updated each forward call

- [ ] **T3** Slot classifier (shared by CE and GRL heads):
  - `class SlotClassifier(nn.Module)` — `Linear(z_d_dim, hidden) → ReLU →
    Linear(hidden, n_classes)`; hidden=64 is sufficient for 2D input

- [ ] **T4** Loss class: `class SPVAELoss(nn.Module)` in `src/models/sp_vae/loss.py`
  - Constructor takes `n_wall_classes=10, n_floor_classes=10, vicreg_eps=1.0,
    orth_weight=1.0, mi_weight=1.0, ce_weight=1.0, grl_weight=1.0`
  - `forward(fwd_out, x, wall_labels, floor_labels, beta_c, beta_d, dann_lambda)`
    where `fwd_out` is the tuple from `SPVAE.forward`
  - Reconstruction: MSE / batch
  - KL_c: `-0.5*(1 + lv_c - mu_c.pow(2) - lv_c.exp()).sum(1).mean()`
  - KL_d1, KL_d2: same, independently
  - CE_wall: `CrossEntropy(classifier_d1(mu_d1), wall_labels)`
  - CE_floor: `CrossEntropy(classifier_d2(mu_d2), floor_labels)`
  - GRL_wall: `CrossEntropy(classifier_grl_d1(GRL(z_d1)), floor_labels)`
    — gradient reversed, suppresses floor_hue information in z_d1
  - GRL_floor: `CrossEntropy(classifier_grl_d2(GRL(z_d2)), wall_labels)`
  - Orth: `C = (z_d1 - z_d1.mean(0)).T @ (z_d2 - z_d2.mean(0)) / (B-1)`;
    loss = `(C**2).sum()`
  - MI (Gaussian lower bound): estimate via `H(z_d1) + H(z_d2) - H([z_d1, z_d2])`
    using diagonal Gaussian entropy `0.5 * sum(1 + log_var)` per slot, with the
    joint log_var estimated from the batch covariance diagonal
  - VICReg guard: `sum_i max(0, eps - std(z_d1[:, i]))^2 + same for z_d2`
  - Total: `recon + beta_c*KL_c + beta_d*(KL_d1+KL_d2) + ce_weight*(CE_wall+CE_floor)
    + grl_weight*(GRL_wall+GRL_floor) + orth_weight*Orth + mi_weight*MI + VICReg`
  - Returns `dict` of all named scalars for W&B logging

- [ ] **T5** Config: `configs/shapes3d/sp_vae/default.yml`
  - `z_c_dim: 8`, `z_d_dim: 2`, `beta_c_max: 1.0`, `beta_d_max: 4.0`,
    `dann_lambda_max: 1.0`, `warmup_frac: 0.2`, `orth_weight: 1.0`,
    `mi_weight: 0.1`, `ce_weight: 1.0`, `grl_weight: 1.0`, `vicreg_eps: 1.0`,
    `lr: 1.0e-4`, `batch_size: 64`, `epochs: 100`, `seed: 42`

## Engagement Instructions

```bash
cd /home/molef/PhD/Paper2/disentangled-vae-poe
python - <<'EOF'
import torch, yaml
from src.models.sp_vae.model import SPVAE
from src.models.sp_vae.loss import SPVAELoss
from src.training.schedulers import BetaAnneal, DannLambda

model   = SPVAE(z_c_dim=8, z_d_dim=2)
loss_fn = SPVAELoss()
b_sch   = BetaAnneal(warmup_frac=0.2, beta_max=4.0)
d_sch   = DannLambda(warmup_frac=0.2, lambda_max=1.0)

x            = torch.randn(4, 3, 64, 64)
wall_labels  = torch.randint(0, 10, (4,))
floor_labels = torch.randint(0, 10, (4,))

fwd  = model(x)
beta = b_sch.get(step=200, total_steps=1000)
lam  = d_sch.get(step=200, total_steps=1000)
terms = loss_fn(fwd, x, wall_labels, floor_labels,
                beta_c=1.0, beta_d=beta, dann_lambda=lam)

EXPECTED = {"recon","kl_c","kl_d1","kl_d2","ce_wall","ce_floor",
            "grl_wall","grl_floor","orth","mi","vicreg","total"}
assert EXPECTED == set(terms.keys()), f"Missing: {EXPECTED - set(terms.keys())}"
print("PASS — all loss terms present")
print({k: round(v.item(), 4) for k, v in terms.items()})

assert b_sch.get(0, 1000)   == 0.0
assert b_sch.get(200, 1000) == 4.0   # 200 >= 0.2*1000 → fully ramped
assert b_sch.get(100, 1000) == 2.0   # halfway through warmup
print("PASS — scheduler values OK")

cfg = yaml.safe_load(open("configs/shapes3d/sp_vae/default.yml"))
assert "z_c_dim" in cfg and "beta_d_max" in cfg
print("PASS — config valid")
EOF
```

## Recommended skill

— custom; no skill fits (project-specific multi-term loss derivation).
