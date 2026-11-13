# Plan 01 — SP-VAE Architecture

> **Scope:** sp-vae | **Stage:** 1 | **Objective:** O1
> **Goals covered:** G1.1 G1.2 G1.3 G1.5
> **W&B project:** disentangled-vae-poe | **tag:** sp-vae

## Objective

Implement the SP-VAE encoder-decoder stack: a ConvVAE trunk upgraded with ResBlocks
and SE channel-attention, fanning out into three independent Gaussian heads
(ContextHead for z_c, two SlotHeads for z_d1 and z_d2), and a decoder that takes
their concatenation. Deliverable: a `SPVAE` model class with `encode`, `decode`, and
`compose` methods that passes a shape-check smoke test.

## Context

The existing `src/models/conv_vae.py` is a single-head model (4 Conv → flat → one
linear projection to (μ, log σ²)). SP-VAE reuses the same 4-conv channel progression
(3→32→32→64→64) as its trunk, adds a ResBlock after each conv layer and an SE block
after the last, then fans out to three independent heads before the flat step. The
decoder is an unchanged mirror-ConvVAE.

The new code lives in `src/models/sp_vae/` — a new package, not a modification of
`conv_vae.py`. The existing ConvVAE is untouched; `train_vae_shapes3d.py` imports it
for all other variants.

## Tasks

- [ ] **T1** Create package: `src/models/sp_vae/__init__.py` (empty)

- [ ] **T2** Encoder trunk: `src/models/sp_vae/encoder.py`
  - `class ResBlock(nn.Module)` — two 3×3 Conv2d + BN + ReLU with a 1×1 skip
    projection when channel count changes
  - `class SEBlock(nn.Module)` — global avg-pool → Linear(C, C//16) → ReLU →
    Linear(C//16, C) → Sigmoid → channel-wise scale; r=16
  - `class SPVAEEncoder(nn.Module)` — 4-stage conv stack (3→32→32→64→64,
    stride-2 at each stage), one ResBlock per stage, SEBlock after stage 4,
    flatten to 64×4×4=1024, Linear(1024, 256) → ReLU → output feature vector

- [ ] **T3** Three-head fan-out: `src/models/sp_vae/heads.py`
  - `class ContextHead(nn.Module)` — `Linear(256, z_c_dim * 2)`, splits to
    (μ_c, log σ²_c)
  - `class SlotHead(nn.Module)` — `Linear(256, z_d_dim * 2)`, splits to
    (μ_d, log σ²_d); instantiated twice for z_d1 and z_d2

- [ ] **T4** Decoder: `src/models/sp_vae/decoder.py`
  - `class SPVAEDecoder(nn.Module)` — input dim = z_c_dim + z_d1_dim + z_d2_dim;
    Linear → Linear(256, 64×4×4) → reshape → 4× ConvTranspose2d stride-2 (mirrors
    ConvVAE channel order 64→64→32→32→3); Sigmoid on output

- [ ] **T5** Model wrapper: `src/models/sp_vae/model.py`
  - `class SPVAE(nn.Module)` — composes encoder, context head, two slot heads,
    decoder
  - `encode(x)` → `(z_c, z_d1, z_d2)` using reparameterisation per slot
  - `decode(z_c, z_d1, z_d2)` → x̂
  - `compose(z_c, z_d1_src, z_d2_src)` — concat and decode (no grad needed at eval)
  - `forward(x)` → `(x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2)`

## Engagement Instructions

```bash
cd /home/molef/PhD/Paper2/disentangled-vae-poe
python - <<'EOF'
import torch
from src.models.sp_vae.model import SPVAE
m = SPVAE(z_c_dim=8, z_d_dim=2)
x = torch.randn(4, 3, 64, 64)
z_c, z_d1, z_d2 = m.encode(x)
assert z_c.shape  == (4, 8),        z_c.shape
assert z_d1.shape == (4, 2),        z_d1.shape
assert z_d2.shape == (4, 2),        z_d2.shape
x_hat = m.decode(z_c, z_d1, z_d2)
assert x_hat.shape == (4, 3, 64, 64), x_hat.shape
comp  = m.compose(z_c, z_d1, z_d2)
assert comp.shape  == (4, 3, 64, 64), comp.shape
print("PASS — shapes OK")
EOF
```

## Recommended skill

▶ `/scaffold` ✅ — scaffold the `src/models/sp_vae/` package from the spec above.
