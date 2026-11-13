# 01 — SP-VAE Architecture

## Section context

**Description:** Implement the SP-VAE encoder-decoder stack — ConvVAE trunk upgraded with ResBlocks and SE channel-attention, fanning out into three independent Gaussian heads (ContextHead for z_c, two SlotHeads for z_d1 and z_d2), and a decoder taking their concatenation.
**Objective:** O1 — Architecture: build the partitioned-latent ConvVAE with ResBlock+SE trunk and three independent posterior heads.
**Goal:** `SPVAE` model class in `src/models/sp_vae/` with `encode`, `decode`, and `compose` methods passing the shape-check smoke test.
**Verify:** `python - <<'EOF'` smoke test — asserts `z_c (4,8)`, `z_d1 (4,2)`, `z_d2 (4,2)`, `x_hat (4,3,64,64)`, `comp (4,3,64,64)`, prints `PASS — shapes OK`.

**Reference (open beside you):** plan `disentangled-vae-poe/plans/sp-vae/plans/01-architecture.md` · code `disentangled-vae-poe/src/models/sp_vae/`

## Tasks

- T1: Create package `src/models/sp_vae/__init__.py` (empty)
- T2: Write `src/models/sp_vae/encoder.py` — ResBlock (two 3×3 Conv2d + BN + ReLU + 1×1 skip), SEBlock (avg-pool → fc → sigmoid, r=16), SPVAEEncoder (4-stage stride-2 conv, one ResBlock per stage, SE after stage 4, fc → 256-d feature)
- T3: Write `src/models/sp_vae/heads.py` — ContextHead (Linear → mu_c, lv_c), SlotHead (Linear → mu_d, lv_d); two SlotHeads instantiated for z_d1 and z_d2
- T4: Write `src/models/sp_vae/decoder.py` — SPVAEDecoder (concat z_c+z_d1+z_d2 → fc → 4× ConvTranspose2d stride-2 → Sigmoid; 64×64 RGB output)
- T5: Write `src/models/sp_vae/model.py` — SPVAE wrapping encoder + heads + decoder; expose encode / decode / compose / forward (returns 10-tuple)
