# Plan 05 — SP-VAE Composition Test

> **Scope:** sp-vae | **Stage:** 4 | **Objective:** O4
> **Goals covered:** G4.1 G4.2
> **W&B project:** disentangled-vae-poe | **tag:** sp-vae

## Objective

Demonstrate factor-addressed composition: encode two reference images, swap z_d1 and
z_d2, decode, and verify the composite has the correct hue combination. Deliverable:
`figures/sp_vae_composition_grid.png` plus `composition_wall_acc` and
`composition_floor_acc` ≥ 80% from a linear probe on the generated composites.

## Context

`SPVAE.compose(z_c, z_d1_src, z_d2_src)` wraps the swap-and-decode step. Reference
images are selected by factor value from the 3D Shapes val set: for each
(wall_hue_a, floor_hue_b) cell in a 4×4 grid, find one image with wall_hue=a (any
floor_hue) and one with floor_hue=b (any wall_hue), encode, swap, decode.

The composition probe re-fits LogisticRegression probes from the val set (same as
Plan 04 T1), then applies them to each composite's (z_d1, z_d2) to measure whether
the composite inherited the correct factor values from its source images.

## Tasks

- [ ] **T1** Write `scripts/eval_spvae_composition.py`
  - `--checkpoint` flag; loads SPVAE and 3D Shapes val set
  - Build a factor-indexed lookup: for each unique (wall_hue, floor_hue) pair in the
    val set, store one image; wall_hue has 10 values (0–9), floor_hue has 10 values
  - Select a 4×4 sub-grid: wall_hue ∈ {0, 3, 6, 9}, floor_hue ∈ {0, 3, 6, 9}
  - For each cell (a, b):
    - Encode ref_A (wall_hue=a) → (z_c_a, z_d1_a, z_d2_a)
    - Encode ref_B (floor_hue=b) → (z_c_b, z_d1_b, z_d2_b)
    - `composite = model.compose(z_c_a, z_d1_a, z_d2_b)` — z_c and z_d1 from A,
      z_d2 from B
  - Arrange 16 composites in a 4×4 image grid; label rows with wall_hue value,
    columns with floor_hue value
  - Save `figures/sp_vae_composition_grid.png`
  - W&B log `viz/sp_vae_composition`

- [ ] **T2** Composition probe (inside the same script or a `--probe` flag):
  - Encode all 16 composites → (z_c, z_d1, z_d2)
  - Re-fit `LogisticRegression` probes on val set:
    - `probe_wall`: z_d1 → wall_hue
    - `probe_floor`: z_d2 → floor_hue
  - Apply each probe to the corresponding composite slot
  - `composition_wall_acc` = fraction where predicted wall_hue == source A wall_hue
  - `composition_floor_acc` = fraction where predicted floor_hue == source B floor_hue
  - Print both with pass/fail markers (≥ 80%)
  - W&B log `metrics/composition_wall_acc` and `metrics/composition_floor_acc`

## Engagement Instructions

```bash
cd /home/molef/PhD/Paper2/disentangled-vae-poe

python scripts/eval_spvae_composition.py \
    --checkpoint checkpoints/shapes3d/vae/spvae/best.pt

# Console output must include:
#   composition_wall_acc  >= 0.80  [PASS]
#   composition_floor_acc >= 0.80  [PASS]

# File check:
python - <<'EOF'
import os
assert os.path.exists("figures/sp_vae_composition_grid.png"), "Grid not saved"
print("PASS — figures/sp_vae_composition_grid.png exists")
EOF

# Visual inspection of the grid:
#   - Each row should show images with the same wall colour (rows = wall_hue value)
#   - Each column should show images with the same floor colour (cols = floor_hue value)
#   - Off-diagonal cells should display both hues correctly combined
```

## Recommended skill

— custom; no skill fits (project-specific composition evaluation).
