# 05 — SP-VAE Composition Test

## Section context

**Description:** Demonstrate factor-addressed composition: encode two reference images, swap z_d1 and z_d2, decode, and verify the composite has the correct hue combination via a linear probe on the generated composites.
**Objective:** O4 — Factor-Addressed Composition: slot substitution at inference produces visually correct composites; composition_wall_acc ≥ 80% and composition_floor_acc ≥ 80%.
**Goal:** `figures/sp_vae_composition_grid.png` saved; composition probe metrics ≥ 80% printed and W&B-logged.
**Verify:** `python scripts/eval_spvae_composition.py --checkpoint checkpoints/shapes3d/vae/spvae/best.pt` prints `composition_wall_acc ≥ 0.80 [PASS]` and `composition_floor_acc ≥ 0.80 [PASS]`; `figures/sp_vae_composition_grid.png` exists. Visual inspection: rows share wall_hue, columns share floor_hue, off-diagonal cells show both correctly.

**Reference (open beside you):** plan `disentangled-vae-poe/plans/sp-vae/plans/05-composition.md` · code `disentangled-vae-poe/scripts/eval_spvae_composition.py`

## Tasks

- T1: Write `scripts/eval_spvae_composition.py` — build factor-indexed val lookup, select 4×4 grid (wall_hue ∈ {0,3,6,9}, floor_hue ∈ {0,3,6,9}), encode ref_A and ref_B per cell, call model.compose(z_c_a, z_d1_a, z_d2_b), save grid + log viz/sp_vae_composition
- T2: Composition probe (inside same script) — re-fit probe_wall (z_d1→wall_hue) and probe_floor (z_d2→floor_hue) on val set; apply to each composite's slots; report composition_wall_acc and composition_floor_acc; W&B log metrics/composition_wall_acc and metrics/composition_floor_acc
