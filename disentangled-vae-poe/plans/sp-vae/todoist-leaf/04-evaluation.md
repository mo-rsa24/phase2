# 04 — SP-VAE Disentanglement Evaluation

## Section context

**Description:** Verify slot isolation quantitatively and visually. Four artefacts: a CSV with all four probe accuracies, a MIG comparison vs β-VAE, a latent traversal figure, and a slot-variance figure.
**Objective:** O3 — Slot Isolation: confirm z_d1 encodes wall_hue (≥90% probe) and z_d2 encodes floor_hue (≥90% probe) with cross-slot leakage below 20% in both directions.
**Goal:** `results/shapes3d/spvae_disentanglement.csv` written; traversal and slot-variance figures saved; MIG for both hue factors exceeds β-VAE baseline.
**Verify:** `python scripts/eval_sp_vae_probes.py --checkpoint checkpoints/shapes3d/vae/spvae/best.pt` prints `acc_d1_wall ≥ 0.90 [PASS]`, `acc_d2_floor ≥ 0.90 [PASS]`, `acc_d2_wall < 0.20 [PASS]`, `acc_d1_floor < 0.20 [PASS]` and writes CSV. `figures/sp_vae_traversals.png` and `figures/sp_vae_slot_variance.png` exist after running plot scripts.

**Reference (open beside you):** plan `disentangled-vae-poe/plans/sp-vae/plans/04-evaluation.md` · code `disentangled-vae-poe/scripts/eval_sp_vae_probes.py`

## Tasks

- T1: Write `scripts/eval_sp_vae_probes.py` — encode full val set, fit 4 LogisticRegression probes (z_d1→wall, z_d2→floor, z_d2→wall, z_d1→floor), print + W&B log all four, write `results/shapes3d/spvae_disentanglement.csv`
- T2: MIG evaluation — add spvae branch to `scripts/eval_disentanglement.py` (or write standalone); compute MIG over concat(z_d1,z_d2); compare vs β-VAE baseline; append to CSV
- T3: Write `scripts/plot_sp_vae_traversals.py` — encode val_ds[0], traverse z_d1[0/1] and z_d2[0/1] in 8 steps (−3σ to +3σ), save `figures/sp_vae_traversals.png`, log `viz/sp_vae_traversals`
- T4: Write `scripts/plot_sp_vae_slot_variance.py` — encode 5000 val samples, plot per-dim std histograms per slot with 0.05 collapse threshold line, save `figures/sp_vae_slot_variance.png`
