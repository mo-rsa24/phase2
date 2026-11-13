# Plan 04 — SP-VAE Disentanglement Evaluation

> **Scope:** sp-vae | **Stage:** 3 | **Objective:** O3
> **Goals covered:** G3.1 G3.2 G3.3 G3.4
> **W&B project:** disentangled-vae-poe | **tag:** sp-vae

## Objective

Verify slot isolation quantitatively and visually. Deliverable: four artefacts — a
CSV with all four probe accuracies, a MIG comparison vs β-VAE baseline, a latent
traversal figure, and a slot-variance figure — plus W&B log entries for each.

## Context

`SPVAE.encode(x)` returns `(z_c, z_d1, z_d2)` directly; probe scripts use these
slots independently. Factor index mapping: `wall_hue = labels[:, 2]`,
`floor_hue = labels[:, 3]`.

The MIG eval loads the sp-vae checkpoint and computes MIG over the concatenation of
z_d1 and z_d2 (z_c excluded — MIG is measured on the disentangled partition only).
The β-VAE baseline MIG values come from the parent-scope Stage 3 results stored in
`results/shapes3d/` or W&B.

## Tasks

- [ ] **T1** Write `scripts/eval_sp_vae_probes.py`
  - `--checkpoint` flag; loads SPVAE and the 3D Shapes val set
  - Encode full val set: collect `(z_d1, z_d2, wall_labels, floor_labels)` in batches
  - Fit four `LogisticRegression(max_iter=1000)` probes from sklearn:
    - `acc_d1_wall`: z_d1 → wall_hue (correct slot)
    - `acc_d2_floor`: z_d2 → floor_hue (correct slot)
    - `acc_d2_wall`: z_d2 → wall_hue (leakage probe)
    - `acc_d1_floor`: z_d1 → floor_hue (leakage probe)
  - Print all four accuracies with pass/fail markers (≥ 90% for correct, < 20%
    for leakage)
  - W&B log all four under current run or fresh `wandb.init` if none active
  - Write `results/shapes3d/spvae_disentanglement.csv` with columns `probe,accuracy`;
    create `results/shapes3d/` if absent

- [ ] **T2** MIG evaluation
  - Check if `scripts/eval_disentanglement.py` supports loading SPVAE; if not, add
    a branch that loads the checkpoint, encodes the val set, and concatenates
    (z_d1, z_d2) as the evaluated representation
  - Run for spvae and for the best β-VAE checkpoint from the parent scope
  - Append MIG_wall and MIG_floor for both variants to
    `results/shapes3d/spvae_disentanglement.csv`
  - Print a two-row comparison table

- [ ] **T3** Write `scripts/plot_sp_vae_traversals.py`
  - `--checkpoint` flag; loads SPVAE; encode `val_ds[0]` to get (z_c, z_d1, z_d2)
  - Traverse z_d1[0], z_d1[1], z_d2[0], z_d2[1] independently in 8 steps from
    −3σ to +3σ; hold z_c and the other slot's values fixed at their encoded mean
  - 4-row traversal grid; label rows with slot/dim names
  - Save `figures/sp_vae_traversals.png`
  - W&B log `viz/sp_vae_traversals`

- [ ] **T4** Write `scripts/plot_sp_vae_slot_variance.py`
  - `--checkpoint` flag; encode up to 5000 val samples
  - Plot per-dimension std histograms in 3 side-by-side subplots (z_c, z_d1, z_d2)
  - Draw a vertical red dashed line at std = 0.05 (collapse threshold) in each panel
  - Save `figures/sp_vae_slot_variance.png`

## Engagement Instructions

```bash
cd /home/molef/PhD/Paper2/disentangled-vae-poe
CKPT=checkpoints/shapes3d/vae/spvae/best.pt

python scripts/eval_sp_vae_probes.py --checkpoint $CKPT
# Must print: acc_d1_wall  >= 0.90  [PASS]
# Must print: acc_d2_floor >= 0.90  [PASS]
# Must print: acc_d2_wall  <  0.20  [PASS]
# Must print: acc_d1_floor <  0.20  [PASS]
# Must write: results/shapes3d/spvae_disentanglement.csv

python scripts/plot_sp_vae_traversals.py --checkpoint $CKPT
# Check: figures/sp_vae_traversals.png exists
# Visual: z_d1 rows should show a clean wall_hue colour sweep;
#         z_d2 rows should show a clean floor_hue colour sweep

python scripts/plot_sp_vae_slot_variance.py --checkpoint $CKPT
# Check: figures/sp_vae_slot_variance.png exists
# Visual: no bar in z_d1 or z_d2 panels crosses the 0.05 threshold line

# MIG comparison (adapt path if eval_disentanglement.py needs a --model flag):
python scripts/eval_disentanglement.py --checkpoint $CKPT --model spvae
# Must print: spvae MIG_wall > beta-VAE MIG_wall
# Must print: spvae MIG_floor > beta-VAE MIG_floor
```

## Recommended skill

▶ `/analyze-save` ✅ — post-checkpoint analysis sweep; invoke after T1–T4 are written.
   alt: `/analyze-run` if a live W&B run ID is preferred over a checkpoint path.
