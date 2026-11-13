# MASTER PLAN — SP-VAE

> **Scope root:** `/home/molef/PhD/Paper2/disentangled-vae-poe/plans/sp-vae/`
> **Parent scope:** [disentangled-vae-poe](../../MASTER_PLAN.md) — Stage 3a variant
> **W&B project:** `disentangled-vae-poe` | **tag:** `sp-vae`
> **Created:** 2026-06-16

---

## 1. Mission

Build SP-VAE (Regime D): a partitioned-representation VAE for 3D Shapes that
explicitly disentangles two target generative factors — wall_hue and floor_hue —
into dedicated 2D latent slots (z_d1, z_d2), while routing all remaining variation
(shape, scale, orientation) through a shared context slot (z_c). A single ConvVAE
encoder trunk with residual blocks and SE channel-attention fans out into three
independent posterior heads. The model is trained with a multi-objective loss
combining supervised slot assignment, GRL adversarial cross-suppression with a DANN
λ-ramp, cross-partition covariance orthogonality penalty, mutual information
penalty between slots, and per-slot VICReg variance guards. β weights are annealed
upward over the first 20% of training. The downstream use is **factor-addressed
composition**: substitute z_d1 from one image and z_d2 from another, decode, and
obtain a valid composite showing the expected combination.

---

## 2. Objectives

### O1 — Architecture

Implement the SP-VAE encoder-decoder pair: a ConvVAE trunk upgraded with ResBlocks
and SE channel-attention, fanning out into three independent Gaussian heads.
The decoder takes concat(z_c, z_d1, z_d2) and mirrors the encoder in reverse.
All three posterior distributions are diagonal Gaussians; reparameterisation is
applied independently per slot.

---

### O2 — Training Stability

Train SP-VAE to convergence on 3D Shapes without posterior collapse in any slot.
This requires: per-slot β weights annealed upward over the first 20% of training,
per-slot VICReg variance guards (std ≥ ε per dimension), and a DANN λ-ramp on
the GRL adversarial heads. No slot should degenerate to a point posterior at any
point during or after training.

---

### O3 — Slot Isolation

Confirm that z_d1 encodes wall_hue and z_d2 encodes floor_hue, with assignments
clean enough to pass linear probe thresholds, and that cross-slot leakage is below
the falsification threshold. Isolation is enforced by the loss stack; this objective
verifies the result holds at eval time.

---

### O4 — Factor-Addressed Composition

Demonstrate that slot substitution at inference produces visually correct composites:
encode images A and B, take z_d1 from A, z_d2 from B, combine with z_c, decode —
the result must exhibit A's wall_hue and B's floor_hue. Verify with a linear probe
on the generated composites.

---

## 3. Goals

### Stage 1 — Model and Loss Implementation

- [ ] **G1.1** Encoder trunk: `src/models/sp_vae/encoder.py`
  - ConvVAE backbone with residual blocks (same channel progression as existing
    ConvVAE)
  - SE channel-attention block after last ResBlock (reduction ratio r=16)
  - Output: flat feature vector before the head fan-out

- [ ] **G1.2** Three-head fan-out: `src/models/sp_vae/heads.py`
  - `ContextHead(in_dim, z_c_dim)` → (μ_c, log σ²_c)
  - `SlotHead(in_dim, z_d_dim)` → (μ_d, log σ²_d); instantiated twice
  - All heads are independent linear projections from the trunk output

- [ ] **G1.3** Decoder: `src/models/sp_vae/decoder.py`
  - Input: concat(z_c, z_d1, z_d2)
  - Mirror-ConvVAE transpose-conv stack; output 64×64 RGB

- [ ] **G1.4** Multi-objective loss: `src/models/sp_vae/loss.py`
  - ELBO per slot with per-slot β weights; β(t) = β_max · min(1, t / 0.2T)
  - Correct-head CE: CrossEntropy(MLP(μ_d1), wall_hue_label) +
    CrossEntropy(MLP(μ_d2), floor_hue_label)
  - Wrong-head GRL: GRL(MLP(μ_d1)) predicts floor_hue (gradient reversed);
    GRL(MLP(μ_d2)) predicts wall_hue; DANN λ-ramp
  - Cross-partition orth penalty: ‖Cov(z_d1, z_d2)‖²_F
  - MI penalty: Î(z_d1; z_d2) — closed-form Gaussian lower bound preferred over
    MINE for stability
  - Per-slot VICReg variance guard: Σ_i max(0, ε − std(z_di_batch))²

- [ ] **G1.5** SP-VAE model wrapper: `src/models/sp_vae/model.py`
  - `encode(x)` → (z_c, z_d1, z_d2)
  - `decode(z_c, z_d1, z_d2)` → x̂
  - `compose(z_c, z_d1_src, z_d2_src)` for slot substitution at eval

---

### Stage 2 — Training

- [ ] **G2.1** Config: `configs/shapes3d/sp_vae/default.yml`
  - z_c_dim, z_d1_dim=2, z_d2_dim=2, β_max per slot, λ_max (DANN), orth_weight,
    mi_weight, vicreg_ε
  - All loss weights exposed as sweep-ready parameters

- [ ] **G2.2** Training script: `scripts/train_sp_vae.py`
  - `--config`, `--seed` flags; one W&B run per call
  - Logs: per-slot KL, per-loss-term scalars, per-slot dimension std, CE accuracy
    per epoch

- [ ] **G2.3** β and λ schedulers: `src/training/schedulers.py`
  - `BetaAnneal(warmup_frac=0.2, beta_max)` and `DannLambda(warmup_frac=0.2, lambda_max)`
  - Shared ramp function; callable by training loop at each step

- [ ] **G2.4** Convergence check: at epoch 100, per-slot KL > 0.05 for all
  dimensions; logged to W&B; training run fails with a warning if any slot collapses

---

### Stage 3 — Evaluation

- [ ] **G3.1** Linear probe script: `scripts/eval_sp_vae_probes.py`
  - Encodes full 3D Shapes val set
  - Fits LogisticRegression on (z_d1 → wall_hue), (z_d2 → floor_hue),
    (z_d2 → wall_hue), (z_d1 → floor_hue)
  - Reports and W&B-logs all four accuracies

- [ ] **G3.2** MIG evaluation: `scripts/eval_disentanglement.py` (shared with
  parent scope)
  - Computes MIG for wall_hue and floor_hue from sp-vae checkpoint
  - Compared against β-VAE baseline MIG from parent Stage 3

- [ ] **G3.3** Latent traversal figure: `scripts/plot_sp_vae_traversals.py`
  - Vary z_d1[0], z_d1[1], z_d2[0], z_d2[1] independently; hold all others fixed
  - Saves `figures/sp_vae_traversals.png`; logs `viz/sp_vae_traversals`

- [ ] **G3.4** Slot variance figure: `scripts/plot_sp_vae_slot_variance.py`
  - Per-dimension std histogram across val set, per slot
  - Confirms no dimension std < 0.05
  - Saves `figures/sp_vae_slot_variance.png`

---

### Stage 4 — Composition

- [ ] **G4.1** Composition eval script: `scripts/eval_sp_vae_composition.py`
  - Grid of (wall_hue_a × floor_hue_b) pairs; encode references, substitute
    z_d1 from A and z_d2 from B, decode
  - Saves `figures/sp_vae_composition_grid.png`
  - Logs `viz/sp_vae_composition`

- [ ] **G4.2** Composition probe: encode each generated composite; probe z_d1 for
  wall_hue (should match A) and z_d2 for floor_hue (should match B)
  - Reports `composition_wall_acc` and `composition_floor_acc`

---

## 4. Sub-Scopes

_None at creation time._

## 5. Plans

| File | Stage | Objectives | Status |
| --- | --- | --- | --- |
| [01-architecture.md](plans/01-architecture.md) | 1 | O1 | ⚠️ not started |
| [02-loss-function.md](plans/02-loss-function.md) | 1 | O2 | ⚠️ not started |
| [03-training.md](plans/03-training.md) | 2 | O2 | ⚠️ not started |
| [04-evaluation.md](plans/04-evaluation.md) | 3 | O3 | ⚠️ not started |
| [05-composition.md](plans/05-composition.md) | 4 | O4 | ⚠️ not started |

---

## 6. Expected Outcomes

### If the full chain holds

> SP-VAE trains without posterior collapse in any slot. z_d1 achieves ≥ 90%
> linear probe accuracy for wall_hue; z_d2 achieves ≥ 90% for floor_hue.
> Cross-slot leakage stays below 20% in both directions. Slot substitution at
> inference produces visually correct composites. MIG scores for both hue factors
> exceed the β-VAE baseline. SP-VAE is ready to replace the tentative SepVAE
> entry in Stage 3a of the parent scope.

### Partial-failure contingencies

| Failure mode | What it means | How to handle |
| --- | --- | --- |
| Posterior collapse in z_d1 or z_d2 | VICReg guard or β-annealing insufficient | Tune vicreg_ε, lower β_max, or extend annealing window |
| Correct-slot probe < 90% | CE supervision insufficient | Increase CE loss weight; audit label quality; try z_d_dim=4 |
| Cross-slot leakage > 20% | GRL or orth penalty insufficient | Increase λ_max, orth_weight |
| Composition produces wrong hue | Slot substitution does not generalise | Audit decoder entanglement; add swap-consistency loss |
| MIG does not beat β-VAE | Global disentanglement not strictly better despite slot supervision | Report supervised vs unsupervised MIG separately; discuss |

---

## 7. Definition of Done

- [ ] SP-VAE trains to convergence on 3D Shapes; per-slot KL > 0.05 for all
  dimensions at epoch 100
- [ ] wall_hue linear probe accuracy on z_d1 alone ≥ 90%
- [ ] floor_hue linear probe accuracy on z_d2 alone ≥ 90%
- [ ] Cross-slot leakage: z_d2 → wall_hue probe < 20% and z_d1 → floor_hue probe < 20%
- [ ] `figures/sp_vae_composition_grid.png` saved; images visually correct
- [ ] `composition_wall_acc` ≥ 80% and `composition_floor_acc` ≥ 80%
- [ ] MIG for wall_hue and floor_hue exceeds β-VAE baseline (Stage 3 parent)
- [ ] All W&B runs tagged `sp-vae` under project `disentangled-vae-poe`
- [ ] `figures/sp_vae_traversals.png` and `figures/sp_vae_slot_variance.png` saved
