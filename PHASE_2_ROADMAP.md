# Phase 2 Roadmap — Beyond the IID Baseline

> **Companion to** [PROJECT_GUIDE.md](PROJECT_GUIDE.md) (overall project context)
> and [EXPLORER_GUIDE.md](EXPLORER_GUIDE.md) (current browser features).
>
> **Purpose:** lay out the four remaining phases of work, with explicit
> code-change checklists per phase and the corresponding extensions the
> browser app needs to surface the results.

---

## Where we are

✅ **Phase 2a (IID baseline)** — Exp 1–4 (β-VAE family) and Exp 5
(FactorVAE) are trained on the IID split. The browser supports MIG,
correlation heatmap, KL spectrum, single-dim traversal modal,
factor-conditional histograms, 2D KL scatter with hover→original+recon.

✅ **Dataset infra** — [src/datasets/correlated_dsprites.py](src/datasets/correlated_dsprites.py)
already implements `make_correlated_split` and `make_heldout_pair_split`;
no changes needed there.

🟡 **Empty stubs awaiting implementation:**
| Path | Purpose | Phase |
|---|---|---|
| [src/metrics/dci.py](src/metrics/dci.py) | DCI metric | 2c |
| [src/metrics/recombination.py](src/metrics/recombination.py) | held-out recombination metrics | 2e |
| [src/metrics/probes.py](src/metrics/probes.py) | swap-loss / GRL probes for SepVAE | 2d |
| [src/models/sepvae.py](src/models/sepvae.py) | partitioned VAE | 2d |
| [src/models/grl.py](src/models/grl.py) | gradient reversal layer | 2d |
| [src/models/beta_vae.py](src/models/beta_vae.py) | (currently subsumed by `--beta` flag) | n/a |
| [scripts/eval_dci.py](scripts/eval_dci.py) | offline DCI evaluation script | 2c |

---

## At a glance

| Phase | Adds | Browser surface |
|---|---|---|
| **2b — Correlated split** | 5 new sweep cells (Exp 6–10) using `make_correlated_split('scale','orientation')`; `--split` flag in trainers | Split-aware run cards, provenance bar, side-by-side compare mode |
| **2c — DCI metric** | `compute_dci()` in `src/metrics/dci.py`; `/api/dci` endpoint | New "DCI" panel in Analysis tab (importance heatmap + per-factor D/C/I bars + click-to-traversal) |
| **2d — SepVAE** | Design note → `grl.py` → `sepvae.py` → trainer → config → Exp 11 | Partitioned latent panel (μ bars colour-coded by group); group-aware 2D scatter; swap-decoded gallery |
| **2e — Held-out split** | `--split heldout` flag; cells 12–16; `eval_recombination` metric | Held-out gallery (factor combos never seen at train); per-cell recon MSE bar |

**Cross-cutting browser evolution** (covers all phases):
- Multi-checkpoint **Compare tab**.
- Cross-experiment **Summary table** (3 splits × 5 architectures × MIG / DCI / TC / heldout-MSE).
- **Provenance bar** (split, hyperparameters, seed) on every loaded checkpoint.

---

## Phase 2b — Correlated split: the central RQ test

### Why this phase

The research question — *"do scale and orientation get encoded
independently when training data correlates them?"* — only becomes
testable when the model has been trained on correlated data. Phase 2a
gave you the *control* (IID); Phase 2b gives you the *experimental
condition*. Comparing the same architecture's metrics under both is the
direct evidence the thesis needs.

### Code changes

**1. Trainers** — add a `--split` flag to both
[scripts/train_vae.py](scripts/train_vae.py) and
[scripts/train_factorvae.py](scripts/train_factorvae.py):
```python
p.add_argument("--split", type=str, default="iid",
               choices=["iid", "correlated", "heldout"])
p.add_argument("--corr-factor-a", type=str, default="scale")
p.add_argument("--corr-factor-b", type=str, default="orientation")
p.add_argument("--corr-direction", type=str, default="positive",
               choices=["positive", "negative"])
```

In `main()`, branch on `args.split`:
```python
if args.split == "iid":
    train_idx, val_idx, _ = make_iid_split(dataset, ...)
elif args.split == "correlated":
    train_idx, val_idx, _ = make_correlated_split(
        dataset, factor_a=args.corr_factor_a,
        factor_b=args.corr_factor_b,
        correlation=args.corr_direction, ...)
elif args.split == "heldout":
    # Phase 2e — see below.
    ...
```

**2. Sweep** — extend `EXPERIMENTS` in
[scripts/sweep_disentanglement.py](scripts/sweep_disentanglement.py)
with cells 6–10 (mirroring 1–5 but with `"split": "correlated"`):

```python
{"id": 6, "trainer": "vae", "split": "correlated", "latent_dim": 10,
 "beta": 1.0,  "purpose": "baseline-corr",
 "notes": "Standard VAE on correlated (scale, orientation)"},
# ... cells 7, 8, 9, 10 mirroring Exps 2, 3, 4, 5 with correlated split
```

`build_train_argv` adds `--split` to the argv list. `run_name` and
checkpoint paths gain a split prefix:

```python
def run_name(exp, seed):
    split = exp.get("split", "iid")
    base  = "factorvae" if exp["trainer"] == "factor_vae" else "vae"
    hp    = f"gamma{exp['gamma']}" if exp["trainer"] == "factor_vae" else f"beta{exp['beta']}"
    return f"{split}_{base}_z{exp['latent_dim']}_{hp}_seed{seed}"
```

So Exp 6 lives at
`checkpoints/vae/correlated_vae_z10_beta1.0_seed42/best.pt`.

**3. Wandb tags** — add `split:correlated` so dashboards can filter.

**4. Bash launcher** — extend `EXP_TRAINER`, `EXP_SPLIT`, etc. arrays in
[scripts/launch_sweep.sh](scripts/launch_sweep.sh) for cells 6–10. New
node assignments:
- `--node hippo`: all 10 cells sequentially.
- `--node mscluster106 / 107`: parallel pairs across IID and correlated.

### Browser extensions for Phase 2b

**1. Split-aware Run cards.** The Select Run tab currently shows 5 cards
in a single grid; with 10 it becomes cluttered. Group cards into two
sections with a clear divider:
```
─── IID baseline (5 cards) ───
[Exp 1] [Exp 2] [Exp 3] [Exp 4] [Exp 5]
─── Correlated split (5 cards) ───
[Exp 6] [Exp 7] [Exp 8] [Exp 9] [Exp 10]
```
Each correlated card carries a small badge `corr(scale, orient)` so the
provenance is unambiguous.

**2. Provenance bar.** A thin strip below the navbar showing
`split: correlated · z=10 · β=1.0 · seed=42 · checkpoint: …`. The
information is in the boot data already (extend
`window.__BOOT__` with `split`, `corr_factor_a`, `corr_factor_b`); the
display is one new component.

**3. Side-by-side compare mode** (the biggest browser change in this
phase — but reusable for Phases 2c/d/e). New top-level tab **Compare**
that lets you select 2 checkpoints and render their metrics in pairs:
- KL spectrum bars overlaid (or two separate bar charts).
- Correlation heatmaps stacked.
- MIG / DCI scores side-by-side.
- Cached-encodings 2D scatter for both, identical (dim_x, dim_y, colour)
  controls.

The point: a researcher loads "Exp 1 (IID baseline)" and "Exp 6
(correlated baseline)" and *immediately sees* whether the same
architecture behaves differently under correlation.

Implementation sketch:
- New endpoint `GET /api/sample_metrics?ckpt=<path>` that returns a
  small JSON blob (KL spectrum, MIG, DCI, top-dim-per-factor) for any
  loaded-but-not-active checkpoint without disturbing main `_state`.
- Front-end: maintains an array of "compared" checkpoints (≤4) in
  memory; renders each metric panel once per checkpoint.
- A "Pin to Compare" button on the Run cards adds the current
  checkpoint to the comparison set.

### Verification (Phase 2b)

1. **Sweep dispatch** — `python scripts/sweep_disentanglement.py
   --experiment-id 6 --runtime hippo --node hippo --print-only` produces
   `python train_vae.py … --split correlated --corr-factor-a scale
   --corr-factor-b orientation --beta 1.0 …` and the right output dir.
2. **Single-cell smoke** — 1-epoch run on correlated split completes
   and writes a checkpoint at the new path. Confirm checkpoint loads in
   the explorer with the new provenance bar.
3. **Visual sanity check on the 2D scatter** — for an Exp 6 checkpoint,
   colour the scatter by scale and then by orientation. The two
   gradients should now run *along the same axis* if the model
   entangled them (the predicted failure mode); orthogonal gradients
   would mean the model resisted the correlation.
4. **Metric sanity check** — Exp 6's correlation heatmap should show
   higher off-diagonal entries on the (scale, orientation) row pair
   than Exp 1's.

---

## Phase 2c — DCI: the second-opinion metric

### Why this phase

A single MIG number is rarely enough for a thesis chapter. DCI
(Eastwood & Williams, ICLR 2018) gives three complementary numbers:

- **Disentanglement (D)** — *per latent dim*: how concentrated its
  importance is on a single factor.
- **Completeness (C)** — *per factor*: how concentrated its
  representation is on a single latent dim.
- **Informativeness (I)** — *per factor*: how well a regressor predicts
  the factor from the latents.

D and C address the two failure modes separately:
- High D, low C → each dim is clean but factors are *split* across multiple dims.
- Low D, high C → each factor lives in one dim, but each dim *also* carries other factors.
- Both high → axis-aligned, factor-faithful code.

This is the metric every disentanglement paper reports alongside MIG.

### Code changes

**1. `src/metrics/dci.py`** — implement `compute_dci(mu, factors)`:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy

def compute_dci(
    mu: np.ndarray,            # (N, latent_dim)
    factors: np.ndarray,       # (N, num_factors)
    *,
    factor_sizes: list[int],   # to skip constant factors
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """Eastwood & Williams (2018) DCI metric.

    Returns:
      {
        "importance":  (latent_dim, num_active_factors) array, normalised by row,
        "D":           (latent_dim,) per-latent disentanglement score,
        "D_overall":   weighted mean of D,
        "C":           (num_active_factors,) per-factor completeness,
        "C_overall":   mean of C,
        "I":           (num_active_factors,) per-factor informativeness,
        "I_overall":   mean of I,
        "factor_names": list of factor names corresponding to columns,
      }
    """
    # 1. Train a Random Forest per factor: predict factor from mu.
    # 2. Use feature_importances_ to build the importance matrix R (latent_dim × num_factors).
    # 3. Compute D[i] = 1 - H(P_i) / log(num_factors) where P_i = R[i] / R[i].sum().
    # 4. Compute C[k] = 1 - H(P_k) / log(latent_dim) where P_k = R[:, k] / R[:, k].sum().
    # 5. Compute I[k] = 1 - prediction error of the regressor on a held-out split.
    # 6. Compute D_overall as the importance-weighted mean of D.
```

(The full implementation is ~120 lines; the scipy/sklearn helpers do
the heavy lifting.)

**2. `scripts/eval_dci.py`** — standalone script that loads a
checkpoint, computes DCI on the IID test split, prints the table, and
saves a JSON to `<ckpt-dir>/dci.json`. Useful for sweep-wide DCI
analysis without launching the explorer.

**3. Explorer endpoint** — add to
[scripts/disentanglement_explorer.py](scripts/disentanglement_explorer.py):

```python
@app.route("/api/dci", methods=["POST"])
def api_dci():
    """Background-thread DCI computation; same idle/computing/done/error
    state machine as MIG, since this also takes ~30 s.
    """
    # ... mirror the MIG endpoint pattern
```

Reuses cached `_state["enc_mu"]` and `_state["enc_factors"]`; no
re-encoding.

### Browser extensions for Phase 2c

A new **DCI** panel in the Analysis tab, mirroring the MIG panel's
shape:

**1. Importance heatmap.** Shape (latent_dim, num_factors); same
visual grammar as the existing |Spearman ρ| heatmap. Sorted by KL
descending. **Click a cell** opens the same 9-frame traversal modal as
the correlation heatmap (the click handler is already there for that
panel — reuse it).

**2. Per-latent D bars.** One row per latent dim (sorted by KL),
colour-graded:
- 🟢 D ≥ 0.5 (single-factor)
- 🟡 0.25 ≤ D < 0.5 (mostly one factor, some leakage)
- 🔴 D < 0.25 (multiplexing).

**3. Per-factor C and I bars.** Two parallel bar charts. Highlight
`scale` and `orientation` rows in blue (the RQ factors). Use the same
green/amber/red conventions as MIG (≥ 0.4, ≥ 0.2, < 0.2).

**4. Click-MIG-style integration.** Click a factor's C bar → conditional
histogram modal (re-using the existing `/api/factor_conditional_histogram`
endpoint and modal infrastructure). The C bar tells you "this factor
should live in one dim"; the histogram shows whether it actually does.

**5. New help-icon slugs.** Add `dci_d`, `dci_c`, `dci_i`,
`importance` to the HELP dict in
[scripts/explorer_help.py](scripts/explorer_help.py). Each with its
KaTeX formula:
- `D_i = 1 − H(P_i) / log K` where P_i is row-normalised importance.
- `C_k = 1 − H(P_k) / log d` where P_k is column-normalised.
- `I_k = 1 − E_test[(\hat v_k − v_k)^2] / Var(v_k)`.

### Verification (Phase 2c)

1. **Importance ranges** — every row of `R` sums to 1 (within FP
   tolerance); same for columns when normalised.
2. **Score ranges** — D, C, I all in [0, 1].
3. **Sanity check on a vanilla VAE** — the column corresponding to
   `color` (size=1) is correctly excluded; if included, D would be
   undefined.
4. **Reproducibility** — `compute_dci(...)` with same seed and same
   inputs is deterministic to within RF non-determinism (set
   `random_state=seed`).
5. **Cross-validation against MIG** — for a clean disentangled run
   (e.g. β=4 if it lands well), DCI's `D_overall` and MIG should both
   be in the moderate-to-high range; for a vanilla VAE both should be
   low. They should *correlate* across the 5 IID checkpoints, even if
   they don't agree exactly.

---

## Phase 2d — SepVAE: design-then-build

### Why this phase

SepVAE is the model that's *meant to succeed* on the correlated split
where vanilla / β / FactorVAE all fail. It hard-codes the
disentanglement target into the architecture by **partitioning the
latent space**:

```
z = [ z_common, z_scale, z_orient ]
```

…and adding losses that:
1. Force `z_scale` to be informative about the scale factor.
2. Force `z_orient` to be informative about the orientation factor.
3. Use a **gradient reversal layer (GRL)** to make `z_scale`
   *uninformative* about orientation (and vice versa) — this is the
   key adversarial bit that suppresses leakage.

Per [parking_lot.md](parking_lot.md), this work is meant to start with
a **design-thinking phase** before implementation:
*failure → mathematical target → architecture/loss*.

### Step 1 — Design note

Before writing code, produce
[notes/design_logic_correlated_dsprites.md](notes/design_logic_correlated_dsprites.md)
with the structure parking_lot.md specifies:

```markdown
# Design Logic: Correlated dSprites

## Failure
Correlated training data lets the encoder merge scale and orientation
into one joint explanation.

## Mathematical Target
I(z_scale; v_scale)   high
I(z_scale; v_orient)  ≈ 0
I(z_orient; v_orient) high
I(z_orient; v_scale)  ≈ 0

## Architecture
z = [z_common (k_c dims), z_scale (k_s dims), z_orient (k_o dims)]

## Losses
1. Reconstruction (BCE)
2. KL (each sub-block to N(0, I))
3. Correct-head prediction:  CE(predict v_scale  | z_scale )  +
                              CE(predict v_orient | z_orient)
4. GRL wrong-head loss:       −α · CE(predict v_orient | z_scale )
                              −α · CE(predict v_scale  | z_orient)
5. Optional swap loss: decode swapped z's and check factor recovery.

## Main metric
Held-out recombination accuracy + DCI(D, C).
```

This goes in `notes/` as a Zettelkasten-style note (not source) — the
design is the deliverable of this step.

### Step 2 — `src/models/grl.py`

```python
import torch
from torch.autograd import Function

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)
```

### Step 3 — `src/models/sepvae.py`

```python
class SepVAE(nn.Module):
    def __init__(self, k_common=4, k_scale=3, k_orient=3, img_size=...):
        # encoder (shared), then split heads:
        #   mu_common,  logvar_common  (k_common dims)
        #   mu_scale,   logvar_scale   (k_scale dims)
        #   mu_orient,  logvar_orient  (k_orient dims)
        # decoder takes the concatenation
        # right-head probes:  z_scale  → predict v_scale (multi-class)
        # right-head probes:  z_orient → predict v_orient (multi-class)
        # wrong-head probes (GRL): z_scale → predict v_orient
        #                          z_orient → predict v_scale
        ...

    def forward(self, x, v_scale=None, v_orient=None, return_components=False):
        # returns (x_hat, mu_blocks, logvar_blocks, z_blocks, head_logits) when return_components
        ...
```

Critical compat constraint from earlier in the project:
**`encoder.encode(x)` and the saved `encoder_state_dict` must remain
loadable by the explorer's `infer_encoder_config`**. Two options:

- **Option A (simpler)**: SepVAE saves the *concatenated* μ/logvar
  under the same `mu.weight`, `logvar.weight`, `encoder.0.weight`,
  `decoder.0.weight`, `decoder.11.weight` keys. The explorer sees a
  flat 10-d latent and doesn't know about partitioning. Partition
  metadata is stored separately in `config["partition"] = [4, 3, 3]`.
- **Option B (more honest)**: a separate explorer adapter that knows
  about partitioned models.

**Recommend Option A** — it keeps all existing tooling working; the
explorer's *new* SepVAE-aware viz reads `config["partition"]` from the
checkpoint and groups dims accordingly.

### Step 4 — `scripts/train_sepvae.py`

Mirrors `train_factorvae.py`'s structure:
- Single optimiser (no adversarial alternation needed — GRL handles it
  in one backward pass).
- New training-loop additions: per-batch the trainer collects head
  prediction accuracies for both correct-head and wrong-head probes.
- New wandb keys: `train/head_scale_acc`, `train/head_orient_acc`,
  `train/wrong_head_scale_acc`, `train/wrong_head_orient_acc`.

The "wrong-head" accuracy should *fall* to chance (1/6 for scale,
1/40 for orientation) over training — that's the GRL working.

### Step 5 — Config + sweep

[configs/sepvae.yml](configs/sepvae.yml) with:
```yaml
model:
  partition: {common: 4, scale: 3, orient: 3}  # → latent_dim=10
training:
  epochs: 50
  alpha_grl: 1.0       # adversarial weight on wrong-head loss
  beta_factor_a: scale
  beta_factor_b: orientation
  ...
```

Add as **Exp 11** in `EXPERIMENTS`. Optionally Exp 12 for SepVAE on the
correlated split — that's the *headline* experiment of the project
(SepVAE is *designed* for correlated data).

### Browser extensions for Phase 2d

This is where the explorer needs the most new viz.

**1. Partition-aware μ panel.** The Posterior μ panel currently shows
each dim as a flat row. For a partitioned model:
- Group rows visually with thin horizontal dividers labelled `z_common`,
  `z_scale`, `z_orient`.
- Colour the dim labels by group: grey for common, blue for scale,
  orange for orientation.
- Group-level summary at the top: "z_scale total KL = 4.2 nats /
  z_orient total KL = 3.8 nats".

**2. Partition-aware 1D KL geometry plot.** The dim-selector dropdown
groups dims by partition (`-- z_common --`, `-- z_scale --`, etc.).
Default = first dim of `z_scale` (the most diagnostic for the RQ).

**3. Partition-aware 2D KL scatter.** Pre-select intelligent defaults:
`dim x = z_scale[0]`, `dim y = z_orient[0]`, colour by `scale`. This
should produce the cleanest possible axis-aligned plot for a working
SepVAE — and the *most damning* misalignment plot if the model failed.

**4. Swap-decoded gallery (new card on the Explore tab).** This is the
SepVAE-specific diagnostic. Pick two real images A and B from the cache:
- decode(z_A) → reconstruction of A.
- decode(z_B) → reconstruction of B.
- decode([z_A.common, z_A.scale, z_B.orient]) → A's shape/scale at B's orientation.
- decode([z_A.common, z_B.scale, z_A.orient]) → A's shape/orientation at B's scale.

If the partitioning works, the swap decodings should look like genuine
re-combinations: A's sprite size at B's rotation. If it doesn't,
they'll look like blurry artifacts. **This is the single most
visually persuasive figure for the SepVAE chapter.** New endpoint
`POST /api/swap_decode` takes two cache indices and a partition pair to
swap.

**5. Wrong-head probe diagnostic in Analysis tab.** A small bar chart
showing for each (latent block, factor) pair the held-out probe
accuracy. Colour-coded:
- Diagonal (correct heads, e.g. `z_scale` predicting scale): should be
  green / high accuracy.
- Off-diagonal (wrong heads): should be red / chance accuracy.

Reuses cached encodings + the saved probe heads. New endpoint
`/api/probe_accuracy`.

**6. New help slugs.** `partition`, `grl`, `wrong_head`, `swap_loss` —
each linking the methodology to the visual.

### Verification (Phase 2d)

1. **GRL gradient sanity** — a unit test that confirms the gradient
   sign flips after passing through `grad_reverse`.
2. **Partitioning preserves total dim count** — `k_common + k_scale +
   k_orient == latent_dim` and the saved encoder/decoder accept the
   concatenated latent.
3. **Explorer compatibility** — even before the partition-aware viz is
   built, a SepVAE checkpoint must be loadable by the existing
   explorer (Option A above). Smoke: load Exp 11 → KL spectrum +
   correlation + MIG all run without errors.
4. **Wrong-head accuracy collapses** — over training, the wrong-head
   probe accuracy should drop from baseline (~10–20%) to chance level.
5. **Swap-decoded sanity** — the swap-decode endpoint produces sprites
   that have visually-consistent properties from each source. (Visual
   inspection only.)

---

## Phase 2e — Held-out pair split: compositional generalisation

### Why this phase

This is the strongest version of the RQ: *can the model imagine factor
combinations it never saw?* If you train on every (shape, scale)
combination *except* (heart, large), can the decoder produce a
plausible large heart from a latent vector at the right point in
latent space?

A truly disentangled model — one that has axis-aligned scale and
shape — should generalise to the held-out cell. An entangled model
should fail visibly.

This phase reuses *most* of the Phase 2b infrastructure plus a
focused evaluation step on the held-out cells.

### Code changes

**1. Trainer** — third value for `--split`:
```python
elif args.split == "heldout":
    train_idx, val_idx, test_idx, heldout_idx = make_heldout_pair_split(
        dataset,
        factor_a=args.heldout_factor_a,           # default: shape
        factor_b=args.heldout_factor_b,           # default: scale
        held_a_vals=args.heldout_a_vals,          # default: [2]   (heart)
        held_b_vals=args.heldout_b_vals,          # default: [4, 5] (large)
        seed=t_cfg["seed"],
    )
    # Save heldout_idx into the checkpoint so the explorer can
    # reconstruct the gallery without re-running the split.
```

**2. Checkpoint** — save the held-out indices:
```python
torch.save({
    ...,
    "heldout_idx": heldout_idx.tolist(),  # so the explorer knows what was excluded
}, best_ckpt)
```

**3. Sweep** — cells 12–16 (or 11–15 if SepVAE doesn't get its own
correlated cell): mirror cells 6–10 with `"split": "heldout"`.

**4. Metric** — `src/metrics/recombination.py` implements:
```python
def heldout_reconstruction_error(model, dataset, heldout_idx, *, device, ...):
    """Mean BCE / MSE on the held-out cells. Returns a per-(shape, scale)-cell
    breakdown plus the overall scalar."""
```

Plus an `eval_recombination.py` script that runs after each cell in
this phase.

### Browser extensions for Phase 2e

**1. Held-out gallery (new card on Explore tab).** Shows a small grid:
```
[ heldout ground-truth image ]   [ model's reconstruction ]   [ |diff| ]
```
…iterated over a sample of held-out indices (say 16). For each, the
caption shows the factor combination that was never seen at training.

This card is only populated if the loaded checkpoint has
`heldout_idx` in its config. New endpoint
`/api/heldout_sample?n=16&seed=0` returns the originals + recons.

**2. Compositional-generalisation MSE bar.** In the Analysis tab, a
small new card showing two numbers per loaded checkpoint:
- Mean recon MSE on the test split (in-distribution).
- Mean recon MSE on the held-out cells.

The *gap* between them is the compositional generalisation deficit.
Per-architecture, this is the single number that answers the RQ for
this phase.

**3. Per-cell breakdown.** Optional drill-down: a heatmap with
ground-truth shape (rows) vs scale (columns), each cell coloured by
recon MSE on samples in that cell. The held-out cells are outlined
in red. A model that generalises well should have **uniform colour**;
a failing model will have a *bright (high-error) island* exactly on
the held-out cells.

### Verification (Phase 2e)

1. **`heldout_idx` is saved in the checkpoint** — `torch.load(...)
   ['heldout_idx']` returns a non-empty list of valid dataset indices.
2. **No leakage** — the held-out indices are a strict subset of those
   not in `train_idx ∪ val_idx`.
3. **Gallery rendering** — opening an Exp 13 checkpoint in the explorer
   shows the held-out gallery with at least 16 unique held-out cells.
4. **Metric directionality** — for the same architecture, training on
   the held-out split should *worsen* held-out MSE compared to
   training IID (the IID model has seen those cells during training).
   If not, something's wrong with the split.
5. **The headline test** — across the 5 architectures × heldout split,
   the held-out MSE gap is smallest for SepVAE (Exp 16 if you build
   it) and largest for vanilla VAE (Exp 12). This is the
   thesis-defending result.

---

## Cross-cutting browser evolution

These extensions are not phase-specific — they elevate the explorer
from a per-checkpoint inspector to a thesis-quality comparison tool.

### Extension A — Multi-checkpoint Compare tab

**Why now:** with 10–16 checkpoints across splits and architectures,
loading them one at a time is unworkable. The defining moment of the
research happens *between* checkpoints, not within one.

**UI:** new top-level tab "Compare". Layout:

```
┌─ Comparison set ──────────────────────────────────────────────────┐
│  [ pinned: Exp 1 (IID, baseline)  ✕ ]                              │
│  [ pinned: Exp 6 (correlated, baseline)  ✕ ]                       │
│  [ + Add a checkpoint ]                                             │
└────────────────────────────────────────────────────────────────────┘

┌─ Metrics table ────────────────────────────────────────────────────┐
│              | MIG  | DCI-D | DCI-C | TC   | scale ρ | orient ρ   │
│ Exp 1 (IID)  | 0.07 | 0.18  | 0.21  | 1.2  | 0.57    | 0.08       │
│ Exp 6 (corr) | 0.04 | 0.10  | 0.11  | 2.4  | 0.61    | 0.65 ⚠     │
└────────────────────────────────────────────────────────────────────┘

┌─ Side-by-side panels ─────────────────────────────────────────────┐
│  KL spectrum (Exp 1)         |   KL spectrum (Exp 6)               │
│  Correlation heatmap (Exp 1) |   Correlation heatmap (Exp 6)        │
│  2D scatter (Exp 1)          |   2D scatter (Exp 6)                 │
└────────────────────────────────────────────────────────────────────┘
```

**Mechanism:** a new in-memory side-state holds up to 4 checkpoints
in addition to the active one. New endpoint:
```
POST /api/compare/load
Body: {checkpoint: "..."}
Response: {compare_id: 0..3, metrics: {mig, dci, ...}, kl_arr: [...]}
```
The browser keeps a separate cache of `enc_mu` per compare slot for
the side-by-side scatter rendering. Total memory: 4 × 240 KB ≈ 1 MB.

The metrics table is the *primary* comparison surface; the
side-by-side panels are for visual deep-dive when a number stands out.

### Extension B — Cross-experiment Summary table

**Why:** at the end of a thesis chapter, you want one figure that says
"here's everything we ran". A static-rendered table inside the
explorer beats curating a markdown table by hand.

**UI:** new top-level tab "Summary". Reads from a single endpoint
`GET /api/summary` that scans `checkpoints/` for any `best.pt` files
matching the EXPERIMENTS table, loads each in a background thread,
computes (or loads cached) MIG / DCI / KL / heldout-MSE, and renders:

| Run | split | latent_dim | hp | active dims | MIG | DCI-D | DCI-C | scale dim | orient dim | heldout MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp 1 | iid | 10 | β=1 | 9/10 | 0.07 | 0.18 | 0.21 | z7 (ρ=0.57) | z4 (ρ=0.08) | n/a |
| Exp 2 | iid | 10 | β=4 | 6/10 | 0.21 | 0.34 | 0.40 | z2 (ρ=0.71) | z9 (ρ=0.42) | n/a |
| Exp 6 | corr | 10 | β=1 | 9/10 | 0.04 | … | … | z7 (ρ=0.61) | z7 (ρ=0.65) ⚠ | n/a |
| Exp 12 | heldout | 10 | β=1 | 8/10 | 0.06 | … | … | … | … | 0.0042 |

The `⚠` flag highlights when scale and orientation share a top dim
(sign of entanglement). Clickable rows route to the Compare tab with
that checkpoint pre-pinned.

This is the "results section figure" of the thesis, generated live.

### Extension C — Provenance bar

A thin strip below the navbar (always visible):

```
checkpoint: checkpoints/vae/correlated_vae_z10_beta1.0_seed42/best.pt
split: correlated(scale, orientation, +)   architecture: VAE
hyperparameters: z=10, β=1.0, seed=42       trained: 50 epochs, val_loss=84.21
```

Reads from the checkpoint's saved `config`. Implemented once and used
by Phases 2b, 2c, 2d, 2e.

### Extension D — Routing UX

Currently, switching tabs preserves all state. With the Compare and
Summary tabs added, deep-link routes (`#compare?set=1,6` and
`#summary`) help researchers share specific views during supervisor
meetings. Implementation: hash-based router, no new dependencies.

---

## Recommended ordering and timeline

**Sequence:** 2c → 2b → cross-cutting Compare → 2e → 2d.

Reasoning:
1. **2c (DCI) first** — adds zero training time (analyses existing
   checkpoints), gives you a second metric for the Phase 2a write-up
   immediately, validates the implementation against existing
   well-understood IID checkpoints.
2. **2b (correlated) second** — the central RQ test. By the time it
   runs, both MIG and DCI exist to evaluate it.
3. **Compare tab next** — once you have IID + correlated checkpoints,
   the Compare tab is what makes them comparable. Build it before SepVAE
   so you don't have to retrofit it later.
4. **2e (held-out) fourth** — close to 2b mechanically; can run on
   existing architectures while SepVAE is being designed.
5. **2d (SepVAE) last** — the most novel and labour-intensive piece;
   benefits from having all the comparison infrastructure already built.

**Rough effort estimates:**
- 2c: ~1.5 days (metric + endpoint + browser panel).
- 2b: ~1.5 days (training wiring + sweep cells + Run-card grouping).
- Compare tab: ~2 days (significant new UI).
- 2e: ~1 day (mostly reuses 2b's `--split` mechanism).
- 2d: ~3–4 days (design note → GRL → SepVAE class → trainer → 5 new
  browser extensions).
- Cross-cutting (Provenance, Summary): ~1 day on top.

**Total to thesis-ready Phase 2:** ~10–12 working days of
implementation, plus cluster training time.

---

## Open questions to resolve

1. **Default correlated factors.** The RQ specifies (scale,
   orientation) but `make_correlated_split` is general. Confirm the
   sweep should use those two; consider an additional cell with
   (shape, scale) for a non-cyclic comparison.

2. **DCI regressor choice.** Random Forest is the standard but Lasso is
   used in some papers. Random Forest is more sensitive to the
   importance signal but slower. **Recommend** Random Forest with
   default depth; flag if computation time becomes an issue.

3. **SepVAE partition sizes.** `(4, 3, 3)` is a sensible default
   summing to 10; it could be `(2, 4, 4)` to give more capacity to
   the constrained sub-blocks. **Recommend** start at `(4, 3, 3)`;
   sweep if the metric is poor.

4. **Held-out cell choice.** Default `(shape=heart, scale ∈ {4, 5})`
   tests shape × scale generalisation. The dataset module also
   supports scale × orientation, which is more aligned with the RQ
   but may have weaker signal because both are continuous.
   **Recommend** start with the default; add the second variant if
   time permits.

5. **Multi-seed runs.** Per Locatello et al. 2019, a single seed isn't
   reliable evidence. **Recommend** add seeds 0/1/2 to Exp 1, Exp 5,
   Exp 11 (the three architecturally-distinct flagships) for a 3-seed
   confidence interval at the headline numbers. This adds a pure
   compute cost, not an implementation cost.

---

*Companion documents:*
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — overall project context and
  Phase 2a write-up.
- [EXPLORER_GUIDE.md](EXPLORER_GUIDE.md) — current browser features.
- [parking_lot.md](parking_lot.md) — original design-thinking notes for
  the SepVAE methodology (Phase 2d).
- [PHASE_2_DATASET_SUMMARY.md](PHASE_2_DATASET_SUMMARY.md) — dataset
  module API for the three splits.
