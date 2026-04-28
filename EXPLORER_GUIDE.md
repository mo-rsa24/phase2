# Disentanglement Explorer — User Guidebook

> **What this is:** an interactive Flask app for inspecting any trained
> VAE / β-VAE on dSprites. Lets you compose images by ground-truth factors,
> watch the encoder respond, traverse latent dimensions, and quantify how
> cleanly each factor maps to a single latent dim.
>
> **Why it exists:** to make every number in your disentanglement metrics
> visually concrete — every KL, ρ, and MIG should be paired with a sprite
> you can see change.
>
> **Source:** [scripts/disentanglement_explorer.py](scripts/disentanglement_explorer.py),
> [scripts/templates/disentanglement_explorer.html](scripts/templates/disentanglement_explorer.html),
> [scripts/static/](scripts/static/).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [The Model & Objective Strip](#2-the-model--objective-strip)
3. [Tab 1 — Select Run](#3-tab-1--select-run)
4. [Tab 2 — Explore](#4-tab-2--explore)
   - 4.1 [dSprites factor sliders](#41-dsprites-factor-sliders)
   - 4.2 [Original / Reconstruction / |Difference|](#42-original--reconstruction--difference)
   - 4.3 [Posterior μ panel](#43-posterior-μ-panel)
   - 4.4 [Latent Dimensions panel](#44-latent-dimensions-panel)
   - 4.5 [Latent Traversal Reconstruction](#45-latent-traversal-reconstruction)
   - 4.6 [KL Geometry — 1D](#46-kl-geometry--1d)
   - 4.7 [KL Geometry — 2D scatter](#47-kl-geometry--2d-scatter)
5. [Tab 3 — Traversal Grid](#5-tab-3--traversal-grid)
6. [Tab 4 — Analysis](#6-tab-4--analysis)
   - 6.1 [KL Activity Spectrum](#61-kl-activity-spectrum)
   - 6.2 [Factor–Latent Correlation](#62-factor-latent-correlation)
   - 6.3 [Mutual Information Gap (MIG)](#63-mutual-information-gap-mig)
   - 6.4 [Click interactions](#64-click-interactions)
7. [The Help System](#7-the-help-system)
8. [Worked Example — Answering the Research Question](#8-worked-example--answering-the-research-question)
9. [Glossary & Cheat Sheet](#9-glossary--cheat-sheet)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. Quick Start

### From VS Code

The repository ships with [.vscode/launch.json](.vscode/launch.json) containing
five debug configurations — one per sweep checkpoint plus a no-checkpoint
launcher.

1. Open the **Run and Debug** panel (`Ctrl+Shift+D`).
2. Pick a config (e.g. *"Disentanglement Explorer (z=10, β=1)"*).
3. Press **F5**.
4. The integrated terminal prints `Running on http://127.0.0.1:5050`. The
   `serverReadyAction` hook auto-forwards the port and opens your local
   browser.

### From the CLI

```bash
micromamba activate phase2-repr
python scripts/disentanglement_explorer.py \
  --checkpoint checkpoints/vae/vae_z10_beta1.0_seed42/best.pt \
  --device cpu \
  --port 5050
```

If the app is on a remote node and you want to view it on a laptop, either
let VS Code Remote-SSH auto-forward the port, or use plain `ssh -L`:

```bash
ssh -L 5050:localhost:5050 molefe@<cluster-node>
```

Then open `http://localhost:5050` on your laptop.

### Status indicator

Top-right of the navbar:

| Dot       | Meaning                                                |
|-----------|--------------------------------------------------------|
| 🔴 red    | No model loaded — most actions disabled.               |
| 🟠 spin   | Loading checkpoint or encoding the 3000-sample cache.  |
| 🟢 green  | Model ready. Latent dim shown beside the dot.          |

The 3000-sample cache (used by the KL spectrum, correlation heatmap, MIG,
and the 2D scatter) is built in a background thread on every model load —
typically takes ~5–15 s depending on `--device`.

---

## 2. The Model & Objective Strip

Click the chevron at the top of any tab. It expands to show the **β-VAE
objective** (rendered with KaTeX):

$$\mathcal{L}_{\beta\text{-VAE}}
  = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}}
  - \beta \cdot \underbrace{D_{KL}(q_\phi(z|x) \,\|\, p(z))}_{\text{regularisation}}$$

with three annotation cards mapping each term to a UI element:

| Term              | Where to look in the UI                                            |
|-------------------|--------------------------------------------------------------------|
| Reconstruction    | The **\|Difference\|** panel + **MSE** value in *Explore*          |
| KL term           | The **KL Activity Spectrum** bar chart in *Analysis*               |
| β (KL weight)     | Shown next to the strip header (auto-detected from checkpoint name) |

This strip stays on every tab so the formula is always one click away.
β is auto-resolved from the loaded checkpoint via the experiments table in
[scripts/disentanglement_explorer.py:56](scripts/disentanglement_explorer.py#L56).

---

## 3. Tab 1 — Select Run

### 3.1 The 4 sweep experiments

The default experiments correspond to the four cells of the sweep plan.
Click any card to load its checkpoint; the loaded card outlines green.

| ID | Label              | latent_dim | β   | Purpose                | What to expect |
|----|--------------------|-----------|-----|------------------------|----------------|
| 1  | Exp 1              | 10        | 1.0 | Baseline VAE           | Modest disentanglement; some entanglement between scale and orientation. |
| 2  | Exp 2              | 10        | 4.0 | β-VAE                  | More dim collapse, but the surviving dims tend to be cleaner — better MIG, sharper traversals. |
| 3  | Exp 3              | 4         | 1.0 | Undercomplete bottleneck | Forced multiplexing — 4 dims for 5 non-trivial factors. Expect heavy entanglement. |
| 4  | Exp 4              | 20        | 1.0 | Overcomplete           | Many redundant / collapsed dims. Active dims should still capture factors but more often duplicated. |

### 3.2 Custom checkpoint

Paste a path under "Custom Checkpoint" and click **Load**. The path is
resolved relative to the project root. Useful for inspecting checkpoints
that aren't in the sweep (intermediate epochs, ablations, fine-tunes).

### 3.3 What happens on load

1. The encoder + decoder are loaded onto the configured device.
2. A background thread encodes 3000 random dSprites images into the cache
   (`enc_mu`, `enc_logvar`, `enc_factors`, `kl_arr`).
3. The Explore tab is initialised with default factor values (square, mid
   scale, orientation 0, centred position) and the encoder responds with
   the first encoding.

The dot turns 🟢 once the model is loaded. The spectral plots in the
Analysis tab can then be computed; they will block briefly if the cache
isn't quite ready.

---

## 4. Tab 2 — Explore

This is the workhorse tab. The layout is three columns: left = factor
sliders + posterior, centre = images + traversal + KL geometry, right =
latent sliders.

### 4.1 dSprites factor sliders

Five sliders correspond to the non-constant dSprites factors:

- **shape**: 0=square, 1=ellipse, 2=heart
- **scale**: 0..5 (six discrete sizes)
- **orientation**: 0..39 (cyclic, ~9° per step → 0°..351°)
- **pos_x, pos_y**: 0..31 pixel positions

Move any slider → the encoder receives the corresponding dSprites image →
the three image panels and the posterior bars update in ~120 ms.

The orientation label also shows degrees: `27° (3)` means orientation
index 3 ≈ 27°.

### 4.2 Original / Reconstruction / |Difference|

| Panel              | What you see                                                    |
|--------------------|-----------------------------------------------------------------|
| **Original**       | The exact dSprites image at the chosen factor combination.     |
| **Reconstruction** | What the decoder produces from `μ` (posterior mean — no sampling). |
| **\|Difference\|** | Absolute pixel error, normalised to its own max so it's always visible. |

The **MSE** value below the triplet is the mean squared error between
original and reconstruction. Typical numbers for these checkpoints are
in the range 1e-4 to 5e-4.

**What to look for:**

- **Sharp residuals on the |Difference| panel** = under-fitting; the
  decoder couldn't produce that exact pixel pattern.
- **Diffuse, low-contrast residuals** = good reconstruction (the
  difference panel is auto-rescaled, so even a low-residual image will
  appear to have visible noise — read the MSE for absolute accuracy).
- **Reconstruction doesn't look like the original** for an extreme factor
  combination (e.g. tiny heart at the corner) = factor is poorly
  encoded; the decoder is averaging over training-set means.

### 4.3 Posterior μ panel

For each latent dim the encoder outputs a Gaussian posterior
$q(z_i|x) = \mathcal{N}(\mu_i, \sigma_i^2)$. This panel visualises that
posterior compactly:

```
z3  [sparkline]  ████████░░░░░░░░░░░░  −1.22  KL=4.22  Δ +0.20
│        │              │                │       │       │
│        │              │                │       │       └ Δμ from anchor (when set)
│        │              │                │       └ mean KL of this dim across 3000 samples
│        │              │                └ μ value (sign-coded, blue=+, red=−)
│        │              └ μ bar; width ∝ |μ|; colour intensity ∝ KL
│        └ 60×18 px sparkline: prior 𝒩(0,1) (grey) vs current posterior (blue) + dashed μ marker
└ latent dim index
```

**Concrete reading example.** A row showing
`z3  ▆██▆▁  ████  −1.22  KL=4.22`:

- The encoder placed the posterior for this image **1.22 units below
  zero** on dim 3. (Sign is arbitrary — the mirror image −1.22 → +1.22
  may correspond to the opposite end of the same factor.)
- KL = 4.22 nats on this dim averaged over 3000 random images — this is
  one of the most-active dims (≈ 6 bits of information per image).
- The sparkline shows a tall, narrow blue spike at −1.22 against the
  wider grey unit Gaussian — this image has triggered a sharp posterior,
  the encoder is **confident** about z<sub>3</sub> for this image.

**Hover any row** for a detailed Bootstrap tooltip:

> **z3**: μ=−1.22, σ=0.18 KL=4.22 nats (≈ 6.09 bits); posterior is 6.8σ
> from prior centre.

The bars are sorted top-to-bottom by mean KL (most active first). After
running **Compute** in the KL panel, the global KL spectrum is used; before
that, the per-encoding KLs of the current image are used.

When you have set an **Anchor** (Section 4.5), each row also shows:
- A thin grey vertical tick on the bar = anchor's μ for that dim.
- A `Δ` text overlay = current μ minus anchor μ.

This makes it easy to track which dims have moved as you sweep factor
sliders away from your anchor.

### 4.4 Latent Dimensions panel

Right column. Each row is a single latent dim with:

- A **label** like `z3  KL=4.22` (only when an anchor is set).
- A **↺ reset** button that sets just this dim back to anchor μ.
- A **range slider** spanning [−4, +4] in the latent space.
- The **current value**.
- A **Δσ** indicator: `(current − anchor μ) / anchor σ` — how far you've
  moved this dim, in units of the encoder's own confidence:
  - grey: |Δσ| < 0.5 (barely moved)
  - blue: 0.5 ≤ |Δσ| < 2 (within posterior support)
  - orange: |Δσ| ≥ 2 (out of distribution — expect surprises)

**Header buttons:**

- **↺ Reset all** — restores every slider to its anchor μ.

The traversal panel below the centre image triplet updates live as you
drag any slider (debounced ~80 ms). This is the primary way to ask "what
is z<sub>3</sub> *for*?": move it alone, watch the sprite change.

**Tip.** Drag z<sub>i</sub> and watch the Δσ indicator. If the sprite stops
changing visibly past |Δσ|=3, that's the limit of the encoder's training
distribution along that dim — beyond, the decoder is extrapolating.

### 4.5 Latent Traversal Reconstruction

A two-image card that appears once you click **"Use as Anchor →"** in the
factor panel. It locks in the current `μ` as the *anchor* — your reference
point for traversals.

| Image     | What it is                                                         |
|-----------|--------------------------------------------------------------------|
| **Anchor** | Decoded from anchor μ — should match the Reconstruction at the moment you anchored. |
| **Traversal** | Decoded from the current latent slider values — updates live as you drag. |

The **Reset to Anchor** button under the panel restores all sliders.

The anchor enables three downstream features:

- The **Traversal Grid** tab can run.
- The **single-dim traversal modal** in the Analysis tab can run (it
  needs anchor μ + logvar to construct the sweep).
- The 1D KL geometry plot can show the anchor's σ as the live posterior
  width.

### 4.6 KL Geometry — 1D

Below the image triplet. A live Plotly figure showing two PDFs on a
shared axis:

- **Grey, filled** — the prior 𝒩(0, 1).
- **Blue, filled** — the current posterior 𝒩(μ<sub>i</sub>, σ<sub>i</sub>²).
- **Dashed vertical line** — the current μ.
- **Title** — `z<sub>i</sub>: μ=…, σ=…, KL=… nats` (computed client-side).

**Controls:** a single dropdown to pick the dim. Default = the highest-KL
dim. The dropdown labels include each dim's KL: `z3 (KL=4.22)`.

**Interactivity.** When the chosen dim's slider in the latent panel
moves, the live μ in this plot translates with it (σ is held at the
encoder's posterior σ for the current input). This is the single most
useful chart for grounding "KL=4.22" — you literally see the cost of
the gap between blue and grey curves.

**What to read out:**

- **Posterior overlapping prior** → KL ≈ 0 → dim is collapsed.
- **Narrow blue spike far from origin** → high KL → dim is highly
  informative for this image.
- **Wide blue curve near origin** → encoder is unsure but not committed
  to any direction → marginal information.

### 4.7 KL Geometry — 2D scatter

Below the 1D plot. Pairwise scatter of 3000 cached encodings projected
onto two latent dims, coloured by a chosen ground-truth factor, with a
**hover/click side-panel** that anchors every dot to its real dSprites
image and the model's reconstruction.

**Controls:**
- `dim x`, `dim y` — pick any two latent dims (defaults: top-2 KL).
- `colour` — `scale`, `orientation` (cyclic), `shape`, `pos_x`, `pos_y`.
- **Render** button — fetches the cached encodings (lazy, ~240 KB).

**Plot elements:**
- 3000 small dots — each a sample's posterior mean.
- **Grey dotted ring** — prior 95% contour (radius ≈ 2.45σ).
- **Blue dashed ellipse** — the current encoding's 2σ posterior.
- **Blue ✕** — current μ.
- **Red ★** — anchor μ (when set).

**Cyclic colourmap for orientation.** When you choose `orientation` as
the colour, the scatter switches from viridis to a cyclic HSV-style
colourmap. dSprites orientation wraps from index 0 (≈0°) to index 39
(≈351°) — those are visually almost identical but linear viridis would
paint them as the most distant colours. The cyclic ramp keeps "near in
angle" mapped to "near in colour", which is the only honest way to read
orientation gradients on the plot.

**The hover/click side-panel.** To the right of the scatter is a card
that shows the original dSprites image and the model's reconstruction
for the dot under the cursor:

- **Hover** any dot → side-panel updates after a 100 ms debounce with
  that sample's original, reconstruction, and ground-truth factors.
- **Click** any dot → pins the side-panel to that sample (turns the
  background blue). The panel stops following the cursor. Click the
  same dot again, or press the **unpin** button, to release.
- The factors readout highlights `scale` and `orientation` in blue —
  these are the RQ factors. Useful when scanning for anomalies.

This is your **three-way consistency check**: colour (ground-truth
label) ↔ original (visual confirmation) ↔ recon (what the encoder–
decoder pair actually preserved). When all three agree, the dot is a
clean disentangled encoding. When they disagree — the dot is yellow
(orientation=39), the original is a sprite rotated 351°, but the
recon comes out at 0° — you've found a specific failure mode that the
metrics smooth over.

**The disentanglement reading.** This card answers two questions:

1. *"Is the cloud's spatial trend consistent with one factor on each
   axis?"* (the global view, from colour gradients)
2. *"At any given (x, y), does the recon visually match the colour
   label?"* (the local view, from the side-panel)

A clean disentangled run gives "yes" to both:

- **Clean horizontal gradient** when coloured by `scale` and you've
  picked dim a as the x-axis ⇒ scale is encoded along dim a.
- **Hovering rightward** along the x-axis shows recons whose *size
  grows* as the cursor moves, regardless of where on the y-axis you
  are.
- Switch colour to `orientation`: now you see a vertical cyclic
  gradient, and hovering upward shows recons whose *rotation cycles*
  while size stays roughly constant.

An entangled run breaks at one of these checks:

- **No gradient or diagonal gradient** under colour ⇒ the chosen pair
  doesn't axis-align factors.
- **Hover trends mismatch the colour:** as you scan rightward, the
  recon's size sometimes grows and sometimes rotates instead — the
  factor isn't isolated to one axis.
- **Recon ≠ original mismatch** at individual dots: encoder dropped a
  factor entirely for that sample.

**Independence of two factors** is visible in a single pair of
renders: pick the candidate scale dim as x and the candidate
orientation dim as y; colour by scale, note the gradient is
horizontal; switch the colour to orientation, note the gradient is
vertical. **Orthogonal gradients = independent codes.** Same gradient
axis under both colorings = those factors are entangled in this pair.

**Reading the cloud silhouette itself.** Even before colour, the
*shape* of the cloud is informative:
- **Per-axis spread** → a thin vertical / wide horizontal cloud means
  the y-dim is collapsed and the x-dim is active.
- **Banding** → discrete factors (scale, shape) produce visible bands
  perpendicular to the encoding axis. Diagonal bands = factor split
  between x and y (entangled).
- **Cluster structure** → categorical factors (shape) produce ~3
  clusters; off-axis clusters suggest factor mixing.

**Caveats to bear in mind:**

- You're showing 2 of *N* dims. The recon reflects the full μ vector,
  not "what z<sub>x</sub> and z<sub>y</sub> alone do" — for *that* question, the
  1D traversal grid is the dedicated tool.
- The "trend" is across many hovers, not within any single one. Each
  recon also varies in shape/position because every sample has its
  own values for those (they live on other dims).
- Pre-select candidate dims via the correlation heatmap or MIG. Random
  pair-picking can produce false-negative views (a clean-looking
  projection where neither dim encodes the factor of interest).

---

## 5. Tab 3 — Traversal Grid

This tab produces the canonical disentanglement figure: a grid where
**each row sweeps a single latent dim** and each column shows the decoded
sprite at that step.

**Settings (left column):**

| Control                  | Default | What it does                                              |
|--------------------------|---------|-----------------------------------------------------------|
| Steps per dimension      | 11      | Columns in the grid (5–15).                               |
| Range (σ units)          | 3σ      | Sweep range; defines `[μ − kσ, μ + kσ]` per row (1–5σ).   |
| Max dims to show         | 10      | Number of rows. Dims are picked top-down by mean KL.      |
| Use anchor's σ           | off     | If on, σ comes from the encoder; if off, σ = 1 (uniform). |

**Output (right column):** a single PNG with the row labelled by `zi  KL=…`
and column headers showing `±k.kσ`. Most-active dim on top.

**Reading the grid for the research question:**

For each row, ask: *"What changes as I scan left → right?"*

- **Only one factor varies** (size grows; sprite rotates; sprite slides) →
  this dim cleanly encodes that one factor → axis-aligned code.
- **Multiple factors change together** (size grows AND sprite rotates) →
  this dim mixes those factors → entangled code.
- **Nothing visible changes** → this dim is collapsed (KL near 0) or
  encodes something the decoder ignores.

Compare across the 4 runs:
- **Exp 2 (β=4)** typically produces the cleanest grid — fewer rows
  visibly active, but each one isolates a single factor.
- **Exp 3 (z=4)** crowds five factors into four dims → at least one row
  will visibly multiplex.

### When you can't generate

The button is disabled until an **anchor** is set in the Explore tab. The
grid uses the anchor's `μ` as the starting point for every row.

---

## 6. Tab 4 — Analysis

Three quantitative panels plus several click-driven qualitative views.
None of the panels run automatically — click their **Compute** button
once the model and cache are ready.

### 6.1 KL Activity Spectrum

A bar chart where each bar is the **mean KL of one latent dim** averaged
over 3000 cached encodings.

**Axes:**
- *x*: latent dim index `z0..z(d−1)` (in dim-order, **not** sorted by KL).
- *y*: mean KL in **nats** (1 nat ≈ 1.443 bits).

**Visual cues:**
- **Blue bar** = active dim (KL > 0.1 nats).
- **Grey bar** = collapsed dim (effectively ignored by the decoder).
- **Red dotted line** = the 0.1-nat threshold.
- A small annotation in the top-right shows `Active: K/d` — the count of
  surviving dims.

**What it tells you about the research question:**
- Only active dims can encode scale or orientation. If your model has 6
  active dims out of 20 (Exp 4), only those 6 are candidates.
- A model with too few active dims (e.g. Exp 2 may collapse to 4 active
  dims) might lack the capacity to put scale and orientation on
  different dims.

**Hover interaction.** Hovering any bar shows a small popover with a
3-frame `[−2σ, 0, +2σ]` traversal of that dim from the current anchor.
Spotting a high-KL dim that produces no visible sprite change is the
classic signature of "active by KL, dead by content" — the encoder
crammed information into it that the decoder doesn't use.

### 6.2 Factor–Latent Correlation

A heatmap of `|Spearman ρ|` per (latent dim, ground-truth factor)
combination.

**Axes:**
- *x*: 6 dSprites factors. `color` is always 0 (constant).
- *y*: latent dims, **sorted by KL descending** (most active at top).

**Cell value:** rank correlation between μ<sub>i</sub> and the integer factor
class — 0 means no monotonic relationship, 1 means perfect monotonic.
Computed via [factor_latent_correlation](src/metrics/disentanglement.py)
on the 3000 cached encodings.

**Interpretation rules:**
- A perfectly disentangled model has **exactly one bright cell per
  column** — each factor "owned" by one dim.
- A perfectly disentangled model has **at most one bright cell per row**
  — each dim owns at most one factor.
- **A column with several mid-tone cells** = factor is split across
  multiple dims (entangled).
- **A row with two bright cells** = that dim multiplexes two factors.

**The orientation caveat (yellow banner).** Spearman ρ assumes a
monotonic factor → latent mapping. dSprites orientation is **cyclic**
(index 0 ≈ 0°, index 39 ≈ 351°, almost identical), so any model that
encodes orientation perfectly along a circle will produce a low ρ on
that column. **Read the orientation column with this in mind**, and
cross-check with the conditional histogram (Section 6.3) which uses
mutual information instead.

**Click any cell** to open a *single-dim traversal modal* — see
Section 6.4.

### 6.3 Mutual Information Gap (MIG)

The most rigorous of the three metrics. Background thread — takes
30–120 s.

**Formula:**

$$\mathrm{MIG} = \frac{1}{K} \sum_{k=1}^{K} \frac{I(z_*; v_k) - I(z_2; v_k)}{H(v_k)}$$

For each factor v<sub>k</sub>, MIG measures the *gap* between the most
informative latent dim z<sub>\*</sub> and the second-most informative z<sub>2</sub>,
normalised by the factor's entropy. Big gap = one dim cleanly owns the
factor (axis-aligned). Small gap = the factor is shared.

**Output panel:**

- **Overall MIG** — large number with a colour-coded badge:
  - 🟢 **strong** ≥ 0.4
  - 🟡 **moderate** ≥ 0.2
  - 🔴 **poor** < 0.2
  Thresholds loosely follow Chen et al. 2018 reported ranges on dSprites.
- **Per-factor bars** — one for each non-trivial dSprites factor, each
  with its own badge.
  - The two bars relevant to your research question — `scale` and
    `orientation` — are highlighted with a thicker border and pinned to
    the front of the row, even if their values are low.

**What to write up.** A small table per run captures what you need:

| Run | Active dims | Scale ρ (top dim) | Orient ρ (top dim) | MIG_scale | MIG_orient | Same dim? |

The "same dim" column is the key research-question check: scale and
orientation should **not** be owned by the same dim (that would be
entanglement).

### 6.4 Click interactions

These are powerful but not obvious — they connect the analysis numbers
back to actual sprites.

#### Click any heatmap cell (correlation)

Opens a modal containing a **9-frame ±3σ traversal** of that dim, with
the title `z<sub>i</sub> · factor: v<sub>k</sub> · |ρ| = X.XX`. This is the
direct visual answer: *"does sweeping z<sub>i</sub> actually change v<sub>k</sub>?"*

Common shapes you'll see:
- |ρ|=0.8 cell, traversal cleanly varies size only → real, axis-aligned
  scale code.
- |ρ|=0.6 cell, traversal varies size *and* shape → factor leakage.
- Low |ρ| cell on orientation, traversal cleanly rotates the sprite →
  the cyclic-Spearman undercount in action.

The modal requires an **anchor** to be set — the traversal is `μ_anchor
± kσ_anchor`.

#### Hover any KL spectrum bar

A small popover shows three thumbnails: `[−2σ, 0, +2σ]` decoded from
that dim. Useful for quickly auditing many dims without opening a full
modal.

#### Click any MIG factor bar

Opens the **conditional μ histogram modal**. For the chosen factor:

1. The server picks the dim that best encodes that factor:
   - For non-orientation factors: `argmax_i |corr[i, v_k]|` from the
     cached correlation matrix.
   - For orientation: `argmax_i I(μ_i; v_k)` via
     `mutual_info_classif` — handles cyclicity.
2. Returns the μ values grouped by factor level (e.g. 6 groups for
   `scale`, 40 for `orientation`).
3. Plots overlay histograms — one trace per factor level.

**Reading the histograms:**

- **Cleanly separated humps** along the μ axis → factor cleanly encoded
  by that dim.
- **Overlapping humps** → factor is poorly captured (or split across
  multiple dims).
- For orientation specifically, the **MI rank** in the title (e.g.
  `z7 (MI rank — orientation is cyclic)`) tells you the selection used
  the cyclic-aware metric. If you see two clean clusters at opposite
  μ values, the model has folded the cyclic factor onto a line.

---

## 7. The Help System

Every term that has a non-obvious definition is wrapped with a help
hook. Two ways to surface them:

### 7.1 The `?` icons

Small blue circles next to panel titles. Click → popover with:
- A title
- A 2–3 paragraph plain-language explanation
- The relevant formula (KaTeX)
- Brief cross-link pulses on UI elements that match (e.g. clicking the
  KL `?` makes the KL panel briefly glow blue).

There are 7 of these, one per major panel.

### 7.2 Help-Mode toggle (navbar)

Flip the **Help mode** checkbox in the top-right. Every defined term
inline (e.g. "posterior", "Spearman ρ", "active dim") gets a dotted
underline; hover for a one-line tooltip.

Off by default to keep the UI uncluttered.

### 7.3 Centralised glossary

All popover content is sourced from
[scripts/explorer_help.py](scripts/explorer_help.py). To add a new term:

1. Add a slug entry to the `HELP` dict (keys: `title`, `short`, `long`,
   `formula`, `links`).
2. In the template, place `<span class="help-icon" data-help-slug="my_slug">?</span>`
   wherever you want a `?` icon.
3. To make the term Help-Mode-aware, wrap it with
   `<span class="term" data-help-slug="my_slug">my term</span>`.

---

## 8. Worked Example — Answering the Research Question

> *Are scale and orientation captured independently by single, distinct
> latent dimensions across the 4 sweep runs?*

For each of the 4 checkpoints (Exp 1–4), follow this procedure. Total
time per run ≈ 5 minutes.

### Step 1 — Load and stabilise

1. *Select Run* → click the experiment card. Wait until the dot turns
   🟢 (model loaded) AND the cached-encodings spinner finishes (the
   Analysis-tab buttons remain disabled until the cache is ready).
2. Note the latent_dim and β shown in the navbar / objective strip.

### Step 2 — KL spectrum

3. *Analysis* → click **Compute** on KL Activity Spectrum.
4. Record `K_active = number of active dims`. Note the visible drop —
   for Exp 4 (z=20), expect ~6–10 active out of 20.
5. Hover the top 2–3 bars to confirm their traversals look meaningful.

### Step 3 — Correlation heatmap

6. Click **Compute** on Factor–Latent Correlation.
7. Read the `scale` column: which dim has the highest |ρ|? Record:
   `scale_dim = z<sub>i</sub>`, `scale_ρ = ...`.
8. Read the `orientation` column. Note the warning: orientation is
   cyclic, so ρ underestimates encoding. Record the top dim anyway, but
   mark it provisional.
9. **Independence check:** are `scale_dim` and `orientation_dim`
   different? If they're the same, you've already found entanglement.

### Step 4 — Visual confirmation via heatmap clicks

10. Click the heatmap cell for the top scale dim → confirm the
    traversal cleanly grows/shrinks the sprite.
11. Click the heatmap cell for the top orientation dim → confirm the
    traversal rotates the sprite (or cleanly varies *some* property —
    even if ρ is low, the visual evidence is what matters).

### Step 5 — MIG with thresholds

12. Click **Compute MIG** (wait 30–120 s).
13. Read the **per-factor scale and orientation bars**:
    - Both green ⇒ both factors cleanly captured.
    - Scale green, orientation amber/red ⇒ scale is fine; orientation
      is either cyclically-encoded (likely) or genuinely entangled.
    - Both amber/red ⇒ poor disentanglement.
14. Click the `scale` bar → conditional histogram. Tight separated
    humps on its top dim ⇒ confirmed clean encoding.
15. Click the `orientation` bar → MI-selected top dim. Multiple
    well-separated humps ⇒ orientation is encoded somewhere (cyclic or
    not).

### Step 6 — Geometric independence check

16. *Explore* → KL Geometry (2D) → render with `dim x = scale_dim`,
    `dim y = orientation_dim`, colour by `scale`. Expect a horizontal
    gradient.
17. Switch colour to `orientation`. Expect a vertical gradient (or
    cyclic colour pattern along *y*).
18. Both clean ⇒ scale and orientation occupy independent latent axes.

### Step 7 — Final write-up table

| Run                | active | scale dim · ρ | orient dim · ρ | MIG_scale | MIG_orient | Independent? |
|--------------------|-------:|---------------|----------------|----------:|-----------:|--------------|
| Exp 1 (z=10, β=1)  |    9   | z7 · 0.57     | z4 · 0.08      |     0.08  |      0.00  |   ✓ partial  |
| Exp 2 (z=10, β=4)  |    …   | …             | …              |        …  |         …  | …            |
| Exp 3 (z=4, β=1)   |    …   | …             | …              |        …  |         …  | …            |
| Exp 4 (z=20, β=1)  |    …   | …             | …              |        …  |         …  | …            |

The "Independent?" column should be:
- ✓ if `scale_dim ≠ orientation_dim` AND both have visually clean
  single-dim traversals AND MI-based orientation conditional histogram
  shows separable humps.
- ✗ otherwise.

---

## 9. Glossary & Cheat Sheet

### 9.1 Variables and quantities

| Symbol / Term | Definition |
|---|---|
| `μ_i` (μ) | Posterior mean for latent dim *i*; centre of the encoder's Gaussian for this image. |
| `σ_i` (σ) | Posterior standard deviation. Computed as `exp(½·logvar_i)`. |
| `logvar_i` | What the encoder actually outputs (numerically stable). |
| Prior `p(z)` | 𝒩(0, *I*); independent unit Gaussians per dim. |
| Posterior `q(z\|x)` | 𝒩(μ(x), σ²(x)); what the encoder produces for image *x*. |
| Anchor μ | Snapshotted `μ` used as the starting point for traversals. |
| Δσ | `(slider value − anchor μ) / anchor σ`; how far you've moved a dim in posterior-σ units. |

### 9.2 KL & entropy

| Quantity | Formula | Units | Notes |
|---|---|---|---|
| Per-image, per-dim KL | `½·(μ² + σ² − 1 − log σ²)` | nats | Closed form for two Gaussians. |
| **Mean KL_i** (KL spectrum) | `(1/N) Σ_n KL_i(x_n)`, *N*=3000 | nats | What the bar chart shows. |
| Total KL term in ELBO | `Σ_i KL_i` | nats | Penalised by β in the objective. |
| 1 nat | `log₂(e)` ≈ 1.443 bits | — | Conversion factor. |
| Active-dim threshold | KL > 0.1 nats | — | Heuristic; below this dim is collapsed. |

### 9.3 Disentanglement metrics

| Metric | Range | Source | Interpretation |
|---|---|---|---|
| `\|ρ_S(z_i, v_k)\|` | [0, 1] | Heatmap | Spearman rank correlation. 1 = perfect monotonic; 0 = none. **Underestimates cyclic factors.** |
| Per-factor MIG | [0, 1] | MIG panel | Gap between most-informative and 2nd-most-informative dim, normalised by H(v_k). |
| Overall MIG | [0, 1] | Top of MIG panel | Average across factors with size > 1 (excludes `color`). |

### 9.4 Threshold conventions

| Quantity | Green / Strong | Amber / Moderate | Red / Poor |
|---|---|---|---|
| Per-factor MIG | ≥ 0.4 | 0.2 – 0.4 | < 0.2 |
| Overall MIG (rough) | ≥ 0.3 | 0.1 – 0.3 | < 0.1 |
| KL active threshold | KL > 0.1 nats | — | KL ≤ 0.1 nats (collapsed) |
| Δσ from anchor | < 0.5 | < 2 | ≥ 2 (out of distribution) |
| `\|ρ\|` (very rough) | ≥ 0.7 | 0.4 – 0.7 | < 0.4 |

### 9.5 Top-dim selection in conditional histogram

| Factor | Method | Why |
|---|---|---|
| `shape, scale, pos_x, pos_y` | `argmax_i \|ρ_S\|` | Fast, consistent with the heatmap. |
| `orientation` | `argmax_i I(μ_i; v_k)` via sklearn `mutual_info_classif` | Spearman undercounts cyclic factors; MI handles them. |

The selection method is shown in the modal title.

---

## 10. Common Pitfalls

1. **"Compute MIG" never returns.** It really does take 30–120 s. The
   spinner is up. Don't reload — the background thread will write
   `_state["mig_result"]` when done and the next `pollMIG` (every 2s)
   picks it up.

2. **Heatmap click does nothing.** You haven't set an anchor. The mini
   traversal modal needs `anchor_mu` and `anchor_logvar` to construct
   the sweep. Go to *Explore*, encode any image, click "Use as Anchor".

3. **2D scatter says "Loading cached encodings…" forever.** The
   3000-sample cache is still being computed in the background thread.
   Wait 5–15 s, click **Render** again. On CUDA devices it's near-instant.

4. **All MIG bars are red.** This is *not* a bug for vanilla VAE on
   dSprites (Exp 1, Exp 4). MIG of 0.05–0.15 is normal. Use the
   conditional histogram and traversal grid as the visual evidence; MIG
   thresholds are calibrated against what dedicated disentanglement
   models can achieve.

5. **Latent slider values clip at ±4.** The slider range is hard-coded;
   if your encoder produces μ outside this, the slider visually clips
   but the actual μ used for decoding is preserved (the slider's `value`
   attribute holds the unrounded float). For traversals beyond ±4 use
   the **Traversal Grid** tab with a larger σ range.

6. **The orientation row in the heatmap is mostly dark.** Expected —
   see the cyclic caveat banner. The conditional histogram for
   orientation uses MI selection and will show the dim that actually
   encodes it.

7. **Reset to Anchor doesn't restore the reconstruction exactly.**
   Floating-point drift in the slider step (0.05) means the post-reset
   z is within ±0.025 of the true anchor μ. Visually identical;
   numerically not bit-exact.

8. **Hover popover on a KL bar lingers after I move the mouse.**
   Bootstrap popovers anchor to the element under the cursor when
   shown; if Plotly redraws the bar, the popover may stay anchored to
   stale geometry. Click anywhere outside to dismiss.

9. **The objective strip shows `β = ?`.** The β passthrough relies on
   matching the loaded checkpoint path against the `EXPERIMENTS` table.
   Custom checkpoints outside that table show `?`. To fix, add an entry
   to [scripts/disentanglement_explorer.py:56](scripts/disentanglement_explorer.py#L56)
   or rename the checkpoint to match an existing pattern.

10. **Help-Mode underlines disappear when I open a modal.** Bootstrap
    re-renders modal content; underlined terms inside modals are not
    yet wired to Help-Mode. The popover-based `?` icons inside modals
    work correctly.

---

## Appendix — Architectural notes

- **Backend**: Flask + threading. State lives in `_state` dict, guarded
  by `_lock`. The 3000-sample encoding cache is rebuilt on every model
  load.
- **Frontend**: Bootstrap 5.3.2 + Plotly 2.27.0 + KaTeX 0.16.9. JS split
  across [explorer-core.js](scripts/static/js/explorer-core.js)
  (load/encode/decode/sliders),
  [explorer-analysis.js](scripts/static/js/explorer-analysis.js) (KL/corr/MIG +
  modals), [explorer-geometric.js](scripts/static/js/explorer-geometric.js)
  (1D/2D KL viz), [explorer-help.js](scripts/static/js/explorer-help.js)
  (popovers, KaTeX, Help-Mode).
- **Help dictionary**: [scripts/explorer_help.py](scripts/explorer_help.py)
  is the single source of truth, JSON-serialised into `window.HELP`
  on every page load.
- **Endpoints added in this iteration**:
  - `GET /api/cached_encodings` — 3000 cached `enc_mu` + factor labels.
  - `POST /api/single_dim_traversal` — 1-row traversal for modals/hover.
  - `POST /api/factor_conditional_histogram` — μ samples grouped by
    factor value, with corr- or MI-based top-dim selection.

To rebuild a PDF (mirroring the existing `PROJECT_GUIDE.pdf` workflow):

```bash
# Adjust docs/build_pdf.sh to point SRC at EXPLORER_GUIDE.md, then:
bash docs/build_pdf.sh
```
