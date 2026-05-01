# Targeted Weak-Supervision VAE on dSprites

This document is the reference for `scripts/train_supervised_vae.py` and the
Phase-2f sweep (Exps 20–23 in `scripts/sweep_disentanglement.py`). It covers
the motivation, the loss derivation, why the recommended configuration looks
the way it does, the angular-reconstruction metric, and every command needed
to reproduce the sweep and read its results.

---

## 1. Motivation

Locatello et al. (2019, *Challenging Common Assumptions in the Unsupervised
Learning of Disentangled Representations*) prove that **disentanglement is
unidentifiable without inductive bias or supervision**: an unsupervised VAE
can match the data distribution while assigning ground-truth factors to any
latent dimension, and which assignment it lands on is a function of seed and
initialisation. Empirically on dSprites, vanilla β-VAE / FactorVAE reliably
isolate scale and pos_x / pos_y because those factors carry the most pixel
variance, but **which** of the latent dims encodes them changes from run to
run.

For our downstream goal — **compositional diffusion conditioned on (scale,
orientation)** — we need the opposite: a guaranteed mapping from a known set
of latent indices to a known set of factors, so the diffusion conditioner can
intervene on `(z_s, z_o)` without an alignment step.

Locatello's impossibility says we cannot do this *without supervision*, but it
also says we don't need much: if we name the two dims and provide labels for
the two factors, axis-controllability is achievable. This trainer is the
minimal realisation of that idea.

Two factors, two structural choices:

* **Scale.** A bounded scalar (6 classes on dSprites). Trivially regressable
  to a single latent dim with MSE.
* **Orientation.** A circular variable on S¹, complicated by rotational
  symmetry: the square has 4-fold symmetry (orientations 0°, 90°, 180°, 270°
  produce identical pixels), the ellipse 2-fold, the heart 1-fold. A naive
  scalar regression cannot represent S¹, and naive `(sin θ, cos θ)`
  regression gives the encoder contradictory gradient on the square.

The trainer addresses both with a 2-D `(sin(k·θ), cos(k·θ))` target where
`k` is the shape's rotational symmetry order; this folds the symmetry group's
fundamental domain bijectively onto the unit circle.

---

## 2. Loss derivation

### 2.1 β-VAE ELBO baseline

For a factorised Gaussian posterior `q(z | x) = ∏_d N(μ_d, σ_d²)` and a
standard Normal prior `p(z) = N(0, I)`, the β-VAE objective is

$$
\mathcal{L}_{\text{ELBO}}
\;=\;
\mathbb{E}_{q}\!\bigl[\log p(x \mid z)\bigr]
\;-\;
\beta \,
D_{\mathrm{KL}}\!\bigl(q(z \mid x)\;\big\|\;p(z)\bigr) .
$$

Because the prior factorises across dims, **so does the KL**:

$$
D_{\mathrm{KL}}(q\Vert p) \;=\; \sum_{d=1}^{D} D_{\mathrm{KL},d},
\qquad
D_{\mathrm{KL},d} \;=\; -\tfrac{1}{2}\bigl(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2\bigr).
$$

So a *per-dimension* β is well-defined and corresponds exactly to a different
KL weight on each dim:

$$
\mathcal{L}_{\text{KL}} \;=\; \sum_d \beta_d \, D_{\mathrm{KL},d}.
$$

The trainer uses `β_d = β_supervised` for `d ∈ {0, 1, 2}` and
`β_d = β_free` for the rest.

### 2.2 Auxiliary supervision

Two MSE terms, computed on the deterministic posterior mean `μ` (not the
sampled `z`) so the supervision is invariant to reparameterisation noise:

$$
\mathcal{L}_{\text{aux}}
\;=\;
\lambda_s \,
\bigl\lVert \mu_0 - \tilde s \bigr\rVert^2
\;+\;
\lambda_o \,
\bigl\lVert \mu_{1{:}3} - (\sin k\theta,\; \cos k\theta) \bigr\rVert^2 ,
$$

with `s̃ = (scale_idx − 2.5) / 2.5` (rescaled to ≈ [-1, 1]) and `k`
the shape's symmetry order.

### 2.3 Why `k·θ` and not just `θ`

For a shape with rotational symmetry order `k`, the orientations
`{θ, θ + 2π/k, θ + 4π/k, …}` produce **identical pixels**. If we regressed
to `(sin θ, cos θ)` directly, the encoder would be asked to produce different
targets for indistinguishable inputs — a contradiction that flattens the
gradient on those slots. Mapping through `k` folds the symmetry group's
fundamental domain onto a single revolution of the circle, so each pixel
configuration has a unique target on S¹. This is what makes the recommended
config work for the square (k=4) and ellipse (k=2) without filtering them out
of the dataset.

### 2.4 Free-bits floor (optional)

Free-bits (Kingma et al., 2016) clamps the per-dim KL contribution from below:

$$
\widetilde{D}_{\mathrm{KL},d} \;=\; \max(D_{\mathrm{KL},d}, F),
$$

applied only to the *free* dims (so the supervised dims aren't artificially
inflated). Set with `--free-bits F` (in nats per dim per sample). 0 disables.
Useful when β is pushing free dims into posterior collapse.

### 2.5 Total loss

$$
\boxed{
\mathcal{L}
\;=\;
\underbrace{\mathrm{BCE}(\hat x, x) / B}_{\text{recon}}
\;+\;
\underbrace{\sum_d \beta_d\, \widetilde D_{\mathrm{KL},d}}_{\text{KL (per-dim β + free-bits)}}
\;+\;
\lambda_s \,\mathcal{L}_s
\;+\;
\lambda_o \,\mathcal{L}_o .
}
$$

---

## 3. Why `β_supervised = 0` (and not a small positive number)

The v1 run (`run-20260428_173903-lyz4o63h`, `λ=1, β_sup=1`) gave a clean
diagnostic of the failure mode. At the end of training:

| term       | value (nats / sample) | share of total |
|------------|-----------------------|----------------|
| recon (BCE summed over pixels, /B) | 16.4 | 40 % |
| β·KL (β=1) | 24.1 | 58 % |
| λ_s · aux_scale  (λ_s=1) | 0.34 | 0.8 % |
| λ_o · aux_orient (λ_o=1) | 0.38 | 0.9 % |
| **total**  | **41.2** | 100 % |

`aux_orient = 0.38` against a predict-zero baseline of 0.5 — only 24%
reduction. `z_1` collapsed (KL = 0.27); `z_2` absorbed orientation alone.

**Diagnosis.** On supervised dims, the KL term wants `μ → 0, σ → 1` while
the MSE wants `μ → target on the unit circle`. With `β_sup = 1`, the KL
reliably wins on the dim with smaller MSE gradient (z_1), driving it to
collapse, and the model takes the easier path of putting all the orientation
signal on z_2 alone — at which point *that* gradient is partially fighting
KL too.

**Fix.** Treat `z_s` and `z_o` as **labeled regression outputs**, not latent
variables. With `β_sup = 0` the supervised dims have no KL pressure, the
encoder is free to place `μ` wherever the supervision asks (notably on the
unit circle for orientation), and `σ` collapses to small values naturally
because the MSE supervises `μ`. The remaining `D − 3` free dims keep the
β-VAE prior and behave as before.

This decoupling also means **the same recipe transfers across β values** —
if you sweep β on the free dims, the supervised dims do not change behaviour,
because they are no longer governed by β at all.

---

## 4. Validation metrics

Two metrics that don't suffer from the circular-target problem (Spearman, MI,
and Random-Forest importance all underestimate circular encodings, sometimes
giving zero for a perfect ring):

* **`val/scale_r2`** — coefficient of determination of `μ_0` against the
  normalised scale target. ≥ 0.9 strong, 0.5–0.9 moderate, < 0.5 weak.
* **`val/orient_angular_err_deg`** — the formula is
  `mean( |wrap_to_π( atan2(z_2, z_1) − k·θ_true )| / k )` in degrees, where
  the wrap-to-π fold is the only correct way to compute angular distance.
  Thresholds: ≤ 10° strong, 10–30° moderate, ≥ 30° essentially no signal.

Both are computed in `src/utils/factor_targets.py` (`scale_r2`,
`orient_angular_error_deg`), reusable from any eval / analysis script.

---

## 5. Run commands

All commands assume the `phase2-repr` micromamba env, the project root as cwd,
and `wandb login` already done.

### 5.1 Single training run

```bash
python scripts/train_supervised_vae.py \
    --config configs/supervised_vae.yaml \
    --runtime hippo \
    --lambda-scale 10 --lambda-orient 10 --beta-supervised 0 \
    --seed 0
```

Knobs:

| flag                    | role                                                                | default |
|-------------------------|---------------------------------------------------------------------|---------|
| `--lambda-scale`        | weight on `MSE(μ_0, scale_target)`                                  | 5.0     |
| `--lambda-orient`       | weight on `MSE(μ_{1:3}, orient_target)`                             | 5.0     |
| `--beta-supervised`     | KL weight on `z_0, z_1, z_2`. 0 = labeled outputs                   | 0.0     |
| `--free-bits`           | floor (nats) on per-dim KL of *free* dims; 0 = disabled             | 0.0     |
| `--beta`                | KL weight on free dims (β-VAE β)                                    | 1.0     |
| `--latent-dim`          | total D                                                             | 10      |
| `--seed` / `--epochs` / `--batch-size` / `--num-workers` | standard | per-YAML |

### 5.2 One sweep cell via the dispatcher

The dispatcher pins a sweep cell's hyperparameters by `--experiment-id`, so
you don't have to remember the four sets of `λ` × `β_sup` values:

```bash
python scripts/sweep_disentanglement.py \
    --experiment-id 23 --seed 0 --runtime hippo
```

Cells: 20 = sup-A diagnostic, 21 = sup-B brute-λ, 22 = sup-C per-dim β alone,
23 = sup-D recommended.

### 5.3 Whole sweep, three seeds

```bash
for s in 0 1 2; do
    bash scripts/launch_sweep.sh --node hippo-supervised --seed "$s"
done
```

This launches all four cells per seed sequentially on a single GPU
(~25 min each at default batch_size=1024, so ≈100 min/seed). Yields 12
checkpoints under `checkpoints/vae/supervised_vae_*`.

For the cluster (2× 48 GB), use `--node mscluster106-supervised` (cells 20 & 21
in parallel) and `--node mscluster107-supervised` (cells 22 & 23 in parallel).

Dry-run to inspect the commands without executing:

```bash
bash scripts/launch_sweep.sh --node hippo-supervised --seed 0 --dry-run
```

### 5.4 Reading the results in the explorer

```bash
python scripts/disentanglement_explorer.py --port 5050 --device cpu
```

The "Supervised (targeted)" group renders Exps 20–23. Each card's seed
dropdown will populate as more seeds are trained.

---

## 6. Reading the results

For each sweep cell, the *minimum* set of plots to look at:

1. **KL spectrum** (Analysis tab). After sup-D, expect z_0 KL ≈ 1–2 nats
   (it has to encode 6 scale classes), z_1 + z_2 ≈ 4–6 nats together (the
   ring), free dims with KL spread roughly like the β-VAE baseline.
2. **`val/orient_angular_err_deg`** in W&B. Expected per cell:
   - sup-A: ≈ 50° (random would be ≈ 60° on a uniform draw)
   - sup-B: 15–30° (brute-λ partially works)
   - sup-C: 10–25° (per-dim β alone helps)
   - sup-D: ≤ 10° (both fixes)
3. **KL Geometry 2D scatter** (Analysis tab):
   - `dim_x = z0, dim_y = z3, colour = scale` → smooth horizontal gradient on
     z_0, no structure on z_3.
   - `dim_x = z1, dim_y = z2, colour = orientation` (cyclic colormap) → for
     sup-D, expect points on a ring at radius ≈ 1, colour sweeping smoothly
     around it. For sup-A this view is the diagnostic for collapse.
4. **Cross-leakage check.** `dim_x = z0, dim_y = z1, colour = pos_x` → no
   gradient on either axis (pos_x lives in z_free).

Multi-seed stability — load each seed via the card's seed dropdown and
confirm the ring/scale geometry is the same. This is the controllability
claim the paper rests on.

---

## 7. Known limitations & semantic caveats

* **`(z_1, z_2)` encodes the orientation equivalence class within a shape.**
  Because the target depends on `k`, the same ring point means different
  physical angles for square vs. heart. This is a feature, not a bug, for
  conditional diffusion (where you condition on shape anyway), but should be
  flagged in the paper. Naive interventions that move the ring point without
  touching shape will misrender.
* **Spearman / MIG / Random-Forest importance are blind to circular
  encodings.** The viewer banners this in the Factor–Latent Correlation
  panel. Always cross-check orientation with `val/orient_angular_err_deg`.
* **Free-bits is off by default.** Turn it on (`--free-bits 0.5`) only if a
  sweep cell shows widespread collapse on free dims; otherwise it's noise.

---

## 8. File map

| component                      | path                                                                   |
|--------------------------------|------------------------------------------------------------------------|
| Trainer                        | [scripts/train_supervised_vae.py](../scripts/train_supervised_vae.py)  |
| Config                         | [configs/supervised_vae.yaml](../configs/supervised_vae.yaml)          |
| Targets + metrics              | [src/utils/factor_targets.py](../src/utils/factor_targets.py)          |
| Sweep dispatcher (Exps 20–23)  | [scripts/sweep_disentanglement.py](../scripts/sweep_disentanglement.py) |
| Sweep launcher (`-supervised`) | [scripts/launch_sweep.sh](../scripts/launch_sweep.sh)                  |
| Explorer registry              | [scripts/disentanglement_explorer.py](../scripts/disentanglement_explorer.py) |
