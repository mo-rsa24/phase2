# Visual Sample Inspection — gaussian_rotation

**Generated**: 2026-06-16  
**Modality**: 2-D synthetic point cloud  
**Script**: `figures/eda/scripts/03_visualize_gaussian_rotation.py`

---

## Figure 1 — Data distribution at three rotation angles

![data scatter](01_data_scatter.png)

**Top row** coloured by c₁, **bottom row** by c₂, for θ ∈ {0, π/8, π/4}.

The data cloud is **always circular** (isotropic): the rotation R(θ) preserves
the distribution shape because c₁ and c₂ are drawn from the same uniform and
R is orthogonal.  What changes is the **colour gradient direction**:

| θ | What you see |
|---|---|
| 0 | c₁ stripes run left–right (x₀ ≈ c₁); c₂ stripes run up–down (x₁ ≈ c₂) |
| π/8 | gradients start to tilt — each xᵢ carries partial info on both factors |
| π/4 | gradients are diagonal; x₀ and x₁ each mix c₁ and c₂ **equally** |

**Key implication for a VAE**: at θ=0 a model that simply learns to copy xᵢ → zᵢ
already disentangles.  At θ=π/4 the model *must* learn a non-trivial inverse
rotation to recover the factors.

---

## Figure 2 — What the ideal VAE encoder learns

![latent space](02_latent_space.png)

Rows: observation space x (top) → latent space z coloured by c₁ (middle) →
z coloured by c₂ (bottom).  The latent z uses the closed-form optimal encoder

```
q*(z|x) = N(W·x, σ²·I),  W = (20/21)·R(−θ),  σ = 1/√21 ≈ 0.218
```

**The latent grid is identical across all three θ columns** for the middle row
(c₁ → z₀ left–right gradient) and for the bottom row (c₂ → z₁ up–down
gradient).  This shows:

- `z₀ ≈ c₁` and `z₁ ≈ c₂` regardless of how rotated the observations are.
- The encoder undoes the rotation: W = (20/21)·R(−θ).
- Disentanglement in latent space is achievable at every θ — as long as the
  encoder is powerful enough to learn a full 2×2 matrix (not just diagonal weights).

**Practical note**: the trained checkpoints in `checkpoints/gaussian/` have
collapsed posteriors (std ≈ 1 ≈ prior, encoder weights near zero).  The figures
above use the analytic optimal encoder.  Better training (lower β, more epochs,
or a learning-rate schedule) is needed before the numerical PoE experiments.

---

## Figure 3 — PoE posterior geometry in latent space

![PoE geometry](03_poe_geometry.png)

For θ = 0 (left) and θ = π/4 (right):

- **Blue cluster + ellipse**: reference set {c₁ ≈ 1, c₂ ~ U[−1,1]} encoded by
  the ideal encoder.  The ellipse captures both per-sample noise and the
  variation from c₂ being unknown.
- **Red cluster + ellipse**: reference set {c₂ ≈ 1, c₁ ~ U[−1,1]}.
- **Green ellipse**: PoE product of the two reference posteriors
  (Λ_PoE = Λ₁ + Λ₂ − Λ_prior, μ_PoE = Λ_PoE⁻¹·(Λ₁μ₁ + Λ₂μ₂)).
- **★**: true target — encoding of the joint point (c₁=1, c₂=1).

**What to look for**:

At θ=0 the blue ellipse is a **vertical slab** (certain about z₀ = c₁, uncertain
about z₁ = c₂) and the red ellipse is a **horizontal slab** (certain about z₁,
uncertain about z₀).  Their PoE product is a small ellipse pinned near the target.

At θ=π/4 the ideal encoder still maps z ≈ c, so the ellipses are similarly
shaped — and PoE still lands near the target.  **With the ideal encoder** the
reference-set precision is anisotropic at both angles, so PoE works.

The PoE failure (scaling as |sin(2θ)|) arises when the encoder is constrained to
use **per-sample isotropic precision** (constant σ, no reference-set averaging):
in that case the PoE mean is always (μ₁+μ₂)/2, undershooting the target by ≈50%.
That failure is what Plans 01 T4–T6 measure numerically.

---

## What to flag

- Trained checkpoints have collapsed posteriors — fix training before running
  `scripts/plot_poe_error_curve.py` or `scripts/plot_posterior_geometry.py`.
- The data generator itself (`src/data/gaussian_rotation.py`) is correct — the
  data scatter and latent recovery both look as expected.
