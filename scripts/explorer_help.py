"""Glossary / interpretation strings for the disentanglement explorer UI.

Each entry has:
  title    — short header shown at the top of the popover
  short    — one-line tooltip (≤ 120 chars)
  long     — HTML body for the popover
  formula  — optional KaTeX string rendered into the popover
  links    — optional list of CSS selectors (UI elements that the term refers to;
             these get a brief pulse animation when the popover is opened)

The dict is serialized to JSON and exposed to the browser as `window.HELP`.
"""

HELP: dict[str, dict] = {
    # ── Latent variables ────────────────────────────────────────────────────
    "mu": {
        "title": "Posterior mean μ",
        "short": "μ = where the encoder places this image's distribution along this latent dim. 0 = on the prior.",
        "long": (
            "<p>For each image <i>x</i>, the encoder outputs a Gaussian posterior "
            "<i>q(z|x)</i> = 𝒩(μ, σ²) per latent dim. The <b>μ</b> value is the "
            "centre of that distribution.</p>"
            "<p>μ = 0 means the posterior sits exactly on the prior 𝒩(0,1). "
            "Large |μ| means the encoder pushed this image far from the prior — "
            "an image-specific signature.</p>"
            "<p>Sign carries information too: μ = −1.22 vs μ = +1.22 may correspond "
            "to opposite ends of the same factor (e.g. small vs large scale).</p>"
        ),
        "formula": r"q_\phi(z_i | x) = \mathcal{N}(\mu_i(x),\, \sigma_i^2(x))",
        "links": ["#mu-bars"],
    },
    "sigma": {
        "title": "Posterior std σ",
        "short": "σ = how peaked vs vague the encoder is about this dim. σ ≈ 1 = ~prior; σ ≪ 1 = confident.",
        "long": (
            "<p>σ is the posterior standard deviation per latent dim, derived "
            "from the encoder's <code>logvar</code> output: σ = exp(½·logvar).</p>"
            "<p>σ ≈ 1 means the posterior is as wide as the prior — the model "
            "can't distinguish this image's value of <i>z<sub>i</sub></i> from any other. "
            "σ ≪ 1 means a sharp spike: the encoder has committed to a precise value.</p>"
        ),
        "formula": r"\sigma_i = \exp\!\left(\tfrac{1}{2}\,\log\sigma_i^2\right)",
        "links": [],
    },
    "logvar": {
        "title": "Log-variance",
        "short": "What the encoder actually outputs. Reparametrized to σ via exp(½·logvar).",
        "long": (
            "<p>The encoder outputs <code>logvar = log(σ²)</code> directly because "
            "it's numerically stable across many orders of magnitude.</p>"
            "<p>logvar = 0 → σ = 1 (matches prior). logvar = −4 → σ ≈ 0.14.</p>"
        ),
        "formula": r"\sigma_i^2 = e^{\log\sigma_i^2}",
        "links": [],
    },
    "prior": {
        "title": "Prior p(z)",
        "short": "Standard isotropic Gaussian 𝒩(0, I). The KL pulls the posterior toward this.",
        "long": (
            "<p>The VAE assumes <i>z</i> is drawn from 𝒩(0, I) — independent unit "
            "Gaussians, centred at the origin.</p>"
            "<p>The objective penalises any encoder posterior that drifts away from "
            "this. Far-from-prior posteriors only survive when the reconstruction "
            "term gains enough to pay the KL cost.</p>"
        ),
        "formula": r"p(z) = \mathcal{N}(0, I)",
        "links": ["#geom-1d-card"],
    },
    "posterior": {
        "title": "Posterior q(z|x)",
        "short": "What the encoder produces — a Gaussian over z for each image x.",
        "long": (
            "<p>The encoder is the variational posterior. Per latent dim it returns "
            "(μ, σ) defining a Gaussian over <i>z<sub>i</sub></i> conditional on the input.</p>"
            "<p>Disentanglement = each posterior dim peaks for distinct factor values "
            "and stays near-prior for unused dims.</p>"
        ),
        "formula": r"q_\phi(z|x) = \prod_i \mathcal{N}(\mu_i(x),\, \sigma_i^2(x))",
        "links": ["#mu-bars", "#geom-1d-card"],
    },

    # ── KL divergence ───────────────────────────────────────────────────────
    "kl": {
        "title": "KL divergence (per dim)",
        "short": "Information cost of moving the posterior away from the prior. 0 = collapsed; high = active.",
        "long": (
            "<p>The KL divergence between two Gaussians has a closed form. "
            "For dim <i>i</i>:</p>"
            "<p>KL = 0 → posterior = prior → dim is <b>collapsed</b> (encodes "
            "nothing). KL ≫ 0 → encoder commits to a specific (μ, σ) for each "
            "image → dim is <b>active</b>.</p>"
            "<p>Counted in <b>nats</b> (natural-log units). 1 nat ≈ 1.443 bits.</p>"
        ),
        "formula": (
            r"\mathrm{KL}\!\left(\mathcal{N}(\mu_i, \sigma_i^2) \,\|\, \mathcal{N}(0,1)\right) "
            r"= \tfrac{1}{2}\!\left(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2\right)"
        ),
        "links": ["#kl-plot", "#geom-1d-card"],
    },
    "kl_term": {
        "title": "KL term in the ELBO",
        "short": "Sum of per-dim KLs — total information cost of the posterior.",
        "long": (
            "<p>The full KL term in the β-VAE objective is the sum of per-dim KLs "
            "(latent dims are independent under the diagonal-Gaussian assumption).</p>"
            "<p>The KL spectrum visualises each summand in this total.</p>"
        ),
        "formula": (
            r"D_{KL}\!\left(q_\phi(z|x) \,\|\, p(z)\right) = "
            r"\sum_{i=1}^{d_z} \mathrm{KL}_i"
        ),
        "links": ["#kl-plot"],
    },
    "active_dim": {
        "title": "Active dim (KL > 0.1)",
        "short": "A dim with mean KL > 0.1 nats — meaningfully encoding information about the input.",
        "long": (
            "<p>0.1 nats is a heuristic threshold: dims below it are effectively "
            "ignored by the decoder (a phenomenon called <b>posterior collapse</b>).</p>"
            "<p>Only active dims can encode dSprites factors like scale or "
            "orientation. Collapsed dims are dead capacity.</p>"
        ),
        "links": ["#kl-plot"],
    },
    "nat": {
        "title": "Nat",
        "short": "Natural-log unit of information. 1 nat = log₂(e) ≈ 1.443 bits.",
        "long": (
            "<p>KL and entropy in this app are reported in <b>nats</b> because "
            "the underlying formulas use natural logarithms.</p>"
            "<p>To convert: <i>n</i> nats × 1.443 = bits. So KL = 4.0 nats ≈ "
            "5.77 bits of information in this dim per image.</p>"
        ),
        "formula": r"1\ \mathrm{nat} = \log_2(e) \approx 1.443\ \mathrm{bits}",
        "links": [],
    },

    # ── Objective ───────────────────────────────────────────────────────────
    "elbo": {
        "title": "β-VAE objective (ELBO)",
        "short": "Reconstruction term minus β times KL. Maximised during training.",
        "long": (
            "<p>The β-VAE training loss is the negative ELBO:</p>"
            "<p><b>Reconstruction</b>: how well the decoder rebuilds <i>x</i> from "
            "<i>z</i>. Visible in the |Difference| panel (Explore tab).</p>"
            "<p><b>KL term</b>: total information cost. Visible in the KL "
            "Activity Spectrum (Analysis tab).</p>"
            "<p><b>β</b>: weight on the KL term. β > 1 promotes disentanglement at "
            "the cost of reconstruction quality.</p>"
        ),
        "formula": (
            r"\mathcal{L}_{\beta\text{-VAE}} = "
            r"\underbrace{\mathbb{E}_{q_\phi(z|x)}\bigl[\log p_\theta(x|z)\bigr]}_{\text{reconstruction}} "
            r"- \beta \cdot \underbrace{D_{KL}\bigl(q_\phi(z|x)\,\|\,p(z)\bigr)}_{\text{regularisation}}"
        ),
        "links": ["#diff-box", "#kl-plot"],
    },
    "beta": {
        "title": "β (KL weight)",
        "short": "How hard the regulariser pulls posteriors toward the prior. β=1 is vanilla VAE.",
        "long": (
            "<p>β = 1 → standard VAE (well-calibrated probabilistic model).</p>"
            "<p>β > 1 → β-VAE. Higher β = stronger prior pull = more dims collapse "
            "and the survivors are pushed to be axis-aligned with factors of "
            "variation. This is the disentanglement / reconstruction trade-off.</p>"
        ),
        "links": [],
    },
    "recon": {
        "title": "Reconstruction term",
        "short": "Negative reconstruction error (BCE or MSE). Higher = decoder rebuilds the input faithfully.",
        "long": (
            "<p>For binary dSprites images, this is the negative binary "
            "cross-entropy between the decoder output and the input pixels.</p>"
            "<p>The <b>|Difference|</b> panel visualises where the reconstruction "
            "is failing. Sharp residuals = under-fitting; uniform low residuals = "
            "good reconstruction.</p>"
        ),
        "links": ["#diff-box"],
    },

    # ── Disentanglement metrics ─────────────────────────────────────────────
    "rho": {
        "title": "|Spearman ρ|",
        "short": "Rank correlation between a latent dim and a factor. 1 = perfect monotonic; 0 = none.",
        "long": (
            "<p>Spearman's ρ measures monotonic association between μ<sub>i</sub> "
            "and a factor value (e.g. scale level 0..5).</p>"
            "<p>We take the absolute value because direction (+/−) is an arbitrary "
            "convention of the encoder.</p>"
            "<p><b>Caveat</b>: ρ assumes a monotonic relationship. For cyclic "
            "factors (orientation), it underestimates true alignment.</p>"
        ),
        "formula": r"|\rho_S(z_i, v_k)| \in [0, 1]",
        "links": ["#corr-plot"],
    },
    "spearman": {
        "title": "Spearman rank correlation",
        "short": "Pearson correlation on ranks instead of raw values — robust to monotonic non-linearities.",
        "long": (
            "<p>Spearman replaces values with their ranks then computes Pearson "
            "correlation. This makes it invariant to any monotonic transformation "
            "of either variable.</p>"
            "<p>Useful when the encoder learns a non-linear (but monotonic) mapping "
            "from factor → latent.</p>"
        ),
        "links": ["#corr-plot"],
    },
    "cyclic_orientation": {
        "title": "Why orientation is special",
        "short": "Orientation is cyclic (40 bins on a circle). Linear/rank correlation underestimates encoding.",
        "long": (
            "<p>dSprites orientation is a <b>cyclic</b> factor: index 0 and 39 are "
            "almost identical (≈ 0° and ≈ 351°), but rank correlation treats them "
            "as far apart.</p>"
            "<p>Effect: a model can perfectly encode orientation in a single latent "
            "dim and still show low |Spearman ρ|.</p>"
            "<p><b>Workaround</b>: cross-check with the latent traversal grid "
            "(visual) and the orientation conditional histogram (uses MI, which "
            "handles cyclic factors correctly).</p>"
        ),
        "links": ["#corr-plot"],
    },
    "mig": {
        "title": "Mutual Information Gap (MIG)",
        "short": "How clearly each factor is owned by exactly one latent dim. ∈ [0,1]; higher = better.",
        "long": (
            "<p>For each factor <i>v<sub>k</sub></i>, MIG measures the gap between "
            "the most informative latent dim and the second-most informative one, "
            "normalised by the factor's entropy.</p>"
            "<p>Big gap → one dim cleanly owns the factor (axis-aligned code). "
            "Small gap → the factor is split across multiple dims (entangled).</p>"
            "<p>Rough guidance: ≥ 0.4 strong, 0.2–0.4 moderate, &lt; 0.2 poor "
            "(loosely from Chen et al. 2018).</p>"
        ),
        "formula": (
            r"\mathrm{MIG} = \frac{1}{K} \sum_{k=1}^{K} "
            r"\frac{I(z_*; v_k) - I(z_2; v_k)}{H(v_k)}"
        ),
        "links": ["#mig-overall", "#mig-per-factor"],
    },
    "mig_overall": {
        "title": "Overall MIG score",
        "short": "Average per-factor MIG across all non-trivial factors. ∈ [0,1].",
        "long": (
            "<p>Single scalar summary of disentanglement. Higher is better.</p>"
            "<p>For β-VAE on dSprites, expected ranges (approx.): "
            "vanilla VAE ≈ 0.05–0.10, β=4 ≈ 0.15–0.30, dedicated disentanglement "
            "models can reach 0.4+.</p>"
        ),
        "links": ["#mig-overall"],
    },
    "mig_per_factor": {
        "title": "Per-factor MIG",
        "short": "Per-factor breakdown — tells you which factor is cleanly captured and which is entangled.",
        "long": (
            "<p>For your research question, the rows that matter are <b>scale</b> "
            "and <b>orientation</b>. A high score on either means a single latent "
            "dim owns that factor; a low score means it's distributed.</p>"
            "<p>Click any factor's bar (Phase 3) to see the conditional histogram "
            "of μ values for that factor's top dim — the visual evidence behind "
            "the number.</p>"
        ),
        "links": ["#mig-per-factor"],
    },

    # ── DCI metric (Eastwood & Williams 2018) ──────────────────────────────
    "dci": {
        "title": "DCI (Disentanglement / Completeness / Informativeness)",
        "short": "Three complementary scores from a Random-Forest importance matrix R(latent × factor).",
        "long": (
            "<p>For each ground-truth factor, a Random Forest is trained to "
            "predict it from the latents <i>μ</i>. The forest's "
            "<b>feature_importances_</b> form a matrix R[i, k] of how much "
            "latent <i>i</i> contributes to predicting factor <i>k</i>.</p>"
            "<p>Three scores are derived:</p>"
            "<ul><li><b>D</b>isentanglement: per-latent — is each dim "
            "concentrated on one factor?</li>"
            "<li><b>C</b>ompleteness: per-factor — is each factor "
            "concentrated in one dim?</li>"
            "<li><b>I</b>nformativeness: per-factor — can the factor be "
            "predicted at all? (held-out R²)</li></ul>"
            "<p>D and C address different failure modes: high D + low C = "
            "clean dims but factors are split; low D + high C = factors "
            "are localised but each dim multiplexes; both high = "
            "axis-aligned.</p>"
        ),
        "links": ["#dci-panel"],
    },
    "dci_d": {
        "title": "Disentanglement (D)",
        "short": "Per-latent: does this dim concentrate its predictive power on a single factor?",
        "long": (
            "<p>For latent dim <i>i</i>, normalise its row in the importance "
            "matrix to a probability vector P_i, then:</p>"
            "<p>D_i = 1 − H(P_i) / log K, with K = number of factors.</p>"
            "<p>D = 1 means the dim's importance is concentrated on a single "
            "factor (P_i is one-hot). D = 0 means the dim spreads its "
            "importance evenly across all K factors.</p>"
            "<p>Overall D is an importance-weighted mean — uninformative "
            "dims do not dilute the score.</p>"
        ),
        "formula": (
            r"D_i = 1 - \frac{H(P_i)}{\log K}, \quad "
            r"P_i = \frac{R[i, :]}{\sum_k R[i, k]}"
        ),
        "links": ["#dci-d"],
    },
    "dci_c": {
        "title": "Completeness (C)",
        "short": "Per-factor: is this factor's representation concentrated in a single latent dim?",
        "long": (
            "<p>For factor <i>k</i>, normalise its column in the importance "
            "matrix to a probability vector P_k, then:</p>"
            "<p>C_k = 1 − H(P_k) / log d, with d = latent_dim.</p>"
            "<p>C = 1 means the factor lives in one dim. C = 0 means it is "
            "split equally across every dim.</p>"
            "<p>For your RQ, scale and orientation should each have high C."
            " Low C on either means the factor is multi-dim entangled — even "
            "if MIG is moderate.</p>"
        ),
        "formula": (
            r"C_k = 1 - \frac{H(P_k)}{\log d}, \quad "
            r"P_k = \frac{R[:, k]}{\sum_i R[i, k]}"
        ),
        "links": ["#dci-c"],
    },
    "dci_i": {
        "title": "Informativeness (I)",
        "short": "Per-factor: can this factor be predicted from the latents at all? (held-out R²)",
        "long": (
            "<p>Held-out R² of the regressor that predicts factor <i>k</i> "
            "from <i>μ</i>. I = 1 → perfect prediction; I = 0 → as good as "
            "predicting the factor's mean.</p>"
            "<p>Low I means the factor isn't encoded anywhere. A model can "
            "have high D and high C but low I — meaning the latents are "
            "<i>structured</i> but the structure isn't <i>informative</i>. "
            "All three are needed.</p>"
        ),
        "formula": r"I_k = \max\bigl(0,\ R^2_{\text{test}}(\hat v_k(\mu),\, v_k)\bigr)",
        "links": ["#dci-i"],
    },
    "importance": {
        "title": "Feature importance matrix",
        "short": "R[i, k] = importance of latent dim i for predicting factor k (from a Random Forest).",
        "long": (
            "<p>The heart of DCI. A 2D matrix where each cell measures how "
            "much a Random Forest relies on a given latent dim to predict a "
            "given factor.</p>"
            "<p>Read like the |Spearman ρ| heatmap: ideal = exactly one "
            "bright cell per column (each factor owned by one dim) and at "
            "most one per row (each dim owns at most one factor).</p>"
            "<p>RF importance is non-linear (handles non-monotonic encodings "
            "better than Spearman) but treats the factor as a continuous "
            "regression target — for cyclic factors like orientation it "
            "still under-counts. Cross-check with the conditional histogram.</p>"
        ),
        "links": ["#dci-heatmap"],
    },

    # ── Other ────────────────────────────────────────────────────────────────
    "traversal": {
        "title": "Latent traversal",
        "short": "Sweep one latent dim through ±k σ while holding others fixed; decode each step.",
        "long": (
            "<p>The defining diagnostic for disentanglement: if a single dim "
            "encodes only scale, sweeping it should change <b>only</b> sprite "
            "size — not shape, position, or rotation.</p>"
            "<p>Cross-bleed between factors (e.g. sweeping a 'scale' dim also "
            "rotates the sprite) is the visual signature of entanglement.</p>"
        ),
        "links": ["#tab-grid"],
    },
    "delta_sigma": {
        "title": "Δσ from anchor",
        "short": "How far this slider has moved from the anchor μ, in units of the anchor's posterior σ.",
        "long": (
            "<p>Tells you the magnitude of your traversal in the posterior's own "
            "units. |Δσ| &lt; 0.5 = barely moved (still in-distribution). "
            "|Δσ| &gt; 2 = far out of the posterior — expect surprising decodings.</p>"
        ),
        "links": [],
    },
}
