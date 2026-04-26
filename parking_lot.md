Yes. This should be included, but not as another full implementation target. It should become a **design-thinking phase** for the weekend.

Add it as a dedicated block:

# New Weekend Phase: Method Design Logic

## Objective

Learn how to move from:

```text
failure → mathematical property → architecture/loss
```

This is the thinking pattern you need for Phase 2.

## Time

90–120 minutes.

## When

Put it **after VAE/dSprites setup** and **before FactorVAE/DCI/GRL skeletons**.

---

# What to Produce

Create one Zettelkasten note:

```text
notes/design_logic_correlated_dsprites.md
```

With this structure:

```markdown
# Design Logic: Correlated dSprites

## Failure
Correlated data lets the encoder merge scale and orientation into one joint explanation.

## Mathematical Target
z_scale should contain scale:
I(z_scale; c_scale) high

z_scale should not contain orientation:
I(z_scale; c_orientation) ≈ 0

z_orientation should contain orientation:
I(z_orientation; c_orientation) high

z_orientation should not contain scale:
I(z_orientation; c_scale) ≈ 0

## Architecture
Partition the latent space:
z = [z_common, z_scale, z_orientation]

## Losses
1. Reconstruction: keep image information.
2. KL: keep latent space regularized.
3. Correct-head prediction: make assigned factors recoverable.
4. GRL wrong-head loss: suppress leakage.
5. Swap loss: test/control interventions.

## Main Metric
Held-out joint recombination accuracy.
```

---

# Coding Task for This Phase

Do **not** train the whole method yet.

Create only a design skeleton:

```text
src/models/partitioned_vae.py
src/models/grl.py
src/losses/partitioned_losses.py
```

Minimum code objects:

```python
class PartitionedVAE(nn.Module):
    # outputs z_common, z_scale, z_orientation
```

```python
class GradientReversal(torch.autograd.Function):
    # forward identity, backward negative gradient
```

```python
def correct_head_loss(...):
    pass

def wrong_head_grl_loss(...):
    pass

def swap_consistency_loss(...):
    pass
```

The point is not performance. The point is to encode the **method design logic** into code structure.

---

# Updated Weekend Priority Order

Use this revised order:

1. **Repo + Zettelkasten setup**
2. **VAE math videos and compressed notes**
3. **dSprites loader + EDA**
4. **Vanilla VAE + β-VAE quick implementation**
5. **Design Logic: correlated dSprites**
6. **GRL + partitioned VAE skeleton**
7. **DCI/probe skeleton**
8. **FactorVAE skeleton only if time remains**

I would move **FactorVAE below GRL/partitioned VAE** because your actual research idea depends more directly on latent partitioning, leakage, and swapping.

---

# What This Adds to the Weekend

By including this phase, you are not just learning methods. You are practicing the research skill:

```text
What failure am I targeting?
What mathematical property would prevent it?
What architectural or loss constraint enforces that property?
How will I measure whether it worked?
```

That is exactly the mindset you need before using Claude Code.

Final adjusted weekend success target:

```text
[ ] I can explain VAE/β-VAE mathematically.
[ ] I can load and visualize dSprites.
[ ] I can train a baseline VAE.
[ ] I can construct correlated factor splits.
[ ] I can write the design logic from failure → math → loss.
[ ] I have a partitioned VAE + GRL skeleton.
[ ] I know what DCI and FactorVAE are for, even if not fully implemented.
```

So yes: include it. It is probably one of the most important blocks of the weekend.
You design it by moving through **three levels**:

1. **Intuition:** what failure are we trying to prevent?
2. **Math:** what property should the latent space satisfy?
3. **Architecture/loss:** what constraints force the model toward that property?

For correlated dSprites, the core failure is:

> The model sees that size and orientation often co-occur, so it treats them as one joint explanation instead of two separately controllable causes.

So the goal is not just reconstruction. The goal is:

> Learn latent variables where changing “size” changes size, while orientation stays fixed.

---

# 1. Start from the intuition

A normal VAE learns:

[
x \rightarrow z \rightarrow \hat{x}
]

The encoder asks:

[
q_\phi(z \mid x)
]

meaning:

> Given this image, what latent code explains it?

But if size and orientation are correlated in the dataset, the encoder may learn:

[
z_1 = \text{size} + \text{orientation}
]

instead of:

[
z_{\text{size}} = \text{size only}
]

[
z_{\text{orientation}} = \text{orientation only}
]

So your first-principles diagnosis is:

> The encoder is allowed to use correlations as shortcuts. It is not forced to separate causal factors.

Then the disentanglement design problem becomes:

> How do we prevent the latent representation from storing one factor inside the latent space of another factor?

---

# 2. Turn the intuition into a mathematical target

Suppose the true factors are:

[
c = (c_{\text{size}}, c_{\text{orientation}})
]

and the latent space is partitioned as:

[
z = (z_{\text{size}}, z_{\text{orientation}}, z_{\text{common}})
]

A disentangled representation should satisfy three things.

## A. Sufficiency

The full latent code should still reconstruct the image:

[
x \approx D_\theta(z_{\text{size}}, z_{\text{orientation}}, z_{\text{common}})
]

So we need reconstruction quality.

This gives the usual VAE reconstruction term:

[
\mathcal{L}_{\text{rec}}
========================

-\mathbb{E}*{q*\phi(z \mid x)}
\left[
\log p_\theta(x \mid z)
\right]
]

Plainly:

> The latent code must contain enough information to rebuild the image.

---

## B. Correct factor information

The size latent should contain size information:

[
z_{\text{size}} \rightarrow c_{\text{size}}
]

The orientation latent should contain orientation information:

[
z_{\text{orientation}} \rightarrow c_{\text{orientation}}
]

So we want:

[
I(z_{\text{size}}; c_{\text{size}}) \text{ high}
]

[
I(z_{\text{orientation}}; c_{\text{orientation}}) \text{ high}
]

Plainly:

> The correct latent block should know about its assigned factor.

In practice, you approximate this with small classifier heads:

[
h_{\text{size}}(z_{\text{size}}) \rightarrow c_{\text{size}}
]

[
h_{\text{orientation}}(z_{\text{orientation}}) \rightarrow c_{\text{orientation}}
]

and add supervised or weakly supervised prediction losses.

---

## C. Cross-factor suppression

The size latent should **not** contain orientation information:

[
I(z_{\text{size}}; c_{\text{orientation}}) \approx 0
]

The orientation latent should **not** contain size information:

[
I(z_{\text{orientation}}; c_{\text{size}}) \approx 0
]

Plainly:

> If I look only at (z_{\text{size}}), I should not be able to predict orientation.

This is the most important disentanglement idea.

You are not only asking:

> Can the latent code reconstruct the data?

You are asking:

> Is each latent block forbidden from carrying the wrong information?

That is where architecture and loss design enter.

---

# 3. Translate the math into architecture

A standard VAE has one encoder output:

[
q_\phi(z \mid x)
]

For disentanglement, you make the latent structure explicit:

[
q_\phi(z \mid x)
================

q_\phi(z_{\text{common}}, z_{\text{size}}, z_{\text{orientation}} \mid x)
]

Then implement the encoder as multiple heads:

```text
Image x
  |
Shared CNN encoder
  |
  |--- z_common
  |--- z_size
  |--- z_orientation
```

Then the decoder receives all blocks:

```text
[z_common, z_size, z_orientation]
  |
Decoder
  |
Reconstructed image x_hat
```

The architectural bias says:

> There is a dedicated place for size information and a dedicated place for orientation information.

But architecture alone is not enough. The model can still leak orientation into (z_{\text{size}}). So you need losses.

---

# 4. Translate the math into losses

The full objective could look like:

[
\mathcal{L}
===========

\mathcal{L}*{\text{rec}}
+
\beta \mathcal{L}*{\text{KL}}
+
\lambda_{\text{right}}\mathcal{L}*{\text{right}}
+
\lambda*{\text{wrong}}\mathcal{L}*{\text{wrong}}
+
\lambda*{\text{TC}}\mathcal{L}*{\text{TC}}
+
\lambda*{\text{swap}}\mathcal{L}_{\text{swap}}
]

Each term has a role.

---

## A. Reconstruction loss

[
\mathcal{L}_{\text{rec}}
========================

-\mathbb{E}*{q*\phi(z \mid x)}
[
\log p_\theta(x \mid z)
]
]

This says:

> The latent code must still explain the image.

Without this, the representation may become clean but useless.

---

## B. KL loss

[
\mathcal{L}_{\text{KL}}
=======================

D_{\text{KL}}
\left(
q_\phi(z \mid x)
\Vert
p(z)
\right)
]

This says:

> Keep the latent space organized and sampleable.

But be careful: if (\beta) is too high, the model may ignore useful information and reconstruction collapses. So you usually schedule (\beta) gradually.

---

## C. Correct-head prediction loss

For size:

[
\mathcal{L}_{\text{size}}
=========================

\text{CE}
\left(
h_{\text{size}}(z_{\text{size}}), c_{\text{size}}
\right)
]

For orientation:

[
\mathcal{L}_{\text{orientation}}
================================

\text{CE}
\left(
h_{\text{orientation}}(z_{\text{orientation}}), c_{\text{orientation}}
\right)
]

This says:

> The size head must predict size. The orientation head must predict orientation.

So:

[
\mathcal{L}_{\text{right}}
==========================

\mathcal{L}*{\text{size}}
+
\mathcal{L}*{\text{orientation}}
]

---

## D. Wrong-head suppression loss

Now add adversarial heads.

Try to predict orientation from (z_{\text{size}}):

[
a_{\text{orientation}}(z_{\text{size}}) \rightarrow c_{\text{orientation}}
]

Try to predict size from (z_{\text{orientation}}):

[
a_{\text{size}}(z_{\text{orientation}}) \rightarrow c_{\text{size}}
]

But through a gradient reversal layer, the encoder is trained to make those predictions fail.

So the adversary asks:

> Can I recover the wrong factor from this latent block?

The encoder learns:

> Make that impossible.

This gives:

[
z_{\text{size}} \not\rightarrow c_{\text{orientation}}
]

[
z_{\text{orientation}} \not\rightarrow c_{\text{size}}
]

Plainly:

> The size latent should be useful for size, but useless for orientation.

This is one of the most direct ways to turn your intuition into a loss.

---

## E. Total correlation penalty

A FactorVAE-style loss penalizes dependence among latent dimensions:

[
\mathcal{L}_{\text{TC}}
=======================

D_{\text{KL}}
\left(
q(z)
\Vert
\prod_j q(z_j)
\right)
]

This says:

> The latent dimensions should not be statistically tangled together.

This helps, but by itself it may not solve correlated dSprites because the dataset correlation can still dominate. So TC is useful, but I would not rely on it alone.

---

## F. Swap or intervention consistency loss

This is the most “causal” part.

Take two examples:

[
x_a = \text{small object, vertical}
]

[
x_b = \text{large object, diagonal}
]

Encode both:

[
z_a = (z_{\text{size}}^a, z_{\text{orientation}}^a)
]

[
z_b = (z_{\text{size}}^b, z_{\text{orientation}}^b)
]

Now swap only size:

[
\tilde{z}
=========

(z_{\text{size}}^b, z_{\text{orientation}}^a)
]

Decode:

[
\tilde{x}
=========

D_\theta(\tilde{z})
]

The desired result is:

> Same orientation as (x_a), but size from (x_b).

Then use external factor classifiers to enforce:

[
f_{\text{size}}(\tilde{x}) = c_{\text{size}}^b
]

[
f_{\text{orientation}}(\tilde{x}) = c_{\text{orientation}}^a
]

This gives a recombination loss:

[
\mathcal{L}_{\text{swap}}
=========================

\text{CE}(f_{\text{size}}(\tilde{x}), c_{\text{size}}^b)
+
\text{CE}(f_{\text{orientation}}(\tilde{x}), c_{\text{orientation}}^a)
]

Plainly:

> If I swap only the size latent, only size should change.

This is the strongest test of disentanglement because it checks controllability, not just reconstruction.

---

# 5. The design logic in one pipeline

You can think of the whole method like this:

```text
Problem:
  Correlated data makes the encoder merge factors.

Mathematical goal:
  z_size should contain size and not orientation.
  z_orientation should contain orientation and not size.

Architecture:
  Use separate latent heads:
    z_common, z_size, z_orientation.

Losses:
  Reconstruction:
    preserve image information.

  KL:
    regularize the latent space.

  Correct-head prediction:
    make z_size informative about size.
    make z_orientation informative about orientation.

  Wrong-head adversarial loss:
    prevent z_size from carrying orientation.
    prevent z_orientation from carrying size.

  Total correlation:
    discourage statistical dependence among latent dimensions.

  Swap/intervention loss:
    enforce that editing one latent block changes only one factor.
```

---

# 6. The most important principle

The key shift is this:

A normal VAE asks:

> Can (z) reconstruct (x)?

A disentangled VAE asks:

> Can each part of (z) explain exactly one controllable cause of (x), without carrying the others?

That means you are designing for **controlled intervention**, not just compression.

---

# 7. A clean research framing

You could write the idea like this:

> Starting from the view that the encoder infers latent causes from observations, disentanglement can be framed as a constraint on the quality of this inference. In correlated datasets, a standard encoder may compress co-occurring factors into shared latent directions, thereby representing empirical correlations rather than independently controllable causes. We therefore introduce an architectural partition of the latent space into factor-specific blocks and train each block to be informative about its assigned factor while adversarially suppressing information about other factors. Reconstruction preserves generative sufficiency, KL regularization maintains a structured latent prior, and intervention-based swapping losses test whether manipulating one block changes only its corresponding factor. In this view, disentanglement is not merely improved reconstruction; it is the enforcement of factor-specific sufficiency and cross-factor invariance.

---

# 8. For your Phase 2 experiment, the clean design would be

For correlated dSprites:

```text
Latents:
  z_common
  z_shape
  z_scale
  z_orientation
  z_position_x
  z_position_y
```

For your first controlled experiment, keep it simpler:

```text
z_common
z_scale
z_orientation
```

because your main test is:

> Can the model separate scale and orientation even when the dataset correlates them?

Use these metrics:

```text
1. Reconstruction quality
2. DCI disentanglement
3. Correct factor prediction from correct latent block
4. Failure of wrong-factor prediction from wrong latent block
5. Latent traversal purity
6. Held-out recombination accuracy
```

The most important final metric is held-out recombination:

[
\text{Rec}_{\text{joint}}
=========================

\frac{1}{N}
\sum_i
\mathbf{1}
[
(\hat{c}*{\text{scale}}^{(i)}, \hat{c}*{\text{orientation}}^{(i)})
==================================================================

(c_{\text{scale}}^{(i)}, c_{\text{orientation}}^{(i)})
]
]

because it directly tests whether the model can produce unseen factor combinations.

---

# 9. The simplest version of the method

Start with this minimal loss:

[
\mathcal{L}
===========

\mathcal{L}*{\text{rec}}
+
\beta \mathcal{L}*{\text{KL}}
+
\lambda_1 \mathcal{L}*{\text{right}}
+
\lambda_2 \mathcal{L}*{\text{wrong-GRL}}
+
\lambda_3 \mathcal{L}_{\text{swap}}
]

This is enough to express your core idea.

You do not need to begin with every possible disentanglement loss. The minimal story is:

```text
Reconstruct the image.
Put each factor in the right latent block.
Remove wrong-factor information from each block.
Verify by swapping blocks and checking whether only the intended factor changes.
```

That is the clean bridge from intuition, to math, to architecture, to loss design.
