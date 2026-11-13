---
name: project-pivot-2026
description: Paper2 pivoted from dSprites VAE baseline to three-stage Gaussian/PoE/DDPM scope in June 2026
metadata:
  type: project
---

As of 2026-06-14, the project has pivoted from its original dSprites disentanglement scope to a new three-stage experiment:

- **Stage 1:** Gaussian analytic proof — rotation-mixing toy, formal PoE failure derivation
- **Stage 2:** Gaussian numerical verification — three model variants on rotated synthetic data
- **Stage 3:** 3D Shapes + FiLM-conditioned DDPM — full compositional generation pipeline

**Why:** The SepVAE/GRL/dSprites approach (Phases 2b/2d/2e/2g) was superseded. The new direction tests compositional generation through Product-of-Experts composition, motivated by the failure of entangled posteriors.

**What is complete from dSprites work:**
- IID sweep Exps 1–5 (β-VAE family + FactorVAE γ=35) — all checkpoints present
- Multi-seed Exp 1 (β=1, seeds 0–3, 42)
- DCI metric (`src/metrics/dci.py`, reusable)
- Supervised VAE Exp 20 only
- Dataset split infra (correlated_dsprites.py)

**How to apply:** MASTER_PLAN.md is the canonical active roadmap. PHASE_2_ROADMAP.md is historical with superseded banners. dSprites docs archived to `docs/archived-dsprites/`.
