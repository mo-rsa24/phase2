# Master Plan — Compositional Generation with Diffusion Models

> **Last audited:** 2026-06-14
> **Current phase:** Pivoting from dSprites baseline to three-stage Gaussian/PoE/DDPM scope.

---

## Completed Work (dSprites Phase)

All tasks below are verified against the repo as of the audit date.

| Scope | Status | Key artifacts |
|---|---|---|
| IID sweep — Exps 1–5 (β-VAE family + FactorVAE) | ✅ Done | `checkpoints/vae/`, `checkpoints/factor_vae/` |
| Multi-seed replication — Exp 1 (β=1, seeds 0–3,42) | ✅ Done | `checkpoints/vae/vae_z10_beta1.0_seed*/` |
| DCI metric — `compute_dci()` | ✅ Done | `src/metrics/dci.py` (188 lines) |
| DCI eval script + explorer endpoint | ✅ Done | `scripts/eval_dci.py`, `/api/dci/start` |
| Dataset split infra (IID / correlated / held-out) | ✅ Done | `src/datasets/correlated_dsprites.py` |
| Supervised VAE trainer + docs (Exp 20 only) | ✅ Partial | `scripts/train_supervised_vae.py`, `docs/targeted_supervision.md` |
| Explorer Compare tab endpoint | ✅ Done | `/api/compare/metrics` in `scripts/disentanglement_explorer.py` |

**Superseded / dropped (detail in [PHASE_2_ROADMAP.md](PHASE_2_ROADMAP.md)):**

- Phase 2b (correlated split training), 2d (SepVAE), 2e (held-out), 2g (β hardening) — all superseded.
- Browser extensions (Summary table, Provenance bar, Routing UX) — not built, dropped.
- Supervised VAE Exps 21–23 — dropped.

---

## Active Scope — Three New Stages

### Stage 1 — Gaussian Analytic Proof

**Goal:** Derive formally why standard PoE composition fails when VAE posteriors
are entangled. Use a rotation-mixing toy to show the failure mode analytically.

**Deliverables:**

- [ ] Rotation-mixing toy setup (2D Gaussian, two correlated factors)
- [ ] Analytic derivation of PoE failure: entangled posteriors → incorrect product
- [ ] Closed-form expression for the error term
- [ ] Written note: `notes/poe_failure_derivation.md`

**Key questions to answer:**
- Under what posterior correlation does PoE composition break down?
- Is the failure mode a bias (wrong mean) or a variance problem, or both?

---

### Stage 2 — Gaussian Numerical Verification

**Goal:** Confirm the Stage 1 failure mode holds empirically across three model
variants on rotated data; establish which variant is clean enough to use in Stage 3.

**Three model variants** (to be defined):

- [ ] Variant A: standard VAE (entangled posterior — expected to fail PoE)
- [ ] Variant B: (TBD — e.g. β-VAE or factored prior)
- [ ] Variant C: (TBD — e.g. weakly supervised / ground-truth partition)

**Deliverables:**

- [ ] Rotated-data synthetic dataset (Gaussian, known ground truth)
- [ ] Training scripts for all three variants
- [ ] PoE composition evaluation: ground-truth vs. composed posterior
- [ ] Quantitative comparison table (posterior correlation, PoE error, MIG/DCI)
- [ ] Decision: which variant to carry into Stage 3

---

### Stage 3 — 3D Shapes + FiLM-Conditioned DDPM

**Goal:** Full compositional generation pipeline — a disentangled VAE encoder
(selected from Stage 2) conditioning a DDPM via FiLM layers on the 3D Shapes
dataset.

**Deliverables:**

- [ ] 3D Shapes dataset loader (`src/datasets/shapes3d.py`)
- [ ] FiLM-conditioning module (`src/models/film.py`)
- [ ] FiLM-conditioned DDPM (`src/models/ddpm_film.py`)
- [ ] Training script: `scripts/train_ddpm_film.py`
- [ ] Evaluation: compositional generation of held-out factor combinations
- [ ] Inspection tooling (replace dSprites explorer with Shapes-aware equivalent)

---

## Carry-Forward from dSprites Work

These artifacts from the dSprites phase are reusable in the new stages:

| Artifact | Reuse in |
|---|---|
| `compute_dci()` in `src/metrics/dci.py` | Stage 2 model comparison |
| `src/utils/train_utils.py`, `seed.py` | All stages |
| `src/models/vae.py` (encoder/decoder pattern) | Stage 2 backbone |
| W&B logging pattern from `scripts/train_vae.py` | Stage 2/3 trainers |
| `Instruction.md` (env/Git workflow) | All stages |

---

## Archived Reference (dSprites chapter)

All dSprites-specific docs are in [`docs/archived-dsprites/`](docs/archived-dsprites/):

- `PROJECT_GUIDE.md` — architecture, training guide, result interpretation
- `PHASE_2_DATASET_SUMMARY.md` — split system API
- `EXPLORER_GUIDE.md` — browser feature reference
- `sepvae_design_logic.md` — GRL/partition/PoE design derivation (relevant background for Stage 1)

Detail on what was built and what was dropped: [PHASE_2_ROADMAP.md](PHASE_2_ROADMAP.md).
