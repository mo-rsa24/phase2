# Phase 2 Representation Learning Instructions

This document is the operational guide for setting up, versioning, and running the weekend scaffold for **beta-VAE, FactorVAE, DCI, GRL, and SepVAE-style experiments on dSprites**.

The goal is not to finish every model in full detail immediately. The goal is to build a **clean, reproducible, research-friendly system** that lets us:

1. set up the environment on any machine,
2. organize code and notes consistently,
3. run small experiments safely,
4. recover old states of the project,
5. collaborate across computers without chaos.

---

## 1. Project Objective

By the end of the setup, this repository should support a scientific question like:

> How do VAE variants change latent factorization on dSprites, and how does that help with held-out recombination experiments?

The immediate deliverable is a starter repository with:

- runnable environment setup,
- a clear folder structure,
- lightweight experiment scripts,
- notes that are linked to code,
- version control habits that preserve milestones.

---

## 2. Working Philosophy

Use this rule throughout the project:

```text
Watch -> iPad breakdown -> compressed note -> code only the minimum needed -> link note to code
```

Every study session should produce at least one concrete artifact:

- a note,
- an equation summary,
- a diagram prompt,
- runnable code,
- a debugging checklist,
- or an experiment result.

The idea is to avoid passive learning. Theory should immediately feed implementation, and implementation should immediately feed documentation.

---

## 3. Recommended Repository Structure

The repository should evolve toward this layout:

```text
phase2_representation/
  README.md
  Instruction.md
  requirements.txt
  environment.yml
  configs/
    vae.yaml
    beta_vae.yaml
    factor_vae.yaml
    sepvae.yaml
  src/
    datasets/
      dsprites.py
    models/
      vae.py
      beta_vae.py
      factor_vae.py
      sepvae.py
      grl.py
    losses/
      vae_losses.py
      tc_loss.py
    metrics/
      dci.py
      probes.py
      recombination.py
    utils/
      viz.py
      seed.py
      train_utils.py
  scripts/
    train_vae.py
    train_factorvae.py
    eval_dci.py
  notebooks/
    01_dsprites_eda.ipynb
    02_vae_sanity.ipynb
    03_latent_traversals.ipynb
    04_dci_probe.ipynb
  notes/
    00_index.md
    vae.md
    elbo.md
    reparameterization.md
    beta_vae.md
    factor_vae.md
    total_correlation.md
    dci.md
    grl.md
    sepvae.md
    correlated_dsprites.md
  data/
  outputs/
  parking_lot.md
```

### Folder intent

- `configs/`: experiment definitions and hyperparameters.
- `src/`: reusable research code.
- `scripts/`: thin command-line entry points.
- `notebooks/`: exploration, sanity checks, and visualizations.
- `notes/`: compressed theory and implementation notes.
- `data/`: local datasets, usually not committed.
- `outputs/`: checkpoints, plots, logs, and evaluation summaries.
- `parking_lot.md`: ideas that matter, but not right now.

---

## 4. Environment Setup with Micromamba

These instructions assume **Micromamba is already installed**.

### 4.1 Create the environment

From the project root:

```bash
micromamba create -n phase2-repr python=3.11 -y
micromamba activate phase2-repr
```

If activation does not work yet, initialize shell support once:

```bash
micromamba shell init -s bash -r ~/micromamba
```

Then restart the terminal and run:

```bash
micromamba activate phase2-repr
```

### 4.2 Install project requirements

If `requirements.txt` contains the Python dependencies:

```bash
pip install -r requirements.txt
```

Typical starter packages for this project may include:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`
- `ipykernel`
- `pyyaml`
- `tqdm`

If you add packages manually during setup, immediately record them:

```bash
pip freeze > requirements.txt
```

That is the simplest path, but for **better reproducibility across machines**, also create an `environment.yml`.

### 4.3 Create a reproducible `environment.yml`

Use:

```bash
micromamba env export -n phase2-repr > environment.yml
```

This captures the current environment in a form another machine can recreate.

### 4.4 Recreate the environment on another machine

On the second machine:

```bash
micromamba env create -f environment.yml
micromamba activate phase2-repr
```

If the exported file is too platform-specific, use this practical alternative:

1. keep `requirements.txt` for Python packages,
2. keep a short hand-written `environment.yml` for Python version and core dependencies,
3. rebuild with those two files rather than relying only on a fully frozen export.

### 4.5 Practical reproducibility rule

Use this rule:

- `requirements.txt` is the easy Python dependency list.
- `environment.yml` is the reproducible environment definition.
- commit both whenever dependencies change.

For research code, perfect lockfile reproducibility is less important than:

- same Python version,
- same major ML libraries,
- documented installation steps,
- and the ability to rerun notebooks and scripts consistently.

---

## 5. Git and GitHub Setup

This section covers setting up Git identity, SSH, repository creation, and pushing the project.

### 5.1 Configure Git identity

Run these once on each computer:

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

Check:

```bash
git config --global --list
```

### 5.2 Check whether SSH keys already exist

Run:

```bash
ls -la ~/.ssh
```

Look for files like:

- `id_ed25519`
- `id_ed25519.pub`
- `id_rsa`
- `id_rsa.pub`

If you already have an SSH key pair, you may reuse it.

### 5.3 Create an SSH key if needed

Recommended:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept the default file location. You may set a passphrase or leave it empty, but for secure everyday use, a passphrase is better.

### 5.4 Start the SSH agent and add the key

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 5.5 Copy the public key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the full output and add it to GitHub:

1. GitHub
2. `Settings`
3. `SSH and GPG keys`
4. `New SSH key`

### 5.6 Test passwordless GitHub login

Run:

```bash
ssh -T git@github.com
```

If everything is configured properly, GitHub should authenticate you without asking for your GitHub password during pushes.

---

## 6. Create the GitHub Repository and Push the Project

### 6.1 Initialize Git locally

From the project root:

```bash
git init
git branch -M main
```

### 6.2 Add a `.gitignore`

Make sure the repository ignores things that should stay local, especially:

```text
__pycache__/
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
.env
data/
outputs/
*.pt
*.pth
*.ckpt
```

Notes:

- `data/` is often too large or too machine-specific to commit.
- `outputs/` should usually stay local unless you are deliberately versioning lightweight figures or summaries.

### 6.3 Make the first commit

```bash
git add .
git commit -m "Initial scaffold for phase 2 representation learning"
```

### 6.4 Create the remote repository on GitHub

Create a new repository on GitHub, for example:

```text
phase2_representation
```

Do not initialize it with a README if this local folder already contains files.

### 6.5 Add the remote and push

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin git@github.com:YOUR_USERNAME/phase2_representation.git
git push -u origin main
```

After this, future pushes are simply:

```bash
git push
```

---

## 7. Pulling the Project on Another Computer

On the second computer, first repeat the Git identity and SSH setup.

Then clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/phase2_representation.git
cd phase2_representation
```

Recreate the environment:

```bash
micromamba env create -f environment.yml
micromamba activate phase2-repr
```

If needed, also run:

```bash
pip install -r requirements.txt
```

To pull the latest changes in the future:

```bash
git pull origin main
```

If you are working on a feature branch:

```bash
git checkout your-branch-name
git pull origin your-branch-name
```

---

## 8. Daily Git Workflow for Research

Use Git as a research memory system, not just a backup tool.

### 8.1 Branches

Use:

- `main` for stable, meaningful states,
- short-lived feature branches for active work,
- tags for milestones or snapshots worth remembering.

Recommended branch names:

- `feature/dsprites-loader`
- `feature/beta-vae-loss`
- `feature/factorvae-discriminator`
- `feature/dci-metric`
- `fix/training-seed-bug`
- `docs/weekend-notes`

### 8.2 Basic feature branch workflow

Create a branch:

```bash
git checkout -b feature/beta-vae-loss
```

Work, then commit in small logical chunks:

```bash
git add .
git commit -m "Implement beta-VAE KL weighting"
```

Push the branch:

```bash
git push -u origin feature/beta-vae-loss
```

When the work is stable, merge it back into `main`.

### 8.3 Commit philosophy

Good commits are:

- small,
- meaningful,
- reversible,
- and named after a clear unit of work.

Good examples:

- `Add dSprites dataset loader`
- `Implement ELBO loss and reconstruction term`
- `Add latent traversal notebook`
- `Document total correlation intuition`

Avoid giant mixed commits like:

- model code + notebook cleanup + environment changes + note edits all together.

---

## 9. Tags and Snapshots

Tags are useful for research milestones. They answer the question:

> What exact project state produced this result or this phase of understanding?

### 9.1 When to tag

Create a tag when you reach a meaningful milestone, such as:

- first runnable vanilla VAE,
- beta-VAE baseline completed,
- FactorVAE skeleton completed,
- first DCI evaluation working,
- results used in slides or a report.

### 9.2 Example tags

```text
v0.1-scaffold
v0.2-vae-sanity
v0.3-beta-vae-baseline
v0.4-factorvae-skeleton
v0.5-dci-first-pass
```

### 9.3 Create and push a tag

```bash
git tag -a v0.1-scaffold -m "Repository scaffold and environment setup"
git push origin v0.1-scaffold
```

Tags are especially important when you later ask:

- Which code produced this figure?
- Which implementation version gave these traversals?
- Which state existed before I refactored the loss code?

---

## 10. Managing Merge Conflicts Calmly

Conflicts are normal. They usually mean Git found overlapping edits, not that anything is broken beyond repair.

### 10.1 Prevent conflicts early

Pull often:

```bash
git checkout main
git pull origin main
```

Before starting a branch, branch from an up-to-date `main`.

Commit often enough that your work is easy to recover.

### 10.2 If a conflict happens

Git will mark the conflicting sections in the file. They look like this:

```text
<<<<<<< HEAD
your version
=======
incoming version
>>>>>>> other-branch
```

Resolve by editing the file into the final intended version, then:

```bash
git add path/to/conflicted_file.py
git commit
```

### 10.3 Conflict resolution philosophy

Do not ask:

> Which side do I keep?

Ask:

> What should the final file look like now that both lines of work exist?

That mindset is much better for research code, where both sides may contain useful ideas.

### 10.4 Helpful practices

- keep notebooks exploratory, but avoid treating them as the only source of truth,
- move reusable logic into `src/`,
- keep configs separate from code,
- keep notes in `notes/` rather than burying reasoning in commit history.

---

## 11. Experiment Organization

The experiments should be structured so that code, configs, and outputs can be traced back to each other.

### 11.1 Config-first habit

Every experiment should have a config file or clear argument set.

Examples:

- `configs/vae.yaml`
- `configs/beta_vae.yaml`
- `configs/factor_vae.yaml`
- `configs/sepvae.yaml`

This helps with:

- reproducibility,
- comparison across runs,
- and debugging unexpected results.

### 11.2 Output naming

Store outputs using dated or descriptive run folders, for example:

```text
outputs/
  2026-04-25_vae_sanity/
  2026-04-25_beta_vae_b4/
  2026-04-26_factorvae_tc/
```

Each run folder should ideally contain:

- training config copy,
- logs,
- final metrics,
- model checkpoint,
- traversal images,
- any quick interpretation notes.

### 11.3 Link notes to code

If you study total correlation, the note in `notes/total_correlation.md` should point to:

- the relevant loss implementation,
- the script that uses it,
- and the notebook or output that visualizes it.

This is one of the best habits you can build in a research codebase.

---

## 12. Suggested Experimental Progression

Do not implement everything at once.

### Stage 1: Vanilla VAE

Goal:

- dSprites loader,
- encoder/decoder,
- ELBO loss,
- sample reconstructions,
- latent traversals.

### Stage 2: beta-VAE

Goal:

- add KL weighting,
- observe stronger disentanglement pressure,
- compare traversals against the vanilla VAE.

### Stage 3: FactorVAE

Goal:

- add total correlation regularization,
- implement or stub the discriminator logic,
- document clearly how it differs from beta-VAE.

### Stage 4: DCI

Goal:

- compute disentanglement/completeness/informativeness,
- use probes or supervised predictors on latent codes,
- compare models quantitatively.

### Stage 5: GRL and SepVAE-style ideas

Goal:

- build minimal runnable scaffolds,
- focus on representation partitioning intuition,
- document what is prototype-level versus fully validated.

This ordering matters because each stage builds the foundation for the next.

---

## 13. Notebook and Note Discipline

Use notebooks for:

- visual checks,
- quick ablations,
- latent traversals,
- and small metric exploration.

Use `src/` and `scripts/` for:

- reusable code,
- training loops,
- metrics,
- and anything you may want to run again.

Use `notes/` for:

- theory compression,
- equations,
- reading summaries,
- experiment interpretation,
- and implementation decisions.

The rule is:

- notebooks explore,
- scripts run,
- `src/` generalizes,
- notes explain.

---

## 14. Minimal Weekly or Weekend Operating Routine

For each work block:

1. pull the latest code,
2. activate the environment,
3. choose exactly one small objective,
4. study only what supports that objective,
5. implement the smallest runnable version,
6. record what changed,
7. commit with a meaningful message.

Example:

```text
Objective: implement beta-VAE KL weighting
Study artifact: notes/beta_vae.md
Code artifact: src/models/beta_vae.py
Run artifact: outputs/2026-04-25_beta_vae_b4/
Version artifact: git commit + optional tag
```

That is a healthy research loop.

---

## 15. What Not to Do

Avoid these common failure modes:

- watching many hours of theory without producing notes,
- implementing several models before vanilla VAE is stable,
- storing important reasoning only in your head,
- committing notebooks and code changes in one giant unclear commit,
- changing dependencies without updating `requirements.txt` and `environment.yml`,
- treating `main` like a scratchpad.

---

## 16. Definition of a Good Setup

The setup is successful when:

- a new computer can clone the repo and recreate the environment,
- GitHub push and pull work over SSH without password prompts,
- the folder structure makes it obvious where work belongs,
- experiments can be rerun from configs and scripts,
- notes explain why code exists,
- tags preserve important milestones,
- and feature branches keep active work from destabilizing `main`.

That is the real purpose of this repository: not just to store code, but to create a **stable research operating system** for disentanglement experiments.
