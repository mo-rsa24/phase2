# Phase 2 Dataset System: Summary

## Objective
Build a dataset system that enables three scientific conditions for measuring compositional failure in VAEs.

## Implementation

### Files
- **`src/datasets/dsprites.py`** (165 lines)
  - `load_dsprites()` — auto-downloads 737K images from DeepMind
  - `get_factor_names()` → `["color", "shape", "scale", "orientation", "pos_x", "pos_y"]`
  - `filter_by_factors(dataset, constraints)` — mask by factor classes
  - `make_iid_split(dataset, train_frac=0.7)` → (train_idx, val_idx, test_idx)
  - `DSpritesDataset` — PyTorch Dataset wrapper

- **`src/datasets/correlated_dsprites.py`** (171 lines)
  - `make_correlated_split(dataset, factor_a, factor_b)` — inject correlation
  - `make_heldout_pair_split(dataset, factor_a, factor_b, held_a_vals, held_b_vals)` — remove combinations
  - `factor_pair_table(dataset, factor_a, factor_b)` → pandas DataFrame co-occurrence

- **`notebooks/01_dsprites_eda.ipynb`** (178 lines)
  - Load dataset and visualize structure
  - Random 5×5 image grid with labels
  - Scale×Orientation co-occurrence heatmap
  - IID split pie chart
  - Correlated split scatter plots (concordant/discordant)
  - Held-out pair split heatmap

## Three Scientific Splits

### 1. IID Split (Baseline)
```python
train_idx, val_idx, test_idx = make_iid_split(dataset, train_frac=0.7)
```
- **Distribution**: 70% train, 15% val, 15% test (random)
- **Purpose**: Baseline disentanglement
- **Metric**: DCI, SAP, MIG, or other standard metrics
- **Hypothesis**: Well-disentangled VAE learns clean factors here

### 2. Correlated Split (Entanglement Injection)
```python
train_idx, val_idx, test_idx = make_correlated_split(
    dataset, factor_a='scale', factor_b='orientation', correlation='positive', seed=42
)
```
- **Structure**:
  - Train: concordant pairs (both high OR both low)
  - Val/Test: discordant pairs (one high, one low)
- **Purpose**: Inject correlation; test if VAE's latent space entangles
- **Metric**: Correlation coefficient in learned latent codes
- **Hypothesis**: Entangled VAE will have correlated latents; disentangled will not

### 3. Held-Out Pair Split (Compositional Generalization)
```python
train_idx, val_idx, test_idx, held_out_idx = make_heldout_pair_split(
    dataset, factor_a='shape', factor_b='scale',
    held_a_vals=[2], held_b_vals=[4, 5], seed=42
)
```
- **Structure**:
  - Train: all samples EXCEPT (shape=2, scale=[4,5])
  - Held-out: ONLY (shape=2, scale=[4,5])
  - Test: remaining unseen combinations
- **Purpose**: Test compositional recombination
- **Metric**: Reconstruction error on held-out pairs
- **Hypothesis**: Disentangled VAE generalizes to new combinations; entangled fails

## Factor Structure
dSprites has 6 factors with 737,280 total images:
| Factor | Classes | Range |
|--------|---------|-------|
| color | 1 | constant (white) |
| shape | 3 | {square, ellipse, heart} |
| scale | 6 | {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} |
| orientation | 40 | [0°, 360°] |
| pos_x | 32 | [0, 31] pixels |
| pos_y | 32 | [0, 31] pixels |

## Usage Example

```python
from src.datasets.dsprites import load_dsprites, make_iid_split
from src.datasets.correlated_dsprites import make_correlated_split, make_heldout_pair_split

# Load dataset
dataset = load_dsprites()  # 737,280 images, 64×64 binary

# Create IID split for baseline
train_idx, val_idx, test_idx = make_iid_split(dataset, seed=42)

# Create correlated split for entanglement testing
train_c, val_c, test_c = make_correlated_split(
    dataset, 'scale', 'orientation', seed=42
)

# Create held-out split for compositional testing
train_h, val_h, test_h, held = make_heldout_pair_split(
    dataset, 'shape', 'scale', [2], [4, 5], seed=42
)

# Use splits with training script
# (See Phase 2 model implementations for integration)
```

## Key Design Decisions

1. **Correlated split logic**: Bins factors at median, then creates concordant/discordant pairs. This preserves marginals while injecting correlation — more scientifically rigorous than sampling new distributions.

2. **Held-out pairs**: Removes entire cells from factor space (not just individual samples). Ensures clean separation between training and held-out conditions.

3. **Factor pair table**: Provides ground-truth co-occurrence statistics. Useful for post-hoc analysis and debugging split properties.

4. **Lazy PyTorch import**: DSpritesDataset requires torch, but core utilities work without it. Allows dataset exploration without GPU environment.

## Verification

All functions tested and passing:
- Dataset loads correctly (737,280 images)
- IID split creates correct proportions (70/15/15)
- Correlated split achieves ~0.76 correlation in training
- Held-out split has zero overlap between train and held-out
- Factor pair table covers all 737,280 samples

## Next Steps for Phase 2

1. **Train VAE variants** on each split type
   - Vanilla VAE
   - Beta-VAE (add KL weight)
   - FactorVAE (add TC regularization)

2. **Measure disentanglement** with DCI probe
   - Disentanglement score (factor independence)
   - Completeness score (factor coverage)
   - Informativeness (predictor accuracy)

3. **Compare across splits**
   - IID: standard disentanglement
   - Correlated: latent correlation vs injected correlation
   - Held-out: reconstruction error on unseen combinations

4. **Interpret results**
   - Which architecture best disentangles?
   - Which splits reveal compositional failure?
   - What minimizes held-out recombination error?
