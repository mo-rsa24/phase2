"""Correlated and held-out pair split utilities for dSprites."""

import numpy as np
import pandas as pd
from .dsprites import FACTOR_NAMES, FACTOR_SIZES


def make_correlated_split(
    dataset: dict,
    factor_a: str,
    factor_b: str,
    correlation: str = "positive",
    train_frac: float = 0.7,
    seed: int = 42,
) -> tuple:
    """Create split with correlated factors in training data.

    For held-out recombination tests: train on correlations, test on independence.

    Args:
        dataset: dict from load_dsprites()
        factor_a, factor_b: factor names to correlate
        correlation: "positive" (both high/low) or "negative" (high/low opposite)
        train_frac: fraction of concordant pairs for training
        seed: random seed

    Returns:
        (train_idx, val_idx, test_idx): indices where correlation is injected into train
    """
    latents = dataset["latents_classes"]
    rng = np.random.RandomState(seed)

    a_idx = FACTOR_NAMES.index(factor_a)
    b_idx = FACTOR_NAMES.index(factor_b)

    a_vals = latents[:, a_idx]
    b_vals = latents[:, b_idx]

    # Bin at median
    a_median = np.median(a_vals)
    b_median = np.median(b_vals)

    a_high = a_vals >= a_median
    b_high = b_vals >= b_median

    if correlation == "positive":
        # Concordant: both high or both low
        concordant = (a_high & b_high) | (~a_high & ~b_high)
        discordant = ~concordant
    elif correlation == "negative":
        # Discordant: opposite signs
        discordant = (a_high & b_high) | (~a_high & ~b_high)
        concordant = ~discordant
    else:
        raise ValueError(f"correlation must be 'positive' or 'negative', got {correlation}")

    concordant_idx = np.where(concordant)[0]
    discordant_idx = np.where(discordant)[0]

    # Shuffle concordant and keep train_frac for training
    rng.shuffle(concordant_idx)
    n_train = int(len(concordant_idx) * train_frac)
    train_idx = concordant_idx[:n_train]
    val_idx = np.concatenate([concordant_idx[n_train:], discordant_idx])

    # Random split val_idx into val and test
    rng.shuffle(val_idx)
    n_val = len(val_idx) // 2
    val_idx = val_idx[:n_val]
    test_idx = val_idx[n_val:]

    return train_idx, val_idx, test_idx


def make_heldout_pair_split(
    dataset: dict,
    factor_a: str,
    factor_b: str,
    held_a_vals: list,
    held_b_vals: list,
    train_frac: float = 0.7,
    seed: int = 42,
) -> tuple:
    """Create split with held-out (factor_a, factor_b) pairs.

    Removes rows where factor_a in held_a_vals AND factor_b in held_b_vals.
    These become the held_out split.

    Args:
        dataset: dict from load_dsprites()
        factor_a, factor_b: factor names
        held_a_vals: list of class indices for factor_a to hold out
        held_b_vals: list of class indices for factor_b to hold out
        train_frac: fraction of non-held-out data for training
        seed: random seed

    Returns:
        (train_idx, val_idx, test_idx, held_out_idx): indices
    """
    latents = dataset["latents_classes"]
    rng = np.random.RandomState(seed)

    a_idx = FACTOR_NAMES.index(factor_a)
    b_idx = FACTOR_NAMES.index(factor_b)

    a_vals = latents[:, a_idx]
    b_vals = latents[:, b_idx]

    # Mask for held-out pairs
    held_out_mask = np.isin(a_vals, held_a_vals) & np.isin(b_vals, held_b_vals)
    held_out_idx = np.where(held_out_mask)[0]

    # Remaining data
    available_idx = np.where(~held_out_mask)[0]

    # Split available into train/val/test
    rng.shuffle(available_idx)
    n_train = int(len(available_idx) * train_frac)
    n_val = (len(available_idx) - n_train) // 2

    train_idx = available_idx[:n_train]
    val_idx = available_idx[n_train:n_train + n_val]
    test_idx = available_idx[n_train + n_val:]

    # Verify no overlap
    assert len(np.intersect1d(train_idx, held_out_idx)) == 0
    assert len(np.intersect1d(val_idx, held_out_idx)) == 0
    assert len(np.intersect1d(test_idx, held_out_idx)) == 0

    return train_idx, val_idx, test_idx, held_out_idx


def factor_pair_table(
    dataset: dict,
    factor_a: str,
    factor_b: str,
) -> pd.DataFrame:
    """Create DataFrame with co-occurrence counts for two factors.

    Args:
        dataset: dict from load_dsprites()
        factor_a, factor_b: factor names

    Returns:
        DataFrame with factor_a classes as rows, factor_b classes as columns,
        cell values are co-occurrence counts
    """
    latents = dataset["latents_classes"]

    a_idx = FACTOR_NAMES.index(factor_a)
    b_idx = FACTOR_NAMES.index(factor_b)

    a_vals = latents[:, a_idx]
    b_vals = latents[:, b_idx]

    a_size = FACTOR_SIZES[a_idx]
    b_size = FACTOR_SIZES[b_idx]

    # Build contingency table
    table = np.zeros((a_size, b_size), dtype=int)
    for a_class in range(a_size):
        for b_class in range(b_size):
            table[a_class, b_class] = np.sum((a_vals == a_class) & (b_vals == b_class))

    df = pd.DataFrame(
        table,
        index=pd.Index(range(a_size), name=factor_a),
        columns=pd.Index(range(b_size), name=factor_b),
    )

    return df
