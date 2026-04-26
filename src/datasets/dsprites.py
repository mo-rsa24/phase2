"""dSprites dataset loader and utilities."""

import os
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/blob/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
)

FACTOR_NAMES = ["color", "shape", "scale", "orientation", "pos_x", "pos_y"]
FACTOR_SIZES = [1, 3, 6, 40, 32, 32]


def download_dsprites(data_dir: str = "data/") -> None:
    """Download dSprites dataset if it doesn't exist."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / "dsprites.npz"
    if filepath.exists():
        return

    print(f"Downloading dSprites from {DSPRITES_URL}...")
    response = requests.get(DSPRITES_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Downloaded to {filepath}")


def load_dsprites(data_dir: str = "data/") -> dict:
    """Load dSprites dataset.

    Returns dict with keys:
    - imgs: (737280, 64, 64) binary array
    - latents_values: (737280, 6) continuous factor values
    - latents_classes: (737280, 6) discrete factor indices
    - metadata: dict with factor info
    """
    download_dsprites(data_dir)

    filepath = Path(data_dir) / "dsprites.npz"
    data = np.load(filepath, allow_pickle=True)

    imgs = data["imgs"]
    latents_values = data["latents_values"]
    latents_classes = data["latents_classes"]

    return {
        "imgs": imgs,
        "latents_values": latents_values,
        "latents_classes": latents_classes,
        "metadata": {
            "factor_names": FACTOR_NAMES,
            "factor_sizes": FACTOR_SIZES,
        }
    }


def get_factor_names() -> list:
    """Return list of factor names in dSprites."""
    return FACTOR_NAMES


def filter_by_factors(dataset: dict, factor_constraints: dict) -> np.ndarray:
    """Get boolean mask for samples matching factor constraints.

    Args:
        dataset: dict from load_dsprites()
        factor_constraints: dict mapping factor name -> list of allowed class indices
                           e.g. {"shape": [0, 1], "scale": [3, 4, 5]}

    Returns:
        Boolean array of length n_samples where True indicates matching sample
    """
    latents = dataset["latents_classes"]
    factor_names = FACTOR_NAMES

    mask = np.ones(len(latents), dtype=bool)
    for factor_name, allowed_classes in factor_constraints.items():
        if factor_name not in factor_names:
            raise ValueError(f"Unknown factor: {factor_name}")

        factor_idx = factor_names.index(factor_name)
        factor_mask = np.isin(latents[:, factor_idx], allowed_classes)
        mask = mask & factor_mask

    return mask


def make_iid_split(
    dataset: dict,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Create random train/val/test split.

    Returns:
        (train_idx, val_idx, test_idx): arrays of indices
    """
    n_samples = len(dataset["imgs"])
    rng = np.random.RandomState(seed)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


class DSpritesDataset(Dataset):
    """PyTorch Dataset wrapper for dSprites."""

    def __init__(self, dataset: dict, indices: np.ndarray, normalize: bool = True):
        """
        Args:
            dataset: dict from load_dsprites()
            indices: array of indices to use
            normalize: if True, scale images to [0, 1]; else keep binary
        """
        if torch is None:
            raise ImportError("PyTorch is required for DSpritesDataset. Install with: pip install torch")

        self.imgs = dataset["imgs"][indices].astype(np.float32)
        if normalize:
            # Already binary, just cast to float
            pass
        self.latents = dataset["latents_classes"][indices]
        self.normalize = normalize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx])
        # Add channel dimension: (64, 64) -> (1, 64, 64)
        img = img.unsqueeze(0)
        latents = torch.from_numpy(self.latents[idx]).long()
        return img, latents
