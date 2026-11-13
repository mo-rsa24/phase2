"""3D Shapes dataset loader (Stage 3 — Plan 03).

DeepMind 3D Shapes: 480 000 synthetic images with 6 independent generative
factors. The HDF5 file (~480 MB) is downloaded once and cached on disk.

Factor ordering (slowest → fastest varying in the HDF5):
    floor_hue (10), wall_hue (10), object_hue (10),
    scale (8), shape (4), orientation (15)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_URL     = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
_N_TOTAL = 480_000

FACTOR_SIZES = [10, 10, 10, 8, 4, 15]
FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]

# stride[j] = product of all factor sizes to the right of j
_STRIDES = np.array(
    [int(np.prod(FACTOR_SIZES[j + 1:])) for j in range(6)],
    dtype=np.int64,
)


def _default_cache_dir() -> Path:
    env = os.environ.get("SHAPES3D_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "shapes3d"


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        from tqdm import tqdm
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name) as pbar:
            def _hook(count, bsize, total):
                pbar.total = total
                pbar.update(bsize)
            urllib.request.urlretrieve(_URL, tmp, reporthook=_hook)
    except ImportError:
        print(f"Downloading {_URL} …")
        urllib.request.urlretrieve(_URL, tmp)
    tmp.rename(dest)
    print(f"Saved: {dest}")


def _decode_factors(global_idx: int) -> np.ndarray:
    """Decode a global HDF5 index into 6 integer factor indices.

    The dataset cycles orientation fastest and floor_hue slowest, so
    we simply unravel the index through FACTOR_SIZES right-to-left.
    """
    factors   = np.empty(6, dtype=np.int64)
    remaining = int(global_idx)
    for j in range(5, -1, -1):
        factors[j]  = remaining % FACTOR_SIZES[j]
        remaining  //= FACTOR_SIZES[j]
    return factors


def _encode_factors(factors: np.ndarray) -> int:
    """Encode 6 integer factor indices back into a global HDF5 index."""
    return int(np.dot(factors, _STRIDES))


class Shapes3D(Dataset):
    """3D Shapes with a seeded IID 70 / 15 / 15 train / val / test split.

    Args:
        split:     ``"train"``, ``"val"``, or ``"test"``.
        cache_dir: Directory for the HDF5 file.  Falls back to
                   ``$SHAPES3D_CACHE_DIR`` or ``~/.cache/shapes3d/``.
        seed:      RNG seed for the split permutation.

    Returns per sample:
        img     – ``(3, 64, 64)`` float32 in [0, 1]
        latents – ``(6,)`` int64, one integer index per factor
    """

    FACTOR_SIZES = FACTOR_SIZES
    FACTOR_NAMES = FACTOR_NAMES

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        self._split   = split
        self._h5_path = Path(cache_dir or _default_cache_dir()) / "3dshapes.h5"
        self._file: h5py.File | None = None   # opened lazily per process (DataLoader-safe)

        if not self._h5_path.exists():
            print(f"3D Shapes not cached at {self._h5_path} — downloading …")
            _download(self._h5_path)

        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(_N_TOTAL)
        n_train = int(0.70 * _N_TOTAL)   # 336 000
        n_val   = int(0.15 * _N_TOTAL)   #  72 000

        if split == "train":
            self._indices = perm[:n_train]
        elif split == "val":
            self._indices = perm[n_train : n_train + n_val]
        else:
            self._indices = perm[n_train + n_val :]

    # ------------------------------------------------------------------
    # Lazy open: each DataLoader worker inherits a None handle after
    # fork and opens its own file descriptor on first access.
    # ------------------------------------------------------------------
    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self._h5_path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g       = int(self._indices[idx])
        img_hwc = self._open()["images"][g]                          # (64,64,3) uint8
        img     = torch.from_numpy(img_hwc).permute(2, 0, 1).float().div(255.0)
        latents = torch.from_numpy(_decode_factors(g))
        return img, latents

    def __repr__(self) -> str:
        return f"Shapes3D(split={self._split!r}, n={len(self)})"


class PairedShapes3DDataset(Dataset):
    """Paired iterator for Ada-GVAE training.

    Each item is a triplet ``(x_a, x_b, shared_mask)`` where x_a and x_b
    differ in exactly one randomly chosen factor.

    Args:
        split:      ``"train"``, ``"val"``, or ``"test"``.
        cache_dir:  Passed through to ``Shapes3D``.
        seed:       RNG seed for the IID split (same as ``Shapes3D``).
        pair_seed:  Additional seed offset that controls which factor is
                    changed and to what value.  Call ``set_epoch(e)`` each
                    epoch to get fresh pairs without re-instantiating.

    Returns per sample:
        x_a          – ``(3, 64, 64)`` float32 in [0, 1]
        x_b          – ``(3, 64, 64)`` float32 in [0, 1]  (one factor changed)
        shared_mask  – ``(6,)`` bool  True where the factor is held constant
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | Path | None = None,
        seed: int = 42,
        pair_seed: int = 0,
    ) -> None:
        self._base      = Shapes3D(split=split, cache_dir=cache_dir, seed=seed)
        self._pair_seed = pair_seed

    def set_epoch(self, epoch: int) -> None:
        """Rotate pair assignments each epoch — call from your training loop."""
        self._pair_seed = epoch * len(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self._pair_seed + idx)

        g_a      = int(self._base._indices[idx])
        factors_a = _decode_factors(g_a)

        # pick which factor to vary
        k = int(rng.integers(6))

        # pick a different value for that factor
        choices = np.delete(np.arange(FACTOR_SIZES[k]), factors_a[k])
        new_val = int(rng.choice(choices))

        factors_b    = factors_a.copy()
        factors_b[k] = new_val
        g_b          = _encode_factors(factors_b)

        f      = self._base._open()["images"]
        x_a    = torch.from_numpy(f[g_a]).permute(2, 0, 1).float().div(255.0)
        x_b    = torch.from_numpy(f[g_b]).permute(2, 0, 1).float().div(255.0)

        shared_mask      = torch.ones(6, dtype=torch.bool)
        shared_mask[k]   = False

        return x_a, x_b, shared_mask

    def __repr__(self) -> str:
        return (
            f"PairedShapes3DDataset(split={self._base._split!r}, "
            f"n={len(self)}, pair_seed={self._pair_seed})"
        )
