#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES, load_dsprites
from src.utils.vae_inspection import load_encoder_decoder, run_vae_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single dSprites example through the VAE encoder bottleneck and decoder."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing dsprites.npz")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset index to inspect")
    parser.add_argument("--latent-dim", type=int, default=10, help="Latent dimension if no checkpoints are given")
    parser.add_argument("--img-channels", type=int, default=1, choices=[1, 3], help="Encoder/decoder input channels")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None, help="Optional encoder checkpoint path")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None, help="Optional decoder checkpoint path")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda")
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Sample z from q(z|x) instead of decoding from the mean vector mu",
    )
    parser.add_argument("--save-path", type=Path, default=None, help="Optional path for saving the figure")

    for factor_name, factor_size in zip(FACTOR_NAMES, FACTOR_SIZES):
        parser.add_argument(
            f"--{factor_name}",
            type=int,
            default=None,
            help=f"Override sample selection with {factor_name} in [0, {factor_size - 1}]",
        )
    return parser.parse_args()


def build_factor_override(args: argparse.Namespace) -> dict[str, int] | None:
    overrides = {}
    for factor_name in FACTOR_NAMES:
        value = getattr(args, factor_name)
        if value is not None:
            overrides[factor_name] = value
    return overrides or None


def plot_pass(result, save_path: Path | None) -> None:
    latent_names = [f"z{i}" for i in range(len(result.mu))]
    recon_image = result.reconstruction
    if recon_image.ndim == 3 and recon_image.shape[-1] == 1:
        recon_image = recon_image[..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].imshow(result.input_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input dSprite")
    axes[0].axis("off")

    axes[1].imshow(recon_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Reconstruction\nMSE={result.mse:.5f}")
    axes[1].axis("off")

    x = np.arange(len(latent_names))
    width = 0.38
    axes[2].bar(x - width / 2, result.mu, width=width, label="mu", color="#1f77b4")
    axes[2].bar(x + width / 2, result.z, width=width, label="z", color="#ff7f0e")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(latent_names, rotation=45, ha="right")
    axes[2].set_title("Bottleneck Values")
    axes[2].legend()

    factor_text = " | ".join(f"{name}={value}" for name, value in result.factors.items())
    fig.suptitle(f"sample_index={result.sample_index} | {factor_text}", fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    dataset = load_dsprites(args.data_dir)
    factor_values = build_factor_override(args)

    encoder, decoder = load_encoder_decoder(
        latent_dim=args.latent_dim,
        img_channels=args.img_channels,
        encoder_checkpoint=args.encoder_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        device=args.device,
    )

    result = run_vae_pass(
        dataset,
        encoder,
        decoder,
        sample_index=None if factor_values else args.sample_index,
        factor_values=factor_values,
        use_mean_latent=not args.sample_latent,
    )

    print(f"sample_index={result.sample_index}")
    print("factors:", result.factors)
    print("mu:", np.array2string(result.mu, precision=4))
    print("logvar:", np.array2string(result.logvar, precision=4))
    print("z:", np.array2string(result.z, precision=4))
    print(f"reconstruction_mse={result.mse:.6f}")

    plot_pass(result, args.save_path)


if __name__ == "__main__":
    main()
