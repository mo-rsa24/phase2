from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES
from src.models.vae import Decoder, Encoder


@dataclass
class VAEPassResult:
    sample_index: int
    factors: dict[str, int]
    input_image: np.ndarray
    reconstruction: np.ndarray
    mu: np.ndarray
    logvar: np.ndarray
    z: np.ndarray
    mse: float


def build_factor_index(dataset: Mapping[str, Any]) -> dict[tuple[int, ...], int]:
    latents = np.asarray(dataset["latents_classes"], dtype=np.int64)
    return {tuple(row.tolist()): idx for idx, row in enumerate(latents)}


def find_sample_index(
    dataset: Mapping[str, Any],
    *,
    factor_values: Mapping[str, int] | None = None,
    sample_index: int | None = None,
    factor_index: Mapping[tuple[int, ...], int] | None = None,
) -> int:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(dataset["imgs"]):
            raise IndexError(f"sample_index={sample_index} is out of range for dataset of size {len(dataset['imgs'])}")
        return int(sample_index)

    if factor_values is None:
        raise ValueError("Provide either sample_index or factor_values.")

    factor_tuple = []
    for name, size in zip(FACTOR_NAMES, FACTOR_SIZES):
        value = int(factor_values.get(name, 0))
        if value < 0 or value >= size:
            raise ValueError(f"{name} must be in [0, {size - 1}], got {value}")
        factor_tuple.append(value)

    factor_tuple = tuple(factor_tuple)
    lookup = factor_index if factor_index is not None else build_factor_index(dataset)
    if factor_tuple not in lookup:
        raise KeyError(f"No dSprites example found for factors {dict(zip(FACTOR_NAMES, factor_tuple))}")
    return int(lookup[factor_tuple])


def _extract_state_dict(payload: Any, module_name: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type for {module_name}: {type(payload)!r}")

    direct_keys = [
        f"{module_name}_state_dict",
        f"{module_name}_dict",
        module_name,
        "state_dict",
        "model_state_dict",
        "model",
    ]
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
            state_dict = value
            prefix = f"{module_name}."
            if any(k.startswith(prefix) for k in state_dict):
                return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            return state_dict

    prefix = f"{module_name}."
    prefixed = {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}
    if prefixed:
        return prefixed

    raise KeyError(
        f"Could not find a state dict for '{module_name}' in checkpoint payload. "
        f"Available keys: {sorted(payload.keys())}"
    )


def _load_checkpoint_state_dict(path: str | Path, module_name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    return _extract_state_dict(payload, module_name)


def infer_encoder_config(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    latent_dim = int(state_dict["mu.weight"].shape[0])
    img_channels = int(state_dict["encoder.0.weight"].shape[1])
    return latent_dim, img_channels


def infer_decoder_config(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    latent_dim = int(state_dict["decoder.0.weight"].shape[1])
    img_channels = int(state_dict["decoder.9.weight"].shape[1])
    return latent_dim, img_channels


def load_encoder_decoder(
    *,
    latent_dim: int = 10,
    img_channels: int = 1,
    encoder_checkpoint: str | Path | None = None,
    decoder_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Encoder, Decoder]:
    encoder_state = None
    decoder_state = None

    if encoder_checkpoint is not None:
        encoder_state = _load_checkpoint_state_dict(encoder_checkpoint, "encoder")
        latent_dim, img_channels = infer_encoder_config(encoder_state)

    if decoder_checkpoint is not None:
        decoder_state = _load_checkpoint_state_dict(decoder_checkpoint, "decoder")
        dec_latent_dim, dec_img_channels = infer_decoder_config(decoder_state)
        if encoder_state is None:
            latent_dim, img_channels = dec_latent_dim, dec_img_channels
        else:
            if dec_latent_dim != latent_dim or dec_img_channels != img_channels:
                raise ValueError(
                    "Encoder and decoder checkpoints disagree on configuration: "
                    f"encoder=(latent_dim={latent_dim}, img_channels={img_channels}), "
                    f"decoder=(latent_dim={dec_latent_dim}, img_channels={dec_img_channels})"
                )

    encoder = Encoder(latent_dim=latent_dim, img_size=torch.Size([img_channels, 64, 64]))
    decoder = Decoder(latent_dim=latent_dim, img_size=torch.Size([img_channels, 64, 64]))

    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    if decoder_state is not None:
        decoder.load_state_dict(decoder_state)

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    return encoder, decoder


def prepare_model_input(image: np.ndarray, img_channels: int) -> torch.Tensor:
    image = np.asarray(image, dtype=np.float32)
    if image.shape != (64, 64):
        raise ValueError(f"Expected a single dSprites image of shape (64, 64), got {image.shape}")

    tensor = torch.from_numpy(image).unsqueeze(0)
    if img_channels == 1:
        return tensor.unsqueeze(0)
    if img_channels == 3:
        return tensor.repeat(3, 1, 1).unsqueeze(0)
    raise ValueError(f"Unsupported img_channels={img_channels}. Expected 1 or 3.")


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return np.clip(array, 0.0, 1.0)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected grayscale or RGB image, got shape {image.shape}")
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


@torch.no_grad()
def run_vae_pass(
    dataset: Mapping[str, Any],
    encoder: Encoder,
    decoder: Decoder,
    *,
    sample_index: int | None = None,
    factor_values: Mapping[str, int] | None = None,
    use_mean_latent: bool = True,
    factor_index: Mapping[tuple[int, ...], int] | None = None,
) -> VAEPassResult:
    resolved_index = find_sample_index(
        dataset,
        factor_values=factor_values,
        sample_index=sample_index,
        factor_index=factor_index,
    )

    image = np.asarray(dataset["imgs"][resolved_index], dtype=np.float32)
    factors = {
        name: int(dataset["latents_classes"][resolved_index, idx])
        for idx, name in enumerate(FACTOR_NAMES)
    }

    img_channels = int(encoder.img_size[0])
    x = prepare_model_input(image, img_channels).to(next(encoder.parameters()).device)
    mu, logvar = encoder.encode(x)
    z = mu if use_mean_latent else encoder.reparameterize(mu, logvar)
    reconstruction = decoder(z)

    reconstruction_image = tensor_to_display_image(reconstruction)
    if reconstruction_image.ndim == 3:
        reconstruction_gray = reconstruction_image[..., 0]
    else:
        reconstruction_gray = reconstruction_image

    mse = float(np.mean((image - reconstruction_gray) ** 2))
    return VAEPassResult(
        sample_index=resolved_index,
        factors=factors,
        input_image=image,
        reconstruction=reconstruction_image,
        mu=mu[0].detach().cpu().numpy(),
        logvar=logvar[0].detach().cpu().numpy(),
        z=z[0].detach().cpu().numpy(),
        mse=mse,
    )
