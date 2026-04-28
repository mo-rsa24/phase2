"""FactorVAE (Kim & Mnih, 2018) — total-correlation regularised VAE.

The VAE branch reuses the Encoder/Decoder from src.models.vae unchanged so
that checkpoints can be loaded by any tooling that already understands the
standard VAE state_dict layout (e.g. the disentanglement explorer's
load_encoder_decoder).

The FactorVAE objective adds a total-correlation penalty estimated via an
adversarial discriminator that distinguishes samples from the joint q(z)
from samples from the product of marginals (obtained by permuting each dim
independently across the batch).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.vae import Decoder, Encoder


class Discriminator(nn.Module):
    """6-layer MLP per Kim & Mnih (2018), 1000 hidden units, 2-logit head.

    Two-logit output (rather than one) makes the density-ratio estimate
        log D(z) / (1 - D(z))
    numerically stable as a difference of logits, avoiding log of values
    near 0 or 1.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 1000, num_layers: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (batch, 2) logits


class FactorVAE(nn.Module):
    """Encoder + Decoder + Discriminator wrapper.

    forward(x) returns (x_hat, mu, logvar) by default — a drop-in replacement
    for the vanilla VAE.forward signature so that visualisation helpers
    like make_recon_grid / make_pca_manifold work unchanged. Pass
    return_z=True to additionally get the sampled latent (the trainer's
    main loop needs it for the TC term).
    """

    def __init__(
        self,
        latent_dim: int = 10,
        img_size: torch.Size = torch.Size([1, 64, 64]),
        disc_hidden_dim: int = 1000,
        disc_num_layers: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)
        self.discriminator = Discriminator(
            latent_dim,
            hidden_dim=disc_hidden_dim,
            num_layers=disc_num_layers,
        )

    def forward(self, x: torch.Tensor, *, return_z: bool = False):
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        if return_z:
            return x_hat, mu, logvar, z
        return x_hat, mu, logvar


def permute_dims(z: torch.Tensor) -> torch.Tensor:
    """Permute each latent dim independently across the batch.

    Given z of shape (B, D), returns z_perm where each column has been
    independently shuffled. Per-dim marginals q(z_i) are preserved;
    cross-dim dependencies are destroyed — i.e. z_perm is a sample from
    ∏_i q(z_i), the product of the marginals.
    """
    if z.dim() != 2:
        raise ValueError(f"permute_dims expects 2D (B, D), got shape {tuple(z.shape)}")
    B, D = z.shape
    perms = torch.stack(
        [torch.randperm(B, device=z.device) for _ in range(D)],
        dim=1,
    )  # (B, D)
    return torch.gather(z, 0, perms)
