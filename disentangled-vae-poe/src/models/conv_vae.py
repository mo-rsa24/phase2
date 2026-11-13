"""Convolutional VAE for 64×64 images (Stage 3 — Plan 03).

Architecture from Locatello et al. 2019 (disentanglement benchmark):
    Encoder : 4 × Conv(stride=2) → 2 × Linear → (mu, log_var)
    Decoder : 2 × Linear → 4 × ConvTranspose(stride=2) → Sigmoid
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


class ConvVAE(nn.Module):
    """Convolutional β-VAE for 3×64×64 images.

    Args:
        latent_dim : dimensionality of the latent code z.
    """

    def __init__(self, latent_dim: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim * 2),
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 4 * 4), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mu, log_var), each (B, latent_dim)."""
        h = self.encoder(x).flatten(1)
        mu, log_var = self.enc_fc(h).chunk(2, dim=-1)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Return reconstruction (B, 3, 64, 64) ∈ [0, 1]."""
        h = self.dec_fc(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = (0.5 * log_var).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (x_hat, mu, log_var, z)."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z
