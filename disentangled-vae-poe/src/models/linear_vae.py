import numpy as np
import torch
from torch import nn, Tensor


class LinearVAE(nn.Module):
    """Minimal linear VAE for the 2-D Gaussian rotation experiment.

    Encoder: one nn.Linear(2 → 2·latent_dim), no activation; output is split
             into (mu, log_var).
    Decoder: one nn.Linear(latent_dim → 2), no activation.

    State-dict sub-module names (`encoder`, `decoder`) follow the same
    convention as src/models/vae.py so existing checkpoint-loading helpers
    are compatible.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(2, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, 2)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map x → (mu, log_var) via a single linear projection."""
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Map latent sample z → reconstructed observation."""
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Draw z ~ N(mu, exp(0.5·log_var)) using the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, reparameterize, decode.

        Returns
        -------
        x_hat : (batch, 2)  reconstructed observation
        mu    : (batch, latent_dim)  posterior mean
        log_var : (batch, latent_dim)  posterior log-variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
