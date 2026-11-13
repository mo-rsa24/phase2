from __future__ import annotations

import torch
from torch import nn, Tensor

from .encoder import SPVAEEncoder
from .heads import ContextHead, SlotHead
from .decoder import SPVAEDecoder


class SPVAE(nn.Module):
    """Partitioned-representation VAE with three independent posterior slots.

    Latent code: z = (z_c, z_d1, z_d2)
      z_c   — shared context (shape, scale, orientation)
      z_d1  — wall_hue slot
      z_d2  — floor_hue slot
    """

    def __init__(self, z_c_dim: int = 8, z_d_dim: int = 2) -> None:
        super().__init__()
        self.z_c_dim = z_c_dim
        self.z_d_dim = z_d_dim

        self.encoder   = SPVAEEncoder()
        self.ctx_head  = ContextHead(256, z_c_dim)
        self.slot_head1 = SlotHead(256, z_d_dim)
        self.slot_head2 = SlotHead(256, z_d_dim)
        self.decoder   = SPVAEDecoder(z_c_dim, z_d_dim)

    @staticmethod
    def _reparam(mu: Tensor, lv: Tensor) -> Tensor:
        std = (0.5 * lv).exp()
        return mu + std * torch.randn_like(std)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (z_c, z_d1, z_d2) — reparameterised samples."""
        h = self.encoder(x)
        mu_c,  lv_c  = self.ctx_head(h)
        mu_d1, lv_d1 = self.slot_head1(h)
        mu_d2, lv_d2 = self.slot_head2(h)
        z_c  = self._reparam(mu_c,  lv_c)
        z_d1 = self._reparam(mu_d1, lv_d1)
        z_d2 = self._reparam(mu_d2, lv_d2)
        return z_c, z_d1, z_d2

    def decode(self, z_c: Tensor, z_d1: Tensor, z_d2: Tensor) -> Tensor:
        """Return reconstruction (B, 3, 64, 64)."""
        return self.decoder(torch.cat([z_c, z_d1, z_d2], dim=-1))

    def compose(self, z_c: Tensor, z_d1_src: Tensor, z_d2_src: Tensor) -> Tensor:
        """Slot substitution: decode a composite from independently sourced slots."""
        return self.decode(z_c, z_d1_src, z_d2_src)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return (x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2)."""
        h = self.encoder(x)
        mu_c,  lv_c  = self.ctx_head(h)
        mu_d1, lv_d1 = self.slot_head1(h)
        mu_d2, lv_d2 = self.slot_head2(h)
        z_c  = self._reparam(mu_c,  lv_c)
        z_d1 = self._reparam(mu_d1, lv_d1)
        z_d2 = self._reparam(mu_d2, lv_d2)
        x_hat = self.decode(z_c, z_d1, z_d2)
        return x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2
