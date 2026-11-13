from __future__ import annotations

from torch import nn, Tensor


class ContextHead(nn.Module):
    """Maps the trunk feature vector to (μ_c, log σ²_c)."""

    def __init__(self, in_dim: int, z_c_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, z_c_dim * 2)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        mu, lv = self.fc(h).chunk(2, dim=-1)
        return mu, lv


class SlotHead(nn.Module):
    """Maps the trunk feature vector to (μ_d, log σ²_d) for one hue slot."""

    def __init__(self, in_dim: int, z_d_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, z_d_dim * 2)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        mu, lv = self.fc(h).chunk(2, dim=-1)
        return mu, lv
