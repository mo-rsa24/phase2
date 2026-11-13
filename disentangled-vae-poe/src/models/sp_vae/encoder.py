from __future__ import annotations

import torch
from torch import nn, Tensor


class ResBlock(nn.Module):
    """Two 3×3 convs with BN + ReLU and a 1×1 skip projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (reduction ratio r=16)."""

    def __init__(self, channels: int, r: int = 16) -> None:
        super().__init__()
        mid = max(channels // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class SPVAEEncoder(nn.Module):
    """ConvVAE trunk with ResBlocks and SE attention.

    Produces a 256-d feature vector consumed by the three posterior heads.
    Channel progression mirrors ConvVAE: 3→32→32→64→64 with stride-2 at each stage.
    """

    def __init__(self) -> None:
        super().__init__()

        def stage(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                ResBlock(out_c, out_c),
            )

        self.stages = nn.Sequential(
            stage(3,  32),
            stage(32, 32),
            stage(32, 64),
            stage(64, 64),
        )
        self.se = SEBlock(64)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.stages(x)        # (B, 64, 4, 4)
        h = self.se(h)
        return self.fc(h.flatten(1))   # (B, 256)
