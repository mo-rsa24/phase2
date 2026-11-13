from __future__ import annotations

from torch import nn, Tensor


class SPVAEDecoder(nn.Module):
    """Mirror-ConvVAE decoder.

    Input: concat(z_c, z_d1, z_d2) of dim z_c_dim + 2*z_d_dim.
    Output: (B, 3, 64, 64) ∈ [0, 1].
    """

    def __init__(self, z_c_dim: int, z_d_dim: int) -> None:
        super().__init__()
        latent_dim = z_c_dim + 2 * z_d_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, 64, 4, 4)
        return self.deconv(h)
