from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (T2)
# ---------------------------------------------------------------------------

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lam: float) -> Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (lam,) = ctx.saved_tensors
        return -lam.item() * grad_output, None


class GRLLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lam", torch.tensor(0.0))

    def forward(self, x: Tensor, lam: float | None = None) -> Tensor:
        if lam is not None:
            self.lam.fill_(lam)
        return GRLFunction.apply(x, self.lam.item())


# ---------------------------------------------------------------------------
# Slot classifier — shared by CE and GRL heads (T3)
# ---------------------------------------------------------------------------

class SlotClassifier(nn.Module):
    def __init__(self, z_d_dim: int, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_d_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# SPVAELoss (T4)
# ---------------------------------------------------------------------------

def _kl(mu: Tensor, lv: Tensor) -> Tensor:
    return -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()


def _diag_gaussian_entropy(lv: Tensor) -> Tensor:
    """Differential entropy of a diagonal Gaussian: 0.5 * sum(1 + lv)."""
    return 0.5 * (1 + lv).sum(1).mean()


class SPVAELoss(nn.Module):
    def __init__(
        self,
        z_d_dim: int = 2,
        n_wall_classes: int = 10,
        n_floor_classes: int = 10,
        vicreg_eps: float = 1.0,
        orth_weight: float = 1.0,
        mi_weight: float = 0.1,
        ce_weight: float = 1.0,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.vicreg_eps = vicreg_eps
        self.orth_weight = orth_weight
        self.mi_weight = mi_weight
        self.ce_weight = ce_weight
        self.grl_weight = grl_weight

        # correct-head classifiers
        self.clf_d1 = SlotClassifier(z_d_dim, n_wall_classes)
        self.clf_d2 = SlotClassifier(z_d_dim, n_floor_classes)

        # wrong-head classifiers (receive GRL output)
        self.clf_grl_d1 = SlotClassifier(z_d_dim, n_floor_classes)
        self.clf_grl_d2 = SlotClassifier(z_d_dim, n_wall_classes)

        self.grl = GRLLayer()

    def forward(
        self,
        fwd_out: tuple[Tensor, ...],
        x: Tensor,
        wall_labels: Tensor,
        floor_labels: Tensor,
        beta_c: float,
        beta_d: float,
        dann_lambda: float,
    ) -> dict[str, Tensor]:
        x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2 = fwd_out
        B = x.shape[0]

        # -- reconstruction
        recon = F.mse_loss(x_hat, x)

        # -- KL terms
        kl_c  = _kl(mu_c,  lv_c)
        kl_d1 = _kl(mu_d1, lv_d1)
        kl_d2 = _kl(mu_d2, lv_d2)

        # -- correct-head CE (use means for stability)
        ce_wall  = F.cross_entropy(self.clf_d1(mu_d1), wall_labels)
        ce_floor = F.cross_entropy(self.clf_d2(mu_d2), floor_labels)

        # -- wrong-head GRL
        z_d1_rev = self.grl(z_d1, dann_lambda)
        z_d2_rev = self.grl(z_d2, dann_lambda)
        grl_wall  = F.cross_entropy(self.clf_grl_d1(z_d1_rev), floor_labels)
        grl_floor = F.cross_entropy(self.clf_grl_d2(z_d2_rev), wall_labels)

        # -- orthogonality penalty ‖Cov(z_d1, z_d2)‖²_F
        z1c = z_d1 - z_d1.mean(0)
        z2c = z_d2 - z_d2.mean(0)
        C = z1c.T @ z2c / (B - 1)
        orth = (C ** 2).sum()

        # -- MI lower bound via Gaussian entropy: H(z_d1) + H(z_d2) - H([z_d1, z_d2])
        h1 = _diag_gaussian_entropy(lv_d1)
        h2 = _diag_gaussian_entropy(lv_d2)
        # joint log-var from diagonal of empirical covariance of concat
        z_joint = torch.cat([z_d1, z_d2], dim=1)
        var_joint = z_joint.var(dim=0).clamp(min=1e-6)
        lv_joint = var_joint.log()
        h_joint = 0.5 * (1 + lv_joint).sum()
        mi = h1 + h2 - h_joint

        # -- VICReg variance guard
        std_d1 = z_d1.std(dim=0)
        std_d2 = z_d2.std(dim=0)
        vicreg = (
            F.relu(self.vicreg_eps - std_d1).pow(2).sum()
            + F.relu(self.vicreg_eps - std_d2).pow(2).sum()
        )

        # -- total
        total = (
            recon
            + beta_c * kl_c
            + beta_d * (kl_d1 + kl_d2)
            + self.ce_weight  * (ce_wall + ce_floor)
            + self.grl_weight * (grl_wall + grl_floor)
            + self.orth_weight * orth
            + self.mi_weight  * mi
            + vicreg
        )

        return {
            "recon":     recon,
            "kl_c":      kl_c,
            "kl_d1":     kl_d1,
            "kl_d2":     kl_d2,
            "ce_wall":   ce_wall,
            "ce_floor":  ce_floor,
            "grl_wall":  grl_wall,
            "grl_floor": grl_floor,
            "orth":      orth,
            "mi":        mi,
            "vicreg":    vicreg,
            "total":     total,
        }
