#!/usr/bin/env python3
"""SP-VAE slot-isolation probe evaluation (Plan 04, T1).

Fits four logistic-regression probes on the val set:
  acc_d1_wall  — z_d1 → wall_hue  (correct slot, expect >= 0.90)
  acc_d2_floor — z_d2 → floor_hue (correct slot, expect >= 0.90)
  acc_d2_wall  — z_d2 → wall_hue  (leakage probe,  expect < 0.20)
  acc_d1_floor — z_d1 → floor_hue (leakage probe,  expect < 0.20)

Writes results/shapes3d/spvae_disentanglement.csv and logs to W&B.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

_HERE       = os.path.dirname(os.path.abspath(__file__))
_SCOPE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_SCOPE_ROOT)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SCOPE_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from src.models.sp_vae.model import SPVAE
from src.datasets.shapes3d import Shapes3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size",   type=int, default=256)
    p.add_argument("--num-workers",  type=int, default=0)
    p.add_argument("--no-wandb",     action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def encode_val(model, val_loader, device):
    model.eval()
    mu_d1s, mu_d2s, wall_ls, floor_ls = [], [], [], []
    for x, labels in val_loader:
        fwd = model(x.to(device))
        # fwd: (x_hat, mu_c, lv_c, z_c, mu_d1, lv_d1, z_d1, mu_d2, lv_d2, z_d2)
        mu_d1s.append(fwd[4].cpu())
        mu_d2s.append(fwd[7].cpu())
        wall_ls.append(labels[:, 2])   # object_hue index (plan calls it wall_hue)
        floor_ls.append(labels[:, 3])  # scale index      (plan calls it floor_hue)
    model.train()
    return (
        torch.cat(mu_d1s).numpy(),
        torch.cat(mu_d2s).numpy(),
        torch.cat(wall_ls).numpy(),
        torch.cat(floor_ls).numpy(),
    )


def probe(X, y, name):
    clf = LogisticRegression(max_iter=1000, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    acc = clf.score(X, y)
    return acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = SPVAE(z_c_dim=cfg["z_c_dim"], z_d_dim=cfg["z_d_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_ds     = Shapes3D(split="val", seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print("Encoding val set …")
    mu_d1, mu_d2, wall_labels, floor_labels = encode_val(model, val_loader, device)

    probes = {
        "acc_d1_wall":  (mu_d1,  wall_labels),
        "acc_d2_floor": (mu_d2,  floor_labels),
        "acc_d2_wall":  (mu_d2,  wall_labels),
        "acc_d1_floor": (mu_d1,  floor_labels),
    }
    thresholds = {
        "acc_d1_wall":  (">=", 0.90),
        "acc_d2_floor": (">=", 0.90),
        "acc_d2_wall":  ("<",  0.20),
        "acc_d1_floor": ("<",  0.20),
    }

    results = {}
    for name, (X, y) in probes.items():
        print(f"  fitting {name} …")
        acc = probe(X, y, name)
        results[name] = acc
        op, thresh = thresholds[name]
        passed = (acc >= thresh) if op == ">=" else (acc < thresh)
        tag = "PASS" if passed else "FAIL"
        print(f"  {name:<16}  {acc:.4f}  [{tag}]")

    # Write CSV
    out_dir = Path(_SCOPE_ROOT) / "results" / "shapes3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "spvae_disentanglement.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe", "accuracy"])
        for name, acc in results.items():
            writer.writerow([name, f"{acc:.6f}"])
    print(f"\nWrote {csv_path}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project="disentangled-vae-poe",
                         name="spvae_probe_eval", tags=["sp-vae", "eval"],
                         resume="allow")
        wandb.log({f"probe/{k}": v for k, v in results.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
