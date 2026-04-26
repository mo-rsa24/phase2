"""Sanity-check the VAE and stress the GPU.

Phase 1 — small forward pass, print every tensor shape, assert correctness.
Phase 2 — large-batch forward+backward loop with bf16 autocast, TF32, and
torch.compile, reporting throughput and peak VRAM. Watch `nvidia-smi -l 1`
in another shell to see utilization.
"""

from pathlib import Path
import sys
import time

import torch
from torch import optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vae import VAE


def shape_sanity(device: torch.device, latent_dim: int, img_size: torch.Size) -> None:
    batch_size = 4
    model = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    model.eval()

    x = torch.rand(batch_size, *img_size, device=device)
    print(f"input x         : {tuple(x.shape)}")

    with torch.no_grad():
        z, mu, logvar = model.encoder(x)
        print(f"encoder z       : {tuple(z.shape)}")
        print(f"encoder mu      : {tuple(mu.shape)}")
        print(f"encoder logvar  : {tuple(logvar.shape)}")

        recon_from_z = model.decoder(z)
        print(f"decoder(z)      : {tuple(recon_from_z.shape)}")

        recon_x, mu_full, logvar_full = model(x)
        print(f"VAE recon_x     : {tuple(recon_x.shape)}")
        print(f"VAE mu          : {tuple(mu_full.shape)}")
        print(f"VAE logvar      : {tuple(logvar_full.shape)}")
        print(f"VAE kl (scalar) : {model.kl.item():.4f}")

    assert recon_x.shape == x.shape, f"recon shape {recon_x.shape} != input {x.shape}"
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert recon_x.min() >= 0.0 and recon_x.max() <= 1.0, "decoder output should be in [0, 1]"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params          : {n_params:,}")
    print("sanity check    : OK")


def gpu_stress(
    device: torch.device,
    latent_dim: int,
    img_size: torch.Size,
    batch_size: int = 8192,
    iters: int = 200,
    warmup: int = 25,
) -> None:
    print()
    print("-- gpu stress --")
    print(f"batch           : {batch_size}")
    print(f"iters           : {iters} (warmup {warmup})")

    model = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4, fused=True)

    compiled = False
    try:
        model = torch.compile(model)
        compiled = True
    except Exception as e:
        print(f"torch.compile skipped: {e}")
    print(f"torch.compile   : {'on' if compiled else 'off'}")

    x = torch.rand(batch_size, *img_size, device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_val = float("nan")
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            recon_x, mu, logvar = model(x)
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            loss = recon_loss + kl
        loss.backward()
        opt.step()
        loss_val = loss.detach()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    samples = iters * batch_size
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"throughput      : {samples / elapsed:,.0f} samples/sec")
    print(f"step time       : {1000 * elapsed / iters:.2f} ms")
    print(f"peak vram       : {peak_gb:.2f} / {total_gb:.1f} GB ({100 * peak_gb / total_gb:.1f}%)")
    print(f"final loss      : {loss_val.item():.4f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device          : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu             : {props.name}")
        print(f"sm              : sm_{props.major}{props.minor}")
        print(f"vram total      : {props.total_memory / 1e9:.1f} GB")
    print(f"torch           : {torch.__version__}")
    print()

    # Perf knobs.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    latent_dim = 10
    img_size = torch.Size([1, 64, 64])

    shape_sanity(device, latent_dim, img_size)
    if device.type == "cuda":
        gpu_stress(device, latent_dim, img_size)


if __name__ == "__main__":
    main()
