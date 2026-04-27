#!/usr/bin/env python3
"""VAE Disentanglement Explorer — interactive analysis suite.

Tabs
----
  Select Run     Load any of the 4 sweep experiments or a custom checkpoint.
  Explore        dSprites factor sliders → encode; latent sliders → decode live.
  Traversal Grid Rows = latent dims (sorted by KL), cols = traversal values.
  Analysis       KL activity spectrum · factor-latent correlation · MIG score.

Usage
-----
    # Preload a checkpoint:
    python scripts/disentanglement_explorer.py \\
        --checkpoint checkpoints/vae/vae_z10_beta1.0_seed42/best.pt

    # Or just open the UI and click a preset:
    python scripts/disentanglement_explorer.py

Then open http://localhost:5050
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import base64
import io
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES, load_dsprites
from src.metrics.disentanglement import (
    compute_mig,
    factor_latent_correlation,
    kl_per_dim,
)
from src.utils.vae_inspection import (
    build_factor_index,
    load_encoder_decoder,
    run_vae_pass,
    to_uint8_rgb,
)


# ── Sweep experiment presets ──────────────────────────────────────────────────

EXPERIMENTS = [
    {"id": 1, "label": "Exp 1 — z=10, β=1.0", "purpose": "Baseline VAE",
     "latent_dim": 10, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z10_beta1.0_seed42/best.pt"},
    {"id": 2, "label": "Exp 2 — z=10, β=4.0", "purpose": "β-VAE",
     "latent_dim": 10, "beta": 4.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z10_beta4.0_seed42/best.pt"},
    {"id": 3, "label": "Exp 3 — z=4, β=1.0",  "purpose": "Undercomplete",
     "latent_dim":  4, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z4_beta1.0_seed42/best.pt"},
    {"id": 4, "label": "Exp 4 — z=20, β=1.0", "purpose": "Overcomplete",
     "latent_dim": 20, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z20_beta1.0_seed42/best.pt"},
]


# ── App state ─────────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state = {
    "encoder":    None,
    "decoder":    None,
    "device":     "cpu",
    "latent_dim": None,
    "ckpt_path":  None,
    # Cached batch encodings (filled in a background thread after load)
    "enc_mu":      None,   # (N, latent_dim)
    "enc_logvar":  None,   # (N, latent_dim)
    "enc_factors": None,   # (N, 6)
    "kl_arr":      None,   # (latent_dim,)
    # MIG background task
    "mig_status": "idle",  # idle | computing | done | error
    "mig_result": None,
    "mig_error":  None,
}

_dataset:      dict | None = None
_factor_index: dict | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64png(arr: np.ndarray) -> str:
    """Float (H,W) or (H,W,3) in [0,1] → base64 PNG data-URI."""
    rgb = to_uint8_rgb(arr)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def _encode_imgs(imgs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 64, 64) float32 → (mu, logvar) each (N, latent_dim)."""
    encoder = _state["encoder"]
    device  = _state["device"]
    encoder.eval()
    x       = torch.from_numpy(imgs_np[:, np.newaxis]).float().to(device)
    mu, lv  = encoder.encode(x)
    return mu.cpu().numpy(), lv.cpu().numpy()


def _cache_encoded_samples(n: int = 3000) -> None:
    """Encode N random dataset images and cache the results."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(_dataset["imgs"]), min(n, len(_dataset["imgs"])), replace=False)
    imgs    = _dataset["imgs"][idx].astype(np.float32)
    factors = _dataset["latents_classes"][idx]

    mu_list, lv_list = [], []
    for i in range(0, len(imgs), 512):
        mu_b, lv_b = _encode_imgs(imgs[i : i + 512])
        mu_list.append(mu_b)
        lv_list.append(lv_b)

    mu_arr = np.concatenate(mu_list)
    lv_arr = np.concatenate(lv_list)

    with _lock:
        _state["enc_mu"]      = mu_arr
        _state["enc_logvar"]  = lv_arr
        _state["enc_factors"] = factors
        _state["kl_arr"]      = kl_per_dim(mu_arr, lv_arr)


# ── Flask ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE Disentanglement Explorer")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--port",       type=int, default=5050)
    return p.parse_args()


args = parse_args()
app  = Flask(__name__, template_folder="templates")

print("Loading dSprites …")
_dataset      = load_dsprites(args.data_dir)
_factor_index = build_factor_index(_dataset)
print(f"Dataset ready: {len(_dataset['imgs'])} images")

_state["device"] = args.device

if args.checkpoint and Path(args.checkpoint).exists():
    enc, dec = load_encoder_decoder(
        encoder_checkpoint=args.checkpoint,
        decoder_checkpoint=args.checkpoint,
        device=args.device,
    )
    with _lock:
        _state.update(encoder=enc, decoder=dec,
                      latent_dim=int(enc.latent_dim), ckpt_path=args.checkpoint)
    threading.Thread(target=_cache_encoded_samples, daemon=True).start()
    print(f"Model preloaded: latent_dim={enc.latent_dim}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "disentanglement_explorer.html",
        factor_names=list(FACTOR_NAMES),
        factor_sizes=list(FACTOR_SIZES),
        experiments=EXPERIMENTS,
        loaded_ckpt=_state["ckpt_path"],
        loaded_latent_dim=_state["latent_dim"],
    )


@app.route("/api/load", methods=["POST"])
def api_load():
    path = Path((request.json or {}).get("checkpoint", "").strip())
    if not str(path):
        return jsonify({"error": "No checkpoint path provided"}), 400
    if not path.exists():
        return jsonify({"error": f"File not found: {path}"}), 404
    try:
        enc, dec = load_encoder_decoder(
            encoder_checkpoint=str(path),
            decoder_checkpoint=str(path),
            device=args.device,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with _lock:
        _state.update(
            encoder=enc, decoder=dec,
            latent_dim=int(enc.latent_dim), ckpt_path=str(path),
            enc_mu=None, enc_logvar=None, enc_factors=None, kl_arr=None,
            mig_status="idle", mig_result=None, mig_error=None,
        )
    threading.Thread(target=_cache_encoded_samples, daemon=True).start()
    return jsonify({"ok": True, "latent_dim": _state["latent_dim"], "checkpoint": str(path)})


@app.route("/api/encode_factors", methods=["POST"])
def api_encode_factors():
    if _state["encoder"] is None:
        return jsonify({"error": "No model loaded"}), 400

    data          = request.json or {}
    factor_values = {n: int(data.get(n, 0)) for n in FACTOR_NAMES}

    try:
        result = run_vae_pass(
            _dataset, _state["encoder"], _state["decoder"],
            factor_values=factor_values,
            use_mean_latent=True,
            factor_index=_factor_index,
        )
    except (KeyError, IndexError, ValueError) as e:
        return jsonify({"error": str(e)}), 404

    recon = result.reconstruction
    if recon.ndim == 3 and recon.shape[-1] == 1:
        recon = recon[..., 0]
    diff = np.abs(result.input_image.astype(np.float32) - recon)
    if diff.max() > 0:
        diff /= diff.max()

    lv_clip = np.clip(result.logvar, -10, 10)
    kl_dims = (-0.5 * (1 + lv_clip - result.mu ** 2 - np.exp(lv_clip))).tolist()

    return jsonify({
        "original":       _b64png(result.input_image),
        "reconstruction": _b64png(recon),
        "difference":     _b64png(diff),
        "mu":             result.mu.tolist(),
        "logvar":         result.logvar.tolist(),
        "mse":            float(result.mse),
        "kl_per_dim":     kl_dims,
    })


@app.route("/api/decode", methods=["POST"])
def api_decode():
    if _state["decoder"] is None:
        return jsonify({"error": "No model loaded"}), 400

    z = np.array((request.json or {}).get("z", []), dtype=np.float32)
    if len(z) != _state["latent_dim"]:
        return jsonify({"error": f"Expected z of length {_state['latent_dim']}"}), 400

    z_t = torch.from_numpy(z[np.newaxis]).to(args.device)
    with torch.no_grad():
        img = _state["decoder"](z_t).cpu().squeeze().numpy()
    return jsonify({"reconstruction": _b64png(img)})


@app.route("/api/traversal_grid", methods=["POST"])
def api_traversal_grid():
    if _state["encoder"] is None:
        return jsonify({"error": "No model loaded"}), 400

    data        = request.json or {}
    mu          = np.array(data.get("mu",     []), dtype=np.float32)
    logvar      = np.array(data.get("logvar", []), dtype=np.float32)
    n_steps     = int(data.get("n_steps",     11))
    sigma_range = float(data.get("sigma_range", 3.0))
    max_dims    = int(data.get("max_dims",    10))
    use_anchor_sigma = bool(data.get("use_anchor_sigma", False))

    with _lock:
        kl = _state["kl_arr"]

    # Fall back to anchor's own KL if global cache not ready
    lv_clip = np.clip(logvar, -10, 10)
    kl_anchor = (-0.5 * (1 + lv_clip - mu ** 2 - np.exp(lv_clip)))
    kl_use    = kl if kl is not None else kl_anchor

    sorted_dims = np.argsort(kl_use)[::-1][:max_dims]
    n_show      = len(sorted_dims)

    cell  = 64
    inch  = 80           # pixels per inch
    fw    = n_steps * cell / inch + 1.6
    fh    = n_show  * cell / inch + 0.6
    fig, axes = plt.subplots(n_show, n_steps, figsize=(fw, fh), squeeze=False)

    decoder = _state["decoder"]
    device  = args.device

    for row, dim in enumerate(sorted_dims):
        if use_anchor_sigma:
            sigma = float(np.exp(0.5 * np.clip(logvar[dim], -10, 10)))
            lo    = float(mu[dim]) - sigma_range * sigma
            hi    = float(mu[dim]) + sigma_range * sigma
        else:
            lo = float(mu[dim]) - sigma_range
            hi = float(mu[dim]) + sigma_range

        for col, val in enumerate(np.linspace(lo, hi, n_steps)):
            z       = mu.copy()
            z[dim]  = float(val)
            z_t     = torch.from_numpy(z[np.newaxis]).float().to(device)
            with torch.no_grad():
                img = decoder(z_t).cpu().squeeze().numpy()
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"z{dim}\nKL={kl_use[dim]:.2f}", fontsize=7,
            rotation=0, labelpad=36, va="center",
        )

    for col, sv in enumerate(np.linspace(-sigma_range, sigma_range, n_steps)):
        axes[0, col].set_title(f"{sv:+.1f}σ", fontsize=7)

    fig.suptitle("Latent Traversal Grid", fontsize=9, y=1.01)
    plt.tight_layout(pad=0.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return jsonify({"grid": b64})


@app.route("/api/kl_spectrum", methods=["POST"])
def api_kl_spectrum():
    with _lock:
        kl  = _state["kl_arr"]
        ldim = _state["latent_dim"]
    if kl is None:
        return jsonify({"error": "Still computing — try again in a moment."}), 503

    active = kl > 0.1
    colors = ["#2563eb" if a else "#d1d5db" for a in active]
    dims   = [f"z{i}" for i in range(ldim)]

    return jsonify({
        "data": [{
            "type": "bar", "x": dims, "y": kl.tolist(),
            "marker": {"color": colors},
            "text": [f"{v:.3f}" for v in kl], "textposition": "outside",
        }],
        "layout": {
            "title": {"text": "Mean KL per Latent Dimension", "font": {"size": 14}},
            "xaxis": {"title": "Latent dim"},
            "yaxis": {"title": "Mean KL (nats)", "rangemode": "tozero"},
            "margin": {"t": 55, "b": 55, "l": 60, "r": 20},
            "plot_bgcolor": "#f9fafb", "paper_bgcolor": "white",
            "height": 360,
            "shapes": [{"type": "line", "x0": -0.5, "x1": ldim - 0.5,
                         "y0": 0.1, "y1": 0.1,
                         "line": {"color": "#ef4444", "dash": "dot", "width": 1}}],
            "annotations": [{
                "x": 0.99, "y": 0.97, "xref": "paper", "yref": "paper",
                "text": f"Active (KL > 0.1): {int(active.sum())}/{ldim}",
                "showarrow": False, "align": "right",
                "bgcolor": "white", "bordercolor": "#d1d5db", "borderwidth": 1,
                "font": {"size": 11},
            }],
        },
    })


@app.route("/api/correlation", methods=["POST"])
def api_correlation():
    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
        kl      = _state["kl_arr"]
    if mu is None:
        return jsonify({"error": "Still computing — try again in a moment."}), 503

    corr  = factor_latent_correlation(mu, factors)
    order = np.argsort(kl)[::-1] if kl is not None else np.arange(corr.shape[0])
    cs    = corr[order]
    ks    = (kl[order] if kl is not None else np.zeros(len(order)))
    ylabs = [f"z{i} (KL={ks[r]:.2f})" for r, i in enumerate(order)]

    return jsonify({
        "data": [{
            "type": "heatmap",
            "z": cs.tolist(), "x": list(FACTOR_NAMES), "y": ylabs,
            "colorscale": "Blues", "zmin": 0, "zmax": 1,
            "text": [[f"{v:.2f}" for v in row] for row in cs],
            "texttemplate": "%{text}", "showscale": True,
        }],
        "layout": {
            "title": {"text": "|Spearman ρ|: Latent × Factor", "font": {"size": 14}},
            "xaxis": {"title": "Ground-truth factor", "side": "bottom"},
            "yaxis": {"title": "Latent dim (↑ KL)", "autorange": "reversed"},
            "margin": {"t": 55, "b": 70, "l": 130, "r": 20},
            "height": max(320, len(order) * 28 + 120),
        },
    })


# ── MIG (background task) ─────────────────────────────────────────────────────

def _run_mig() -> None:
    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
    if mu is None:
        with _lock:
            _state["mig_status"] = "error"
            _state["mig_error"]  = "No encoded samples available yet."
        return
    try:
        score, per_factor = compute_mig(mu, factors)
        with _lock:
            _state["mig_result"] = {"score": score, "per_factor": per_factor}
            _state["mig_status"] = "done"
    except Exception as e:
        with _lock:
            _state["mig_status"] = "error"
            _state["mig_error"]  = str(e)


@app.route("/api/mig/start", methods=["POST"])
def api_mig_start():
    with _lock:
        if _state["mig_status"] == "computing":
            return jsonify({"status": "computing"})
        _state["mig_status"] = "computing"
        _state["mig_result"] = None
        _state["mig_error"]  = None
    threading.Thread(target=_run_mig, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/mig/status")
def api_mig_status():
    with _lock:
        resp = {"status": _state["mig_status"]}
        if _state["mig_result"]:
            resp["result"] = _state["mig_result"]
        if _state["mig_error"]:
            resp["error"] = _state["mig_error"]
    return jsonify(resp)


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*58}")
    print("  VAE Disentanglement Explorer")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*58}\n")
    app.run(debug=False, use_reloader=False, port=args.port)
