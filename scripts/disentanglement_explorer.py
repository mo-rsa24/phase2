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
import json
import re
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.feature_selection import mutual_info_classif

from explorer_help import HELP
from src.datasets.dsprites import FACTOR_NAMES, FACTOR_SIZES, load_dsprites
from src.metrics.dci import compute_dci
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
    # ---- Phase 2a: IID baseline (Exp 1–5) ------------------------------
    {"id": 1, "label": "Exp 1 — z=10, β=1.0", "purpose": "Baseline VAE",
     "split": "iid",
     "latent_dim": 10, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z10_beta1.0_seed42/best.pt"},
    {"id": 2, "label": "Exp 2 — z=10, β=4.0", "purpose": "β-VAE",
     "split": "iid",
     "latent_dim": 10, "beta": 4.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z10_beta4.0_seed42/best.pt"},
    {"id": 3, "label": "Exp 3 — z=4, β=1.0",  "purpose": "Undercomplete",
     "split": "iid",
     "latent_dim":  4, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z4_beta1.0_seed42/best.pt"},
    {"id": 4, "label": "Exp 4 — z=20, β=1.0", "purpose": "Overcomplete",
     "split": "iid",
     "latent_dim": 20, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/vae_z20_beta1.0_seed42/best.pt"},
    {"id": 5, "label": "Exp 5 — z=10, γ=35.0", "purpose": "FactorVAE",
     "split": "iid",
     "latent_dim": 10, "beta": None, "gamma": 35.0, "seed": 42,
     "checkpoint": "checkpoints/factor_vae/factorvae_z10_gamma35.0_seed42/best.pt"},
    # ---- Phase 2b: correlated (scale, orientation+) split (Exp 6–10) ---
    {"id": 6, "label": "Exp 6 — corr · z=10, β=1.0", "purpose": "Baseline VAE (corr)",
     "split": "correlated", "corr_factor_a": "scale", "corr_factor_b": "orientation",
     "corr_direction": "positive",
     "latent_dim": 10, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/correlated_vae_z10_beta1.0_seed42/best.pt"},
    {"id": 7, "label": "Exp 7 — corr · z=10, β=4.0", "purpose": "β-VAE (corr)",
     "split": "correlated", "corr_factor_a": "scale", "corr_factor_b": "orientation",
     "corr_direction": "positive",
     "latent_dim": 10, "beta": 4.0, "seed": 42,
     "checkpoint": "checkpoints/vae/correlated_vae_z10_beta4.0_seed42/best.pt"},
    {"id": 8, "label": "Exp 8 — corr · z=4, β=1.0",  "purpose": "Undercomplete (corr)",
     "split": "correlated", "corr_factor_a": "scale", "corr_factor_b": "orientation",
     "corr_direction": "positive",
     "latent_dim":  4, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/correlated_vae_z4_beta1.0_seed42/best.pt"},
    {"id": 9, "label": "Exp 9 — corr · z=20, β=1.0", "purpose": "Overcomplete (corr)",
     "split": "correlated", "corr_factor_a": "scale", "corr_factor_b": "orientation",
     "corr_direction": "positive",
     "latent_dim": 20, "beta": 1.0, "seed": 42,
     "checkpoint": "checkpoints/vae/correlated_vae_z20_beta1.0_seed42/best.pt"},
    {"id": 10, "label": "Exp 10 — corr · z=10, γ=35.0", "purpose": "FactorVAE (corr)",
     "split": "correlated", "corr_factor_a": "scale", "corr_factor_b": "orientation",
     "corr_direction": "positive",
     "latent_dim": 10, "beta": None, "gamma": 35.0, "seed": 42,
     "checkpoint": "checkpoints/factor_vae/correlated_factorvae_z10_gamma35.0_seed42/best.pt"},
]


# ── App state ─────────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state = {
    "encoder":    None,
    "decoder":    None,
    "device":     "cpu",
    "latent_dim": None,
    "ckpt_path":  None,
    "beta":       None,    # β value of the loaded checkpoint (matched from EXPERIMENTS)
    "gamma":      None,    # γ (FactorVAE) of the loaded checkpoint, if applicable
    "split":      None,    # 'iid' / 'correlated' / 'heldout' / None (unknown)
    "corr_factor_a":   None,
    "corr_factor_b":   None,
    "corr_direction":  None,
    # Cached batch encodings (filled in a background thread after load)
    "enc_mu":       None,  # (N, latent_dim)
    "enc_logvar":   None,  # (N, latent_dim)
    "enc_factors":  None,  # (N, 6)
    "enc_indices":  None,  # (N,) — original dSprites dataset indices for cached samples
    "kl_arr":       None,  # (latent_dim,)
    # Lazy: filled the first time `/api/factor_conditional_histogram` runs
    "corr_matrix":           None,  # (latent_dim, 6) — |Spearman ρ|
    "mig_top_dim_per_factor": {},   # {factor_name: int}
    # MIG background task
    "mig_status": "idle",  # idle | computing | done | error
    "mig_result": None,
    "mig_error":  None,
    # DCI background task (same state machine as MIG)
    "dci_status": "idle",
    "dci_result": None,
    "dci_error":  None,
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
        _state["enc_indices"] = idx
        _state["kl_arr"]      = kl_per_dim(mu_arr, lv_arr)
        # Invalidate derived caches that depend on the encoded samples
        _state["corr_matrix"]           = None
        _state["mig_top_dim_per_factor"] = {}


# ── Flask ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE Disentanglement Explorer")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--port",       type=int, default=5050)
    return p.parse_args()


args = parse_args()
app  = Flask(__name__, template_folder="templates", static_folder="static")


def _seeds_by_exp() -> dict[int, list[dict]]:
    """Discover available seeds on disk for each EXPERIMENTS entry.

    For each experiment, glob the checkpoint path with the seed segment
    replaced by a wildcard (e.g. ``vae_z10_beta1.0_seed*/best.pt``) and
    parse the seed back from each match. Returns ``{exp.id: [{seed, path}, ...]}``
    sorted ascending by seed. Empty list if no checkpoints exist yet.
    """
    result: dict[int, list[dict]] = {}
    for exp in EXPERIMENTS:
        ckpt = exp["checkpoint"]
        pattern = re.sub(r"_seed\d+/", "_seed*/", ckpt)
        seeds: list[dict] = []
        for match in Path(".").glob(pattern):
            mm = re.search(r"_seed(\d+)/", str(match))
            if mm:
                seeds.append({"seed": int(mm.group(1)), "checkpoint": str(match)})
        seeds.sort(key=lambda s: s["seed"])
        result[exp["id"]] = seeds
    return result


def _disent_param_for_checkpoint(ckpt_path: str | None) -> dict:
    """Look up disentanglement hyperparameters + split for a checkpoint path.

    Returns a dict with keys 'beta', 'gamma', 'split', 'corr_factor_a',
    'corr_factor_b', 'corr_direction'. Values are None when unknown
    (custom path not in EXPERIMENTS, or fields irrelevant to the family).
    """
    empty = {
        "beta": None, "gamma": None, "split": None,
        "corr_factor_a": None, "corr_factor_b": None, "corr_direction": None,
    }
    if not ckpt_path:
        return empty
    # Compare seed-agnostic forms so non-default seeds still resolve to their
    # parent experiment's β/γ/split metadata.
    p_norm = re.sub(r"_seed\d+/", "_seed*/", str(ckpt_path))
    for exp in EXPERIMENTS:
        exp_norm = re.sub(r"_seed\d+/", "_seed*/", exp["checkpoint"])
        if exp_norm in p_norm:
            beta  = exp.get("beta")
            gamma = exp.get("gamma")
            return {
                "beta":  float(beta)  if beta  is not None else None,
                "gamma": float(gamma) if gamma is not None else None,
                "split": exp.get("split"),
                "corr_factor_a":  exp.get("corr_factor_a"),
                "corr_factor_b":  exp.get("corr_factor_b"),
                "corr_direction": exp.get("corr_direction"),
            }
    return empty

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
    _params = _disent_param_for_checkpoint(args.checkpoint)
    with _lock:
        _state.update(encoder=enc, decoder=dec,
                      latent_dim=int(enc.latent_dim),
                      ckpt_path=args.checkpoint,
                      beta=_params["beta"],
                      gamma=_params["gamma"],
                      split=_params["split"],
                      corr_factor_a=_params["corr_factor_a"],
                      corr_factor_b=_params["corr_factor_b"],
                      corr_direction=_params["corr_direction"])
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
        seeds_by_exp=_seeds_by_exp(),
        loaded_ckpt=_state["ckpt_path"],
        loaded_latent_dim=_state["latent_dim"],
        loaded_beta=_state["beta"],
        loaded_gamma=_state["gamma"],
        loaded_split=_state["split"],
        loaded_corr_a=_state["corr_factor_a"],
        loaded_corr_b=_state["corr_factor_b"],
        loaded_corr_direction=_state["corr_direction"],
        help_json=json.dumps(HELP),
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

    params = _disent_param_for_checkpoint(str(path))
    with _lock:
        _state.update(
            encoder=enc, decoder=dec,
            latent_dim=int(enc.latent_dim), ckpt_path=str(path),
            beta=params["beta"], gamma=params["gamma"],
            split=params["split"],
            corr_factor_a=params["corr_factor_a"],
            corr_factor_b=params["corr_factor_b"],
            corr_direction=params["corr_direction"],
            enc_mu=None, enc_logvar=None, enc_factors=None,
            enc_indices=None, kl_arr=None,
            corr_matrix=None, mig_top_dim_per_factor={},
            mig_status="idle", mig_result=None, mig_error=None,
            dci_status="idle", dci_result=None, dci_error=None,
        )
    threading.Thread(target=_cache_encoded_samples, daemon=True).start()
    return jsonify({
        "ok":         True,
        "latent_dim": _state["latent_dim"],
        "checkpoint": str(path),
        "beta":       params["beta"],
        "gamma":      params["gamma"],
        "split":      params["split"],
        "corr_factor_a":  params["corr_factor_a"],
        "corr_factor_b":  params["corr_factor_b"],
        "corr_direction": params["corr_direction"],
    })


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


# ── DCI (background task) ────────────────────────────────────────────────────
#
# Same state machine as MIG: idle → computing → done | error. The metric
# typically takes ~10–30 s on 3000 cached samples (RF n_estimators=50).
# Reads from `_state["enc_mu"]` and `_state["enc_factors"]`.

def _run_dci() -> None:
    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
    if mu is None or factors is None:
        with _lock:
            _state["dci_status"] = "error"
            _state["dci_error"]  = "No encoded samples available yet."
        return
    try:
        result = compute_dci(
            mu, factors,
            factor_names=FACTOR_NAMES,
            factor_sizes=FACTOR_SIZES,
            n_estimators=50,
            seed=0,
        )
        # numpy → JSON-friendly
        payload = {
            "importance":   result["importance"].tolist(),
            "D_per_latent": result["D_per_latent"].tolist(),
            "D":            float(result["D"]),
            "C_per_factor": result["C_per_factor"].tolist(),
            "C":            float(result["C"]),
            "I_per_factor": result["I_per_factor"].tolist(),
            "I":            float(result["I"]),
            "factor_names": list(result["factor_names"]),
        }
        with _lock:
            _state["dci_result"] = payload
            _state["dci_status"] = "done"
    except Exception as e:
        with _lock:
            _state["dci_status"] = "error"
            _state["dci_error"]  = str(e)


@app.route("/api/dci/start", methods=["POST"])
def api_dci_start():
    with _lock:
        if _state["dci_status"] == "computing":
            return jsonify({"status": "computing"})
        _state["dci_status"] = "computing"
        _state["dci_result"] = None
        _state["dci_error"]  = None
    threading.Thread(target=_run_dci, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/dci/status")
def api_dci_status():
    with _lock:
        resp = {"status": _state["dci_status"]}
        if _state["dci_result"]:
            resp["result"] = _state["dci_result"]
        if _state["dci_error"]:
            resp["error"] = _state["dci_error"]
    return jsonify(resp)


# ── Phase-3 endpoints ─────────────────────────────────────────────────────────

@app.route("/api/cached_encodings")
def api_cached_encodings():
    """Return cached encoder posterior μ + factor labels for the 2D scatter."""
    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
    if mu is None or factors is None:
        return jsonify({"ready": False})
    return jsonify({
        "ready": True,
        "mu":           mu.tolist(),
        "factors":      factors.tolist(),
        "factor_names": list(FACTOR_NAMES),
    })


@app.route("/api/cached_sample/<int:cache_idx>")
def api_cached_sample(cache_idx: int):
    """Return the original image, decoded reconstruction, and ground-truth
    factors for one cached encoding. Used by the 2D scatter hover/click panel.
    """
    with _lock:
        indices = _state["enc_indices"]
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
        decoder = _state["decoder"]
    if indices is None or mu is None or decoder is None:
        return jsonify({"error": "Cache not ready"}), 503
    if not (0 <= cache_idx < len(indices)):
        return jsonify({"error": f"cache_idx {cache_idx} out of range"}), 400

    dataset_idx = int(indices[cache_idx])
    original    = _dataset["imgs"][dataset_idx].astype(np.float32)

    z   = mu[cache_idx]
    z_t = torch.from_numpy(z[np.newaxis]).float().to(args.device)
    with torch.no_grad():
        recon = decoder(z_t).cpu().squeeze().numpy()

    factor_dict = {
        name: int(factors[cache_idx, i])
        for i, name in enumerate(FACTOR_NAMES)
    }
    return jsonify({
        "cache_idx":      cache_idx,
        "dataset_idx":    dataset_idx,
        "original":       _b64png(original),
        "reconstruction": _b64png(recon),
        "factors":        factor_dict,
    })


@app.route("/api/single_dim_traversal", methods=["POST"])
def api_single_dim_traversal():
    """Decode a 1-row latent traversal sweeping a single dim across ±range·σ."""
    if _state["decoder"] is None:
        return jsonify({"error": "No model loaded"}), 400

    data    = request.json or {}
    try:
        dim     = int(data.get("dim", -1))
        n_steps = int(data.get("n_steps", 9))
        rng_sig = float(data.get("range_sigma", 3.0))
        anchor_mu     = np.asarray(data.get("anchor_mu",     []), dtype=np.float32)
        anchor_logvar = np.asarray(data.get("anchor_logvar", []), dtype=np.float32)
    except Exception as e:
        return jsonify({"error": f"bad payload: {e}"}), 400

    if anchor_mu.size != _state["latent_dim"]:
        return jsonify({"error": f"anchor_mu length {anchor_mu.size} != ldim"}), 400
    if not (0 <= dim < _state["latent_dim"]):
        return jsonify({"error": f"dim {dim} out of range"}), 400

    sigma = float(np.exp(0.5 * np.clip(anchor_logvar[dim], -10, 10)))
    z_values = np.linspace(
        float(anchor_mu[dim]) - rng_sig * sigma,
        float(anchor_mu[dim]) + rng_sig * sigma,
        n_steps,
    ).astype(np.float32)

    decoder = _state["decoder"]
    device  = args.device
    images: list[str] = []
    with torch.no_grad():
        for v in z_values:
            z = anchor_mu.copy()
            z[dim] = float(v)
            z_t = torch.from_numpy(z[np.newaxis]).float().to(device)
            img = decoder(z_t).cpu().squeeze().numpy()
            images.append(_b64png(img))

    return jsonify({
        "images":   images,
        "z_values": z_values.tolist(),
        "dim":      dim,
        "anchor_mu_dim":     float(anchor_mu[dim]),
        "anchor_sigma_dim":  sigma,
    })


def _ensure_corr_matrix() -> np.ndarray | None:
    """Compute (and cache) the |Spearman ρ| matrix on cached encodings."""
    with _lock:
        cached = _state["corr_matrix"]
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
    if cached is not None:
        return cached
    if mu is None or factors is None:
        return None
    corr = factor_latent_correlation(mu, factors)
    with _lock:
        _state["corr_matrix"] = corr
    return corr


def _top_dim_for_factor(factor_name: str) -> tuple[int, str]:
    """Pick the latent dim that best encodes the named factor.

    Returns (dim, method) where method is "corr" for non-orientation factors and
    "mi" for orientation (since orientation is cyclic and rank correlation
    underestimates true alignment).
    """
    factor_idx = FACTOR_NAMES.index(factor_name)
    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
        cached  = _state["mig_top_dim_per_factor"].get(factor_name)
    if cached is not None:
        method = "mi" if factor_name == "orientation" else "corr"
        return int(cached), method
    if mu is None or factors is None:
        raise RuntimeError("Encoded samples not ready yet.")

    if factor_name == "orientation":
        mi = mutual_info_classif(mu, factors[:, factor_idx], discrete_features=False)
        dim = int(np.argmax(mi))
        method = "mi"
    else:
        corr = _ensure_corr_matrix()
        dim = int(np.argmax(np.abs(corr[:, factor_idx])))
        method = "corr"

    with _lock:
        _state["mig_top_dim_per_factor"][factor_name] = dim
    return dim, method


@app.route("/api/factor_conditional_histogram", methods=["POST"])
def api_factor_conditional_histogram():
    """Return μ samples grouped by factor value, for the dim that best encodes it."""
    data        = request.json or {}
    factor_name = str(data.get("factor", ""))
    if factor_name not in FACTOR_NAMES:
        return jsonify({"error": f"unknown factor: {factor_name}"}), 400
    if FACTOR_SIZES[FACTOR_NAMES.index(factor_name)] <= 1:
        return jsonify({"error": f"factor '{factor_name}' is constant"}), 400

    with _lock:
        mu      = _state["enc_mu"]
        factors = _state["enc_factors"]
    if mu is None or factors is None:
        return jsonify({"error": "Still computing — try again in a moment."}), 503

    try:
        dim, method = _top_dim_for_factor(factor_name)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    factor_idx = FACTOR_NAMES.index(factor_name)
    factor_col = factors[:, factor_idx]
    mu_col     = mu[:, dim]
    unique     = sorted(np.unique(factor_col).tolist())
    histograms = [
        {"value": int(v), "mu": mu_col[factor_col == v].tolist()}
        for v in unique
    ]
    return jsonify({
        "factor":           factor_name,
        "factor_values":    unique,
        "histograms":       histograms,
        "dim_used":         int(dim),
        "selection_method": method,
    })


# ── Phase-2b: Compare endpoint ────────────────────────────────────────────────
#
# The Compare tab can pin up to 4 checkpoints and view their metrics
# side-by-side. We compute KL/correlation/MIG/DCI for each pinned checkpoint
# *without* mutating the main `_state` (so the user's currently-loaded
# checkpoint and its cached encodings aren't disturbed).
#
# Each call loads the encoder, encodes 3000 random samples, and runs all
# four metrics. Result is cached by checkpoint path so repeat hits are free.

_compare_cache: dict[str, dict] = {}
_compare_lock = threading.Lock()


def _compute_compare_metrics(ckpt_path: str, n_samples: int = 3000) -> dict:
    """Encode a checkpoint and compute its summary metrics.

    Cached by path so subsequent calls are free.
    """
    with _compare_lock:
        if ckpt_path in _compare_cache:
            return _compare_cache[ckpt_path]

    enc, _dec = load_encoder_decoder(
        encoder_checkpoint=ckpt_path,
        decoder_checkpoint=ckpt_path,
        device=args.device,
    )
    enc.eval()
    rng = np.random.RandomState(0)
    idx = rng.choice(len(_dataset["imgs"]),
                     min(n_samples, len(_dataset["imgs"])),
                     replace=False)
    imgs    = _dataset["imgs"][idx].astype(np.float32)
    factors = _dataset["latents_classes"][idx]

    mu_list, lv_list = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), 512):
            x = torch.from_numpy(imgs[i:i+512][:, np.newaxis]).float().to(args.device)
            mu_b, lv_b = enc.encode(x)
            mu_list.append(mu_b.cpu().numpy())
            lv_list.append(lv_b.cpu().numpy())
    mu_arr = np.concatenate(mu_list)
    lv_arr = np.concatenate(lv_list)

    kl = kl_per_dim(mu_arr, lv_arr)
    corr = factor_latent_correlation(mu_arr, factors)
    try:
        mig_score, mig_per_factor = compute_mig(mu_arr, factors)
    except Exception:
        mig_score, mig_per_factor = float("nan"), {}
    try:
        dci_result = compute_dci(
            mu_arr, factors,
            factor_names=FACTOR_NAMES,
            factor_sizes=FACTOR_SIZES,
            n_estimators=30, seed=0,
        )
        dci = {
            "D": float(dci_result["D"]),
            "C": float(dci_result["C"]),
            "I": float(dci_result["I"]),
            "C_per_factor": dci_result["C_per_factor"].tolist(),
            "I_per_factor": dci_result["I_per_factor"].tolist(),
            "factor_names": list(dci_result["factor_names"]),
        }
    except Exception as e:
        dci = {"error": str(e)}

    metrics = {
        "checkpoint": ckpt_path,
        "latent_dim": int(enc.latent_dim),
        "kl_per_dim": kl.tolist(),
        "active_dims": int((kl > 0.1).sum()),
        "corr_max_per_factor": [float(np.abs(corr[:, k]).max())
                                for k in range(corr.shape[1])],
        "corr_top_dim_per_factor": [int(np.argmax(np.abs(corr[:, k])))
                                     for k in range(corr.shape[1])],
        "factor_names": list(FACTOR_NAMES),
        "mig":         float(mig_score),
        "mig_per_factor": mig_per_factor,
        "dci":         dci,
    }

    with _compare_lock:
        _compare_cache[ckpt_path] = metrics
    return metrics


@app.route("/api/compare/metrics", methods=["POST"])
def api_compare_metrics():
    """Compute (or return cached) summary metrics for a checkpoint.

    Body: {"checkpoint": "checkpoints/.../best.pt"}.
    Used by the Compare tab to pin checkpoints. May take 30-60s the first
    time per checkpoint, then instant.
    """
    data = request.json or {}
    p    = (data.get("checkpoint") or "").strip()
    if not p:
        return jsonify({"error": "No checkpoint path provided"}), 400
    if not Path(p).exists():
        return jsonify({"error": f"File not found: {p}"}), 404
    try:
        m = _compute_compare_metrics(p)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Match against EXPERIMENTS for split / β / γ provenance.
    params = _disent_param_for_checkpoint(p)
    m["beta"]            = params["beta"]
    m["gamma"]           = params["gamma"]
    m["split"]           = params["split"]
    m["corr_factor_a"]   = params["corr_factor_a"]
    m["corr_factor_b"]   = params["corr_factor_b"]
    m["corr_direction"]  = params["corr_direction"]
    return jsonify(m)


@app.route("/api/experiments")
def api_experiments():
    """Return the EXPERIMENTS table to the front-end (used by Compare picker)."""
    return jsonify({"experiments": EXPERIMENTS})


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*58}")
    print("  VAE Disentanglement Explorer")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*58}\n")
    app.run(debug=False, use_reloader=False, port=args.port)
