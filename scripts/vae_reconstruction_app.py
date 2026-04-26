#!/usr/bin/env python3
"""
VAE Reconstruction Browser.

Launch:
  python scripts/vae_reconstruction_app.py --checkpoint checkpoints/vae/best.pt

Then open http://localhost:5001 in your browser.
Use the factor sliders to pick any dSprite and watch the VAE encode + reconstruct it live.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import io
import base64

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from src.datasets.dsprites import (
    FACTOR_NAMES,
    FACTOR_SIZES,
    load_dsprites,
)
from src.utils.vae_inspection import (
    build_factor_index,
    load_encoder_decoder,
    run_vae_pass,
    to_uint8_rgb,
)


def parse_args():
    p = argparse.ArgumentParser(description="VAE Reconstruction Browser")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt checkpoint (encoder_state_dict key required)")
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--data-dir",   type=str, default="data")
    p.add_argument("--port",       type=int, default=5001)
    return p.parse_args()


# ---------------------------------------------------------------------------
# App-level startup (module scope so Flask sees it before first request)
# ---------------------------------------------------------------------------
args = parse_args()

app = Flask(__name__, template_folder="templates")

print("Loading dSprites dataset...")
_dataset       = load_dsprites(args.data_dir)
_factor_names  = list(FACTOR_NAMES)
_factor_maxes  = {name: int(FACTOR_SIZES[i] - 1) for i, name in enumerate(_factor_names)}

print("Building factor index (one-time O(n) cost)...")
_factor_index  = build_factor_index(_dataset)

print(f"Loading VAE (checkpoint={args.checkpoint or 'random init'})...")
_encoder, _decoder = load_encoder_decoder(
    latent_dim=args.latent_dim,
    img_channels=1,
    encoder_checkpoint=args.checkpoint,
    decoder_checkpoint=args.checkpoint,
    device=args.device,
)
_latent_dim = int(_encoder.latent_dim)
print(f"Ready. latent_dim={_latent_dim}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _array_to_b64_png(img: np.ndarray) -> str:
    """Float [0,1] (H,W) array → base64 PNG data-URI."""
    rgb = to_uint8_rgb(img)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _diff_image(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    diff = np.abs(orig.astype(np.float32) - recon.astype(np.float32))
    if diff.max() > 0:
        diff /= diff.max()
    return diff


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "vae_reconstruction.html",
        factor_names=_factor_names,
        factor_maxes=_factor_maxes,
        latent_dim=_latent_dim,
    )


@app.route("/api/reconstruct", methods=["POST"])
def reconstruct():
    data          = request.json
    factor_values = {name: int(data.get(name, 0)) for name in _factor_names}

    try:
        result = run_vae_pass(
            _dataset,
            _encoder,
            _decoder,
            factor_values=factor_values,
            use_mean_latent=True,
            factor_index=_factor_index,
        )
    except (KeyError, IndexError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404

    # result.reconstruction is (H, W) or (H, W, 1) after tensor_to_display_image
    recon = result.reconstruction
    if recon.ndim == 3 and recon.shape[-1] == 1:
        recon = recon[..., 0]

    diff = _diff_image(result.input_image, recon)

    return jsonify({
        "original":       _array_to_b64_png(result.input_image),
        "reconstruction": _array_to_b64_png(recon),
        "difference":     _array_to_b64_png(diff),
        "mu":             result.mu.tolist(),
        "logvar":         result.logvar.tolist(),
        "mse":            result.mse,
        "sample_index":   result.sample_index,
        "factors":        result.factors,
    })


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("VAE Reconstruction Browser starting...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("="*60 + "\n")
    app.run(debug=False, use_reloader=False, port=args.port)
