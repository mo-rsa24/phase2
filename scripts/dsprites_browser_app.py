#!/usr/bin/env python3
"""
Web-based dSprites browser with live slider controls.
Launch with: python scripts/dsprites_browser_app.py
Then open http://localhost:5000 in your browser
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from src.datasets.dsprites import load_dsprites, get_factor_names

app = Flask(__name__)

# Load dataset once at startup
print("Loading dSprites dataset...")
dataset = load_dsprites()
factor_names = get_factor_names()
latents_classes = dataset['latents_classes']
factor_maxes = {name: int(np.max(latents_classes[:, i]))
                for i, name in enumerate(factor_names)}
print(f"Dataset loaded: {len(dataset['imgs'])} images")


def get_image_base64(img_array):
    """Convert numpy image to base64 PNG for embedding in HTML."""
    # Convert grayscale to RGB
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
    pil_img = Image.fromarray((img_rgb * 255).astype(np.uint8))

    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


def get_image_for_factors(color, shape, scale, orientation, pos_x, pos_y):
    """Get image matching exact factor values."""
    latents = dataset['latents_classes']

    mask = (
        (latents[:, 0] == color) &
        (latents[:, 1] == shape) &
        (latents[:, 2] == scale) &
        (latents[:, 3] == orientation) &
        (latents[:, 4] == pos_x) &
        (latents[:, 5] == pos_y)
    )

    if mask.sum() == 0:
        return None, None

    img_idx = np.where(mask)[0][0]
    return dataset['imgs'][img_idx], img_idx


@app.route('/')
def index():
    return render_template('dsprites.html', factor_maxes=factor_maxes, factor_names=factor_names)


@app.route('/api/image', methods=['POST'])
def get_image():
    """API endpoint to get image for given factors."""
    data = request.json

    color = int(data.get('color', 0))
    shape = int(data.get('shape', 0))
    scale = int(data.get('scale', 0))
    orientation = int(data.get('orientation', 0))
    pos_x = int(data.get('pos_x', 0))
    pos_y = int(data.get('pos_y', 0))

    img, img_idx = get_image_for_factors(color, shape, scale, orientation, pos_x, pos_y)

    if img is None:
        return jsonify({'error': 'No image found for this factor combination'}), 404

    img_base64 = get_image_base64(img)

    return jsonify({
        'image': img_base64,
        'index': int(img_idx),
        'factors': {
            'color': color,
            'shape': shape,
            'scale': scale,
            'orientation': orientation,
            'pos_x': pos_x,
            'pos_y': pos_y
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("dSprites Browser starting...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, use_reloader=False)
