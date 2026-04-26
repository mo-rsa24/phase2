#!/usr/bin/env python3
"""
Terminal-based dSprites browser with keyboard controls.
Shows individual dSprites images with factor values and saves to disk.

Usage:
    python scripts/browse_dsprites.py

Controls:
    - Use arrow keys or hjkl to navigate: color, shape, scale, orientation
    - Use +/- keys to adjust pos_x and pos_y
    - Press 's' to save current image to file
    - Press 'q' to quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from src.datasets.dsprites import load_dsprites, get_factor_names


def display_image_terminal(img: np.ndarray, height: int = 24):
    """Display a 64x64 grayscale image in terminal using ASCII/Unicode."""
    # Resize to terminal-friendly size
    img_small = Image.fromarray((img * 255).astype(np.uint8))
    img_resized = img_small.resize((height * 2, height), Image.Resampling.BILINEAR)
    arr = np.array(img_resized)

    # Convert to ASCII art
    chars = " .:-=+*#%@"
    normalized = arr / 255.0
    char_indices = (normalized * (len(chars) - 1)).astype(int)

    for row in char_indices:
        print("".join(chars[i] for i in row))


def get_image_for_factors(dataset, factors: dict):
    """Get image matching exact factor values."""
    latents = dataset['latents_classes']
    factor_names = get_factor_names()

    mask = np.ones(len(latents), dtype=bool)
    for name, value in factors.items():
        idx = factor_names.index(name)
        mask = mask & (latents[:, idx] == value)

    if mask.sum() == 0:
        return None

    img_idx = np.where(mask)[0][0]
    return dataset['imgs'][img_idx], img_idx


def print_status(factors: dict, factor_maxes: dict):
    """Print current factor values and max values."""
    print("\n" + "=" * 60)
    print("dSprites Factor Explorer")
    print("=" * 60)
    for name in get_factor_names():
        val = factors[name]
        max_val = factor_maxes[name]
        bar_width = 30
        filled = int((val / max_val) * bar_width) if max_val > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {name:12} {val:2d}/{max_val:2d}  [{bar}]")
    print("=" * 60)


def main():
    print("Loading dSprites dataset...")
    dataset = load_dsprites()
    factor_names = get_factor_names()

    # Get max values for each factor
    latents_classes = dataset['latents_classes']
    factor_maxes = {name: int(np.max(latents_classes[:, i]))
                    for i, name in enumerate(factor_names)}

    # Initialize factors
    factors = {name: 0 for name in factor_names}

    print(f"Dataset loaded: {len(dataset['imgs'])} images")
    print("Press 'h' for help\n")

    try:
        import readline  # Enable line editing in interactive mode
    except ImportError:
        pass

    while True:
        # Get image for current factors
        result = get_image_for_factors(dataset, factors)
        if result is None:
            print("No image found for this factor combination")
            continue

        img, img_idx = result

        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')

        print_status(factors, factor_maxes)
        print("\nImage preview:")
        print("(Use terminal zoom to see details better)\n")
        display_image_terminal(img, height=16)

        print("\nControls:")
        print("  color:       [a/d]  orientation: [j/k]")
        print("  shape:       [s/f]  pos_x:       [u/o]")
        print("  scale:       [e/r]  pos_y:       [i/p]")
        print("  [v] Save image  [q] Quit  [h] Help")

        try:
            cmd = input("\nCommand: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if cmd == 'q':
            print("Exiting...")
            break
        elif cmd == 'h':
            print(__doc__)
            input("Press Enter to continue...")
        elif cmd == 'v':
            # Save image to disk
            output_dir = "browsed_images"
            os.makedirs(output_dir, exist_ok=True)

            factor_str = "_".join(f"{name}={factors[name]}" for name in factor_names)
            filename = f"{output_dir}/dsprites_{factor_str}.png"

            # Convert grayscale to RGB for better viewing
            img_rgb = np.stack([img, img, img], axis=-1)
            pil_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
            pil_img.save(filename)
            print(f"✓ Saved to {filename}")
            input("Press Enter to continue...")
        elif cmd in ['a', 'd']:  # color
            factors['color'] = max(0, min(factor_maxes['color'],
                                         factors['color'] + (1 if cmd == 'd' else -1)))
        elif cmd in ['s', 'f']:  # shape
            factors['shape'] = max(0, min(factor_maxes['shape'],
                                         factors['shape'] + (1 if cmd == 'f' else -1)))
        elif cmd in ['e', 'r']:  # scale
            factors['scale'] = max(0, min(factor_maxes['scale'],
                                         factors['scale'] + (1 if cmd == 'r' else -1)))
        elif cmd in ['j', 'k']:  # orientation
            factors['orientation'] = max(0, min(factor_maxes['orientation'],
                                               factors['orientation'] + (1 if cmd == 'k' else -1)))
        elif cmd in ['u', 'o']:  # pos_x
            factors['pos_x'] = max(0, min(factor_maxes['pos_x'],
                                         factors['pos_x'] + (1 if cmd == 'o' else -1)))
        elif cmd in ['i', 'p']:  # pos_y
            factors['pos_y'] = max(0, min(factor_maxes['pos_y'],
                                         factors['pos_y'] + (1 if cmd == 'p' else -1)))


if __name__ == "__main__":
    main()
