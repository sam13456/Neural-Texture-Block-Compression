"""
Absolute Difference Map Generator

Compares two images pixel-by-pixel and outputs a difference map:
  - Black pixels = identical
  - Brighter pixels = larger difference

Edit the CONFIG paths below, then run:
    python diff_map.py
"""

from pathlib import Path
import numpy as np
from PIL import Image


# ==================== CONFIG ====================

# Reference image (ground truth / Compressonator BC1 decode)
REFERENCE_IMAGE = r"D:\BC1 extract\VNTBC_Outputs\BC4\MetalPlates013_4K-PNG_model_vbc4_am\inference_output\MetalPlates013_4K-PNG_AmbientOcclusion_ref_bc4.png"

# Output image (NTBC inference preview)
OUTPUT_IMAGE = r"D:\BC1 extract\VNTBC_Outputs\BC4\MetalPlates013_4K-PNG_model_vbc4_am\inference_output\ntbc_out_preview_AO.png"

# Amplification factor — higher = subtle differences become more visible
AMPLIFY = 5.0

# Where to save the diff map (set to None for auto-naming next to output)
SAVE_PATH = None

# ================================================


def generate_diff_map(ref: np.ndarray, out: np.ndarray, amplify: float = 5.0) -> np.ndarray:
    """Compute amplified absolute difference between two RGB uint8 images."""
    diff = np.abs(ref.astype(np.float32) - out.astype(np.float32))
    return np.clip(diff * amplify, 0, 255).astype(np.uint8)


def main():
    ref_path = Path(REFERENCE_IMAGE)
    out_path = Path(OUTPUT_IMAGE)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not out_path.exists():
        raise FileNotFoundError(f"Output image not found: {out_path}")

    ref_img = np.array(Image.open(ref_path).convert("RGB"))
    out_img = np.array(Image.open(out_path).convert("RGB"))

    if ref_img.shape != out_img.shape:
        raise ValueError(f"Image dimensions don't match: ref={ref_img.shape} vs out={out_img.shape}")

    diff = generate_diff_map(ref_img, out_img, amplify=AMPLIFY)

    # Stats
    raw_diff = np.abs(ref_img.astype(np.float32) - out_img.astype(np.float32))
    print(f"Reference:  {ref_path.name} ({ref_img.shape[1]}x{ref_img.shape[0]})")
    print(f"Output:     {out_path.name}")
    print(f"Amplify:    {AMPLIFY}x")
    print(f"Mean error: {raw_diff.mean():.2f} / 255")
    print(f"Max error:  {raw_diff.max():.0f} / 255")
    print(f"Identical:  {(raw_diff.sum(axis=-1) == 0).mean() * 100:.1f}% of pixels")

    # Save
    if SAVE_PATH:
        save_path = Path(SAVE_PATH)
    else:
        save_path = out_path.parent / f"{out_path.stem}_diff_x{int(AMPLIFY)}{out_path.suffix}"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff).save(save_path)
    print(f"Saved:      {save_path}")


if __name__ == "__main__":
    main()
