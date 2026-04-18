"""
VNTBC Texture Classifier

Scans a texture folder and classifies each PNG as BC1 (RGB) or BC4 (single-channel).
Uses filename keyword matching with a PIL image-mode fallback.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re


@dataclass
class TextureInfo:
    """Information about a detected texture file."""
    path: str           # Full absolute path to the PNG file
    filename: str       # Just the filename
    detected_type: str  # "BC1" or "BC4"
    display_name: str   # Human-readable name (e.g., "Color", "Normal", "Displacement")
    enabled: bool       # Whether to include in processing


def _classify_by_name(filename: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Classify texture type from filename using keyword matching.

    Returns (type, display_name, auto_exclude) or (None, None, False) if unknown.
    """
    stem = Path(filename).stem.lower()
    segments = set(re.split(r'[_\-]', stem))

    # --- Normal map variants (specific before generic) ---
    if "normalgl" in stem or ("nor" in segments and "gl" in segments):
        return "BC1", "NormalGL", True  # Auto-excluded
    if "normaldx" in stem or ("nor" in segments and "dx" in segments):
        return "BC1", "NormalDX", False

    # --- BC1 (RGB) patterns ---
    if "color" in segments or "colour" in segments:
        return "BC1", "Color", False
    if "diff" in segments or "diffuse" in segments:
        return "BC1", "Diffuse", False
    if "albedo" in segments or "basecolor" in segments:
        return "BC1", "BaseColor", False
    if "normal" in segments or "nor" in segments:
        return "BC1", "Normal", False

    # --- BC4 (grayscale) patterns ---
    if "ambientocclusion" in stem or "ao" in segments:
        return "BC4", "AO", False
    if "displacement" in stem or "disp" in segments:
        return "BC4", "Displacement", False
    if "height" in segments:
        return "BC4", "Height", False
    if "roughness" in stem or "rough" in segments:
        return "BC4", "Roughness", False
    if "metalness" in stem or "metallic" in stem:
        return "BC4", "Metalness", False
    if "specular" in stem or "spec" in segments:
        return "BC4", "Specular", False
    if "opacity" in segments:
        return "BC4", "Opacity", False
    if "alpha" in segments:
        return "BC4", "Alpha", False
    if "mask" in segments:
        return "BC4", "Mask", False

    # --- BC1 packed maps ---
    if "arm" in segments:
        return "BC1", "ARM", False

    return None, None, False


def _classify_by_image(path: str) -> str:
    """Fallback: classify by inspecting image mode/channels."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(path)
        if img.mode == "L":
            return "BC4"
        if img.mode in ("RGB", "RGBA"):
            arr = np.array(img.convert("RGB"))
            h, w = arr.shape[:2]
            cy, cx = h // 2, w // 2
            patch = arr[max(0, cy - 128):cy + 128, max(0, cx - 128):cx + 128]
            if (patch[:, :, 0] == patch[:, :, 1]).all() and \
               (patch[:, :, 1] == patch[:, :, 2]).all():
                return "BC4"
        return "BC1"
    except Exception:
        return "BC1"


def scan_texture_folder(folder_path: str) -> List[TextureInfo]:
    """Scan a folder for PNG textures and classify each as BC1 or BC4.

    Args:
        folder_path: Path to the texture folder.

    Returns:
        List of TextureInfo objects, one per PNG file found.

    Raises:
        ValueError: If the folder doesn't exist or contains no PNGs.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        raise ValueError(f"No PNG files found in: {folder}")

    results = []
    for png in pngs:
        tex_type, display_name, auto_exclude = _classify_by_name(png.name)

        if tex_type is None:
            tex_type = _classify_by_image(str(png))
            parts = re.split(r'[_\-]', png.stem)
            display_name = parts[-1] if parts else png.stem

        results.append(TextureInfo(
            path=str(png),
            filename=png.name,
            detected_type=tex_type,
            display_name=display_name,
            enabled=not auto_exclude,
        ))

    return results


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\BC1 extract\Data\MetalPlates013_4K-PNG"
    textures = scan_texture_folder(folder)
    for t in textures:
        status = "Y" if t.enabled else "N"
        print(f"  [{status}] {t.filename:45s} -> {t.detected_type}  ({t.display_name})")
