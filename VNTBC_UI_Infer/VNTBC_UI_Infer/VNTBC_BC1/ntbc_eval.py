#!/usr/bin/env python3
"""
ntbc_eval.py  [MULTI-RGB-TEXTURE]

Paper-style evaluation:
- PSNR/SSIM against the *uncompressed source texture*.
- Compare:
    Ref BC1  : source vs decode(ref_bc1.dds)
    Your NTBC: source vs decode(ntbc_out_*.dds)
    Delta    : NTBC - Ref

Multi-texture:
- If you trained on multiple RGB textures for a material, evaluate each one separately and print a summary.

Recommended setup (from your pipeline):
- material_dir contains:
    Inference_input.json
    Train_dataset.json  (with meta.source_images + meta.texture_names)
    *_ref_bc1.dds        (written by Dataset_Input_Extract.py)
    ntbc_out_*.dds        (written by Inference_DDS.py)

"""

# =============================================================================
# CONFIG (paths from config.py)
# =============================================================================
from config import (
    SOURCE_IMAGES, REF_DDS, TEST_DDS,
)

CONFIG = {
    "source_images": SOURCE_IMAGES,
    "ref_dds": REF_DDS,
    "test_dds": TEST_DDS,

    # Evaluate resolution caps
    "eval_max_side": 4096,     # PSNR
    "compute_ssim": True,
    "ssim_max_side": 4096,
}

# =============================================================================
# IMPLEMENTATION
# =============================================================================

import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim_fn


# -----------------------------
# Image I/O + resizing / padding
# -----------------------------
def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_max_side(arr_u8: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if max_side is None:
        return arr_u8
    h, w = arr_u8.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return arr_u8
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr_u8, mode="RGB")
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)

def pad_to_shape_edge(arr_u8: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr_u8.shape[:2]
    if h == target_h and w == target_w:
        return arr_u8
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        # if arr is larger (shouldn't happen), crop
        return arr_u8[:target_h, :target_w, :]
    return np.pad(arr_u8, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


# -----------------------------
# DDS / BC1 (DXT1) decode
# -----------------------------
class DDSDecodeError(RuntimeError):
    pass

def _u32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def _fourcc(b: bytes, off: int) -> str:
    return b[off:off+4].decode("ascii", errors="replace")

def _rgb565_to_rgb888(c_u16: np.ndarray) -> np.ndarray:
    c = c_u16.astype(np.uint16)
    r = ((c >> 11) & 31).astype(np.float32) * (255.0 / 31.0)
    g = ((c >> 5) & 63).astype(np.float32) * (255.0 / 63.0)
    b = (c & 31).astype(np.float32) * (255.0 / 31.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

def decode_dds_bc1(path: str) -> np.ndarray:
    """Decode a BC1 (DXT1) DDS file — fully vectorized, no Python loops."""
    p = Path(path)
    data = p.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise DDSDecodeError(f"{p}: not a valid DDS")

    header = data[4:4+124]
    if _u32_le(header, 0) != 124:
        raise DDSDecodeError(f"{p}: unexpected DDS header size")

    height = _u32_le(header, 8)
    width  = _u32_le(header, 12)

    pf_off = 72
    fourcc = _fourcc(header, pf_off + 8)

    offset = 4 + 124
    if fourcc == "DX10":
        if len(data) < offset + 20:
            raise DDSDecodeError(f"{p}: truncated DX10 header")
        dxgi_format = struct.unpack_from("<I", data, offset)[0]
        offset += 20
        if dxgi_format not in (71, 72):
            raise DDSDecodeError(f"{p}: DX10 format {dxgi_format} not BC1")
    else:
        if fourcc != "DXT1":
            raise DDSDecodeError(f"{p}: FourCC {fourcc} not supported (need DXT1/BC1)")

    bw = (width + 3) // 4
    bh = (height + 3) // 4
    N = bw * bh
    top_mip_bytes = N * 8
    if len(data) < offset + top_mip_bytes:
        raise DDSDecodeError(f"{p}: truncated (need {top_mip_bytes} bytes for top mip)")

    # Read all blocks at once: (N, 8) uint8
    buf = np.frombuffer(data, dtype=np.uint8, offset=offset, count=N * 8).reshape(N, 8)

    # Extract c0, c1 as uint16 and index bits as uint32
    c0 = buf[:, 0].astype(np.uint16) | (buf[:, 1].astype(np.uint16) << 8)
    c1 = buf[:, 2].astype(np.uint16) | (buf[:, 3].astype(np.uint16) << 8)
    idx = (buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8)
           | (buf[:, 6].astype(np.uint32) << 16) | (buf[:, 7].astype(np.uint32) << 24))

    # Decode endpoints to RGB888 float for interpolation
    rgb0 = _rgb565_to_rgb888(c0).astype(np.float32)  # (N, 3)
    rgb1 = _rgb565_to_rgb888(c1).astype(np.float32)  # (N, 3)

    # Build palette: (N, 4, 3) float32
    mode4 = (c0 > c1)[:, None]  # (N, 1) for broadcasting
    pal = np.empty((N, 4, 3), dtype=np.float32)
    pal[:, 0] = rgb0
    pal[:, 1] = rgb1
    # 4-color mode: c2 = (2*c0+c1)/3, c3 = (c0+2*c1)/3
    # 3-color mode: c2 = (c0+c1)/2,   c3 = 0
    pal[:, 2] = np.where(mode4, (2.0 * rgb0 + rgb1) / 3.0, (rgb0 + rgb1) / 2.0)
    pal[:, 3] = np.where(mode4, (rgb0 + 2.0 * rgb1) / 3.0, 0.0)

    pal = np.clip(np.rint(pal), 0, 255).astype(np.uint8)

    # Unpack 16 2-bit indices per block: (N, 16)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, :]
    tex_idx = ((idx[:, None] >> shifts) & 3).astype(np.intp)  # (N, 16)

    # Gather colors: (N, 16, 3)
    colors = pal[np.arange(N)[:, None], tex_idx]

    # Reshape from block-linear to image: (N,16,3) -> (bh,bw,4,4,3) -> (bh*4, bw*4, 3)
    colors = colors.reshape(bh, bw, 4, 4, 3)
    out = colors.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)

    return out[:height, :width, :]


# -----------------------------
# Metrics
# -----------------------------
def psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"Shape mismatch: {a_u8.shape} vs {b_u8.shape}")
    diff = a_u8.astype(np.float32) - b_u8.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)

def ssim_rgb(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    return float(ssim_fn(a_u8, b_u8, data_range=255, channel_axis=2))

def main():
    cfg = CONFIG

    srcs = list(cfg["source_images"])
    refs = list(cfg["ref_dds"])
    tests = list(cfg["test_dds"])
    names = [Path(s).stem for s in srcs]

    if not (srcs and refs and tests) or len(srcs) != len(refs) or len(srcs) != len(tests):
        raise RuntimeError(
            "Evaluation inputs mismatch.\n"
            f"Resolved: srcs={len(srcs)} refs={len(refs)} tests={len(tests)}\n"
            "Fix paths in config.py."
        )

    eval_max_side = cfg.get("eval_max_side", None)
    compute_ssim = bool(cfg.get("compute_ssim", False))
    ssim_max_side = cfg.get("ssim_max_side", None)

    rows = []
    for i, (name, src_p, ref_p, test_p) in enumerate(zip(names, srcs, refs, tests)):
        if not (src_p and ref_p and test_p):
            print(f"[SKIP] {name}: missing path(s)")
            continue

        src = load_rgb(src_p)
        ref = decode_dds_bc1(ref_p)
        test = decode_dds_bc1(test_p)

        # Match shapes via edge-padding on source (DDS may be padded to multiple of 4)
        th, tw = ref.shape[0], ref.shape[1]
        src_padded = pad_to_shape_edge(src, th, tw)
        test = pad_to_shape_edge(test, th, tw)  # should already match, but safe

        src_eval = resize_max_side(src_padded, eval_max_side)
        ref_eval = resize_max_side(ref, eval_max_side)
        test_eval = resize_max_side(test, eval_max_side)

        psnr_ref = psnr(src_eval, ref_eval)
        psnr_test = psnr(src_eval, test_eval)

        ssim_ref = ssim_test = None
        if compute_ssim:
            src_s = resize_max_side(src_padded, ssim_max_side)
            ref_s = resize_max_side(ref, ssim_max_side)
            test_s = resize_max_side(test, ssim_max_side)
            ssim_ref = ssim_rgb(src_s, ref_s)
            ssim_test = ssim_rgb(src_s, test_s)

        row = {
            "name": name,
            "psnr_ref": psnr_ref,
            "psnr_ntbc": psnr_test,
            "psnr_delta": psnr_test - psnr_ref,
        }
        if compute_ssim:
            row.update({
                "ssim_ref": float(ssim_ref),
                "ssim_ntbc": float(ssim_test),
                "ssim_delta": float(ssim_test - ssim_ref),
            })
        rows.append(row)

    # Print report
    print("\n==================== NTBC EVAL ====================")
    for r in rows:
        if "ssim_ref" in r:
            print(
                f"{r['name']}:  "
                f"PSNR ref={r['psnr_ref']:.3f}  ntbc={r['psnr_ntbc']:.3f}  Delta={r['psnr_delta']:+.3f} | "
                f"SSIM ref={r['ssim_ref']:.4f} ntbc={r['ssim_ntbc']:.4f} Delta={r['ssim_delta']:+.4f}"
            )
        else:
            print(
                f"{r['name']}:  "
                f"PSNR ref={r['psnr_ref']:.3f}  ntbc={r['psnr_ntbc']:.3f}  Delta={r['psnr_delta']:+.3f}"
            )

    if rows:
        avg_psnr_delta = sum(r["psnr_delta"] for r in rows) / len(rows)
        print(f"\n[AVG] PSNR Delta: {avg_psnr_delta:+.3f} dB over {len(rows)} texture(s)")
        if rows and "ssim_delta" in rows[0]:
            avg_ssim_delta = sum(r["ssim_delta"] for r in rows) / len(rows)
            print(f"[AVG] SSIM Delta: {avg_ssim_delta:+.4f}")




if __name__ == "__main__":
    main()
