#!/usr/bin/env python3
"""
ntbc_eval.py  [MULTI-SINGLE-CHANNEL-TEXTURE / BC4]

Paper-style evaluation for BC4:
- PSNR/SSIM against the *uncompressed source texture* (grayscale).
- Compare:
    Ref BC4  : source vs decode(ref_bc4.dds)
    Your NTBC: source vs decode(ntbc_out_*.dds)
    Delta    : NTBC - Ref
"""

# =============================================================================
# CONFIG
# =============================================================================
from config_BC4 import (
    SOURCE_IMAGES, REF_DDS, TEST_DDS,
)

CONFIG = {
    "source_images": SOURCE_IMAGES,
    "ref_dds": REF_DDS,
    "test_dds": TEST_DDS,

    "eval_max_side": 4096,
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
# Image I/O
# -----------------------------
def load_grayscale(path: str) -> np.ndarray:
    """Load image as grayscale uint8 (H,W)."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32)
        return (arr / 256.0).clip(0, 255).astype(np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


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
    mode = "L" if arr_u8.ndim == 2 else "RGB"
    img = Image.fromarray(arr_u8, mode=mode)
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)

def pad_to_shape_edge(arr_u8: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr_u8.shape[:2]
    if h == target_h and w == target_w:
        return arr_u8
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr_u8[:target_h, :target_w]
    if arr_u8.ndim == 2:
        return np.pad(arr_u8, ((0, pad_h), (0, pad_w)), mode="edge")
    return np.pad(arr_u8, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


# -----------------------------
# DDS / BC4 (ATI1) decode
# -----------------------------
class DDSDecodeError(RuntimeError):
    pass

def _u32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def _fourcc(b: bytes, off: int) -> str:
    return b[off:off+4].decode("ascii", errors="replace")

def decode_dds_bc4(path: str) -> np.ndarray:
    """Decode a BC4 (ATI1/BC4U) DDS file into a grayscale uint8 array (H,W)."""
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
        if dxgi_format not in (80, 81):  # BC4_UNORM, BC4_SNORM
            raise DDSDecodeError(f"{p}: DX10 format {dxgi_format} not BC4")
    else:
        if fourcc not in ("ATI1", "BC4U"):
            raise DDSDecodeError(f"{p}: FourCC {fourcc} not supported (need ATI1/BC4)")

    bw = (width + 3) // 4
    bh = (height + 3) // 4
    N = bw * bh
    top_mip_bytes = N * 8
    if len(data) < offset + top_mip_bytes:
        raise DDSDecodeError(f"{p}: truncated (need {top_mip_bytes} bytes for top mip)")

    buf = np.frombuffer(data, dtype=np.uint8, offset=offset, count=N * 8).reshape(N, 8)

    e0 = buf[:, 0].astype(np.float32)  # (N,)
    e1 = buf[:, 1].astype(np.float32)  # (N,)

    # Unpack 48-bit selectors -> 16 x 3-bit indices
    sel_bytes = buf[:, 2:8].astype(np.uint64)
    packed = np.zeros(N, dtype=np.uint64)
    for byte_i in range(6):
        packed |= sel_bytes[:, byte_i] << (byte_i * 8)

    indices = np.zeros((N, 16), dtype=np.uint8)
    for px in range(16):
        indices[:, px] = ((packed >> (px * 3)) & 7).astype(np.uint8)

    # Build 8-value palette (when e0 > e1)
    pal = np.zeros((N, 8), dtype=np.float32)
    pal[:, 0] = e0
    pal[:, 1] = e1
    # 8-value mode: e0 > e1
    mode8 = (e0 > e1)
    for k in range(1, 7):
        pal[:, k + 1] = np.where(mode8,
                                  ((7 - k) * e0 + k * e1) / 7.0,
                                  # 6-value mode (e0 <= e1): only 6 interpolated + 2 special
                                  # sel 2..5: interpolated, sel 6: 0, sel 7: 255
                                  0.0)
    # Handle 6-value mode separately
    mode6 = ~mode8
    if np.any(mode6):
        for k in range(1, 5):
            pal[mode6, k + 1] = ((5 - k) * e0[mode6] + k * e1[mode6]) / 5.0
        pal[mode6, 6] = 0.0
        pal[mode6, 7] = 255.0

    pal = np.clip(np.rint(pal), 0, 255).astype(np.uint8)

    # Gather: (N, 16)
    colors = pal[np.arange(N)[:, None], indices]

    # Reshape to image: (bh, bw, 4, 4) -> (bh*4, bw*4)
    colors = colors.reshape(bh, bw, 4, 4)
    out = colors.transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)

    return out[:height, :width]


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

def ssim_gray(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    return float(ssim_fn(a_u8, b_u8, data_range=255))


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

        src = load_grayscale(src_p)
        ref = decode_dds_bc4(ref_p)
        test = decode_dds_bc4(test_p)

        # Match shapes
        th, tw = ref.shape[0], ref.shape[1]
        src_padded = pad_to_shape_edge(src, th, tw)
        test = pad_to_shape_edge(test, th, tw)

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
            ssim_ref = ssim_gray(src_s, ref_s)
            ssim_test = ssim_gray(src_s, test_s)

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
    print("\n==================== NTBC BC4 EVAL ====================")
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
