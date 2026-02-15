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
    *_ref_bc1.dds        (written by Dataset_Input_Extract_v2.py)
    ntbc_out_*.dds        (written by Inference_DDS_v2.py)

If your Train_dataset.json does NOT include meta (include_meta=False), then fill CONFIG lists manually.
"""

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Option A (recommended): auto-resolve everything from this directory
    #"material_dir": ,

    # Option B: manual lists (used only if auto-resolve fails / material_dir is empty)
    "source_images": [r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_Color.png", r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_NormalDX.png",r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_NormalGL.png" ],   # list[str]
    "ref_dds": [r"D:\BC1 extract\Bricks090_4K-PNG_model\Bricks090_4K-PNG_Color_ref_bc1.dds",r"D:\BC1 extract\Bricks090_4K-PNG_model\Bricks090_4K-PNG_NormalDX_ref_bc1.dds",r"D:\BC1 extract\Bricks090_4K-PNG_model\Bricks090_4K-PNG_NormalGL_ref_bc1.dds"],         # list[str] same length
    "test_dds": [r"D:\BC1 extract\Bricks090_4K-PNG_model\ntbc_out_Color.dds",r"D:\BC1 extract\Bricks090_4K-PNG_model\ntbc_out_NormalDX.dds",r"D:\BC1 extract\Bricks090_4K-PNG_model\ntbc_out_NormalGL.dds"],        # list[str] same length

    # Outputs
    "out_dir": r"metrics_out",
    "save_decoded_images": True,

    # Evaluate resolution caps
    "eval_max_side": 4096,     # PSNR
    "compute_ssim": True,
    "ssim_max_side": 4096,
}

# =============================================================================
# IMPLEMENTATION
# =============================================================================

import json
import math
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None


# -----------------------------
# Image I/O + resizing / padding
# -----------------------------
def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_rgb(path: str, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)

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
    top_mip_bytes = bw * bh * 8
    if len(data) < offset + top_mip_bytes:
        raise DDSDecodeError(f"{p}: truncated (need {top_mip_bytes} bytes for top mip)")

    blocks = memoryview(data)[offset:offset + top_mip_bytes]

    out = np.zeros((bh * 4, bw * 4, 3), dtype=np.uint8)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, :]

    for by in range(bh):
        row = blocks[by * bw * 8:(by + 1) * bw * 8]
        buf = np.frombuffer(row, dtype=np.uint8).reshape(bw, 8)

        c0 = (buf[:, 0].astype(np.uint16) | (buf[:, 1].astype(np.uint16) << 8))
        c1 = (buf[:, 2].astype(np.uint16) | (buf[:, 3].astype(np.uint16) << 8))

        idx = (buf[:, 4].astype(np.uint32)
               | (buf[:, 5].astype(np.uint32) << 8)
               | (buf[:, 6].astype(np.uint32) << 16)
               | (buf[:, 7].astype(np.uint32) << 24))

        rgb0 = _rgb565_to_rgb888(c0)
        rgb1 = _rgb565_to_rgb888(c1)

        pal = np.empty((bw, 4, 3), dtype=np.uint8)
        pal[:, 0, :] = rgb0
        pal[:, 1, :] = rgb1

        mode4 = c0 > c1
        c2 = np.empty((bw, 3), dtype=np.uint8)
        c3 = np.empty((bw, 3), dtype=np.uint8)

        if np.any(mode4):
            a = rgb0[mode4].astype(np.float32)
            b = rgb1[mode4].astype(np.float32)
            c2[mode4] = np.clip(np.rint((2.0 * a + b) / 3.0), 0, 255).astype(np.uint8)
            c3[mode4] = np.clip(np.rint((a + 2.0 * b) / 3.0), 0, 255).astype(np.uint8)

        if np.any(~mode4):
            a = rgb0[~mode4].astype(np.float32)
            b = rgb1[~mode4].astype(np.float32)
            c2[~mode4] = np.clip(np.rint((a + b) / 2.0), 0, 255).astype(np.uint8)
            c3[~mode4] = 0

        pal[:, 2, :] = c2
        pal[:, 3, :] = c3

        tex_idx = ((idx[:, None] >> shifts) & 3).astype(np.intp)

        colors = pal[np.arange(bw)[:, None], tex_idx].reshape(bw, 4, 4, 3)
        row_pixels = colors.transpose(1, 0, 2, 3).reshape(4, bw * 4, 3)
        out[by * 4:(by + 1) * 4, :, :] = row_pixels

    return out[:height, :width, :]


# -----------------------------
# Metrics
# -----------------------------
def psnr_chunked(a_u8: np.ndarray, b_u8: np.ndarray, chunk_rows: int = 256) -> float:
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"Shape mismatch: {a_u8.shape} vs {b_u8.shape}")
    mse = 0.0
    n = 0
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    h = a.shape[0]
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        diff = a[y0:y1] - b[y0:y1]
        mse += float(np.mean(diff * diff)) * (y1 - y0)
        n += (y1 - y0)
    mse /= max(1, n)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

def ssim_rgb(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    if ssim_fn is None:
        raise RuntimeError("scikit-image not installed (pip install scikit-image) for SSIM.")
    vals = []
    for c in range(3):
        v = ssim_fn(a_u8[:, :, c], b_u8[:, :, c], data_range=255)
        vals.append(float(v))
    return float(np.mean(vals))


# -----------------------------
# Auto resolve inputs
# -----------------------------
def auto_resolve(material_dir: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns:
      names, source_images, ref_dds, test_dds
    """
    names = []
    source_images = []
    ref_dds = []
    test_dds = []

    # Try read Train_dataset.json meta
    train_json = material_dir / "Train_dataset.json"
    if train_json.exists():
        d = json.loads(train_json.read_text())
        meta = d.get("meta", {}) or {}
        names = meta.get("texture_names") or []
        source_images = meta.get("source_images") or []

    # Read Inference_input.json for fallback names / count
    coords_json = material_dir / "Inference_input.json"
    if coords_json.exists():
        c = json.loads(coords_json.read_text())
        if not names:
            names = c.get("texture_names") or []
        if not names:
            nt = int(c.get("num_textures", 1))
            names = [f"tex{i:02d}" for i in range(nt)]

    if not names:
        names = ["tex00"]

    # Resolve ref_dds: prefer exact stems from source_images (extractor naming)
    if source_images:
        for s in source_images:
            stem = Path(s).stem
            cand = material_dir / f"{stem}_ref_bc1.dds"
            if cand.exists():
                ref_dds.append(str(cand))
            else:
                # fallback: any *_ref_bc1.dds that contains stem
                hits = sorted(material_dir.glob(f"*{stem}*_ref_bc1.dds"))
                ref_dds.append(str(hits[0]) if hits else "")
    else:
        # fallback: take N files in directory
        hits = sorted(material_dir.glob("*_ref_bc1.dds"))
        ref_dds = [str(h) for h in hits]

    # Resolve test_dds: default output naming from Inference_DDS_v2.py
    # base = ntbc_out.dds or any "ntbc_out*.dds"
    for n in names:
        safe = (n or "").replace(" ", "_")
        cand = material_dir / f"ntbc_out_{safe}.dds"
        if cand.exists():
            test_dds.append(str(cand))
        else:
            # fallback: try any file that contains name
            hits = sorted(material_dir.glob(f"*{safe}*.dds"))
            test_dds.append(str(hits[0]) if hits else "")

    # If source_images empty, try to find image files that match names
    if not source_images:
        for n in names:
            safe = (n or "").replace(" ", "_")
            hits = sorted(material_dir.glob(f"*{safe}*.png")) + sorted(material_dir.glob(f"*{safe}*.jpg"))
            source_images.append(str(hits[0]) if hits else "")

    # Ensure lengths
    N = len(names)
    def _pad(lst: List[str]) -> List[str]:
        return (lst + [""] * N)[:N]
    return names, _pad(source_images), _pad(ref_dds), _pad(test_dds)


def main():
    cfg = CONFIG
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    srcs: List[str] = []
    refs: List[str] = []
    tests: List[str] = []

    material_dir = str(cfg.get("material_dir") or "").strip()
    if material_dir:
        md = Path(material_dir).expanduser().resolve()
        if md.exists():
            names, srcs, refs, tests = auto_resolve(md)

    # Manual override if auto failed
    if not srcs or any(not s for s in srcs):
        if cfg["source_images"]:
            srcs = list(cfg["source_images"])
            names = names or [f"tex{i:02d}" for i in range(len(srcs))]
    if not refs or any(not r for r in refs):
        if cfg["ref_dds"]:
            refs = list(cfg["ref_dds"])
    if not tests or any(not t for t in tests):
        if cfg["test_dds"]:
            tests = list(cfg["test_dds"])

    if not (srcs and refs and tests) or len(srcs) != len(refs) or len(srcs) != len(tests):
        raise RuntimeError(
            "Could not resolve evaluation inputs.\n"
            f"Resolved: srcs={len(srcs)} refs={len(refs)} tests={len(tests)}\n"
            "Fix by setting CONFIG['material_dir'] correctly, or fill CONFIG lists manually."
        )

    # Evaluate each texture
    eval_max_side = cfg.get("eval_max_side", None)
    compute_ssim = bool(cfg.get("compute_ssim", False))
    ssim_max_side = cfg.get("ssim_max_side", None)
    save_dec = bool(cfg.get("save_decoded_images", False))

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

        psnr_ref = psnr_chunked(src_eval, ref_eval)
        psnr_test = psnr_chunked(src_eval, test_eval)

        ssim_ref = ssim_test = None
        if compute_ssim:
            if ssim_fn is None:
                print("[WARN] SSIM requested but scikit-image missing; skipping SSIM.")
                compute_ssim = False
            else:
                src_s = resize_max_side(src_padded, ssim_max_side)
                ref_s = resize_max_side(ref, ssim_max_side)
                test_s = resize_max_side(test, ssim_max_side)
                ssim_ref = ssim_rgb(src_s, ref_s)
                ssim_test = ssim_rgb(src_s, test_s)

        if save_dec:
            save_rgb(str(out_dir / f"{name}_source.png"), src_padded)
            save_rgb(str(out_dir / f"{name}_ref_decoded.png"), ref)
            save_rgb(str(out_dir / f"{name}_ntbc_decoded.png"), test)

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
    print("\n==================== NTBC EVAL (MULTI) ====================")
    for r in rows:
        if "ssim_ref" in r:
            print(
                f"{r['name']}:  "
                f"PSNR ref={r['psnr_ref']:.3f}  ntbc={r['psnr_ntbc']:.3f}  Δ={r['psnr_delta']:+.3f} | "
                f"SSIM ref={r['ssim_ref']:.4f} ntbc={r['ssim_ntbc']:.4f} Δ={r['ssim_delta']:+.4f}"
            )
        else:
            print(
                f"{r['name']}:  "
                f"PSNR ref={r['psnr_ref']:.3f}  ntbc={r['psnr_ntbc']:.3f}  Δ={r['psnr_delta']:+.3f}"
            )

    if rows:
        avg_psnr_delta = sum(r["psnr_delta"] for r in rows) / len(rows)
        print(f"\n[AVG] PSNR Δ: {avg_psnr_delta:+.3f} dB over {len(rows)} texture(s)")
        if rows and "ssim_delta" in rows[0]:
            avg_ssim_delta = sum(r["ssim_delta"] for r in rows) / len(rows)
            print(f"[AVG] SSIM Δ: {avg_ssim_delta:+.4f}")

    print(f"\nDecoded outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
