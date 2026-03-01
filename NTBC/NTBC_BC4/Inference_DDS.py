"""
NTBC BC4 FULL INFERENCE -> DDS (ATI1)  [MULTI-SINGLE-CHANNEL-TEXTURE]

Loads a merged compressed state dict and writes one BC4 DDS per texture.

BC4 block layout (8 bytes per 4x4 block):
  - Byte 0: endpoint e0 (uint8)
  - Byte 1: endpoint e1 (uint8)
  - Bytes 2-7: 16 × 3-bit selectors (48 bits, little-endian packed)

BC4 interpolation (when e0 > e1, 8-value mode):
  palette[0] = e0
  palette[1] = e1
  palette[2..7] = linearly interpolated (6e0+1e1)/7 .. (1e0+6e1)/7
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch

from Network_endpoint import EndpointNetwork, bc4_palette_bc4order
from Network_color import ColorNetwork

from Model_param_compress import decompress_state_dict


# =========================
# CONFIG
# =========================
from config import (
    INFERENCE_INPUT_JSON, MERGED_CHECKPOINT, OUT_DDS, OUT_PREVIEW_PNG,
    REF_DDS, INFERENCE_OUTPUT_DIR,
)

CONFIG = {
    "coords_json": INFERENCE_INPUT_JSON,
    "merged_ckpt": MERGED_CHECKPOINT,
    "out_dds": OUT_DDS,
    "save_preview_png": True,
    "out_preview_png": OUT_PREVIEW_PNG,
    "ref_dds": REF_DDS,
    "inference_output_dir": INFERENCE_OUTPUT_DIR,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "block_batch": 65536,
}


# =========================
# DDS (ATI1/BC4) writer
# =========================
def dds_header_bc4(width: int, height: int) -> bytes:
    """Build a DDS header with ATI1 FourCC for BC4."""
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000

    DDPF_FOURCC = 0x4
    DDSCAPS_TEXTURE = 0x1000

    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    linear_size = blocks_x * blocks_y * 8  # 8 bytes per BC4 block

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = linear_size
    dwDepth = 0
    dwMipMapCount = 0
    dwReserved1 = [0] * 11

    pfSize = 32
    pfFlags = DDPF_FOURCC
    pfFourCC = b"ATI1"   # BC4 FourCC
    pfRGBBitCount = 0
    pfRBitMask = 0
    pfGBitMask = 0
    pfBBitMask = 0
    pfABitMask = 0

    dwCaps = DDSCAPS_TEXTURE
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I",
        dwSize, dwFlags, dwHeight, dwWidth,
        dwPitchOrLinearSize, dwDepth, dwMipMapCount,
        *dwReserved1
    )

    pixel_format = struct.pack(
        "<I I 4s I I I I I",
        pfSize, pfFlags, pfFourCC,
        pfRGBBitCount, pfRBitMask, pfGBitMask, pfBBitMask, pfABitMask
    )

    caps = struct.pack("<I I I I I", dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2)
    return b"DDS " + header + pixel_format + caps


def write_dds_bc4(path: Path, width: int, height: int, bc4_blocks: bytes) -> None:
    hdr = dds_header_bc4(width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + bc4_blocks)


# =========================
# Model loading
# =========================
def split_merged_state_dict(merged: dict) -> tuple:
    endpoint_sd = {}
    color_sd = {}
    for k, v in merged.items():
        if k.startswith("endpoint."):
            endpoint_sd[k[len("endpoint."):]] = v
        elif k.startswith("color."):
            color_sd[k[len("color."):]] = v
        else:
            print(f"[WARN] Unknown key prefix in merged file: {k}")
    return endpoint_sd, color_sd


def _infer_grid_params_from_state(state: Dict[str, torch.Tensor], prefix: str = "encoding.grids."):
    grid_keys = [k for k in state.keys() if k.startswith(prefix)]
    if not grid_keys:
        raise ValueError("No grid keys found in state_dict.")

    def idx_of(k: str) -> int:
        try:
            return int(k.split(".")[2])
        except Exception:
            return -1

    grid_keys = sorted(grid_keys, key=idx_of)
    num_levels = len(grid_keys)

    g0 = state[grid_keys[0]]
    glast = state[grid_keys[-1]]
    base_res = int(round(math.sqrt(g0.shape[0])))
    finest_res = int(round(math.sqrt(glast.shape[0])))
    feature_dim = int(g0.shape[1])
    dtype = g0.dtype
    return num_levels, base_res, finest_res, feature_dim, dtype


def _infer_out_dim_from_mlp(state: Dict[str, torch.Tensor]) -> int:
    best_i = -1
    best_out = None
    for k, v in state.items():
        if k.startswith("mlp.") and k.endswith(".weight") and v.ndim == 2:
            try:
                i = int(k.split(".")[1])
            except Exception:
                continue
            if i > best_i:
                best_i = i
                best_out = int(v.shape[0])
    if best_out is None:
        raise ValueError("Could not infer MLP output dim from state_dict keys.")
    return best_out


# =========================
# BC4 helpers
# =========================

_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
_OFF_Y = torch.tensor([0, 0, 0, 0,
                       1, 1, 1, 1,
                       2, 2, 2, 2,
                       3, 3, 3, 3], dtype=torch.long)


def pack_bc4_indices_48bit(indices_16: torch.Tensor) -> np.ndarray:
    """Pack 16 x 3-bit BC4 selectors into 6 bytes (48 bits) per block.

    indices_16: (B,16) uint8 values in [0..7]
    Returns: (B,6) uint8 numpy array

    BC4 packing: bits are packed LSB-first into 6 consecutive bytes.
    """
    idx = indices_16.to(torch.int64).cpu().numpy()  # (B,16)
    B = idx.shape[0]

    # Pack 16 x 3-bit values into a 48-bit integer
    packed = np.zeros(B, dtype=np.uint64)
    for i in range(16):
        packed |= (idx[:, i].astype(np.uint64) & 7) << (i * 3)

    # Split into 6 bytes
    out = np.zeros((B, 6), dtype=np.uint8)
    for byte_i in range(6):
        out[:, byte_i] = ((packed >> (byte_i * 8)) & 0xFF).astype(np.uint8)

    return out


# =========================
# Output paths
# =========================

def _make_output_paths(base_dds: Path, base_png: Optional[Path], names: List[str]) -> Tuple[List[Path], List[Optional[Path]]]:
    if len(names) == 1:
        return [base_dds], [base_png]
    base_stem = base_dds.stem
    base_dir = base_dds.parent
    base_suf = base_dds.suffix
    out_dds = []
    out_png = []
    for i, n in enumerate(names):
        safe = (n or f"tex{i:02d}").replace(" ", "_")
        out_dds.append(base_dir / f"{base_stem}_{safe}{base_suf}")
        if base_png is None:
            out_png.append(None)
        else:
            out_png.append(base_png.parent / f"{base_png.stem}_{safe}{base_png.suffix}")
    return out_dds, out_png


# =========================
# Main inference
# =========================
@dataclass
class InferOutputs:
    width: int
    height: int
    blocks_x: int
    blocks_y: int
    out_dds_paths: List[Path]
    out_png_paths: List[Path]


@torch.no_grad()
def infer_ntbc_bc4_to_dds_multi(
    coords_json: Path,
    merged_ckpt: Path,
    out_dds: Path,
    device: str = "cuda",
    use_amp: bool = True,
    block_batch: int = 4096,
    save_preview_png: bool = True,
    out_preview_png: Path | None = None,
) -> InferOutputs:
    coords = json.loads(coords_json.read_text())
    blocks_x = int(coords["blocks_x"])
    blocks_y = int(coords["blocks_y"])
    W = blocks_x * 4
    H = blocks_y * 4

    names = coords.get("texture_names") or []
    T_json = coords.get("num_textures", None)
    if not names and T_json is not None:
        names = [f"tex{i:02d}" for i in range(int(T_json))]
    if not names:
        names = ["tex00"]

    print(f"[INFO] Texture size: {W}x{H} blocks=({blocks_x},{blocks_y})")

    # Load merged checkpoint
    merged = torch.load(merged_ckpt, map_location="cpu")
    ep_comp, col_comp = split_merged_state_dict(merged)

    ep_state = decompress_state_dict(ep_comp)
    col_state = decompress_state_dict(col_comp)

    # Infer T
    ep_out = _infer_out_dim_from_mlp(ep_state)   # 2*T for BC4
    col_out = _infer_out_dim_from_mlp(col_state)  # 1*T for BC4
    if ep_out % 2 != 0:
        raise ValueError(f"Endpoint MLP out_dim={ep_out} not divisible by 2.")
    T_ep = ep_out // 2
    T_col = col_out  # 1 per texture
    if T_ep != T_col:
        raise ValueError(f"Mismatch: endpoint infers T={T_ep}, color infers T={T_col}.")
    T = T_ep

    if len(names) != T:
        print(f"[WARN] texture_names length ({len(names)}) != model T ({T}). Using model T.")
        names = [names[i] if i < len(names) else f"tex{i:02d}" for i in range(T)]

    print(f"[INFO] Multi-texture BC4 inference: T={T} -> writing {T} DDS files")

    # Instantiate models
    nl, br, fr, fd, dt = _infer_grid_params_from_state(ep_state, prefix="encoding.grids.")
    param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32
    ep_net = EndpointNetwork(
        num_textures=T,
        num_levels=nl, base_resolution=br, finest_resolution=fr,
        feature_dim=fd, hidden_dim=64, num_hidden_layers=3,
        param_dtype=param_dtype,
    ).to(device)
    ep_net.load_state_dict(ep_state, strict=True)
    ep_net.eval()

    nl, br, fr, fd, dt = _infer_grid_params_from_state(col_state, prefix="encoding.grids.")
    param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32
    col_net = ColorNetwork(
        num_textures=T,
        param_dtype=param_dtype, finest_resolution=fr,
        base_resolution=br, num_levels=nl,
    ).to(device)
    col_net.load_state_dict(col_state, strict=True)
    col_net.eval()

    amp_enabled = bool(use_amp) and (device == "cuda")
    autocast_device = "cuda" if device == "cuda" else "cpu"

    # All blocks in row-major order
    bx = torch.arange(blocks_x, dtype=torch.int64)
    by = torch.arange(blocks_y, dtype=torch.int64)
    grid_by, grid_bx = torch.meshgrid(by, bx, indexing="ij")
    bxby = torch.stack([grid_bx.reshape(-1), grid_by.reshape(-1)], dim=1)
    N = int(bxby.shape[0])

    denom_x = float(max(1, blocks_x - 1))
    denom_y = float(max(1, blocks_y - 1))
    st = torch.stack([bxby[:, 0].to(torch.float32) / denom_x,
                      bxby[:, 1].to(torch.float32) / denom_y], dim=1)

    # Output buffers: 8 bytes per block per texture
    bc4_bytes = [bytearray(N * 8) for _ in range(T)]

    previews = None
    if save_preview_png:
        previews = [np.zeros((H, W), dtype=np.uint8) for _ in range(T)]

    base_dds = out_dds.expanduser().resolve()
    base_png = out_preview_png.expanduser().resolve() if (save_preview_png and out_preview_png is not None) else None
    out_dds_paths, out_png_paths = _make_output_paths(base_dds, base_png, names)

    for start in range(0, N, block_batch):
        end = min(N, start + block_batch)
        b = end - start

        bxby_b = bxby[start:end]
        st_b = st[start:end].to(device=device, non_blocking=True)

        # Endpoints: (b, 2*T) -> (b, T, 2)
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            ep_pred_flat = ep_net(st_b).to(torch.float32)
        ep_pred = ep_pred_flat.view(b, T, 2)

        # Quantize endpoints to uint8
        e0_f = ep_pred[:, :, 0]  # (b,T) in [0,1]
        e1_f = ep_pred[:, :, 1]  # (b,T) in [0,1]
        e0_u8 = (e0_f * 255.0).round().clamp(0, 255).to(torch.uint8)
        e1_u8 = (e1_f * 255.0).round().clamp(0, 255).to(torch.uint8)

        # Recalculate quantized endpoint values in [0,1] for palette building
        e0_q01 = e0_u8.to(torch.float32) / 255.0  # (b,T)
        e1_q01 = e1_u8.to(torch.float32) / 255.0

        # Pixel coords inside each block
        base_x = (bxby_b[:, 0] * 4).view(-1, 1)
        base_y = (bxby_b[:, 1] * 4).view(-1, 1)
        x = (base_x + _OFF_X.view(1, 16)).to(torch.int64)
        y = (base_y + _OFF_Y.view(1, 16)).to(torch.int64)

        u = (x.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (y.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=-1).reshape(-1, 2)  # (b*16, 2)

        # Predict colors: (b*16, T) -> (b, 16, T) -> (b, T, 16)
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            pred_flat = col_net(uv).to(torch.float32)
        pred = pred_flat.view(b, 16, T).permute(0, 2, 1).contiguous()  # (b, T, 16)
        # Build BC4 palette in BC4 selector order (handles both BC4 modes)
        endpoints_q01 = torch.stack([e0_q01, e1_q01], dim=-1).to(device=device, non_blocking=True)  # (b,T,2)
        pal = bc4_palette_bc4order(endpoints_q01)  # (b,T,8)

        # Nearest palette selector per texel
        diff = pred.unsqueeze(3) - pal.unsqueeze(2)  # (b,T,16,8)
        d2 = diff * diff
        idx_bc4 = torch.argmin(d2, dim=-1).to(torch.uint8)  # (b,T,16) in BC4 selector order

        # Pack and write per texture
        e0_cpu = e0_u8.cpu().numpy()
        e1_cpu = e1_u8.cpu().numpy()

        for t in range(T):
            packed_sel = pack_bc4_indices_48bit(idx_bc4[:, t, :])  # (b, 6) uint8
            bb = bc4_bytes[t]
            for i in range(b):
                off = (start + i) * 8
                bb[off] = int(e0_cpu[i, t])
                bb[off + 1] = int(e1_cpu[i, t])
                bb[off + 2:off + 8] = packed_sel[i].tobytes()
        # Preview: grayscale (decode using the same BC4 palette/indices)
        if previews is not None:
            pal_cpu = pal.detach().cpu()  # (b,T,8)
            idx_cpu = idx_bc4.detach().cpu().to(torch.int64)  # (b,T,16)
            for t in range(T):
                decoded = torch.gather(pal_cpu[:, t, :], 1, idx_cpu[:, t, :])  # (b,16)
                dec_u8 = (decoded.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
                x_np = x.numpy().reshape(-1)
                y_np = y.numpy().reshape(-1)
                previews[t][y_np, x_np] = dec_u8.reshape(-1)


        if (start // block_batch) % 10 == 0:
            print(f"[INFO] Blocks {start}..{end-1} / {N}")

    # Write DDS and PNGs
    out_png_written: List[Path] = []
    for t in range(T):
        write_dds_bc4(out_dds_paths[t], W, H, bytes(bc4_bytes[t]))
        print(f"[DONE] Wrote BC4 DDS: {out_dds_paths[t]}")
        if previews is not None and out_png_paths[t] is not None:
            Image.fromarray(previews[t], mode="L").save(out_png_paths[t])
            out_png_written.append(out_png_paths[t])
            print(f"[DONE] Wrote grayscale preview PNG: {out_png_paths[t]}")

    return InferOutputs(width=W, height=H, blocks_x=blocks_x, blocks_y=blocks_y,
                        out_dds_paths=out_dds_paths, out_png_paths=out_png_written)


# =========================
# DDS BC4 decoder (for generating ref PNGs)
# =========================
def decode_dds_bc4(path: str) -> np.ndarray:
    """Decode a BC4 (ATI1) DDS file into a grayscale uint8 numpy array (H,W)."""
    p = Path(path)
    data = p.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise RuntimeError(f"{p}: not a valid DDS")

    header = data[4:4+124]
    height = struct.unpack_from("<I", header, 8)[0]
    width = struct.unpack_from("<I", header, 12)[0]

    fourcc = header[80:84]
    offset = 128
    if fourcc == b"DX10":
        offset += 20
    elif fourcc not in (b"ATI1", b"BC4U"):
        raise RuntimeError(f"{p}: FourCC {fourcc!r} not supported (need ATI1/BC4)")

    bw = (width + 3) // 4
    bh = (height + 3) // 4
    out = np.zeros((bh * 4, bw * 4), dtype=np.uint8)

    for by_row in range(bh):
        row_data = data[offset + by_row * bw * 8: offset + (by_row + 1) * bw * 8]
        blocks = np.frombuffer(row_data, dtype=np.uint8).reshape(bw, 8)
        e0 = blocks[:, 0].astype(np.float32)
        e1 = blocks[:, 1].astype(np.float32)

        # Unpack 48-bit selectors -> 16 x 3-bit indices
        sel_bytes = blocks[:, 2:8].astype(np.uint64)
        packed = np.zeros(bw, dtype=np.uint64)
        for byte_i in range(6):
            packed |= sel_bytes[:, byte_i] << (byte_i * 8)

        indices = np.zeros((bw, 16), dtype=np.uint8)
        for px in range(16):
            indices[:, px] = ((packed >> (px * 3)) & 7).astype(np.uint8)

        # Build palette (BC4 selector order) with proper mode handling
        pal = np.zeros((bw, 8), dtype=np.float32)
        pal[:, 0] = e0
        pal[:, 1] = e1

        mode8 = (e0 > e1)
        # 8-value mode: sel2..7
        for k in range(1, 7):
            pal[:, k + 1] = np.where(mode8, ((7 - k) * e0 + k * e1) / 7.0, pal[:, k + 1])

        # 6-value mode: sel2..5 plus sel6=0, sel7=255
        mode6 = ~mode8
        if np.any(mode6):
            for k in range(1, 5):
                pal[mode6, k + 1] = ((5 - k) * e0[mode6] + k * e1[mode6]) / 5.0
            pal[mode6, 6] = 0.0
            pal[mode6, 7] = 255.0

        pal = np.clip(np.rint(pal), 0, 255).astype(np.uint8)

        # Decode
        colors = pal[np.arange(bw)[:, None], indices]  # (bw, 16)
        colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8).reshape(bw, 4, 4)
        row_pixels = colors.transpose(1, 0, 2).reshape(4, bw * 4)
        out[by_row * 4:(by_row + 1) * 4, :] = row_pixels

    return out[:height, :width]


def main():
    cfg = CONFIG
    print("\n==================== NTBC BC4 Inference ====================")
    res = infer_ntbc_bc4_to_dds_multi(
        coords_json=Path(cfg["coords_json"]),
        merged_ckpt=Path(cfg["merged_ckpt"]),
        out_dds=Path(cfg["out_dds"]),
        device=str(cfg["device"]),
        use_amp=bool(cfg["use_amp"]),
        block_batch=int(cfg["block_batch"]),
        save_preview_png=bool(cfg["save_preview_png"]),
        out_preview_png=Path(cfg["out_preview_png"]) if cfg.get("out_preview_png") else None,
    )

    # Decode reference BC4 DDS to PNGs
    ref_dds_list = cfg.get("ref_dds") or []
    out_dir = Path(cfg["inference_output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ref_path_str in ref_dds_list:
        ref_path = Path(ref_path_str)
        if not ref_path.exists():
            print(f"[WARN] Ref DDS not found, skipping: {ref_path}")
            continue
        ref_img = decode_dds_bc4(str(ref_path))
        ref_png = out_dir / f"{ref_path.stem}.png"
        Image.fromarray(ref_img, mode="L").save(ref_png)
        print(f"[DONE] Wrote ref PNG: {ref_png}")


if __name__ == "__main__":
    main()
