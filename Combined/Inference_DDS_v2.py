"""
NTBC BC1 FULL INFERENCE -> DDS (DXT1)  [MULTI-RGB-TEXTURE]
Loads a merged compressed state dict and writes one BC1 DDS per RGB texture.

This matches the paper's "train one model per material with multiple RGB textures":
- EndpointNet outputs 6*T per block
- ColorNet outputs 3*T per texel
Then at inference we split outputs and write separate BC1 DDS files.

Inputs:
- Inference_input.json (written by Dataset_Input_Extract_v2.py)
    { blocks_x, blocks_y, width, height, num_textures, texture_names }
- ntbc_bc1_merged_compressed.pt (written by Train_combined_v2.py)

Outputs:
- For T==1: exactly CONFIG['out_dds'] (same as your old script)
- For T>1 : writes CONFIG['out_dds'] with suffix per texture:
    base.dds -> base_<textureName>.dds  (fallback base_tex00.dds ...)
And optional preview PNGs the same way.

NOTE:
- This script does NOT need the original source images.
- The DDS size is blocks_x*4 by blocks_y*4 (already padded to multiple-of-4).
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

from Network_endpoint_v2 import EndpointNetwork, pack_rgb565_from_epq01, bc1_palette_from_endpoints, _BC1_W
from Network_color_v2 import ColorNetwork

from Model_param_compress import decompress_state_dict


# =========================
# CONFIG (edit these)
# =========================
CONFIG = {
    # Written by Dataset_Input_Extract_v2.py
    "coords_json": r"D:\BC1 extract\Bricks090_4K_test\Inference_input.json",

    # Written by Train_combined_v2.py
    "merged_ckpt": r"D:\BC1 extract\Bricks090_4K-PNG_model\ntbc_bc1_merged_compressed.pt",

    # Output DDS path (DXT1 / BC1). If multiple textures, suffixes will be added.
    "out_dds": r"D:\BC1 extract\Bricks090_4K-PNG_model\ntbc_out.dds",

    # Optional preview PNG(s)
    "save_preview_png": True,
    "out_preview_png": r"D:\BC1 extract\Bricks090_4K_test\ntbc_out_preview.png",

    # Runtime
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "block_batch": 65536,
    "pause_on_exit": False,
}


# =========================
# DDS (DXT1) writer
# =========================
def dds_header_dxt1(width: int, height: int) -> bytes:
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000

    DDPF_FOURCC = 0x4
    DDSCAPS_TEXTURE = 0x1000

    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    linear_size = blocks_x * blocks_y * 8

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
    pfFourCC = b"DXT1"
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


def write_dds_dxt1(path: Path, width: int, height: int, bc1_blocks: bytes) -> None:
    hdr = dds_header_dxt1(width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + bc1_blocks)


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
        raise ValueError("No grid keys found in state_dict. Expected keys like 'encoding.grids.0' ...")

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
    # pick last Linear weight in mlp.<idx>.weight
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
        raise ValueError("Could not infer MLP output dim from state_dict keys (mlp.*.weight).")
    return best_out


# =========================
# BC1 helpers
# =========================
def rgb565_to_q01_t(c: torch.Tensor) -> torch.Tensor:
    c = c.to(torch.int32)
    r5 = (c >> 11) & 31
    g6 = (c >> 5) & 63
    b5 = c & 31
    return torch.stack([r5.to(torch.float32) / 31.0,
                        g6.to(torch.float32) / 63.0,
                        b5.to(torch.float32) / 31.0], dim=-1)


_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
_OFF_Y = torch.tensor([0, 0, 0, 0,
                       1, 1, 1, 1,
                       2, 2, 2, 2,
                       3, 3, 3, 3], dtype=torch.long)


def pack_indices_u32(indices_16: torch.Tensor) -> torch.Tensor:
    idx = indices_16.to(torch.int64)
    shifts = (2 * torch.arange(16, device=idx.device, dtype=torch.int64)).view(1, 16)
    packed_i64 = torch.sum(((idx & 3) << shifts), dim=1)
    return packed_i64.to(torch.uint32)


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


@torch.no_grad()
def infer_ntbc_bc1_to_dds_multi(
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
        # fallback: infer later from model
        names = ["tex00"]

    print(f"[INFO] Texture size: {W}x{H} blocks=({blocks_x},{blocks_y})")

    # Load merged checkpoint and split
    merged = torch.load(merged_ckpt, map_location="cpu")
    ep_comp, col_comp = split_merged_state_dict(merged)

    ep_state = decompress_state_dict(ep_comp)
    col_state = decompress_state_dict(col_comp)

    # Infer T from state dicts (more reliable than JSON if user edited JSON)
    ep_out = _infer_out_dim_from_mlp(ep_state)    # 6*T
    col_out = _infer_out_dim_from_mlp(col_state)  # 3*T
    if ep_out % 6 != 0:
        raise ValueError(f"Endpoint MLP out_dim={ep_out} not divisible by 6.")
    if col_out % 3 != 0:
        raise ValueError(f"Color MLP out_dim={col_out} not divisible by 3.")
    T_ep = ep_out // 6
    T_col = col_out // 3
    if T_ep != T_col:
        raise ValueError(f"Mismatch: endpoint infers T={T_ep}, color infers T={T_col}.")
    T = T_ep

    if len(names) != T:
        print(f"[WARN] texture_names length ({len(names)}) != model T ({T}). Using model T.")
        names = [names[i] if i < len(names) else f"tex{i:02d}" for i in range(T)]

    print(f"[INFO] Multi-texture inference: T={T} -> writing {T} DDS files")

    # Instantiate models (infer grid params)
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
    bxby = torch.stack([grid_bx.reshape(-1), grid_by.reshape(-1)], dim=1)  # (N,2)
    N = int(bxby.shape[0])

    denom_x = float(max(1, blocks_x - 1))
    denom_y = float(max(1, blocks_y - 1))
    st = torch.stack([bxby[:, 0].to(torch.float32) / denom_x,
                      bxby[:, 1].to(torch.float32) / denom_y], dim=1)  # (N,2)

    # Output buffers per texture
    bc1_bytes = [bytearray(N * 8) for _ in range(T)]

    previews = None
    if save_preview_png:
        previews = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(T)]

    base_dds = out_dds.expanduser().resolve()
    base_png = out_preview_png.expanduser().resolve() if (save_preview_png and out_preview_png is not None) else None
    out_dds_paths, out_png_paths = _make_output_paths(base_dds, base_png, names)

    # Precompute mapping tensor for indices: paper order -> BC1 selector order
    map_idx = torch.tensor([0, 2, 3, 1], dtype=torch.uint8)

    for start in range(0, N, block_batch):
        end = min(N, start + block_batch)
        b = end - start

        bxby_b = bxby[start:end]  # CPU
        st_b = st[start:end].to(device=device, non_blocking=True)

        # Endpoints: (b,6*T) -> (b,T,6)
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            ep_pred_flat = ep_net(st_b).to(torch.float32)
        ep_pred = ep_pred_flat.view(b, T, 6)

        # Quantize endpoints to RGB565 per texture
        ep565 = []
        for t in range(T):
            ep565.append(pack_rgb565_from_epq01(ep_pred[:, t, :]).to(torch.int32))  # (b,2)
        ep565 = torch.stack(ep565, dim=1)  # (b,T,2)
        c0 = ep565[:, :, 0]
        c1 = ep565[:, :, 1]

        # Enforce 4-color mode per texture: c0 > c1
        swap = (c0 <= c1)
        c0_s = torch.where(swap, c1, c0)
        c1_s = torch.where(swap, c0, c1)
        c0, c1 = c0_s, c1_s

        # Fix equality to avoid 3-color/transparent mode
        equal = (c0 == c1)
        if equal.any():
            can_inc = (c0 < 0xFFFF)
            c0 = torch.where(equal & can_inc, c0 + 1, c0)
            c1 = torch.where(equal & ~can_inc, c1 - 1, c1)

        # Quantized endpoints to q01 (b,T,3)
        e0_q = rgb565_to_q01_t(c0)  # (b,T,3)
        e1_q = rgb565_to_q01_t(c1)  # (b,T,3)

        # Pixel coords inside each block
        base_x = (bxby_b[:, 0] * 4).view(-1, 1)  # (b,1)
        base_y = (bxby_b[:, 1] * 4).view(-1, 1)
        x = (base_x + _OFF_X.view(1, 16)).to(torch.int64)  # (b,16)
        y = (base_y + _OFF_Y.view(1, 16)).to(torch.int64)  # (b,16)

        # UV in [0,1] for all pixels
        u = (x.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (y.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=-1).reshape(-1, 2)  # (b*16,2)

        # Predict colors: (b*16, 3*T) -> (b,16,T,3) -> (b,T,16,3)
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            pred_flat = col_net(uv).to(torch.float32)
        pred = pred_flat.view(b * 16, T, 3).view(b, 16, T, 3).permute(0, 2, 1, 3).contiguous()  # (b,T,16,3)

        # Palette from quantized endpoints in paper order (b,T,4,3)
        w = _BC1_W.to(device=device, dtype=torch.float32)  # (4,)
        pal = bc1_palette_from_endpoints(e0_q.to(device), e1_q.to(device), w=w)  # (b,T,4,3)

        # Nearest palette index per pixel (paper order): (b,T,16,4)
        diff = pred.unsqueeze(3) - pal.unsqueeze(2)  # (b,T,16,4,3)
        d2 = (diff * diff).sum(dim=-1)  # (b,T,16,4)
        idx_paper = torch.argmin(d2, dim=-1).to(torch.uint8)  # (b,T,16)

        # Map to BC1 selector order
        idx_bc1 = map_idx.to(device=idx_paper.device)[idx_paper.long()]  # (b,T,16)

        # Pack and write bytes per texture
        packed_idx = pack_indices_u32(idx_bc1.view(b * T, 16)).to("cpu").numpy().astype(np.uint32).reshape(b, T)
        c0_cpu = c0.to("cpu").to(torch.uint16).numpy()
        c1_cpu = c1.to("cpu").to(torch.uint16).numpy()

        for t in range(T):
            bb = bc1_bytes[t]
            for i in range(b):
                off = (start + i) * 8
                struct.pack_into("<HHI", bb, off, int(c0_cpu[i, t]), int(c1_cpu[i, t]), int(packed_idx[i, t]))

        # Optional preview per texture
        if previews is not None:
            # BC1 order palette: [c0, c1, (2c0+c1)/3, (c0+2c1)/3] in q01
            e0 = e0_q.to(torch.float32)
            e1 = e1_q.to(torch.float32)
            pal_bc1 = torch.stack([e0, e1, (2.0 * e0 + e1) / 3.0, (e0 + 2.0 * e1) / 3.0], dim=2)  # (b,T,4,3)
            pal_cpu = pal_bc1.to("cpu")
            idx_cpu = idx_bc1.to("cpu").to(torch.int64)  # (b,T,16)

            # Gather and scatter
            dec = torch.gather(
                pal_cpu.unsqueeze(2).expand(-1, -1, 16, -1, -1),
                3,
                idx_cpu.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3),
            ).squeeze(3)  # (b,T,16,3)
            dec_u8 = (dec.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()  # (b,T,16,3)

            x_np = x.numpy().reshape(-1)
            y_np = y.numpy().reshape(-1)
            for t in range(T):
                previews[t][y_np, x_np] = dec_u8[:, t, :, :].reshape(-1, 3)

        if (start // block_batch) % 10 == 0:
            print(f"[INFO] Blocks {start}..{end-1} / {N}")

    # Write DDS (and PNGs)
    out_png_written: List[Path] = []
    for t in range(T):
        write_dds_dxt1(out_dds_paths[t], W, H, bytes(bc1_bytes[t]))
        print(f"[DONE] Wrote DDS: {out_dds_paths[t]}")
        if previews is not None and out_png_paths[t] is not None:
            Image.fromarray(previews[t], mode="RGB").save(out_png_paths[t])
            out_png_written.append(out_png_paths[t])
            print(f"[DONE] Wrote preview PNG: {out_png_paths[t]}")

    return InferOutputs(width=W, height=H, blocks_x=blocks_x, blocks_y=blocks_y,
                        out_dds_paths=out_dds_paths, out_png_paths=out_png_written)


def main():
    cfg = CONFIG
    res = infer_ntbc_bc1_to_dds_multi(
        coords_json=Path(cfg["coords_json"]),
        merged_ckpt=Path(cfg["merged_ckpt"]),
        out_dds=Path(cfg["out_dds"]),
        device=str(cfg["device"]),
        use_amp=bool(cfg["use_amp"]),
        block_batch=int(cfg["block_batch"]),
        save_preview_png=bool(cfg["save_preview_png"]),
        out_preview_png=Path(cfg["out_preview_png"]) if cfg.get("out_preview_png") else None,
    )
    print(f"[SUMMARY] {res.width}x{res.height} blocks={res.blocks_x}x{res.blocks_y} outputs={len(res.out_dds_paths)}")
    if bool(cfg["pause_on_exit"]):
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
