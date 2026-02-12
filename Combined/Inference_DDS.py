"""
NTBC BC1 FULL INFERENCE -> DDS (DXT1)
(Compressed variant: loads uint8-quantized state dicts.)

What this does
--------------
Given a texture-specific trained EndpointNetwork + ColorNetwork (trained separately like the paper),
this script runs inference to produce a BC1-compressed DDS (DXT1 / BC1).

Pipeline (paper-style inference)
--------------------------------
For each 4x4 block:
  1) EndpointNet predicts endpoints (e0,e1) from block coords (s,t).
  2) Endpoints are quantized to RGB565 (c0,c1).
  3) Enforce 4-color mode: if c0 <= c1, swap c0/c1 (and swap endpoints).
  4) ColorNet predicts RGB for each texel coord (u,v).
  5) Build the 4-color palette from the (quantized) endpoints.
  6) Pick indices in paper palette order, then remap to BC1/DXT1 selector order.
  7) Pack into BC1 blocks and write a DDS (DXT1).

NOTE
----
- NTBC is per-texture. You should run this on the SAME texture you trained on
  (or at least same resolution), otherwise quality may be garbage.
- BC1 requires width/height multiples of 4. We pad the input image by edge replication.
  The DDS header uses the padded size.

No "bash" needed: just edit CONFIG below and run the .py normally from your IDE / double click.

"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

import torch

from Network_endpoint import EndpointNetwork, pack_rgb565_from_epq01, bc1_palette_from_endpoints, endpoints6_to_e0e1, _BC1_W
from Network_color import ColorNetwork

from state_dict_compress import decompress_state_dict


# =========================
# CONFIG (edit these)
# =========================
CONFIG = {
    # The image you trained NTBC on (used here mainly for width/height and optional preview comparison)
    "input_image": r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",

    # Endpoint network checkpoint (either a full checkpoint with {"model_state":...} OR pure state_dict)
    "endpoint_ckpt": r"D:\BC1 extract\Bricks090_diffuse\runs_endpoint_bc1_steps_compressed\endpoint_net_bc1_final_state_dict_compressed.pt",

    # Color network checkpoint (compressed state_dict)
    "color_ckpt": r"D:\BC1 extract\Bricks090_diffuse\runs_color_bc1_steps_compressed\color_net_bc1_final_state_dict_compressed.pt",

    # Output DDS path (DXT1 / BC1)
    "out_dds": r"D:\BC1 extract\Bricks090_diffuse\ntbc_out.dds",

    # Optional: also save a decoded preview PNG (what the DDS would decode to)
    "save_preview_png": True,
    "out_preview_png": r"D:\BC1 extract\Bricks090_diffuse\ntbc_out_preview.png",

    # Runtime
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,           # AMP helps on CUDA
    "block_batch": 4096,       # blocks per batch (each block = 16 pixels)
    "pause_on_exit": False,    # set True if double-clicking and you want the window to stay
}


# =========================
# DDS (DXT1) writer
# =========================
def dds_header_dxt1(width: int, height: int) -> bytes:
    """
    Writes a minimal DDS header for BC1/DXT1 (no DX10 header).
    """
    # DDS constants
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000

    DDPF_FOURCC = 0x4

    DDSCAPS_TEXTURE = 0x1000

    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    linear_size = blocks_x * blocks_y * 8  # BC1 block = 8 bytes

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = linear_size
    dwDepth = 0
    dwMipMapCount = 0
    dwReserved1 = [0] * 11

    # DDS_PIXELFORMAT (32 bytes)
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
        "<I I I I I I I 11I",  # main header fields
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
# Image helpers
# =========================
def load_image_rgb_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)  # HWC


def pad_image_to_multiple_of_4(img: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    pad_h = (-H) % 4
    pad_w = (-W) % 4
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


# =========================
# Model loading (robust)
# =========================
def _extract_state_dict(ckpt_obj):
    """Extract state dict from checkpoint, decompressing uint8 grids if needed."""
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        sd = ckpt_obj["model_state"]
    else:
        sd = ckpt_obj
    # Decompress uint8-quantized grids back to float
    return decompress_state_dict(sd)


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


@torch.no_grad()
def load_endpoint_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    nl, br, fr, fd, dt = _infer_grid_params_from_state(state, prefix="encoding.grids.")
    param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32

    net = EndpointNetwork(
        num_levels=nl,
        base_resolution=br,
        finest_resolution=fr,
        feature_dim=fd,
        hidden_dim=64,
        num_hidden_layers=3,
        param_dtype=param_dtype,
    ).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


@torch.no_grad()
def load_color_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    nl, br, fr, fd, dt = _infer_grid_params_from_state(state, prefix="encoding.grids.")
    # ColorNetwork hardcodes feature_dim=2, but we validate
    if fd != 2:
        print(f"[WARN] Color grid feature_dim is {fd}, but code assumes 2. Proceeding anyway.")
    param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32

    net = ColorNetwork(
        param_dtype=param_dtype,
        finest_resolution=fr,
        base_resolution=br,
        num_levels=nl,
    ).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


# =========================
# BC1 helpers
# =========================
def rgb565_to_q01_t(c: torch.Tensor) -> torch.Tensor:
    """
    c: (...,) int/uint
    returns (...,3) float32 in [0,1] with RGB565 quantization
    """
    c = c.to(torch.int32)
    r5 = (c >> 11) & 31
    g6 = (c >> 5) & 63
    b5 = c & 31
    out = torch.stack([r5.to(torch.float32) / 31.0,
                       g6.to(torch.float32) / 63.0,
                       b5.to(torch.float32) / 31.0], dim=-1)
    return out


_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)  # (16,)
_OFF_Y = torch.tensor([0, 0, 0, 0,
                       1, 1, 1, 1,
                       2, 2, 2, 2,
                       3, 3, 3, 3], dtype=torch.long)      # (16,)


def pack_indices_u32(indices_16: torch.Tensor) -> torch.Tensor:
    """
    indices_16: (B,16) uint8 in [0..3]
    returns: (B,) uint32 where bits are packed as BC1 expects:
      bits  1:0   -> pixel 0
      bits  3:2   -> pixel 1
      ...
      bits 31:30  -> pixel 15

    NOTE: torch.arange(..., dtype=torch.uint32) is not implemented on CUDA.
    We do all math in int64 and cast at the end.
    """
    idx = indices_16.to(torch.int64)
    shifts = (2 * torch.arange(16, device=idx.device, dtype=torch.int64)).view(1, 16)
    packed_i64 = torch.sum(((idx & 3) << shifts), dim=1)
    return packed_i64.to(torch.uint32)


# =========================
# Main inference
# =========================
@dataclass
class InferResult:
    width: int
    height: int
    blocks_x: int
    blocks_y: int
    out_dds: Path
    out_preview_png: Path | None


@torch.no_grad()
def infer_ntbc_bc1_to_dds(
    input_image: Path,
    endpoint_ckpt: Path,
    color_ckpt: Path,
    out_dds: Path,
    device: str = "cuda",
    use_amp: bool = True,
    block_batch: int = 4096,
    save_preview_png: bool = True,
    out_preview_png: Path | None = None,
) -> InferResult:
    # Load and pad image
    img_u8 = load_image_rgb_u8(input_image)
    img_u8 = pad_image_to_multiple_of_4(img_u8)
    H, W, _ = img_u8.shape
    blocks_x = W // 4
    blocks_y = H // 4

    print(f"[INFO] Input image (padded): {W}x{H}  blocks=({blocks_x},{blocks_y})")

    # Models
    print("[INFO] Loading EndpointNet...")
    ep_net = load_endpoint_model(endpoint_ckpt, device=device)
    print("[INFO] Loading ColorNet...")
    col_net = load_color_model(color_ckpt, device=device)

    amp_enabled = bool(use_amp) and (device == "cuda")
    autocast_device = "cuda" if device == "cuda" else "cpu"

    # Prepare all block indices in DDS row-major order: y-major then x
    # Linear block index should be: idx = by * blocks_x + bx (x changes fastest).
    bx = torch.arange(blocks_x, dtype=torch.int64)
    by = torch.arange(blocks_y, dtype=torch.int64)
    grid_by, grid_bx = torch.meshgrid(by, bx, indexing="ij")  # (blocks_y, blocks_x)
    bxby = torch.stack([grid_bx.reshape(-1), grid_by.reshape(-1)], dim=1)  # (N,2)
    N = bxby.shape[0]

    # Block coords (s,t) exactly like Endpoints_Extract.py:
    # s=bx/(Bx-1), t=by/(By-1)
    denom_x = float(max(1, blocks_x - 1))
    denom_y = float(max(1, blocks_y - 1))
    st = torch.stack([bxby[:, 0].to(torch.float32) / denom_x,
                      bxby[:, 1].to(torch.float32) / denom_y], dim=1)  # (N,2)

    # Output buffers
    bc1_bytes = bytearray(N * 8)

    preview = None
    if save_preview_png:
        preview = np.zeros((H, W, 3), dtype=np.uint8)

    # Process in batches of blocks
    for start in range(0, N, block_batch):
        end = min(N, start + block_batch)
        b = end - start

        bxby_b = bxby[start:end]  # CPU
        st_b = st[start:end].to(device=device, non_blocking=True)

        # Predict endpoints (float in [0,1]) and quantize to RGB565
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            ep_pred = ep_net(st_b).to(torch.float32)  # (b,6)

        ep565 = pack_rgb565_from_epq01(ep_pred).to(torch.int32)  # (b,2) [c0,c1] uint16-ish

        c0 = ep565[:, 0]
        c1 = ep565[:, 1]

        # Enforce 4-color mode: make sure c0 > c1
        swap = (c0 <= c1)
        if swap.any():
            c0_s = torch.where(swap, c1, c0)
            c1_s = torch.where(swap, c0, c1)
            c0, c1 = c0_s, c1_s

        # Quantized endpoints in q01 (so palette matches what GPU will decode)
        e0_q = rgb565_to_q01_t(c0)  # (b,3)
        e1_q = rgb565_to_q01_t(c1)  # (b,3)

        # Expand endpoints to 16 pixels
        e0_16 = e0_q.unsqueeze(1).expand(-1, 16, -1)  # (b,16,3)
        e1_16 = e1_q.unsqueeze(1).expand(-1, 16, -1)

        # Pixel coordinates inside each block
        base_x = (bxby_b[:, 0] * 4).view(-1, 1)  # (b,1)
        base_y = (bxby_b[:, 1] * 4).view(-1, 1)

        x = (base_x + _OFF_X.view(1, 16)).to(torch.int64)  # (b,16)
        y = (base_y + _OFF_Y.view(1, 16)).to(torch.int64)  # (b,16)

        # UV in [0,1]
        u = (x.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (y.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=-1).reshape(-1, 2)  # (b*16,2)

        # Predict colors
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            pred_c = col_net(uv).to(torch.float32)  # (p,3)

        pred_c = pred_c.view(b, 16, 3)

        # Palette from quantized endpoints (b,4,3)
        w = _BC1_W.to(device=device, dtype=torch.float32)
        pal = bc1_palette_from_endpoints(e0_q.to(device), e1_q.to(device), w=w)  # (b,4,3)

        # Nearest palette index per pixel
        # dist^2: (b,16,4)
        diff = pred_c.unsqueeze(2) - pal.unsqueeze(1)  # (b,16,4,3)
        d2 = (diff * diff).sum(dim=-1)  # (b,16,4)
        idx = torch.argmin(d2, dim=-1).to(torch.uint8)  # (b,16)  (paper palette order)

        # Remap from paper palette order [c0, c(2/3), c(1/3), c1]
        # to BC1/DXT1 selector order   [c0, c1,    c(2/3), c(1/3)]
        # mapping: paper 0->0, 1->2, 2->3, 3->1
        map_idx = torch.tensor([0, 2, 3, 1], device=idx.device, dtype=torch.uint8)
        idx_bc1 = map_idx[idx.long()]

        packed_idx = pack_indices_u32(idx_bc1).to("cpu")  # (b,) uint32 on CPU
        c0_cpu = c0.to("cpu").to(torch.uint16).numpy()
        c1_cpu = c1.to("cpu").to(torch.uint16).numpy()
        packed_cpu = packed_idx.numpy().astype(np.uint32)

        # Write block bytes into bytearray (8 bytes per block)
        for i in range(b):
            off = (start + i) * 8
            struct.pack_into("<HHI", bc1_bytes, off, int(c0_cpu[i]), int(c1_cpu[i]), int(packed_cpu[i]))

        # Optional preview: decode with palette + indices and write to an RGB image
        if preview is not None:
            # decode colors per pixel (b,16,3) in float then to uint8
            # gather palette colors by indices
            # Decode preview using BC1 selector order so it matches the DDS viewer.
            # BC1 order: [c0, c1, (2c0+c1)/3, (c0+2c1)/3]
            pal_bc1 = torch.stack([
                e0_q.to(torch.float32),
                e1_q.to(torch.float32),
                (2.0 * e0_q.to(torch.float32) + e1_q.to(torch.float32)) / 3.0,
                (e0_q.to(torch.float32) + 2.0 * e1_q.to(torch.float32)) / 3.0,
            ], dim=1)  # (b,4,3)
            pal_cpu = pal_bc1.to("cpu")
            idx_cpu = idx_bc1.to("cpu").to(torch.int64)  # (b,16)
            dec = torch.gather(
                pal_cpu.unsqueeze(1).expand(-1, 16, -1, -1),
                2,
                idx_cpu.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
            ).squeeze(2)  # (b,16,3)
            dec_u8 = (dec.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()

            # scatter
            x_np = x.numpy().reshape(-1)
            y_np = y.numpy().reshape(-1)
            preview[y_np, x_np] = dec_u8.reshape(-1, 3)

        if (start // block_batch) % 10 == 0:
            print(f"[INFO] Blocks {start}..{end-1} / {N}")

    # Write DDS
    out_dds = out_dds.expanduser().resolve()
    write_dds_dxt1(out_dds, W, H, bytes(bc1_bytes))
    print(f"[DONE] Wrote DDS (DXT1): {out_dds}")

    out_png = None
    if preview is not None:
        out_png = (out_preview_png or (out_dds.with_suffix(".png"))).expanduser().resolve()
        Image.fromarray(preview, mode="RGB").save(out_png)
        print(f"[DONE] Wrote preview PNG: {out_png}")

    return InferResult(width=W, height=H, blocks_x=blocks_x, blocks_y=blocks_y, out_dds=out_dds, out_preview_png=out_png)


def main():
    cfg = CONFIG
    res = infer_ntbc_bc1_to_dds(
        input_image=Path(cfg["input_image"]),
        endpoint_ckpt=Path(cfg["endpoint_ckpt"]),
        color_ckpt=Path(cfg["color_ckpt"]),
        out_dds=Path(cfg["out_dds"]),
        device=str(cfg["device"]),
        use_amp=bool(cfg["use_amp"]),
        block_batch=int(cfg["block_batch"]),
        save_preview_png=bool(cfg["save_preview_png"]),
        out_preview_png=Path(cfg["out_preview_png"]) if cfg.get("out_preview_png") else None,
    )
    print(f"[SUMMARY] {res.width}x{res.height}  blocks={res.blocks_x}x{res.blocks_y}")

    if bool(cfg["pause_on_exit"]):
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
