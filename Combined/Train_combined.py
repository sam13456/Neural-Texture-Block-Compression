"""
Train both NTBC BC1 networks the way the paper does it: SEPARATELY.

- Endpoint network:
    input  : block coords (s,t) from endpoint dataset JSON
    target : reference endpoints (Compressonator) + decoded-color loss using source image

- Color network:
    input  : texel coords (u,v) sampled from blocks in the endpoint dataset
    target : reference RGB from source image + decoded-color loss using REFERENCE endpoints

This script intentionally does NOT generate datasets. You must already have:
  1) bc1_endpoint_dataset.json (from your Endpoints_Extract.py)
  2) original uncompressed source image (RGB)

Drop this file next to:
  - Network.py
  - Network_color.py
and run it.

Paper-aligned schedule:
  main_steps + 10% QAT tail, with 10-step warmup + cosine decay.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch


# -------------------------
# Robust imports (your repo has had both names at different times)
# -------------------------

from Network_endpoint import EndpointNetwork, endpoint_loss_bc1

from Network_color import ColorNetwork, color_loss_bc1

from Model_param_compress import compress_state_dict, print_size_comparison

# =========================
# CONFIG (edit these)
# =========================
CONFIG = {
    # Shared paths
    "endpoint_dataset_json": r"D:\BC1 extract\Bricks090_diffuse_2K_test\Train_dataset.json",
    "source_image": r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",

    # Run toggles
    "run_endpoint_training": True,
    "run_color_training": True,

    # Output folders (single base dir for both)
    "out_dir_endpoint": r"D:\BC1 extract\Bricks090_diffuse_2K_test\runs_endpoint",
    "out_dir_color": r"D:\BC1 extract\Bricks090_diffuse_2K_test\runs_color",

    # Merged output (both networks in one file)
    "out_dir_merged": r"D:\BC1 extract\Bricks090_diffuse_2K_test",

    # Training schedule (paper)
    "main_steps": 20_000,
    "qat_tail_fraction": 0.10,  # +10% steps
    "warmup_steps": 10,

    # Optimizer (paper)
    "lr_grid": 1e-2,
    "lr_mlp": 5e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-15,

    # STE temperature (paper)
    "temperature": 0.01,

    # QAT (paper Sec 3.3)
    "qat_bits": 8,
    "freeze_grids_during_qat": True,  # close to "fine-tune MLPs with quantized grids"


    # Batch sizes
    "batch_size_blocks": 2048,     # endpoint net: blocks per step
    "batch_size_texels": 65536,   # color net: texels per step

    # Mixed precision
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,              # AMP on CUDA
    "param_dtype": "float32",     # keep params float32 for stability (recommended)

    # Logging / saving
    "log_every_steps": 50,
    "save_every_steps": 10000,

    # Repro
    "seed": 0,
}


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_scale_warmup_cos(step: int, total_steps: int, warmup_steps: int) -> float:
    """Paper: warmup 10, cosine to 0."""
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    t = float(step - warmup_steps) / float(denom)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def set_lrs(optimizer: torch.optim.Optimizer, lr_scale: float, lr_grid: float, lr_mlp: float, grid_lr_mul: float = 1.0):
    optimizer.param_groups[0]["lr"] = (lr_grid * grid_lr_mul) * lr_scale
    optimizer.param_groups[1]["lr"] = lr_mlp * lr_scale


def save_checkpoint(out_dir: Path, name: str, cfg: dict, meta: dict, net: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int):
    ckpt = {
        "global_step": step,
        "config": cfg,
        "meta": meta,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path = out_dir / name
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_image_rgb_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)  # HWC


def pad_image_to_blocks(img: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    """Pad image to (blocks_y*4, blocks_x*4) using edge replication."""
    H, W, C = img.shape
    target_h = blocks_y * 4
    target_w = blocks_x * 4
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


# 16 offsets for a 4x4 block in row-major order (endpoint training)
_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)  # (16,)
_OFF_Y = torch.tensor([0, 0, 0, 0,
                       1, 1, 1, 1,
                       2, 2, 2, 2,
                       3, 3, 3, 3], dtype=torch.long)      # (16,)


def extract_block_colors_batch_u8(img_hwc_u8_t: torch.Tensor, bxby_batch: torch.Tensor) -> torch.Tensor:
    """
    img_hwc_u8_t: (H,W,3) uint8 CPU tensor
    bxby_batch:   (B,2) int64 CPU tensor with bx,by
    Returns:      (B,16,3) uint8 CPU tensor
    """
    bx = bxby_batch[:, 0]
    by = bxby_batch[:, 1]
    base_x = bx * 4
    base_y = by * 4
    x = base_x[:, None] + _OFF_X[None, :]
    y = base_y[:, None] + _OFF_Y[None, :]
    return img_hwc_u8_t[y, x]  # (B,16,3) uint8


# =========================
# Dataset loading
# =========================
def load_endpoint_dataset_json(path: Path):
    """
    Expected schema (from Dataset_Extract.py):
      inputs:
        - bxby : (N,2) block indices
      targets:
        - ep_q01 : (N,6) endpoints in [0,1]
      meta (optional):
        - blocks_x, blocks_y, width, height
    """
    d = json.loads(path.read_text())
    bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int64)
    ep = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)
    meta = d.get("meta", {})
    return bxby, ep, meta


# =========================
# Training: Endpoint network
# =========================
def train_endpoint_network(cfg: dict, endpoint_dataset_json: Path, source_image: Path, out_dir: Path):
    device = str(cfg["device"])
    use_amp = bool(cfg["use_amp"]) and device.startswith("cuda")
    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32

    bxby_np, ep_np, meta = load_endpoint_dataset_json(endpoint_dataset_json)
    N = int(bxby_np.shape[0])

    blocks_x = int(meta.get("blocks_x", int(bxby_np[:, 0].max() + 1)))
    blocks_y = int(meta.get("blocks_y", int(bxby_np[:, 1].max() + 1)))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))
    print(f"[Endpoint] Dataset: N={N} blocks, blocks=({blocks_x},{blocks_y}), tex={width}x{height}")

    # Compute st from bxby on the fly
    st_np = np.zeros((N, 2), dtype=np.float32)
    st_np[:, 0] = bxby_np[:, 0] / max(blocks_x - 1, 1)
    st_np[:, 1] = bxby_np[:, 1] / max(blocks_y - 1, 1)

    # CPU tensors
    st_t = torch.from_numpy(st_np)                       # (N,2) float32 CPU
    bxby_t = torch.from_numpy(bxby_np)                   # (N,2) int64 CPU
    ep_t = torch.from_numpy(ep_np)                       # (N,6) float32 CPU

    # Reference image (uint8 CPU)
    img_u8 = load_image_rgb_u8(source_image)
    img_u8 = pad_image_to_blocks(img_u8, blocks_x, blocks_y)
    img_u8_t = torch.from_numpy(img_u8)                  # (H,W,3) uint8 CPU
    print(f"[Endpoint] Image loaded (padded): {img_u8_t.shape[1]}x{img_u8_t.shape[0]}")

    # Model
    net = EndpointNetwork(param_dtype=param_dtype).to(device)
    net.train()

    optimizer = torch.optim.Adam(
        [
            {"params": net.encoding.parameters(), "lr": float(cfg["lr_grid"])},
            {"params": net.mlp.parameters(), "lr": float(cfg["lr_mlp"])},
        ],
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    main_steps = int(cfg["main_steps"])
    qat_tail_steps = int(round(main_steps * float(cfg["qat_tail_fraction"])))
    total_steps = main_steps + qat_tail_steps
    qat_start_step = main_steps
    warmup_steps = int(cfg["warmup_steps"])
    batch_size = int(cfg["batch_size_blocks"])
    temperature = float(cfg["temperature"])

    log_every = int(cfg["log_every_steps"])
    save_every = int(cfg["save_every_steps"])
    freeze_grids = bool(cfg["freeze_grids_during_qat"])
    grid_lr_mul = 1.0

    print(f"[Endpoint] Train steps: main={main_steps}, qat_tail={qat_tail_steps}, total={total_steps}")
    print(f"[Endpoint] QAT enables at step {qat_start_step} (0-indexed). Batch={batch_size} blocks/step")

    run_total = run_le = run_lcd = 0.0
    run_count = 0
    t0 = time.time()

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for step in range(total_steps):
        if step == qat_start_step and qat_tail_steps > 0:
            print("[Endpoint] >>> Enabling grid QAT (fake quant) <<<")
            net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
            if freeze_grids:
                print("[Endpoint] >>> Freezing grids during QAT tail (lr_grid -> 0) <<<")
                grid_lr_mul = 0.0
            save_checkpoint(out_dir, f"endpoint_net_bc1_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        # Sample blocks
        didx = torch.randint(0, N, (batch_size,), device="cpu", dtype=torch.long)

        coords = st_t[didx].to(device=device, non_blocking=True)           # (B,2)
        ref_ep = ep_t[didx].to(device=device, non_blocking=True)           # (B,6)
        bxby_batch = bxby_t[didx]                                          # (B,2) CPU

        # Reference colors for each block: (B,16,3)
        cols_u8 = extract_block_colors_batch_u8(img_u8_t, bxby_batch)
        cols = cols_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0

        # LR schedule
        lr_scale = lr_scale_warmup_cos(step, total_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, float(cfg["lr_grid"]), float(cfg["lr_mlp"]), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            pred = net(coords)  # (B,6)
            loss_out = endpoint_loss_bc1(
                pred_endpoints6=pred,
                ref_endpoints6=ref_ep,
                ref_colors=cols,
                temperature=temperature,
                reduction="mean",
            )
            loss = loss_out.total

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_total += float(loss_out.total.detach().cpu().item())
        run_le += float(loss_out.le.detach().cpu().item())
        run_lcd += float(loss_out.lcd.detach().cpu().item())
        run_count += 1

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"[Endpoint step {step+1:06d}/{total_steps}] "
                f"loss={run_total/run_count:.6f} (Le={run_le/run_count:.6f}, Lcd={run_lcd/run_count:.6f}) "
                f"lr_grid={optimizer.param_groups[0]['lr']:.5g} lr_mlp={optimizer.param_groups[1]['lr']:.5g} "
                f"time={dt:.1f}s"
            )
            run_total = run_le = run_lcd = 0.0
            run_count = 0
            t0 = time.time()

        if save_every > 0 and ((step + 1) % save_every == 0):
            save_checkpoint(out_dir, f"endpoint_net_bc1_step{step+1:06d}.pt", cfg, meta, net, optimizer, step + 1)

    save_checkpoint(out_dir, "endpoint_net_bc1_final.pt", cfg, meta, net, optimizer, total_steps)

    # Compress state dict (grids -> uint8) and return it (merging happens in main)
    original_sd = net.state_dict()
    compressed_sd = compress_state_dict(original_sd)
    print("[Endpoint] Compressed state_dict:")
    print_size_comparison(original_sd, compressed_sd)
    return compressed_sd


# =========================
# Training: Color network
# =========================
def train_color_network(cfg: dict, endpoint_dataset_json: Path, source_image: Path, out_dir: Path):
    device = str(cfg["device"])
    use_amp = bool(cfg["use_amp"]) and device.startswith("cuda")
    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32

    # Load endpoint dataset (for bxby + reference endpoints)
    bxby_np, ep_np, meta = load_endpoint_dataset_json(endpoint_dataset_json)
    N = int(bxby_np.shape[0])

    blocks_x = int(meta.get("blocks_x", int(bxby_np[:, 0].max() + 1)))
    blocks_y = int(meta.get("blocks_y", int(bxby_np[:, 1].max() + 1)))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))
    print(f"[Color] Endpoint dataset: N={N}, blocks=({blocks_x},{blocks_y}), tex={width}x{height}")

    # Reference image
    img_u8 = load_image_rgb_u8(source_image)
    img_u8 = pad_image_to_blocks(img_u8, blocks_x, blocks_y)
    H, W, _ = img_u8.shape
    print(f"[Color] Image loaded (padded): {W}x{H}")

    # CPU tensors
    img_u8_t = torch.from_numpy(img_u8)                  # (H,W,3) uint8 CPU
    bxby_t = torch.from_numpy(bxby_np.astype(np.int64))  # (N,2) int64 CPU
    ep_t = torch.from_numpy(ep_np.astype(np.float32))    # (N,6) float32 CPU

    # Model
    # Paper uses finest 2048 for 2K textures; you can change in Network_color if needed.
    net = ColorNetwork(param_dtype=param_dtype, finest_resolution=2048).to(device)
    net.train()

    optimizer = torch.optim.Adam(
        [
            {"params": net.encoding.parameters(), "lr": float(cfg["lr_grid"])},
            {"params": net.mlp.parameters(), "lr": float(cfg["lr_mlp"])},
        ],
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    main_steps = int(cfg["main_steps"])
    qat_tail_steps = int(round(main_steps * float(cfg["qat_tail_fraction"])))
    total_steps = main_steps + qat_tail_steps
    qat_start_step = main_steps
    warmup_steps = int(cfg["warmup_steps"])
    batch_size = int(cfg["batch_size_texels"])
    temperature = float(cfg["temperature"])

    log_every = int(cfg["log_every_steps"])
    save_every = int(cfg["save_every_steps"])
    freeze_grids = bool(cfg["freeze_grids_during_qat"])
    grid_lr_mul = 1.0

    print(f"[Color] Train steps: main={main_steps}, qat_tail={qat_tail_steps}, total={total_steps}")
    print(f"[Color] QAT enables at step {qat_start_step} (0-indexed). Batch={batch_size} texels/step")

    run_total = run_lc = run_lcd = 0.0
    run_count = 0
    t0 = time.time()

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for step in range(total_steps):
        if step == qat_start_step and qat_tail_steps > 0:
            print("[Color] >>> Enabling grid QAT (fake quant) <<<")
            net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
            if freeze_grids:
                print("[Color] >>> Freezing grids during QAT tail (lr_grid -> 0) <<<")
                grid_lr_mul = 0.0
            save_checkpoint(out_dir, f"color_net_bc1_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        # Sample blocks, then a random texel in each 4x4 block
        didx = torch.randint(0, N, (batch_size,), device="cpu", dtype=torch.long)

        bxby = bxby_t[didx]      # (B,2) CPU
        ref_ep = ep_t[didx]      # (B,6) CPU

        ox = torch.randint(0, 4, (batch_size,), device="cpu", dtype=torch.long)
        oy = torch.randint(0, 4, (batch_size,), device="cpu", dtype=torch.long)

        px = (bxby[:, 0] * 4 + ox).clamp(0, W - 1)
        py = (bxby[:, 1] * 4 + oy).clamp(0, H - 1)

        # Reference RGB (uint8 -> float)
        ref_rgb_u8 = img_u8_t[py, px]  # (B,3) uint8 CPU
        ref_rgb = (ref_rgb_u8.to(torch.float32) / 255.0).to(device=device, non_blocking=True)

        # UV coords in [0,1]
        u = (px.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (py.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=1)

        ref_ep = ref_ep.to(device=device, dtype=torch.float32, non_blocking=True)

        # LR schedule
        lr_scale = lr_scale_warmup_cos(step, total_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, float(cfg["lr_grid"]), float(cfg["lr_mlp"]), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            pred = net(uv)  # (B,3)
            loss_out = color_loss_bc1(
                pred_color=pred,
                ref_color=ref_rgb,
                ref_endpoints6=ref_ep,
                temperature=temperature,
                reduction="mean",
            )
            loss = loss_out.total

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_total += float(loss_out.total.detach().cpu().item())
        run_lc += float(loss_out.lc.detach().cpu().item())
        run_lcd += float(loss_out.lcd.detach().cpu().item())
        run_count += 1

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            print(
                f"[Color step {step+1:06d}/{total_steps}] "
                f"loss={run_total/run_count:.6f} (Lc={run_lc/run_count:.6f}, Lcd={run_lcd/run_count:.6f}) "
                f"lr_grid={optimizer.param_groups[0]['lr']:.5g} lr_mlp={optimizer.param_groups[1]['lr']:.5g} "
                f"time={dt:.1f}s"
            )
            run_total = run_lc = run_lcd = 0.0
            run_count = 0
            t0 = time.time()

        if save_every > 0 and ((step + 1) % save_every == 0):
            save_checkpoint(out_dir, f"color_net_bc1_step{step+1:06d}.pt", cfg, meta, net, optimizer, step + 1)

    save_checkpoint(out_dir, "color_net_bc1_final.pt", cfg, meta, net, optimizer, total_steps)

    # Compress state dict (grids -> uint8) and return it (merging happens in main)
    original_sd = net.state_dict()
    compressed_sd = compress_state_dict(original_sd)
    print("[Color] Compressed state_dict:")
    print_size_comparison(original_sd, compressed_sd)
    return compressed_sd


# =========================
# Main
# =========================
def merge_compressed_state_dicts(
    endpoint_sd: Dict[str, torch.Tensor],
    color_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Merge two compressed state dicts into one file.

    Keys are prefixed with 'endpoint.' and 'color.' so they can be
    cleanly separated at load time.
    """
    merged: Dict[str, torch.Tensor] = {}
    for k, v in endpoint_sd.items():
        merged["endpoint." + k] = v
    for k, v in color_sd.items():
        merged["color." + k] = v
    return merged


def main():
    cfg = CONFIG
    set_seed(int(cfg["seed"]))

    endpoint_dataset_json = Path(cfg["endpoint_dataset_json"]).expanduser().resolve()
    source_image = Path(cfg["source_image"]).expanduser().resolve()
    out_dir_ep = Path(cfg["out_dir_endpoint"]).expanduser().resolve()
    out_dir_co = Path(cfg["out_dir_color"]).expanduser().resolve()
    out_dir_merged = Path(cfg["out_dir_merged"]).expanduser().resolve()

    if not endpoint_dataset_json.exists():
        raise FileNotFoundError(f"Endpoint dataset JSON not found: {endpoint_dataset_json}")
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    out_dir_ep.mkdir(parents=True, exist_ok=True)
    out_dir_co.mkdir(parents=True, exist_ok=True)
    out_dir_merged.mkdir(parents=True, exist_ok=True)

    print("Device:", cfg["device"])
    print("AMP:", cfg["use_amp"], "| param_dtype:", cfg["param_dtype"])
    print("Paper schedule:", cfg["main_steps"], "+", int(round(cfg["main_steps"] * cfg["qat_tail_fraction"])), "QAT steps")

    endpoint_compressed_sd = None
    color_compressed_sd = None

    if bool(cfg["run_endpoint_training"]):
        print("\n====================\nTRAIN ENDPOINT NET\n====================")
        endpoint_compressed_sd = train_endpoint_network(cfg, endpoint_dataset_json, source_image, out_dir_ep)

    if bool(cfg["run_color_training"]):
        print("\n====================\nTRAIN COLOR NET\n====================")
        color_compressed_sd = train_color_network(cfg, endpoint_dataset_json, source_image, out_dir_co)

    # Merge both compressed state dicts into one file
    if endpoint_compressed_sd is not None and color_compressed_sd is not None:
        merged = merge_compressed_state_dicts(endpoint_compressed_sd, color_compressed_sd)
        merged_path = out_dir_merged / "ntbc_bc1_merged_compressed.pt"
        torch.save(merged, merged_path)

        total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
        print(f"\n[MERGED] Saved both networks into one file: {merged_path}")
        print(f"[MERGED] Total size: {total_bytes / 1024 / 1024:.2f} MB")

        # Clean up individual run directories now that the merged file exists
        import shutil
        for d in (out_dir_ep, out_dir_co):
            if d.exists():
                shutil.rmtree(d)
                print(f"[CLEANUP] Deleted: {d}")
    else:
        print("\n[WARN] Skipped merging: both networks must be trained to produce merged file.")


if __name__ == "__main__":
    main()
