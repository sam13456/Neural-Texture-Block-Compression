"""
Train NTBC Color Network (BC1) separately (paper-aligned).

Core paper facts:
- Color network input: texture coords (u,v) in [0,1] encoded via 8-level multi-res grids (finest 2048) (Sec. 3.4)
- MLP: 3 hidden layers, 64 neurons, selu, sigmoid output (Sec. 3.4)
- Loss: L_color = L_c + L_cd (Eq. 12)
  * Use REFERENCE endpoints (Compressonator) + predicted colors to reconstruct index (Eq. 9–10) (Sec. 3.2)
- STE through argmax, softmax temperature T=0.01 (Eq. 13 / Sec. 3.4)
- QAT: train in half precision first, then +10% steps with grid fake-quant (Sec. 3.3)

This runner:
- exactly 20k main steps + 10% QAT tail (22k total by default)
- resolution-independent: samples random texels using block indices from the endpoint dataset.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch

from Network_color import ColorNetwork, color_loss_bc1

CONFIG = {
    # >>> EDIT THESE PATHS <<<
    # Endpoint dataset JSON produced by Endpoints_Extract.py (must include bxby + endpoints)
    "endpoint_dataset_json": r"D:\BC1 extract\Bricks090_diffuse\bc1_endpoint_dataset.json",
    # Original uncompressed image (RGB)
    "source_image": r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",
    # Output folder for checkpoints/logs
    "out_dir": r"D:\BC1 extract\Bricks090_diffuse\runs_color_bc1_steps",

    # Training steps
    "main_steps": 20_000,
    "qat_tail_fraction": 0.10,
    "warmup_steps": 10,

    # Batch size = number of texels sampled per step
    "batch_size": 65_536,

    # Grid finest resolution. Paper uses 2048 for 2K textures.
    # Set to None to auto-use max(image_width, image_height).
    "grid_finest_resolution": 2048,


    # Optim
    "lr_grid": 1e-2,
    "lr_mlp": 5e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-15,

    # Softmax temp (paper uses 0.01)
    "temperature": 0.01,

    # QAT
    "qat_bits": 8,
    "freeze_grids_during_qat": True,

    # If you are following your 4-color-only simplification:
    # True  => only sample blocks where c0>c1 in the reference BC1
    # False => sample all blocks (recommended if you handle both modes)
    "filter_c0_gt_c1": False,

    # Mixed precision
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "param_dtype": "float32",  # keep float32 params for stability; use AMP for speed

    # I/O
    "log_every_steps": 50,
    "save_every_steps": 2000,
}


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


def load_endpoint_dataset(path: Path):
    """
    Reads bc1_endpoint_dataset.json:
      inputs: st (unused here), bxby
      targets: ep_q01 (r0,g0,b0,r1,g1,b1 in [0,1])
      flags: c0_gt_c1 (optional)
      meta: blocks_x, blocks_y, width, height (optional)
    """
    d = json.loads(path.read_text())
    bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int64)          # (N,2)
    ep = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)       # (N,6)
    flags = d.get("flags", {})
    c0_gt_c1 = np.asarray(flags.get("c0_gt_c1", np.ones((bxby.shape[0],), dtype=np.uint8)), dtype=np.uint8)
    meta = d.get("meta", {})
    return bxby, ep, c0_gt_c1, meta


def lr_scale_warmup_cos(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    t = float(step - warmup_steps) / float(denom)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def set_lrs(optimizer: torch.optim.Optimizer, lr_scale: float, base_lrs: Tuple[float, float], grid_lr_mul: float):
    lr_grid, lr_mlp = base_lrs
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


def main():
    cfg = CONFIG

    endpoint_dataset_json = Path(cfg["endpoint_dataset_json"]).expanduser().resolve()
    source_image = Path(cfg["source_image"]).expanduser().resolve()
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not endpoint_dataset_json.exists():
        raise FileNotFoundError(f"Endpoint dataset JSON not found: {endpoint_dataset_json}")
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    device = str(cfg["device"])
    print("Device:", device)

    bxby_np, ep_np, c0_gt_c1_np, meta = load_endpoint_dataset(endpoint_dataset_json)
    N = int(bxby_np.shape[0])
    blocks_x = int(meta.get("blocks_x", bxby_np[:, 0].max() + 1))
    blocks_y = int(meta.get("blocks_y", bxby_np[:, 1].max() + 1))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))
    print(f"Endpoint dataset: N={N}, blocks=({blocks_x},{blocks_y}), tex={width}x{height}")

    # Build a list of valid blocks (indices into bxby/ep arrays)
    if bool(cfg["filter_c0_gt_c1"]):
        valid_idx = np.nonzero(c0_gt_c1_np.astype(np.uint8) == 1)[0].astype(np.int64)
        print(f"Filtering c0>c1: keeping {valid_idx.size}/{N} blocks")
    else:
        valid_idx = np.arange(N, dtype=np.int64)

    if valid_idx.size == 0:
        raise RuntimeError("No valid blocks after filtering.")

    # Load and pad image to block grid so block sampling is always in-bounds
    img_u8 = load_image_rgb_u8(source_image)
    img_u8 = pad_image_to_blocks(img_u8, blocks_x, blocks_y)
    H, W, _ = img_u8.shape
    print(f"Image loaded (padded): {W}x{H}")

    # Torch CPU copies
    img_u8_t = torch.from_numpy(img_u8)                     # (H,W,3) uint8 CPU
    bxby_t = torch.from_numpy(bxby_np)                      # (N,2) int64 CPU
    ep_t = torch.from_numpy(ep_np)                          # (N,6) float32 CPU
    valid_idx_t = torch.from_numpy(valid_idx)               # (M,) int64 CPU

    # Create network
    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32
    finest = cfg.get("grid_finest_resolution", 2048)
    if finest is None:
        finest = int(max(W, H))
    net = ColorNetwork(param_dtype=param_dtype, finest_resolution=int(finest)).to(device)
    net.train()

    optimizer = torch.optim.Adam(
        [
            {"params": net.encoding.parameters(), "lr": float(cfg["lr_grid"])},
            {"params": net.mlp.parameters(), "lr": float(cfg["lr_mlp"])},
        ],
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
    )

    use_amp = bool(cfg["use_amp"]) and (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    main_steps = int(cfg["main_steps"])
    qat_tail_steps = int(round(main_steps * float(cfg["qat_tail_fraction"])))
    total_steps = main_steps + qat_tail_steps
    qat_start_step = main_steps
    warmup_steps = int(cfg["warmup_steps"])
    batch_size = int(cfg["batch_size"])
    temperature = float(cfg["temperature"])

    print(f"Train steps: main={main_steps}, qat_tail={qat_tail_steps}, total={total_steps}")
    print(f"QAT enables at step {qat_start_step} (0-indexed)")
    print(f"Batch size: {batch_size} texels per step, sampled from {valid_idx_t.numel()} blocks")

    log_every = int(cfg["log_every_steps"])
    save_every = int(cfg["save_every_steps"])

    run_total = run_lc = run_lcd = 0.0
    run_count = 0
    t0 = time.time()

    freeze_grids = bool(cfg["freeze_grids_during_qat"])
    grid_lr_mul = 1.0

    # Precompute random 4x4 offsets each step cheaply on GPU
    for step in range(total_steps):
        if step == qat_start_step and qat_tail_steps > 0:
            print(">>> Enabling grid QAT (fake quant) <<<")
            net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
            if freeze_grids:
                print(">>> Freezing grids during QAT tail (lr_grid -> 0) <<<")
                grid_lr_mul = 0.0
            save_checkpoint(out_dir, f"color_net_bc1_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        # Sample blocks, then sample a texel within each block (uniform within 4x4)
        # 1) choose block rows in dataset
        pos = torch.randint(0, valid_idx_t.numel(), (batch_size,), device="cpu", dtype=torch.long)
        didx = valid_idx_t[pos]                 # dataset indices (CPU)
        bxby = bxby_t[didx]                     # (B,2) CPU
        ref_ep = ep_t[didx]                     # (B,6) CPU

        # 2) offsets inside block
        ox = torch.randint(0, 4, (batch_size,), device="cpu", dtype=torch.long)
        oy = torch.randint(0, 4, (batch_size,), device="cpu", dtype=torch.long)

        px = bxby[:, 0] * 4 + ox  # (B,) pixel x
        py = bxby[:, 1] * 4 + oy  # (B,) pixel y

        # Safety clamp (should be unnecessary after padding, but keep safe)
        px = px.clamp(0, W - 1)
        py = py.clamp(0, H - 1)

        # Reference RGB (uint8 -> float)
        ref_rgb_u8 = img_u8_t[py, px]  # (B,3) uint8 CPU
        ref_rgb = (ref_rgb_u8.to(torch.float32) / 255.0).to(device=device, non_blocking=True)  # (B,3)

        # UV coords in [0,1] (paper uses normalized texture coords)
        # Use (W-1,H-1) so corners map to exactly 0 and 1; clamp in encoding avoids 1.0 overflow.
        u = (px.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (py.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=1)  # (B,2)

        ref_ep = ref_ep.to(device=device, dtype=torch.float32, non_blocking=True)

        # LR schedule
        lr_scale = lr_scale_warmup_cos(step, total_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, (float(cfg["lr_grid"]), float(cfg["lr_mlp"])), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = net(uv)  # (B,3)
            loss_out = color_loss_bc1(
                pred_color=pred,
                ref_color=ref_rgb,
                ref_endpoints6=ref_ep,
                temperature=temperature,
                reduction="mean",
            )
            loss = loss_out.total

        if use_amp:
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
                f"[step {step+1:06d}/{total_steps}] "
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
    final_sd = out_dir / "color_net_bc1_final_state_dict.pt"
    torch.save(net.state_dict(), final_sd)
    print("Saved final model state_dict:", final_sd)


if __name__ == "__main__":
    main()
