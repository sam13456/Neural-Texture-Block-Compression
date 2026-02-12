"""
Trains the endpoint network using the paper's step-based approach.

20k main steps + 10% QAT tail = 22k total steps.
Samples random blocks each step instead of iterating through epochs.

Edit the CONFIG paths below and run this file.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from Network_endpoint import EndpointNetwork, endpoint_loss_bc1

CONFIG = {
    # >>> EDIT THESE PATHS <<<
    "dataset_json": r"D:\BC1 extract\Bricks090_diffuse\bc1_endpoint_dataset.json",
    "source_image": r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",
    "out_dir": r"D:\BC1 extract\Bricks090_diffuse\runs_endpoint_bc1_steps",

    "main_steps": 20_000,
    "qat_tail_fraction": 0.10,       # +10% extra steps after main_steps
    "warmup_steps": 10,
    "batch_size": 1024,

    "lr_grid": 1e-2,
    "lr_mlp": 5e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-15,

    "qat_bits": 8,
    "freeze_grids_during_qat": True,  # closer to “fine-tune MLPs with quantized grids”

    "filter_c0_gt_c1": False,  # recommended if you use the 4-color palette (Eq.7)

    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,            # mixed precision on CUDA
    "param_dtype": "float32",   

    "log_every_steps": 50,
    "save_every_steps": 2000,   
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_json(path: Path):
    d = json.loads(path.read_text())
    st = np.asarray(d["inputs"]["st"], dtype=np.float32)
    bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int32)
    y = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)
    c0_gt_c1 = np.asarray(d["flags"]["c0_gt_c1"], dtype=np.uint8)
    meta = d.get("meta", {})
    return st, bxby, y, c0_gt_c1, meta


def load_image_rgb_u8(image_path: Path) -> np.ndarray:
    """Loads image as uint8 RGB. Keeps uint8 to save memory."""
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    return arr


def pad_image_u8_to_blocks(img_hwc_u8: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    """
    Pad uint8 HWC image so it becomes exactly (blocks_y*4, blocks_x*4, 3),
    using edge replication (matches prior runner intent).
    """
    H, W, C = img_hwc_u8.shape
    target_h = blocks_y * 4
    target_w = blocks_x * 4
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    if pad_h == 0 and pad_w == 0:
        return img_hwc_u8

    # pad ((top,bottom),(left,right),(channel))
    out = np.pad(
        img_hwc_u8,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )
    return out


# Precompute 16 offsets for a 4x4 block in row-major order
_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)  # (16,)
_OFF_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)  # (16,)


def extract_block_colors_batch_u8(img_hwc_u8_t: torch.Tensor, bxby_batch: torch.Tensor) -> torch.Tensor:
    """
    Vectorized extraction of reference colors for a batch of blocks.
    img_hwc_u8_t: (H,W,3) uint8 CPU tensor
    bxby_batch:   (B,2) int64 CPU tensor with bx,by
    Returns:      (B,16,3) uint8 CPU tensor
    """
    bx = bxby_batch[:, 0]  # (B,)
    by = bxby_batch[:, 1]  # (B,)

    base_x = bx * 4  # (B,)
    base_y = by * 4  # (B,)

    x = base_x[:, None] + _OFF_X[None, :]  # (B,16)
    y = base_y[:, None] + _OFF_Y[None, :]  # (B,16)

    # Advanced indexing yields (B,16,3)
    patch_u8 = img_hwc_u8_t[y, x]  # uint8
    return patch_u8


def lr_scale_warmup_cos(step: int, total_steps: int, warmup_steps: int) -> float:
    """
    Paper: warmup 10, cosine to 0.
    """
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    t = float(step - warmup_steps) / float(denom)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def set_lrs(optimizer: torch.optim.Optimizer, lr_scale: float, base_lrs: Tuple[float, float], grid_lr_mul: float = 1.0):
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
    set_seed(int(cfg["seed"]))

    dataset_json = Path(cfg["dataset_json"]).expanduser().resolve()
    source_image = Path(cfg["source_image"]).expanduser().resolve()
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    if not source_image.exists():
        raise FileNotFoundError(
            f"Source image not found: {source_image}\n"
            "You MUST provide the original uncompressed image so L_cd can be computed."
        )

    device = str(cfg["device"])
    print("Device:", device)

    # --- Load dataset ---
    st_np, bxby_np, y_np, c0_gt_c1_np, meta = load_dataset_json(dataset_json)
    N = int(st_np.shape[0])

    blocks_x = int(meta.get("blocks_x", bxby_np[:, 0].max() + 1))
    blocks_y = int(meta.get("blocks_y", bxby_np[:, 1].max() + 1))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))
    print(f"Dataset: N={N} blocks, blocks=({blocks_x},{blocks_y}), tex={width}x{height}")

    if bool(cfg["filter_c0_gt_c1"]):
        valid_mask = (c0_gt_c1_np.astype(np.uint8) == 1)
        valid_idx = np.nonzero(valid_mask)[0].astype(np.int64)
        print(f"Filtering: keeping {valid_idx.size} / {N} blocks with c0_gt_c1==1")
    else:
        valid_idx = np.arange(N, dtype=np.int64)

    if valid_idx.size == 0:
        raise RuntimeError("No valid blocks after filtering. Check your dataset or filter setting.")

    st_t = torch.from_numpy(st_np)              # (N,2) float32 CPU
    y_t = torch.from_numpy(y_np)                # (N,6) float32 CPU
    bxby_t = torch.from_numpy(bxby_np.astype(np.int64))  # (N,2) int64 CPU
    valid_idx_t = torch.from_numpy(valid_idx)   # (M,) int64 CPU

    img_u8 = load_image_rgb_u8(source_image)  # (H,W,3) uint8
    img_u8 = pad_image_u8_to_blocks(img_u8, blocks_x, blocks_y)
    img_u8_t = torch.from_numpy(img_u8)       # (H,W,3) uint8 CPU
    print(f"Image loaded (padded): {img_u8_t.shape[1]}x{img_u8_t.shape[0]}")

    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32
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

    use_amp = bool(cfg["use_amp"]) and (device == "cuda")
    if use_amp and param_dtype == torch.float16:
        print("WARNING: param_dtype=float16 + GradScaler is unstable in vanilla PyTorch. "
              "Use param_dtype='float32' with autocast for a paper-like effect.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Step-based schedule (paper) ---
    main_steps = int(cfg["main_steps"])
    qat_tail_steps = int(round(main_steps * float(cfg["qat_tail_fraction"])))
    total_steps = main_steps + qat_tail_steps
    qat_start_step = main_steps  # enable QAT exactly after main training
    warmup_steps = int(cfg["warmup_steps"])
    batch_size = int(cfg["batch_size"])

    print(f"Train steps: main={main_steps}, qat_tail={qat_tail_steps} (fraction={cfg['qat_tail_fraction']:.2f}), total={total_steps}")
    print(f"QAT enables at step {qat_start_step} (0-indexed).")
    print(f"Batch size: {batch_size}. Sampling uniformly over {valid_idx_t.numel()} valid blocks.")

    # Logging accumulators
    log_every = int(cfg["log_every_steps"])
    save_every = int(cfg["save_every_steps"])
    t0 = time.time()
    run_total = run_le = run_lcd = 0.0
    run_count = 0

    # Freeze-grid multiplier (applied only during QAT tail if enabled)
    freeze_grids = bool(cfg["freeze_grids_during_qat"])
    grid_lr_mul = 1.0

    for step in range(total_steps):
        # Enable QAT exactly at boundary
        if step == qat_start_step and qat_tail_steps > 0:
            print(">>> Enabling grid QAT (fake quant) <<<")
            net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
            if freeze_grids:
                print(">>> Freezing grids during QAT tail (lr_grid -> 0) <<<")
                grid_lr_mul = 0.0

            # Save a boundary checkpoint (useful for debugging)
            save_checkpoint(out_dir, f"endpoint_net_bc1_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        # Sample batch indices (resolution-independent)
        pos = torch.randint(
            low=0,
            high=valid_idx_t.numel(),
            size=(batch_size,),
            device="cpu",
            dtype=torch.long,
        )
        bidx = valid_idx_t[pos]  # (B,) indices into dataset arrays on CPU

        coords = st_t[bidx].to(device=device, non_blocking=True)      # (B,2)
        ref_ep = y_t[bidx].to(device=device, non_blocking=True)       # (B,6)
        bxby_batch = bxby_t[bidx]                                     # (B,2) CPU

        # Extract reference colors for this batch, then move to device
        cols_u8 = extract_block_colors_batch_u8(img_u8_t, bxby_batch)  # (B,16,3) uint8 CPU
        cols = cols_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0

        # LR schedule
        lr_scale = lr_scale_warmup_cos(step, total_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, (float(cfg["lr_grid"]), float(cfg["lr_mlp"])), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = net(coords)
            loss_out = endpoint_loss_bc1(
                pred_endpoints6=pred,
                ref_endpoints6=ref_ep,
                ref_colors=cols,
                temperature=0.01,       # paper
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

        # Logging stats (running average over log window)
        run_total += float(loss_out.total.detach().cpu().item())
        run_le += float(loss_out.le.detach().cpu().item())
        run_lcd += float(loss_out.lcd.detach().cpu().item())
        run_count += 1

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            avg_total = run_total / run_count
            avg_le = run_le / run_count
            avg_lcd = run_lcd / run_count
            print(
                f"[step {step+1:06d}/{total_steps}] "
                f"loss={avg_total:.6f} (Le={avg_le:.6f}, Lcd={avg_lcd:.6f}) "
                f"lr_grid={optimizer.param_groups[0]['lr']:.5g} lr_mlp={optimizer.param_groups[1]['lr']:.5g} "
                f"time={dt:.1f}s"
            )
            run_total = run_le = run_lcd = 0.0
            run_count = 0
            t0 = time.time()

        # Periodic checkpointing
        if save_every > 0 and ((step + 1) % save_every == 0):
            save_checkpoint(out_dir, f"endpoint_net_bc1_step{step+1:06d}.pt", cfg, meta, net, optimizer, step + 1)

    # Final saves
    save_checkpoint(out_dir, "endpoint_net_bc1_final.pt", cfg, meta, net, optimizer, total_steps)
    final_sd = out_dir / "endpoint_net_bc1_final_state_dict.pt"
    torch.save(net.state_dict(), final_sd)
    print("Saved final model state_dict:", final_sd)


if __name__ == "__main__":
    main()
