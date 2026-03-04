"""
V-NTBC: Train both BC4 networks SEPARATELY with VBQ.
(MULTI-single-channel-texture capable, BC4-only)

BC4 compresses single-channel data. This script:
- Loads grayscale source images (or converts RGB to grayscale)
- Trains endpoint network (predicts 2*T scalar endpoints per block)
- Trains color network (predicts 1*T scalar per texel)

V-NTBC enhancements over base NTBC BC4:
  - VBQ: per-level quantization bits via bits_list during QAT tail
  - Residual fusion grids (handled inside Network_endpoint / Network_color)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

import torch

from Network_endpoint import EndpointNetwork, endpoint_loss_bc4_multi
from Network_color import ColorNetwork, color_loss_bc4_multi

from Model_param_compress import compress_state_dict, print_size_comparison


# =========================
# CONFIG
# =========================
from config import (
    TRAIN_DATASET_JSON, SOURCE_IMAGES,
    OUT_DIR_ENDPOINT, OUT_DIR_COLOR, MODEL_DIR,
    QAT_BITS_ENDPOINT, QAT_BITS_COLOR,
    USE_LPE, LPE_N, LPE_N_FREQ, LPE_D0,
)

CONFIG = {
    "endpoint_dataset_json": TRAIN_DATASET_JSON,
    "source_images": SOURCE_IMAGES,

    "run_endpoint_training": True,
    "run_color_training": True,

    "out_dir_endpoint": OUT_DIR_ENDPOINT,
    "out_dir_color": OUT_DIR_COLOR,
    "out_dir_merged": MODEL_DIR,

    # Training schedule
    "main_steps": 20000,
    "qat_tail_fraction": 0.50,
    "warmup_steps": 10,
    "qat_warmup_steps": 10,

    # Optimizer
    "lr_grid": 1.3e-2,
    "lr_mlp": 5e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-15,

    # STE temperature
    "temperature": 0.01,

    # VBQ: per-level quantization bits
    "qat_bits_endpoint": QAT_BITS_ENDPOINT,
    "qat_bits_color": QAT_BITS_COLOR,
    "freeze_grids_during_qat": False,

    # Local Positional Encoding (LPE)
    "use_lpe": USE_LPE,
    "lpe_N": LPE_N,
    "lpe_n_freq": LPE_N_FREQ,
    "lpe_d0": LPE_D0,

    # Batch sizes
    "batch_size_blocks": 5096,
    "batch_size_texels": 131072,

    # Mixed precision
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "param_dtype": "float32",

    # Logging / saving
    "log_every_steps": 100,
    "save_every_steps": 5000,

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


def load_image_grayscale_u8(path: Path) -> np.ndarray:
    """Load image as single-channel grayscale uint8. (H,W)"""
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def pad_image_to_blocks(img: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    H, W = img.shape[:2]
    target_h = blocks_y * 4
    target_w = blocks_x * 4
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    if pad_h == 0 and pad_w == 0:
        return img
    if img.ndim == 2:
        return np.pad(img, pad_width=((0, pad_h), (0, pad_w)), mode="edge")
    return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def load_images_stack_grayscale_u8(paths: List[Path], blocks_x: int, blocks_y: int) -> torch.Tensor:
    """Load and stack grayscale images. Returns (T,H,W) uint8 tensor."""
    imgs = []
    W0 = H0 = None
    for p in paths:
        arr = load_image_grayscale_u8(p)
        arr = pad_image_to_blocks(arr, blocks_x, blocks_y)
        H, W = arr.shape
        if W0 is None:
            W0, H0 = W, H
        else:
            if W != W0 or H != H0:
                raise ValueError(f"All source images must match after padding. {p} -> {W}x{H}, expected {W0}x{H0}")
        imgs.append(torch.from_numpy(arr))
    return torch.stack(imgs, dim=0)  # (T,H,W) uint8 CPU


# 16 offsets for 4x4 block
_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
_OFF_Y = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
], dtype=torch.long)


def extract_block_values_batch_u8_multi(imgs_thw_u8: torch.Tensor, bxby_batch: torch.Tensor) -> torch.Tensor:
    """imgs_thw_u8: (T,H,W) uint8; bxby_batch: (B,2) -> (B,T,16) uint8"""
    dev = bxby_batch.device
    bx = bxby_batch[:, 0]
    by = bxby_batch[:, 1]
    base_x = bx * 4
    base_y = by * 4
    off_x = _OFF_X.to(dev)
    off_y = _OFF_Y.to(dev)
    x = base_x[:, None] + off_x[None, :]
    y = base_y[:, None] + off_y[None, :]

    # imgs[:, y, x] -> (T,B,16)
    out = imgs_thw_u8[:, y, x]
    return out.permute(1, 0, 2).contiguous()  # (B,T,16)


# =========================
# Dataset loading
# =========================

def load_endpoint_dataset_json(path: Path):
    d = json.loads(path.read_text())
    bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int64)
    ep = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)
    meta = d.get("meta", {})
    return bxby, ep, meta


def infer_num_textures(ep_np: np.ndarray, meta: dict) -> int:
    if "num_textures" in meta:
        return int(meta["num_textures"])
    if ep_np.ndim != 2 or ep_np.shape[1] % 2 != 0:
        raise ValueError(f"Cannot infer num_textures from ep_q01 shape {ep_np.shape}")
    return int(ep_np.shape[1] // 2)


def resolve_source_images(cfg: dict, meta: dict) -> List[Path]:
    srcs = cfg.get("source_images") or []
    if srcs:
        return [Path(s).expanduser().resolve() for s in srcs]
    meta_srcs = meta.get("source_images") or []
    if not meta_srcs:
        raise ValueError("No source images found.")
    return [Path(s).expanduser().resolve() for s in meta_srcs]


# =========================
# Training: Endpoint network
# =========================

def train_endpoint_network(cfg: dict, endpoint_dataset_json: Path, out_dir: Path):
    device = str(cfg["device"])
    use_amp = bool(cfg["use_amp"]) and device.startswith("cuda")
    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32

    bxby_np, ep_np, meta = load_endpoint_dataset_json(endpoint_dataset_json)
    N = int(bxby_np.shape[0])

    blocks_x = int(meta.get("blocks_x", int(bxby_np[:, 0].max() + 1)))
    blocks_y = int(meta.get("blocks_y", int(bxby_np[:, 1].max() + 1)))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))

    T = infer_num_textures(ep_np, meta)
    print(f"[Endpoint] Dataset: N={N} blocks, blocks=({blocks_x},{blocks_y}), tex={width}x{height}, T={T}")

    st_np = np.zeros((N, 2), dtype=np.float32)
    st_np[:, 0] = bxby_np[:, 0] / max(blocks_x - 1, 1)
    st_np[:, 1] = bxby_np[:, 1] / max(blocks_y - 1, 1)

    st_t = torch.from_numpy(st_np).to(device)
    bxby_t = torch.from_numpy(bxby_np).to(device)
    ep_t = torch.from_numpy(ep_np).to(device)

    src_paths = resolve_source_images(cfg, meta)
    for p in src_paths:
        if not p.exists():
            raise FileNotFoundError(f"Source image not found: {p}")
    # Load as grayscale: (T,H,W) uint8
    imgs_thw_u8 = load_images_stack_grayscale_u8(src_paths, blocks_x, blocks_y).to(device)
    H, W = int(imgs_thw_u8.shape[1]), int(imgs_thw_u8.shape[2])
    print(f"[Endpoint] Loaded {len(src_paths)} grayscale images (padded): {W}x{H}, all data on {device}")

    net = EndpointNetwork(
        num_textures=T,
        param_dtype=param_dtype,
        use_lpe=bool(cfg.get("use_lpe", False)),
        lpe_N=int(cfg.get("lpe_N", 128)),
        lpe_n_freq=int(cfg.get("lpe_n_freq", 4)),
        lpe_d0=int(cfg.get("lpe_d0", 8)),
    ).to(device)
    net.train()

    opt_params = [
        {"params": net.encoding.parameters(), "lr": float(cfg["lr_grid"])},
        {"params": net.mlp.parameters(), "lr": float(cfg["lr_mlp"])},
    ]
    if getattr(net, "use_lpe", False) and net.lpe is not None:
        opt_params.append({"params": net.lpe.parameters(), "lr": float(cfg["lr_grid"])})

    optimizer = torch.optim.Adam(
        opt_params,
        betas=tuple(cfg["betas"]), eps=float(cfg["eps"]))
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

    qat_warmup_steps = int(cfg.get("qat_warmup_steps", 10))
    in_qat = False

    print(f"[Endpoint] Train steps: main={main_steps}, qat_tail={qat_tail_steps}, total={total_steps}")

    run_total = run_le = run_lcd = 0.0
    run_count = 0
    t_train_start = time.time()
    t0 = time.time()

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for step in range(total_steps):
        if step == qat_start_step and qat_tail_steps > 0:
            print("[Endpoint] >>> Enabling grid VBQ QAT <<<")
            net.encoding.enable_qat(bits_list=cfg["qat_bits_endpoint"])
            in_qat = True
            if freeze_grids:
                grid_lr_mul = 0.0
            save_checkpoint(out_dir, f"endpoint_net_bc4_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        didx = torch.randint(0, N, (batch_size,), device=device, dtype=torch.long)

        st = st_t[didx]
        bxby = bxby_t[didx]
        ref_ep = ep_t[didx]  # (B, 2*T)

        # (B,T,16) uint8 -> float
        ref_vals_u8 = extract_block_values_batch_u8_multi(imgs_thw_u8, bxby)
        ref_vals = ref_vals_u8.to(torch.float32) / 255.0

        if in_qat:
            qat_local_step = step - qat_start_step
            lr_scale = lr_scale_warmup_cos(qat_local_step, qat_tail_steps, qat_warmup_steps)
        else:
            lr_scale = lr_scale_warmup_cos(step, main_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, float(cfg["lr_grid"]), float(cfg["lr_mlp"]), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            pred_ep = net(st)  # (B, 2*T)
            loss_out = endpoint_loss_bc4_multi(
                pred_endpoints=pred_ep,
                ref_endpoints=ref_ep,
                ref_values=ref_vals,
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
            save_checkpoint(out_dir, f"endpoint_net_bc4_step{step+1:06d}.pt", cfg, meta, net, optimizer, step + 1)

    save_checkpoint(out_dir, "endpoint_net_bc4_final.pt", cfg, meta, net, optimizer, total_steps)

    elapsed = time.time() - t_train_start
    mins, secs = divmod(elapsed, 60)
    print(f"[Endpoint] Training complete in {int(mins)}m {secs:.1f}s")

    original_sd = net.state_dict()
    compressed_sd = compress_state_dict(original_sd, bits_list=cfg["qat_bits_endpoint"])
    print("[Endpoint] Compressed state_dict:")
    print_size_comparison(original_sd, compressed_sd)
    return compressed_sd


# =========================
# Training: Color network
# =========================

def train_color_network(cfg: dict, endpoint_dataset_json: Path, out_dir: Path):
    device = str(cfg["device"])
    use_amp = bool(cfg["use_amp"]) and device.startswith("cuda")
    param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32

    bxby_np, ep_np, meta = load_endpoint_dataset_json(endpoint_dataset_json)
    N = int(bxby_np.shape[0])

    blocks_x = int(meta.get("blocks_x", int(bxby_np[:, 0].max() + 1)))
    blocks_y = int(meta.get("blocks_y", int(bxby_np[:, 1].max() + 1)))
    width = int(meta.get("width", blocks_x * 4))
    height = int(meta.get("height", blocks_y * 4))

    T = infer_num_textures(ep_np, meta)
    print(f"[Color] Dataset: N={N}, blocks=({blocks_x},{blocks_y}), tex={width}x{height}, T={T}")

    bxby_t = torch.from_numpy(bxby_np.astype(np.int64)).to(device)
    ep_t = torch.from_numpy(ep_np.astype(np.float32)).to(device)

    src_paths = resolve_source_images(cfg, meta)
    for p in src_paths:
        if not p.exists():
            raise FileNotFoundError(f"Source image not found: {p}")
    imgs_thw_u8 = load_images_stack_grayscale_u8(src_paths, blocks_x, blocks_y).to(device)
    H, W = int(imgs_thw_u8.shape[1]), int(imgs_thw_u8.shape[2])
    print(f"[Color] Loaded {len(src_paths)} grayscale images (padded): {W}x{H}, all data on {device}")

    net = ColorNetwork(
        num_textures=T,
        param_dtype=param_dtype,
        finest_resolution=2048,
        use_lpe=bool(cfg.get("use_lpe", False)),
        lpe_N=int(cfg.get("lpe_N", 128)),
        lpe_n_freq=int(cfg.get("lpe_n_freq", 4)),
        lpe_d0=int(cfg.get("lpe_d0", 8)),
    ).to(device)
    net.train()

    opt_params = [
        {"params": net.encoding.parameters(), "lr": float(cfg["lr_grid"])},
        {"params": net.mlp.parameters(), "lr": float(cfg["lr_mlp"])},
    ]
    if getattr(net, "use_lpe", False) and net.lpe is not None:
        opt_params.append({"params": net.lpe.parameters(), "lr": float(cfg["lr_grid"])})

    optimizer = torch.optim.Adam(
        opt_params,
        betas=tuple(cfg["betas"]), eps=float(cfg["eps"]))
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

    qat_warmup_steps = int(cfg.get("qat_warmup_steps", 10))
    in_qat = False

    print(f"[Color] Train steps: main={main_steps}, qat_tail={qat_tail_steps}, total={total_steps}")

    run_total = run_lc = run_lcd = 0.0
    run_count = 0
    t_train_start = time.time()
    t0 = time.time()

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for step in range(total_steps):
        if step == qat_start_step and qat_tail_steps > 0:
            print("[Color] >>> Enabling grid VBQ QAT <<<")
            net.encoding.enable_qat(bits_list=cfg["qat_bits_color"])
            in_qat = True
            if freeze_grids:
                grid_lr_mul = 0.0
            save_checkpoint(out_dir, f"color_net_bc4_step{step:06d}_qat_start.pt", cfg, meta, net, optimizer, step)

        didx = torch.randint(0, N, (batch_size,), device=device, dtype=torch.long)

        bxby = bxby_t[didx]
        ref_ep = ep_t[didx]  # (B, 2*T)

        ox = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)
        oy = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)

        px = (bxby[:, 0] * 4 + ox).clamp(0, W - 1)
        py = (bxby[:, 1] * 4 + oy).clamp(0, H - 1)

        # Reference per-texel grayscale values for ALL textures
        # imgs_thw_u8: (T,H,W), indexing: (T,B) -> permute to (B,T)
        ref_val_u8 = imgs_thw_u8[:, py, px]  # (T,B)
        ref_val = ref_val_u8.permute(1, 0).to(torch.float32) / 255.0  # (B,T)

        u = px.to(torch.float32) / float(max(1, W - 1))
        v = py.to(torch.float32) / float(max(1, H - 1))
        uv = torch.stack([u, v], dim=1)

        if in_qat:
            qat_local_step = step - qat_start_step
            lr_scale = lr_scale_warmup_cos(qat_local_step, qat_tail_steps, qat_warmup_steps)
        else:
            lr_scale = lr_scale_warmup_cos(step, main_steps, warmup_steps)
        set_lrs(optimizer, lr_scale, float(cfg["lr_grid"]), float(cfg["lr_mlp"]), grid_lr_mul=grid_lr_mul)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            pred = net(uv)  # (B,T)
            loss_out = color_loss_bc4_multi(
                pred_values=pred,
                ref_values=ref_val,
                ref_endpoints=ref_ep,
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
            save_checkpoint(out_dir, f"color_net_bc4_step{step+1:06d}.pt", cfg, meta, net, optimizer, step + 1)

    save_checkpoint(out_dir, "color_net_bc4_final.pt", cfg, meta, net, optimizer, total_steps)

    elapsed = time.time() - t_train_start
    mins, secs = divmod(elapsed, 60)
    print(f"[Color] Training complete in {int(mins)}m {secs:.1f}s")

    original_sd = net.state_dict()
    compressed_sd = compress_state_dict(original_sd, bits_list=cfg["qat_bits_color"])
    print("[Color] Compressed state_dict:")
    print_size_comparison(original_sd, compressed_sd)
    return compressed_sd


# =========================
# Main
# =========================

def merge_compressed_state_dicts(endpoint_sd: Dict[str, torch.Tensor], color_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in endpoint_sd.items():
        merged["endpoint." + k] = v
    for k, v in color_sd.items():
        merged["color." + k] = v
    return merged


def main():
    t_total_start = time.time()
    cfg = CONFIG
    set_seed(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True

    endpoint_dataset_json = Path(cfg["endpoint_dataset_json"]).expanduser().resolve()
    out_dir_ep = Path(cfg["out_dir_endpoint"]).expanduser().resolve()
    out_dir_co = Path(cfg["out_dir_color"]).expanduser().resolve()
    out_dir_merged = Path(cfg["out_dir_merged"]).expanduser().resolve()

    if not endpoint_dataset_json.exists():
        raise FileNotFoundError(f"Endpoint dataset JSON not found: {endpoint_dataset_json}")

    out_dir_ep.mkdir(parents=True, exist_ok=True)
    out_dir_co.mkdir(parents=True, exist_ok=True)
    out_dir_merged.mkdir(parents=True, exist_ok=True)

    print("Device:", cfg["device"])
    print("AMP:", cfg["use_amp"], "| param_dtype:", cfg["param_dtype"])

    endpoint_sd = None
    color_sd = None

    if bool(cfg["run_endpoint_training"]):
        print("\n====================\nTRAIN ENDPOINT NET (V-NTBC BC4)\n====================")
        endpoint_sd = train_endpoint_network(cfg, endpoint_dataset_json, out_dir_ep)

    if bool(cfg["run_color_training"]):
        print("\n====================\nTRAIN COLOR NET (V-NTBC BC4)\n====================")
        color_sd = train_color_network(cfg, endpoint_dataset_json, out_dir_co)

    if endpoint_sd is not None and color_sd is not None:
        merged = merge_compressed_state_dicts(endpoint_sd, color_sd)
        merged_path = out_dir_merged / "v_ntbc_bc4_merged_compressed.pt"
        torch.save(merged, merged_path)

        total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
        print(f"\n[MERGED] Saved: {merged_path}")
        print(f"[MERGED] Total size: {total_bytes / 1024 / 1024:.2f} MB")

        import shutil
        for d in (out_dir_ep, out_dir_co):
            if d.exists():
                shutil.rmtree(d)
                print(f"[CLEANUP] Deleted: {d}")
    else:
        print("\n[WARN] Skipped merging: train both networks to produce merged file.")

    total_elapsed = time.time() - t_total_start
    mins, secs = divmod(total_elapsed, 60)
    print(f"\n[TOTAL] Pipeline complete in {int(mins)}m {secs:.1f}s")


if __name__ == "__main__":
    main()
