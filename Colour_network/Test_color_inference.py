"""
Inference / verification for NTBC Color Network trained separately.

Reports:
- L_c metrics: MSE/MAE between predicted uncompressed color c_hat and original color c.
- "Decoded" metrics: choose BC1 indices by nearest palette color (using reference endpoints)
  and decode using those indices (paper-aligned for evaluating L_cd).

Also saves:
- pred_color.png (direct network RGB)
- decoded_from_ref_endpoints.png (BC1-like decoded using reference endpoints + predicted indices)

NOTE: This is NOT the full NTBC pipeline (which would use predicted endpoints + predicted indices).
This isolates the color network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from Network_color import ColorNetwork, endpoints6_to_e0e1, bc1_palette_from_endpoints, _BC1_W


CONFIG = {
    # Edit these
    "endpoint_dataset_json": r"D:\BC1 extract\Bricks090_diffuse\bc1_endpoint_dataset.json",
    "source_image": r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",

    # Load either:
    # 1) checkpoint with {"model_state": ...}
    "checkpoint_path": r"D:\BC1 extract\Bricks090_diffuse\runs_color_bc1_steps\color_net_bc1_final_state_dict.pt",

    # OR 2) pure state_dict
    # "checkpoint_path": r"D:\...\color_net_bc1_final_state_dict.pt",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,

    # batching
    "block_batch": 4096,  # blocks per batch (each block has 16 pixels)
}


def load_endpoint_dataset(path: Path):
    d = json.loads(path.read_text())
    bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int64)          # (N,2)
    ep = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)       # (N,6)
    meta = d.get("meta", {})
    return bxby, ep, meta


def load_image_rgb_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)  # HWC


def pad_image_to_blocks(img: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    H, W, C = img.shape
    target_h = blocks_y * 4
    target_w = blocks_x * 4
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


@torch.no_grad()
def load_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Determine state_dict + (optional) config
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    else:
        state = ckpt
        cfg = {}

    # Try to infer finest_resolution so the grid shapes match
    finest = cfg.get("grid_finest_resolution", None)
    if finest is None:
        # infer from last grid tensor in state_dict: shape is (r*r, feature_dim)
        grid_keys = [k for k in state.keys() if k.startswith("encoding.grids.")]
        if grid_keys:
            # pick the highest index grid
            def grid_idx(k: str) -> int:
                try:
                    return int(k.split(".")[2])
                except Exception:
                    return -1
            last_key = sorted(grid_keys, key=grid_idx)[-1]
            rr = state[last_key].shape[0]
            finest = int(round(math.sqrt(rr)))
        else:
            finest = 2048

    net = ColorNetwork(param_dtype=torch.float32, finest_resolution=int(finest)).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net



@torch.no_grad()
def decode_indices_and_colors(pred_color: torch.Tensor, ref_endpoints6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pred_color: (P,3)
    ref_endpoints6: (P,6)
    returns:
      hard_idx: (P,) uint8 in [0..3]
      decoded:  (P,3) decoded colors using reference endpoints + hard indices
    """
    e0, e1 = endpoints6_to_e0e1(ref_endpoints6)  # (P,3),(P,3)
    w = _BC1_W.to(device=pred_color.device, dtype=pred_color.dtype)
    pal = bc1_palette_from_endpoints(e0, e1, w=w)  # (P,4,3)

    diff = pred_color.unsqueeze(1) - pal  # (P,4,3)
    dn = -torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (P,4)
    hard = torch.argmax(dn, dim=-1).to(torch.uint8)

    w_h = w[hard.long()]
    decoded = (1.0 - w_h).unsqueeze(-1) * e0 + w_h.unsqueeze(-1) * e1
    return hard, decoded


def main():
    cfg = CONFIG

    endpoint_dataset_json = Path(cfg["endpoint_dataset_json"]).expanduser().resolve()
    source_image = Path(cfg["source_image"]).expanduser().resolve()
    checkpoint_path = Path(cfg["checkpoint_path"]).expanduser().resolve()

    if not endpoint_dataset_json.exists():
        raise FileNotFoundError(endpoint_dataset_json)
    if not source_image.exists():
        raise FileNotFoundError(source_image)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = str(cfg["device"])
    use_amp = bool(cfg["use_amp"]) and (device == "cuda")
    print("Device:", device)

    bxby_np, ep_np, meta = load_endpoint_dataset(endpoint_dataset_json)
    blocks_x = int(meta.get("blocks_x", bxby_np[:, 0].max() + 1))
    blocks_y = int(meta.get("blocks_y", bxby_np[:, 1].max() + 1))

    img_u8 = load_image_rgb_u8(source_image)
    img_u8 = pad_image_to_blocks(img_u8, blocks_x, blocks_y)
    H, W, _ = img_u8.shape
    print(f"Image (padded): {W}x{H}, blocks=({blocks_x},{blocks_y})")

    net = load_model(checkpoint_path, device=device)
    print("Loaded model:", checkpoint_path.name)

    # Torch CPU tensors
    img_u8_t = torch.from_numpy(img_u8)                    # (H,W,3) uint8 CPU
    bxby_t = torch.from_numpy(bxby_np)                     # (N,2) int64 CPU
    ep_t = torch.from_numpy(ep_np)                         # (N,6) float32 CPU

    # Precompute offsets for 4x4
    off_x = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)  # (16,)
    off_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)

    N = bxby_t.shape[0]
    block_batch = int(cfg["block_batch"])

    # Outputs (optional)
    pred_img = np.zeros((H, W, 3), dtype=np.uint8)
    dec_img = np.zeros((H, W, 3), dtype=np.uint8)

    # Metrics accumulators
    sum_se_pred = 0.0
    sum_ae_pred = 0.0
    sum_se_dec = 0.0
    sum_ae_dec = 0.0
    count = 0

    for start in range(0, N, block_batch):
        end = min(N, start + block_batch)
        b = end - start

        bxby = bxby_t[start:end]  # (b,2) CPU
        ep = ep_t[start:end]      # (b,6) CPU

        base_x = bxby[:, 0] * 4   # (b,)
        base_y = bxby[:, 1] * 4

        x = (base_x[:, None] + off_x[None, :]).clamp(0, W - 1)  # (b,16)
        y = (base_y[:, None] + off_y[None, :]).clamp(0, H - 1)  # (b,16)

        # ref colors (CPU) -> device float
        ref_u8 = img_u8_t[y, x]  # (b,16,3) uint8
        ref = (ref_u8.to(torch.float32) / 255.0).to(device=device, non_blocking=True)  # (b,16,3)

        # UV coords -> (b*16,2)
        u = (x.to(torch.float32) / float(max(1, W - 1))).to(device=device, non_blocking=True)
        v = (y.to(torch.float32) / float(max(1, H - 1))).to(device=device, non_blocking=True)
        uv = torch.stack([u, v], dim=-1).reshape(-1, 2)  # (p,2) where p=b*16

        ep_rep = ep.to(device=device, non_blocking=True).unsqueeze(1).expand(-1, 16, -1).reshape(-1, 6)  # (p,6)
        ref_flat = ref.reshape(-1, 3)  # (p,3)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred_flat = net(uv).to(torch.float32)  # (p,3)

        hard_idx, dec_flat = decode_indices_and_colors(pred_flat, ep_rep)

        # Metrics
        diff_p = pred_flat - ref_flat
        diff_d = dec_flat - ref_flat
        sum_se_pred += float((diff_p * diff_p).sum().cpu().item())
        sum_ae_pred += float(diff_p.abs().sum().cpu().item())
        sum_se_dec += float((diff_d * diff_d).sum().cpu().item())
        sum_ae_dec += float(diff_d.abs().sum().cpu().item())
        count += int(ref_flat.numel())  # 3 * pixels

        # Write images back (CPU)
        pred_u8 = (pred_flat.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().reshape(b, 16, 3)
        dec_u8 = (dec_flat.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().reshape(b, 16, 3)

        # Scatter into output images
        # Flatten coordinates for assignment
        x_cpu = x.numpy().reshape(-1)
        y_cpu = y.numpy().reshape(-1)
        pred_img[y_cpu, x_cpu] = pred_u8.numpy().reshape(-1, 3)
        dec_img[y_cpu, x_cpu] = dec_u8.numpy().reshape(-1, 3)

        if (start // block_batch) % 10 == 0:
            print(f"Processed blocks {start}..{end-1} / {N}")

    # Final metrics
    num_pixels = (count // 3)
    mse_pred = sum_se_pred / count
    mae_pred = sum_ae_pred / count
    mse_dec = sum_se_dec / count
    mae_dec = sum_ae_dec / count

    def psnr(mse: float) -> float:
        if mse <= 0:
            return 99.0
        import math
        return 10.0 * math.log10(1.0 / mse)

    print("\n============================")
    print("COLOR NETWORK METRICS")
    print("============================")
    print(f"Pixels: {num_pixels:,}")
    print(f"L_c  (pred vs ref)  MSE: {mse_pred:.8f}  MAE: {mae_pred:.8f}  PSNR: {psnr(mse_pred):.2f} dB")
    print(f"L_cd (decoded vs ref) MSE: {mse_dec:.8f}  MAE: {mae_dec:.8f}  PSNR: {psnr(mse_dec):.2f} dB")

    out_dir = checkpoint_path.parent
    pred_path = out_dir / "pred_color.png"
    dec_path = out_dir / "decoded_from_ref_endpoints.png"
    Image.fromarray(pred_img, mode="RGB").save(pred_path)
    Image.fromarray(dec_img, mode="RGB").save(dec_path)
    print("\nSaved:", pred_path)
    print("Saved:", dec_path)


if __name__ == "__main__":
    main()