"""
Runs inference on the endpoint network and compares predictions with ground truth.

This uses the dense multi-resolution grid from Network_paper.py to match the trained checkpoint.
Just update the dataset and model paths below, then run it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch


from Network_paper import EndpointNetwork


DATASET_PATH = r"D:\BC1 extract\Bricks090_diffuse\bc1_endpoint_dataset.json"
MODEL_PATH   = r"D:\BC1 extract\Bricks090_diffuse\runs_endpoint_bc1_steps\endpoint_net_bc1_step022000.pt"


def load_dataset(json_path: str):
    d = json.loads(Path(json_path).read_text())
    coords = torch.tensor(d["inputs"]["st"], dtype=torch.float32)
    targets_q01 = torch.tensor(d["targets"]["ep_q01"], dtype=torch.float32)
    targets_rgb565 = torch.tensor(d["targets"]["ep_rgb565"], dtype=torch.int64)
    return coords, targets_q01, targets_rgb565


def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (int(r), int(g), int(b))


def load_model(model_path: str, device: torch.device):
    raw = torch.load(model_path, map_location=device)
    if isinstance(raw, dict) and "model_state" in raw:
        state_dict = raw["model_state"]
    else:
        state_dict = raw

    model = EndpointNetwork(
        num_levels=7,
        base_resolution=16,
        finest_resolution=1024,
        feature_dim=2,
        hidden_dim=64,
        num_hidden_layers=3,
        param_dtype=torch.float32,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("NTBC Endpoint Network - Inference Test")
    print("=" * 80)
    print(f"Device: {device}\n")

    coords, targets_q01, targets_rgb565 = load_dataset(DATASET_PATH)
    print(f"Dataset size: {coords.shape[0]} blocks\n")

    print("Loading model...")
    model = load_model(MODEL_PATH, device)
    print("Model loaded.\n")

    coords = coords.to(device)
    targets_q01 = targets_q01.to(device)
    targets_rgb565 = targets_rgb565.to(device)

    PRINT_TO_CONSOLE = False
    SAVE_CSV_REPORT = True
    REPORT_PATH = Path(MODEL_PATH).parent / "all_blocks_report.csv"

    with torch.no_grad():
        # Predictions in batches
        preds = model(coords)
        preds565 = model.predict_rgb565(coords).to(torch.int64)

        if SAVE_CSV_REPORT:
            import csv
            report_path = REPORT_PATH
            report_path.parent.mkdir(parents=True, exist_ok=True)

            coords_cpu = coords.detach().cpu()
            preds565_cpu = preds565.detach().cpu()
            targets565_cpu = targets_rgb565.detach().cpu()

            with report_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "block_idx",
                    "s", "t",
                    "pred_c0_dec", "pred_c1_dec",
                    "gt_c0_dec", "gt_c1_dec",
                    "match_both",
                    "pred_c0_rgb888", "pred_c1_rgb888",
                    "gt_c0_rgb888", "gt_c1_rgb888",
                ])

                for i in range(coords_cpu.shape[0]):
                    s = float(coords_cpu[i, 0].item())
                    t = float(coords_cpu[i, 1].item())

                    pc0 = int(preds565_cpu[i, 0].item())
                    pc1 = int(preds565_cpu[i, 1].item())
                    gc0 = int(targets565_cpu[i, 0].item())
                    gc1 = int(targets565_cpu[i, 1].item())

                    match = int((pc0 == gc0) and (pc1 == gc1))

                    pred_c0_rgb = rgb565_to_rgb888(pc0)
                    pred_c1_rgb = rgb565_to_rgb888(pc1)
                    gt_c0_rgb = rgb565_to_rgb888(gc0)
                    gt_c1_rgb = rgb565_to_rgb888(gc1)

                    w.writerow([
                        i,
                        f"{s:.6f}", f"{t:.6f}",
                        #f"0x{pc0:04X}", f"0x{pc1:04X}",
                        #f"0x{gc0:04X}", f"0x{gc1:04X}",
                        pc0, pc1,
                        gc0, gc1,
                        match,
                        pred_c0_rgb, pred_c1_rgb,
                        gt_c0_rgb, gt_c1_rgb,
                    ])

            print(f"Saved ALL-block report CSV: {report_path}")

        if PRINT_TO_CONSOLE:
            print("=" * 80)
            print("ALL BLOCKS (console)")
            print("=" * 80)
            for i in range(coords.shape[0]):
                coord = coords[i:i+1]
                gt_565 = targets_rgb565[i]
                pred_565 = preds565[i]

                pred_c0 = rgb565_to_rgb888(int(pred_565[0].item()))
                pred_c1 = rgb565_to_rgb888(int(pred_565[1].item()))
                gt_c0 = rgb565_to_rgb888(int(gt_565[0].item()))
                gt_c1 = rgb565_to_rgb888(int(gt_565[1].item()))

                print(f"Block {i:5d}  st=({coord[0,0].item():.4f},{coord[0,1].item():.4f}) "
                      f"pred: 0x{int(pred_565[0].item()):04X} 0x{int(pred_565[1].item()):04X} "
                      f"gt: 0x{int(gt_565[0].item()):04X} 0x{int(gt_565[1].item()):04X} "
                      f"RGB pred {pred_c0} {pred_c1} | gt {gt_c0} {gt_c1}")

        # Quantize predictions to match format
        preds_q = torch.stack([
            torch.round(preds[:, 0] * 31) / 31.0,
            torch.round(preds[:, 1] * 63) / 63.0,
            torch.round(preds[:, 2] * 31) / 31.0,
            torch.round(preds[:, 3] * 31) / 31.0,
            torch.round(preds[:, 4] * 63) / 63.0,
            torch.round(preds[:, 5] * 31) / 31.0,
        ], dim=1)

        err = preds_q - targets_q01
        mse = torch.mean(err ** 2).item()
        mae = torch.mean(torch.abs(err)).item()

        perfect = torch.all(preds565 == targets_rgb565, dim=1).sum().item()
        acc = perfect / coords.shape[0] * 100.0

        print("\n" + "=" * 80)
        print("FULL DATASET METRICS")
        print("=" * 80)
        print(f"Q01 MSE: {mse:.8f}")
        print(f"Q01 MAE: {mae:.8f}")
        print(f"RGB565 exact-match: {perfect}/{coords.shape[0]} ({acc:.2f}%)")
        print("=" * 80)


if __name__ == "__main__":
    main()
