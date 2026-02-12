"""BC1 Reference Endpoints + Dataset Builder

Compresses an image to BC1 using CompressonatorCLI, extracts the endpoints,
and converts them into a training dataset JSON for the endpoint network.


"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DXGI_FORMAT_BC1_UNORM = 71
DXGI_FORMAT_BC1_UNORM_SRGB = 72


# =========================
# CONFIG (edit these)
# =========================
CONFIG = {
    "cli": r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe",
    "source_image": r"D:\BC1 extract\Bricks090_diffuse_8K\Bricks090_8K-PNG_Color.png",
    "out_dir": r"D:\BC1 extract\Bricks090_diffuse_8K_model",
    "encode_with": "HPC",
    "refine_steps": 2,
    "include_meta": False,
}


def rgb565_to_q01(c: int) -> List[float]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    return [r5 / 31.0, g6 / 63.0, b5 / 31.0]


def _run(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def detect_refine_flag(cli_path: Path) -> Optional[str]:
    rc, out, err = _run([str(cli_path), "-help"])
    text = (out or "") + "\n" + (err or "")
    candidates = [
        "-RefineSteps", "-refinesteps", "-refineSteps",
        "-RefineStep", "-refinestep", "-refine"
    ]
    for c in candidates:
        if c.lower() in text.lower():
            return c
    m = re.search(r"(-{1,2}[A-Za-z0-9_]*refine[A-Za-z0-9_]*)", text, re.IGNORECASE)
    return m.group(1) if m else None


def compress_bc1_dds(
    cli_path: Path,
    src_img: Path,
    out_dds: Path,
    encode_with: str = "HPC",
    refine_steps: int = 2,
    nomipmap: bool = True,
) -> Dict[str, Any]:
    out_dds.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(cli_path), "-fd", "BC1", "-EncodeWith", encode_with]
    if nomipmap:
        cmd.append("-nomipmap")
    cmd += [str(src_img), str(out_dds)]

    refine_flag = detect_refine_flag(cli_path)
    used_refine = False
    if refine_flag:
        cmd.insert(-2, str(refine_steps))
        cmd.insert(-2, refine_flag)
        used_refine = True

    rc, out, err = _run(cmd)
    if rc != 0:
        raise RuntimeError(
            "CompressonatorCLI failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{out}\n"
            f"STDERR:\n{err}\n"
        )

    return {
        "encode_with": encode_with,
        "nomipmap": bool(nomipmap),
        "refine_steps": int(refine_steps),
        "refine_flag": refine_flag,
        "refine_used": used_refine,
    }


def parse_dds_bc1_endpoints(dds_path: Path) -> Dict[str, Any]:
    data = dds_path.read_bytes()
    if len(data) < 128 or data[0:4] != b"DDS ":
        raise ValueError("Not a valid DDS file (missing DDS magic).")

    header = data[4:4+124]
    if struct.unpack_from("<I", header, 0)[0] != 124:
        raise ValueError("Unexpected DDS header size.")

    height = struct.unpack_from("<I", header, 8)[0]
    width  = struct.unpack_from("<I", header, 12)[0]

    ddspf_off = 72
    fourcc = header[ddspf_off + 8: ddspf_off + 12]

    offset = 4 + 124
    if fourcc == b"DXT1":
        pass
    elif fourcc == b"DX10":
        dx10 = data[offset:offset+20]
        dxgi_format = struct.unpack_from("<I", dx10, 0)[0]
        if dxgi_format not in (DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB):
            raise ValueError(f"DDS DX10 format is not BC1 (dxgiFormat={dxgi_format}).")
        offset += 20
    else:
        raise ValueError(f"Unsupported DDS FourCC for BC1 extraction: {fourcc!r}")

    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    num_blocks = blocks_x * blocks_y

    needed = offset + num_blocks * 8
    if len(data) < needed:
        raise ValueError("DDS truncated: not enough BC1 blocks.")

    # Vectorised: read all BC1 blocks at once (8 bytes each: 2B c0, 2B c1, 4B indices)
    import numpy as np
    raw = np.frombuffer(data, dtype=np.uint16, offset=offset, count=num_blocks * 4)
    # raw is [c0_0, c1_0, idx_lo_0, idx_hi_0, c0_1, c1_1, ...]
    c0_arr = raw[0::4].astype(np.int32)  # (N,) every 4th uint16 starting at 0
    c1_arr = raw[1::4].astype(np.int32)  # (N,) every 4th uint16 starting at 1

    endpoints_rgb565 = np.stack([c0_arr, c1_arr], axis=-1)              # (N, 2)

    return {
        "width": int(width),
        "height": int(height),
        "blocks_x": int(blocks_x),
        "blocks_y": int(blocks_y),
        "block_order": "row_major",
        "format": "BC1",
        "endpoints_rgb565": endpoints_rgb565,
    }


def get_reference_endpoints_bc1(
    cli_path: str,
    image_path: str,
    work_dir: str,
    encode_with: str = "HPC",
    refine_steps: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    cli_path_p = Path(cli_path).expanduser().resolve()
    image_path_p = Path(image_path).expanduser().resolve()
    work_dir_p = Path(work_dir).expanduser().resolve()
    work_dir_p.mkdir(parents=True, exist_ok=True)

    out_dds = work_dir_p / (image_path_p.stem + "_ref_bc1.dds")
    run_meta = compress_bc1_dds(cli_path_p, image_path_p, out_dds, encode_with=encode_with, refine_steps=refine_steps)
    ref = parse_dds_bc1_endpoints(out_dds)

    ref_meta = {
        "source_image": str(image_path_p),
        "compressonator_cli": str(cli_path_p),
        **run_meta,
    }
    return ref, ref_meta, out_dds


def convert_reference_to_dataset(
    ref: Dict[str, Any],
    out_json: Path,
    source_image: Optional[str] = None,
    include_meta: bool = True,
) -> None:
    """Convert parsed reference endpoints directly into a training dataset JSON."""
    import numpy as np

    W, H = int(ref["width"]), int(ref["height"])
    Bx, By = int(ref["blocks_x"]), int(ref["blocks_y"])
    eps = ref["endpoints_rgb565"]  # numpy (N,2) int32 from parse_dds_bc1_endpoints
    n = int(eps.shape[0]) if hasattr(eps, 'shape') else len(eps)

    # Vectorised bxby: row-major block indices
    idx = np.arange(n, dtype=np.int32)
    bxby = np.stack([idx % Bx, idx // Bx], axis=-1)  # (N, 2) int32

    # Vectorised RGB565 -> q01 for both c0 and c1
    if not isinstance(eps, np.ndarray):
        eps = np.asarray(eps, dtype=np.int32)
    c0 = eps[:, 0]
    c1 = eps[:, 1]

    def rgb565_to_q01_vec(c):
        r5 = ((c >> 11) & 0x1F).astype(np.float32) / 31.0
        g6 = ((c >> 5) & 0x3F).astype(np.float32) / 63.0
        b5 = (c & 0x1F).astype(np.float32) / 31.0
        return np.stack([r5, g6, b5], axis=-1)  # (N, 3)

    q01_c0 = rgb565_to_q01_vec(c0)  # (N, 3)
    q01_c1 = rgb565_to_q01_vec(c1)  # (N, 3)
    ep_q01 = np.concatenate([q01_c0, q01_c1], axis=-1)  # (N, 6)

    out: Dict[str, Any] = {
        "inputs": {"bxby": bxby.tolist()},
        "targets": {"ep_q01": ep_q01.tolist()},
    }

    if include_meta:
        out["meta"] = {
            "width": W,
            "height": H,
            "blocks_x": Bx,
            "blocks_y": By,
            "block_order": ref.get("block_order", "row_major"),
            "format": ref.get("format", "BC1"),
            "num_blocks_total": n,
            "source_image": source_image,
        }

    out_json.write_text(json.dumps(out))
    print("Wrote dataset:", out_json)

    # Compact coords file: just the grid dimensions (st/bxby are derived from these)
    coords_json = out_json.parent / "Inference_input.json"
    coords_json.write_text(json.dumps({"blocks_x": Bx, "blocks_y": By}))
    print("Wrote coords:", coords_json)



if __name__ == "__main__":
    cfg = CONFIG

    ref, ref_meta, out_dds = get_reference_endpoints_bc1(
        cfg["cli"], cfg["source_image"], cfg["out_dir"],
        encode_with=cfg["encode_with"], refine_steps=cfg["refine_steps"],
    )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print("DDS written to:", out_dds)

    convert_reference_to_dataset(
        ref=ref,
        out_json=out_dir / "Train_dataset.json",
        source_image=cfg["source_image"],
        include_meta=cfg["include_meta"],
    )
