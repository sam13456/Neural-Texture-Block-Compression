"""BC1 Reference Endpoints + Dataset Builder

Compresses an image to BC1 using CompressonatorCLI, extracts the endpoints,
and converts them into a training dataset JSON for the endpoint network.


"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DXGI_FORMAT_BC1_UNORM = 71
DXGI_FORMAT_BC1_UNORM_SRGB = 72


def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (int(r), int(g), int(b))


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

    endpoints_rgb565: List[List[int]] = []
    endpoints_rgb888: List[List[List[int]]] = []

    p = offset
    for _ in range(num_blocks):
        c0 = int.from_bytes(data[p:p+2], "little")
        c1 = int.from_bytes(data[p+2:p+4], "little")
        endpoints_rgb565.append([c0, c1])
        endpoints_rgb888.append([list(rgb565_to_rgb888(c0)), list(rgb565_to_rgb888(c1))])
        p += 8

    return {
        "width": int(width),
        "height": int(height),
        "blocks_x": int(blocks_x),
        "blocks_y": int(blocks_y),
        "block_order": "row_major",
        "format": "BC1",
        "endpoints_rgb565": endpoints_rgb565,
        "endpoints_rgb888": endpoints_rgb888,
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
    ref_json: Path,
    out_json: Path,
    include_meta: bool = True,
    keep_only_c0_gt_c1: bool = False,
) -> None:
    d = json.loads(ref_json.read_text())

    W, H = int(d["width"]), int(d["height"])
    Bx, By = int(d["blocks_x"]), int(d["blocks_y"])
    eps = d["endpoints_rgb565"]
    n = len(eps)

    st: List[List[float]] = []
    bxby: List[List[int]] = []
    ep_q01: List[List[float]] = []
    c0_gt_c1: List[int] = []
    ep_rgb565: List[List[int]] = []

    for i, pair in enumerate(eps):
        c0, c1 = int(pair[0]), int(pair[1])
        bx = i % Bx
        by = i // Bx

        s = bx / (Bx - 1) if Bx > 1 else 0.0
        t = by / (By - 1) if By > 1 else 0.0

        flag = 1 if (c0 > c1) else 0
        if keep_only_c0_gt_c1 and flag == 0:
            continue

        bxby.append([bx, by])
        st.append([float(s), float(t)])

        ep_rgb565.append([c0, c1])
        ep_q01.append(rgb565_to_q01(c0) + rgb565_to_q01(c1))
        c0_gt_c1.append(flag)

    out: Dict[str, Any] = {
        "inputs": {"st": st, "bxby": bxby},
        "targets": {"ep_rgb565": ep_rgb565, "ep_q01": ep_q01},
        "flags": {"c0_gt_c1": c0_gt_c1},
    }

    if include_meta:
        out["meta"] = {
            "width": W,
            "height": H,
            "blocks_x": Bx,
            "blocks_y": By,
            "block_order": d.get("block_order", "row_major"),
            "format": d.get("format", "BC1"),
            "num_blocks_total": n,
            "num_blocks_kept": len(ep_rgb565),
            "filtered_c0_gt_c1": bool(keep_only_c0_gt_c1),
            "source_image": d.get("compress_meta", {}).get("source_image"),
        }

    out_json.write_text(json.dumps(out, indent=2))
    print("Wrote dataset:", out_json)


@dataclass
class Config:
    CLI: str
    IMG: str
    OUT: str
    encode_with: str = "HPC"
    refine_steps: int = 2
    KEEP_ONLY_C0_GT_C1: bool = False
    INCLUDE_META: bool = True


if __name__ == "__main__":
    cfg = Config(
        CLI=r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe",
        IMG=r"D:\BC1 extract\Bricks090_diffuse\Bricks090_2K-PNG_Color.png",
        OUT=r"D:\BC1 extract\Bricks090_diffuse",
        encode_with="HPC",
        refine_steps=2,
        KEEP_ONLY_C0_GT_C1=False,  # set True to drop BC1 3-color mode blocks
        INCLUDE_META=False,
    )

    ref, ref_meta, out_dds = get_reference_endpoints_bc1(
        cfg.CLI, cfg.IMG, cfg.OUT, encode_with=cfg.encode_with, refine_steps=cfg.refine_steps
    )

    out_dir = Path(cfg.OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_json = out_dir / "bc1_reference_endpoints.json"
    ref_out = dict(ref)
    ref_out["compress_meta"] = ref_meta
    ref_json.write_text(json.dumps(ref_out, indent=2))
    print("Saved reference endpoints:", ref_json)
    print("DDS written to:", out_dds)

    convert_reference_to_dataset(
        ref_json=ref_json,
        out_json=out_dir / "bc1_endpoint_dataset.json",
        include_meta=cfg.INCLUDE_META,
        keep_only_c0_gt_c1=cfg.KEEP_ONLY_C0_GT_C1,
    )
