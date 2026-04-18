"""BC4 Reference Endpoints + Dataset Builder (Single or Multi-Texture)

Uses CompressonatorCLI to compress single-channel source images to BC4,
extracts per-block endpoints (2 × uint8), and writes a JSON dataset.

BC4 block layout (8 bytes per 4×4 block):
  - Byte 0: endpoint e0 (uint8)
  - Byte 1: endpoint e1 (uint8)
  - Bytes 2-7: 16 × 3-bit selectors (48 bits, packed little-endian)

For multi-texture, the endpoint network predicts 2*T floats per block.
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DXGI_FORMAT_BC4_UNORM = 80
DXGI_FORMAT_BC4_SNORM = 81


# =========================
# CONFIG
# =========================
from config_BC4 import (
    COMPRESSONATOR_CLI, SOURCE_IMAGES, TEXTURE_NAMES, MODEL_DIR,
)

CONFIG: Dict[str, Any] = {
    "cli": COMPRESSONATOR_CLI,
    "source_images": SOURCE_IMAGES,
    "texture_names": TEXTURE_NAMES,
    "out_dir": MODEL_DIR,
    "encode_with": "HPC",
    "refine_steps": 2,
    "include_meta": False,
}


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


def compress_bc4_dds(
    cli_path: Path,
    src_img: Path,
    out_dds: Path,
    encode_with: str = "HPC",
    refine_steps: int = 2,
    nomipmap: bool = True,
) -> Dict[str, Any]:
    out_dds.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(cli_path), "-fd", "BC4", "-EncodeWith", encode_with]
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


def parse_dds_bc4_endpoints(dds_path: Path) -> Dict[str, Any]:
    """Parse a BC4 DDS and return endpoints_u8 (N,2) in row-major block order."""
    data = dds_path.read_bytes()
    if len(data) < 128 or data[0:4] != b"DDS ":
        raise ValueError("Not a valid DDS file (missing DDS magic).")

    header = data[4:4 + 124]
    if struct.unpack_from("<I", header, 0)[0] != 124:
        raise ValueError("Unexpected DDS header size.")

    height = struct.unpack_from("<I", header, 8)[0]
    width = struct.unpack_from("<I", header, 12)[0]

    ddspf_off = 72
    fourcc = header[ddspf_off + 8: ddspf_off + 12]

    offset = 4 + 124
    if fourcc == b"ATI1":
        pass
    elif fourcc == b"BC4U":
        pass
    elif fourcc == b"DX10":
        dx10 = data[offset:offset + 20]
        dxgi_format = struct.unpack_from("<I", dx10, 0)[0]
        if dxgi_format not in (DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM):
            raise ValueError(f"DDS DX10 format is not BC4 (dxgiFormat={dxgi_format}).")
        offset += 20
    else:
        raise ValueError(f"Unsupported DDS FourCC for BC4 extraction: {fourcc!r}")

    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    num_blocks = blocks_x * blocks_y

    needed = offset + num_blocks * 8
    if len(data) < needed:
        raise ValueError("DDS truncated: not enough BC4 blocks.")

    # Each BC4 block: 8 bytes = [e0, e1, 6 bytes of 3-bit selectors]
    raw = np.frombuffer(data, dtype=np.uint8, offset=offset, count=num_blocks * 8)
    raw = raw.reshape(num_blocks, 8)
    e0_arr = raw[:, 0].astype(np.int32)
    e1_arr = raw[:, 1].astype(np.int32)

    endpoints_u8 = np.stack([e0_arr, e1_arr], axis=-1)  # (N, 2)

    return {
        "width": int(width),
        "height": int(height),
        "blocks_x": int(blocks_x),
        "blocks_y": int(blocks_y),
        "block_order": "row_major",
        "format": "BC4",
        "endpoints_u8": endpoints_u8,
    }


def _ensure_source_images(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    srcs = list(cfg.get("source_images") or [])
    if not srcs:
        raise ValueError("No input images. Provide CONFIG['source_images'].")

    names_cfg = list(cfg.get("texture_names") or [])
    if names_cfg and len(names_cfg) != len(srcs):
        raise ValueError("CONFIG['texture_names'] length must match CONFIG['source_images'].")

    names = names_cfg if names_cfg else [Path(s).stem for s in srcs]
    return srcs, names


def get_reference_endpoints_bc4(
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

    out_dds = work_dir_p / (image_path_p.stem + "_ref_bc4.dds")
    run_meta = compress_bc4_dds(cli_path_p, image_path_p, out_dds, encode_with=encode_with, refine_steps=refine_steps)
    ref = parse_dds_bc4_endpoints(out_dds)

    ref_meta = {
        "source_image": str(image_path_p),
        "reference_dds": str(out_dds),
        "compressonator_cli": str(cli_path_p),
        **run_meta,
    }
    return ref, ref_meta, out_dds


def get_reference_endpoints_bc4_multi(
    cli_path: str,
    image_paths: List[str],
    work_dir: str,
    encode_with: str = "HPC",
    refine_steps: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    refs: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    dds_paths: List[Path] = []

    w0 = h0 = bx0 = by0 = None

    for img in image_paths:
        ref, meta, dds = get_reference_endpoints_bc4(
            cli_path=cli_path,
            image_path=img,
            work_dir=work_dir,
            encode_with=encode_with,
            refine_steps=refine_steps,
        )
        W, H = int(ref["width"]), int(ref["height"])
        Bx, By = int(ref["blocks_x"]), int(ref["blocks_y"])

        if w0 is None:
            w0, h0, bx0, by0 = W, H, Bx, By
        else:
            if (W, H, Bx, By) != (w0, h0, bx0, by0):
                raise ValueError(
                    "All textures must have identical dimensions.\n"
                    f"First: {w0}x{h0} ({bx0}x{by0} blocks)\n"
                    f"Now:   {W}x{H} ({Bx}x{By} blocks)\n"
                    f"Image: {img}"
                )

        refs.append(ref)
        metas.append(meta)
        dds_paths.append(dds)

    return refs, metas, dds_paths


def convert_reference_to_dataset_multi(
    refs: List[Dict[str, Any]],
    out_json: Path,
    source_images: Optional[List[str]] = None,
    texture_names: Optional[List[str]] = None,
    include_meta: bool = True,
) -> None:
    """Multi: refs -> dataset with targets.ep_q01 per block = 2*num_textures."""
    if not refs:
        raise ValueError("refs is empty")

    W, H = int(refs[0]["width"]), int(refs[0]["height"])
    Bx, By = int(refs[0]["blocks_x"]), int(refs[0]["blocks_y"])

    eps0 = refs[0]["endpoints_u8"]
    if not isinstance(eps0, np.ndarray):
        eps0 = np.asarray(eps0, dtype=np.int32)
    n = int(eps0.shape[0])

    # Build bxby in row-major order
    idx = np.arange(n, dtype=np.int32)
    bxby = np.stack([idx % Bx, idx // Bx], axis=-1)  # (N, 2)

    # Per-texture endpoints -> Q01 -> concat
    ep_q01_list: List[np.ndarray] = []
    for k, ref in enumerate(refs):
        eps = ref["endpoints_u8"]
        if not isinstance(eps, np.ndarray):
            eps = np.asarray(eps, dtype=np.int32)
        e0_arr = eps[:, 0]
        e1_arr = eps[:, 1]
        eps = np.stack([e0_arr, e1_arr], axis=-1)
        # BC4: just 2 uint8 values per block -> normalize to [0,1]
        ep_q01 = eps.astype(np.float32) / 255.0  # (N, 2)
        ep_q01_list.append(ep_q01)

    # (N, 2*T)
    ep_q01_all = np.concatenate(ep_q01_list, axis=-1)

    out: Dict[str, Any] = {
        "inputs": {"bxby": bxby.tolist()},
        "targets": {"ep_q01": ep_q01_all.tolist()},
    }

    num_textures = len(refs)

    if include_meta:
        out["meta"] = {
            "width": W,
            "height": H,
            "blocks_x": Bx,
            "blocks_y": By,
            "block_order": refs[0].get("block_order", "row_major"),
            "format": refs[0].get("format", "BC4"),
            "num_blocks_total": n,
            "num_textures": num_textures,
            "texture_names": texture_names,
            "source_images": source_images,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    print("Wrote dataset:", out_json)

    # Inference helper
    coords_json = out_json.parent / "Inference_input.json"
    coords_payload = {
        "blocks_x": Bx,
        "blocks_y": By,
        "num_textures": num_textures,
        "texture_names": texture_names,
    }
    coords_json.write_text(json.dumps(coords_payload))
    print("Wrote coords:", coords_json)


if __name__ == "__main__":
    cfg = CONFIG

    src_images, tex_names = _ensure_source_images(cfg)

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    refs, metas, dds_paths = get_reference_endpoints_bc4_multi(
        cfg["cli"], src_images, str(out_dir),
        encode_with=cfg["encode_with"], refine_steps=cfg["refine_steps"],
    )
    print("DDS written to:")
    for d in dds_paths:
        print(" ", d)

    convert_reference_to_dataset_multi(
        refs=refs,
        out_json=out_dir / "Train_dataset.json",
        source_images=src_images,
        texture_names=tex_names,
        include_meta=cfg["include_meta"],
    )
