"""BC1 Reference Endpoints + Dataset Builder (Single or Multi-Texture)

This script uses CompressonatorCLI to compress one or more *RGB* source images to BC1,
extracts the per-block BC1 endpoints (RGB565 c0/c1), and writes a JSON dataset.

Why multi-texture?
- NTBC (conservative approach) can train ONE endpoint network for a *material* that has
  multiple RGB textures (e.g., albedo, normalRGB, ORM-as-RGB, etc.).
- For N textures, the endpoint network predicts 6*N floats per block (two RGB endpoints per texture).

Outputs:
- Train_dataset.json:
    inputs.bxby:  [ [bx,by], ... ] for every 4x4 block (row-major)
    targets.ep_q01: [ [e0.rgb, e1.rgb] concatenated per texture ] in Q01 float
        shape per block = 6 * num_textures
- Inference_input.json:
    blocks_x, blocks_y, width, height, num_textures, texture_names

Notes:
- This script only builds the *endpoint* dataset. The color network training additionally
  needs the uncompressed source images, and optionally other per-texel derived data.
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
CONFIG: Dict[str, Any] = {
    "cli": r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe",

    # ---- SINGLE TEXTURE (legacy) ----
    # If you keep only "source_image", the script behaves like before.
    #"source_image": r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_Color.png",

    # ---- MULTI TEXTURE (new) ----
    # If you set "source_images" to a non-empty list, it overrides "source_image".
    # All textures must have the SAME resolution.
    # Example:
    # "source_images": [
    #     r"D:\mat\albedo.png",
    #     r"D:\mat\normal_rgb.png",
    #     r"D:\mat\orm_rgb.png",
    # ],
    "source_images": [r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_Color.png", r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_NormalDX.png", r"D:\BC1 extract\Bricks090_4K-PNG\Bricks090_4K-PNG_NormalGL.png"],

    # Optional explicit names matching source_images length (otherwise uses file stems)
    "texture_names": ["Color", "NormalDX", "NormalGL"],

    "out_dir": r"D:\BC1 extract\Bricks090_4K_test",
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
    """Parse a BC1 DDS and return endpoints_rgb565 (N,2) in row-major block order."""
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
    if fourcc == b"DXT1":
        pass
    elif fourcc == b"DX10":
        dx10 = data[offset:offset + 20]
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
    c0_arr = raw[0::4].astype(np.int32)
    c1_arr = raw[1::4].astype(np.int32)

    endpoints_rgb565 = np.stack([c0_arr, c1_arr], axis=-1)  # (N, 2)

    return {
        "width": int(width),
        "height": int(height),
        "blocks_x": int(blocks_x),
        "blocks_y": int(blocks_y),
        "block_order": "row_major",
        "format": "BC1",
        "endpoints_rgb565": endpoints_rgb565,
    }


def _ensure_source_images(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (source_images, texture_names) with sane defaults."""
    srcs: List[str] = []
    if cfg.get("source_images"):
        srcs = list(cfg["source_images"])
    else:
        # legacy single
        si = cfg.get("source_image")
        if si:
            srcs = [si]

    if not srcs:
        raise ValueError("No input images. Provide CONFIG['source_image'] or CONFIG['source_images'].")

    # Names
    names_cfg = list(cfg.get("texture_names") or [])
    if names_cfg and len(names_cfg) != len(srcs):
        raise ValueError("CONFIG['texture_names'] length must match CONFIG['source_images'].")

    if names_cfg:
        names = names_cfg
    else:
        names = [Path(s).stem for s in srcs]

    return srcs, names


def get_reference_endpoints_bc1(
    cli_path: str,
    image_path: str,
    work_dir: str,
    encode_with: str = "HPC",
    refine_steps: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Legacy: single image -> (ref, meta, dds_path)."""
    cli_path_p = Path(cli_path).expanduser().resolve()
    image_path_p = Path(image_path).expanduser().resolve()
    work_dir_p = Path(work_dir).expanduser().resolve()
    work_dir_p.mkdir(parents=True, exist_ok=True)

    out_dds = work_dir_p / (image_path_p.stem + "_ref_bc1.dds")
    run_meta = compress_bc1_dds(cli_path_p, image_path_p, out_dds, encode_with=encode_with, refine_steps=refine_steps)
    ref = parse_dds_bc1_endpoints(out_dds)

    ref_meta = {
        "source_image": str(image_path_p),
        "reference_dds": str(out_dds),
        "compressonator_cli": str(cli_path_p),
        **run_meta,
    }
    return ref, ref_meta, out_dds


def get_reference_endpoints_bc1_multi(
    cli_path: str,
    image_paths: List[str],
    work_dir: str,
    encode_with: str = "HPC",
    refine_steps: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    """Multi: list of images -> ([ref_k], [meta_k], [dds_k]). Enforces same resolution."""
    refs: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    dds_paths: List[Path] = []

    w0 = h0 = bx0 = by0 = None

    for img in image_paths:
        ref, meta, dds = get_reference_endpoints_bc1(
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


def convert_reference_to_dataset(
    ref: Dict[str, Any],
    out_json: Path,
    source_image: Optional[str] = None,
    include_meta: bool = True,
) -> None:
    """Legacy: single ref -> endpoint dataset (same schema as before)."""
    convert_reference_to_dataset_multi(
        refs=[ref],
        out_json=out_json,
        source_images=[source_image] if source_image else None,
        texture_names=None,
        include_meta=include_meta,
    )


def convert_reference_to_dataset_multi(
    refs: List[Dict[str, Any]],
    out_json: Path,
    source_images: Optional[List[str]] = None,
    texture_names: Optional[List[str]] = None,
    include_meta: bool = True,
) -> None:
    """Multi: refs -> dataset with targets.ep_q01 per block = 6*num_textures."""
    import numpy as np

    if not refs:
        raise ValueError("refs is empty")

    # Shared dimensions (already validated in get_reference_endpoints_bc1_multi)
    W, H = int(refs[0]["width"]), int(refs[0]["height"])
    Bx, By = int(refs[0]["blocks_x"]), int(refs[0]["blocks_y"])

    eps0 = refs[0]["endpoints_rgb565"]
    if not isinstance(eps0, np.ndarray):
        eps0 = np.asarray(eps0, dtype=np.int32)
    n = int(eps0.shape[0])

    # Build bxby in row-major block order
    idx = np.arange(n, dtype=np.int32)
    bxby = np.stack([idx % Bx, idx // Bx], axis=-1)  # (N, 2)

    def rgb565_to_q01_vec(c: np.ndarray) -> np.ndarray:
        r5 = ((c >> 11) & 0x1F).astype(np.float32) / 31.0
        g6 = ((c >> 5) & 0x3F).astype(np.float32) / 63.0
        b5 = (c & 0x1F).astype(np.float32) / 31.0
        return np.stack([r5, g6, b5], axis=-1)  # (N, 3)

    # Per-texture endpoints -> Q01 -> concat along last dim
    ep_q01_list: List[np.ndarray] = []
    for k, ref in enumerate(refs):
        eps = ref["endpoints_rgb565"]
        if not isinstance(eps, np.ndarray):
            eps = np.asarray(eps, dtype=np.int32)
        if eps.shape[0] != n:
            raise ValueError(f"Texture {k} has different number of blocks.")

        c0 = eps[:, 0]
        c1 = eps[:, 1]
        q01_c0 = rgb565_to_q01_vec(c0)  # (N, 3)
        q01_c1 = rgb565_to_q01_vec(c1)  # (N, 3)
        ep_q01 = np.concatenate([q01_c0, q01_c1], axis=-1)  # (N, 6)
        ep_q01_list.append(ep_q01)

    # (N, 6*T)
    ep_q01_all = np.concatenate(ep_q01_list, axis=-1)

    out: Dict[str, Any] = {
        "inputs": {"bxby": bxby.tolist()},
        "targets": {"ep_q01": ep_q01_all.tolist()},
    }

    num_textures = len(refs)

    # Meta
    if include_meta:
        out["meta"] = {
            "width": W,
            "height": H,
            "blocks_x": Bx,
            "blocks_y": By,
            "block_order": refs[0].get("block_order", "row_major"),
            "format": refs[0].get("format", "BC1"),
            "num_blocks_total": n,
            "num_textures": num_textures,
            "texture_names": texture_names,
            "source_images": source_images,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out))
    print("Wrote dataset:", out_json)

    # Inference helper (lets inference know how to split outputs back to textures)
    coords_json = out_json.parent / "Inference_input.json"
    coords_payload = {
        "width": W,
        "height": H,
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

    # Multi path (also covers single, but we keep legacy prints readable)
    if len(src_images) == 1:
        ref, ref_meta, out_dds = get_reference_endpoints_bc1(
            cfg["cli"], src_images[0], str(out_dir),
            encode_with=cfg["encode_with"], refine_steps=cfg["refine_steps"],
        )
        print("DDS written to:", out_dds)

        convert_reference_to_dataset(
            ref=ref,
            out_json=out_dir / "Train_dataset.json",
            source_image=src_images[0],
            include_meta=cfg["include_meta"],
        )
    else:
        refs, metas, dds_paths = get_reference_endpoints_bc1_multi(
            cfg["cli"], src_images, str(out_dir),
            encode_with=cfg["encode_with"], refine_steps=cfg["refine_steps"],
        )
        print("DDS written to:")
        for d in dds_paths:
            print("  ", d)

        convert_reference_to_dataset_multi(
            refs=refs,
            out_json=out_dir / "Train_dataset.json",
            source_images=src_images,
            texture_names=tex_names,
            include_meta=cfg["include_meta"],
        )
