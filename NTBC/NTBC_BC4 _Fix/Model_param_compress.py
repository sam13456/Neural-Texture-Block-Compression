"""
uint8 State Dict Compression / Decompression

After QAT training, multi-resolution grid parameters are meant to live at 8-bit precision.
This module quantizes those grids to uint8 with per-tensor min/max metadata,
cutting grid storage to ~1/4 of float32 or ~1/2 of float16.

MLP weights (tiny in comparison) are left untouched.

Usage:
    from state_dict_compress import compress_state_dict, decompress_state_dict

    # Save
    compressed = compress_state_dict(net.state_dict())
    torch.save(compressed, "model_compressed.pt")

    # Load
    compressed = torch.load("model_compressed.pt", map_location="cpu")
    real_sd = decompress_state_dict(compressed)
    net.load_state_dict(real_sd)
"""

from __future__ import annotations

from typing import Dict

import torch


# Metadata keys appended to grid parameter names
_QMIN_SUFFIX = ".__qmin"
_QMAX_SUFFIX = ".__qmax"
_QDTYPE_SUFFIX = ".__qdtype"  # original dtype string so we can restore exactly


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    grid_keyword: str = "grids",
) -> Dict[str, torch.Tensor]:
    """
    Quantize grid parameters to uint8 with per-tensor min/max.

    Only floating-point tensors whose key contains `grid_keyword` are quantized.
    Everything else (MLP weights, buffers, etc.) is kept as-is.

    This is lossless relative to what QAT trains for: the fake-quant during
    training already rounds values to 8-bit levels, so re-quantizing here
    just stores what the network was already using.
    """
    compressed: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if grid_keyword in key and tensor.is_floating_point():
            t = tensor.float()
            t_min = t.min()
            t_max = t.max()

            # Avoid division by zero for constant grids
            rng = t_max - t_min
            if rng == 0:
                # All values identical: store as zeros, reconstruct from min
                quantized = torch.zeros_like(t, dtype=torch.uint8)
            else:
                scale = rng / 255.0
                quantized = ((t - t_min) / scale).round().clamp(0, 255).to(torch.uint8)

            compressed[key] = quantized
            compressed[key + _QMIN_SUFFIX] = t_min.cpu()
            compressed[key + _QMAX_SUFFIX] = t_max.cpu()
            compressed[key + _QDTYPE_SUFFIX] = torch.tensor(
                # Store dtype as an int flag: 0=float32, 1=float16, 2=bfloat16
                {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}.get(tensor.dtype, 0)
            )
        else:
            compressed[key] = tensor

    return compressed


_DTYPE_MAP = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}


def decompress_state_dict(
    compressed: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct floating-point grids from uint8 + per-tensor min/max.

    Returns a state_dict that can be loaded with model.load_state_dict().
    """
    state_dict: Dict[str, torch.Tensor] = {}

    # Collect metadata keys so we can skip them
    meta_keys = {
        k for k in compressed
        if k.endswith(_QMIN_SUFFIX) or k.endswith(_QMAX_SUFFIX) or k.endswith(_QDTYPE_SUFFIX)
    }

    for key, tensor in compressed.items():
        if key in meta_keys:
            continue

        qmin_key = key + _QMIN_SUFFIX
        if qmin_key in compressed:
            # This is a quantized grid, reconstruct it
            t_min = compressed[qmin_key].float()
            t_max = compressed[key + _QMAX_SUFFIX].float()
            rng = t_max - t_min

            if rng == 0:
                reconstructed = torch.full_like(tensor, fill_value=t_min.item(), dtype=torch.float32)
            else:
                scale = rng / 255.0
                reconstructed = tensor.float() * scale + t_min

            # Restore original dtype
            dtype_flag = int(compressed.get(key + _QDTYPE_SUFFIX, torch.tensor(0)).item())
            target_dtype = _DTYPE_MAP.get(dtype_flag, torch.float32)
            state_dict[key] = reconstructed.to(target_dtype)
        else:
            state_dict[key] = tensor

    return state_dict


def print_size_comparison(original_sd: Dict[str, torch.Tensor], compressed_sd: Dict[str, torch.Tensor]):
    """Print a summary of size savings."""
    orig_bytes = sum(t.numel() * t.element_size() for t in original_sd.values())
    comp_bytes = sum(t.numel() * t.element_size() for t in compressed_sd.values())
    ratio = comp_bytes / orig_bytes if orig_bytes > 0 else 1.0
    print(f"Original:   {orig_bytes / 1024 / 1024:.2f} MB")
    print(f"Compressed: {comp_bytes / 1024 / 1024:.2f} MB")
    print(f"Ratio:      {ratio:.2%}  (saved {(1.0 - ratio):.1%})")
