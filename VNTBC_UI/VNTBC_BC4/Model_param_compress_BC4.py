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
import math

import torch


# Metadata keys appended to grid parameter names
_QMIN_SUFFIX = ".__qmin"
_QMAX_SUFFIX = ".__qmax"
_QBITS_SUFFIX = ".__qbits"
_QSHAPE_SUFFIX = ".__qshape"
_QDTYPE_SUFFIX = ".__qdtype"  # original dtype string so we can restore exactly


def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    bits_list: list[int] = None,
    grid_keyword: str = "grids",
    lpe_keyword: str = "lpe.grids",
) -> Dict[str, torch.Tensor]:
    """
    Quantize grid parameters to uint8 with per-tensor min/max, applying 
    bit-packing for 1-bit, 2-bit, and 4-bit quantizations.
    """
    compressed: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if (grid_keyword in key or lpe_keyword in key) and tensor.is_floating_point():
            if lpe_keyword in key:
                bits = 8  # Default LPE quantization to 8-bit
            else:
                try:
                    lvl = int(key.split('.')[-1])
                    bits = bits_list[lvl] if (bits_list and lvl < len(bits_list)) else 8
                except ValueError:
                    bits = 8
                
            t = tensor.float()
            t_min = t.min()
            t_max = t.max()

            rng = t_max - t_min
            if rng == 0:
                quantized = torch.zeros(0, dtype=torch.uint8)
            else:
                qmax = float((1 << bits) - 1)
                scale = rng / qmax
                zero_point = torch.round(-t_min / scale).clamp(0.0, qmax)
                q_vals = torch.round(t / scale + zero_point).clamp(0.0, qmax).to(torch.uint8)

                if bits in [1, 2, 4]:
                    epb = 8 // bits
                    flat = q_vals.flatten()
                    pad_len = (epb - (len(flat) % epb)) % epb
                    if pad_len > 0:
                        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
                    
                    flat_reshaped = flat.view(-1, epb)
                    packed = torch.zeros(flat_reshaped.shape[0], dtype=torch.uint8, device=flat.device)
                    for i in range(epb):
                        packed |= (flat_reshaped[:, i] << (i * bits))
                    quantized = packed.cpu()
                else:
                    quantized = q_vals.cpu()

            compressed[key] = quantized
            compressed[key + _QMIN_SUFFIX] = t_min.cpu()
            compressed[key + _QMAX_SUFFIX] = t_max.cpu()
            compressed[key + _QBITS_SUFFIX] = torch.tensor(bits, dtype=torch.uint8)
            compressed[key + _QSHAPE_SUFFIX] = torch.tensor(list(tensor.shape), dtype=torch.int32)
            compressed[key + _QDTYPE_SUFFIX] = torch.tensor(
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

    meta_keys = {
        k for k in compressed
        if k.endswith(_QMIN_SUFFIX) or k.endswith(_QMAX_SUFFIX) or k.endswith(_QDTYPE_SUFFIX) or k.endswith(_QBITS_SUFFIX) or k.endswith(_QSHAPE_SUFFIX)
    }

    for key, tensor in compressed.items():
        if key in meta_keys:
            continue

        qmin_key = key + _QMIN_SUFFIX
        if qmin_key in compressed:
            t_min = compressed[qmin_key].float()
            t_max = compressed[key + _QMAX_SUFFIX].float()
            bits = int(compressed.get(key + _QBITS_SUFFIX, torch.tensor(8)).item())
            
            if (key + _QSHAPE_SUFFIX) in compressed:
                orig_shape = tuple(compressed[key + _QSHAPE_SUFFIX].tolist())
            else:
                orig_shape = tensor.shape
                
            rng = t_max - t_min

            if rng == 0:
                reconstructed = torch.full(orig_shape, fill_value=t_min.item(), dtype=torch.float32)
            else:
                qmax = float((1 << bits) - 1)
                scale = rng / qmax
                zero_point = torch.round(-t_min / scale).clamp(0.0, qmax)
                
                if bits in [1, 2, 4]:
                    epb = 8 // bits
                    numel = math.prod(orig_shape)
                    
                    flat_reshaped = tensor.unsqueeze(1).expand(-1, epb)
                    shifts = (torch.arange(epb, dtype=torch.uint8, device=tensor.device) * bits).unsqueeze(0)
                    masks = torch.tensor(qmax, dtype=torch.uint8, device=tensor.device)
                    
                    unpacked = (flat_reshaped >> shifts) & masks
                    q_vals = unpacked.flatten()[:numel]
                else:
                    q_vals = tensor.flatten()
                    
                reconstructed = ((q_vals.float() - zero_point) * scale).view(orig_shape)

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
