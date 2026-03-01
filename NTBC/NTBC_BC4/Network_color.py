"""
NTBC Color Network for BC4 compression (multi-single-channel-texture capable).
Based on: Neural Texture Block Compression (arXiv:2407.09543)

BC4 compresses single-channel textures. The color network predicts 1 scalar
value per texel per texture (instead of 3 for BC1).

If num_textures=T, the network outputs (1*T) values per texel in [0,1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Network_endpoint import bc4_palette_bc4order


# ---------- Quantization helpers ----------

def _fake_quantize_asymmetric_with_range(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    """Asymmetric fake quantization with straight-through estimator."""
    qmin = 0.0
    qmax = float((1 << bits) - 1)

    x_clamped = torch.clamp(x, alpha, beta)
    scale = (beta - alpha) / (qmax - qmin)
    zero_point = torch.round(-alpha / scale).clamp(qmin, qmax)

    q = torch.round(x_clamped / scale + zero_point).clamp(qmin, qmax)
    x_q = (q - zero_point) * scale
    return x_clamped + (x_q - x_clamped).detach()


# ---------- BC4 helpers ----------

# BC4 palette weights: w_n = n/7 for n=0..7
_BC4_W = torch.tensor([i / 7.0 for i in range(8)], dtype=torch.float32)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return coords.clamp(0.0, 1.0 - eps)


def endpoints2_to_e0e1(endpoints2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e0 = endpoints2[..., 0:1]
    e1 = endpoints2[..., 1:2]
    return e0, e1


def bc4_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """e0, e1: (...,1), w: (8,) -> (...,8,1)"""
    wv = w.view(*([1] * (e0.ndim - 1)), -1, 1).to(device=e0.device, dtype=e0.dtype)
    e0v = e0.unsqueeze(-2)
    e1v = e1.unsqueeze(-2)
    return (1.0 - wv) * e0v + wv * e1v


def _infer_num_textures_from_flat_values(values_flat: torch.Tensor) -> int:
    """values_flat: (B, 1*T) -> T"""
    if values_flat.ndim != 2:
        raise ValueError(f"values_flat must be (B, T). Got {tuple(values_flat.shape)}")
    return int(values_flat.shape[1])


def split_values_flat(values_flat: torch.Tensor) -> torch.Tensor:
    """(B,T) -> (B,T,1)"""
    return values_flat.unsqueeze(-1)


# ---------- Multi-resolution Feature Grids ----------

class MultiResFeatureGrid2D(nn.Module):
    """Multi-resolution feature grids with bilinear interpolation."""

    def __init__(
        self,
        num_levels: int = 8,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        feature_dim: int = 2,
        init_range: float = 1e-4,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        if base_resolution < 2 or finest_resolution < 2:
            raise ValueError("resolutions must be >= 2")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")

        self.num_levels = int(num_levels)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.feature_dim = int(feature_dim)
        self.init_range = float(init_range)
        self.param_dtype = param_dtype

        if self.num_levels == 1:
            self._resolutions = [self.base_resolution]
        else:
            b = math.exp((math.log(self.finest_resolution) - math.log(self.base_resolution)) / (self.num_levels - 1))
            self._resolutions = [int(math.floor(self.base_resolution * (b ** l) + 1e-9)) for l in range(self.num_levels)]
            self._resolutions[-1] = self.finest_resolution

        grids = []
        for r in self._resolutions:
            g = nn.Parameter(torch.empty((r * r, self.feature_dim), dtype=self.param_dtype))
            nn.init.uniform_(g, a=-self.init_range, b=+self.init_range)
            grids.append(g)

        self.grids = nn.ParameterList(grids)
        self.output_dim = self.num_levels * self.feature_dim

        self.qat_enabled = False
        self.qat_bits = 8

    def enable_qat(self, bits: int = 8) -> None:
        self.qat_enabled = True
        self.qat_bits = int(bits)

    def disable_qat(self) -> None:
        self.qat_enabled = False

    @property
    def resolutions(self):
        return list(self._resolutions)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (B,2), got {tuple(coords.shape)}")

        coords = clamp_coords01(coords).to(dtype=torch.float32)
        feats = []
        x = coords[:, 0]
        y = coords[:, 1]

        for lvl, r in enumerate(self._resolutions):
            grid = self.grids[lvl]

            if self.qat_enabled:
                alpha = grid.min().detach()
                beta = grid.max().detach()
            else:
                alpha = beta = None

            xs = x * (r - 1)
            ys = y * (r - 1)

            x0 = torch.floor(xs).to(torch.int64).clamp(0, r - 2)
            y0 = torch.floor(ys).to(torch.int64).clamp(0, r - 2)
            fx = (xs - x0.to(xs.dtype)).unsqueeze(1)
            fy = (ys - y0.to(ys.dtype)).unsqueeze(1)

            x1 = x0 + 1
            y1 = y0 + 1

            idx00 = x0 + y0 * r
            idx10 = x1 + y0 * r
            idx01 = x0 + y1 * r
            idx11 = x1 + y1 * r

            f00 = grid[idx00].to(torch.float32)
            f10 = grid[idx10].to(torch.float32)
            f01 = grid[idx01].to(torch.float32)
            f11 = grid[idx11].to(torch.float32)

            if self.qat_enabled:
                f00 = _fake_quantize_asymmetric_with_range(f00, alpha, beta, bits=self.qat_bits)
                f10 = _fake_quantize_asymmetric_with_range(f10, alpha, beta, bits=self.qat_bits)
                f01 = _fake_quantize_asymmetric_with_range(f01, alpha, beta, bits=self.qat_bits)
                f11 = _fake_quantize_asymmetric_with_range(f11, alpha, beta, bits=self.qat_bits)

            f0 = f00 * (1.0 - fx) + f10 * fx
            f1 = f01 * (1.0 - fx) + f11 * fx
            f = f0 * (1.0 - fy) + f1 * fy

            feats.append(f.to(dtype=grid.dtype))

        return torch.cat(feats, dim=1)


# ---------- Color Network ----------

class ColorNetwork(nn.Module):
    """
    Predicts single-channel value(s) from 2D texture coordinates (u,v).
    If num_textures=T, outputs (B, T) with a sigmoid.
    """

    def __init__(
        self,
        num_textures: int = 1,
        param_dtype: torch.dtype = torch.float32,
        finest_resolution: int = 2048,
        base_resolution: int = 16,
        num_levels: int = 8,
    ):
        super().__init__()
        if num_textures < 1:
            raise ValueError("num_textures must be >= 1")
        self.num_textures = int(num_textures)

        self.encoding = MultiResFeatureGrid2D(
            num_levels=int(num_levels),
            base_resolution=int(base_resolution),
            finest_resolution=int(finest_resolution),
            feature_dim=2,
            init_range=1e-4,
            param_dtype=(torch.float16 if param_dtype == torch.float16 else torch.float32),
        )
        in_dim = self.encoding.output_dim

        # BC4: 1 channel per texture
        out_dim = 1 * self.num_textures
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SELU(inplace=True),
            nn.Linear(64, 64),
            nn.SELU(inplace=True),
            nn.Linear(64, 64),
            nn.SELU(inplace=True),
            nn.Linear(64, out_dim),
            nn.Sigmoid(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        feats = self.encoding(uv)
        out = self.mlp(feats.to(torch.float32))
        return out  # (B, T)


@dataclass
class ColorLossOutput:
    total: torch.Tensor
    lc: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor  # (B,) for single, (B,T) for multi


def color_loss_bc4(
    pred_value: torch.Tensor,       # (B,1) in [0,1]
    ref_value: torch.Tensor,        # (B,1) in [0,1]
    ref_endpoints2: torch.Tensor,   # (B,2) in [0,1]
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    if pred_value.shape != ref_value.shape:
        raise ValueError(f"pred/ref values must have same shape, got {pred_value.shape} vs {ref_value.shape}")
    if pred_value.ndim != 2 or pred_value.shape[1] != 1:
        raise ValueError(f"pred_value must be (B,1), got {tuple(pred_value.shape)}")
    if ref_endpoints2.ndim != 2 or ref_endpoints2.shape[1] != 2:
        raise ValueError(f"ref_endpoints2 must be (B,2), got {tuple(ref_endpoints2.shape)}")

    # L_c: MSE on predicted vs reference value
    lc = F.mse_loss(pred_value, ref_value, reduction=reduction)
    # Build BC4 palette from REFERENCE endpoints (BC4 selector order, with proper mode handling)
    pal = bc4_palette_bc4order(ref_endpoints2)  # (B,8)

    # d_n = - (c_hat - p_n)^2
    diff = pred_value - pal  # (B,8) broadcast from (B,1)
    dn = -(diff * diff)  # (B,8) squared L2

    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)  # (B,)

    # Hard decode (exactly matches BC4 selector decode)
    decoded_hard = torch.gather(pal, 1, hard_n.long().unsqueeze(1))  # (B,1)

    # Soft decode (for gradients)
    p = F.softmax(dn / float(temperature), dim=-1)  # (B,8)
    decoded_soft = (p * pal).sum(dim=-1, keepdim=True)  # (B,1)

    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())

    lcd = F.mse_loss(decoded, ref_value, reduction=reduction)
    total = lc + lcd
    return ColorLossOutput(total=total, lc=lc, lcd=lcd, hard_indices=hard_n)


def color_loss_bc4_multi(
    pred_values: torch.Tensor,      # (B,T)
    ref_values: torch.Tensor,       # (B,T)
    ref_endpoints: torch.Tensor,    # (B,2*T) or (B,T,2)
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    """Multi-texture BC4 color loss."""
    # pred_values / ref_values: (B,T) -> each texel has 1 scalar per texture
    if pred_values.ndim != 2:
        raise ValueError(f"pred_values must be (B,T). Got {tuple(pred_values.shape)}")
    B, T = pred_values.shape

    if ref_values.ndim != 2 or ref_values.shape != (B, T):
        raise ValueError(f"ref_values must be (B,T)={B,T}. Got {tuple(ref_values.shape)}")

    # Handle endpoints
    if ref_endpoints.ndim == 2:
        D = int(ref_endpoints.shape[1])
        if D % 2 != 0:
            raise ValueError(f"ref_endpoints second dim must be multiple of 2. Got {D}")
        T_e = D // 2
        ref_e = ref_endpoints.view(B, T_e, 2)
    elif ref_endpoints.ndim == 3 and ref_endpoints.shape[-1] == 2:
        ref_e = ref_endpoints
    else:
        raise ValueError(f"ref_endpoints must be (B,2*T) or (B,T,2). Got {tuple(ref_endpoints.shape)}")

    if ref_e.shape[:2] != (B, T):
        raise ValueError(f"ref_endpoints mismatch: expected (B,{T},2), got {tuple(ref_e.shape)}")

    totals, lcs, lcds, hards = [], [], [], []
    for t in range(T):
        out_t = color_loss_bc4(
            pred_values[:, t:t+1],    # (B,1)
            ref_values[:, t:t+1],     # (B,1)
            ref_e[:, t, :],           # (B,2)
            temperature=temperature,
            reduction=reduction,
        )
        totals.append(out_t.total)
        lcs.append(out_t.lc)
        lcds.append(out_t.lcd)
        hards.append(out_t.hard_indices)

    total = torch.stack(totals).sum()
    lc = torch.stack(lcs).sum()
    lcd = torch.stack(lcds).sum()
    hard_indices = torch.stack(hards, dim=1)  # (B,T)
    return ColorLossOutput(total=total, lc=lc, lcd=lcd, hard_indices=hard_indices)


if __name__ == "__main__":
    print("Quick test: multi-texture BC4 ColorNetwork forward + loss shapes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 2
    net = ColorNetwork(num_textures=T).to(device)

    B = 32
    uv = torch.rand(B, 2, device=device)
    pred = net(uv)  # (B,T)

    ref_v = torch.rand(B, T, device=device)
    ref_ep = torch.rand(B, 2*T, device=device)

    out = color_loss_bc4_multi(pred, ref_v, ref_ep)
    print("pred:", pred.shape)
    print("loss:", out.total.item(), "lc:", out.lc.item(), "lcd:", out.lcd.item(), "indices:", out.hard_indices.shape)
