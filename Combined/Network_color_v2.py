"""
NTBC Color Network for BC1 compression (multi-RGB-texture capable).
Based on: Neural Texture Block Compression (arXiv:2407.09543)

What changed vs your original:
- Supports training a *single* color model that predicts colors for multiple RGB textures jointly.
- If num_textures=T, the network outputs (3*T) values per texel:
    [r,g,b] repeated T times (each in [0,1]).
- Provides color_loss_bc1_multi(...) which loops over textures and averages losses.

Backwards compatible:
- num_textures defaults to 1, so your single-texture pipeline still works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Quantization helpers (paper Sec. 3.3 / Eq. 2–6) ----------

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


# ---------- BC1 helpers ----------

_BC1_W = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return coords.clamp(0.0, 1.0 - eps)


def endpoints6_to_e0e1(endpoints6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e0 = endpoints6[..., 0:3]
    e1 = endpoints6[..., 3:6]
    return e0, e1


def bc1_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wv = w.view(*([1] * (e0.ndim - 1)), -1, 1).to(device=e0.device, dtype=e0.dtype)
    e0v = e0.unsqueeze(-2)
    e1v = e1.unsqueeze(-2)
    return (1.0 - wv) * e0v + wv * e1v  # (...,4,3)


def _infer_num_textures_from_flat_colors(colors_flat: torch.Tensor) -> int:
    if colors_flat.ndim != 2:
        raise ValueError(f"colors_flat must be (B, 3*T). Got {tuple(colors_flat.shape)}")
    D = int(colors_flat.shape[1])
    if D % 3 != 0:
        raise ValueError(f"Color dimension must be multiple of 3. Got {D}")
    return D // 3


def split_colors_flat(colors_flat: torch.Tensor) -> torch.Tensor:
    """(B,3*T) -> (B,T,3)"""
    T = _infer_num_textures_from_flat_colors(colors_flat)
    return colors_flat.view(colors_flat.shape[0], T, 3)


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
    Predicts uncompressed RGB color(s) c_hat from 2D texture coordinates (u,v).
    If num_textures=T, outputs (B, 3*T) with a sigmoid.
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

        out_dim = 3 * self.num_textures
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
        return out


@dataclass
class ColorLossOutput:
    total: torch.Tensor
    lc: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor  # (B,) for single, (B,T) for multi


def color_loss_bc1(
    pred_color: torch.Tensor,       # (B,3) in [0,1]
    ref_color: torch.Tensor,        # (B,3) in [0,1]
    ref_endpoints6: torch.Tensor,   # (B,6) in [0,1]
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    if pred_color.shape != ref_color.shape:
        raise ValueError(f"pred/ref colors must have same shape, got {pred_color.shape} vs {ref_color.shape}")
    if pred_color.ndim != 2 or pred_color.shape[1] != 3:
        raise ValueError(f"pred_color must be (B,3), got {tuple(pred_color.shape)}")
    if ref_endpoints6.ndim != 2 or ref_endpoints6.shape[1] != 6:
        raise ValueError(f"ref_endpoints6 must be (B,6), got {tuple(ref_endpoints6.shape)}")

    lc = F.mse_loss(pred_color, ref_color, reduction=reduction)

    # Palette from REFERENCE endpoints (paper: color net uses ref endpoints + predicted colors)
    ref_e0, ref_e1 = endpoints6_to_e0e1(ref_endpoints6)
    w_levels = _BC1_W.to(device=pred_color.device, dtype=pred_color.dtype)
    pal = bc1_palette_from_endpoints(ref_e0, ref_e1, w=w_levels)  # (B,4,3)

    # d_n = -||c_hat - c_n|| (Eq. 9)
    diff = pred_color.unsqueeze(1) - pal  # (B,4,3)
    dn = -torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (B,4)

    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)  # (B,)

    # Hard decode (forward)
    w_hard = w_levels[hard_n.long()]  # (B,)
    decoded_hard = (1.0 - w_hard).unsqueeze(-1) * ref_e0 + w_hard.unsqueeze(-1) * ref_e1  # (B,3)

    # Soft decode (backward)
    p = F.softmax(dn / float(temperature), dim=-1)  # (B,4)
    w_soft = (p * w_levels.view(1, 4)).sum(dim=-1)  # (B,)
    decoded_soft = (1.0 - w_soft).unsqueeze(-1) * ref_e0 + w_soft.unsqueeze(-1) * ref_e1  # (B,3)

    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())

    lcd = F.mse_loss(decoded, ref_color, reduction=reduction)
    total = lc + lcd
    return ColorLossOutput(total=total, lc=lc, lcd=lcd, hard_indices=hard_n)


def color_loss_bc1_multi(
    pred_colors: torch.Tensor,      # (B,3*T) or (B,T,3)
    ref_colors: torch.Tensor,       # (B,3*T) or (B,T,3)
    ref_endpoints: torch.Tensor,    # (B,6*T) or (B,T,6)
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    """
    Multi-texture color loss: average over textures of (L_c + L_cd).

    Shapes:
      pred_colors  : (B,3*T) or (B,T,3)
      ref_colors   : (B,3*T) or (B,T,3)
      ref_endpoints: (B,6*T) or (B,T,6)
    """
    if pred_colors.ndim == 2:
        pred_c = split_colors_flat(pred_colors)  # (B,T,3)
    elif pred_colors.ndim == 3 and pred_colors.shape[-1] == 3:
        pred_c = pred_colors
    else:
        raise ValueError(f"pred_colors must be (B,3*T) or (B,T,3). Got {tuple(pred_colors.shape)}")

    if ref_colors.ndim == 2:
        ref_c = split_colors_flat(ref_colors)
    elif ref_colors.ndim == 3 and ref_colors.shape[-1] == 3:
        ref_c = ref_colors
    else:
        raise ValueError(f"ref_colors must be (B,3*T) or (B,T,3). Got {tuple(ref_colors.shape)}")

    if ref_endpoints.ndim == 2:
        # (B,6*T) -> (B,T,6)
        D = int(ref_endpoints.shape[1])
        if D % 6 != 0:
            raise ValueError(f"ref_endpoints second dim must be multiple of 6. Got {D}")
        T = D // 6
        ref_e = ref_endpoints.view(ref_endpoints.shape[0], T, 6)
    elif ref_endpoints.ndim == 3 and ref_endpoints.shape[-1] == 6:
        ref_e = ref_endpoints
    else:
        raise ValueError(f"ref_endpoints must be (B,6*T) or (B,T,6). Got {tuple(ref_endpoints.shape)}")

    B, T, _ = pred_c.shape
    if ref_c.shape[:2] != (B, T):
        raise ValueError(f"pred/ref colors mismatch: {tuple(pred_c.shape)} vs {tuple(ref_c.shape)}")
    if ref_e.shape[:2] != (B, T):
        raise ValueError(f"ref_endpoints mismatch: expected (B,T,6), got {tuple(ref_e.shape)}")

    totals, lcs, lcds, hards = [], [], [], []
    for t in range(T):
        out_t = color_loss_bc1(
            pred_c[:, t, :],
            ref_c[:, t, :],
            ref_e[:, t, :],
            temperature=temperature,
            reduction=reduction,
        )
        totals.append(out_t.total)
        lcs.append(out_t.lc)
        lcds.append(out_t.lcd)
        hards.append(out_t.hard_indices)

    total = torch.stack(totals).mean()
    lc = torch.stack(lcs).mean()
    lcd = torch.stack(lcds).mean()
    hard_indices = torch.stack(hards, dim=1)  # (B,T)
    return ColorLossOutput(total=total, lc=lc, lcd=lcd, hard_indices=hard_indices)


if __name__ == "__main__":
    print("Quick test: multi-texture ColorNetwork forward + loss shapes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 3
    net = ColorNetwork(num_textures=T).to(device)

    B = 32
    uv = torch.rand(B, 2, device=device)
    pred = net(uv)  # (B,3*T)

    ref_c = torch.rand(B, 3*T, device=device)
    ref_ep = torch.rand(B, 6*T, device=device)

    out = color_loss_bc1_multi(pred, ref_c, ref_ep)
    print("pred:", pred.shape)
    print("loss:", out.total.item(), "lc:", out.lc.item(), "lcd:", out.lcd.item(), "indices:", out.hard_indices.shape)
