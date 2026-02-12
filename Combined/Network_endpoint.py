"""
NTBC Endpoint Network for BC1 compression.
Based on: Neural Texture Block Compression (arXiv:2407.09543v2)

Multi-resolution feature grids + MLP to predict BC1 endpoints from 2D block coordinates.
Includes the paper's endpoint training loss (L_e + L_cd).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


# ---------- Utilities (BC1) ----------

_BC1_W = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32)  # w_n = n/3 for n=0..3 (Eq. 7)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp normalized coords to [0, 1-eps] to avoid edge indexing issues at exactly 1.0."""
    return coords.clamp(0.0, 1.0 - eps)


def endpoints6_to_e0e1(endpoints6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    endpoints6: (B,6) = [r0,g0,b0,r1,g1,b1] in [0,1]
    returns e0,e1: (B,3), (B,3)
    """
    e0 = endpoints6[:, 0:3]
    e1 = endpoints6[:, 3:6]
    return e0, e1


def bc1_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Builds BC1 palette colors from endpoints using linear interpolation."""
    wv = w.view(1, -1, 1).to(device=e0.device, dtype=e0.dtype)
    e0v = e0.unsqueeze(1)
    e1v = e1.unsqueeze(1)
    return (1.0 - wv) * e0v + wv * e1v


def pack_rgb565_from_epq01(endpoints6: torch.Tensor) -> torch.Tensor:
    """Converts predicted endpoints to packed RGB565 format."""
    r0 = (endpoints6[:, 0] * 31.0).round().clamp(0, 31).to(torch.int32)
    g0 = (endpoints6[:, 1] * 63.0).round().clamp(0, 63).to(torch.int32)
    b0 = (endpoints6[:, 2] * 31.0).round().clamp(0, 31).to(torch.int32)
    r1 = (endpoints6[:, 3] * 31.0).round().clamp(0, 31).to(torch.int32)
    g1 = (endpoints6[:, 4] * 63.0).round().clamp(0, 63).to(torch.int32)
    b1 = (endpoints6[:, 5] * 31.0).round().clamp(0, 31).to(torch.int32)

    c0 = (r0 << 11) | (g0 << 5) | b0
    c1 = (r1 << 11) | (g1 << 5) | b1
    out = torch.stack([c0, c1], dim=1).to(torch.uint16)
    return out


# ---------- Multi-resolution Feature Grids  ----------

class MultiResFeatureGrid2D(nn.Module):
    """Multi-resolution feature grids with bilinear interpolation."""

    def __init__(
        self,
        num_levels: int = 7,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
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

        # QAT 
        self.qat_enabled = False
        self.qat_bits = 8

    def enable_qat(self, bits: int = 8) -> None:
        """Enables fake quantization for feature grids."""
        self.qat_enabled = True
        self.qat_bits = int(bits)

    def disable_qat(self) -> None:
        """Disables fake quantization."""
        self.qat_enabled = False

    @property
    def resolutions(self):
        return list(self._resolutions)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Returns concatenated features from all grid levels."""
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (B,2), got {tuple(coords.shape)}")

        coords = clamp_coords01(coords).to(dtype=torch.float32)
        B = coords.shape[0]
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


# ---------- Endpoint Network  ----------

class EndpointNetwork(nn.Module):
    """Predicts BC1 endpoints from normalized block coordinates."""

    def __init__(
        self,
        num_levels: int = 7,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        feature_dim: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.encoding = MultiResFeatureGrid2D(
            num_levels=num_levels,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            feature_dim=feature_dim,
            init_range=1e-4,
            param_dtype=param_dtype,
        )

        layers = []
        in_dim = self.encoding.output_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, 6))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        self._init_mlp_he()

        if param_dtype == torch.float16:
            self.mlp = self.mlp.half()

        self.register_buffer("bc1_w", _BC1_W.clone(), persistent=False)

    def _init_mlp_he(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoding(coords)
        return self.mlp(feats)

    @torch.no_grad()
    def predict_rgb565(self, coords: torch.Tensor) -> torch.Tensor:
        """Predicts and quantizes endpoints to RGB565 format."""
        ep = self.forward(coords)
        return pack_rgb565_from_epq01(ep)


# ---------- Paper-aligned endpoint loss (L_e + L_cd) ----------

@dataclass
class EndpointLossOutput:
    total: torch.Tensor
    le: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor  


def endpoint_loss_bc1(
    pred_endpoints6: torch.Tensor,
    ref_endpoints6: torch.Tensor,
    ref_colors: torch.Tensor,
    temperature: float = 0.01,
    reduction: str = "mean",
) -> EndpointLossOutput:
    """
    Computes endpoint training loss: L_e + L_cd.
    
    Uses straight-through estimator for gradient flow through argmax.
    """
    if pred_endpoints6.shape != ref_endpoints6.shape:
        raise ValueError(f"pred/ref endpoints must have same shape, got {pred_endpoints6.shape} vs {ref_endpoints6.shape}")
    if ref_colors.ndim != 3 or ref_colors.shape[1:] != (16, 3):
        raise ValueError(f"ref_colors must be (B,16,3), got {tuple(ref_colors.shape)}")

    # L_e: MSE between predicted and reference endpoints (paper: L2)
    le = F.mse_loss(pred_endpoints6, ref_endpoints6, reduction=reduction)

    # Build palette from predicted endpoints (for distance computation)
    pred_e0, pred_e1 = endpoints6_to_e0e1(pred_endpoints6)
    w = _BC1_W.to(device=pred_endpoints6.device, dtype=pred_endpoints6.dtype)
    pal_pred = bc1_palette_from_endpoints(pred_e0, pred_e1, w=w)

    # Distances d_n = -||c - c_hat_n||  (Eq. 9)
    # (B,16,4)
    diff = ref_colors.unsqueeze(2) - pal_pred.unsqueeze(1)
    dn = -torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)

    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)

    # Decode with reference endpoints using Eq.7 (paper says use reference endpoints for final decoded colors)
    ref_e0, ref_e1 = endpoints6_to_e0e1(ref_endpoints6)
    w_levels = _BC1_W.to(device=ref_endpoints6.device, dtype=ref_endpoints6.dtype)
    w_hard = w_levels[hard_n.long()]
    decoded_hard = (1.0 - w_hard).unsqueeze(-1) * ref_e0.unsqueeze(1) + w_hard.unsqueeze(-1) * ref_e1.unsqueeze(1)

    p = F.softmax(dn / float(temperature), dim=-1)
    w_soft = (p * w_levels.view(1, 1, 4)).sum(dim=-1)
    decoded_soft = (1.0 - w_soft).unsqueeze(-1) * ref_e0.unsqueeze(1) + w_soft.unsqueeze(-1) * ref_e1.unsqueeze(1)

    # STE combine (forward hard, backward soft)
    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())

    lcd = F.mse_loss(decoded, ref_colors, reduction=reduction)

    total = le + lcd
    return EndpointLossOutput(total=total, le=le, lcd=lcd, hard_indices=hard_n)


if __name__ == "__main__":
    print("Quick test: EndpointNetwork forward + loss shapes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = EndpointNetwork(param_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    B = 8
    coords = torch.rand(B, 2, device=device)
    pred = net(coords)

    ref_ep = torch.rand(B, 6, device=device)
    ref_colors = torch.rand(B, 16, 3, device=device)

    out = endpoint_loss_bc1(pred, ref_ep, ref_colors)
    print("pred:", pred.shape, pred.min().item(), pred.max().item())
    print("loss:", out.total.item(), "le:", out.le.item(), "lcd:", out.lcd.item(), "indices:", out.hard_indices.shape)
