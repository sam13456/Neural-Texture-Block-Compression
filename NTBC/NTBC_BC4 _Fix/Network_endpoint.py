"""
NTBC Endpoint Network for BC4 compression (multi-single-channel-texture capable).
Based on: Neural Texture Block Compression (arXiv:2407.09543)

BC4 stores two 8-bit scalar endpoints (e0, e1) per 4x4 block.
The palette has 8 entries, but BC4 has TWO modes:
  - e0 > e1 : 8-value interpolation
  - e0 <= e1: 6-value interpolation + two special values (0 and 1)
The NTBC paper requires handling both cases (Eq. 8).

If num_textures=T, the network outputs (2*T) values per block:
    [e0, e1] repeated T times (each in [0,1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _fake_quantize_u8_ste(endpoints2: torch.Tensor) -> torch.Tensor:
    """Fake-quantize 2-channel BC4 endpoints to 8-bit (256 levels) with STE.

    Input/output: (..., 2) in [0,1] = [e0, e1]
    Each channel: round(x*255)/255
    """
    levels = 255.0
    scaled = (endpoints2 * levels).round()
    clamped = scaled.clamp(0, levels)
    quantized = clamped / levels
    return endpoints2 + (quantized - endpoints2).detach()


# ---------- Utilities (BC4) ----------

# BC4 palette weights: w_n = n/7 for n=0..7
_BC4_W = torch.tensor([i / 7.0 for i in range(8)], dtype=torch.float32)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return coords.clamp(0.0, 1.0 - eps)


def endpoints2_to_e0e1(endpoints2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """endpoints2: (...,2) = [e0, e1] in [0,1] -> e0, e1: (...,1), (...,1)"""
    e0 = endpoints2[..., 0:1]
    e1 = endpoints2[..., 1:2]
    return e0, e1


def bc4_palette_bc4order(endpoints2: torch.Tensor) -> torch.Tensor:
    """Build BC4 palette in *BC4 selector order* (0..7).

    endpoints2: (...,2) in [0,1] where endpoints2[...,0]=e0 and endpoints2[...,1]=e1.

    Returns:
      palette: (...,8) in [0,1] such that palette[...,sel] matches BC4 decoding.

    Mode rules:
      - if e0 > e1 (8-value mode):
          sel0=e0, sel1=e1,
          sel2..7 = ((7-k)*e0 + k*e1)/7 for k=1..6
      - else (6-value mode):
          sel0=e0, sel1=e1,
          sel2..5 = ((5-k)*e0 + k*e1)/5 for k=1..4,
          sel6=0, sel7=1
    """
    if endpoints2.shape[-1] != 2:
        raise ValueError(f"endpoints2 last dim must be 2, got {tuple(endpoints2.shape)}")

    e0 = endpoints2[..., 0]
    e1 = endpoints2[..., 1]

    # Mode-8 palette (e0 > e1)
    t8 = endpoints2.new_tensor([1, 2, 3, 4, 5, 6])
    vals8 = ((7.0 - t8) * e0.unsqueeze(-1) + t8 * e1.unsqueeze(-1)) / 7.0  # (...,6)
    pal8 = torch.empty((*endpoints2.shape[:-1], 8), device=endpoints2.device, dtype=endpoints2.dtype)
    pal8[..., 0] = e0
    pal8[..., 1] = e1
    pal8[..., 2:8] = vals8

    # Mode-6 palette (e0 <= e1)
    t6 = endpoints2.new_tensor([1, 2, 3, 4])
    vals6 = ((5.0 - t6) * e0.unsqueeze(-1) + t6 * e1.unsqueeze(-1)) / 5.0  # (...,4)
    pal6 = torch.empty((*endpoints2.shape[:-1], 8), device=endpoints2.device, dtype=endpoints2.dtype)
    pal6[..., 0] = e0
    pal6[..., 1] = e1
    pal6[..., 2:6] = vals6
    pal6[..., 6] = torch.zeros((), device=endpoints2.device, dtype=endpoints2.dtype)
    pal6[..., 7] = torch.ones((), device=endpoints2.device, dtype=endpoints2.dtype)

    mode8 = (e0 > e1).unsqueeze(-1)
    return torch.where(mode8, pal8, pal6)


def bc4_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Build BC4 8-entry palette from two scalar endpoints.

    e0, e1: (...,1)  w: (8,)
    returns palette: (...,8,1)
    """
    wv = w.view(*([1] * (e0.ndim - 1)), -1, 1).to(device=e0.device, dtype=e0.dtype)
    e0v = e0.unsqueeze(-2)
    e1v = e1.unsqueeze(-2)
    return (1.0 - wv) * e0v + wv * e1v


def pack_u8_from_epq01(endpoints2: torch.Tensor) -> torch.Tensor:
    """Converts predicted endpoints to uint8 format. endpoints2: (B,2) -> (B,2) uint8"""
    return (endpoints2 * 255.0).round().clamp(0, 255).to(torch.uint8)


def _infer_num_textures_from_flat(endpoints_flat: torch.Tensor) -> int:
    if endpoints_flat.ndim != 2:
        raise ValueError(f"endpoints_flat must be (B, 2*T). Got {tuple(endpoints_flat.shape)}")
    D = int(endpoints_flat.shape[1])
    if D % 2 != 0:
        raise ValueError(f"Endpoint dimension must be multiple of 2. Got {D}")
    return D // 2


def split_endpoints_flat(endpoints_flat: torch.Tensor) -> torch.Tensor:
    """(B,2*T) -> (B,T,2)"""
    T = _infer_num_textures_from_flat(endpoints_flat)
    return endpoints_flat.view(endpoints_flat.shape[0], T, 2)


# ---------- Multi-resolution Feature Grids ----------

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


# ---------- Endpoint Network ----------

class EndpointNetwork(nn.Module):
    """Predicts BC4 endpoints (2 scalars per texture) from normalized block coordinates."""

    def __init__(
        self,
        num_textures: int = 1,
        num_levels: int = 7,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        feature_dim: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        if num_textures < 1:
            raise ValueError("num_textures must be >= 1")
        self.num_textures = int(num_textures)

        self.encoding = MultiResFeatureGrid2D(
            num_levels=num_levels,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            feature_dim=feature_dim,
            init_range=1e-4,
            param_dtype=param_dtype,
        )

        # BC4: 2 endpoints per texture (e0, e1 scalars)
        out_dim = 2 * self.num_textures

        layers = []
        in_dim = self.encoding.output_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        self._init_mlp_he()

        if param_dtype == torch.float16:
            self.mlp = self.mlp.half()

        self.register_buffer("bc4_w", _BC4_W.clone(), persistent=False)

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
    def predict_u8(self, coords: torch.Tensor) -> torch.Tensor:
        """Predicts and quantizes endpoints to uint8.

        Returns:
          - if num_textures==1: (B,2) uint8
          - else:              (B,T,2) uint8
        """
        ep_flat = self.forward(coords)  # (B,2*T)
        if self.num_textures == 1:
            return pack_u8_from_epq01(ep_flat)
        ep = ep_flat.view(ep_flat.shape[0], self.num_textures, 2)
        outs = []
        for t in range(self.num_textures):
            outs.append(pack_u8_from_epq01(ep[:, t, :]))
        return torch.stack(outs, dim=1)


# ---------- BC4 endpoint loss (L_e + L_cd) ----------

@dataclass
class EndpointLossOutput:
    total: torch.Tensor
    le: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor  # (B,16) for single, (B,T,16) for multi


def endpoint_loss_bc4(
    pred_endpoints2: torch.Tensor,    # (B,2) in [0,1]
    ref_endpoints2: torch.Tensor,     # (B,2) in [0,1]
    ref_values: torch.Tensor,         # (B,16) single-channel pixel values in [0,1]
    temperature: float = 0.01,
    reduction: str = "mean",
) -> EndpointLossOutput:
    """
    Computes BC4 endpoint loss: L_e + L_cd.

    BC4 has two modes; palette/index/decoding must follow BC4 rules (paper Eq. 8).
    """
    if pred_endpoints2.shape != ref_endpoints2.shape:
        raise ValueError(f"pred/ref endpoints must have same shape, got {pred_endpoints2.shape} vs {ref_endpoints2.shape}")
    if ref_values.ndim != 2 or ref_values.shape[1] != 16:
        raise ValueError(f"ref_values must be (B,16), got {tuple(ref_values.shape)}")

    # L_e: MSE between predicted and reference endpoints
    le = F.mse_loss(pred_endpoints2, ref_endpoints2, reduction=reduction)

    # Fake-quantize predicted endpoints to uint8 (STE)
    pred_q = _fake_quantize_u8_ste(pred_endpoints2)
    # Build BC4 palette (BC4 selector order) from quantized endpoints
    pal_pred = bc4_palette_bc4order(pred_q)  # (B,8)

    # Squared L2 Distance: d_n = - (v - p_n)^2
    diff = ref_values.unsqueeze(2) - pal_pred.unsqueeze(1)  # (B,16,8)
    dn = -(diff * diff)  # (B,16,8) squared L2

    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)  # (B,16)

    # Decode using PREDICTED palette (so gradients flow smoothly into endpoints)
    # Hard decode (exactly matches BC4 selector decode)
    decoded_hard = torch.gather(pal_pred, 1, hard_n.long())  # (B,16)

    # Soft decode (for gradients)
    p = F.softmax(dn / float(temperature), dim=-1)  # (B,16,8)
    decoded_soft = (p * pal_pred.unsqueeze(1)).sum(dim=-1)  # (B,16)

    # STE combine
    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())

    lcd = F.mse_loss(decoded, ref_values, reduction=reduction)

    total = le + lcd
    return EndpointLossOutput(total=total, le=le, lcd=lcd, hard_indices=hard_n)


def endpoint_loss_bc4_multi(
    pred_endpoints: torch.Tensor,   # (B,2*T) or (B,T,2)
    ref_endpoints: torch.Tensor,    # (B,2*T) or (B,T,2)
    ref_values: torch.Tensor,       # (B,T,16) single-channel pixel values
    temperature: float = 0.01,
    reduction: str = "mean",
) -> EndpointLossOutput:
    """Multi-texture version of BC4 endpoint loss."""
    if pred_endpoints.ndim == 2:
        pred_e = split_endpoints_flat(pred_endpoints)  # (B,T,2)
    elif pred_endpoints.ndim == 3 and pred_endpoints.shape[-1] == 2:
        pred_e = pred_endpoints
    else:
        raise ValueError(f"pred_endpoints must be (B,2*T) or (B,T,2). Got {tuple(pred_endpoints.shape)}")

    if ref_endpoints.ndim == 2:
        ref_e = split_endpoints_flat(ref_endpoints)
    elif ref_endpoints.ndim == 3 and ref_endpoints.shape[-1] == 2:
        ref_e = ref_endpoints
    else:
        raise ValueError(f"ref_endpoints must be (B,2*T) or (B,T,2). Got {tuple(ref_endpoints.shape)}")

    B, T, _ = pred_e.shape
    if ref_e.shape[:2] != (B, T):
        raise ValueError(f"pred/ref endpoints mismatch: {tuple(pred_e.shape)} vs {tuple(ref_e.shape)}")

    if T == 1 and ref_values.ndim == 2:
        ref_v = ref_values.unsqueeze(1)  # (B,1,16)
    elif ref_values.ndim == 3 and ref_values.shape[1] == T:
        ref_v = ref_values
    else:
        raise ValueError(f"ref_values must be (B,T,16) (or (B,16) when T=1). Got {tuple(ref_values.shape)}")

    totals, les, lcds, hards = [], [], [], []
    for t in range(T):
        out_t = endpoint_loss_bc4(
            pred_e[:, t, :],
            ref_e[:, t, :],
            ref_v[:, t, :],
            temperature=temperature,
            reduction=reduction,
        )
        totals.append(out_t.total)
        les.append(out_t.le)
        lcds.append(out_t.lcd)
        hards.append(out_t.hard_indices)

    total = torch.stack(totals).sum()
    le = torch.stack(les).sum()
    lcd = torch.stack(lcds).sum()
    hard_indices = torch.stack(hards, dim=1)  # (B,T,16)
    return EndpointLossOutput(total=total, le=le, lcd=lcd, hard_indices=hard_indices)


if __name__ == "__main__":
    print("Quick test: multi-texture BC4 EndpointNetwork forward + loss shapes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 2
    net = EndpointNetwork(num_textures=T, param_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    B = 8
    coords = torch.rand(B, 2, device=device)
    pred = net(coords)  # (B,2*T)

    ref_ep = torch.rand(B, 2*T, device=device)
    ref_values = torch.rand(B, T, 16, device=device)

    out = endpoint_loss_bc4_multi(pred, ref_ep, ref_values)
    print("pred:", pred.shape, pred.min().item(), pred.max().item())
    print("loss:", out.total.item(), "le:", out.le.item(), "lcd:", out.lcd.item(), "indices:", out.hard_indices.shape)
