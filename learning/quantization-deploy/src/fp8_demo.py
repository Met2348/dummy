"""FP8 demo — emulated quantization without requiring Hopper hardware.

We round to the nearest representable E4M3 value via a precomputed table.
This stays accurate for the workshop without depending on torch>=2.5 fp8
ops or H100 GPUs.
"""
from __future__ import annotations

import math
import torch


def _build_e4m3_table() -> torch.Tensor:
    """Enumerate all 256 E4M3 representations and return their fp32 values."""
    vals = set()
    for sign in (0, 1):
        for exp in range(16):       # 4-bit exponent
            for mant in range(8):   # 3-bit mantissa
                if exp == 0:
                    val = 2 ** (1 - 7) * (mant / 8)
                else:
                    val = 2 ** (exp - 7) * (1 + mant / 8)
                if sign == 1:
                    val = -val
                vals.add(val)
    return torch.tensor(sorted(vals))


_TABLE = None


def fp8_round(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest E4M3 representable values."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_e4m3_table()
    flat = x.flatten()
    idx = torch.bucketize(flat, _TABLE)
    idx = idx.clamp(1, len(_TABLE) - 1)
    low = _TABLE[idx - 1]
    high = _TABLE[idx]
    pick = torch.where((flat - low).abs() < (high - flat).abs(), low, high)
    return pick.reshape(x.shape)


def fp8_matmul_mock(W: torch.Tensor, X: torch.Tensor, w_scale: float = 1.0, x_scale: float = 1.0) -> torch.Tensor:
    """Quantize both operands to FP8, do fp32 matmul, return scaled fp32."""
    W_q = fp8_round(W / w_scale) * w_scale
    X_q = fp8_round(X / x_scale) * x_scale
    return W_q @ X_q.t()


def relative_error(approx: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((approx - ref).norm() / max(ref.norm().item(), 1e-9)).item())
