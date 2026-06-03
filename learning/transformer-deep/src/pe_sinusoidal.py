"""Sinusoidal Positional Encoding (Vaswani 2017)."""
from __future__ import annotations

import math

import torch


def sinusoidal_pe(t: int, d: int, base: float = 10000.0,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """Return PE matrix of shape (t, d)."""
    pos = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)    # (t, 1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=dtype)
                    * -(math.log(base) / d))                           # (d/2,)
    pe = torch.zeros(t, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def add_sinusoidal(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: (b, t, d)  →  x + PE(t, d) (broadcast)."""
    b, t, d = x.shape
    pe = sinusoidal_pe(t, d, base=base, device=x.device, dtype=x.dtype)
    return x + pe.unsqueeze(0)


if __name__ == "__main__":
    pe = sinusoidal_pe(t=8, d=16)
    print("shape:", pe.shape)
    print("first row (pos=0):", pe[0])
    print("symmetry check: PE[0,0::2] all sin(0)=0 →", pe[0, ::2].abs().max().item())
