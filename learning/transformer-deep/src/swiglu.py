"""SwiGLU / GeGLU MLP — Shazeer 2020 / Llama-2 标准."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """Llama-2 style SwiGLU MLP."""
    def __init__(self, d_model: int, d_ff: int | None = None,
                 bias: bool = False):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)  # Llama-2 ratio
        self.w_g = nn.Linear(d_model, d_ff, bias=bias)
        self.w_v = nn.Linear(d_model, d_ff, bias=bias)
        self.w_o = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_o(F.silu(self.w_g(x)) * self.w_v(x))


class GeGLUMLP(nn.Module):
    """GeGLU 变体 — Llama-1 用前一代."""
    def __init__(self, d_model: int, d_ff: int | None = None,
                 bias: bool = False):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        self.w_g = nn.Linear(d_model, d_ff, bias=bias)
        self.w_v = nn.Linear(d_model, d_ff, bias=bias)
        self.w_o = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w_o(F.gelu(self.w_g(x)) * self.w_v(x))


class GELUMLP(nn.Module):
    """传统 4x GELU MLP (GPT-2 风格)."""
    def __init__(self, d_model: int, d_ff: int | None = None,
                 bias: bool = True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


if __name__ == "__main__":
    d = 64
    x = torch.randn(2, 8, d)
    for cls in [SwiGLUMLP, GeGLUMLP, GELUMLP]:
        m = cls(d_model=d)
        y = m(x)
        n_p = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__:10s} out {tuple(y.shape)}  params {n_p}")
