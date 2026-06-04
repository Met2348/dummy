"""Mamba-2 SSD-form block — naive 教学版 (chunk matmul)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_block import selective_scan_naive


class Mamba2Block(nn.Module):
    """Mamba-2 style block — 简化版，复用 selective_scan_naive."""
    def __init__(self, d_model: int, d_state: int = 16, chunk_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        A_init = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A_init.unsqueeze(0).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)
        dt = F.softplus(self.dt(x))
        B = self.B_proj(x)
        C = self.C_proj(x)
        A = -torch.exp(self.A_log)
        y = selective_scan_naive(u, dt, A, B, C, self.D)
        return self.out_proj(y)


if __name__ == "__main__":
    m = Mamba2Block(d_model=16, d_state=8)
    x = torch.randn(1, 10, 16)
    print(f"Mamba2 out {m(x).shape}, params {sum(p.numel() for p in m.parameters())}")
