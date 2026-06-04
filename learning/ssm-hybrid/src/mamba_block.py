"""Mamba block — naive selective scan (无 CUDA kernel).

教学版，性能差但数学等价于 mamba-ssm CUDA kernel。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_naive(u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                          B: torch.Tensor, C: torch.Tensor,
                          D: torch.Tensor | None = None) -> torch.Tensor:
    """Naive selective scan.

    u:  (b, t, d)
    dt: (b, t, d)
    A:  (d, d_state)
    B:  (b, t, d_state)
    C:  (b, t, d_state)
    D:  (d,) skip connection (optional)
    Returns y: (b, t, d)
    """
    b, t, d = u.shape
    d_state = A.shape[-1]
    h = torch.zeros(b, d, d_state, device=u.device, dtype=u.dtype)
    out = []
    for i in range(t):
        # dA: (b, d, d_state)
        dA = torch.exp(dt[:, i].unsqueeze(-1) * A.unsqueeze(0))
        dB = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1)  # (b, d, d_state)
        h = h * dA + dB * u[:, i].unsqueeze(-1)              # (b, d, d_state)
        y = (h * C[:, i].unsqueeze(1)).sum(-1)               # (b, d)
        out.append(y)
    y_out = torch.stack(out, dim=1)
    if D is not None:
        y_out = y_out + u * D
    return y_out


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                              padding=d_conv - 1, groups=self.d_inner)
        # 投影出 dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state,
                                bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # A in log space (negative diagonal)
        A_init = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A_init.unsqueeze(0).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        # conv 1d on inner
        x_conv = self.conv(x_in.transpose(1, 2))[:, :, :t].transpose(1, 2)
        x_act = F.silu(x_conv)
        # selective params
        x_dbl = self.x_proj(x_act)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = selective_scan_naive(x_act, dt, A, B, C, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MambaBlock(d_model=32, d_state=8, d_conv=4)
    x = torch.randn(1, 16, 32)
    y = m(x)
    print(f"Mamba out {y.shape}, params {sum(p.numel() for p in m.parameters())}")
