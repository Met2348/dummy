"""Infini-Attention — 简化 compressive memory."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiniBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int = 4):
        super().__init__()
        self.h = n_head
        self.d_h = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(n_head))

    def forward(self, x: torch.Tensor, M_prev: torch.Tensor | None = None,
                Z_prev: torch.Tensor | None = None):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d_h).transpose(1, 2)

        # 1. local attention (causal)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h)
        mask = torch.full((t, t), float("-inf"), device=x.device,
                          dtype=x.dtype).triu_(1)
        local = F.softmax(scores + mask, dim=-1) @ v

        # 2. memory retrieval (使用 sigmoid 替 softmax)
        sig_k = torch.sigmoid(k)
        sig_q = torch.sigmoid(q)
        # M shape: (b, h, d_h, d_h);  Z shape: (b, h, d_h)
        if M_prev is None:
            M_prev = torch.zeros(b, self.h, self.d_h, self.d_h,
                                  device=x.device, dtype=x.dtype)
            Z_prev = torch.zeros(b, self.h, self.d_h,
                                  device=x.device, dtype=x.dtype)
        # retrieval
        retrieval = torch.einsum("bhtd,bhde->bhte", sig_q, M_prev) / (
            torch.einsum("bhtd,bhd->bht", sig_q, Z_prev).unsqueeze(-1) + 1e-6
        )

        # 3. update memory
        M_new = M_prev + torch.einsum("bhtd,bhte->bhde", sig_k, v)
        Z_new = Z_prev + sig_k.sum(dim=-2)

        # 4. 混合 local + retrieval (per-head gate)
        gate = torch.sigmoid(self.gate).view(1, self.h, 1, 1)
        out = gate * retrieval + (1 - gate) * local

        out = out.transpose(1, 2).reshape(b, t, d)
        return self.o(out), M_new, Z_new


if __name__ == "__main__":
    m = InfiniBlock(d_model=32, n_head=4)
    x = torch.randn(1, 8, 32)
    y, M, Z = m(x)
    print(f"Infini out {y.shape}  M {M.shape}  Z {Z.shape}")
