"""RWKV-style block — 简化教学版 (无 RWKV-7 完整 channel mix)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVBlock(nn.Module):
    """简化 RWKV: weighted sum with learned time decay."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.R = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        r = torch.sigmoid(self.R(x))
        k = self.K(x)
        v = self.V(x)
        # 简化版 time-mix: 加权累加
        out = torch.zeros_like(v)
        state_num = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        state_den = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        w = -torch.exp(self.time_decay)
        for i in range(t):
            k_i = k[:, i]
            v_i = v[:, i]
            # numerator / denominator weighted sum
            num = state_num + torch.exp(k_i + self.time_first) * v_i
            den = state_den + torch.exp(k_i + self.time_first)
            wkv = num / (den + 1e-9)
            out[:, i] = r[:, i] * wkv
            # update state with decay
            state_num = state_num * torch.exp(w) + torch.exp(k_i) * v_i
            state_den = state_den * torch.exp(w) + torch.exp(k_i)
        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = RWKVBlock(d_model=32)
    x = torch.randn(1, 10, 32)
    print(f"RWKV out {m(x).shape}, params {sum(p.numel() for p in m.parameters())}")
