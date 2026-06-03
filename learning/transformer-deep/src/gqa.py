"""GQA — Grouped-Query Attention (Ainslie 2023)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    """h Q head, g KV head, 每 h/g Q 共享 1 K, 1 V."""

    def __init__(self, d_model: int, n_head: int, n_kv_head: int,
                 bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_head == 0, f"{n_head} % {n_kv_head} != 0"
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_head = d_model // n_head
        self.group_size = n_head // n_kv_head
        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.d_head, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.d_head).transpose(1, 2)
        # repeat K, V to match h heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(out)


if __name__ == "__main__":
    m = GQA(d_model=64, n_head=8, n_kv_head=2)
    x = torch.randn(2, 8, 64)
    print("GQA out:", m(x).shape)
    print("GQA params:", sum(p.numel() for p in m.parameters()))
