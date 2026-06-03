"""MQA — Multi-Query Attention (Shazeer 2019)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MQA(nn.Module):
    """h Q head 共享 1 K, 1 V head."""

    def __init__(self, d_model: int, n_head: int, bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.d_head, bias=bias)    # 单 head
        self.v_proj = nn.Linear(d_model, self.d_head, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, 1, self.d_head).transpose(1, 2)   # (b, 1, t, d_head)
        v = self.v_proj(x).view(b, t, 1, self.d_head).transpose(1, 2)
        # broadcast K, V to all h heads
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(out)


if __name__ == "__main__":
    m = MQA(d_model=64, n_head=4)
    x = torch.randn(2, 8, 64)
    print("MQA out:", m(x).shape)
    print("MQA params:", sum(p.numel() for p in m.parameters()))
