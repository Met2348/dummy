"""MHA — Multi-Head Attention (Vaswani 2017)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, d_model: int, n_head: int, bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                          # (b, h, t, d_head)
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(out)


def kv_cache_size(n_layer: int, n_head: int, d_head: int, t: int,
                  dtype_bytes: int = 2) -> int:
    """B = 2 (K+V) · n_layer · n_head · t · d_head · dtype_bytes."""
    return 2 * n_layer * n_head * t * d_head * dtype_bytes


if __name__ == "__main__":
    model = MHA(d_model=64, n_head=4)
    x = torch.randn(2, 8, 64)
    y = model(x)
    print("MHA out:", y.shape)
    print("MHA params:", sum(p.numel() for p in model.parameters()))
