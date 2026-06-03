"""Sliding Window Attention — Mistral / Gemma 风格."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swa_mask(t: int, window: int, device=None,
             dtype=torch.float32) -> torch.Tensor:
    """SWA + causal mask, shape (t, t) with -inf where masked."""
    pos = torch.arange(t, device=device, dtype=torch.long)
    diff = pos[None, :] - pos[:, None]
    # mask out: diff > 0 (future)  OR  diff < -window (太远)
    bad = (diff > 0) | (diff < -window)
    out = torch.zeros(t, t, device=device, dtype=dtype)
    out.masked_fill_(bad, float("-inf"))
    return out


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, window: int,
                 bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.window = window
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        mask = swa_mask(t, self.window, device=x.device, dtype=x.dtype)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(out)


if __name__ == "__main__":
    m = SlidingWindowAttention(d_model=64, n_head=4, window=2)
    x = torch.randn(1, 8, 64)
    y = m(x)
    print("SWA out:", y.shape)
    # 可视化 mask
    mask = swa_mask(8, 2)
    print("\nMask (8 token, W=2):")
    print((mask == 0).int())
