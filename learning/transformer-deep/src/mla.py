"""MLA — Multi-head Latent Attention (DeepSeek-V2 2024).

教学版：去掉 RoPE 分离的复杂部分，保留低秩压缩核心。
KV cache 只存 d_low 维 latent c，推理时再解压。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLA(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_low: int = 64,
                 bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_low = d_low
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        # 关键: K, V 共享 一个 d_low compression
        self.kv_down = nn.Linear(d_model, d_low, bias=bias)
        self.k_up = nn.Linear(d_low, d_model, bias=bias)
        self.v_up = nn.Linear(d_low, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                cache_c: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (output, c_kv) — c_kv 是新生成的 latent 可被 cache."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        # 压缩 K, V
        c = self.kv_down(x)                                       # (b, t, d_low)
        if cache_c is not None:
            c = torch.cat([cache_c, c], dim=1)
        # 推理时再解压
        k_full = self.k_up(c).view(b, c.shape[1], self.n_head, self.d_head
                                  ).transpose(1, 2)
        v_full = self.v_up(c).view(b, c.shape[1], self.n_head, self.d_head
                                  ).transpose(1, 2)
        scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v_full).transpose(1, 2).reshape(b, t, self.d_model)
        return self.o_proj(out), c


def kv_cache_size_mla(n_layer: int, t: int, d_low: int,
                      dtype_bytes: int = 2) -> int:
    """MLA 只 cache 1 个 c_kv (d_low 维)."""
    return n_layer * t * d_low * dtype_bytes


if __name__ == "__main__":
    m = MLA(d_model=64, n_head=4, d_low=16)
    x = torch.randn(2, 8, 64)
    out, c = m(x)
    print("MLA out:", out.shape, "c_kv:", c.shape)
    print("MLA params:", sum(p.numel() for p in m.parameters()))
