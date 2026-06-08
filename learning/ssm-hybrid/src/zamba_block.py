"""Zamba - shared attention across multiple layers."""
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_block import MambaBlock
from jamba_block import SimpleAttn


class ZambaModel(nn.Module):
    def __init__(self, d_model: int = 64, n_layer: int = 8,
                 attn_period: int = 4, d_state: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layer)
        )
        # 单个 attention shared across层
        self.shared_attn = SimpleAttn(d_model, h=4)
        self.attn_period = attn_period
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = x + l(x)
            if (i + 1) % self.attn_period == 0:
                x = x + self.shared_attn(x)
        return self.norm_f(x)


if __name__ == "__main__":
    m = ZambaModel(d_model=32, n_layer=4, attn_period=2)
    x = torch.randn(1, 10, 32)
    y = m(x)
    print(f"Zamba out {y.shape}, params {sum(p.numel() for p in m.parameters())}")
