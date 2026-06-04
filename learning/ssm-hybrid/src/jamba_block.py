"""Jamba-style hybrid block — attn / mamba 按层配比."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_block import MambaBlock


class SimpleAttn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h = h
        self.d_h = d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h)
        mask = torch.full((t, t), float("-inf"), device=x.device,
                          dtype=x.dtype).triu_(1)
        attn = F.softmax(scores + mask, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.o(out)


class JambaLayer(nn.Module):
    def __init__(self, d_model: int, layer_type: str = "mamba",
                 d_state: int = 8):
        super().__init__()
        assert layer_type in {"mamba", "attn"}
        self.norm1 = nn.LayerNorm(d_model)
        if layer_type == "attn":
            self.mixer = SimpleAttn(d_model, h=4)
        else:
            self.mixer = MambaBlock(d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class JambaModel(nn.Module):
    """每 8 层 1 个 attention，其余 Mamba (Jamba 配比)."""
    def __init__(self, d_model: int = 64, n_layer: int = 8,
                 d_state: int = 8):
        super().__init__()
        layers = []
        for i in range(n_layer):
            t = "attn" if i % 8 == 4 else "mamba"
            layers.append(JambaLayer(d_model, layer_type=t, d_state=d_state))
        self.layers = nn.ModuleList(layers)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


if __name__ == "__main__":
    m = JambaModel(d_model=32, n_layer=4, d_state=8)
    x = torch.randn(1, 16, 32)
    y = m(x)
    print(f"Jamba out {y.shape}, params {sum(p.numel() for p in m.parameters())}")
