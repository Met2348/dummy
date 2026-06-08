"""Mini-Mamba - capstone 130M-scale idea, toy 32M for demo."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mamba_block import MambaBlock


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()


@dataclass
class MiniMambaConfig:
    vocab_size: int = 1024
    n_layer: int = 4
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    max_seq: int = 1024


class MiniMamba(nn.Module):
    def __init__(self, cfg: MiniMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            MambaBlock(cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv)
            for _ in range(cfg.n_layer)
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight     # tie

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.lm_head(self.norm_f(h))


if __name__ == "__main__":
    cfg = MiniMambaConfig(vocab_size=256, n_layer=2, d_model=64, d_state=8)
    m = MiniMamba(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    y = m(x)
    print(f"MiniMamba out {y.shape}, params {sum(p.numel() for p in m.parameters())}")
