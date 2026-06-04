"""Vanilla GPT-2 124M baseline (ckpt A)."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    seq_len: int = 1024
    hidden: int = 768
    n_head: int = 12
    n_layer: int = 12


class GPT2Block(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            c.hidden, c.n_head, batch_first=True, bias=True)
        self.ln1 = nn.LayerNorm(c.hidden)
        self.mlp = nn.Sequential(
            nn.Linear(c.hidden, 4 * c.hidden),
            nn.GELU(),
            nn.Linear(4 * c.hidden, c.hidden),
        )
        self.ln2 = nn.LayerNorm(c.hidden)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device,
                                       dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


class VanillaGPT2(nn.Module):
    def __init__(self, c: GPT2Config):
        super().__init__()
        self.cfg = c
        self.tok_embed = nn.Embedding(c.vocab_size, c.hidden)
        self.pos_embed = nn.Embedding(c.seq_len, c.hidden)
        self.blocks = nn.ModuleList([GPT2Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.hidden)
        self.lm_head = nn.Linear(c.hidden, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight
        self._init()

    def _init(self):
        for n, p in self.named_parameters():
            if "weight" in n and p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer) \
                    if "out_proj" in n or "mlp.2" in n else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for b in self.blocks:
            h = b(h)
        return self.lm_head(self.ln_f(h))


if __name__ == "__main__":
    c = GPT2Config()
    m = VanillaGPT2(c)
    n = sum(p.numel() for p in m.parameters())
    n_no_embed = n - m.tok_embed.weight.numel()
    print(f"Vanilla GPT-2 124M:")
    print(f"  total params {n/1e6:.1f}M (tied embed)")
    print(f"  excl embed   {n_no_embed/1e6:.1f}M")
    x = torch.randint(0, c.vocab_size, (2, 64))
    print(f"  fwd shape: {m(x).shape}")
