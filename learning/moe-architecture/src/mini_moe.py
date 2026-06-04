"""Mini-MoE — 4-expert + Aux-Free 集成 (capstone).

把 GPT-mini 风格 base 的 MLP 替换为 4-routed + 1-shared MoE。
路由用 Aux-Free（DeepSeek-V3 风格）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from aux_loss_free import AuxFreeRouter
from moe_layer_naive import Expert


@dataclass
class MiniMoEConfig:
    vocab_size: int = 1024
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    n_kv: int = 2
    d_ff_routed: int = 128
    d_ff_shared: int = 256
    n_routed: int = 4
    top_k: int = 2
    max_seq: int = 64
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()


class SimpleAttention(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.h = cfg.n_head; self.d_h = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d_h).transpose(1, 2)
        mask = torch.full((t, t), float("-inf"), device=x.device,
                          dtype=x.dtype).triu_(1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_h) + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o(out)


class MoEBlock(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SimpleAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.router = AuxFreeRouter(cfg.d_model, n_experts=cfg.n_routed,
                                     top_k=cfg.top_k)
        self.routed = nn.ModuleList(
            Expert(cfg.d_model, cfg.d_ff_routed) for _ in range(cfg.n_routed)
        )
        self.shared = Expert(cfg.d_model, cfg.d_ff_shared)
        self.cfg = cfg

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        # MoE FFN
        h = self.n2(x)
        b, t, d = h.shape
        h_flat = h.view(-1, d)
        out = self.shared(h_flat)
        gates, idx, _ = self.router(h_flat)
        for e in range(self.cfg.n_routed):
            mask = (idx == e)
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            tok_in = h_flat[tok_idx]
            expert_out = self.routed[e](tok_in)
            weight = gates[tok_idx, slot].unsqueeze(-1)
            out.index_add_(0, tok_idx, expert_out * weight)
        return x + out.view(b, t, d)


class MiniMoE(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, x):
        h = self.tok_embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.norm_f(h))


if __name__ == "__main__":
    cfg = MiniMoEConfig(vocab_size=256, n_layer=2, d_model=64,
                        n_head=4, n_routed=4, top_k=2)
    m = MiniMoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = m(x)
    print(f"out {y.shape}  params {sum(p.numel() for p in m.parameters())}")
