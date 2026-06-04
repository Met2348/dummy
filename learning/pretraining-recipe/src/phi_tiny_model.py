"""Phi-tiny 270M model - Pre-RMSNorm + GQA + RoPE + SwiGLU + tied."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PhiTinyConfig:
    vocab_size: int = 50257
    hidden: int = 1024
    n_head: int = 16
    n_kv_head: int = 4
    head_dim: int = 64
    n_layer: int = 24
    intermediate: int = 2730
    seq_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def precompute_rope(head_dim: int, seq_len: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() /
                                head_dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        self.n_head = c.n_head
        self.n_kv_head = c.n_kv_head
        self.head_dim = c.head_dim
        self.q_proj = nn.Linear(c.hidden, c.n_head * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.n_kv_head * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.n_kv_head * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.n_head * c.head_dim, c.hidden, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim
                                 ).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim
                                 ).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim
                                 ).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGluMlp(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        d_ff = c.intermediate
        self.w1 = nn.Linear(c.hidden, d_ff, bias=False)
        self.w2 = nn.Linear(c.hidden, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, c.hidden, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        self.attn_ln = RMSNorm(c.hidden, c.norm_eps)
        self.attn = GroupedQueryAttention(c)
        self.mlp_ln = RMSNorm(c.hidden, c.norm_eps)
        self.mlp = SwiGluMlp(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_ln(x), cos, sin)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class PhiTiny(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.final_ln = RMSNorm(c.hidden, c.norm_eps)
        self.lm_head = nn.Linear(c.hidden, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = precompute_rope(c.head_dim, c.seq_len, c.rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                std = 0.02
                if "o_proj" in name or "w3" in name:
                    std /= math.sqrt(2 * self.cfg.n_layer)
                nn.init.normal_(p, std=std)

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, self.rope_cos, self.rope_sin)
        h = self.final_ln(h)
        return self.lm_head(h)


if __name__ == "__main__":
    c = PhiTinyConfig()
    m = PhiTiny(c)
    n = sum(p.numel() for p in m.parameters())
    n_no_embed = n - m.embed.weight.numel()
    print(f"PhiTiny params: total {n/1e6:.1f}M")
    print(f"  excl embed tied: {n_no_embed/1e6:.1f}M")
    x = torch.randint(0, c.vocab_size, (2, 64))
    out = m(x)
    print(f"  fwd out shape {out.shape}")
