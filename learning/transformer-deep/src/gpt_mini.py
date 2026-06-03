"""GPT-mini — 80M model with RoPE + RMSNorm + GQA + SwiGLU.

Capstone of transformer-deep topic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import causal_mask, init_weights_   # noqa
from rmsnorm import RMSNorm                       # noqa
from rope import RoPE                              # noqa
from swiglu import SwiGLUMLP                       # noqa


@dataclass
class GPTMiniConfig:
    vocab_size: int = 8192
    n_layer: int = 12
    n_head: int = 12
    n_kv: int = 2
    d_model: int = 768
    d_ff: int = 2048
    max_seq: int = 1024
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-6

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def group_size(self) -> int:
        return self.n_head // self.n_kv


class GQAWithRoPE(nn.Module):
    """GQA with RoPE — used inside GPT-mini Block."""
    def __init__(self, cfg: GPTMiniConfig):
        super().__init__()
        self.cfg = cfg
        d_kv = cfg.n_kv * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, d_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, d_kv, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RoPE(dim=cfg.d_head, base=cfg.rope_base)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                cache_kv: tuple | None = None):
        b, t, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(b, t, cfg.n_head, cfg.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.n_kv, cfg.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.n_kv, cfg.d_head).transpose(1, 2)
        # 应用 RoPE
        pos_offset = cache_kv[0].shape[-2] if cache_kv else 0
        q, k = self.rope(q, k, pos_offset=pos_offset)
        # 合并 cache
        if cache_kv:
            cached_k, cached_v = cache_kv
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)
        new_cache = (k, v)
        # GQA: 把 k, v 复制到 n_head 个 head
        k_full = k.repeat_interleave(cfg.group_size, dim=1)
        v_full = v.repeat_interleave(cfg.group_size, dim=1)
        scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(cfg.d_head)
        if mask is not None:
            # mask shape (t_q, t_kv); 取与 q, k 后部分对齐
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v_full).transpose(1, 2).reshape(b, t, cfg.d_model)
        return self.o_proj(out), new_cache


class Block(nn.Module):
    def __init__(self, cfg: GPTMiniConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GQAWithRoPE(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg.d_model, cfg.d_ff, bias=False)

    def forward(self, x, mask=None, cache_kv=None):
        h, new_cache = self.attn(self.norm1(x), mask=mask, cache_kv=cache_kv)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPTMini(nn.Module):
    def __init__(self, cfg: GPTMiniConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self.apply(init_weights_)

    def forward(self, x: torch.Tensor, cache: Optional[list] = None
                ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """forward.

        x:     (b, t) token ids
        cache: optional list of (k, v) per layer for KV cache
        """
        h = self.tok_embed(x)
        b, t = x.shape
        # causal mask 大小取决于 (t_q, t_kv)
        new_caches = [] if cache is None or len(cache) == 0 else None
        if cache is None:
            cache = [None] * self.cfg.n_layer
        cached_len = cache[0][0].shape[-2] if cache[0] is not None else 0
        full_kv_len = cached_len + t
        # mask: query 长度 t, key 长度 full_kv_len
        mask = torch.full((t, full_kv_len), float("-inf"),
                          device=x.device, dtype=h.dtype)
        for i in range(t):
            mask[i, : cached_len + i + 1] = 0.0
        new_caches = []
        for blk, c in zip(self.blocks, cache):
            h, new_c = blk(h, mask=mask, cache_kv=c)
            new_caches.append(new_c)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        if cache[0] is None:
            return logits
        return logits, new_caches

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new: int = 50,
                 temperature: float = 0.0) -> torch.Tensor:
        """KV-cache generate. x: (b, t)."""
        cache: list = [None] * self.cfg.n_layer
        out_ids = x.clone()
        # prefill
        logits, cache = self.forward(x, cache=cache)
        for _ in range(max_new):
            last = logits[:, -1]                  # (b, vocab)
            if temperature == 0.0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                probs = (last / temperature).softmax(-1)
                nxt = torch.multinomial(probs, 1)
            out_ids = torch.cat([out_ids, nxt], dim=-1)
            logits, cache = self.forward(nxt, cache=cache)
        return out_ids


if __name__ == "__main__":
    cfg = GPTMiniConfig(vocab_size=512, n_layer=2, n_head=4, n_kv=2,
                        d_model=64, d_ff=128, max_seq=64)
    m = GPTMini(cfg)
    x = torch.randint(0, 512, (2, 8))
    y = m(x)
    print(f"forward: {y.shape}")
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    out = m.generate(x, max_new=5)
    print(f"generated: {out.shape}")
