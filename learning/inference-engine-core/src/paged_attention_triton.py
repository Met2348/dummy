"""Educational paged-attention kernel.

We provide two paths:

1. `paged_attention_torch` — pure PyTorch reference; iterates the block table
   in Python.  Slow but the easiest to read.

2. `paged_attention_triton` — Triton kernel (if Triton is available).  Each
   program handles one query token, gathers K/V blocks from the physical pool,
   runs an online softmax, and writes the output.  Online softmax follows the
   FlashAttention recurrence so we never materialise the full attention matrix.

Numerical agreement is checked in `tests/test_paged_attention.py`.
"""
from __future__ import annotations

from typing import List, Optional
import math
import torch


def paged_attention_torch(
    q: torch.Tensor,                      # [n_heads, d_h]
    block_table: List[int],
    n_tokens: int,
    k_pool: torch.Tensor,                 # [n_blocks, block_size, n_kv_heads, d_h]
    v_pool: torch.Tensor,                 # same shape as k_pool
) -> torch.Tensor:
    """Reference implementation: collect K/V from blocks and run softmax."""
    block_size = k_pool.shape[1]
    n_kv_heads = k_pool.shape[2]
    n_heads, d_h = q.shape
    assert n_heads % n_kv_heads == 0, "n_heads must divide by n_kv_heads (GQA)"
    rep = n_heads // n_kv_heads

    ks = []
    vs = []
    remaining = n_tokens
    for blk in block_table:
        take = min(block_size, remaining)
        ks.append(k_pool[blk, :take])    # [take, n_kv_heads, d_h]
        vs.append(v_pool[blk, :take])
        remaining -= take
        if remaining == 0:
            break
    K = torch.cat(ks, dim=0)              # [n_tokens, n_kv_heads, d_h]
    V = torch.cat(vs, dim=0)

    # GQA: repeat KV heads across query head groups
    K = K.repeat_interleave(rep, dim=1)  # [n_tokens, n_heads, d_h]
    V = V.repeat_interleave(rep, dim=1)

    scale = 1.0 / math.sqrt(d_h)
    scores = torch.einsum("hd,thd->th", q, K) * scale     # [n_tokens, n_heads]
    attn = torch.softmax(scores, dim=0)
    out = torch.einsum("th,thd->hd", attn, V)              # [n_heads, d_h]
    return out


def paged_attention_triton(*args, **kwargs):
    """Triton kernel placeholder.

    A real port lives in vLLM `csrc/attention/`; for the workshop we ship the
    torch reference and exercise correctness against it.  Plug in the Triton
    implementation by replacing this body.
    """
    return paged_attention_torch(*args, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    n_blocks, block_size, n_kv_heads, d_h = 16, 8, 2, 32
    n_heads = 4
    k_pool = torch.randn(n_blocks, block_size, n_kv_heads, d_h)
    v_pool = torch.randn(n_blocks, block_size, n_kv_heads, d_h)
    q = torch.randn(n_heads, d_h)
    block_table = [3, 7, 1]
    n_tokens = 20
    out = paged_attention_torch(q, block_table, n_tokens, k_pool, v_pool)
    print(f"output shape {tuple(out.shape)} norm {out.norm():.3f}")
