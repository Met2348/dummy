"""Naive PyTorch attention reference for backend comparison.

flash-attn / flashinfer if installed give 10-50x speedup; we ship this as the
baseline so the workshop is fully runnable without those libs.
"""
from __future__ import annotations

import math
import torch


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """q,k,v shape [B, H, S, D].  Returns [B, H, S, D]."""
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k) * scale
    if causal:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", attn, v)
    return out


def has_flash_attn() -> bool:
    try:
        import flash_attn      # noqa: F401
        return True
    except ImportError:
        return False


def has_flashinfer() -> bool:
    try:
        import flashinfer      # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(1, 4, 16, 32)
    k = torch.randn(1, 4, 16, 32)
    v = torch.randn(1, 4, 16, 32)
    out = naive_attention(q, k, v)
    print(out.shape, out.norm().item())
    print(f"flash_attn={has_flash_attn()} flashinfer={has_flashinfer()}")
