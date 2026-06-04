"""Tree attention mask construction for batched verification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math
import torch


def build_tree_mask(parent_idx: List[int]) -> torch.Tensor:
    """Return [N, N] bool mask: mask[i, j] = j is an ancestor of i (incl. i)."""
    n = len(parent_idx)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        cur = i
        while cur != -1:
            mask[i, cur] = True
            cur = parent_idx[cur]
    return mask


def tree_attention_torch(
    q: torch.Tensor,             # [N, d]
    k: torch.Tensor,             # [N, d]
    v: torch.Tensor,             # [N, d]
    parent_idx: List[int],
) -> torch.Tensor:
    """Compute self-attn with a tree mask."""
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    scores = q @ k.transpose(0, 1) * scale     # [N, N]
    mask = build_tree_mask(parent_idx)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
