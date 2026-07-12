"""causal_attention 参考实现——见 dictation/specs/attention_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/attention.py，但按闭卷 spec 手写数值稳定 softmax。
"""
from __future__ import annotations

import math

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, h, t, d = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)          # (B,H,T,T)

    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, float("-inf"))

    # 手写数值稳定 softmax：减去行最大值再 exp、按行归一化
    row_max = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - row_max)
    attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    return attn @ v


if __name__ == "__main__":
    from dictation.checks.attention_check import check
    check(causal_attention)
    print("[PASS] attention_solution 自检通过")
