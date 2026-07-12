"""sample_top_k_top_p 参考实现——见 dictation/specs/sampling_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/sampling.py 里的 top_k_filter/top_p_filter，
这里把 temperature -> top_k -> top_p -> softmax 归一化整合进一个函数。
"""
from __future__ import annotations

import torch


def sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    logits = logits / temperature
    vocab = logits.size(-1)

    if top_k is not None and 0 < top_k < vocab:
        kth = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs_sorted = sorted_logits.softmax(dim=-1)
        cum = probs_sorted.cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False           # nucleus 至少保留 1 个
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    return logits.softmax(dim=-1)


if __name__ == "__main__":
    from dictation.checks.sampling_check import check
    check(sample_top_k_top_p)
    print("[PASS] sampling_solution 自检通过")
