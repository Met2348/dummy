"""check(target) for sample_top_k_top_p —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. top-k 过滤后恰好保留 k 个非零概率，且正是 logits 最大的 k 个。
2. top-p 过滤后保留的集合恰好是"累计概率首次 >= 阈值"的最小前缀（独立重新计算这个
   nucleus 集合，不依赖被测函数以外的任何实现）。
3. 极端情况：top_k=1 退化为 greedy（one-hot），且不受 top_p 影响。
4. temperature 越高分布越平坦（熵增），不做过滤时可验证。
5. 输出恒为合法概率分布（非负、按最后一维求和为 1）。
"""
from __future__ import annotations

import torch


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1)


def check(target) -> None:
    torch.manual_seed(0)
    vocab = 10
    logits = torch.randn(vocab)

    # 1) top-k=3 精确保留 3 个，且是最大的 3 个
    out_k = target(logits, top_k=3, top_p=1.0, temperature=1.0)
    assert out_k.shape == logits.shape
    nonzero_idx = set(out_k.nonzero().flatten().tolist())
    assert len(nonzero_idx) == 3, f"top_k=3 应恰好保留 3 个非零概率，实际 {len(nonzero_idx)} 个"
    expected_idx = set(logits.topk(3).indices.tolist())
    assert nonzero_idx == expected_idx, f"保留的下标应为最大的3个 {expected_idx}，实际 {nonzero_idx}"
    assert torch.allclose(out_k.sum(), torch.tensor(1.0), atol=1e-5)

    # 2) top-p：独立重新计算 nucleus 集合，和被测函数的支持集对拍
    p_thresh = 0.6
    out_p = target(logits, top_k=0, top_p=p_thresh, temperature=1.0)
    assert torch.allclose(out_p.sum(), torch.tensor(1.0), atol=1e-5)

    sorted_logits, sorted_idx = logits.sort(descending=True)
    probs_sorted = sorted_logits.softmax(dim=-1)
    cum = probs_sorted.cumsum(dim=-1)
    remove = cum > p_thresh
    remove[1:] = remove[:-1].clone()   # 累计概率首次 >= 阈值的那个仍要保留，故右移一位
    remove[0] = False
    expected_nucleus = set(sorted_idx[~remove].tolist())

    actual_nonzero = set(out_p.nonzero().flatten().tolist())
    assert actual_nonzero == expected_nucleus, (
        f"top_p={p_thresh} 的 nucleus 集合不对: 期望 {expected_nucleus}, 实际 {actual_nonzero}"
    )

    # 3) top_k=1 退化为 greedy，且不受 top_p 影响（即使 top_p 很小）
    argmax_idx = logits.argmax().item()
    for tiny_p in (0.01, 0.5, 1.0):
        out_greedy = target(logits, top_k=1, top_p=tiny_p, temperature=1.0)
        nz = out_greedy.nonzero().flatten().tolist()
        assert nz == [argmax_idx], f"top_k=1 应退化为 one-hot(argmax), top_p={tiny_p} 时得到 {nz}"
        assert abs(out_greedy[argmax_idx].item() - 1.0) < 1e-5

    # 4) temperature 越高越平坦（不过滤时）
    flat_logits = torch.tensor([3.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    e_cold = _entropy(target(flat_logits, top_k=0, top_p=1.0, temperature=0.5))
    e_normal = _entropy(target(flat_logits, top_k=0, top_p=1.0, temperature=1.0))
    e_hot = _entropy(target(flat_logits, top_k=0, top_p=1.0, temperature=2.0))
    assert e_cold < e_normal < e_hot, f"熵应随 temperature 单调递增: {e_cold.item()}, {e_normal.item()}, {e_hot.item()}"

    # 5) 合法概率分布：非负 + 求和为 1（批量维也测一下）
    batch_logits = torch.randn(4, vocab)
    out_batch = target(batch_logits, top_k=4, top_p=0.9, temperature=0.8)
    assert (out_batch >= 0).all()
    assert torch.allclose(out_batch.sum(dim=-1), torch.ones(4), atol=1e-5)
