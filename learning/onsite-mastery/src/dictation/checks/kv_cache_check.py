"""check(target) for kv_cache_step —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. 逐 token 增量拼接的最终结果，与"一次性拥有全量序列"数值一致（误差 < 1e-5）。
2. 首步 cache=None 正确处理，形状随步数正确增长。
3. 不能原地修改旧 cache：历史步骤里保存下来的 cache 快照，在后续 step 调用之后
   形状/数值都不应该被污染（防止用预分配 buffer 原地赋值的偷懒实现）。
4. 用缓存增量结果做因果注意力，逐步输出应等于对全量 K/V 一次性做因果注意力
   （更强的端到端等价性验证，呼应"KV cache 只是省算力，不改变结果"这条核心不变量）。
"""
from __future__ import annotations

import math

import torch


def _causal_attend(q_t: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """单步 query 对全部（含自己）历史 key 做注意力，无需掩码（未来还不存在）。"""
    d = q_t.size(-1)
    scores = q_t @ k.transpose(-2, -1) / math.sqrt(d)
    attn = scores.softmax(dim=-1)
    return attn @ v


def check(target) -> None:
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 5, 8
    k_full = torch.randn(b, h, t, d)
    v_full = torch.randn(b, h, t, d)
    q_full = torch.randn(b, h, t, d)

    cache_k, cache_v = None, None
    snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    stepwise_outs = []
    for step in range(t):
        new_k = k_full[:, :, step:step + 1, :]
        new_v = v_full[:, :, step:step + 1, :]
        cache_k, cache_v = target(new_k, new_v, cache_k, cache_v)

        assert cache_k.shape == (b, h, step + 1, d), f"step {step} cache_k 形状错误: {cache_k.shape}"
        assert cache_v.shape == (b, h, step + 1, d), f"step {step} cache_v 形状错误: {cache_v.shape}"

        # 保存快照的克隆（预期值），以及函数返回的实际引用，之后对比两者是否被后续调用污染
        snapshots.append((step, cache_k.clone(), cache_k))
        q_t = q_full[:, :, step:step + 1, :]
        stepwise_outs.append(_causal_attend(q_t, cache_k, cache_v))

    # 1) 最终缓存与全量序列数值一致
    err_k = (cache_k - k_full).abs().max().item()
    err_v = (cache_v - v_full).abs().max().item()
    assert err_k < 1e-5, f"增量缓存的 K 与全量不一致 err={err_k}"
    assert err_v < 1e-5, f"增量缓存的 V 与全量不一致 err={err_v}"

    # 2) 不能原地修改旧 cache：每个历史快照的"当时克隆值"必须仍等于"当时返回的引用现在的值"
    for step, expected, actual_ref in snapshots:
        assert actual_ref.shape[2] == step + 1, (
            f"step {step} 的旧 cache 引用形状被后续调用改变了（疑似原地修改/共享内存），"
            f"当前形状={actual_ref.shape}"
        )
        assert torch.allclose(expected, actual_ref, atol=1e-6), (
            f"step {step} 的旧 cache 数值被后续调用污染了（疑似原地修改/共享内存）"
        )

    # 3) 端到端等价：逐步 KV cache 注意力输出 == 一次性对全量 K/V 做因果注意力
    stepwise = torch.cat(stepwise_outs, dim=2)
    scores_full = q_full @ k_full.transpose(-2, -1) / math.sqrt(d)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
    scores_full = scores_full.masked_fill(~mask, float("-inf"))
    full_out = scores_full.softmax(dim=-1) @ v_full
    err_out = (stepwise - full_out).abs().max().item()
    assert err_out < 1e-5, f"逐步缓存注意力与全量因果注意力不一致 err={err_out}"
