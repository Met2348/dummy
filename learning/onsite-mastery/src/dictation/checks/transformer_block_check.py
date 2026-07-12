"""check(target) for transformer_block_forward —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. 残差直通：attn_fn/ffn_fn 都返回全 0 时，输出必须精确等于输入（不管 norm1/norm2 是什么）。
2. pre-norm 顺序：norm 必须加在子层输入上、子层输出与"未归一化"的 x 相加——用
   非平凡（乘法而非加法）的 norm 和恒等子层分别探测两条残差分支，post-norm/漏残差/
   漏归一化 的错误实现都会在数值上和正确答案不一致。
3. 形状不变。
"""
from __future__ import annotations

import torch


def _zero(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def _identity(t: torch.Tensor) -> torch.Tensor:
    return t


def check(target) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)

    # 1) 残差直通：子层全 0 时输出必须精确等于输入
    out = target(x, _zero, _zero, lambda t: t * 999.0, lambda t: t - 999.0)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6), "attn_fn/ffn_fn 全 0 时，输出应精确等于输入"

    # 2a) 探测第一条残差分支的 pre-norm 顺序：norm1 放大 2 倍、attn_fn 恒等、ffn_fn 恒为 0
    #     正确结果 = x + attn_fn(norm1(x)) = x + 2x = 3x
    out_a = target(x, _identity, _zero, lambda t: t * 2.0, _identity)
    assert torch.allclose(out_a, 3.0 * x, atol=1e-5), (
        f"第一条残差分支的 pre-norm 顺序不对：期望约等于 3x，"
        f"实际与 3x 的最大误差 {(out_a - 3.0 * x).abs().max().item()}"
    )

    # 2b) 探测第二条残差分支：attn_fn 恒为 0（第一条分支不改变 x），
    #     norm2 放大 3 倍、ffn_fn 恒等 -> 正确结果 = x + ffn_fn(norm2(x)) = x + 3x = 4x
    out_b = target(x, _zero, _identity, _identity, lambda t: t * 3.0)
    assert torch.allclose(out_b, 4.0 * x, atol=1e-5), (
        f"第二条残差分支的 pre-norm 顺序不对：期望约等于 4x，"
        f"实际与 4x 的最大误差 {(out_b - 4.0 * x).abs().max().item()}"
    )

    # 3) 反传可用
    x2 = x.clone().requires_grad_(True)
    y = target(x2, _identity, _identity, _identity, _identity)
    y.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
