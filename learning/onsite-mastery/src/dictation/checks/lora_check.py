"""check(target) for lora_forward —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. lora_B 全 0 时输出精确等于 base(x)（初始等价性）。
2. lora_A 全 0 时同样精确等于 base(x)。
3. 非零 A/B 时，输出与独立手算的公式对拍（含 alpha/rank 缩放）。
4. 梯度只流向 lora_A/lora_B，不流向 base_weight（用 requires_grad 验证）。
"""
from __future__ import annotations

import torch


def check(target) -> None:
    torch.manual_seed(0)
    in_f, out_f, rank, alpha = 10, 6, 4, 8

    x = torch.randn(5, in_f)
    base_weight = torch.randn(out_f, in_f)
    lora_A = torch.randn(in_f, rank)
    lora_B = torch.randn(rank, out_f)

    base_out = x @ base_weight.T

    # 1) lora_B=0 -> 精确等于 base_out
    zero_B = torch.zeros(rank, out_f)
    out1 = target(x, base_weight, lora_A, zero_B, alpha, rank)
    assert out1.shape == (5, out_f)
    assert torch.allclose(out1, base_out, atol=1e-6), "lora_B=0 时应精确等于 base(x)"

    # 2) lora_A=0 -> 同样精确等于 base_out
    zero_A = torch.zeros(in_f, rank)
    out2 = target(x, base_weight, zero_A, lora_B, alpha, rank)
    assert torch.allclose(out2, base_out, atol=1e-6), "lora_A=0 时应精确等于 base(x)"

    # 3) 非零 A/B：与独立手算公式对拍
    out3 = target(x, base_weight, lora_A, lora_B, alpha, rank)
    ref = base_out + (alpha / rank) * (x @ lora_A @ lora_B)
    assert torch.allclose(out3, ref, atol=1e-5), (out3 - ref).abs().max().item()
    assert not torch.allclose(out3, base_out, atol=1e-3), "非零 A/B 应该真的改变输出"

    # 4) 梯度只流向 A/B，不流向 base_weight
    base_w2 = base_weight.clone().requires_grad_(False)
    A2 = lora_A.clone().requires_grad_(True)
    B2 = lora_B.clone().requires_grad_(True)
    out4 = target(x, base_w2, A2, B2, alpha, rank)
    out4.sum().backward()
    assert base_w2.grad is None, "base_weight 不应该有梯度（应保持冻结）"
    assert A2.grad is not None and torch.isfinite(A2.grad).all()
    assert B2.grad is not None and torch.isfinite(B2.grad).all()
