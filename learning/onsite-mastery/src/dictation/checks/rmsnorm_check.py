"""check(target) for rmsnorm —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. weight=全1 时，输出的 RMS（均方根）应归一到 1。
2. 与公式手算对拍（check 自己独立算一遍公式，不依赖 solution 模块）。
3. eps 加在 sqrt 内部：x 全 0 时输出必须是有限值 0，而不是 NaN/Inf。
4. weight 非平凡时，输出 = 归一化结果 * weight（逐通道验证）。
5. 反传可用。
"""
from __future__ import annotations

import torch


def check(target) -> None:
    torch.manual_seed(0)
    dim = 16
    x = torch.randn(4, 5, dim) * 3.0 + 2.0   # 非零均值/非单位方差，逼真一点
    ones = torch.ones(dim)

    # 1) weight=1 时输出 RMS 归一到 1
    out = target(x, ones, eps=1e-6)
    assert out.shape == x.shape
    out_rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(out_rms, torch.ones(4, 5), atol=1e-3), (
        f"输出 RMS 未归一到 1: {out_rms}"
    )

    # 2) 与独立手算的公式对拍
    eps = 1e-6
    ref = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * ones
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()

    # 3) eps 位置正确：x 全 0 时应输出有限的 0，而不是 NaN（若 eps 加错位置会在这里炸）
    x_zero = torch.zeros(2, 3, dim)
    out_zero = target(x_zero, ones, eps=1e-6)
    assert torch.isfinite(out_zero).all(), "x 全 0 时输出出现 NaN/Inf，eps 大概率加错了位置"
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6)

    # 4) 非平凡 weight：逐通道缩放
    w = torch.randn(dim).abs() + 0.5
    out_w = target(x, w, eps=1e-6)
    ref_w = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w
    assert torch.allclose(out_w, ref_w, atol=1e-5)

    # 5) 反传可用，weight 也应该有梯度
    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    y = target(x2, w2, eps=1e-6)
    y.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
    assert w2.grad is not None and torch.isfinite(w2.grad).all()
