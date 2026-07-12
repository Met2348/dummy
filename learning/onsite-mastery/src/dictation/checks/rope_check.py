"""check(target) for rope_apply —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. 保范数：旋转不改变向量在最后一维上的 L2 范数。
2. 相对位置不变性：<RoPE(u,m), RoPE(v,n)> 只依赖 (m-n)——用不同绝对位置但相同相对
   距离的两组 (m,n) 验证内积一致。positions 故意用不连续/不从 0 开始的下标，
   防止"内部用 arange(T) 覆盖 positions"的偷懒实现蒙混过关。
3. 位置 0 = 恒等（angle=0 时 cos=1,sin=0）。
"""
from __future__ import annotations

import torch


def check(target) -> None:
    torch.manual_seed(0)
    dim = 8

    # 1) 保范数
    t = 6
    positions = torch.arange(t)
    x = torch.randn(2, t, dim)
    xr = target(x, positions)
    assert xr.shape == x.shape, f"输出形状错误: {xr.shape}"
    assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-4), "RoPE 应保范数（正交变换）"

    # 2) 相对位置不变性，用不连续/不从 0 开始的 positions 防止内部覆盖 positions 的偷懒实现
    u = torch.randn(dim)
    v = torch.randn(dim)

    def rope_dot(m: int, n: int) -> torch.Tensor:
        um = target(u.unsqueeze(0), torch.tensor([m])).squeeze(0)
        vn = target(v.unsqueeze(0), torch.tensor([n])).squeeze(0)
        return (um * vn).sum()

    # 两组绝对位置不同、相对距离都是 3；且都不从 0 开始、不连续
    dot_a = rope_dot(17, 14)
    dot_b = rope_dot(103, 100)
    assert torch.allclose(dot_a, dot_b, atol=1e-4), (
        f"相对位置性质被破坏: rope_dot(17,14)={dot_a.item()} != rope_dot(103,100)={dot_b.item()}"
    )

    # 绝对位置相同、相对距离不同 -> 内积一般不应相等（防止函数忽略 positions 恒返回同一个值）
    dot_c = rope_dot(17, 5)
    assert not torch.allclose(dot_a, dot_c, atol=1e-4), "内积没有随相对位置变化，疑似忽略了 positions"

    # 3) 位置 0 = 恒等
    x0 = torch.randn(3, 1, dim)
    out0 = target(x0, torch.tensor([0]))
    assert torch.allclose(out0, x0, atol=1e-5), "position=0 应恰好是恒等变换"

    # 4) 反传可用
    x2 = torch.randn(2, t, dim, requires_grad=True)
    y = target(x2, positions)
    y.sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
