"""check(target) for causal_attention —— 纯断言，独立于 solutions/，不导入参考实现。

验证点：
1. 形状正确。
2. 因果不泄露：扰动未来 v 不应改变更早位置的输出（等价于 softmax 权重在未来位置严格为 0）。
3. 与 F.scaled_dot_product_attention(..., is_causal=True) 数值对拍，误差 < 1e-4。
4. 反传可用（梯度有限）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def check(target) -> None:
    torch.manual_seed(0)
    b, h, t, d = 2, 3, 6, 8
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)

    out = target(q, k, v)
    assert out.shape == (b, h, t, d), f"输出形状错误: {out.shape}"

    # 1) 因果不泄露：只改未来位置(>=3)的 v，位置 0..2 的输出必须不变
    v_perturbed = v.clone()
    v_perturbed[:, :, 3:, :] += 1000.0
    out_perturbed = target(q, k, v_perturbed)
    assert torch.allclose(out[:, :, :3, :], out_perturbed[:, :, :3, :], atol=1e-4), (
        "扰动未来 v 影响了更早位置的输出——因果掩码泄露了未来信息"
    )

    # 2) 与官方 kernel 数值对拍（causal）
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"与 F.scaled_dot_product_attention(is_causal=True) 对不齐, max_err={err}"

    # 3) 位置 0 只能看到自己：out[...,0,:] 必须等于对 v[...,0,:] 的直接拷贝（单 key 无选择）
    #    这是一个更强的结构性验证：query 只有一个可见 key 时，输出必然等于该 key 的 value。
    out0 = target(q, k, v)[:, :, 0, :]
    assert torch.allclose(out0, v[:, :, 0, :], atol=1e-4), (
        "位置 0 只能注意到自己，输出应恰好等于 v[...,0,:]"
    )

    # 4) 反传可用
    q2 = torch.randn(b, h, t, d, requires_grad=True)
    k2 = torch.randn(b, h, t, d, requires_grad=True)
    v2 = torch.randn(b, h, t, d, requires_grad=True)
    y = target(q2, k2, v2)
    y.sum().backward()
    for name, g in (("q", q2.grad), ("k", k2.grad), ("v", v2.grad)):
        assert g is not None and torch.isfinite(g).all(), f"{name} 的梯度缺失或非有限"
