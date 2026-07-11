"""Scaled-dot-product attention + Multi-Head Attention，从零手写。

面试高频度 ★★★★★。得分点：/sqrt(d) 的理由、causal mask 用 -inf 加在 softmax 前、
多头靠 reshape 而非开 H 个矩阵。self_test 与 F.scaled_dot_product_attention 对拍。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,          # (B, H, Tq, d)
    k: torch.Tensor,          # (B, H, Tk, d)
    v: torch.Tensor,          # (B, H, Tk, dv)
    mask: torch.Tensor | None = None,   # (Tq, Tk) or broadcastable, True=keep
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """返回 (B, H, Tq, dv)。手写核心：为何除 sqrt(d)——见模块 docstring。"""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)      # (B,H,Tq,Tk)
    if mask is not None:
        # mask 为 True 处保留，False 处置 -inf（在 softmax 之前，乘法掩码会泄露未来）
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v


def causal_mask(t: int, device=None) -> torch.Tensor:
    """(T, T) 下三角 True 掩码：位置 i 只能看 <= i。"""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    """从零多头自注意力（非 nn.MultiheadAttention）。"""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)   # 融合 QKV，一次投影
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)                                       # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)                          # 3 x (B,T,D)
        # (B,T,D) -> (B,H,T,d_head)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        mask = causal_mask(t, x.device) if causal else None
        o = scaled_dot_product_attention(q, k, v, mask=mask)    # (B,H,T,d_head)
        o = o.transpose(1, 2).contiguous().view(b, t, self.d_model)   # 合并头
        return self.out(o)


def _self_test() -> None:
    torch.manual_seed(0)
    b, h, t, d = 2, 4, 5, 8
    q, k, v = (torch.randn(b, h, t, d) for _ in range(3))

    # 1) 与官方 kernel 对拍（无掩码）
    mine = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max().item()

    # 2) 与官方对拍（因果掩码）
    m = causal_mask(t)
    mine_c = scaled_dot_product_attention(q, k, v, mask=m)
    ref_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine_c, ref_c, atol=1e-5)

    # 3) 因果性：位置 0 的输出只依赖 v[...,0,:]（改 v[...,1:,:] 不应影响 out[...,0,:]）
    v2 = v.clone()
    v2[..., 1:, :] += 100.0
    o1 = scaled_dot_product_attention(q, k, v, mask=m)[..., 0, :]
    o2 = scaled_dot_product_attention(q, k, v2, mask=m)[..., 0, :]
    assert torch.allclose(o1, o2, atol=1e-5), "因果掩码泄露了未来！"

    # 4) MHA 前向 + 反传通
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 6, 16, requires_grad=True)
    y = mha(x, causal=True)
    assert y.shape == (2, 6, 16)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[PASS] attention: SDPA 对拍官方 + 因果性 + MHA 反传")


if __name__ == "__main__":
    _self_test()
