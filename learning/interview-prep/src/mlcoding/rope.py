"""RoPE 旋转位置编码，从零手写（LLaMA/GPT-NeoX 风格）。

面试高频度 ★★★★。得分点：RoPE 把位置编码成"旋转"，因此 <RoPE(q,m),RoPE(k,n)>
只依赖相对位置 (m-n)——这正是它优于绝对位置编码之处。旋转是正交变换，保范数。
"""
from __future__ import annotations

import torch


def build_cos_sin(seq_len: int, dim: int, theta: float = 10000.0):
    """返回 cos,sin，形状均 (seq_len, dim)。dim 必须为偶数。"""
    assert dim % 2 == 0
    i = torch.arange(0, dim, 2, dtype=torch.float32)        # 0,2,...,dim-2
    freqs = 1.0 / (theta ** (i / dim))                      # (dim/2,)
    pos = torch.arange(seq_len, dtype=torch.float32)        # (T,)
    ang = torch.outer(pos, freqs)                           # (T, dim/2)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)         # (T, dim)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, dim)；cos/sin: (T, dim)。广播到 x 的前置维度。"""
    return x * cos + rotate_half(x) * sin


def _self_test() -> None:
    torch.manual_seed(0)
    t, dim = 6, 8
    cos, sin = build_cos_sin(t, dim)
    q = torch.randn(2, t, dim)
    k = torch.randn(2, t, dim)

    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)

    # 1) 保范数（旋转是正交变换）
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4)

    # 2) 相对位置性质：<RoPE(q,m),RoPE(k,n)> 只依赖 m-n
    #    取固定向量 u,v，比较 (m,n)=(4,1) 与 (5,2)，相对位移都为 3
    u = torch.randn(dim)
    v = torch.randn(dim)

    def rope_dot(m, n):
        um = apply_rope(u.unsqueeze(0), cos[m:m + 1], sin[m:m + 1]).squeeze(0)
        vn = apply_rope(v.unsqueeze(0), cos[n:n + 1], sin[n:n + 1]).squeeze(0)
        return (um * vn).sum()

    assert torch.allclose(rope_dot(4, 1), rope_dot(5, 2), atol=1e-4), \
        "相对位置性质被破坏"

    # 3) 位置 0 = 恒等（angle=0 -> cos=1,sin=0）
    assert torch.allclose(apply_rope(q, cos, sin)[:, 0], q[:, 0], atol=1e-5)
    print("[PASS] rope: 保范数 + 相对位置不变 + pos0=恒等")


if __name__ == "__main__":
    _self_test()
