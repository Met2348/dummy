"""RoPE 一致性测试 — interleaved 实现自身性质 + 相对位置不变."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rope import RoPE, build_cos_sin, apply_rope_interleaved   # noqa: E402


def test_rope_preserves_shape():
    rope = RoPE(dim=16)
    q = torch.randn(2, 3, 8, 16)
    k = torch.randn(2, 3, 8, 16)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape
    assert kr.shape == k.shape


def test_rope_relative_position_invariance():
    """<RoPE(q, m), RoPE(k, n)> 只依赖 n-m, 不依赖 m, n 绝对值."""
    torch.manual_seed(42)
    d = 16
    q = torch.randn(1, d)
    k = torch.randn(1, d)
    cos_all, sin_all = build_cos_sin(t=20, dim=d, base=10000.0)

    def score(m, n):
        qr = apply_rope_interleaved(q, cos_all[m:m+1], sin_all[m:m+1])
        kr = apply_rope_interleaved(k, cos_all[n:n+1], sin_all[n:n+1])
        return (qr * kr).sum().item()

    s1 = score(0, 3)
    s2 = score(7, 10)
    s3 = score(12, 15)
    # 相对距离都是 3，三个 score 应一致
    assert abs(s1 - s2) < 1e-4, f"{s1} vs {s2}"
    assert abs(s1 - s3) < 1e-4, f"{s1} vs {s3}"


def test_rope_at_zero_is_identity():
    """RoPE 在 pos=0 应等于 identity."""
    d = 16
    x = torch.randn(1, d)
    cos, sin = build_cos_sin(t=1, dim=d)
    xr = apply_rope_interleaved(x, cos[0:1], sin[0:1])
    assert torch.allclose(xr, x, atol=1e-6), (xr - x).abs().max()


def test_rope_base_changes_freq():
    """base 越大 → cos 周期越长."""
    cos1, _ = build_cos_sin(t=8, dim=8, base=100.0)
    cos2, _ = build_cos_sin(t=8, dim=8, base=10000.0)
    # 对于 高 dim (低频)，base 越大，cos 越平
    # cos1[:, 3] 振荡，cos2[:, 3] 接近 1
    assert cos2[:, 3].std().item() < cos1[:, 3].std().item()
