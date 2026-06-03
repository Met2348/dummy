"""RMSNorm + SwiGLU 测试 — 数值正确性 + 梯度流."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmsnorm import RMSNorm        # noqa: E402
from swiglu import SwiGLUMLP, GELUMLP   # noqa: E402


def test_rmsnorm_output_unit_rms():
    """RMSNorm 输出 RMS ≈ 1."""
    torch.manual_seed(0)
    d = 64
    norm = RMSNorm(d)
    x = torch.randn(4, 10, d) * 3.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().mean() < 0.05, f"rms = {rms.mean()}"


def test_rmsnorm_grad_flows():
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64, requires_grad=True)
    y = norm(x)
    y.sum().backward()
    assert x.grad is not None
    assert norm.gamma.grad is not None


def test_swiglu_shape():
    m = SwiGLUMLP(d_model=64)
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_swiglu_params_match_ratio():
    """SwiGLU d_ff = 8/3 * d → 3 个 matrix ≈ 8 d² params."""
    m = SwiGLUMLP(d_model=64)
    n = sum(p.numel() for p in m.parameters())
    # 3 × 64 × (8/3 × 64) = 8 × 64² = 32768
    assert 30000 < n < 35000, f"params {n} not within ratio"


def test_swiglu_vs_gelu_params_close():
    """对比 GELU 标准 4d MLP, SwiGLU 2.67d 参数接近."""
    sw = SwiGLUMLP(d_model=128)
    ge = GELUMLP(d_model=128)
    n_sw = sum(p.numel() for p in sw.parameters())
    n_ge = sum(p.numel() for p in ge.parameters())
    ratio = n_sw / n_ge
    # 两者应在 ±20% 内
    assert 0.8 < ratio < 1.2, f"ratio {ratio}"
