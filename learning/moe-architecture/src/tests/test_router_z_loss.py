"""z-loss + grouped GEMM 测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router_z_loss import router_z_loss
import grouped_gemm_demo


def test_z_loss_finite_and_positive():
    logits = torch.randn(10, 8)
    z = router_z_loss(logits)
    assert torch.isfinite(z)
    assert z.item() >= 0


def test_z_loss_grad_flows():
    logits = torch.randn(10, 8, requires_grad=True)
    z = router_z_loss(logits)
    z.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_grouped_gemm_naive_correct():
    torch.manual_seed(0)
    xs = [torch.randn(4, 8), torch.randn(4, 8)]
    ws = [torch.randn(8, 8), torch.randn(8, 8)]
    outs = grouped_gemm_demo.grouped_gemm_naive(xs, ws)
    assert len(outs) == 2
    assert outs[0].shape == (4, 8)
    # 数值检查
    assert torch.allclose(outs[0], xs[0] @ ws[0])
