"""S4 sanity: forward 输出 shape 正确."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4_naive import S4Naive


def test_s4_forward_shape():
    m = S4Naive(d_model=4, d_state=8)
    u = torch.randn(1, 10, 4)
    y = m(u)
    assert y.shape == u.shape


def test_s4_grad_flows():
    m = S4Naive(d_model=4, d_state=8)
    u = torch.randn(1, 10, 4, requires_grad=True)
    y = m(u).sum()
    y.backward()
    assert u.grad is not None


def test_s4_no_nan():
    m = S4Naive(d_model=4, d_state=8)
    u = torch.randn(1, 10, 4)
    y = m(u)
    assert torch.isfinite(y).all()
