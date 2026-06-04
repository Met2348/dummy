"""Mamba 数值合法性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamba_block import MambaBlock, selective_scan_naive


def test_mamba_shape():
    m = MambaBlock(d_model=16, d_state=8)
    x = torch.randn(1, 10, 16)
    assert m(x).shape == x.shape


def test_mamba_grad():
    m = MambaBlock(d_model=16, d_state=8)
    x = torch.randn(1, 10, 16, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_selective_scan_no_nan():
    b, t, d, s = 1, 8, 4, 4
    u = torch.randn(b, t, d)
    dt = torch.rand(b, t, d) + 0.1
    A = -torch.arange(1, s + 1).float().unsqueeze(0).repeat(d, 1)
    B = torch.randn(b, t, s)
    C = torch.randn(b, t, s)
    y = selective_scan_naive(u, dt, A, B, C)
    assert y.shape == (b, t, d)
    assert torch.isfinite(y).all()
