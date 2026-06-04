"""Ring attention 数值正确性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ring_attention_naive import vanilla_attn, ring_attention_naive


def test_ring_matches_vanilla():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out_v = vanilla_attn(q, k, v)
    out_r = ring_attention_naive(q, k, v, n_rank=4)
    diff = (out_v - out_r).abs().max().item()
    assert diff < 1e-4, f"diff {diff}"


def test_ring_different_n_rank():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 24, 4)
    k = torch.randn(1, 1, 24, 4)
    v = torch.randn(1, 1, 24, 4)
    out_a = ring_attention_naive(q, k, v, n_rank=2)
    out_b = ring_attention_naive(q, k, v, n_rank=4)
    assert (out_a - out_b).abs().max() < 1e-4
