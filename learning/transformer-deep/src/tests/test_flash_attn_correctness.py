"""FlashAttention naive 数值正确性测试."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flash_attn_naive import vanilla_attn, flash_attn_naive   # noqa: E402


def test_naive_matches_vanilla_no_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, 16)
    k = torch.randn(2, 4, 64, 16)
    v = torch.randn(2, 4, 64, 16)
    out_vanilla = vanilla_attn(q, k, v, causal=False)
    out_flash = flash_attn_naive(q, k, v, block_r=16, block_c=16, causal=False)
    diff = (out_vanilla - out_flash).abs().max().item()
    assert diff < 1e-4, f"diff={diff}"


def test_naive_matches_vanilla_causal():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 8)
    k = torch.randn(1, 2, 32, 8)
    v = torch.randn(1, 2, 32, 8)
    out_vanilla = vanilla_attn(q, k, v, causal=True)
    out_flash = flash_attn_naive(q, k, v, block_r=8, block_c=8, causal=True)
    diff = (out_vanilla - out_flash).abs().max().item()
    assert diff < 1e-4, f"diff={diff}"


def test_different_block_sizes_consistent():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 24, 8)
    k = torch.randn(1, 1, 24, 8)
    v = torch.randn(1, 1, 24, 8)
    out_a = flash_attn_naive(q, k, v, block_r=4, block_c=4)
    out_b = flash_attn_naive(q, k, v, block_r=8, block_c=12)
    assert (out_a - out_b).abs().max() < 1e-4
