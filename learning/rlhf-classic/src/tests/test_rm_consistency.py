"""RM 单元测试 — BT loss 数值 + last-token 取值正确."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from rm_minimal import bt_loss


def test_bt_loss_zero_diff():
    """r_chosen == r_rejected 时 BT loss = -log 0.5 = log 2."""
    r_c = torch.tensor([1.0, 2.0, 0.0])
    r_r = torch.tensor([1.0, 2.0, 0.0])
    L = bt_loss(r_c, r_r).item()
    assert abs(L - 0.6931) < 1e-3, L


def test_bt_loss_positive_gap():
    """r_chosen > r_rejected → loss 显著小于 log 2."""
    r_c = torch.tensor([5.0])
    r_r = torch.tensor([0.0])
    L = bt_loss(r_c, r_r).item()
    assert L < 0.01, L


def test_bt_loss_negative_gap():
    """r_chosen < r_rejected → loss 显著大于 log 2（"反着学了"）."""
    r_c = torch.tensor([0.0])
    r_r = torch.tensor([5.0])
    L = bt_loss(r_c, r_r).item()
    assert L > 4.0, L


def test_last_token_indexing():
    """模拟 attention_mask sum-1 取 last token 正确。"""
    B, T, hidden = 3, 5, 4
    h = torch.randn(B, T, hidden)
    attn = torch.tensor([
        [1, 1, 1, 1, 0],   # last = 3
        [1, 1, 1, 0, 0],   # last = 2
        [1, 1, 1, 1, 1],   # last = 4
    ])
    last_idx = attn.sum(-1) - 1
    last_h = h[torch.arange(B), last_idx]
    assert torch.allclose(last_h[0], h[0, 3])
    assert torch.allclose(last_h[1], h[1, 2])
    assert torch.allclose(last_h[2], h[2, 4])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
