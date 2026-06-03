"""DPO loss 单元测试：数值正确性 + 边界行为."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from dpo_minimal import dpo_loss


def test_dpo_loss_zero_margin():
    """log_ratio_w == log_ratio_l → loss = -log 0.5 = log 2."""
    L = dpo_loss(
        log_p_chosen_actor=torch.tensor([0.0]),
        log_p_chosen_ref=torch.tensor([0.0]),
        log_p_rejected_actor=torch.tensor([0.0]),
        log_p_rejected_ref=torch.tensor([0.0]),
        beta=0.1,
    ).item()
    assert abs(L - 0.6931) < 1e-3, L


def test_dpo_loss_positive_margin():
    """chosen 概率明显大 → loss 显著小于 log 2."""
    L = dpo_loss(
        log_p_chosen_actor=torch.tensor([0.0]),
        log_p_chosen_ref=torch.tensor([-2.0]),    # log_ratio_w = +2
        log_p_rejected_actor=torch.tensor([-2.0]),
        log_p_rejected_ref=torch.tensor([0.0]),   # log_ratio_l = -2
        beta=0.1,
    ).item()
    # β · (2 - (-2)) = 0.4
    # -log sigmoid(0.4) = ~0.51
    assert L < 0.6, L


def test_dpo_loss_beta_effect():
    """β 加大放大 margin 效应。"""
    args = dict(
        log_p_chosen_actor=torch.tensor([1.0]),
        log_p_chosen_ref=torch.tensor([0.0]),
        log_p_rejected_actor=torch.tensor([0.0]),
        log_p_rejected_ref=torch.tensor([0.0]),
    )
    L1 = dpo_loss(beta=0.1, **args).item()
    L5 = dpo_loss(beta=1.0, **args).item()
    assert L5 < L1, (L1, L5)   # 大 β → 更小 loss（chosen 已经赢了）


def test_dpo_loss_negative_margin():
    """chosen 概率更小 → loss 大。"""
    L = dpo_loss(
        log_p_chosen_actor=torch.tensor([-3.0]),
        log_p_chosen_ref=torch.tensor([0.0]),    # log_ratio_w = -3
        log_p_rejected_actor=torch.tensor([0.0]),
        log_p_rejected_ref=torch.tensor([0.0]),  # log_ratio_l = 0
        beta=1.0,                                 # 用大 beta 让对比明显
    ).item()
    # β · (-3 - 0) = -3
    # -log sigmoid(-3) = ~3.05
    assert L > 2.5, L


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
