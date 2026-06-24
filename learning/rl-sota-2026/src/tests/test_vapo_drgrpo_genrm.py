"""VAPO + Dr.GRPO + GenRM + Capstone 测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from vapo_minimal import adaptive_lambda, length_adaptive_gae
from dr_grpo import (
    grpo_advantage,
    dr_grpo_advantage,
    grpo_length_weight,
    dr_grpo_length_weight,
)
from genrm import parse_genrm_score


# ===== VAPO =====

def test_adaptive_lambda_monotone():
    lens = torch.tensor([10.0, 100.0, 1000.0])
    lams = adaptive_lambda(lens)
    assert lams[0].item() < lams[1].item() < lams[2].item()


def test_adaptive_lambda_clamped():
    lens = torch.tensor([1.0, 1e6])
    lams = adaptive_lambda(lens, lam_min=0.5, lam_max=0.99)
    assert lams[0].item() >= 0.5
    assert lams[1].item() <= 0.99


def test_length_adaptive_gae_runs():
    B, T = 2, 5
    rewards = torch.zeros(B, T); rewards[:, -1] = 1.0
    values = torch.zeros(B, T)
    dones = torch.zeros(B, T); dones[:, -1] = 1
    lens = torch.tensor([5.0, 5.0])
    adv = length_adaptive_gae(rewards, values, dones, lens)
    assert adv.shape == (B, T)
    assert adv[0, -1].item() == 1.0


# ===== Dr. GRPO（对齐 Sea AI Lab 论文：去 std 除法 + 去长度归一）=====

def test_dr_grpo_advantage_centered_no_std():
    """Dr.GRPO advantage = R − group_mean，只中心化、不除 std。"""
    rewards = torch.tensor([0.0, 1.0, 2.0, 1.0])  # mean = 1.0
    A = dr_grpo_advantage(rewards, k=4)
    assert torch.allclose(A, rewards - rewards.mean(), atol=1e-6)
    # 且确与 std-normalize 的 GRPO 有别（证明真去掉了 /std）
    assert not torch.allclose(A, grpo_advantage(rewards, k=4), atol=1e-3)


def test_dr_grpo_removes_difficulty_bias():
    """① 低方差组：GRPO 把微小差异 /std 放大到 ~±1.2；Dr.GRPO 保持原始小尺度。"""
    rewards = torch.tensor([0.45, 0.5, 0.5, 0.55])  # std 很小
    assert grpo_advantage(rewards, k=4).abs().max() > 1.0
    assert dr_grpo_advantage(rewards, k=4).abs().max() < 0.1


def test_dr_grpo_removes_length_bias():
    """② Dr.GRPO 用常数归一：不同长度 response 权重相同；GRPO 的 1/|o| 随长度变。"""
    lens = torch.tensor([10.0, 200.0])
    w_dr = dr_grpo_length_weight(lens, l_const=200.0)
    assert torch.allclose(w_dr, w_dr[:1].expand_as(w_dr))  # 全相等 → 去偏
    w_grpo = grpo_length_weight(lens)
    assert w_grpo[0] > w_grpo[1]  # 短 response 权重更大（长度偏置）


# ===== GenRM =====

def test_parse_genrm_score():
    assert abs(parse_genrm_score("blah Score: 8") - 0.8) < 1e-6
    assert abs(parse_genrm_score("Score: 10") - 1.0) < 1e-6
    assert parse_genrm_score("no score here") is None


def test_parse_genrm_score_clamp():
    # 超出 10 也归一化到 ≤ 1
    assert parse_genrm_score("Score: 15") == 1.0


# ===== Capstone =====

def test_capstone_ablation_imports():
    from capstone_dapo_ablation import mock_ablation_run
    out = mock_ablation_run({"clip_higher": True, "dynamic_sampling": False,
                              "token_level": False, "overlong": False})
    assert "accuracy" in out
    assert out["accuracy"] > 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
