"""VAPO + Dr.GRPO + GenRM + Capstone 测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from vapo_minimal import adaptive_lambda, length_adaptive_gae
from dr_grpo import mad_normalize, dr_grpo_advantage
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


# ===== Dr. GRPO =====

def test_mad_outlier_robustness():
    rewards = torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 100.0])
    A_mad = mad_normalize(rewards, k=8)
    # std-based GRPO 会让 normal 样本 advantage 接近 0；MAD 让它们仍有意义
    normal_part = A_mad[:7]
    assert normal_part.abs().mean() > 0.1


def test_dr_grpo_length_penalty():
    rewards = torch.tensor([1.0, 1, 0, 0])
    lens_short = torch.tensor([10.0] * 4)
    lens_long = torch.tensor([100.0] * 4)
    A_s = dr_grpo_advantage(rewards, lens_short, k=4, beta_len=0.05)
    A_l = dr_grpo_advantage(rewards, lens_long, k=4, beta_len=0.05)
    # 长 response 被惩罚
    assert A_l.mean() < A_s.mean()


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
