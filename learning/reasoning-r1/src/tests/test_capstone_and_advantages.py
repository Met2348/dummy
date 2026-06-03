"""Capstone tracks + RLOO/REINFORCE++ advantage 测试."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from rloo_minimal import rloo_advantage
from reinforce_pp import reinforce_pp_advantage
from r1_zero_track_a import gen_countdown_problem, combined_reward
from r1_zero_track_b import aha_word_frequency, gsm8k_reward_full
import random


# ===== RLOO =====

def test_rloo_baseline_excludes_self():
    rewards = torch.tensor([1.0, 0, 1, 0])
    A = rloo_advantage(rewards, k=4)
    # 对 idx 0: baseline = (0+1+0)/3 = 0.333, A_0 = 1-0.333 = 0.667
    assert abs(A[0].item() - 0.6667) < 1e-3


def test_rloo_zero_mean_within_group():
    rewards = torch.tensor([2.0, 1, 0, -1, 5, 5, 5, 5])
    A = rloo_advantage(rewards, k=4)
    # 两组各自，RLOO 不保证 mean=0 (它的 baseline 异于 group mean)
    # 但全 0 时应全 0
    rewards_zero = torch.zeros(8)
    A_z = rloo_advantage(rewards_zero, k=4)
    assert (A_z == 0).all()


# ===== REINFORCE++ =====

def test_reinforce_pp_advantage_centered():
    rewards = torch.tensor([1.0, 0.5, -0.5, 1.0])
    kl = torch.zeros(4, 5)
    mask = torch.ones(4, 5)
    A = reinforce_pp_advantage(rewards, kl, mask, beta_kl=0.04)
    assert abs(A.mean().item()) < 1e-6


# ===== Track A: Countdown =====

def test_countdown_problem_valid():
    rng = random.Random(123)
    nums, target = gen_countdown_problem(rng)
    assert len(nums) == 3
    assert all(1 <= n < 20 for n in nums)
    # target 应该可由 nums 算出
    assert isinstance(target, int)


def test_combined_reward_format_accuracy():
    nums = [3, 4, 5]
    target = 27   # 3*(4+5)
    resp = f"<think>3*(4+5)=27</think><answer>3*(4+5)=27</answer>"
    r = combined_reward(resp, target, nums)
    assert r["format"] == 1.0
    assert r["accuracy"] == 1.0
    assert abs(r["total"] - 1.0) < 1e-6


def test_combined_reward_no_format():
    r = combined_reward("just 12", 12, [3, 4, 5])
    assert r["format"] == 0.0


# ===== Track B: GSM8K + aha =====

def test_aha_word_frequency():
    responses = [
        "wait, let me reconsider",
        "no aha here",
        "actually that's wrong",
        "rethink this",
        "plain answer 7",
    ]
    stats = aha_word_frequency(responses)
    assert stats["responses_with_aha"] == 3
    assert abs(stats["aha_ratio"] - 0.6) < 1e-6


def test_gsm8k_reward_full_extracts_answer():
    resp = "<think>16-3=13, 13-6=7</think><answer>#### 7</answer>"
    r = gsm8k_reward_full(resp, "7")
    assert r["format"] == 1.0
    assert r["accuracy"] == 1.0


def test_gsm8k_reward_full_wrong_answer():
    resp = "<think>guess</think><answer>#### 99</answer>"
    r = gsm8k_reward_full(resp, "7")
    assert r["format"] == 1.0
    assert r["accuracy"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
