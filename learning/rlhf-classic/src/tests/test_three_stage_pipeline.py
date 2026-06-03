"""RLHF 三段管线 smoke tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from sft_minimal import sft_loss
from rm_minimal import RewardModel, bt_loss
from ppo_llm_minimal import build_token_rewards, gather_logp
from reward_hacking_demo import detect_hacking, hackable_reward


# ===== SFT =====

def test_sft_loss_masks_prompt():
    """label = -100 处不计 loss."""
    B, T, V = 2, 8, 50
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, 5:] = torch.randint(0, V, (B, T - 5))
    loss = sft_loss(logits, labels)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


# ===== RM =====

def test_bt_loss_basic():
    """chosen > rejected 时 loss < log 2."""
    r_c = torch.tensor([2.0, 1.0])
    r_r = torch.tensor([0.0, 0.5])
    L = bt_loss(r_c, r_r)
    assert L.item() < 0.7   # < log 2


def test_bt_loss_perfect_separation():
    """rc - rr 很大 → loss → 0."""
    r_c = torch.tensor([10.0])
    r_r = torch.tensor([-10.0])
    L = bt_loss(r_c, r_r)
    assert L.item() < 1e-6


# ===== PPO =====

def test_token_reward_structure():
    """KL 在每步, RM 只在 last response token."""
    B, T = 2, 6
    response_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
    ], dtype=torch.float32)
    raw = torch.tensor([1.0, -0.5])
    log_p_act = torch.zeros(B, T)
    log_p_ref = torch.zeros(B, T)
    rewards = build_token_rewards(raw, response_mask, log_p_act, log_p_ref, beta=0.02)
    # KL=0 → 每步 reward 0；只末位有 RM
    # row 0 last idx = 5: rewards[0,5]=1.0
    # row 1 last idx = 4 (mask): rewards[1,4]=-0.5
    assert abs(rewards[0, 5].item() - 1.0) < 1e-6
    assert abs(rewards[1, 4].item() - (-0.5)) < 1e-6
    assert rewards[1, 5].item() == 0.0  # masked


def test_gather_logp_shape():
    B, T, V = 2, 5, 50
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    log_p = gather_logp(logits, input_ids)
    assert log_p.shape == (B, T - 1)


# ===== Reward Hacking =====

def test_hacking_detection():
    rewards = [0.1] * 10 + [0.5] * 10
    lens = [20.0] * 10 + [50.0] * 10
    d = detect_hacking(rewards, lens)
    assert d["detected"]


def test_hacking_no_false_positive():
    rewards = [0.3] * 20
    lens = [30.0] * 20
    d = detect_hacking(rewards, lens)
    assert not d["detected"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
