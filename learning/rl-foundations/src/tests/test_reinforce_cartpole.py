"""REINFORCE on CartPole 趋势测试 — 跑 200 ep 不要求 solved，但要求 reward 趋势上升。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

import gymnasium as gym

from common import compute_returns, set_seed
from reinforce_minimal import PolicyNet, run_episode, compute_loss


def test_reinforce_trend_up():
    """REINFORCE 跑 200 ep，前 50 ep 与后 50 ep 的均值至少差 30。"""
    set_seed(0)
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    policy = PolicyNet(state_dim=4, hidden=64, n_actions=2)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    returns: list[float] = []
    for ep in range(200):
        log_probs, rewards, ret = run_episode(env, policy, device="cpu")
        loss = compute_loss(log_probs, rewards, gamma=0.99, use_baseline=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        returns.append(ret)
    env.close()

    early = np.mean(returns[:50])
    late = np.mean(returns[-50:])
    assert late > early + 30, (
        f"REINFORCE 未明显学习：early avg {early:.1f}, late avg {late:.1f}"
    )


def test_compute_returns_basic():
    """compute_returns 在 γ=1 时应等于逆序累加。"""
    rewards = [1.0, 2.0, 3.0]
    dones = [False, False, True]
    G = compute_returns(rewards, dones, gamma=1.0)
    assert G == [6.0, 5.0, 3.0]


def test_compute_returns_with_done_reset():
    """在 done 后 G 应重置（下一个 episode）。"""
    rewards = [1.0, 1.0, 1.0, 10.0]
    dones = [False, False, True, True]  # 第 3 步是 ep 末
    G = compute_returns(rewards, dones, gamma=1.0)
    # 注意 compute_returns 的 done 语义：done 在 t 时 G_t 直接是 r_t，下一段累加新算
    # [G0, G1, G2, G3] = [3, 2, 1, 10]
    assert G == [3.0, 2.0, 1.0, 10.0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
