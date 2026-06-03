"""PPO 一致性 / GAE 数值测试.

测试范围：
1. GAE 退化测试 — λ=0 等价 1-step TD；λ=1 等价 MC（reward 用 last_value bootstrap）
2. GAE numerical correctness — 与逐步手算结果一致
3. PPO clip 单元 — A>0 / A<0 时 clip 行为正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

from common import compute_gae
from gae import gae_numpy, gae_lambda_zero_equals_td, gae_lambda_one_equals_mc


def test_gae_lambda_zero_equals_td():
    rng = np.random.default_rng(0)
    T = 8
    rewards = rng.standard_normal(T)
    values = rng.standard_normal(T)
    dones = np.zeros(T)
    dones[-1] = 1.0
    A_gae, _ = gae_numpy(rewards, values, dones, last_value=0.0,
                        gamma=0.99, lam=0.0)
    A_td = gae_lambda_zero_equals_td(rewards, values, dones, last_value=0.0,
                                     gamma=0.99)
    assert np.allclose(A_gae, A_td, atol=1e-10), (A_gae, A_td)


def test_gae_lambda_one_equals_mc():
    rng = np.random.default_rng(1)
    T = 8
    rewards = rng.standard_normal(T)
    values = rng.standard_normal(T)
    dones = np.zeros(T)
    dones[-1] = 1.0
    A_gae, _ = gae_numpy(rewards, values, dones, last_value=0.0,
                        gamma=0.99, lam=1.0)
    A_mc = gae_lambda_one_equals_mc(rewards, values, dones, last_value=0.0,
                                    gamma=0.99)
    assert np.allclose(A_gae, A_mc, atol=1e-10), (A_gae, A_mc)


def test_gae_known_case():
    """lecture L05 的手算例子：rewards=[1,1,1], V=[0.5,0.3,0.2], dones=[F,F,T], γ=1, λ=0.95.

    手算：
      t=2: δ = 1 + 0·0 - 0.2 = 0.8 → A_2 = 0.8
      t=1: δ = 1 + 0.2 - 0.3 = 0.9 → A_1 = 0.9 + 1·0.95·1·0.8 = 1.66
      t=0: δ = 1 + 0.3 - 0.5 = 0.8 → A_0 = 0.8 + 1·0.95·1·1.66 = 2.377
    """
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.3, 0.2])
    dones = np.array([0.0, 0.0, 1.0])
    A, _ = gae_numpy(rewards, values, dones, last_value=0.0,
                    gamma=1.0, lam=0.95)
    expected = np.array([2.377, 1.66, 0.8])
    assert np.allclose(A, expected, atol=1e-3), f"A={A}, expected={expected}"


def test_compute_gae_torch_matches_numpy():
    """common.compute_gae (torch) 与 gae.gae_numpy 应数值一致。"""
    rng = np.random.default_rng(2)
    T, N = 16, 4
    rewards_np = rng.standard_normal((T, N))
    values_np = rng.standard_normal((T, N))
    dones_np = np.zeros((T, N))
    dones_np[-1] = 1.0

    A_np, R_np = gae_numpy(rewards_np, values_np, dones_np, last_value=0.0,
                          gamma=0.99, lam=0.95)
    A_torch, R_torch = compute_gae(
        torch.as_tensor(rewards_np, dtype=torch.float32),
        torch.as_tensor(values_np, dtype=torch.float32),
        torch.as_tensor(dones_np, dtype=torch.float32),
        last_value=0.0, gamma=0.99, lam=0.95,
    )
    assert np.allclose(A_np, A_torch.numpy(), atol=1e-4)
    assert np.allclose(R_np, R_torch.numpy(), atol=1e-4)


def test_ppo_clip_behavior_a_positive():
    """A>0 时，ratio 超过 1+ε 后被 clip 住。"""
    eps = 0.2
    ratio = torch.tensor([0.5, 1.0, 1.1, 1.3, 2.0])
    A = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])    # 所有 A>0
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - eps, 1 + eps) * A
    L = -torch.min(surr1, surr2)
    # ratio=2.0 (>1+ε) → surr2=1.2，surr1=2.0；min=1.2 → -L=-1.2
    # ratio=0.5 (<1-ε) → surr2=0.8，surr1=0.5；min=0.5 → -L=-0.5
    # A>0 时 clip 只在 ratio 超过 1+ε 时生效（min(2.0, 1.2) = 1.2）
    assert torch.allclose(L[4], torch.tensor(-1.2)), L
    # ratio=0.5 不被 clip （A>0 时上界 clip 生效）
    assert torch.allclose(L[0], torch.tensor(-0.5)), L


def test_ppo_clip_behavior_a_negative():
    """A<0 时，ratio 低于 1-ε 后被 clip 住。"""
    eps = 0.2
    ratio = torch.tensor([0.5, 1.0, 1.3])
    A = torch.tensor([-1.0, -1.0, -1.0])
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - eps, 1 + eps) * A
    L = -torch.min(surr1, surr2)
    # ratio=0.5 (<1-ε): surr2 = 0.8·(-1) = -0.8, surr1 = 0.5·(-1) = -0.5
    # min(-0.5, -0.8) = -0.8 → L = 0.8
    assert torch.allclose(L[0], torch.tensor(0.8)), L


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
