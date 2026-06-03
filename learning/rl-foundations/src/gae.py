"""GAE (Generalized Advantage Estimation) 独立模块.

`compute_gae` 已在 common.py。这里追加：
- numpy 版本（便于教学，可读性最强）
- 数值小测试套件
"""
from __future__ import annotations

import numpy as np
import torch


def gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 实现，便于在教学 notebook 里看清逐步。

    rewards/values/dones shape: (T,) 或 (T, N).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def gae_lambda_zero_equals_td(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float = 0.0,
    gamma: float = 0.99,
) -> np.ndarray:
    """λ=0 时 A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t) （朴素 TD）。"""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    for t in range(T):
        v_next = values[t + 1] if t + 1 < T else last_value
        mask = 1.0 - dones[t]
        adv[t] = rewards[t] + gamma * v_next * mask - values[t]
    return adv


def gae_lambda_one_equals_mc(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float = 0.0,
    gamma: float = 0.99,
) -> np.ndarray:
    """λ=1 时 A_t = G_t - V(s_t)（Monte-Carlo）.

    G_t 用 last_value bootstrap：G_T = last_value 然后逆序 G_t = r_t + γ G_{t+1} (mask).
    """
    T = rewards.shape[0]
    G = np.zeros_like(rewards, dtype=np.float64)
    R = last_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        R = rewards[t] + gamma * R * mask
        G[t] = R
    return G - values


def _self_test():
    """跑几个小 case，验证 GAE 退化关系。"""
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.3, 0.2])
    dones = np.array([0.0, 0.0, 1.0])
    last_V = 0.0

    A_lam0, _ = gae_numpy(rewards, values, dones, last_V, gamma=1.0, lam=0.0)
    A_td = gae_lambda_zero_equals_td(rewards, values, dones, last_V, gamma=1.0)
    print("λ=0 vs TD:", np.allclose(A_lam0, A_td), A_lam0, A_td)

    A_lam1, _ = gae_numpy(rewards, values, dones, last_V, gamma=1.0, lam=1.0)
    A_mc = gae_lambda_one_equals_mc(rewards, values, dones, last_V, gamma=1.0)
    print("λ=1 vs MC:", np.allclose(A_lam1, A_mc), A_lam1, A_mc)

    A_095, R_095 = gae_numpy(rewards, values, dones, last_V, gamma=1.0, lam=0.95)
    print(f"λ=0.95 A = {A_095}")
    # 上 lecture 例子：A0=2.377, A1=1.66, A2=0.8
    print(f"  expected approx [2.377, 1.66, 0.8]")


if __name__ == "__main__":
    _self_test()
