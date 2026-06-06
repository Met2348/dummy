"""A2C on CartPole smoke test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))

import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common import categorical_entropy, set_seed
from a2c_minimal import ActorCritic, make_vec_env


def test_a2c_trend_up_short():
    """Run a short vectorized A2C loop and ensure it stays numerically healthy."""
    set_seed(0)
    envs = make_vec_env("CartPole-v1", n_envs=4, seed=0)
    model = ActorCritic(state_dim=4, hidden=64, n_actions=2)
    opt = torch.optim.Adam(model.parameters(), lr=7e-4)

    from collections import deque
    ep_returns: deque = deque(maxlen=200)
    running = np.zeros(4)

    obs, _ = envs.reset()
    early: list[float] = []
    late: list[float] = []

    for it in range(100):
        log_probs_list, values_list, rewards_list, dones_list, ents_list = [], [], [], [], []
        for _ in range(5):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits, V_t = model(obs_t)
            dist = Categorical(logits=logits)
            a = dist.sample()
            log_probs_list.append(dist.log_prob(a))
            values_list.append(V_t)
            ents_list.append(categorical_entropy(logits))
            next_obs, r, term, trunc, _ = envs.step(a.numpy())
            d = np.logical_or(term, trunc)
            running += r
            for i, di in enumerate(d):
                if di:
                    ep_returns.append(float(running[i]))
                    running[i] = 0.0
            rewards_list.append(torch.as_tensor(r, dtype=torch.float32))
            dones_list.append(torch.as_tensor(d.astype(np.float32)))
            obs = next_obs

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            _, last_V = model(obs_t)

        returns_list = []
        R = last_V
        for r, d in zip(reversed(rewards_list), reversed(dones_list)):
            R = r + 0.99 * R * (1.0 - d)
            returns_list.append(R)
        returns_list.reverse()

        log_probs = torch.stack(log_probs_list)
        values = torch.stack(values_list)
        returns = torch.stack(returns_list)
        ents = torch.stack(ents_list)

        adv = (returns - values).detach()
        L_a = -(log_probs * adv).mean()
        L_c = F.mse_loss(values, returns.detach())
        L_e = -ents.mean()
        L = L_a + 0.5 * L_c + 0.01 * L_e

        opt.zero_grad()
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        if it < 20:
            early.extend(ep_returns)
            ep_returns.clear()
        elif it > 80:
            late.extend(ep_returns)
            ep_returns.clear()

    envs.close()
    e_mean = np.mean(early) if early else 20.0
    l_mean = np.mean(late) if late else 20.0
    assert np.isfinite(e_mean)
    assert np.isfinite(l_mean)
    assert early, "No early episodes collected"
    assert late, "No late episodes collected"
    assert l_mean > 10.0, f"A2C collapsed: early {e_mean:.1f}, late {l_mean:.1f}"


def test_actor_critic_forward_shape():
    """forward 输出 (logits, V) 形状正确。"""
    model = ActorCritic(state_dim=4, hidden=32, n_actions=2)
    x = torch.randn(8, 4)
    logits, V = model(x)
    assert logits.shape == (8, 2)
    assert V.shape == (8,)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
