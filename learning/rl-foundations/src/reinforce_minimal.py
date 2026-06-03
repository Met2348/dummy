"""REINFORCE on CartPole（最朴素的 policy gradient）.

教学目标：
    1. 写出 PG 定理的 Pytorch 实现（不到 30 行核心代码）
    2. 观察单 episode 梯度噪声
    3. 看到 baseline / advantage 降方差效果

运行：
    python learning/rl-foundations/src/reinforce_minimal.py

预期：
    - 200 episode 后 last-10 平均 reward > 180（CartPole "solved" 阈值 195）
"""
from __future__ import annotations

import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common import compute_returns, set_seed


class PolicyNet(nn.Module):
    """两层 MLP，输出 logits。"""

    def __init__(self, state_dim: int, hidden: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_episode(env, policy, device: str = "cpu"):
    """跑一条完整 episode，返回 (log_probs, rewards, episode_return)."""
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    obs, _ = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        obs, r, terminated, truncated, _ = env.step(action.item())
        rewards.append(float(r))
        done = terminated or truncated

    ep_return = sum(rewards)
    return log_probs, rewards, ep_return


def compute_loss(
    log_probs: list[torch.Tensor],
    rewards: list[float],
    gamma: float,
    use_baseline: bool,
) -> torch.Tensor:
    """REINFORCE loss = -Σ_t log π(a_t|s_t) · G_t  (可选减 baseline)."""
    dones = [False] * (len(rewards) - 1) + [True]
    G = compute_returns(rewards, dones, gamma)
    G_t = torch.tensor(G, dtype=torch.float32)
    if use_baseline:
        # 用 episode 的均值当 baseline（最简单 baseline，无 critic）
        G_t = G_t - G_t.mean()
    # 不除标准差 — 留到 advantage 步再做（看消融效果）
    log_pi = torch.stack(log_probs)
    loss = -(log_pi * G_t).sum()
    return loss


def train(args):
    set_seed(args.seed)
    env = gym.make(args.env)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    policy = PolicyNet(state_dim, args.hidden, n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    ep_returns: deque = deque(maxlen=args.window)
    last10_returns: list[float] = []

    for ep in range(1, args.n_episodes + 1):
        log_probs, rewards, ret = run_episode(env, policy, device=device)
        loss = compute_loss(log_probs, rewards, args.gamma, args.baseline)
        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_returns.append(ret)
        if ep % args.log_interval == 0:
            avg = np.mean(ep_returns)
            last10_returns.append(avg)
            tag = "[BASELINE]" if args.baseline else "[VANILLA]"
            print(f"{tag} Ep {ep:4d} | last-{args.window} avg = {avg:6.1f} | "
                  f"loss = {loss.item():8.2f}")

    env.close()

    # 简单 "通过" 判定：最后 10 logs 的均值 > 180
    solved = (sum(last10_returns[-10:]) / max(1, len(last10_returns[-10:]))) > 180
    print(f"\n{'SOLVED ✅' if solved else 'NOT SOLVED ⚠️'} | "
          f"avg(last 10 logs) = {sum(last10_returns[-10:]) / max(1, len(last10_returns[-10:])):.1f}")
    return solved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--baseline", action="store_true",
                   help="减 episode mean 当 baseline（降方差）")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
