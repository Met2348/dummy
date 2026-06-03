"""A2C (Synchronous Advantage Actor-Critic) on CartPole.

教学目标：
    1. 写出 actor-critic 共享 backbone
    2. 实现 1-step TD advantage 与 c_v / β 加权
    3. n-step rollout（默认 n=5）

运行：
    python learning/rl-foundations/src/a2c_minimal.py
    python learning/rl-foundations/src/a2c_minimal.py --n-envs 8 --total-steps 100_000

预期：
    - 50k step 后 eval mean reward > 400
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

from common import categorical_entropy, set_seed


class ActorCritic(nn.Module):
    """共享 2 层 MLP backbone + actor head + critic head。"""

    def __init__(self, state_dim: int, hidden: int, n_actions: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def make_vec_env(env_id: str, n_envs: int, seed: int):
    def make(idx):
        def _f():
            env = gym.make(env_id)
            env.reset(seed=seed + idx)
            return env
        return _f
    return gym.vector.SyncVectorEnv([make(i) for i in range(n_envs)])


def train(args):
    set_seed(args.seed)
    envs = make_vec_env(args.env, args.n_envs, args.seed)

    state_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model = ActorCritic(state_dim, args.hidden, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = envs.reset(seed=args.seed)
    ep_returns_window: deque = deque(maxlen=100)
    running_returns = np.zeros(args.n_envs, dtype=np.float64)

    total_iters = args.total_steps // (args.n_steps * args.n_envs)
    for it in range(1, total_iters + 1):
        # ---- rollout n_steps ----
        log_probs_list: list[torch.Tensor] = []
        values_list: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []
        dones_list: list[torch.Tensor] = []
        entropies_list: list[torch.Tensor] = []

        for _ in range(args.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            logits, V_t = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_p = dist.log_prob(action)
            ent = categorical_entropy(logits)

            next_obs, rewards, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            running_returns += rewards
            for i, d in enumerate(done):
                if d:
                    ep_returns_window.append(running_returns[i])
                    running_returns[i] = 0.0

            log_probs_list.append(log_p)
            values_list.append(V_t)
            rewards_list.append(
                torch.as_tensor(rewards, dtype=torch.float32, device=device)
            )
            dones_list.append(
                torch.as_tensor(done, dtype=torch.float32, device=device)
            )
            entropies_list.append(ent)
            obs = next_obs

        # bootstrap value
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, last_V = model(obs_t)

        # ---- compute n-step returns 与 advantages ----
        returns_list: list[torch.Tensor] = []
        R = last_V
        for r, d in zip(reversed(rewards_list), reversed(dones_list)):
            R = r + args.gamma * R * (1.0 - d)
            returns_list.append(R)
        returns_list.reverse()

        log_probs = torch.stack(log_probs_list)               # (n_steps, n_envs)
        values = torch.stack(values_list)                     # (n_steps, n_envs)
        returns = torch.stack(returns_list)                   # (n_steps, n_envs)
        entropies = torch.stack(entropies_list)               # (n_steps,)

        advantages = (returns - values).detach()              # critic 不传梯度回 advantage

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns.detach())
        entropy_loss = -entropies.mean()                       # 加号→鼓励熵

        loss = actor_loss + args.vf_coef * critic_loss + args.ent_coef * entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()

        if it % args.log_interval == 0:
            mean_R = np.mean(ep_returns_window) if ep_returns_window else 0.0
            print(f"Iter {it:5d} | env steps {it * args.n_steps * args.n_envs:7d} | "
                  f"ep mean {mean_R:6.1f} | "
                  f"L_actor {actor_loss.item():7.3f} | L_critic {critic_loss.item():7.3f}")

    envs.close()
    final_mean = np.mean(ep_returns_window) if ep_returns_window else 0.0
    solved = final_mean > 400
    print(f"\n{'SOLVED ✅' if solved else 'NOT SOLVED ⚠️'} | "
          f"final ep mean (over last 100) = {final_mean:.1f}")
    return solved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
