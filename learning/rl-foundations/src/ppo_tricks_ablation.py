"""PPO 7 件套 ablation — 单独关掉每个 trick 看影响.

教学目标：复现 Engstrom 2020 的简化版结论：
    - advantage normalization 是 PPO 训得动的关键
    - 其他 trick 在 CartPole 上影响相对小

运行（约 40 分钟，5 配置 × 100k step）:
    python learning/rl-foundations/src/ppo_tricks_ablation.py
    python learning/rl-foundations/src/ppo_tricks_ablation.py --steps 30_000  # 快速 smoke
"""
from __future__ import annotations

import argparse
import copy
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common import compute_gae, set_seed


def orthogonal_init_(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class PolicyValue(nn.Module):
    def __init__(self, state_dim: int, hidden: int, n_actions: int,
                 use_orthog_init: bool = True) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        if use_orthog_init:
            orthogonal_init_(self.l1, np.sqrt(2))
            orthogonal_init_(self.l2, np.sqrt(2))
            orthogonal_init_(self.actor, 0.01)
            orthogonal_init_(self.critic, 1.0)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.l1(x))
        h = torch.tanh(self.l2(h))
        return self.actor(h), self.critic(h).squeeze(-1)


def run_ppo(cfg: dict) -> float:
    """跑一次 PPO，返回最后 100 ep 平均 reward。"""
    set_seed(cfg["seed"])
    envs = gym.vector.SyncVectorEnv([
        lambda i=i: gym.make(cfg["env"]) for i in range(cfg["n_envs"])
    ])
    envs.reset(seed=cfg["seed"])

    state_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    model = PolicyValue(state_dim, cfg["hidden"], n_actions,
                        use_orthog_init=cfg["orthog_init"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    obs, _ = envs.reset(seed=cfg["seed"])
    ep_returns: deque = deque(maxlen=100)
    running = np.zeros(cfg["n_envs"])

    total_iters = cfg["total_steps"] // (cfg["n_steps"] * cfg["n_envs"])
    batch_size = cfg["n_steps"] * cfg["n_envs"]
    mb_size = batch_size // cfg["n_minibatches"]

    for it in range(1, total_iters + 1):
        obs_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"], state_dim)
        act_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"], dtype=torch.long)
        logp_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"])
        val_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"])
        rew_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"])
        done_buf = torch.zeros(cfg["n_steps"], cfg["n_envs"])

        for t in range(cfg["n_steps"]):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, V = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            next_obs, r, term, trunc, _ = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            obs_buf[t] = obs_t
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = V
            rew_buf[t] = torch.as_tensor(r, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(done.astype(np.float32))
            running += r
            for i, d in enumerate(done):
                if d:
                    ep_returns.append(float(running[i]))
                    running[i] = 0.0
            obs = next_obs

        with torch.no_grad():
            _, last_V = model(torch.as_tensor(obs, dtype=torch.float32))

        adv, ret = compute_gae(rew_buf, val_buf, done_buf,
                               last_value=last_V.mean().item(),
                               gamma=cfg["gamma"], lam=cfg["lam"])
        if cfg["adv_norm"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_flat = obs_buf.reshape(-1, state_dim)
        act_flat = act_buf.reshape(-1)
        logp_old_flat = logp_buf.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        for _ in range(cfg["K_epochs"]):
            idx = np.random.permutation(batch_size)
            for start in range(0, batch_size, mb_size):
                mb = idx[start:start + mb_size]
                logits, V_new = model(obs_flat[mb])
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_flat[mb])
                ratio = (logp_new - logp_old_flat[mb]).exp()
                surr1 = ratio * adv_flat[mb]
                surr2 = ratio.clamp(1 - cfg["eps"], 1 + cfg["eps"]) * adv_flat[mb]
                L_clip = -torch.min(surr1, surr2).mean()
                L_vf = F.mse_loss(V_new, ret_flat[mb])
                entropy = dist.entropy().mean()
                loss = L_clip + cfg["vf_coef"] * L_vf - cfg["ent_coef"] * entropy
                opt.zero_grad()
                loss.backward()
                if cfg["max_grad_norm"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                opt.step()

    envs.close()
    return float(np.mean(ep_returns)) if ep_returns else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = dict(
        env="CartPole-v1", n_envs=8, n_steps=128, hidden=64,
        lr=3e-4, gamma=0.99, lam=0.95, eps=0.2,
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
        K_epochs=10, n_minibatches=4,
        adv_norm=True, orthog_init=True,
        total_steps=args.steps, seed=args.seed,
    )

    configs = [
        ("baseline (all on)", {}),
        ("-adv norm", dict(adv_norm=False)),
        ("-grad clip", dict(max_grad_norm=None)),
        ("-orthog init", dict(orthog_init=False)),
        ("-(adv + grad + orthog)", dict(adv_norm=False, max_grad_norm=None,
                                        orthog_init=False)),
    ]
    print(f"\nPPO ablation @ {args.steps:,} env steps × 5 configs")
    print("=" * 60)
    results = []
    for name, override in configs:
        cfg = copy.deepcopy(base)
        cfg.update(override)
        final = run_ppo(cfg)
        results.append((name, final))
        print(f"  {name:<28} mean reward = {final:6.1f}")

    print("\n" + "=" * 60)
    print("expected (CartPole-v1, ~500 max):")
    print("  baseline ≈ 500, -adv norm ≈ 350, -others ≈ 450~500")


if __name__ == "__main__":
    main()
