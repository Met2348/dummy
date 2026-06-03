"""简化版 TRPO on CartPole.

教学折衷：跳过共轭梯度 + Fisher-vector，把 natural gradient 近似为普通梯度，
但保留 **backtracking line search**（KL 超限就缩半 retry）。

—— 行为类似 TRPO，但少了二阶曲率加速。后续在 PPO 中会看到更优雅的"一行 clip"实现。

运行：
    python learning/rl-foundations/src/trpo_minimal.py

预期：50k env step 后 ep mean ≥ 400
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


class PolicyValue(nn.Module):
    """共享 backbone + actor + critic。"""

    def __init__(self, state_dim: int, hidden: int, n_actions: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.shared(x))


def collect_rollout(envs, model, n_steps: int, obs):
    """收 n_steps 数据。"""
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, V = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        next_obs, r, term, trunc, _ = envs.step(action.numpy())
        done = np.logical_or(term, trunc)

        obs_buf.append(obs_t)
        act_buf.append(action)
        logp_buf.append(logp)
        val_buf.append(V)
        rew_buf.append(torch.as_tensor(r, dtype=torch.float32))
        done_buf.append(torch.as_tensor(done.astype(np.float32)))
        obs = next_obs

    return obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, obs


def flatten(t_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in t_list])


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view(p.shape))
        idx += n


def compute_kl(model: PolicyValue, model_old: PolicyValue, obs: torch.Tensor) -> torch.Tensor:
    """E_s [ KL( π_old(·|s) || π(·|s) ) ]。"""
    with torch.no_grad():
        logits_old = model_old.get_logits(obs)
    logits_new = model.get_logits(obs)
    log_old = F.log_softmax(logits_old, dim=-1)
    log_new = F.log_softmax(logits_new, dim=-1)
    return (log_old.exp() * (log_old - log_new)).sum(-1).mean()


def compute_surrogate(
    model: PolicyValue,
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """L_surr = E [ exp(log π_new - log π_old) · A ]."""
    logits = model.get_logits(obs)
    log_pi_new = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = (log_pi_new - log_probs_old).exp()
    return (ratio * advantages).mean()


def train(args):
    set_seed(args.seed)
    envs = gym.vector.SyncVectorEnv([
        lambda i=i: gym.make(args.env) for i in range(args.n_envs)
    ])
    envs.reset(seed=args.seed)
    state_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    model = PolicyValue(state_dim, args.hidden, n_actions)
    opt_v = torch.optim.Adam(
        list(model.shared.parameters()) + list(model.critic.parameters()),
        lr=args.vf_lr,
    )

    obs, _ = envs.reset(seed=args.seed)
    ep_returns: deque = deque(maxlen=100)
    running = np.zeros(args.n_envs)

    total_iters = args.total_steps // (args.n_steps * args.n_envs)
    for it in range(1, total_iters + 1):
        ob, ac, lp, va, rw, dn, obs = collect_rollout(envs, model, args.n_steps, obs)
        # update running episode returns
        for r_step, d_step in zip(rw, dn):
            running += r_step.numpy()
            for i, d in enumerate(d_step.numpy()):
                if d:
                    ep_returns.append(float(running[i]))
                    running[i] = 0.0

        # ---- GAE ----
        with torch.no_grad():
            _, last_V = model(torch.as_tensor(obs, dtype=torch.float32))
        rewards = torch.stack(rw)        # (T, N)
        values = torch.stack(va)
        dones = torch.stack(dn)
        adv, ret = compute_gae(rewards, values, dones, last_value=last_V.mean().item(),
                               gamma=args.gamma, lam=args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_flat = torch.cat(ob).reshape(-1, state_dim)
        act_flat = torch.cat(ac).reshape(-1)
        logp_old_flat = torch.cat(lp).reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        # ---- compute policy gradient ----
        model_old = copy.deepcopy(model)

        loss_pi = -compute_surrogate(model, obs_flat, act_flat, logp_old_flat, adv_flat)
        grads = torch.autograd.grad(loss_pi, [p for p in model.actor.parameters()] +
                                            [p for p in model.shared.parameters()],
                                    allow_unused=True)
        g = flatten([gg if gg is not None else torch.zeros_like(p)
                     for gg, p in zip(grads,
                                      list(model.actor.parameters()) +
                                      list(model.shared.parameters()))])

        # ---- backtracking line search ----
        flat_old = torch.cat([p.data.view(-1) for p in model.actor.parameters()] +
                             [p.data.view(-1) for p in model.shared.parameters()])
        step = args.step_size
        accepted = False
        for _ in range(args.max_backtracks):
            new_flat = flat_old - step * g   # 注意：L_surr 已是负号(loss)，故 -g
            idx = 0
            for p in list(model.actor.parameters()) + list(model.shared.parameters()):
                n = p.numel()
                p.data.copy_(new_flat[idx:idx + n].view(p.shape))
                idx += n
            kl = compute_kl(model, model_old, obs_flat).item()
            with torch.no_grad():
                new_loss_pi = -compute_surrogate(model, obs_flat, act_flat, logp_old_flat, adv_flat).item()
            if kl < args.max_kl and new_loss_pi < loss_pi.item():
                accepted = True
                break
            step /= 2

        if not accepted:
            # 还原
            idx = 0
            for p in list(model.actor.parameters()) + list(model.shared.parameters()):
                n = p.numel()
                p.data.copy_(flat_old[idx:idx + n].view(p.shape))
                idx += n

        # ---- update critic（普通 SGD） ----
        for _ in range(args.vf_iters):
            opt_v.zero_grad()
            _, V_new = model(obs_flat)
            L_v = F.mse_loss(V_new, ret_flat)
            L_v.backward()
            opt_v.step()

        if it % args.log_interval == 0:
            mean_R = np.mean(ep_returns) if ep_returns else 0.0
            print(f"Iter {it:5d} | env steps {it * args.n_steps * args.n_envs:7d} | "
                  f"ep mean {mean_R:6.1f} | KL {kl:6.4f} | "
                  f"{'accepted' if accepted else 'REJECTED'} step={step:.4e}")

    envs.close()
    final = np.mean(ep_returns) if ep_returns else 0.0
    print(f"\n{'SOLVED ✅' if final > 400 else 'NOT SOLVED ⚠️'} | final ep mean = {final:.1f}")
    return final > 400


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--vf-iters", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--step-size", type=float, default=0.01)
    p.add_argument("--max-kl", type=float, default=0.01)
    p.add_argument("--max-backtracks", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
