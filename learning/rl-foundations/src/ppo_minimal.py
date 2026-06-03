"""手写 PPO-Clip on CartPole.

教学目标：
    1. 一目了然的 PPO clip 实现（30 行核心）
    2. 完整 4 loss: L_clip + c_v · L_vf - c_e · entropy + (optional KL)
    3. K epoch × M minibatch 重用同一 rollout
    4. 含 advantage normalization、grad clip 等关键 trick

运行：
    python learning/rl-foundations/src/ppo_minimal.py
    python learning/rl-foundations/src/ppo_minimal.py --total-steps 200_000

预期：
    - 100k step 后 ep mean > 480 (CartPole-v1 max=500)
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

from common import compute_gae, categorical_entropy, set_seed


class PolicyValue(nn.Module):
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

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def make_vec_env(env_id: str, n_envs: int, seed: int):
    def make(i):
        def _f():
            env = gym.make(env_id)
            env.reset(seed=seed + i)
            return env
        return _f
    return gym.vector.SyncVectorEnv([make(i) for i in range(n_envs)])


def train(args):
    set_seed(args.seed)
    envs = make_vec_env(args.env, args.n_envs, args.seed)
    state_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model = PolicyValue(state_dim, args.hidden, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs, _ = envs.reset(seed=args.seed)
    ep_returns: deque = deque(maxlen=100)
    running = np.zeros(args.n_envs)

    total_iters = args.total_steps // (args.n_steps * args.n_envs)
    batch_size = args.n_steps * args.n_envs
    mb_size = batch_size // args.n_minibatches
    assert batch_size % args.n_minibatches == 0

    for it in range(1, total_iters + 1):
        # ===== rollout =====
        obs_buf = torch.zeros(args.n_steps, args.n_envs, state_dim, device=device)
        act_buf = torch.zeros(args.n_steps, args.n_envs, dtype=torch.long, device=device)
        logp_buf = torch.zeros(args.n_steps, args.n_envs, device=device)
        val_buf = torch.zeros(args.n_steps, args.n_envs, device=device)
        rew_buf = torch.zeros(args.n_steps, args.n_envs, device=device)
        done_buf = torch.zeros(args.n_steps, args.n_envs, device=device)

        for t in range(args.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, V = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, r, term, trunc, _ = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)

            obs_buf[t] = obs_t
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = V
            rew_buf[t] = torch.as_tensor(r, dtype=torch.float32, device=device)
            done_buf[t] = torch.as_tensor(done.astype(np.float32), device=device)

            running += r
            for i, d in enumerate(done):
                if d:
                    ep_returns.append(float(running[i]))
                    running[i] = 0.0
            obs = next_obs

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, last_V = model(obs_t)

        # ===== compute GAE =====
        adv, ret = compute_gae(rew_buf, val_buf, done_buf,
                               last_value=last_V.mean().item(),
                               gamma=args.gamma, lam=args.lam)
        # advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten
        obs_flat = obs_buf.reshape(-1, state_dim)
        act_flat = act_buf.reshape(-1)
        logp_old_flat = logp_buf.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        # ===== K epoch × M minibatch =====
        mean_kl = 0.0
        n_updates = 0
        for epoch in range(args.K_epochs):
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            for start in range(0, batch_size, mb_size):
                idx = indices[start:start + mb_size]
                obs_mb = obs_flat[idx]
                act_mb = act_flat[idx]
                logp_old_mb = logp_old_flat[idx]
                adv_mb = adv_flat[idx]
                ret_mb = ret_flat[idx]

                logits, V_new = model(obs_mb)
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                ratio = (logp_new - logp_old_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1 - args.eps, 1 + args.eps) * adv_mb
                L_clip = -torch.min(surr1, surr2).mean()

                L_vf = F.mse_loss(V_new, ret_mb)
                loss = L_clip + args.vf_coef * L_vf - args.ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    log_r = logp_new - logp_old_mb
                    mean_kl += (log_r.exp() - 1 - log_r).mean().item()  # Schulman approx
                    n_updates += 1

        mean_kl /= max(n_updates, 1)

        if it % args.log_interval == 0:
            mean_R = np.mean(ep_returns) if ep_returns else 0.0
            print(f"Iter {it:4d} | env steps {it * batch_size:7d} | "
                  f"ep mean {mean_R:6.1f} | L_clip {L_clip.item():7.4f} | "
                  f"L_vf {L_vf.item():7.4f} | KL≈{mean_kl:6.4f}")

    envs.close()
    final = np.mean(ep_returns) if ep_returns else 0.0
    solved = final > 450
    print(f"\n{'SOLVED ✅' if solved else 'NOT SOLVED ⚠️'} | final ep mean = {final:.1f}")
    return solved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--K-epochs", type=int, default=10)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
