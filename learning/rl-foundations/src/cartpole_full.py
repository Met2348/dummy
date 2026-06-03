"""CartPole 完整 lab：5 算法一键切换，带 TensorBoard 日志.

支持的 --algo:
    reinforce / a2c / trpo / ppo / sb3_ppo

运行:
    python learning/rl-foundations/src/cartpole_full.py --algo ppo --total-steps 100_000
    # 然后 tensorboard --logdir runs/
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def run_reinforce(args, writer: SummaryWriter):
    import types
    mod = importlib.import_module("reinforce_minimal")
    ns = types.SimpleNamespace(
        env=args.env, n_episodes=args.total_steps // 200,
        hidden=64, lr=1e-3, gamma=0.99, seed=args.seed,
        window=10, log_interval=10, baseline=True, cpu=True,
    )
    mod.train(ns)


def run_a2c(args, writer: SummaryWriter):
    import types
    mod = importlib.import_module("a2c_minimal")
    ns = types.SimpleNamespace(
        env=args.env, n_envs=8, total_steps=args.total_steps,
        n_steps=5, hidden=64, lr=7e-4, gamma=0.99,
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
        seed=args.seed, log_interval=50, cpu=True,
    )
    mod.train(ns)


def run_trpo(args, writer: SummaryWriter):
    import types
    mod = importlib.import_module("trpo_minimal")
    ns = types.SimpleNamespace(
        env=args.env, total_steps=args.total_steps,
        n_envs=8, n_steps=64, hidden=64,
        vf_lr=1e-3, vf_iters=10, gamma=0.99, lam=0.95,
        step_size=0.01, max_kl=0.01, max_backtracks=10,
        seed=args.seed, log_interval=10,
    )
    mod.train(ns)


def run_ppo(args, writer: SummaryWriter):
    import types
    mod = importlib.import_module("ppo_minimal")
    ns = types.SimpleNamespace(
        env=args.env, total_steps=args.total_steps,
        n_envs=8, n_steps=128, hidden=64,
        lr=3e-4, gamma=0.99, lam=0.95, eps=0.2,
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
        K_epochs=10, n_minibatches=4,
        seed=args.seed, log_interval=10, cpu=True,
    )
    mod.train(ns)


def run_sb3_ppo(args, writer: SummaryWriter):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    env = DummyVecEnv([lambda: gym.make(args.env) for _ in range(8)])
    env = VecMonitor(env)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=128, batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                vf_coef=0.5, ent_coef=0.01, seed=args.seed, verbose=1,
                tensorboard_log=str(args.tb_log_dir))
    model.learn(total_timesteps=args.total_steps)


ALGOS = {
    "reinforce": run_reinforce,
    "a2c": run_a2c,
    "trpo": run_trpo,
    "ppo": run_ppo,
    "sb3_ppo": run_sb3_ppo,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(ALGOS), required=True)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tb-log-dir", type=Path, default=Path("runs"))
    args = p.parse_args()
    args.tb_log_dir.mkdir(parents=True, exist_ok=True)

    log_name = f"{args.algo}_seed{args.seed}_{args.total_steps}"
    writer = SummaryWriter(args.tb_log_dir / log_name)
    print(f"==> algo={args.algo}  total_steps={args.total_steps}")
    print(f"==> tensorboard --logdir {args.tb_log_dir}\n")

    ALGOS[args.algo](args, writer)
    writer.close()


if __name__ == "__main__":
    main()
