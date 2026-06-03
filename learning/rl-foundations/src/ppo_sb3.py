"""stable-baselines3 PPO on CartPole — 与 ppo_minimal.py 对照.

教学：sb3 包含 PPO 的所有"工业实现细节"（value clip / orthogonal init / lr
schedule / advantage norm）。一致性测试在 test_ppo_consistency.py。

运行:
    python learning/rl-foundations/src/ppo_sb3.py --total-steps 100_000
"""
from __future__ import annotations

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--K-epochs", type=int, default=10)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-eps", type=int, default=20)
    args = p.parse_args()

    env = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.n_envs)])
    env = VecMonitor(env)

    batch_size = args.n_steps * args.n_envs
    mb_size = batch_size // args.n_minibatches

    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=mb_size,
        n_epochs=args.K_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_steps, log_interval=10)

    eval_env = gym.make(args.env)
    mean_R, std_R = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_eps)
    print(f"\nsb3 PPO eval over {args.eval_eps} ep: mean = {mean_R:.1f} ± {std_R:.1f}")

    solved = mean_R > 450
    print(f"{'SOLVED ✅' if solved else 'NOT SOLVED ⚠️'}")
    return solved


if __name__ == "__main__":
    main()
