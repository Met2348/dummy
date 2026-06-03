"""stable-baselines3 的 A2C 对照实现。

教学目标：
    1. 跑通 sb3 标准 A2C，作为手写 a2c_minimal.py 的参考
    2. 对比同 budget 下两者的收敛曲线（差异通常是 sb3 略好，因有更完善的 advantage norm + lr schedule）

运行：
    python learning/rl-foundations/src/a2c_sb3.py --total-steps 100_000

预期：
    - 50k step 后 eval mean reward > 400
"""
from __future__ import annotations

import argparse

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-eps", type=int, default=20)
    args = p.parse_args()

    train_env = DummyVecEnv([lambda: gym.make(args.env) for _ in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    model = A2C(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=args.gamma,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_steps, log_interval=50)

    # 评估
    eval_env = gym.make(args.env)
    mean_R, std_R = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_eps)
    print(f"\nsb3 A2C eval over {args.eval_eps} ep: mean = {mean_R:.1f} ± {std_R:.1f}")

    solved = mean_R > 400
    print(f"{'SOLVED ✅' if solved else 'NOT SOLVED ⚠️'}")
    return solved


if __name__ == "__main__":
    main()
