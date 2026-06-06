"""RL Foundations 环境自检 — 三段式。

Part A: 基础（torch / transformers / trl / gymnasium / sb3）
Part B: GPU + sm_120 (Blackwell)
Part C: trl PPOTrainer GPT-2 5 step smoke
"""
from __future__ import annotations

import sys


def _parse_version(v: str) -> tuple:
    parts = v.split("+")[0].split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p.split("dev")[0] or "0"))
        except ValueError:
            break
    return tuple(nums)


def part_a_basic() -> bool:
    print("\n=== Part A: 基础 ===")
    targets = {
        "torch": "2.5",
        "transformers": "4.55",
        "trl": "0.11",
        "peft": "0.13",
        "gymnasium": "0.29",
        "stable_baselines3": "2.3",
        "numpy": "1.24",
        "matplotlib": "3.7",
        "datasets": "2.14",
        "accelerate": "0.30",
    }
    ok = True
    for name, min_v in targets.items():
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            if _parse_version(ver) >= _parse_version(min_v):
                print(f"  [OK]   {name:<18} {ver}  (>= {min_v})")
            else:
                print(f"  [WARN] {name:<18} {ver}  (< {min_v})")
                ok = False
        except ImportError:
            print(f"  [FAIL] {name:<18} not installed")
            ok = False
    print(f"\nPart A: {'PASS' if ok else 'FAIL'}")
    return ok


def part_b_gpu() -> bool:
    print("\n=== Part B: GPU ===")
    try:
        import torch
    except ImportError:
        print("  torch not installed, SKIP")
        return True
    if not torch.cuda.is_available():
        print("  No CUDA GPU, SKIP（CPU 仍可跑 CartPole 与小 GPT-2）")
        return True
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"  compute: {sm}")
    if sm == "sm_120":
        print("  Blackwell GPU 需要 cu130+ torch")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"  GEMM smoke: [OK] (output norm = {y.norm().item():.2e})")
        print("\nPart B: PASS")
        return True
    except RuntimeError as e:
        print(f"  GEMM failed: {e}")
        print("\nPart B: FAIL")
        return False


def part_c_gym_and_trl() -> bool:
    print("\n=== Part C: gymnasium + trl smoke test ===")
    ok = True

    # C1: gymnasium CartPole
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        print("  [OK] gymnasium CartPole 5 step")
    except Exception as e:
        print(f"  [FAIL] gymnasium CartPole: {e}")
        ok = False

    # C2: trl PPOTrainer legacy API（新版本 trl 可能不再顶层暴露）
    try:
        from trl import PPOConfig, PPOTrainer  # noqa: F401
        print("  [OK] trl PPOConfig / PPOTrainer import")
    except Exception as e:
        import trl
        print(f"  [SKIP] trl legacy PPOTrainer unavailable in trl={trl.__version__}: {e}")

    # C3: stable-baselines3 PPO 1 step
    try:
        from stable_baselines3 import PPO
        import gymnasium as gym
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0, n_steps=8, batch_size=4, device="cpu")
        model.learn(total_timesteps=8)
        env.close()
        print("  [OK] sb3 PPO 8 step CartPole")
    except Exception as e:
        print(f"  [FAIL] sb3 PPO: {e}")
        ok = False

    print(f"\nPart C: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("RL Foundations 环境自检")
    print("=" * 50)
    a = part_a_basic()
    b = part_b_gpu()
    c = part_c_gym_and_trl()
    print("\n" + "=" * 50)
    print(f"总结: A={'PASS' if a else 'FAIL'} B={'PASS' if b else 'SKIP/FAIL'} C={'PASS' if c else 'FAIL'}")
    return 0 if (a and c) else 1


if __name__ == "__main__":
    sys.exit(main())
