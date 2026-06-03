"""RLHF Classic 环境自检 — 继承专题 1 cu130 nightly."""
from __future__ import annotations

import sys


def main() -> int:
    print("RLHF Classic 环境自检")
    print("=" * 50)
    ok = True

    print("\n=== Part A: 基础（继承 rl-foundations）===")
    try:
        import torch, transformers, trl, peft, datasets, accelerate
        print(f"  [OK] torch={torch.__version__}, trl={trl.__version__}, "
              f"peft={peft.__version__}")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        ok = False

    print("\n=== Part B: trl 4 Trainer (SFT/Reward/PPO/DPO) import ===")
    try:
        from trl import SFTTrainer, RewardTrainer, PPOTrainer, DPOTrainer  # noqa: F401
        print("  [OK] 4 trainer 可用")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        ok = False

    print("\n=== Part C: Anthropic-HH 1k 加载 ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split="train[:50]")
        print(f"  [OK] 加载 {len(ds)} 条样本")
        print(f"  example keys: {list(ds[0].keys())}")
    except Exception as e:
        print(f"  [WARN] HH 加载失败（可能需 huggingface 登录）: {e}")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
