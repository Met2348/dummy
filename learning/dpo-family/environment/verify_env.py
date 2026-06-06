"""DPO Family 环境自检 — 检查 trl 4 Trainer (DPO/KTO/ORPO/CPO)."""
from __future__ import annotations

import sys


def main() -> int:
    print("DPO Family 环境自检")
    print("=" * 50)
    ok = True

    try:
        import trl
        trainers = ["DPOTrainer", "KTOTrainer", "ORPOTrainer", "CPOTrainer"]
        available = [name for name in trainers if hasattr(trl, name)]
        missing = [name for name in trainers if not hasattr(trl, name)]
        if available:
            print(f"  [OK] 可用 Trainer: {', '.join(available)}")
        if missing:
            print(f"  [SKIP] 当前 trl={trl.__version__} 未暴露: {', '.join(missing)}")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        ok = False

    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split="train[:10]")
        print(f"  [OK] HH 数据可用 ({len(ds)} 样本)")
    except Exception as e:
        print(f"  [WARN] HH: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
