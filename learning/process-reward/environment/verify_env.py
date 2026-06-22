"""Process Reward 环境自检."""
from __future__ import annotations

import sys


def main() -> int:
    print("Process Reward 环境自检")
    ok = True
    try:
        import math_verify, sympy, networkx  # noqa
        print(f"  [OK] math-verify / sympy / networkx")
    except ImportError as e:
        print(f"  [WARN] {e}")
        ok = False
    try:
        from datasets import load_dataset
        # 用命名空间 id（datasets 5.x 不再支持裸别名 "gsm8k"，会 redirect/报错）；
        # 离线/缺网时下方 except 只 WARN，不阻塞 demo。
        ds = load_dataset("openai/gsm8k", "main", split="train[:5]")
        print(f"  [OK] GSM8K loaded ({len(ds)})")
    except Exception as e:
        print(f"  [WARN] GSM8K: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
