"""Verify interview-prep track environment（torch 是本 track 的破例依赖）。"""
import os
import sys


def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")

    print("\n=== Part B: 依赖（torch 唯一破例）===")
    try:
        import numpy
        print(f"[OK] numpy {numpy.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] numpy: {e}")
        return 1
    try:
        import torch
        print(f"[OK] torch {torch.__version__} (CPU 即可，无需 GPU)")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] torch 未安装: pip install torch  ({e})")
        return 1

    print("\n=== Part C: 各层可导入 ===")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src)
    try:
        from mlcoding import attention  # noqa: F401
        from leetcode import patterns    # noqa: F401
        from mlqa import qbank           # noqa: F401
        print("[OK] mlcoding / leetcode / mlqa 三层 import 干净")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import error: {e}")
        return 1

    print("\n[SUCCESS] interview-prep 环境就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
