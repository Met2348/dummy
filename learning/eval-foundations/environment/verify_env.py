"""Verify Topic 1 eval-foundations environment.

Three-part check (Module 5 convention):
A. Python basics
B. Optional GPU
C. Optional libraries
"""
import sys


def part_a():
    print("=== Part A: Python basics ===")
    ok = True
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        ok = False
    else:
        print(f"[OK] Python {sys.version_info[:3]}")
    return ok


def part_b():
    print("\n=== Part B: GPU (optional) ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] No GPU — Topic 1 still runs (mock backend).")
    except ImportError:
        print("[INFO] torch not installed — Topic 1 still runs (mock backend).")
    return True


def part_c():
    print("\n=== Part C: Optional libraries ===")
    for pkg in ("lm_eval", "vllm", "transformers"):
        try:
            __import__(pkg)
            print(f"[OK] {pkg} available")
        except ImportError:
            print(f"[INFO] {pkg} not installed — only needed for real-model eval")
    return True


def main() -> int:
    a = part_a()
    b = part_b()
    c = part_c()
    print()
    if a and b and c:
        print("[SUCCESS] Topic 1 environment ready.")
        return 0
    print("[FAIL] See above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
