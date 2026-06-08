"""Verify Topic 3 agent-code-eval environment."""
import sys


def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")

    print("\n=== Part B: Optional GPU ===")
    try:
        import torch
        print(f"[OK] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError:
        print("[INFO] torch not installed — Topic 3 runs without it")

    print("\n=== Part C: Optional libs ===")
    for pkg in ("playwright", "swebench"):
        try:
            __import__(pkg)
            print(f"[OK] {pkg} available")
        except ImportError:
            print(f"[INFO] {pkg} not installed - only needed for real bench")

    print("\n[SUCCESS] Topic 3 environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
