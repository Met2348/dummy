"""Verify Topic 9 agent-harness-design environment."""
import sys
import os


def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")

    print("\n=== Part B: Stdlib only ===")
    print("[OK] No external deps needed (MockModel via deterministic brain).")

    print("\n=== Part C: harness importable ===")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src)
    try:
        from harness.loop import run_loop  # noqa: F401
        from mini_harness import Harness  # noqa: F401
        print("[OK] harness package + mini_harness import cleanly.")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import error: {e}")
        return 1

    print("\n[SUCCESS] Topic 9 environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
