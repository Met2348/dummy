"""Verify Topic 5 red-team-jailbreak environment."""
import sys


def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")

    print("\n=== Part B: Optional libs ===")
    for pkg in ("torch", "transformers"):
        try:
            __import__(pkg)
            print(f"[OK] {pkg} available")
        except ImportError:
            print(f"[INFO] {pkg} not installed — only needed for real-model attack")

    print("\n[SUCCESS] Topic 5 environment ready (educational mocks only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
