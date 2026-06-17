"""Verify Topic 8 agent-design-patterns environment."""
import sys
import os


def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")

    print("\n=== Part B: Stdlib only ===")
    print("[OK] No external deps needed (mock LLM via injected responder).")

    print("\n=== Part C: src importable ===")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src)
    try:
        import common  # noqa: F401
        from patterns import prompt_chaining  # noqa: F401
        print("[OK] common + patterns import cleanly.")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import error: {e}")
        return 1

    print("\n[SUCCESS] Topic 8 environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
