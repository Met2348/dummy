"""Verify Topic 2 rag-essential environment."""
import sys

def main() -> int:
    print("=== Part A: Python ===")
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]} < 3.9")
        return 1
    print(f"[OK] Python {sys.version_info[:3]}")
    print("\n=== Part B: Stdlib only ===")
    print("[OK] No external deps (mock embedding/reranker/LLM).")
    print("\n[SUCCESS] Topic 2 environment ready.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
