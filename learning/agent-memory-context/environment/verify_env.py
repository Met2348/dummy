"""Verify Topic 5 agent-memory-context env."""
import sys

def main() -> int:
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]}"); return 1
    print(f"[OK] Python {sys.version_info[:3]}")
    print("[OK] stdlib only")
    return 0

if __name__ == "__main__":
    sys.exit(main())
