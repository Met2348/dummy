"""Verify Topic 7 agent-graduation env."""
import sys

def main() -> int:
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]}"); return 1
    print(f"[OK] Python {sys.version_info[:3]}")
    print("[OK] stdlib only - Module 7 graduation plus 39-topic portfolio")
    return 0

if __name__ == "__main__":
    sys.exit(main())
