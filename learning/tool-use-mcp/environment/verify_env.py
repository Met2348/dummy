"""Verify Topic 3 tool-use-mcp environment."""
import sys

def main() -> int:
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python {sys.version_info[:2]}"); return 1
    print(f"[OK] Python {sys.version_info[:3]}")
    print("[OK] stdlib only (MCP/A2A in-process mock)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
