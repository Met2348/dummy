"""Multimodal Agent 环境自检."""
import sys
def main():
    ok = True
    try:
        import qwen_vl_utils, playwright  # noqa
        print("[OK] qwen-vl-utils + playwright")
    except ImportError as e:
        print(f"[WARN] {e}")
        ok = False
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
