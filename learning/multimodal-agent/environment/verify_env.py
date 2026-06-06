"""Multimodal Agent 环境自检."""
import sys
def main():
    try:
        import qwen_vl_utils  # noqa
        print("[OK] qwen-vl-utils")
    except ImportError as e:
        print(f"[SKIP] qwen-vl-utils optional for real VLM runs: {e}")
    try:
        import playwright  # noqa
        print("[OK] playwright")
    except ImportError as e:
        print(f"[SKIP] playwright optional for real browser tasks: {e}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
