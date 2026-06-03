"""RL SOTA 2026 环境自检 — 继承专题 5."""
import sys
def main():
    ok = True
    try:
        import verl, vllm, ray  # noqa
        print("[OK] verl + vllm + ray")
    except ImportError as e:
        print(f"[FAIL] {e}")
        ok = False
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
