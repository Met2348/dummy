"""RL SOTA 2026 环境自检 — 继承专题 5."""
import sys
def main():
    for name in ["verl", "vllm", "ray"]:
        try:
            mod = __import__(name)
            print(f"[OK] {name} {getattr(mod, '__version__', '?')}")
        except ImportError as e:
            print(f"[SKIP] {name}: optional WSL2/Linux production stack unavailable ({e})")
    return 0
if __name__ == "__main__":
    sys.exit(main())
