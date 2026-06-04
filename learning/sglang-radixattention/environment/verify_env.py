"""3-part check: torch+cuda / sglang / helpers."""
import sys


def part_a():
    import torch
    print(f"[A] torch {torch.__version__}, cuda={torch.cuda.is_available()}")


def part_b():
    try:
        import sglang
        print(f"[B] sglang {sglang.__version__}")
    except ImportError:
        print("[B] sglang NOT installed (minimal-only flow ok)")
    for lib in ("outlines", "xgrammar"):
        try:
            mod = __import__(lib)
            print(f"[B] {lib} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"[B] {lib} NOT installed")


def part_c():
    import transformers
    print(f"[C] transformers {transformers.__version__}")


if __name__ == "__main__":
    for p in (part_a, part_b, part_c):
        try:
            p()
        except Exception as e:
            print(f"FAIL {p.__name__}: {e}")
            sys.exit(1)
    print("\nverify_env OK")
