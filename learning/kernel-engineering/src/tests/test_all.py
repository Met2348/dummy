"""Aggregate test for Topic 3 kernel-engineering."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = ["triton_style", "cutlass_layout", "flash_attention", "fused_mlp",
           "rmsnorm_kernel", "capstone_attn_speedup"]


def main() -> int:
    fails = []
    for n in MODULES:
        try:
            m = __import__(n, fromlist=["_self_test"])
            m._self_test()
        except Exception as e:
            fails.append((n, e))
            print(f"[FAIL] {n}: {e}")
    print(f"\n=== {len(MODULES) - len(fails)}/{len(MODULES)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
