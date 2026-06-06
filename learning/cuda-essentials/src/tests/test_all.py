"""Aggregate test for Topic 2 cuda-essentials."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = ["common", "vector_add", "warp_primitives", "shared_memory",
           "coalescing", "reduce_kernel", "gemm_tiled", "capstone_softmax"]


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
