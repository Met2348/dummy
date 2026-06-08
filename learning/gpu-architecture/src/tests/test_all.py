"""Aggregate test for Topic 1 gpu-architecture."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = ["common", "roofline", "roofline_original_minimal",
           "memory_hierarchy", "tensor_core", "nvlink_topology",
           "sm_occupancy", "capstone_roofline_zoo"]


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
