"""Aggregate test for Topic 4 cluster-networking."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = ["common", "allreduce_algos", "fabric_topology",
           "nccl_collectives", "sharp_inline", "capstone_cluster_sim"]


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
