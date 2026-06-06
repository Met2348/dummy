"""Aggregate test for Topic 7 agent-graduation."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = [
    "dra.common",
    "dra.tools.dra_tools",
    "dra.planner",
    "dra.retriever",
    "dra.writer",
    "dra.verifier",
    "dra.orchestrator",
    "eval.tau_bench_mock",
    "eval.dra_eval",
    "portfolio_v2",
]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            mod = __import__(name, fromlist=["_self_test"])
            mod._self_test()
        except Exception as e:  # noqa: BLE001
            failed.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
    print()
    print(f"=== {len(MODULES) - len(failed)}/{len(MODULES)} modules passed ===")
    if failed:
        for n, m in failed:
            print(f"  FAIL: {n} -> {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
