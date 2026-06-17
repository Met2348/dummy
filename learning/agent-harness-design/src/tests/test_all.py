"""Aggregate test for Topic 9 agent-harness-design."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = [
    "harness.util",
    "harness.tracing",
    "harness.errors",
    "harness.model",
    "harness.tools",
    "harness.context",
    "harness.permissions",
    "harness.memory",
    "harness.system_prompt",
    "harness.loop",
    "harness.subagents",
    "mini_harness",
    "capstone.run_task",
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
