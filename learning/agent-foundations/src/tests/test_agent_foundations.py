"""Aggregate test runner for Topic 1 agent-foundations."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "tools"))

MODULES = [
    ("common", "common"),
    ("calculator", "tools.calculator"),
    ("search_mock", "tools.search_mock"),
    ("file_op", "tools.file_op"),
    ("web_mock", "tools.web_mock"),
    ("react_loop", "react_loop"),
    ("reflexion_demo", "reflexion_demo"),
    ("plan_execute", "plan_execute"),
    ("router_pattern", "router_pattern"),
    ("state_machine", "state_machine"),
    ("tracing", "tracing"),
    ("capstone_react", "capstone_react"),
]


def main() -> int:
    failed = []
    for short, modname in MODULES:
        try:
            mod = __import__(modname, fromlist=["_self_test"])
            mod._self_test()
        except Exception as e:  # noqa: BLE001
            failed.append((short, str(e)))
            print(f"[FAIL] {short}: {e}")

    print()
    print(f"=== {len(MODULES) - len(failed)}/{len(MODULES)} modules passed ===")
    if failed:
        for short, msg in failed:
            print(f"  FAIL: {short} → {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
