"""Aggregate test for Topic 3 tool-use-mcp."""
import sys
import os

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

MODULES = [
    "common",
    "openai_tools",
    "mcp_protocol",
    "mcp_server",
    "mcp_client",
    "a2a_minimal",
    "computer_use_mock",
    "sandbox_mock",
    "streaming_tools",
    "tool_retry",
    "tool_injection_demo",
    "capstone_mcp_stack",
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
