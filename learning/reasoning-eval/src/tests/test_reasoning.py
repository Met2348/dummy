"""Topic 2 self-test runner."""
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = [
    "common",
    "gsm8k_runner",
    "math_runner",
    "aime_runner",
    "math_verify_demo",
    "gpqa_runner",
    "hle_mock",
    "zebra_logic",
    "tool_aug_math",
    "capstone_reasoning_compare",
]


def run_all() -> int:
    failed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            f = mod._self_test()
            status = "OK" if f == 0 else f"FAILED ({f})"
            print(f"[{name}] {status}")
            if f != 0:
                failed += 1
        except AssertionError as e:
            print(f"[{name}] ASSERTION FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"[{name}] EXCEPTION: {e}")
            failed += 1
    return failed


if __name__ == "__main__":
    fails = run_all()
    total = len(MODULES)
    print(f"\n=== {total - fails}/{total} modules passed ===")
    sys.exit(0 if fails == 0 else 1)
