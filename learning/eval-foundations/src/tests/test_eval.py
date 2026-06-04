"""Topic 1 test runner — no pytest, run with: python -c ... from src dir.

Imports every src module's _self_test and aggregates.
"""
import importlib
import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODULES = [
    "common",
    "mmlu_runner",
    "mmlu_pro_runner",
    "helm_local",
    "bbh_runner",
    "truthfulqa_runner",
    "commonsense_runner",
    "lm_eval_adapter",
    "contamination_check",
    "eval_pipeline",
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
