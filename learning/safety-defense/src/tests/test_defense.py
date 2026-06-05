"""Topic 6 self-test runner."""
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = [
    "common",
    "llama_guard_mock",
    "wildguard_mock",
    "nemo_guardrails_mock",
    "constitutional_classifier",
    "pii_redaction",
    "prompt_injection_defense",
    "safety_eval_runner",
    "defense_pipeline",
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
