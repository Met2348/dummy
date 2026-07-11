"""汇总自测：import 全部 12 个模块并调用各自 _self_test()。

跑法：$env:PYTHONIOENCODING="utf-8"; python learning/interview-prep/src/tests/test_all.py
预期：=== 12/12 modules passed ===
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

from mlcoding import (attention, bpe, kv_cache, lora, norm, rope,   # noqa: E402
                      sampling, training_loop, transformer_block)
from leetcode import patterns, tracker                              # noqa: E402
from mlqa import qbank                                              # noqa: E402

MODULES = [
    ("mlcoding.attention", attention),
    ("mlcoding.norm", norm),
    ("mlcoding.rope", rope),
    ("mlcoding.sampling", sampling),
    ("mlcoding.lora", lora),
    ("mlcoding.bpe", bpe),
    ("mlcoding.transformer_block", transformer_block),
    ("mlcoding.kv_cache", kv_cache),
    ("mlcoding.training_loop", training_loop),
    ("leetcode.patterns", patterns),
    ("leetcode.tracker", tracker),
    ("mlqa.qbank", qbank),
]


def main() -> int:
    passed = 0
    for name, mod in MODULES:
        try:
            mod._self_test()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
    print(f"\n=== {passed}/{len(MODULES)} modules passed ===")
    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
