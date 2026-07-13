"""跑全部 Theory of Computation Mastery 模块的 _self_test():
deep_common 自检 + 7 个 tier 模块自检 + 总聚合校验。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # tier1 浅:自动机正则语言与文法下推自动机基础
    "tier1_shallow.dp_toc_automata_regular_languages",
    "tier1_shallow.dp_toc_grammars_pushdown_automata",
    # tier2 深:图灵机可计算性/复杂性类/归约NP完全性证明深水
    "tier2_deep.dp_toc_turing_machines_computability_deep",
    "tier2_deep.dp_toc_complexity_classes_deep",
    "tier2_deep.dp_toc_reductions_np_completeness_proofs_deep",
    # tier3 社招级别:算法可行性与形式化建模判断
    "tier3_social_hire.sc_toc_computational_feasibility_judgment",
    "tier3_social_hire.sc_toc_formal_modeling_judgment",
    # 总聚合校验
    "theory_of_computation_mastery",
]


def main() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"=== {passed}/{len(MODULES)} modules passed ===")
    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
