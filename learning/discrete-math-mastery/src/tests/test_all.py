"""跑全部 Discrete Math Mastery 模块的 _self_test():
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
    # tier1 浅:逻辑集合函数与关系偏序计数基础
    "tier1_shallow.dp_dm_logic_sets_functions",
    "tier1_shallow.dp_dm_relations_orders_counting_basics",
    # tier2 深:组合数学/图论/证明方法深水
    "tier2_deep.dp_dm_combinatorics_deep",
    "tier2_deep.dp_dm_graph_theory_deep",
    "tier2_deep.dp_dm_proof_techniques_deep",
    # tier3 社招级别:算法建模与证明策略判断
    "tier3_social_hire.sc_dm_algorithm_modeling_judgment",
    "tier3_social_hire.sc_dm_proof_strategy_pitfalls_judgment",
    # 总聚合校验
    "discrete_math_mastery",
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
