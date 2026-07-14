"""跑全部 Algorithms Theory Mastery 模块的 _self_test():
deep_common 自检 + 6 个 tier 模块自检 + 总聚合校验。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # tier1 浅:渐进分析主定理与分治贪心设计基础
    "tier1_shallow.dp_alg_asymptotic_analysis_master_theorem",
    "tier1_shallow.dp_alg_divide_conquer_greedy_basics",
    # tier2 深:动态规划正确性证明/图算法正确性/网络流匹配深水
    "tier2_deep.dp_alg_dynamic_programming_correctness_deep",
    "tier2_deep.dp_alg_graph_algorithms_correctness_deep",
    "tier2_deep.dp_alg_network_flow_matching_deep",
    # tier3 社招级别:算法范式选择与复杂度工程权衡判断
    "tier3_social_hire.sc_alg_paradigm_selection_judgment",
    "tier3_social_hire.sc_alg_complexity_engineering_tradeoff_judgment",
    # 总聚合校验
    "algorithms_theory_mastery",
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
