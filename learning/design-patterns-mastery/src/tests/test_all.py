"""跑全部 Design Patterns Mastery 模块的 _self_test():
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
    # tier1 浅:创建型与结构型模式基础
    "tier1_shallow.dp_pat_creational_patterns",
    "tier1_shallow.dp_pat_structural_patterns",
    # tier2 深:行为型模式深水与现代模式演化
    "tier2_deep.dp_pat_behavioral_patterns_one",
    "tier2_deep.dp_pat_behavioral_patterns_two",
    "tier2_deep.dp_pat_modern_patterns_antipatterns",
    # tier3 社招级别:架构选型与过度设计判断
    "tier3_social_hire.sc_pat_architecture_selection_judgment",
    "tier3_social_hire.sc_pat_overengineering_boundary_judgment",
    # 总聚合校验
    "design_patterns_mastery",
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
