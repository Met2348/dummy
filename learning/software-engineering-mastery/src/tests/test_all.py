"""跑全部 Software Engineering Mastery 模块的 _self_test():
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
    # tier1 浅:SDLC/敏捷基础与版本控制协作流程基础
    "tier1_shallow.dp_se_sdlc_agile_requirements",
    "tier1_shallow.dp_se_version_control_cicd_basics",
    # tier2 深:测试/架构/代码质量方法论深水
    "tier2_deep.dp_se_testing_methodology",
    "tier2_deep.dp_se_architecture_modularity",
    "tier2_deep.dp_se_code_quality_refactoring",
    # tier3 社招级别:工程文化与交付判断
    "tier3_social_hire.sc_se_tech_debt_engineering_culture_judgment",
    "tier3_social_hire.sc_se_delivery_release_judgment",
    # 总聚合校验
    "software_engineering_mastery",
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
