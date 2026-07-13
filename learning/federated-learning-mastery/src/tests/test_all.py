"""跑全部 Federated Learning Mastery 模块的 _self_test():
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
    # tier1 浅:核心概念与系统部署
    "tier1_shallow.dp_fl_basics_fedavg",
    "tier1_shallow.dp_fl_system_deployment",
    # tier2 深:机制深水
    "tier2_deep.dp_fl_noniid_personalization",
    "tier2_deep.dp_fl_communication_efficiency",
    "tier2_deep.dp_fl_privacy_security",
    # tier3 社招级别:资深战略判断
    "tier3_social_hire.sc_fl_production_incentive_judgment",
    "tier3_social_hire.sc_fl_llm_frontier_judgment",
    # 总聚合校验
    "federated_learning_mastery",
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
