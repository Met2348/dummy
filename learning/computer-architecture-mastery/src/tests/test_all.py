"""跑全部 Computer Architecture Mastery 模块的 _self_test():
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
    # tier1 浅:指令集数据通路与存储层次基础
    "tier1_shallow.dp_arch_isa_datapath_basics",
    "tier1_shallow.dp_arch_memory_hierarchy_basics",
    # tier2 深:流水线冒险/乱序执行/多核缓存一致性深水
    "tier2_deep.dp_arch_pipelining_hazards_deep",
    "tier2_deep.dp_arch_out_of_order_execution_deep",
    "tier2_deep.dp_arch_multicore_cache_coherence_deep",
    # tier3 社招级别:性能瓶颈定位与硬件感知优化判断
    "tier3_social_hire.sc_arch_performance_bottleneck_judgment",
    "tier3_social_hire.sc_arch_hardware_aware_optimization_judgment",
    # 总聚合校验
    "computer_architecture_mastery",
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
