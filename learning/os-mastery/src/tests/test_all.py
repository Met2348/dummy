"""跑全部 OS Mastery 模块的 _self_test():
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
    # tier1 浅:进程线程基础与内存管理基础
    "tier1_shallow.dp_os_process_thread_basics",
    "tier1_shallow.dp_os_memory_management_basics",
    # tier2 深:调度同步/文件系统IO/虚拟化容器深水
    "tier2_deep.dp_os_scheduling_synchronization_deep",
    "tier2_deep.dp_os_filesystem_io_deep",
    "tier2_deep.dp_os_virtualization_container_deep",
    # tier3 社招级别:性能定位与系统设计OS层判断
    "tier3_social_hire.sc_os_performance_troubleshooting_judgment",
    "tier3_social_hire.sc_os_system_design_os_layer_judgment",
    # 总聚合校验
    "os_mastery",
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
