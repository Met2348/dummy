"""OS Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第五个，也是最后一个，组织
方式沿用前四个专题已验证的"讲解+追问链/场景判断"混合格式：

- tier1(浅)：进程与线程基础(进程状态转换/PCB/上下文切换/用户级与内核级线程/IPC/孤儿僵尸进程)、
  内存管理基础(虚拟内存/分页分段/多级页表/TLB/缺页中断/进程内存布局)
- tier2(深)：调度与同步深水(CPU调度算法/Linux CFS/锁-信号量-条件变量/死锁四条件/优先级反转)、
  文件系统与IO深水(inode/硬链接软链接/select-poll-epoll演进/零拷贝/Page Cache/日志文件系统)、
  虚拟化与容器深水(VM vs容器/Hypervisor类型/Linux namespace/cgroups/容器镜像分层/内核态用户态
  切换)
- tier3(社招级别)：生产性能问题定位判断(CPU飙升/内存泄漏/IO瓶颈/容器OOMKilled)、系统设计中的
  OS层判断(高并发IO模型选型/内核参数调优边界/CPU亲和性/系统调用开销优化)——无标准答案的资深判断

至此，老手2026-07-13提出的"软件工程/设计模式/数据库/网络/OS各做100+知识点系统化学习且讲仔细"
的五专题队列全部完成。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_os_process_thread_basics import BANK as _t1_proc  # noqa: E402
from tier1_shallow.dp_os_memory_management_basics import BANK as _t1_mem  # noqa: E402
from tier2_deep.dp_os_scheduling_synchronization_deep import BANK as _t2_sched  # noqa: E402
from tier2_deep.dp_os_filesystem_io_deep import BANK as _t2_fs  # noqa: E402
from tier2_deep.dp_os_virtualization_container_deep import BANK as _t2_vm  # noqa: E402
from tier3_social_hire.sc_os_performance_troubleshooting_judgment import BANK as _t3_perf  # noqa: E402
from tier3_social_hire.sc_os_system_design_os_layer_judgment import BANK as _t3_design  # noqa: E402

ALL_DP = list(_t1_proc) + list(_t1_mem) + list(_t2_sched) + list(_t2_fs) + list(_t2_vm)
ALL_SP = list(_t3_perf) + list(_t3_design)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:进程线程与内存管理基础", "kind": "dp",
     "modules": ({"name": "os_process_thread_basics", "cat": "操作系统基础一:进程与线程基础", "bank": _t1_proc},
                 {"name": "os_memory_management_basics", "cat": "操作系统基础二:内存管理基础", "bank": _t1_mem})},
    {"n": 2, "name": "deep", "label": "深:调度同步/文件系统IO/虚拟化容器深水", "kind": "dp",
     "modules": ({"name": "os_scheduling_synchronization_deep", "cat": "操作系统深水一:调度与同步深水", "bank": _t2_sched},
                 {"name": "os_filesystem_io_deep", "cat": "操作系统深水二:文件系统与IO深水", "bank": _t2_fs},
                 {"name": "os_virtualization_container_deep", "cat": "操作系统深水三:虚拟化与容器深水", "bank": _t2_vm})},
    {"n": 3, "name": "social_hire", "label": "社招级别:性能定位与系统设计OS层判断", "kind": "sp",
     "modules": ({"name": "os_performance_troubleshooting_judgment", "cat": "操作系统社招级别一:生产性能问题定位判断", "bank": _t3_perf},
                 {"name": "os_system_design_os_layer_judgment", "cat": "操作系统社招级别二:系统设计中的OS层判断", "bank": _t3_design})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "os-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "os-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-os-") for i in dp_ids), "存在不以dp-os-开头的DeepPoint id"
    assert all(i.startswith("sc-os-") for i in sp_ids), "存在不以sc-os-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "os-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "os-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] os_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
