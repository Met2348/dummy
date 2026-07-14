"""Computer Architecture Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师"仔细思考和科班CS本科相比还差哪些能力"要求补齐的六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)第四个。这里的"计算机体系结构"是**通用CPU
体系结构**,和仓库里已有的 `gpu-architecture`/`cuda-essentials`/`kernel-engineering`(GPU
专用架构)是完全不同的细分领域。`explain` 字段承载"讲仔细"的系统性教学讲解，组织方式沿用
"难度分层"轴：

- tier1(浅)：指令集与数据通路(RISC vs CISC/寻址模式/单周期多周期/IF-ID-EX-MEM-WB)、存储
  层次基础(局部性原理/cache基础/写策略)
- tier2(深)：流水线与冒险深水(结构数据控制冒险/forwarding/分支预测)、乱序执行与高级微架构
  深水(Tomasulo/寄存器重命名/ROB/推测执行)、多核与缓存一致性深水(MESI/一致性协议/内存
  一致性模型/false sharing)
- tier3(社招级别)：性能瓶颈定位判断(CPU-bound/memory-bound/cache miss等)、硬件感知的
  软件优化判断(数据局部性/分支预测友好/false sharing规避)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_arch_isa_datapath_basics import BANK as _t1_isa  # noqa: E402
from tier1_shallow.dp_arch_memory_hierarchy_basics import BANK as _t1_mem  # noqa: E402
from tier2_deep.dp_arch_pipelining_hazards_deep import BANK as _t2_pipe  # noqa: E402
from tier2_deep.dp_arch_out_of_order_execution_deep import BANK as _t2_ooo  # noqa: E402
from tier2_deep.dp_arch_multicore_cache_coherence_deep import BANK as _t2_mc  # noqa: E402
from tier3_social_hire.sc_arch_performance_bottleneck_judgment import BANK as _t3_perf  # noqa: E402
from tier3_social_hire.sc_arch_hardware_aware_optimization_judgment import BANK as _t3_opt  # noqa: E402

ALL_DP = list(_t1_isa) + list(_t1_mem) + list(_t2_pipe) + list(_t2_ooo) + list(_t2_mc)
ALL_SP = list(_t3_perf) + list(_t3_opt)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:指令集数据通路与存储层次基础", "kind": "dp",
     "modules": ({"name": "arch_isa_datapath_basics", "cat": "计算机体系结构基础一:指令集与数据通路", "bank": _t1_isa},
                 {"name": "arch_memory_hierarchy_basics", "cat": "计算机体系结构基础二:存储层次基础", "bank": _t1_mem})},
    {"n": 2, "name": "deep", "label": "深:流水线冒险/乱序执行/多核缓存一致性深水", "kind": "dp",
     "modules": ({"name": "arch_pipelining_hazards_deep", "cat": "计算机体系结构深水一:流水线与冒险深水", "bank": _t2_pipe},
                 {"name": "arch_out_of_order_execution_deep", "cat": "计算机体系结构深水二:乱序执行与高级微架构深水", "bank": _t2_ooo},
                 {"name": "arch_multicore_cache_coherence_deep", "cat": "计算机体系结构深水三:多核与缓存一致性深水", "bank": _t2_mc})},
    {"n": 3, "name": "social_hire", "label": "社招级别:性能瓶颈定位与硬件感知优化判断", "kind": "sp",
     "modules": ({"name": "arch_performance_bottleneck_judgment", "cat": "计算机体系结构社招级别一:性能瓶颈定位判断", "bank": _t3_perf},
                 {"name": "arch_hardware_aware_optimization_judgment", "cat": "计算机体系结构社招级别二:硬件感知的软件优化判断", "bank": _t3_opt})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "computer-architecture-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "computer-architecture-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-arch-") for i in dp_ids), "存在不以dp-arch-开头的DeepPoint id"
    assert all(i.startswith("sc-arch-") for i in sp_ids), "存在不以sc-arch-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "computer-architecture-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "computer-architecture-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] computer_architecture_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
