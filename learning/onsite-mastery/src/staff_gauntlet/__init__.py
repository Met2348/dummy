"""Staff Gauntlet:按面试关卡阶段(而非知识主题)组织的资深/高阶通关训练营(11关,合计219点)。

与 ai_deep/backend_deep/frontier_deep 三个"知识主题包"的组织轴不同,这里复刻的是2026年
frontier AI lab(OpenAI/Anthropic/DeepMind)真实面试loop的阶段结构:screen动机筛选→
research coding(论文转代码)→paper critique(论文批判)→ML基础设施系统设计→agentic系统
设计→values round(价值观关)→资深叙事(方向主导权)→跨团队模糊决策→国内资深社招视角→
定级谈判判断→大规模集群HPC基础设施深水。GATES 按这个真实顺序排列,练法是按顺序过关,不是
随机抽题(见 lectures/00-how-to-defend.md)。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import categories  # noqa: E402

from .gate1_screen_motivation import BANK as _gate1
from .gate2_research_coding_paper2code import BANK as _gate2
from .gate3_paper_critique_fluency import BANK as _gate3
from .gate4_ml_infra_system_design import BANK as _gate4
from .gate5_agentic_system_design import BANK as _gate5
from .gate6_values_safety_stance import BANK as _gate6
from .gate7_staff_ownership_narrative import BANK as _gate7
from .gate8_cross_functional_ambiguity import BANK as _gate8
from .gate9_domestic_senior_socialhire import BANK as _gate9
from .gate10_leveling_negotiation import BANK as _gate10
from .gate11_hpc_cluster_infra import BANK as _gate11

ALL_DP = (
    list(_gate1) + list(_gate2) + list(_gate3) + list(_gate7) + list(_gate9) + list(_gate11)
)
ALL_SP = (
    list(_gate4) + list(_gate5) + list(_gate6) + list(_gate8) + list(_gate10)
)

# GATES:按真实面试loop顺序排列的关卡元数据,体现"层层通关"——练习时应按 n 从小到大顺序过关,
# 而不是像 ai_deep/backend_deep/frontier_deep 那样随机抽题。
GATES: tuple[dict, ...] = (
    {"n": 1, "name": "screen_motivation", "cat": "面试关卡一:动机筛选与方向匹配", "kind": "dp", "bank": _gate1},
    {"n": 2, "name": "research_coding_paper2code", "cat": "面试关卡二:论文转代码研究手写", "kind": "dp", "bank": _gate2},
    {"n": 3, "name": "paper_critique_fluency", "cat": "面试关卡三:论文批判与研究流利度", "kind": "dp", "bank": _gate3},
    {"n": 4, "name": "ml_infra_system_design", "cat": "面试关卡四:ML基础设施系统设计判断", "kind": "sp", "bank": _gate4},
    {"n": 5, "name": "agentic_system_design", "cat": "面试关卡五:Agent生产系统设计判断", "kind": "sp", "bank": _gate5},
    {"n": 6, "name": "values_safety_stance", "cat": "面试关卡六:价值观与安全立场关(Values Round)", "kind": "sp", "bank": _gate6},
    {"n": 7, "name": "staff_ownership_narrative", "cat": "面试关卡七:资深叙事与研究方向主导权", "kind": "dp", "bank": _gate7},
    {"n": 8, "name": "cross_functional_ambiguity", "cat": "面试关卡八:跨团队协作与模糊决策判断", "kind": "sp", "bank": _gate8},
    {"n": 9, "name": "domestic_senior_socialhire", "cat": "面试关卡九:国内大厂资深社招视角", "kind": "dp", "bank": _gate9},
    {"n": 10, "name": "leveling_negotiation", "cat": "面试关卡十:定级与谈判判断", "kind": "sp", "bank": _gate10},
    {"n": 11, "name": "hpc_cluster_infra", "cat": "面试关卡十一:大规模集群HPC基础设施深水", "kind": "dp", "bank": _gate11},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 110, len(ALL_DP)
    assert len(ALL_SP) >= 90, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 190, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 6, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 5, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "staff_gauntlet ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "staff_gauntlet ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-sg-") for i in dp_ids), "存在不以dp-sg-开头的DeepPoint id"
    assert all(i.startswith("sc-sg-") for i in sp_ids), "存在不以sc-sg-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "staff_gauntlet ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "staff_gauntlet ALL_SP 内存在重复 trigger"

    assert len(GATES) == 11, len(GATES)
    assert [g["n"] for g in GATES] == list(range(1, 12)), "GATES 顺序不连续"
    gate_bank_total = sum(len(g["bank"]) for g in GATES)
    assert gate_bank_total == len(ALL_DP) + len(ALL_SP), "GATES 里的bank总数与ALL_DP+ALL_SP不一致"
    dp_gate_count = sum(1 for g in GATES if g["kind"] == "dp")
    sp_gate_count = sum(1 for g in GATES if g["kind"] == "sp")
    assert dp_gate_count == 6, dp_gate_count
    assert sp_gate_count == 5, sp_gate_count

    print(
        f"[PASS] staff_gauntlet: 11关(6个DeepPoint关+5个ScenarioPoint关) 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
