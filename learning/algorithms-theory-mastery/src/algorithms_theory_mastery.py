"""Algorithms Theory Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师"仔细思考和科班CS本科相比还差哪些能力"要求补齐的六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)第三个。和仓库里已有的 `leetcode-mastery`
(刷题练习)性质不同——这个专题是"证明为什么算法是对的"这个理论层面。`explain` 字段承载
"讲仔细"的系统性教学讲解，组织方式沿用"难度分层"轴：

- tier1(浅)：渐进分析与主定理(大O/大Omega/大Theta严格定义/递归式求解/主定理三种情形/均摊
  分析)、分治与贪心算法设计基础(分治三步骤/贪心选择性质/交换论证雏形)
- tier2(深)：动态规划正确性证明深水(最优子结构剪切-粘贴证明/状态定义设计原则/常见DP证明
  陷阱)、图算法正确性深水(Dijkstra/Bellman-Ford/Prim/Kruskal正确性证明/cut property)、
  网络流与匹配深水(Ford-Fulkerson/最大流最小割定理/Edmonds-Karp/Hall定理)
- tier3(社招级别)：算法设计范式选择判断(给定实际问题判断该用贪心/DP/分治)、复杂度权衡与
  工程实现判断(理论最优vs工程实用性权衡)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_alg_asymptotic_analysis_master_theorem import BANK as _t1_asym  # noqa: E402
from tier1_shallow.dp_alg_divide_conquer_greedy_basics import BANK as _t1_dcg  # noqa: E402
from tier2_deep.dp_alg_dynamic_programming_correctness_deep import BANK as _t2_dp  # noqa: E402
from tier2_deep.dp_alg_graph_algorithms_correctness_deep import BANK as _t2_graph  # noqa: E402
from tier2_deep.dp_alg_network_flow_matching_deep import BANK as _t2_flow  # noqa: E402
from tier3_social_hire.sc_alg_paradigm_selection_judgment import BANK as _t3_para  # noqa: E402
from tier3_social_hire.sc_alg_complexity_engineering_tradeoff_judgment import BANK as _t3_eng  # noqa: E402

ALL_DP = list(_t1_asym) + list(_t1_dcg) + list(_t2_dp) + list(_t2_graph) + list(_t2_flow)
ALL_SP = list(_t3_para) + list(_t3_eng)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:渐进分析主定理与分治贪心设计基础", "kind": "dp",
     "modules": ({"name": "alg_asymptotic_analysis_master_theorem", "cat": "算法理论基础一:渐进分析与主定理", "bank": _t1_asym},
                 {"name": "alg_divide_conquer_greedy_basics", "cat": "算法理论基础二:分治与贪心算法设计基础", "bank": _t1_dcg})},
    {"n": 2, "name": "deep", "label": "深:动态规划正确性证明/图算法正确性/网络流匹配深水", "kind": "dp",
     "modules": ({"name": "alg_dynamic_programming_correctness_deep", "cat": "算法理论深水一:动态规划正确性证明深水", "bank": _t2_dp},
                 {"name": "alg_graph_algorithms_correctness_deep", "cat": "算法理论深水二:图算法正确性深水", "bank": _t2_graph},
                 {"name": "alg_network_flow_matching_deep", "cat": "算法理论深水三:网络流与匹配深水", "bank": _t2_flow})},
    {"n": 3, "name": "social_hire", "label": "社招级别:算法范式选择与复杂度工程权衡判断", "kind": "sp",
     "modules": ({"name": "alg_paradigm_selection_judgment", "cat": "算法理论社招级别一:算法设计范式选择判断", "bank": _t3_para},
                 {"name": "alg_complexity_engineering_tradeoff_judgment", "cat": "算法理论社招级别二:复杂度权衡与工程实现判断", "bank": _t3_eng})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 60, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "algorithms-theory-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "algorithms-theory-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-alg-") for i in dp_ids), "存在不以dp-alg-开头的DeepPoint id"
    assert all(i.startswith("sc-alg-") for i in sp_ids), "存在不以sc-alg-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "algorithms-theory-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "algorithms-theory-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] algorithms_theory_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
