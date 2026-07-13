"""Discrete Math Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师指出"和科班CS本科相比还缺哪些能力"之后，确认要补齐六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)，这是第一个。和已完成的五个 CS 基础专题
(software-engineering-mastery 等)一样，`explain` 字段承载"讲仔细"的系统性教学讲解，组织方式
沿用"难度分层"轴：

- tier1(浅)：逻辑与集合函数(命题逻辑/谓词逻辑/集合运算/函数性质)、关系-偏序与计数基础(等价关系/
  偏序/哈斯图/基本计数原理/排列组合/鸽笼原理)
- tier2(深)：组合数学深水(容斥原理/生成函数/递推关系/Catalan数/组合恒等式证明)、图论深水(树/欧拉
  路径/哈密顿路径/图着色/二分图/平面图)、证明方法深水(直接证明/反证法/归纳法/结构归纳法/不变量/
  良序原理)
- tier3(社招级别)：算法建模判断(给定实际问题该用什么离散数学工具建模)、证明策略与常见错误判断
  (识别证明里的逻辑漏洞)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_dm_logic_sets_functions import BANK as _t1_logic  # noqa: E402
from tier1_shallow.dp_dm_relations_orders_counting_basics import BANK as _t1_rel  # noqa: E402
from tier2_deep.dp_dm_combinatorics_deep import BANK as _t2_comb  # noqa: E402
from tier2_deep.dp_dm_graph_theory_deep import BANK as _t2_graph  # noqa: E402
from tier2_deep.dp_dm_proof_techniques_deep import BANK as _t2_proof  # noqa: E402
from tier3_social_hire.sc_dm_algorithm_modeling_judgment import BANK as _t3_model  # noqa: E402
from tier3_social_hire.sc_dm_proof_strategy_pitfalls_judgment import BANK as _t3_proof  # noqa: E402

ALL_DP = list(_t1_logic) + list(_t1_rel) + list(_t2_comb) + list(_t2_graph) + list(_t2_proof)
ALL_SP = list(_t3_model) + list(_t3_proof)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:逻辑集合函数与关系偏序计数基础", "kind": "dp",
     "modules": ({"name": "dm_logic_sets_functions", "cat": "离散数学基础一:逻辑与集合函数", "bank": _t1_logic},
                 {"name": "dm_relations_orders_counting_basics", "cat": "离散数学基础二:关系-偏序与计数基础", "bank": _t1_rel})},
    {"n": 2, "name": "deep", "label": "深:组合数学/图论/证明方法深水", "kind": "dp",
     "modules": ({"name": "dm_combinatorics_deep", "cat": "离散数学深水一:组合数学深水", "bank": _t2_comb},
                 {"name": "dm_graph_theory_deep", "cat": "离散数学深水二:图论深水", "bank": _t2_graph},
                 {"name": "dm_proof_techniques_deep", "cat": "离散数学深水三:证明方法深水", "bank": _t2_proof})},
    {"n": 3, "name": "social_hire", "label": "社招级别:算法建模与证明策略判断", "kind": "sp",
     "modules": ({"name": "dm_algorithm_modeling_judgment", "cat": "离散数学社招级别一:算法建模判断", "bank": _t3_model},
                 {"name": "dm_proof_strategy_pitfalls_judgment", "cat": "离散数学社招级别二:证明策略与常见错误判断", "bank": _t3_proof})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "discrete-math-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "discrete-math-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-dm-") for i in dp_ids), "存在不以dp-dm-开头的DeepPoint id"
    assert all(i.startswith("sc-dm-") for i in sp_ids), "存在不以sc-dm-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "discrete-math-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "discrete-math-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] discrete_math_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
