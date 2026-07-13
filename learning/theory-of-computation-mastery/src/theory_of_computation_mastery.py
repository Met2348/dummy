"""Theory of Computation Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师"仔细思考和科班CS本科相比还差哪些能力"要求补齐的六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)第二个。`explain` 字段承载"讲仔细"的系统性
教学讲解，组织方式沿用"难度分层"轴：

- tier1(浅)：自动机与正则语言(DFA/NFA/子集构造法/正则表达式/泵引理)、上下文无关文法与
  下推自动机(CFG/二义性/乔姆斯基范式/PDA/CFL泵引理)
- tier2(深)：图灵机与可计算性深水(邱奇-图灵论题/通用图灵机/停机问题/Rice定理)、计算复杂性
  类深水(P/NP/co-NP/Cook-Levin定理/经典NP完全问题)、归约与NP完全性证明深水(归约标准套路/
  近似算法/参数化复杂度)
- tier3(社招级别)：算法可行性判断(给定实际问题判断NP难/不可判定及应对策略)、形式化建模判断
  (给定实际问题判断该用什么层级的计算模型描述)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_toc_automata_regular_languages import BANK as _t1_fa  # noqa: E402
from tier1_shallow.dp_toc_grammars_pushdown_automata import BANK as _t1_cfg  # noqa: E402
from tier2_deep.dp_toc_turing_machines_computability_deep import BANK as _t2_tm  # noqa: E402
from tier2_deep.dp_toc_complexity_classes_deep import BANK as _t2_comp  # noqa: E402
from tier2_deep.dp_toc_reductions_np_completeness_proofs_deep import BANK as _t2_red  # noqa: E402
from tier3_social_hire.sc_toc_computational_feasibility_judgment import BANK as _t3_feas  # noqa: E402
from tier3_social_hire.sc_toc_formal_modeling_judgment import BANK as _t3_model  # noqa: E402

ALL_DP = list(_t1_fa) + list(_t1_cfg) + list(_t2_tm) + list(_t2_comp) + list(_t2_red)
ALL_SP = list(_t3_feas) + list(_t3_model)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:自动机正则语言与文法下推自动机基础", "kind": "dp",
     "modules": ({"name": "toc_automata_regular_languages", "cat": "计算理论基础一:自动机与正则语言", "bank": _t1_fa},
                 {"name": "toc_grammars_pushdown_automata", "cat": "计算理论基础二:上下文无关文法与下推自动机", "bank": _t1_cfg})},
    {"n": 2, "name": "deep", "label": "深:图灵机可计算性/复杂性类/归约NP完全性证明深水", "kind": "dp",
     "modules": ({"name": "toc_turing_machines_computability_deep", "cat": "计算理论深水一:图灵机与可计算性深水", "bank": _t2_tm},
                 {"name": "toc_complexity_classes_deep", "cat": "计算理论深水二:计算复杂性类深水", "bank": _t2_comp},
                 {"name": "toc_reductions_np_completeness_proofs_deep", "cat": "计算理论深水三:归约与NP完全性证明深水", "bank": _t2_red})},
    {"n": 3, "name": "social_hire", "label": "社招级别:算法可行性与形式化建模判断", "kind": "sp",
     "modules": ({"name": "toc_computational_feasibility_judgment", "cat": "计算理论社招级别一:算法可行性判断", "bank": _t3_feas},
                 {"name": "toc_formal_modeling_judgment", "cat": "计算理论社招级别二:形式化建模判断", "bank": _t3_model})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "theory-of-computation-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "theory-of-computation-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-toc-") for i in dp_ids), "存在不以dp-toc-开头的DeepPoint id"
    assert all(i.startswith("sc-toc-") for i in sp_ids), "存在不以sc-toc-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "theory-of-computation-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "theory-of-computation-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] theory_of_computation_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
