"""Compilers Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师"仔细思考和科班CS本科相比还差哪些能力"要求补齐的六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)第五个。`explain` 字段承载"讲仔细"的系统性
教学讲解，组织方式沿用"难度分层"轴：

- tier1(浅)：词法分析(token/正则表达式到DFA/最长匹配)、语法分析基础(递归下降/LL(1)/
  FIRST-FOLLOW/消除左递归)
- tier2(深)：LR分析深水(LR(0)/SLR/LALR/分析表/移进规约冲突)、语义分析与类型系统深水
  (符号表作用域/类型检查/Hindley-Milner类型推断)、中间表示与代码生成优化深水(三地址码/
  SSA/CFG/常量折叠死代码消除/图着色寄存器分配)
- tier3(社招级别)：语言设计与实现权衡判断(解析策略选择/DSL设计判断)、编译器优化与调试
  判断(优化引入的bug定位/JIT vs AOT判断)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_comp_lexical_analysis_basics import BANK as _t1_lex  # noqa: E402
from tier1_shallow.dp_comp_parsing_basics import BANK as _t1_parse  # noqa: E402
from tier2_deep.dp_comp_lr_parsing_deep import BANK as _t2_lr  # noqa: E402
from tier2_deep.dp_comp_semantic_analysis_type_systems_deep import BANK as _t2_sem  # noqa: E402
from tier2_deep.dp_comp_ir_codegen_optimization_deep import BANK as _t2_ir  # noqa: E402
from tier3_social_hire.sc_comp_language_design_tradeoff_judgment import BANK as _t3_lang  # noqa: E402
from tier3_social_hire.sc_comp_optimization_debugging_judgment import BANK as _t3_opt  # noqa: E402

ALL_DP = list(_t1_lex) + list(_t1_parse) + list(_t2_lr) + list(_t2_sem) + list(_t2_ir)
ALL_SP = list(_t3_lang) + list(_t3_opt)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:词法分析与语法分析基础", "kind": "dp",
     "modules": ({"name": "comp_lexical_analysis_basics", "cat": "编译原理基础一:词法分析", "bank": _t1_lex},
                 {"name": "comp_parsing_basics", "cat": "编译原理基础二:语法分析基础", "bank": _t1_parse})},
    {"n": 2, "name": "deep", "label": "深:LR分析/语义分析类型系统/中间表示代码生成深水", "kind": "dp",
     "modules": ({"name": "comp_lr_parsing_deep", "cat": "编译原理深水一:LR分析深水", "bank": _t2_lr},
                 {"name": "comp_semantic_analysis_type_systems_deep", "cat": "编译原理深水二:语义分析与类型系统深水", "bank": _t2_sem},
                 {"name": "comp_ir_codegen_optimization_deep", "cat": "编译原理深水三:中间表示与代码生成优化深水", "bank": _t2_ir})},
    {"n": 3, "name": "social_hire", "label": "社招级别:语言设计权衡与编译优化调试判断", "kind": "sp",
     "modules": ({"name": "comp_language_design_tradeoff_judgment", "cat": "编译原理社招级别一:语言设计与实现权衡判断", "bank": _t3_lang},
                 {"name": "comp_optimization_debugging_judgment", "cat": "编译原理社招级别二:编译器优化与调试判断", "bank": _t3_opt})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "compilers-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "compilers-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-comp-") for i in dp_ids), "存在不以dp-comp-开头的DeepPoint id"
    assert all(i.startswith("sc-comp-") for i in sp_ids), "存在不以sc-comp-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "compilers-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "compilers-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] compilers_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
