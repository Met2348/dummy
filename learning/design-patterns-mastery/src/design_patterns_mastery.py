"""Design Patterns Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第二个，组织方式沿用
software-engineering-mastery 已验证的"讲解+追问链/场景判断"混合格式("难度分层"轴)：

- tier1(浅)：GoF创建型模式(Singleton/Factory Method/Abstract Factory/Builder/Prototype)、
  结构型模式(Adapter/Decorator/Proxy/Facade/Composite/Bridge/Flyweight)基础认知
- tier2(深)：行为型模式(一)(Strategy/Observer/Command/Template Method/Iterator)、行为型
  模式(二)(State/Chain of Responsibility/Mediator/Memento/Visitor/Interpreter)、现代模式
  与反模式深水(DI容器/Repository-UoW/CQRS/函数式替代GoF/过度设计与God Object反模式)
- tier3(社招级别)：架构选型判断(什么场景用什么模式/模式组合/隐性模式识别)、过度设计边界判断
  (YAGNI/何时拒绝引入模式/rule of three)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_pat_creational_patterns import BANK as _t1_cre  # noqa: E402
from tier1_shallow.dp_pat_structural_patterns import BANK as _t1_str  # noqa: E402
from tier2_deep.dp_pat_behavioral_patterns_one import BANK as _t2_beh1  # noqa: E402
from tier2_deep.dp_pat_behavioral_patterns_two import BANK as _t2_beh2  # noqa: E402
from tier2_deep.dp_pat_modern_patterns_antipatterns import BANK as _t2_mod  # noqa: E402
from tier3_social_hire.sc_pat_architecture_selection_judgment import BANK as _t3_arch  # noqa: E402
from tier3_social_hire.sc_pat_overengineering_boundary_judgment import BANK as _t3_over  # noqa: E402

ALL_DP = list(_t1_cre) + list(_t1_str) + list(_t2_beh1) + list(_t2_beh2) + list(_t2_mod)
ALL_SP = list(_t3_arch) + list(_t3_over)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:创建型与结构型模式基础", "kind": "dp",
     "modules": ({"name": "pat_creational_patterns", "cat": "设计模式基础一:创建型模式", "bank": _t1_cre},
                 {"name": "pat_structural_patterns", "cat": "设计模式基础二:结构型模式", "bank": _t1_str})},
    {"n": 2, "name": "deep", "label": "深:行为型模式深水与现代模式演化", "kind": "dp",
     "modules": ({"name": "pat_behavioral_patterns_one", "cat": "设计模式深水一:行为型模式(一)", "bank": _t2_beh1},
                 {"name": "pat_behavioral_patterns_two", "cat": "设计模式深水二:行为型模式(二)", "bank": _t2_beh2},
                 {"name": "pat_modern_patterns_antipatterns", "cat": "设计模式深水三:现代模式与反模式深水", "bank": _t2_mod})},
    {"n": 3, "name": "social_hire", "label": "社招级别:架构选型与过度设计判断", "kind": "sp",
     "modules": ({"name": "pat_architecture_selection_judgment", "cat": "设计模式社招级别一:架构选型判断", "bank": _t3_arch},
                 {"name": "pat_overengineering_boundary_judgment", "cat": "设计模式社招级别二:过度设计边界判断", "bank": _t3_over})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "design-patterns-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "design-patterns-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-pat-") for i in dp_ids), "存在不以dp-pat-开头的DeepPoint id"
    assert all(i.startswith("sc-pat-") for i in sp_ids), "存在不以sc-pat-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "design-patterns-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "design-patterns-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] design_patterns_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
