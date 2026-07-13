"""Software Engineering Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老手指出用户 EE 转 NLP/LLM 背景缺少科班 CS 学生默认已掌握的软件工程基础知识，这是老手要求补齐的
五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第一个。与之前 foundation-model-mastery/
federated-learning-mastery/diffusion-mastery 三个专题不同：那三个专题用户已有研究生级基础只需要
整理成追问链，这五个专题用户是完全没系统学过，所以 DeepPoint/ScenarioPoint 都新增了 `explain`
字段承载"讲仔细"的系统性教学讲解，组织方式仍沿用"难度分层"轴：

- tier1(浅)：SDLC与敏捷/Scrum/Kanban基础认知框架、版本控制(git分支模型)与CI/CD协作流程基础
- tier2(深)：测试方法论深水(测试金字塔/TDD-BDD/mock分类/覆盖率陷阱/契约测试/变异测试)、架构与
  模块化深水(六边形架构/Clean Architecture/SOLID深水/DI/微服务vs单体)、代码质量与重构深水
  (code smell/重构手法/DRY真正含义/技术债务利息/评审有效性研究)
- tier3(社招级别)：技术债务与工程文化判断(重写vs重构/优先级判断/测试文化推动)、交付与发布策略
  判断(灰度vs蓝绿/DORA指标/blameless postmortem/契约测试跨团队协调)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_se_sdlc_agile_requirements import BANK as _t1_sdlc  # noqa: E402
from tier1_shallow.dp_se_version_control_cicd_basics import BANK as _t1_vcs  # noqa: E402
from tier2_deep.dp_se_testing_methodology import BANK as _t2_test  # noqa: E402
from tier2_deep.dp_se_architecture_modularity import BANK as _t2_arch  # noqa: E402
from tier2_deep.dp_se_code_quality_refactoring import BANK as _t2_qual  # noqa: E402
from tier3_social_hire.sc_se_tech_debt_engineering_culture_judgment import BANK as _t3_debt  # noqa: E402
from tier3_social_hire.sc_se_delivery_release_judgment import BANK as _t3_rel  # noqa: E402

ALL_DP = list(_t1_sdlc) + list(_t1_vcs) + list(_t2_test) + list(_t2_arch) + list(_t2_qual)
ALL_SP = list(_t3_debt) + list(_t3_rel)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:SDLC/敏捷与版本控制协作流程基础", "kind": "dp",
     "modules": ({"name": "se_sdlc_agile_requirements", "cat": "软件工程基础一:SDLC与需求工程", "bank": _t1_sdlc},
                 {"name": "se_version_control_cicd_basics", "cat": "软件工程基础二:版本控制与协作流程基础", "bank": _t1_vcs})},
    {"n": 2, "name": "deep", "label": "深:测试/架构/代码质量方法论深水", "kind": "dp",
     "modules": ({"name": "se_testing_methodology", "cat": "软件工程深水一:测试方法论深水", "bank": _t2_test},
                 {"name": "se_architecture_modularity", "cat": "软件工程深水二:架构与模块化深水", "bank": _t2_arch},
                 {"name": "se_code_quality_refactoring", "cat": "软件工程深水三:代码质量与重构深水", "bank": _t2_qual})},
    {"n": 3, "name": "social_hire", "label": "社招级别:工程文化与交付判断", "kind": "sp",
     "modules": ({"name": "se_tech_debt_engineering_culture_judgment", "cat": "软件工程社招级别一:技术债务与工程文化判断", "bank": _t3_debt},
                 {"name": "se_delivery_release_judgment", "cat": "软件工程社招级别二:交付与发布策略判断", "bank": _t3_rel})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "software-engineering-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "software-engineering-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-se-") for i in dp_ids), "存在不以dp-se-开头的DeepPoint id"
    assert all(i.startswith("sc-se-") for i in sp_ids), "存在不以sc-se-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "software-engineering-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "software-engineering-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] software_engineering_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
