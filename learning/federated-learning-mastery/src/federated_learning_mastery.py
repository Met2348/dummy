"""Federated Learning Mastery 总聚合:浅→深→社招级别三层,合计7个文件109点。

联邦学习是此前整个题库(898点,含onsite-mastery五子包+foundation-model-mastery)完全没有
覆盖过的全新领域,因此不存在需要刻意规避的重叠内容,7个文件按"难度分层"(而非知识主题或面试
关卡)组织:

- tier1(浅):FL核心概念与FedAvg机制、FL系统角色与部署形态——建立"这个领域是什么、怎么部署"
  的基础认知框架
- tier2(深):Non-IID挑战与个性化(FedProx/SCAFFOLD/FedBN)、通信效率与压缩(量化/稀疏化/
  联邦蒸馏)、隐私与安全深水(梯度反演攻击/差分隐私/安全聚合/投毒攻击/Byzantine-robust聚合)
- tier3(社招级别):生产部署与激励机制判断、FL for LLM时代前沿判断(FedLoRA/联邦RLHF/边缘
  设备部署LLM的现实约束)——无标准答案的资深战略判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_fl_basics_fedavg import BANK as _t1_bas  # noqa: E402
from tier1_shallow.dp_fl_system_deployment import BANK as _t1_sys  # noqa: E402
from tier2_deep.dp_fl_noniid_personalization import BANK as _t2_niid  # noqa: E402
from tier2_deep.dp_fl_communication_efficiency import BANK as _t2_comm  # noqa: E402
from tier2_deep.dp_fl_privacy_security import BANK as _t2_prv  # noqa: E402
from tier3_social_hire.sc_fl_production_incentive_judgment import BANK as _t3_ops  # noqa: E402
from tier3_social_hire.sc_fl_llm_frontier_judgment import BANK as _t3_llm  # noqa: E402

ALL_DP = list(_t1_bas) + list(_t1_sys) + list(_t2_niid) + list(_t2_comm) + list(_t2_prv)
ALL_SP = list(_t3_ops) + list(_t3_llm)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:核心概念与系统部署", "kind": "dp",
     "modules": ({"name": "fl_basics_fedavg", "cat": "联邦学习基础一:核心概念与FedAvg机制", "bank": _t1_bas},
                 {"name": "fl_system_deployment", "cat": "联邦学习基础二:系统角色与部署形态", "bank": _t1_sys})},
    {"n": 2, "name": "deep", "label": "深:机制深水", "kind": "dp",
     "modules": ({"name": "fl_noniid_personalization", "cat": "联邦学习深水一:Non-IID挑战与个性化", "bank": _t2_niid},
                 {"name": "fl_communication_efficiency", "cat": "联邦学习深水二:通信效率与压缩", "bank": _t2_comm},
                 {"name": "fl_privacy_security", "cat": "联邦学习深水三:隐私与安全深水", "bank": _t2_prv})},
    {"n": 3, "name": "social_hire", "label": "社招级别:资深战略判断", "kind": "sp",
     "modules": ({"name": "fl_production_incentive_judgment", "cat": "联邦学习社招级别一:生产部署与激励机制判断", "bank": _t3_ops},
                 {"name": "fl_llm_frontier_judgment", "cat": "联邦学习社招级别二:FL for LLM时代前沿判断", "bank": _t3_llm})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "federated-learning-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "federated-learning-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-fl-") for i in dp_ids), "存在不以dp-fl-开头的DeepPoint id"
    assert all(i.startswith("sc-fl-") for i in sp_ids), "存在不以sc-fl-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "federated-learning-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "federated-learning-mastery ALL_SP 内存在重复 trigger"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] federated_learning_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
