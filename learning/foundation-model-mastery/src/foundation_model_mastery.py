"""Foundation Model Mastery 总聚合:浅→深→社招级别三层,合计7个文件106点。

与 onsite-mastery 的知识主题包(ai_deep/backend_deep/frontier_deep)、面试关卡包
(staff_gauntlet/social_hire_gauntlet)都不同,这里的组织轴是"难度分层"——tier1_shallow
(基础认知框架,面向对基座模型领域相对陌生的候选人)→tier2_deep(深水机制细节,面向已经建立
基础认知、需要接受连续追问的候选人)→tier3_social_hire(无标准答案的资深战略判断力,面向已有
实际工作经验、需要在真金白银预算和组织张力下拍板的候选人)。

刻意避开与已有792点内容(ai_deep+backend_deep+frontier_deep+staff_gauntlet+
social_hire_gauntlet)的重复:标准Transformer内部机制/RLHF/MoE/scaling law本身的数学
规律/预训练数据配比这些已经被覆盖的角度,这里全部跳过,只做真正的知识空白:
- tier1:基座模型宏观分类框架、tokenizer设计(此前任何题库都只是顺带提及,从未独立成篇)
- tier2:非Transformer/混合架构(SSM/Mamba/RWKV/RetNet,此前完全空白的知识维度)、模型家族
  衍生工程(continual pretraining/蒸馏/模型合并/长上下文扩展)、发布前评测方法论(污染/饱和/
  评测方差,区别于dp_eval_safety的RLHF安全评测角度)
- tier3:发布治理判断、训练经济学与算力战略判断(这两个此前完全空白的资深决策维度)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_fm_taxonomy_landscape import BANK as _t1_tax  # noqa: E402
from tier1_shallow.dp_tokenizer_design import BANK as _t1_tok  # noqa: E402
from tier2_deep.dp_non_transformer_architectures import BANK as _t2_arc  # noqa: E402
from tier2_deep.dp_model_family_derivation import BANK as _t2_drv  # noqa: E402
from tier2_deep.dp_fm_evaluation_methodology import BANK as _t2_evl  # noqa: E402
from tier3_social_hire.sc_fm_release_governance import BANK as _t3_gov  # noqa: E402
from tier3_social_hire.sc_fm_training_economics_judgment import BANK as _t3_eco  # noqa: E402

ALL_DP = list(_t1_tax) + list(_t1_tok) + list(_t2_arc) + list(_t2_drv) + list(_t2_evl)
ALL_SP = list(_t3_gov) + list(_t3_eco)

# TIERS:按"浅→深→社招级别"难度顺序排列的分层元数据,练习时建议按tier顺序从1到3推进,
# 而不是从tier3的资深判断题练起——这类题目预设你已经具备tier1/tier2的知识框架。
TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:基础认知框架", "kind": "dp",
     "modules": ({"name": "fm_taxonomy_landscape", "cat": "基座模型基础一:模型谱系与分类框架", "bank": _t1_tax},
                 {"name": "tokenizer_design", "cat": "基座模型基础二:Tokenizer设计", "bank": _t1_tok})},
    {"n": 2, "name": "deep", "label": "深:机制深水", "kind": "mixed",
     "modules": ({"name": "non_transformer_architectures", "cat": "基座模型深水一:非Transformer与混合架构", "bank": _t2_arc},
                 {"name": "model_family_derivation", "cat": "基座模型深水二:模型家族衍生工程", "bank": _t2_drv},
                 {"name": "fm_evaluation_methodology", "cat": "基座模型深水三:发布前评测方法论", "bank": _t2_evl})},
    {"n": 3, "name": "social_hire", "label": "社招级别:资深战略判断", "kind": "sp",
     "modules": ({"name": "fm_release_governance", "cat": "基座模型社招级别一:发布治理判断", "bank": _t3_gov},
                 {"name": "fm_training_economics_judgment", "cat": "基座模型社招级别二:训练经济学与算力战略判断", "bank": _t3_eco})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "foundation-model-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "foundation-model-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-fm-") for i in dp_ids), "存在不以dp-fm-开头的DeepPoint id"
    assert all(i.startswith("sc-fm-") for i in sp_ids), "存在不以sc-fm-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "foundation-model-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "foundation-model-mastery ALL_SP 内存在重复 trigger"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] foundation_model_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
