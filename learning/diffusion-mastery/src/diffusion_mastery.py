"""Diffusion Mastery 总聚合:浅→深→社招级别三层,合计7个文件108点。

这是老手FM/FL/Diffusion三专题队列的第三个,也是最后一个,组织方式沿用foundation-model-
mastery/federated-learning-mastery已验证的"难度分层"轴:

- tier1(浅):DDPM前向反向过程基础机制、采样与引导(DDIM/Classifier-Free Guidance)基础
  ——建立"扩散模型是什么、怎么训练、怎么采样出图"的基础认知框架
- tier2(深):Score-based生成与SDE统一框架(把DDPM和NCSN统一到连续时间视角)、Latent
  Diffusion与DiT架构深水(Stable Diffusion/DiT的具体架构机制)、Flow Matching与一致性
  模型深水(2024-2025年前沿采样加速理论)
- tier3(社招级别):生产部署判断(采样方法选型/蒸馏时机/内容安全/版权合规)、Diffusion
  for LLM/多模态前沿判断(扩散语言模型/机器人扩散策略/视频生成/统一多模态架构这些2026年
  交叉前沿的资深战略判断)——无标准答案的资深战略判断

与仓库已有的diffusion-foundations/diffusion-language-models/dit-latent-diffusion/
flow-matching-sota这几个学术notebook portfolio目录性质不同(那些是"实现深度"的代码
notebook),本track是"面试问答"的DeepPoint/ScenarioPoint题库,部分DeepPoint的
real_world_link字段引用了这些portfolio目录下真实存在的具体文件(已逐一核实)。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_diffusion_forward_reverse_ddpm import BANK as _t1_ddpm  # noqa: E402
from tier1_shallow.dp_diffusion_sampling_guidance_basics import BANK as _t1_samp  # noqa: E402
from tier2_deep.dp_diffusion_score_sde import BANK as _t2_sde  # noqa: E402
from tier2_deep.dp_diffusion_latent_dit import BANK as _t2_dit  # noqa: E402
from tier2_deep.dp_diffusion_flow_matching_consistency import BANK as _t2_flow  # noqa: E402
from tier3_social_hire.sc_diffusion_production_deployment_judgment import BANK as _t3_ops  # noqa: E402
from tier3_social_hire.sc_diffusion_llm_multimodal_frontier_judgment import BANK as _t3_front  # noqa: E402

ALL_DP = list(_t1_ddpm) + list(_t1_samp) + list(_t2_sde) + list(_t2_dit) + list(_t2_flow)
ALL_SP = list(_t3_ops) + list(_t3_front)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:DDPM核心机制与采样引导基础", "kind": "dp",
     "modules": ({"name": "diffusion_forward_reverse_ddpm", "cat": "扩散模型基础一:前向反向过程与DDPM", "bank": _t1_ddpm},
                 {"name": "diffusion_sampling_guidance_basics", "cat": "扩散模型基础二:采样与引导基础", "bank": _t1_samp})},
    {"n": 2, "name": "deep", "label": "深:理论统一框架与架构深水", "kind": "dp",
     "modules": ({"name": "diffusion_score_sde", "cat": "扩散模型深水一:Score-based生成与SDE统一框架", "bank": _t2_sde},
                 {"name": "diffusion_latent_dit", "cat": "扩散模型深水二:Latent Diffusion与DiT架构深水", "bank": _t2_dit},
                 {"name": "diffusion_flow_matching_consistency", "cat": "扩散模型深水三:Flow Matching与一致性模型深水", "bank": _t2_flow})},
    {"n": 3, "name": "social_hire", "label": "社招级别:资深战略判断", "kind": "sp",
     "modules": ({"name": "diffusion_production_deployment_judgment", "cat": "扩散模型社招级别一:生产部署判断", "bank": _t3_ops},
                 {"name": "diffusion_llm_multimodal_frontier_judgment", "cat": "扩散模型社招级别二:Diffusion for LLM/多模态前沿判断", "bank": _t3_front})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "diffusion-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "diffusion-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-diff-") for i in dp_ids), "存在不以dp-diff-开头的DeepPoint id"
    assert all(i.startswith("sc-diff-") for i in sp_ids), "存在不以sc-diff-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "diffusion-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "diffusion-mastery ALL_SP 内存在重复 trigger"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] diffusion_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
