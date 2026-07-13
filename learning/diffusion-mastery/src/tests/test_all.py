"""跑全部 Diffusion Mastery 模块的 _self_test():
deep_common 自检 + 7 个 tier 模块自检 + 总聚合校验。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # tier1 浅:DDPM核心机制与采样引导基础
    "tier1_shallow.dp_diffusion_forward_reverse_ddpm",
    "tier1_shallow.dp_diffusion_sampling_guidance_basics",
    # tier2 深:理论统一框架与架构深水
    "tier2_deep.dp_diffusion_score_sde",
    "tier2_deep.dp_diffusion_latent_dit",
    "tier2_deep.dp_diffusion_flow_matching_consistency",
    # tier3 社招级别:资深战略判断
    "tier3_social_hire.sc_diffusion_production_deployment_judgment",
    "tier3_social_hire.sc_diffusion_llm_multimodal_frontier_judgment",
    # 总聚合校验
    "diffusion_mastery",
]


def main() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"=== {passed}/{len(MODULES)} modules passed ===")
    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
