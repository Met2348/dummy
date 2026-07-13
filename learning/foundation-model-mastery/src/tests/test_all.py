"""跑全部 Foundation Model Mastery 模块的 _self_test():
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
    # tier1 浅:基础认知框架
    "tier1_shallow.dp_fm_taxonomy_landscape",
    "tier1_shallow.dp_tokenizer_design",
    # tier2 深:机制深水
    "tier2_deep.dp_non_transformer_architectures",
    "tier2_deep.dp_model_family_derivation",
    "tier2_deep.dp_fm_evaluation_methodology",
    # tier3 社招级别:资深战略判断
    "tier3_social_hire.sc_fm_release_governance",
    "tier3_social_hire.sc_fm_training_economics_judgment",
    # 总聚合校验
    "foundation_model_mastery",
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
