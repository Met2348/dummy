"""跑全部 Compilers Mastery 模块的 _self_test():
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
    # tier1 浅:词法分析与语法分析基础
    "tier1_shallow.dp_comp_lexical_analysis_basics",
    "tier1_shallow.dp_comp_parsing_basics",
    # tier2 深:LR分析/语义分析类型系统/中间表示代码生成深水
    "tier2_deep.dp_comp_lr_parsing_deep",
    "tier2_deep.dp_comp_semantic_analysis_type_systems_deep",
    "tier2_deep.dp_comp_ir_codegen_optimization_deep",
    # tier3 社招级别:语言设计权衡与编译优化调试判断
    "tier3_social_hire.sc_comp_language_design_tradeoff_judgment",
    "tier3_social_hire.sc_comp_optimization_debugging_judgment",
    # 总聚合校验
    "compilers_mastery",
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
