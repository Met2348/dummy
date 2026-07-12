"""跑全部终面深水区模块的 _self_test()：
deep_common 自检 + 17 个 dp_*.py 类别库 + ai_deep/backend_deep 聚合校验
+ dictation 套件"参考答案必须通过自己的 check()"自检（校验题目本身设计得对，不是校验用户默写）。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # AI 深水区（12 类）
    "ai_deep.dp_transformer_attention",
    "ai_deep.dp_rlhf_alignment",
    "ai_deep.dp_moe_systems",
    "ai_deep.dp_distributed_training",
    "ai_deep.dp_inference_serving",
    "ai_deep.dp_interpretability",
    "ai_deep.dp_rag_agent",
    "ai_deep.dp_pretraining_data",
    "ai_deep.dp_eval_safety",
    "ai_deep.dp_classic_ml_systemdesign",
    "ai_deep.dp_scaling_dynamics",
    "ai_deep.dp_agent_harness",
    # 后端深水区（5 类）
    "backend_deep.dp_os_internals",
    "backend_deep.dp_network_internals",
    "backend_deep.dp_database_internals",
    "backend_deep.dp_distributed_systems",
    "backend_deep.dp_cache_concurrency",
    # 2026 前沿补强（9 类 DeepPoint + 1 类 ScenarioPoint）
    "frontier_deep.dp_reasoning_testtime",
    "frontier_deep.dp_agentic_production",
    "frontier_deep.dp_alignment_oversight",
    "frontier_deep.dp_interpretability_2026",
    "frontier_deep.dp_multimodal_vla",
    "frontier_deep.dp_frontier_oss_models",
    "frontier_deep.dp_llm_infra_2026",
    "frontier_deep.dp_data_scaling_2026",
    "frontier_deep.dp_rag_tooling_2026",
    "frontier_deep.sc_engineering_judgment",
    # 聚合汇总校验
    "ai_deep",
    "backend_deep",
    "frontier_deep",
]


def _run_self_tests() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"=== {passed}/{len(MODULES)} modules passed ===")
    return passed


def _test_dictation_solutions() -> tuple[int, int]:
    from dictation import harness

    ok = 0
    total = len(harness.REGISTRY)
    for name, (spec_mod_name, target_name, check_mod_name) in harness.REGISTRY.items():
        solution_mod_name = spec_mod_name.replace(".specs.", ".solutions.").replace("_spec", "_solution")
        try:
            solution_mod = importlib.import_module(solution_mod_name)
            target = getattr(solution_mod, target_name)
            check_mod = importlib.import_module(check_mod_name)
            check_mod.check(target)
            ok += 1
        except Exception as e:
            print(f"[FAIL] dictation.{name}: solution 未通过自己的 check() — {type(e).__name__}: {e}")
    print(f"=== dictation solutions: {ok}/{total} passed ===")
    return ok, total


def main() -> int:
    passed = _run_self_tests()
    dict_ok, dict_total = _test_dictation_solutions()
    all_ok = passed == len(MODULES) and dict_ok == dict_total
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
