"""跑全部 34 个模块（23 个八股问答库 ai_qa/backend_qa + 11 个代码验证模块
ai_coding/backend_coding）的 _self_test() + tracker 一致性检查。"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "qa_common",
    # AI/算法岗八股（15 类）
    "ai_qa.qbank_optimization",
    "ai_qa.qbank_regularization",
    "ai_qa.qbank_normalization",
    "ai_qa.qbank_metrics",
    "ai_qa.qbank_transformer",
    "ai_qa.qbank_tokenizer_data",
    "ai_qa.qbank_peft",
    "ai_qa.qbank_rlhf",
    "ai_qa.qbank_moe",
    "ai_qa.qbank_distributed_training",
    "ai_qa.qbank_inference_serving",
    "ai_qa.qbank_agent_rag",
    "ai_qa.qbank_interpretability",
    "ai_qa.qbank_classic_ml",
    "ai_qa.qbank_system_design",
    # 后端通用八股（8 类）
    "backend_qa.qbank_os",
    "backend_qa.qbank_network",
    "backend_qa.qbank_database",
    "backend_qa.qbank_jvm_concurrency",
    "backend_qa.qbank_distributed_systems",
    "backend_qa.qbank_cache_storage",
    "backend_qa.qbank_design_patterns",
    "backend_qa.qbank_linux_ops",
    # 聚合汇总校验
    "ai_qa",
    "backend_qa",
    # AI 代码验证模块（5 个）
    "ai_coding.moe_routing",
    "ai_coding.dist_memory_calc",
    "ai_coding.tool_calling_schema",
    "ai_coding.rag_retrieval",
    "ai_coding.quantization_infer",
    # 后端代码验证模块（6 个）
    "backend_coding.sync_primitives",
    "backend_coding.lru_lfu_cache",
    "backend_coding.consistent_hashing",
    "backend_coding.rate_limiter",
    "backend_coding.bplus_tree_sim",
    "backend_coding.design_patterns_demo",
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

    import tracker
    tracker._self_test()

    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
