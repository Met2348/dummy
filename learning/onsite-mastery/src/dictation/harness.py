"""默写练习 CLI harness：`python harness.py <name>`

动态加载 dictation/specs/<name>_spec.py 里用户已经闭卷填好的函数/类，
交给 dictation/checks/<name>_check.py 的 check() 做纯断言检验（check 模块本身不 import
solutions/，防止抄近道），报 PASS/FAIL + 耗时。默写方法见 lectures/00-how-to-defend.md。
"""
from __future__ import annotations

import importlib
import os
import sys
import time

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# name -> (spec 模块, 函数/类名, check 模块)
REGISTRY = {
    "attention": ("dictation.specs.attention_spec", "causal_attention", "dictation.checks.attention_check"),
    "rope": ("dictation.specs.rope_spec", "rope_apply", "dictation.checks.rope_check"),
    "rmsnorm": ("dictation.specs.rmsnorm_spec", "rmsnorm", "dictation.checks.rmsnorm_check"),
    "lora": ("dictation.specs.lora_spec", "lora_forward", "dictation.checks.lora_check"),
    "kv_cache": ("dictation.specs.kv_cache_spec", "kv_cache_step", "dictation.checks.kv_cache_check"),
    "sampling": ("dictation.specs.sampling_spec", "sample_top_k_top_p", "dictation.checks.sampling_check"),
    "training_step": ("dictation.specs.training_step_spec", "training_step", "dictation.checks.training_step_check"),
    "transformer_block": (
        "dictation.specs.transformer_block_spec",
        "transformer_block_forward",
        "dictation.checks.transformer_block_check",
    ),
    "ppo_clip": ("dictation.specs.ppo_clip_spec", "ppo_clip_objective", "dictation.checks.ppo_clip_check"),
    "gae": ("dictation.specs.gae_spec", "gae_advantage", "dictation.checks.gae_check"),
    "dpo_loss": ("dictation.specs.dpo_loss_spec", "dpo_loss", "dictation.checks.dpo_loss_check"),
    "grpo_advantage": (
        "dictation.specs.grpo_advantage_spec",
        "grpo_group_advantage",
        "dictation.checks.grpo_advantage_check",
    ),
    "moe_router": (
        "dictation.specs.moe_router_spec",
        "moe_topk_router_with_aux_loss",
        "dictation.checks.moe_router_check",
    ),
    "consistent_hashing": (
        "dictation.specs.consistent_hashing_spec",
        "ConsistentHashRing",
        "dictation.checks.consistent_hashing_check",
    ),
    "lru_cache": ("dictation.specs.lru_cache_spec", "LRUCache", "dictation.checks.lru_cache_check"),
    "rate_limiter": (
        "dictation.specs.rate_limiter_spec",
        "TokenBucketRateLimiter",
        "dictation.checks.rate_limiter_check",
    ),
    "bm25": ("dictation.specs.bm25_spec", "bm25_score", "dictation.checks.bm25_check"),
}


def run(name: str) -> bool:
    if name not in REGISTRY:
        print(f"[FAIL] 未知默写目标: {name}；可用: {', '.join(sorted(REGISTRY))}")
        return False
    spec_mod_name, target_name, check_mod_name = REGISTRY[name]
    try:
        spec_mod = importlib.import_module(spec_mod_name)
        target = getattr(spec_mod, target_name)
    except (ImportError, AttributeError) as e:
        print(f"[FAIL] {name}: 加载 spec 失败 — {type(e).__name__}: {e}")
        return False

    check_mod = importlib.import_module(check_mod_name)
    start = time.perf_counter()
    try:
        check_mod.check(target)
    except Exception as e:  # noqa: BLE001 —— 默写练习要把任何断言失败/NotImplementedError都报给用户看
        elapsed = time.perf_counter() - start
        print(f"[FAIL] {name}: {type(e).__name__}: {e} ({elapsed:.3f}s)")
        return False
    elapsed = time.perf_counter() - start
    print(f"[PASS] {name}: 通过全部检验 ({elapsed:.3f}s)")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: python harness.py <name>；可用目标:")
        for n in sorted(REGISTRY):
            print(f"  {n}")
        return 1
    return 0 if run(sys.argv[1]) else 1


if __name__ == "__main__":
    sys.exit(main())
