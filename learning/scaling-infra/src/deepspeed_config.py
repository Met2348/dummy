"""DeepSpeed config 生成器."""
from __future__ import annotations

import json


def make_ds_config_zero1(micro_batch: int = 4, grad_accum: int = 1) -> dict:
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
        },
        "gradient_clipping": 1.0,
    }


def make_ds_config_zero2(micro_batch: int = 4, grad_accum: int = 1) -> dict:
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
    }


def make_ds_config_zero3(micro_batch: int = 1, grad_accum: int = 16,
                          cpu_offload: bool = False) -> dict:
    cfg = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        "gradient_clipping": 1.0,
    }
    if cpu_offload:
        cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        cfg["zero_optimization"]["offload_param"] = {"device": "cpu"}
    return cfg


def make_ds_config_megatron(tp: int = 8, pp: int = 8) -> dict:
    return {
        "bf16": {"enabled": True},
        "tensor_parallel": {"tp_size": tp},
        "pipeline_parallel": {"stages": pp, "method": "1f1b"},
        "zero_optimization": {"stage": 1},
    }


if __name__ == "__main__":
    print("=== ZeRO Stage 3 ===")
    print(json.dumps(make_ds_config_zero3(cpu_offload=True), indent=2))

    print("\n=== Megatron-DS (TP=8, PP=8) ===")
    print(json.dumps(make_ds_config_megatron(), indent=2))
