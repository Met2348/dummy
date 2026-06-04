"""并行模式显存账本演示."""
from __future__ import annotations

from common import model_size_bytes, optimizer_state_bytes


def dp_memory(n_params: int, n_gpu: int) -> dict:
    """DP: 每卡全模型."""
    per_gpu = (model_size_bytes(n_params, "bf16")
               + model_size_bytes(n_params, "bf16")
               + optimizer_state_bytes(n_params, "adamw"))
    return {"per_gpu_gb": per_gpu / 1e9,
            "total_gb": per_gpu * n_gpu / 1e9}


def zero1_memory(n_params: int, n_gpu: int) -> dict:
    """ZeRO-1: shard optimizer state."""
    w = model_size_bytes(n_params, "bf16")
    g = model_size_bytes(n_params, "bf16")
    o = optimizer_state_bytes(n_params, "adamw") / n_gpu
    return {"per_gpu_gb": (w + g + o) / 1e9}


def zero2_memory(n_params: int, n_gpu: int) -> dict:
    """ZeRO-2: shard O + G."""
    w = model_size_bytes(n_params, "bf16")
    g = model_size_bytes(n_params, "bf16") / n_gpu
    o = optimizer_state_bytes(n_params, "adamw") / n_gpu
    return {"per_gpu_gb": (w + g + o) / 1e9}


def zero3_memory(n_params: int, n_gpu: int) -> dict:
    """ZeRO-3 / FSDP: shard W + G + O."""
    w = model_size_bytes(n_params, "bf16") / n_gpu
    g = model_size_bytes(n_params, "bf16") / n_gpu
    o = optimizer_state_bytes(n_params, "adamw") / n_gpu
    return {"per_gpu_gb": (w + g + o) / 1e9}


def tp_memory(n_params: int, tp_size: int) -> dict:
    """TP: 切 weight, 每卡 1/tp 模型."""
    w = model_size_bytes(n_params, "bf16") / tp_size
    g = model_size_bytes(n_params, "bf16") / tp_size
    o = optimizer_state_bytes(n_params, "adamw") / tp_size
    return {"per_gpu_gb": (w + g + o) / 1e9}


def pp_memory(n_params: int, pp_size: int) -> dict:
    """PP: 切 layer."""
    w = model_size_bytes(n_params, "bf16") / pp_size
    g = model_size_bytes(n_params, "bf16") / pp_size
    o = optimizer_state_bytes(n_params, "adamw") / pp_size
    return {"per_gpu_gb": (w + g + o) / 1e9}


if __name__ == "__main__":
    N = 70_000_000_000
    G = 64
    print(f"=== Llama-2-70B on {G} GPU (no activation) ===")
    print(f"  DP:      per_gpu={dp_memory(N, G)['per_gpu_gb']:.1f} GB")
    print(f"  ZeRO-1:  per_gpu={zero1_memory(N, G)['per_gpu_gb']:.1f} GB")
    print(f"  ZeRO-2:  per_gpu={zero2_memory(N, G)['per_gpu_gb']:.1f} GB")
    print(f"  ZeRO-3:  per_gpu={zero3_memory(N, G)['per_gpu_gb']:.1f} GB")
    print(f"  TP=8:    per_gpu={tp_memory(N, 8)['per_gpu_gb']:.1f} GB")
    print(f"  PP=8:    per_gpu={pp_memory(N, 8)['per_gpu_gb']:.1f} GB")
