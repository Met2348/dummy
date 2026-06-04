"""Topic 6 scaling-infra 公共工具."""
from __future__ import annotations

import torch


def model_size_bytes(n_params: int, dtype: str = "bf16") -> int:
    """模型参数字节大小."""
    bytes_per = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5}
    return int(n_params * bytes_per.get(dtype, 4))


def optimizer_state_bytes(n_params: int, opt: str = "adamw") -> int:
    """优化器状态字节大小 (Adam = 2× momentum + 2× variance @ fp32)."""
    if opt == "adamw":
        return n_params * 4 * 2
    if opt == "sgd":
        return n_params * 4
    if opt == "8bit":
        return n_params * 1 * 2
    return 0


def activation_bytes(batch, seq_len, hidden, n_layer, dtype="bf16"):
    """estimated activation memory (rough)."""
    bpp = 2 if dtype == "bf16" else 4
    return batch * seq_len * hidden * n_layer * bpp * 10


def kv_cache_bytes(n_layer, n_kv_head, head_dim, ctx_len, dtype="bf16"):
    bpp = 2 if dtype == "bf16" else 4
    return n_layer * n_kv_head * head_dim * 2 * ctx_len * bpp


def total_train_memory(n_params, batch, seq, hidden, n_layer, opt="adamw"):
    """总训练显存估算（GB）."""
    weights = model_size_bytes(n_params, "bf16")
    grad = model_size_bytes(n_params, "bf16")
    optimizer = optimizer_state_bytes(n_params, opt)
    activ = activation_bytes(batch, seq, hidden, n_layer)
    total = weights + grad + optimizer + activ
    return {
        "weights_gb": weights / 1e9,
        "grad_gb": grad / 1e9,
        "optimizer_gb": optimizer / 1e9,
        "activ_gb": activ / 1e9,
        "total_gb": total / 1e9,
    }


if __name__ == "__main__":
    print("Llama-7B AdamW bf16 batch 4 seq 2048:")
    r = total_train_memory(n_params=7_000_000_000, batch=4, seq=2048,
                            hidden=4096, n_layer=32, opt="adamw")
    for k, v in r.items():
        print(f"  {k}: {v:.2f}")
