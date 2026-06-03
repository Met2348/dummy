"""DeepSeek-V3 architecture 关键数字 + 概念图.

不下载 671B 权重；只展示 architecture 配置。
"""
from __future__ import annotations


CONFIG = {
    "total_params": 671_000_000_000,
    "activated_params": 37_000_000_000,
    "n_layer": 61,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "n_head": 128,
    "n_kv_head_mla": "MLA",
    "d_low_kv": 512,
    "d_rope_per_head": 64,
    "d_nope_per_head": 128,
    "moe_n_shared": 1,
    "moe_n_routed": 256,
    "moe_top_k": 8,
    "moe_aux_loss": "Aux-Free (bias update_rate=1e-3)",
    "mtp_n_extra_heads": 1,
    "vocab_size": 129280,
    "context_length": 128_000,
    "train_tokens": 14_800_000_000_000,
    "fp8_training": True,
    "activation": "SwiGLU",
    "norm": "RMSNorm Pre",
}


def kv_cache_size_per_seq(t: int, n_layer: int = 61, d_low: int = 512,
                          dtype_bytes: int = 2) -> int:
    return n_layer * t * d_low * dtype_bytes


def main() -> None:
    print("\n=== DeepSeek-V3 Architecture Summary ===")
    for k, v in CONFIG.items():
        print(f"  {k:<25} = {v}")
    print("\n=== KV cache (per request, 32k context) ===")
    bytes_ = kv_cache_size_per_seq(32_000)
    print(f"  MLA  cache: {bytes_ / 1024**3:.2f} GB")
    bytes_gqa = 2 * 61 * 8 * 32_000 * 128 * 2
    print(f"  GQA  ref:   {bytes_gqa / 1024**3:.2f} GB  (假设 g=8)")


if __name__ == "__main__":
    main()
