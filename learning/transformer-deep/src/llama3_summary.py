"""Llama-3 architecture 关键数字."""
from __future__ import annotations


CONFIGS = {
    "Llama-3 8B": dict(
        n_layer=32, hidden_size=4096, n_head=32, n_kv_head=8, d_head=128,
        intermediate_size=14336, vocab_size=128_256, context=128_000,
        rope_base=500_000.0, activation="SwiGLU", norm="RMSNorm Pre",
    ),
    "Llama-3 70B": dict(
        n_layer=80, hidden_size=8192, n_head=64, n_kv_head=8, d_head=128,
        intermediate_size=28672, vocab_size=128_256, context=128_000,
        rope_base=500_000.0, activation="SwiGLU", norm="RMSNorm Pre",
    ),
    "Llama-3.1 405B": dict(
        n_layer=126, hidden_size=16384, n_head=128, n_kv_head=16, d_head=128,
        intermediate_size=53248, vocab_size=128_256, context=128_000,
        rope_base=500_000.0, activation="SwiGLU", norm="RMSNorm Pre",
    ),
}


def kv_cache_per_seq(cfg: dict, t: int, dtype_bytes: int = 2) -> int:
    return 2 * cfg["n_layer"] * cfg["n_kv_head"] * t * cfg["d_head"] * dtype_bytes


def main() -> None:
    for name, cfg in CONFIGS.items():
        print(f"\n=== {name} ===")
        for k, v in cfg.items():
            print(f"  {k:<20} = {v}")
        kv = kv_cache_per_seq(cfg, t=32_000)
        print(f"  KV cache @ 32k:  {kv / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
