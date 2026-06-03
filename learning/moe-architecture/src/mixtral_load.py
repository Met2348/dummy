"""Mixtral-8x7B 架构统计 — 不下载权重，仅 config."""
from __future__ import annotations


MIXTRAL_8X7B = dict(
    name="Mixtral-8x7B",
    n_layer=32, hidden=4096, n_head=32, n_kv=8, d_head=128,
    d_ff=14336, vocab_size=32000, context=32_768,
    n_experts=8, top_k=2, capacity_factor=1.0,
    total_params=46_700_000_000, activated_params=12_900_000_000,
    routing="GShard-style top-2 + aux loss 0.02",
)

MIXTRAL_8X22B = dict(
    name="Mixtral-8x22B",
    n_layer=56, hidden=6144, n_head=48, n_kv=8, d_head=128,
    d_ff=16384, vocab_size=32000, context=65_536,
    n_experts=8, top_k=2,
    total_params=141_000_000_000, activated_params=39_000_000_000,
    routing="GShard-style top-2 + aux loss",
)


def main() -> None:
    for cfg in [MIXTRAL_8X7B, MIXTRAL_8X22B]:
        print(f"\n=== {cfg['name']} ===")
        for k, v in cfg.items():
            print(f"  {k:<20} = {v}")


if __name__ == "__main__":
    main()
