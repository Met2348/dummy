"""Qwen3-MoE 系列架构 summary."""
CONFIGS = {
    "Qwen3-A3B": dict(
        n_layer=48, hidden=2048, n_head=32, n_kv=4, d_head=64,
        d_ff_routed=1024, vocab_size=151_936, context=131_072,
        n_experts=128, top_k=8,
        total_params=30_000_000_000, activated_params=3_000_000_000,
        routing="GShard top-8 + aux loss",
    ),
    "Qwen3-235B": dict(
        n_layer=94, hidden=4096, n_head=64, n_kv=8, d_head=64,
        d_ff_routed=2048, vocab_size=151_936, context=131_072,
        n_experts=128, top_k=8,
        total_params=235_000_000_000, activated_params=22_000_000_000,
        routing="GShard top-8 + aux loss",
    ),
}


def main() -> None:
    for name, cfg in CONFIGS.items():
        print(f"\n=== {name} ===")
        for k, v in cfg.items():
            print(f"  {k:<20} = {v}")


if __name__ == "__main__":
    main()
