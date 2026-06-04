"""Phi-3.5-MoE / Phi-4-MoE 架构 summary."""
CONFIGS = {
    "Phi-3.5-MoE": dict(
        n_layer=32, hidden=4096, n_head=32, n_kv=8, d_head=128,
        d_ff=6400, vocab_size=32064, context=131_072,
        n_experts=16, top_k=2,
        total_params=61_000_000_000, activated_params=6_600_000_000,
        routing="top-2 + aux loss",
    ),
}


def main() -> None:
    for name, cfg in CONFIGS.items():
        print(f"\n=== {name} ===")
        for k, v in cfg.items():
            print(f"  {k:<20} = {v}")


if __name__ == "__main__":
    main()
