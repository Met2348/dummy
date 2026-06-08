from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from prefix_tuning_original_minimal import (  # noqa: E402
    PrefixConfig,
    attention_weights,
    build_layout,
    causal_reach,
    prefix_kv_shape,
    saved_prefix_parameters,
    summarize,
)


def test_prefix_kv_shape_and_storage() -> None:
    cfg = PrefixConfig(layers=24, hidden_dim=1024, num_heads=16, prefix_len=5)

    assert prefix_kv_shape(cfg) == (24, 2, 16, 5, 64)
    assert saved_prefix_parameters(cfg) == 245_760


def test_prefix_reaches_input_but_infix_does_not() -> None:
    prefix_layout = build_layout("prefix", prefix_len=2, x_len=3, y_len=2)
    infix_layout = build_layout("infix", prefix_len=2, x_len=3, y_len=2)

    assert "P" in causal_reach(prefix_layout)[2]
    assert "P" not in causal_reach(infix_layout)[0]
    assert "P" in causal_reach(prefix_layout)[len(prefix_layout) - 1]
    assert "P" in causal_reach(infix_layout)[len(infix_layout) - 1]


def test_prefix_key_can_dominate_attention() -> None:
    weights = dict(
        attention_weights(
            query=[1.0, 0.0],
            keys=[[0.95, 0.0], [0.2, 0.1], [0.0, 0.9]],
            labels=["prefix_key", "x_key_1", "x_key_2"],
        )
    )

    assert weights["prefix_key"] > weights["x_key_1"]
    assert weights["prefix_key"] > weights["x_key_2"]


def test_summary_contract() -> None:
    result = summarize()

    assert result["kv_shape"] == (24, 2, 16, 5, 64)
    assert result["prefix_100_tasks"] < result["full_finetune_100_tasks"]


if __name__ == "__main__":
    test_prefix_kv_shape_and_storage()
    test_prefix_reaches_input_but_infix_does_not()
    test_prefix_key_can_dominate_attention()
    test_summary_contract()
    print("[PASS] prefix original minimal tests")
