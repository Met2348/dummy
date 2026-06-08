"""Dependency-free examples for Houlsby adapter tuning.

This file teaches the original "Parameter-Efficient Transfer Learning for NLP"
mechanism without loading a Transformer model.

It focuses on:

- bottleneck adapter parameter counts;
- why near-identity initialization preserves the frozen backbone at step zero;
- how adapter storage scales across many tasks;
- why Houlsby adapters use two adapter modules per Transformer layer.

Run:
    .\\.venv\\Scripts\\python.exe learning\\adapter-tuning-family\\src\\adapter_original_minimal.py
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterConfig:
    layers: int
    hidden_dim: int
    adapter_dim: int
    adapters_per_layer: int = 2


def single_adapter_parameters(hidden_dim: int, adapter_dim: int) -> int:
    """Parameter count for down projection, up projection, and biases."""
    down = hidden_dim * adapter_dim + adapter_dim
    up = adapter_dim * hidden_dim + hidden_dim
    return down + up


def houlsby_adapter_parameters(cfg: AdapterConfig) -> int:
    """Task-specific adapter parameters across all Transformer layers."""
    per_adapter = single_adapter_parameters(cfg.hidden_dim, cfg.adapter_dim)
    return cfg.layers * cfg.adapters_per_layer * per_adapter


def full_finetune_total_params(backbone_params: int, num_tasks: int) -> int:
    """Total storage if every task keeps a full model copy."""
    return backbone_params * num_tasks


def adapter_total_params(
    backbone_params: int,
    cfg: AdapterConfig,
    num_tasks: int,
) -> int:
    """Total storage for one frozen backbone plus one adapter set per task."""
    return backbone_params + num_tasks * houlsby_adapter_parameters(cfg)


def relu(xs: list[float]) -> list[float]:
    return [max(0.0, x) for x in xs]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def adapter_forward(
    x: list[float],
    down: list[list[float]],
    up: list[list[float]],
    down_bias: list[float],
    up_bias: list[float],
) -> list[float]:
    """A tiny bottleneck adapter: x + up(relu(down(x)))."""
    z = add(matvec(down, x), down_bias)
    z = relu(z)
    delta = add(matvec(up, z), up_bias)
    return add(x, delta)


def zero_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def near_identity_demo() -> tuple[list[float], list[float]]:
    """Return input and adapter output under zero-initialized up projection."""
    x = [1.0, -2.0, 0.5, 3.0]
    down = [
        [0.10, -0.20, 0.05, 0.00],
        [0.00, 0.10, -0.10, 0.20],
    ]
    down_bias = [0.0, 0.0]

    # Zero up projection makes the adapter branch output zero.
    up = zero_matrix(rows=4, cols=2)
    up_bias = [0.0, 0.0, 0.0, 0.0]

    return x, adapter_forward(x, down, up, down_bias, up_bias)


def summarize() -> dict[str, object]:
    gpt2_base_like = AdapterConfig(
        layers=12,
        hidden_dim=768,
        adapter_dim=16,
        adapters_per_layer=2,
    )
    bert_large_like = AdapterConfig(
        layers=24,
        hidden_dim=1024,
        adapter_dim=64,
        adapters_per_layer=2,
    )
    x, y = near_identity_demo()

    return {
        "single_adapter_gpt2_r16": single_adapter_parameters(768, 16),
        "houlsby_gpt2_r16": houlsby_adapter_parameters(gpt2_base_like),
        "houlsby_bert_large_r64": houlsby_adapter_parameters(bert_large_like),
        "full_finetune_9_tasks": full_finetune_total_params(330_000_000, 9),
        "adapter_9_tasks": adapter_total_params(330_000_000, bert_large_like, 9),
        "near_identity_input": x,
        "near_identity_output": y,
    }


def _self_test() -> None:
    result = summarize()
    assert result["single_adapter_gpt2_r16"] == 25_360
    assert result["houlsby_gpt2_r16"] == 608_640
    assert result["adapter_9_tasks"] < result["full_finetune_9_tasks"]
    assert result["near_identity_input"] == result["near_identity_output"]

    for key, value in result.items():
        print(f"{key}: {value}")
    print("adapter_original_minimal self-test passed")


if __name__ == "__main__":
    _self_test()
