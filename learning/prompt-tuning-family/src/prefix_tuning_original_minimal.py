"""Small, dependency-free examples for the Prefix-Tuning paper.

This file teaches the original Li and Liang mechanism without loading GPT-2.
It focuses on:

- how per-layer prefix key/value tensors are shaped;
- why prefixing can affect both x and y while infixing cannot affect x;
- why saved inference parameters are much smaller than a full model copy;
- how a prefix key changes a causal attention distribution.

Run:
    python learning/prompt-tuning-family/src/prefix_tuning_original_minimal.py
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt


@dataclass(frozen=True)
class PrefixConfig:
    layers: int
    hidden_dim: int
    num_heads: int
    prefix_len: int
    reparam_dim: int = 512

    @property
    def head_dim(self) -> int:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return self.hidden_dim // self.num_heads


def prefix_kv_shape(cfg: PrefixConfig) -> tuple[int, int, int, int, int]:
    """Return the inference-time per-layer KV prefix shape.

    Shape convention:
        (layers, key_or_value, heads, prefix_len, head_dim)

    This is the tensor that can be prepended to past_key_values at inference.
    """
    return (cfg.layers, 2, cfg.num_heads, cfg.prefix_len, cfg.head_dim)


def saved_prefix_parameters(cfg: PrefixConfig) -> int:
    """Parameters kept after training when the reparameterization MLP is dropped."""
    return cfg.layers * 2 * cfg.prefix_len * cfg.hidden_dim


def training_reparam_parameters(cfg: PrefixConfig) -> int:
    """One simple MLP reparameterization parameter count.

    The paper trains a low-dimensional prefix and an MLP, then discards the MLP
    for storage. This count is not meant to reproduce every implementation
    detail exactly; it shows why training-time parameters can be much larger
    than inference-time stored prefix vectors.
    """
    low_prefix = cfg.prefix_len * cfg.reparam_dim
    first = cfg.reparam_dim * cfg.reparam_dim + cfg.reparam_dim
    second_out = cfg.layers * 2 * cfg.hidden_dim
    second = cfg.reparam_dim * second_out + second_out
    return low_prefix + first + second


def full_finetune_parameters(num_lm_params: int, num_tasks: int) -> int:
    """Storage cost for keeping one full fine-tuned LM copy per task."""
    return num_lm_params * num_tasks


def prefix_storage_parameters(cfg: PrefixConfig, num_tasks: int) -> int:
    """Storage cost for one frozen LM plus one prefix per task."""
    return saved_prefix_parameters(cfg) * num_tasks


def causal_reach(layout: list[str]) -> dict[int, set[str]]:
    """For each position, return token categories visible in causal attention."""
    visible: dict[int, set[str]] = {}
    for i in range(len(layout)):
        visible[i] = set(layout[: i + 1])
    return visible


def build_layout(mode: str, prefix_len: int, x_len: int, y_len: int) -> list[str]:
    """Build token category layouts for prefix, infix, and embedding-only cases."""
    if mode == "prefix":
        return ["P"] * prefix_len + ["x"] * x_len + ["y"] * y_len
    if mode == "infix":
        return ["x"] * x_len + ["P"] * prefix_len + ["y"] * y_len
    if mode == "embedding_only":
        return ["P_embed"] * prefix_len + ["x"] * x_len + ["y"] * y_len
    raise ValueError(f"unknown mode: {mode}")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(xs: list[float]) -> list[float]:
    largest = max(xs)
    exps = [exp(x - largest) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def attention_weights(
    query: list[float],
    keys: list[list[float]],
    labels: list[str],
) -> list[tuple[str, float]]:
    """Tiny scaled dot-product attention over a few labeled keys."""
    scale = sqrt(len(query))
    scores = [dot(query, key) / scale for key in keys]
    probs = softmax(scores)
    return list(zip(labels, probs))


def summarize() -> dict[str, object]:
    """Return the facts that the guide points readers to verify."""
    gpt2_medium_like = PrefixConfig(
        layers=24,
        hidden_dim=1024,
        num_heads=16,
        prefix_len=5,
        reparam_dim=512,
    )

    prefix_layout = build_layout("prefix", prefix_len=2, x_len=3, y_len=2)
    infix_layout = build_layout("infix", prefix_len=2, x_len=3, y_len=2)

    query = [1.0, 0.0]
    keys = [
        [0.95, 0.0],  # prefix key aligned with the query
        [0.2, 0.1],   # normal x key
        [0.0, 0.9],   # another normal x key
    ]
    labels = ["prefix_key", "x_key_1", "x_key_2"]

    prefix_last = len(prefix_layout) - 1
    infix_last = len(infix_layout) - 1

    return {
        "kv_shape": prefix_kv_shape(gpt2_medium_like),
        "saved_prefix_params": saved_prefix_parameters(gpt2_medium_like),
        "training_reparam_params": training_reparam_parameters(gpt2_medium_like),
        "full_finetune_100_tasks": full_finetune_parameters(345_000_000, 100),
        "prefix_100_tasks": prefix_storage_parameters(gpt2_medium_like, 100),
        "prefix_first_x_sees": causal_reach(prefix_layout)[2],
        "infix_first_x_sees": causal_reach(infix_layout)[0],
        "last_y_prefix_layout_sees": causal_reach(prefix_layout)[prefix_last],
        "last_y_infix_layout_sees": causal_reach(infix_layout)[infix_last],
        "attention": attention_weights(query, keys, labels),
    }


def _self_test() -> None:
    result = summarize()
    assert result["kv_shape"] == (24, 2, 16, 5, 64)
    assert result["saved_prefix_params"] == 245_760
    assert "P" in result["prefix_first_x_sees"]
    assert "P" not in result["infix_first_x_sees"]
    assert "P" in result["last_y_prefix_layout_sees"]
    assert "P" in result["last_y_infix_layout_sees"]

    attention = dict(result["attention"])
    assert attention["prefix_key"] > attention["x_key_1"]
    assert attention["prefix_key"] > attention["x_key_2"]

    for key, value in result.items():
        print(f"{key}: {value}")
    print("prefix_tuning_original_minimal self-test passed")


if __name__ == "__main__":
    _self_test()
