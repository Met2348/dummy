"""毕业 capstone 公共工具."""
from __future__ import annotations

import torch


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def freeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = False


CKPT_VARIANTS = ["A", "B", "C", "D", "E"]


def variant_desc(v: str) -> str:
    return {
        "A": "Vanilla GPT-2 (124M, baseline)",
        "B": "+ 高质数据 (Cosmopedia)",
        "C": "+ Phi-tiny 架构 (GQA + SwiGLU + RoPE)",
        "D": "+ 长 ctx 扩展 (YaRN)",
        "E": "全部 (= Topic 7 final)",
    }[v]


def gpu_seconds(tok: int, tok_per_s: float) -> float:
    return tok / tok_per_s


if __name__ == "__main__":
    for v in CKPT_VARIANTS:
        print(f"  {v}: {variant_desc(v)}")
