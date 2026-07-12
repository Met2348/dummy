"""transformer_block_forward 参考实现——见 dictation/specs/transformer_block_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/transformer_block.py 里 TransformerBlock.forward
的两行残差公式，这里改成函数式（attn_fn/ffn_fn/norm1/norm2 都是外部传入的可调用对象）。
"""
from __future__ import annotations

from typing import Callable

import torch


def transformer_block_forward(
    x: torch.Tensor,
    attn_fn: Callable[[torch.Tensor], torch.Tensor],
    ffn_fn: Callable[[torch.Tensor], torch.Tensor],
    norm1: Callable[[torch.Tensor], torch.Tensor],
    norm2: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x = x + attn_fn(norm1(x))
    x = x + ffn_fn(norm2(x))
    return x


if __name__ == "__main__":
    from dictation.checks.transformer_block_check import check
    check(transformer_block_forward)
    print("[PASS] transformer_block_solution 自检通过")
