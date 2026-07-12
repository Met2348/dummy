"""lora_forward 参考实现——见 dictation/specs/lora_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/lora.py（那里是 nn.Module 版本，这里改成函数式）。
"""
from __future__ import annotations

import torch


def lora_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    rank: int,
) -> torch.Tensor:
    base_out = x @ base_weight.T
    lora_out = x @ lora_A @ lora_B
    return base_out + (alpha / rank) * lora_out


if __name__ == "__main__":
    from dictation.checks.lora_check import check
    check(lora_forward)
    print("[PASS] lora_solution 自检通过")
