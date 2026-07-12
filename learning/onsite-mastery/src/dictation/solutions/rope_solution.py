"""rope_apply 参考实现——见 dictation/specs/rope_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/rope.py，但把 cos/sin 构造折进同一个函数，
且直接吃调用方传入的 positions（不在内部用 arange(T) 覆盖）。
"""
from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    d = x.size(-1)
    assert d % 2 == 0, "RoPE 要求最后一维为偶数"

    i = torch.arange(0, d, 2, dtype=torch.float32, device=x.device)   # 0,2,...,d-2
    inv_freq = 1.0 / (theta ** (i / d))                                # (d/2,)

    positions = positions.to(dtype=torch.float32, device=x.device)    # (T,)
    angles = torch.outer(positions, inv_freq)                          # (T, d/2)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)              # (T, d)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)              # (T, d)

    return x * cos + _rotate_half(x) * sin


if __name__ == "__main__":
    from dictation.checks.rope_check import check
    check(rope_apply)
    print("[PASS] rope_solution 自检通过")
