"""rmsnorm 参考实现——见 dictation/specs/rmsnorm_spec.py 的接口约定。
风格对齐 interview-prep/src/mlcoding/norm.py 里的 RMSNorm。
"""
from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


if __name__ == "__main__":
    from dictation.checks.rmsnorm_check import check
    check(rmsnorm)
    print("[PASS] rmsnorm_solution 自检通过")
