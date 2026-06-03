"""RMSNorm — Root Mean Square Norm (Zhang & Sennrich 2019)."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在 fp32 算 RMS，避免 bf16 精度问题
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.gamma * xf * rms).to(in_dtype)


def compare_with_pytorch():
    """与 PyTorch 2.5 内置 nn.RMSNorm 对照."""
    import torch
    torch.manual_seed(0)
    d = 64
    x = torch.randn(2, 8, d)
    my = RMSNorm(d)
    if hasattr(nn, "RMSNorm"):
        ref = nn.RMSNorm(d, eps=1e-6)
        # 对齐 gamma init = 1
        ref.weight.data.fill_(1.0)
        my.gamma.data.fill_(1.0)
        out_my = my(x)
        out_ref = ref(x)
        diff = (out_my - out_ref).abs().max().item()
        print(f"my vs torch.nn.RMSNorm  max diff: {diff:.2e}")
    else:
        print("PyTorch 2.5+ 才内置 nn.RMSNorm，跳过对照")


if __name__ == "__main__":
    compare_with_pytorch()
