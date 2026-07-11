"""LayerNorm 与 RMSNorm，从零手写并对拍 torch 官方。

面试高频度 ★★★★。得分点：LN 减均值除标准差（对特征维），RMSNorm 省掉去均值、
只按 RMS 缩放（省算力，LLaMA 用它）。eps 加在 sqrt 内。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # 有偏方差，与 torch 一致
        xhat = (x - mu) / torch.sqrt(var + self.eps)
        return xhat * self.gamma + self.beta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


def _self_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 7, 16)

    ln = LayerNorm(16)
    ref = F.layer_norm(x, (16,), ln.gamma, ln.beta, eps=1e-5)
    assert torch.allclose(ln(x), ref, atol=1e-5), (ln(x) - ref).abs().max().item()

    # LN 输出每行零均值单位方差
    y = ln(x)
    assert torch.allclose(y.mean(-1), torch.zeros(3, 7), atol=1e-5)

    rms = RMSNorm(16)
    r = rms(x)
    # RMSNorm 不保证零均值，但保证按 RMS 缩放到 ~单位 RMS
    assert torch.allclose(r.pow(2).mean(-1).sqrt(), torch.ones(3, 7), atol=1e-3)

    # 反传通
    x2 = x.clone().requires_grad_(True)
    ln(x2).sum().backward()
    assert torch.isfinite(x2.grad).all()
    print("[PASS] norm: LayerNorm 对拍官方 + RMSNorm 单位RMS + 反传")


if __name__ == "__main__":
    _self_test()
