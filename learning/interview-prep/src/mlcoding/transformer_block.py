"""预归一化 Transformer decoder block，从零组装。

面试高频度 ★★★★。得分点：pre-norm（x + attn(norm(x))）比 post-norm 更稳、深层可训；
MLP 是 4x 升维 + GELU + 降维；两处残差。复用同目录的 MHA 与 LayerNorm。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from mlcoding.attention import MultiHeadAttention   # noqa: E402
from mlcoding.norm import LayerNorm                 # noqa: E402


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block：残差包着 (LN->MHA) 和 (LN->MLP)。"""

    def __init__(self, d_model: int, n_heads: int, mult: int = 4):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, mult)

    def forward(self, x, causal: bool = True):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.mlp(self.ln2(x))
        return x


def _self_test() -> None:
    torch.manual_seed(0)
    blk = TransformerBlock(d_model=32, n_heads=4)

    # 1) 形状 + 反传（必须在任何 in-place 改参数之前做）
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = blk(x, causal=True)
    assert y.shape == (2, 7, 32)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()

    # 2) 因果性：block 内注意力不看未来（仍用未被污染的 blk）
    x2 = torch.randn(1, 5, 32)
    full = blk(x2, causal=True)
    x2b = x2.clone()
    x2b[:, 2:] += 10.0
    partial = blk(x2b, causal=True)
    assert torch.allclose(full[:, 0], partial[:, 0], atol=1e-4), "block 泄露未来"

    # 3) 残差直通：另起一个 block 把所有权重清零，输出应精确等于输入
    zblk = TransformerBlock(d_model=32, n_heads=4)
    with torch.no_grad():
        for p in zblk.parameters():
            p.zero_()
    xin = torch.randn(2, 4, 32)
    assert torch.allclose(zblk(xin), xin, atol=1e-5), "残差主干未直通"
    print("[PASS] transformer_block: 形状 + 反传 + 因果 + 残差直通")


if __name__ == "__main__":
    _self_test()
