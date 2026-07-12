"""默写目标：transformer_block_forward —— 手写预归一化(pre-norm) Transformer block（闭卷版）。

## 接口约定
    transformer_block_forward(x, attn_fn, ffn_fn, norm1, norm2) -> torch.Tensor

- x: 形状 (..., d_model)，例如 (B, T, D)。
- attn_fn: 可调用对象，`attn_fn(t) -> t`（同形状），代表注意力子层——本题把它当黑盒，
  内部是否做了causal mask、多头拆分都与本函数无关，只管把它套在正确的位置上。
- ffn_fn: 可调用对象，`ffn_fn(t) -> t`（同形状），代表前馈子层，同样当黑盒。
- norm1, norm2: 可调用对象，`norm(t) -> t`（同形状），分别是 attn 前、ffn 前的归一化
  （LayerNorm/RMSNorm 都行，本函数不关心具体是哪种）。
- 返回：与 x 同形状。

## 数学约定（pre-norm，两处残差）
    x = x + attn_fn(norm1(x))
    x = x + ffn_fn(norm2(x))
    return x

注意"pre-norm"里 norm 是加在**送进子层之前**，子层的输出直接和**原始（未归一化）x**
相加做残差——不是先算 `attn_fn(x) `再对 `x + attn_fn(x)` 做归一化（那是 post-norm，
两种架构在数值上完全不同，post-norm 训练深层网络更容易发散，这也是本题反复强调
"pre-norm"三个字的原因）。

## 禁止使用
- post-norm 写法：`x = norm1(x + attn_fn(x))`。
- 跳过残差直接覆盖：`x = attn_fn(norm1(x))`（丢了 `+ x`）。
- 跳过归一化直接把子层套在原始 x 上：`x = x + attn_fn(x)`（丢了 `norm1`）。
- 给 attn_fn / ffn_fn 的输入或输出做任何本函数职责之外的额外处理（比如自己再加
  dropout、再做一次缩放）——这些都应该封装在 attn_fn/ffn_fn/norm1/norm2 内部，
  本函数只负责拼接顺序。

## 面试常见追问
- 为什么 pre-norm 比 post-norm 更利于训练深层 Transformer？（残差主干上没有被
  归一化"打断"，梯度可以更直接地传回浅层）
- 如果 attn_fn 和 ffn_fn 都恒等于 0（比如权重全零的极端情况），输出应该是什么？
  这条性质在调试"残差连接有没有接对"时有什么用？
- GPT-2 之后的主流大模型（LLaMA/GPT-NeoX 等）用的是 pre-norm 还是 post-norm？
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
    """手写 pre-norm transformer block：x = x + attn_fn(norm1(x))；x = x + ffn_fn(norm2(x))。"""
    raise NotImplementedError
