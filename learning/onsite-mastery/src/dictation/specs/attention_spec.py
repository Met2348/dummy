"""默写目标：causal_attention —— 手写因果自注意力（闭卷版）。

## 接口约定
    causal_attention(q, k, v) -> torch.Tensor

- q, k, v: 形状均为 (B, H, T, d) 的 float 张量（自注意力：Tq == Tk == T，同一批 head 数 H，
  head 维 d）。约定 v 的最后一维也是 d（不单独区分 d_v，简化默写）。
- 返回：形状 (B, H, T, d)，对每个 batch/head/query 位置 i，只允许注意 key 位置 j <= i
  （因果/自回归掩码），softmax 权重在被屏蔽的位置上必须严格为 0。

## 边界条件 / 数值要求
- 必须除以 sqrt(d) 做缩放（否则大 d 时 softmax 会因数值过大饱和、梯度消失）。
- softmax 必须自己写数值稳定版本：对每行减去该行最大值再 exp、再除以行和。
  不允许调用官方 softmax 实现，见下方"禁止使用"。
- 掩码要加在 softmax 之前（用 -inf 或等效的极小值），不能在 softmax 之后再乘 0/1 掩码
  ——那样虽然输出看起来对，但反传会把梯度错误地路由到被掩位置，且不满足"严格为0"的检验。

## 禁止使用（这是考点，不是可选项）
- `torch.nn.functional.scaled_dot_product_attention`
- `torch.nn.functional.softmax` / `Tensor.softmax(...)`（必须手写：减max、exp、按行归一化）
- `torch.nn.MultiheadAttention` 或任何现成注意力封装

允许：`@` / `torch.matmul`、`torch.exp`、`.max(...)`、`.sum(...)`、基本索引与广播、
`torch.tril`/`torch.ones` 等构造掩码用的张量算子。

## 面试常见追问
- 为什么除以 sqrt(d)？（点积方差随 d 增长，除以 sqrt(d) 让 softmax 输入方差恒定）
- 为什么减 max 再 exp？（数值稳定，防止 exp 上溢；减 max 不改变 softmax 结果，因为
  分子分母同时乘了 exp(-max)）
- 掩码为什么必须在 softmax 之前而不是之后？
- 多头是怎么从这个单头函数扩展出去的？（reshape 成 (B,H,T,d)，而不是开 H 份权重）
"""
from __future__ import annotations

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """手写因果自注意力。q,k,v: (B,H,T,d) -> 返回 (B,H,T,d)。"""
    raise NotImplementedError
