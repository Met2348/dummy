"""默写目标：rope_apply —— 手写 RoPE 旋转位置编码（闭卷版，LLaMA/GPT-NeoX 风格）。

## 接口约定
    rope_apply(x, positions, theta=10000.0) -> torch.Tensor

- x: 形状 (..., T, d)，d 必须为偶数（RoPE 把最后一维两两/前后对半分组做二维旋转）。
- positions: 形状 (T,) 的整数或浮点张量，给出 x 在序列维上每个位置对应的"绝对位置下标"。
  注意：positions 不要求从 0 开始连续，也不要求等于 arange(T)——函数必须直接用传入的
  positions 数值算角度，不能在函数内部自己用 arange(T) 覆盖它（这是本题的常见错误，
  会导致"验证相对位置不变性"的检验失败，因为检验会传入不连续/不同起点的 positions）。
- theta: 频率基数，默认 10000.0（与原始 RoPE/LLaMA 一致）。
- 返回：与 x 同形状，对最后一维做旋转后的结果。

## 数学约定（rotate_half 风格）
- 记 half = d // 2。频率 inv_freq[i] = theta^(-2i/d)，i = 0..half-1。
- 角度 angle[t, i] = positions[t] * inv_freq[i]，形状 (T, half)。
- cos/sin 各自把 (T, half) 在最后一维复制拼接成 (T, d)：cos = cat([cos(angle), cos(angle)])。
- rotate_half(x)：把 x 最后一维前一半 x1、后一半 x2 对调并取负：cat([-x2, x1])。
- 输出 = x * cos + rotate_half(x) * sin，cos/sin 需要广播到 x 的所有前置维度
  （x 的形状可以是 (T,d)、(B,T,d)、(B,H,T,d) 等，positions 恒为 (T,)）。

## 禁止使用
- 任何现成 RoPE 实现（如从 transformers 里 import apply_rotary_pos_emb、或调用
  xformers/flash-attn 自带的 rope 接口）。
- 在函数内部用 `torch.arange(T)` 替代/覆盖传入的 `positions`——那样通不过非连续
  positions 的相对位置检验，也不是本题要练的东西（真实 KV cache 场景里 positions
  经常不是从 0 开始的）。

允许：`torch.arange` 用于生成 0..d-2 的频率下标（不是用来生成 positions 本身）、
基本张量算子（cat、view/reshape、广播乘法）。

## 面试常见追问
- 为什么 RoPE 是"旋转"？旋转矩阵的性质是什么（正交、保范数）？
- 证明 <RoPE(q,m), RoPE(k,n)> 只依赖 (m-n)——旋转的可加性从哪来？
- RoPE 相比可学习绝对位置编码/正弦位置编码的优势是什么？
- 长上下文外推时 RoPE 会遇到什么问题（比如超出训练长度后频率没见过），
  业界怎么缓解（NTK-aware scaling、position interpolation）？
"""
from __future__ import annotations

import torch


def rope_apply(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """手写 RoPE。x: (...,T,d)，positions: (T,) -> 返回与 x 同形状的旋转结果。"""
    raise NotImplementedError
