"""默写目标：lora_forward —— 手写 LoRA 前向（闭卷版，函数式，不依赖 nn.Module）。

## 接口约定
    lora_forward(x, base_weight, lora_A, lora_B, alpha, rank) -> torch.Tensor

- x: 形状 (..., in_features)。
- base_weight: 形状 (out_features, in_features)——和 `nn.Linear.weight` **同样的约定**
  （即 base 的线性变换是 `x @ base_weight.T`，不是 `x @ base_weight`）。这是最容易踩的坑：
  记混方向会让"lora_B=0 时输出等于 base 原始输出"这条检验第一眼看起来通过（形状凑巧对得上
  会报错，凑不上直接崩），排查起来很费时间。
- lora_A: 形状 (in_features, rank)。
- lora_B: 形状 (rank, out_features)。
- alpha: 缩放用的超参数（int 或 float）。
- rank: 低秩维度（应等于 lora_A.shape[1] == lora_B.shape[0]，这里显式传入是因为
  真实缩放公式就是按传入的 rank 而不是重新从形状里推断——面试时口头讲这个公式，
  被问"rank 从哪来"要能立刻说出是超参不是形状推断，虽然多数实现里两者相等）。

## 数学约定
    base_out = x @ base_weight.T                     # (..., out_features)
    lora_out = x @ lora_A @ lora_B                    # (..., out_features)
    return base_out + (alpha / rank) * lora_out

## 边界条件
- lora_A 或 lora_B 任一为全 0 时，`lora_out` 必须精确为 0，输出必须精确等于 base_out
  （这是 LoRA "训练初始阶段严格等价于原模型"的关键性质，真实初始化里 B 全 0 就是利用了这点）。
- base_weight 在真实场景中是被冻结的（requires_grad=False），只有 lora_A/lora_B 参与训练；
  函数本身不需要处理 requires_grad（那是调用方的事），但函数**不能对 base_weight 做任何
  会引入梯度依赖之外的操作**（比如原地修改 base_weight）。

## 禁止使用
- `torch.nn.Linear` / 任何 `nn.Module` 封装 base 或 lora 的矩阵（本题就是练"不靠模块，
  手写清楚这几个矩阵乘法怎么拼"）。
- 第三方 `peft` 库的 LoRA 实现。

允许：`@` / `torch.matmul`、`.transpose(...)` / `.T`、基本算术。

## 面试常见追问
- 为什么 lora_B 初始化为全 0 而不是 lora_A？（保证训练开始时旁路贡献为 0，
  不打乱预训练权重的初始行为；如果两个都随机初始化，训练一开始模型输出就被扰动）
- alpha/rank 这个缩放系数的作用是什么？调大 rank 时如果不重新调 alpha 会发生什么？
- 为什么只训 A、B 而冻结 base_weight？显存/计算量上省在哪？
- 合并部署时怎么把 LoRA "焼"回 base_weight？（`base_weight + (alpha/rank) * A @ B` 的转置）
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
    """手写 LoRA 前向。x:(...,in)，base_weight:(out,in)，lora_A:(in,r)，lora_B:(r,out)。"""
    raise NotImplementedError
