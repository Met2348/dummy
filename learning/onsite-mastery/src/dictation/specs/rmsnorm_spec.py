"""默写目标：rmsnorm —— 手写 RMSNorm（闭卷版）。

## 接口约定
    rmsnorm(x, weight, eps=1e-6) -> torch.Tensor

- x: 形状 (..., dim)。
- weight: 形状 (dim,)，逐通道缩放（LLaMA 里的可学习 gamma）。
- eps: 加在 sqrt 内部，防止除零/数值不稳定。
- 返回：形状与 x 相同。

## 数学约定
    rms = sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
    out = x / rms * weight

注意三个容易做错的细节：
1. eps 加在 **sqrt 里面**（对均方值加），不是加在 sqrt 结果上（`sqrt(mean(x**2)) + eps`
   是错的——那样在 x 全 0 时仍会除以 0）。
2. RMSNorm **不减均值**（这是它和 LayerNorm 的核心区别，省了一次 reduce，LLaMA/Qwen
   等模型用它替代 LayerNorm 就是为了这点算力）。
3. weight 是逐元素乘在归一化之后，不是在 rms 计算之前就乘进 x 里。

## 禁止使用
- `torch.nn.RMSNorm`（2.4+ 版本 torch 自带）
- `torch.nn.functional.rms_norm`
- `torch.nn.functional.normalize`（那是 L2 归一化到单位范数，不是 RMS 归一化，
  两者只差一个 sqrt(dim) 的缩放因子，容易被面试官追问区别）

允许：基本张量算子（`.pow(2)` / `**2`、`.mean(...)`、`torch.sqrt`、逐元素乘除）。

## 面试常见追问
- RMSNorm 和 LayerNorm 差在哪？为什么 LLaMA 选 RMSNorm？
- eps 应该加在哪一步？加错位置会有什么后果（x 全零或接近零时的数值行为）？
- RMSNorm 的梯度里，weight 和 x 各自怎么求导？（可以口头推一遍链式法则）
"""
from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """手写 RMSNorm。x: (...,dim)，weight: (dim,) -> 返回与 x 同形状。"""
    raise NotImplementedError
