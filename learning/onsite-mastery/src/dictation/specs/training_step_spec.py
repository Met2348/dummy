"""默写目标：training_step —— 手写一次完整训练步（闭卷版）。

## 接口约定
    training_step(model, optimizer, batch, loss_fn, clip_norm) -> float

- model: `torch.nn.Module`，参数已经绑定给 optimizer。
- optimizer: `torch.optim.Optimizer` 的实例，构造时已经传入了 `model.parameters()`。
- batch: 二元组 `(x, y)`，`x` 是模型输入，`y` 是监督目标。
- loss_fn: 可调用对象 `loss_fn(pred, y) -> scalar tensor`。
- clip_norm: float，梯度裁剪的最大范数（传给 `torch.nn.utils.clip_grad_norm_`）。
- 返回：**这一步的 loss 数值**（Python float，即裁剪/更新之前、基于当前参数算出来的
  loss.item()）——不是更新参数之后重新算的 loss。

## 必须严格按这个顺序（四步循环，一步都不能漏/不能错序）
    1. optimizer.zero_grad()                                   # 清掉上一步遗留的梯度
    2. pred = model(x); loss = loss_fn(pred, y)                 # 前向 + 算 loss
    3. loss.backward()                                          # 反传
    4. torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)   # 裁剪
    5. optimizer.step()                                         # 用（裁剪后的）梯度更新参数
    6. return loss.item()

## 为什么这四步顺序不能乱（面试官最爱在这一题上追问的地方）
- 忘记 `zero_grad()`：PyTorch 的梯度默认是**累加**的，不清零会让上一步的梯度叠加到这一步，
  几步之后梯度会越滚越大，训练直接发散——本题的检验会故意在调用前塞一份"脏"梯度进
  `.grad`，验证你的实现确实把它清掉了而不是累加。
- `clip_grad_norm_` 必须放在 `backward()` 之后、`step()` 之前：裁剪操作的对象是
  `.grad` 张量本身（原地缩放），裁剪太早（backward 前）没有梯度可裁，裁剪太晚
  （step 之后）已经来不及影响这一步的更新。
- 返回值是 backward **之前**算出来的 loss（当前参数下的 loss），不是更新参数之后
  重新跑一遍前向算出来的"新 loss"——这是两个数值上不同的东西，很多人会记混。

## 禁止使用
- `torch.no_grad()` 包住前向计算（那样 backward 会报错，因为计算图没建起来）。
- `torch.nn.utils.clip_grad_value_`（那是逐元素裁剪，不是按整体范数裁剪，是完全不同的操作）。
- 手写"重新计算梯度范数再手动缩放 `.grad`"来替代 `clip_grad_norm_`——直接调用官方 API 即可，
  本题练的是训练循环的骨架顺序，不是重新发明裁剪算法。

## 面试常见追问
- 为什么梯度默认是累加的？这个设计在什么场景下有用（比如梯度累积模拟大 batch）？
- 梯度裁剪解决的是什么问题？裁的是"范数"还是"每个元素的值"，两者的区别是什么？
- 如果 loss 出现 NaN，你会在这四步的哪个位置插入检查/怎么处理？
"""
from __future__ import annotations

from typing import Callable

import torch


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    clip_norm: float,
) -> float:
    """手写一次训练步：zero_grad -> forward -> loss -> backward -> clip -> step。返回 loss 数值。"""
    raise NotImplementedError
