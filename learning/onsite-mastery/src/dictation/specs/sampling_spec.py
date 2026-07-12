"""默写目标：sample_top_k_top_p —— 手写 top-k + top-p(nucleus) 组合采样的概率分布构造（闭卷版）。

## 接口约定
    sample_top_k_top_p(logits, top_k, top_p, temperature) -> torch.Tensor

- logits: 形状 (..., V) 的原始未归一化分数（最后一维是词表维）。
- top_k: int。`top_k <= 0` 或 `top_k >= V` 表示**不做 top-k 过滤**。否则只保留
  logits 最大的 top_k 个候选，其余全部置为不可采样。
- top_p: float，取值范围 (0, 1]。`top_p >= 1.0` 表示**不做 top-p 过滤**。否则按概率
  从大到小累加，保留"累计概率恰好 >= top_p"的最小前缀集合（nucleus），其余置为不可采样。
- temperature: float，必须 > 0。`logits / temperature` 在做任何过滤之前先应用
  （temperature < 1 让分布更尖锐，> 1 更平坦）。
- 返回：与 logits 同形状的**概率分布**（不是采样出的下标！不要调用任何随机采样函数）。
  未被保留的候选，返回概率必须精确为 0；被保留的候选概率必须在最后一维上重新归一化到和为 1。

## 处理顺序（必须按此顺序，顺序错了数值会对不上）
    1. logits = logits / temperature
    2. 应用 top_k 过滤（不满足条件的置 -inf）
    3. 在 top_k 过滤后的结果上应用 top_p 过滤（不满足条件的置 -inf）
    4. 对最终保留的候选做 softmax 归一化（不允许在被过滤的 -inf 位置输出非零概率）

## 边界条件
- top_k=1 时必须退化为 greedy：输出是一个只在 argmax 位置为 1.0、其余全为 0 的
  one-hot 分布，且不受 top_p 影响（top_k 过滤后只剩 1 个候选，top_p 无论多小
  都不能把唯一剩下的候选也过滤掉——nucleus 必须至少保留 1 个）。
- top_p 过滤：若某个候选是"让累计概率首次达到/超过 top_p"的那个，它本身必须被保留
  （nucleus 的定义是"最小的、累计概率 >= top_p 的前缀集合"，不是"严格小于 top_p 的集合"）。

## 禁止使用
- `torch.multinomial` / `torch.distributions.*` / 任何真正做随机采样的 API——
  本题只要求返回确定性的概率张量，方便对拍验证；真去随机采样反而没法写断言。
- `torch.nn.functional.gumbel_softmax`。
- 现成的 `transformers` LogitsWarper（TopKLogitsWarper / TopPLogitsWarper 等）。

允许：`torch.topk`、`.sort(...)`、`.cumsum(...)`、`.softmax(...)`、`masked_fill`。

## 面试常见追问
- top-k 和 top-p 分别解决什么问题？为什么工业界常把两者叠加使用？
- top_p 为什么要"先排序再累加"而不能直接在原始词表顺序上累加？
- temperature 趋近于 0 时的极限行为是什么？（退化为 greedy，对应 top_k=1）
- 如果 top_k 过滤和 top_p 过滤的顺序颠倒，会导致什么不同的结果？
"""
from __future__ import annotations

import torch


def sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    """手写 top-k+top-p 概率分布构造。logits: (...,V) -> 返回同形状、按最后一维归一化的概率。"""
    raise NotImplementedError
