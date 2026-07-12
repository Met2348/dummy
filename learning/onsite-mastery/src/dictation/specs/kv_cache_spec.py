"""默写目标：kv_cache_step —— 手写 KV Cache 单步增量拼接更新（闭卷版）。

## 接口约定
    kv_cache_step(new_k, new_v, cache_k, cache_v) -> tuple[torch.Tensor, torch.Tensor]

- new_k, new_v: 当前这一步新算出来的 K/V，形状 (B, H, 1, d)（序列维恰好长度 1，
  代表自回归解码时"这一步刚生成的那个 token"）。
- cache_k, cache_v: 到上一步为止累积的历史 K/V，形状 (B, H, T_prev, d)；
  **在第一步时 cache_k/cache_v 传入 None**（还没有任何历史）。
- 返回：拼接后的 (updated_k, updated_v)，形状均为 (B, H, T_prev + 1, d)。

## 边界条件
- 必须处理 cache_k is None（等价于 cache_v is None）的首步情况：此时直接返回
  new_k/new_v 本身（形状 (B,H,1,d)）作为"到目前为止"的缓存，不要试图 cat 一个空张量
  进去（容易在 T_prev=0 维度上手滑写错）。
- 拼接必须发生在序列维（也就是 new_k/cache_k 里那个长度会随步数增长的维度，
  这里是 dim=2，对应 (B,H,T,d) 里的 T），历史在前、新 token 在后。
- **禁止原地修改传入的 cache_k/cache_v**（比如用预分配好的大 buffer 做
  `cache_k[:, :, some_slice, :] = new_k` 然后返回同一个 tensor 的引用/视图）。
  真实面试里这是一个高区分度的追问点：如果调用方在别处还持有旧 cache 的引用
  （比如做 beam search 需要保留多个分支各自的历史缓存快照），原地修改会让
  之前保存的"历史快照"被意外污染。本题的检验会显式验证这一点。

## 禁止使用
- 预分配固定长度 buffer + 原地切片赋值（性能优化范畴，超出本题——本题只要求
  最朴素、最不容易犯"意外共享内存"错误的 `torch.cat` 版本）。
- 用 Python list 攒起来再 `torch.stack`（维度语义不对：那样会在拼接维上多插入
  一个新维度，而不是延长已有的序列维）。

允许：`torch.cat`（沿正确维度）、基本形状判断（`if cache_k is None`）。

## 面试常见追问
- 为什么因果注意力才能用 KV cache，双向 encoder 不行？
- 每步计算复杂度从 O(T) 降到 O(1)（相对当步），但 attention 本身对当步 query
  仍要对全部历史 key 做点积，整段解码的总复杂度是多少？
- 为什么不能对传入的 cache 做原地修改？举一个会出错的具体场景（比如 beam search
  多分支共享前缀缓存时的写时复制语义）。
"""
from __future__ import annotations

import torch


def kv_cache_step(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    cache_k: torch.Tensor | None,
    cache_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单步 KV cache 更新。new_k/new_v: (B,H,1,d)；cache_k/cache_v: (B,H,T_prev,d) 或 None。"""
    raise NotImplementedError
