# L07 · Prefix Caching（vLLM 2024 / SGLang Radix）

## 1 · 观察
LLM 服务中大量请求共享 prefix：
- 系统 prompt（所有请求都有的 system message，2k token）
- few-shot 示例（多请求复用同一组）
- agent 多轮（每轮新轮共享前几轮）

朴素：每个请求都从头 prefill → **重复计算**。

## 2 · 解：prefix hash + block reuse
1. 把 prompt 按 BLOCK_SIZE 切块
2. 每块算 hash(token_ids)
3. 全局 hash → physical block id 字典
4. 命中 → 跳过 prefill 该块，直接 mount 到 block table

## 3 · 数据结构
```python
prefix_cache: Dict[hash, (block_id, refcount)]
```

写时：
- 已存在 → refcount++ → mount 到当前 request block_table
- 不存在 → alloc 物理 block + 写入 + 注册 hash

## 4 · 命中率（真实数字）
| 场景 | 命中率 |
|------|-------|
| 单一 system prompt 服务 | 95% |
| 多 system prompt | 60% |
| agent 多轮 | 70% |
| 通用 chat | 10-20% |

## 5 · 失效策略
- LRU：最久不用的 prefix 块先 evict
- 注意 refcount > 0 不能 evict
- vLLM 用 `allocated_block_ids` + `last_access_time`

## 6 · 与 PagedAttention 的天然协同
- block 是 paged 的基础单元 → 共享只需共享 block_id
- 多请求 attention 时 fetch 同一物理块 → 不冲突（只读）

## 7 · SGLang RadixAttention
- 用 **radix tree** 而非平面 hash → 共享部分前缀也算
- 例如 "请回答以下问题：1+1=?" 和 "请回答以下问题：2+2=?" 共享前 9 token

## 8 · vLLM 实现
`--enable-prefix-caching`（默认开启 0.7+）

## 9 · 实现：[prefix_cache.py](../src/prefix_cache.py)
- `PrefixCache` 平面 hash 版（教学）
- 命中率统计
- LRU eviction
