# L03 · PagedAttention（vLLM SOSP'23）

## 1 · 核心类比：虚拟内存
| 操作系统 | vLLM |
|---------|------|
| 进程虚拟地址空间 | request 的 logical KV 序列 |
| 物理页 (4 KB) | KV block (16 token) |
| 页表 | block table |
| 缺页中断 | block 分配 |
| 共享页 | prefix caching |

## 2 · 数据结构
```python
# 每个 block = 16 token (或可调)
block: Tensor[BLOCK_SIZE, n_kv_heads, d_h]

# 全局物理 block 池
physical_blocks: Tensor[N_BLOCKS, BLOCK_SIZE, n_kv_heads, d_h]
free_block_ids: deque[int]

# 每个 request 的 block table
req.block_table: List[int]   # logical -> physical block id
```

## 3 · 写入流程
1. token `t` 来了，`logical_idx = t // BLOCK_SIZE`
2. 若 `block_table[logical_idx]` 不存在 → 从 `free_block_ids.pop()` 拿一个
3. 写到 `physical_blocks[phys_id, t % BLOCK_SIZE]`

## 4 · attention 计算
传统:
```python
q @ K[:t].T   # 连续 K
```
Paged:
```python
for blk_id in block_table:
    K_blk = physical_blocks[blk_id]
    out += q @ K_blk.T
```
**伪连续访问** — kernel 内部 gather 物理块。

## 5 · 碎片彻底消失
- 没有 internal frag：最后一块最多浪费 `BLOCK_SIZE - 1` token
- 没有 external frag：所有 block 等大，可自由复用
- 平均利用率 ≥ 95%

## 6 · 三大附加能力
- **prefix sharing**: 多请求 logical[0]→ 同一 physical block
- **beam search / parallel sample**: fork 时只复制 block table（COW）
- **swap to CPU**: 不活跃 block 换出，激活时换入

## 7 · BLOCK_SIZE 取舍
| BLOCK_SIZE | 优 | 劣 |
|-----------|----|----|
| 16 | 细粒度共享 | 表大 |
| 64+ | 表小 / kernel 快 | 共享浪费多 |

vLLM 默认 16；SGLang 默认 1 (radix tree)。

## 8 · 一句话
> PagedAttention = **给 KV cache 装上 OS 级虚拟内存管理**。

## 9 · 实现：[paged_kv.py](../src/paged_kv.py)
- `PagedKvPool` 物理池
- `BlockTable` per-request
- `allocate(req, n_new_tokens)` 增量分配
- `free(req)` 归还到 `free_block_ids`
