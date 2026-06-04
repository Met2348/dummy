# L02 · Naive KV Cache 及其碎片问题

## 1 · 朴素方案
预留 `[B, max_len, n_kv_heads, d_h]` 张量，每步写入位置 `pos`。

```python
kv = torch.empty(B, max_len, n_kv_heads, d_h, dtype=torch.float16)
kv[b, pos] = new_kv
```

## 2 · 浪费在哪
设 `max_len=2048`，实际请求平均 `512`：
- **internal fragmentation**: 1536 token slot 永远不用 ⇒ **75% 浪费**
- **external fragmentation**: 不同长度请求间不能复用 slot
- **over-reservation**: max_len 估错 → 直接 truncate

## 3 · 用真实数字感受
Qwen-7B fp16, n_kv_heads=8, d_h=128:
- 1 token KV / layer = 2·8·128·2 = 4 KB
- 32 层 = **128 KB / token**
- `max_len=2048` 预留 = 256 MB/request
- 5090 24 GB 最多 ~80 个请求并发

## 4 · padding 的二次浪费
batch decode 时所有请求必须对齐：
- 短请求要 pad → 算 padding 位置的 attention 也得做（或加 mask 但仍占算力）
- attention mask 内存 = `B² · S²`，1k seq, batch 32 → 1 GB attention mask

## 5 · 关键观察
> KV cache 是**按 token 写入**的，但**按 slot 预留**的。两者错配 ⇒ 碎片必然。

## 6 · 修法思路（提示）
- 不预留：写到哪算到哪 → **动态分配**
- 不连续：物理可以非连续 → **块映射**（PagedAttention 来源）

## 7 · 度量碎片
利用率 = `实际使用 / 预留`:
```
util = sum(actual_len) / (B * max_len)
```
naive 平均利用率 25-40%，PagedAttention 可达 95%+。

## 8 · 实现要点（看 [naive_kv.py](../src/naive_kv.py)）
- `NaiveKvPool.alloc(B, max_len)` 预留
- `.write(b, pos, k, v)` 写入
- `.utilization()` 度量
