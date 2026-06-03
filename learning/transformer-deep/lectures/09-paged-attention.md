# L09 · PagedAttention — vLLM KV Cache 分页

> 20 slides | 60 min | Transformer Deep 第 9 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 理解 KV cache 显存碎片化问题
2. 掌握 PagedAttention block table 概念
3. 知道 vLLM 高吞吐的根本原因

---

## Slide 1 · KV cache 显存碎片

```
请求 1: 2k context → 预分配 2k slot
请求 2: 4k context → 4k slot
请求 1 结束: 2k slot 空，但其他请求需 8k → 碎片
```

传统：分配最大 max_seq_len → **巨大浪费**。

---

## Slide 2 · 分页思路

```
KV cache 分为固定大小 block (如 16 token)
每个 sequence 持有一个 block table 指向多个 block
sequence 不需要连续显存
```

类似 OS 虚拟内存分页。

---

## Slide 3 · block table

```
seq 0: [b0, b3, b7, b9]    # 64 token
seq 1: [b1, b2]             # 32 token
seq 2: [b4, b5, b6, b8]     # 64 token

block 池:  [used:0,1,2,3,...]
```

PA 内核读取 block table，访问对应 block 的 K, V。

---

## Slide 4 · 显存利用率

```
传统:   30-40% 利用率
PA:     ≥ 90%
```

→ 同 GPU 显存可跑 2-3× 更多并发请求。

---

## Slide 5 · prefix sharing

多请求共享 system prompt：

```
seq 0: [shared_b0, shared_b1, my_b9]
seq 1: [shared_b0, shared_b1, my_b12]
```

同一 system prompt block 引用计数共享 → 大幅省内存。

---

## Slide 6 · copy-on-write

beam search 多分支时，分支前共享 block，写新 token 时复制。

```
beam 0: [b0, b1, b2 (full)]   write → 新 block b5
beam 1: [b0, b1, b2 (full)]   write → 新 block b6
```

---

## Slide 7 · PagedAttention kernel

```python
def paged_attn(q, K_block_table, V_block_table, block_size, ...):
    # q: 1 token Q
    # 对每 seq 按 block table 读 K, V
    for block_id in block_table:
        K_block = K_pool[block_id]      # (block_size, h_kv, d_head)
        scores = q @ K_block.T
        # online softmax 累加
```

实际 vLLM 用 CUDA 写 fused kernel。

---

## Slide 8 · 性能

```
vanilla:   1k tokens/s
+ FA:       4k tokens/s
+ PA:       16k tokens/s
+ continuous batching: 24k
```

PagedAttention + continuous batching = vLLM 32k tokens/s on A100。

---

## Slide 9 · continuous batching

```
传统 batch: 所有 seq 等齐才 step
continuous: 每 seq 独立 step，谁先 done 谁出
            → GPU 利用率 +30%
```

PA 让 continuous batching 高效（不同长度 seq 同 batch 不浪费）。

---

## Slide 10 · 教学版伪代码

```python
class KVPool:
    def __init__(self, n_blocks, block_size, n_layer, h_kv, d_head):
        shape = (n_layer, 2, n_blocks, block_size, h_kv, d_head)
        self.pool = torch.zeros(shape, ...)
        self.free = list(range(n_blocks))

    def alloc_block(self) -> int:
        return self.free.pop(0)

    def free_block(self, b: int):
        self.free.append(b)
```

详见 `src/paged_attn_demo.py`。

---

## Slide 11 · sequence 类

```python
class Seq:
    def __init__(self, pool, ...):
        self.pool = pool
        self.block_table: list[int] = []
        self.length = 0

    def append(self, k_new, v_new):
        if self.length % block_size == 0:
            self.block_table.append(self.pool.alloc_block())
        # 把 k_new, v_new 写入 pool[ block_table[-1] ]
        self.length += 1
```

---

## Slide 12 · 释放

```
sequence done → pool.free_block(b) for b in block_table
```

引用计数（prefix sharing）：

```python
def free_block(self, b):
    self.ref[b] -= 1
    if self.ref[b] == 0:
        self.free.append(b)
```

---

## Slide 13 · vLLM 集成层

```
1. scheduler 决定哪些 seq run
2. KVPool 分配 block
3. PagedAttention kernel 调用
4. 输出 token append 入 block
```

vLLM 0.1 已开源（2023.06），现版本 0.7+。

---

## Slide 14 · block_size 选择

```
小 (8-16): 碎片小，scheduler 开销大
大 (64-128): 碎片大，scheduler 简单
```

vLLM 默认 16。Llama 长上下文用 32。

---

## Slide 15 · 与 FA 接口

```python
from flash_attn import flash_attn_with_kvcache
out = flash_attn_with_kvcache(
    q, K_cache, V_cache, cache_seqlens=...
)
```

flash-attn 后期 API 已支持 paged cache 接口。

---

## Slide 16 · KV cache 量化

```
fp16 cache → int8 → 2× 内存
int8 → fp4 (Hopper+) → 4×
```

PagedAttention 兼容 int8 cache（v0.5+）。

---

## Slide 17 · CPU offload

```
hot blocks 在 GPU
cold blocks 在 CPU pinned memory
swap-in/swap-out on demand
```

vLLM 0.6+ 支持，长文档场景实用。

---

## Slide 18 · 多 GPU 分页

```
tensor parallel: K, V 沿 h_kv 切分
pipeline parallel: 不同层在不同 GPU
PagedAttention 对每层独立维护 pool
```

L11 / 专题 6 详讲。

---

## Slide 19 · 实务陷阱

```
block_size 与 attention pattern 不齐 → 性能差
prefix 共享识别错误 → cache miss
GPU 显存预算估错 → OOM
```

vLLM 文档 / SGLang 文档专门讲。

---

## Slide 20 · 课后思考

1. block_size 16 vs 64 实测差几倍？
2. prefix sharing 在 chat API 场景能省多少？
3. continuous batching 与 PA 解耦还是必绑？
4. paged 与 ring attention 能组合吗？

---

## 参考

- Kwon 2023 (PagedAttention / vLLM)
- Yu 2022 (Orca / continuous batching)
- vLLM GitHub
