# L09 · KV-Cache Management

## 30 秒

> Transformer decoder 推理时每 token 都看前所有 token 的 K/V → 不缓存就 O(N²) 重算。
> **KV cache** 把每层 K/V 存内存，新 token 只算自己 + reuse 旧的。

## KV cache 基本

```
Token 1: K1, V1 → cache.add(K1, V1)
Token 2: 算 Q2, 与 cached K1 算 attention, K2,V2 → cache.add
...
Token N: 算 Q_N, attention 用 cached K1..K_{N-1} + 新 K_N
```

显存：每层 (K + V) × seq_len × hidden_dim × dtype。

例：Llama-3-70B + 8k context = ~20 GB KV cache。

## PagedAttention（vLLM 2023）

[Kwon 2023 SOSP] 把 KV cache 当**虚拟内存页**管：

| 类比 | vLLM |
|------|------|
| 物理 RAM | GPU memory |
| 进程地址空间 | 1 request 的 KV cache |
| 页表 | block table |
| 物理页 | KV block (16 token / block 默认) |

好处：
- 无内部碎片
- 跨 request 共享 prefix (radix tree)
- KV 可被其他 request 重用

## SGLang RadixAttention（2024）

SGLang 把 PagedAttention + **radix tree** 索引，让 prefix 在多 request 间高效共享：

```
Prefix "System: You are agent..." → 内存中 1 份
多 request 共享同 prefix → 只算后缀 K/V
```

→ Anthropic / OpenAI 的 prompt caching 后台用类似机制。

## 多种 KV optimization

| 技术 | 干啥 |
|------|------|
| **PagedAttention** | 块管理 |
| **RadixAttention** | prefix 共享 |
| **GQA** (Grouped Query Attn) | K/V 头数减少 → cache 小 8× |
| **MQA** (Multi-Query Attn) | K/V 1 个头 → cache 极小 |
| **MLA** (DeepSeek) | low-rank K/V → cache 小 5× |
| **CacheBlend** | retrieve KV from doc cache |

## KV 压缩

| 方法 | 比 |
|------|---|
| INT8 quantize | 2× |
| FP8 quantize | 2× |
| KIVI (INT4) | 4× |
| SnapKV | 选关键 token (rest evict) |
| H2O | top-K heavy hitters |
| StreamingLLM | 留 first + recent |

## 与 prompt caching 关系

| 维度 | KV cache | Prompt cache |
|------|---------|--------------|
| 层级 | GPU 内核 | API |
| 单位 | per layer K/V | per request prefix |
| 谁管 | vLLM/SGLang | Anthropic/OpenAI |
| 看见与否 | 用户透明 | API 暴露 cache_control |

Prompt caching 内核还是用 KV cache + radix tree。

## 实现 (`kv_cache_mock.py` 预告)

简单 KV cache 模拟（用 dict 而非 tensor）：

```python
class KVCache:
    def __init__(self, block_size=16):
        self.blocks: dict[int, dict] = {}  # block_id → {tokens, k, v}
        self.block_size = block_size

    def allocate(self, seq_id, n_tokens) -> list[int]:
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        return [self._new_block() for _ in range(n_blocks)]
```

## 退出条件

- 能讲 KV cache 是什么
- 能讲 PagedAttention 类比
- 能列 6 KV optimization

## 一句话

> KV cache = 每层 K/V 存一遍下次不重算 — PagedAttention 块管 + RadixAttention prefix 共享 = vLLM/SGLang 内核。
