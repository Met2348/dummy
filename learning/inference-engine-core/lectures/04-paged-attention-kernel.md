# L04 · Paged Attention 的 CUDA / Triton kernel

## 1 · 为什么写 kernel
朴素 Python loop 慢 100×。kernel 关键收益：
- block 一次性 load 到 SRAM
- gather 物理 block id 在线程内
- avoid global K reload

## 2 · 朴素 kernel 草图
每个 query token `q` 启动一个 thread block：
```c
// 伪代码
block_table = req.block_table        // 在 shared mem
out = 0
for blk in block_table:
    K_blk = physical_blocks[blk]      // 一次 load 256B/线程
    for t in range(BLOCK_SIZE):
        out += dot(q, K_blk[t]) * V_blk[t]
```

## 3 · Triton 简化版（教学）
```python
@triton.jit
def paged_attn_kernel(
    Q, K_blocks, V_blocks, block_table,
    out,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)            # 1 program / token
    q = tl.load(Q + pid * HEAD_DIM + tl.arange(0, HEAD_DIM))
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    denom = 0.0
    n_blocks = tl.load(block_table_len + pid)
    for i in range(n_blocks):
        blk_id = tl.load(block_table + pid * MAX_BLOCKS + i)
        K_blk = tl.load(K_blocks + blk_id * BLOCK_SIZE * HEAD_DIM + ...)
        scores = tl.sum(q[None, :] * K_blk, axis=1)
        # online softmax 略
        ...
    tl.store(out + pid * HEAD_DIM + tl.arange(0, HEAD_DIM), acc)
```

## 4 · 真实 vLLM kernel
- CUDA 写成，支持 GQA / ALiBi / sliding window
- 支持 mixed precision (fp16/bf16/fp8 KV)
- 支持 fp8 KV cache（节省 50% 显存）

## 5 · 性能数字 (A100, 7B)
| 方案 | 1 token decode | 内存峰值 |
|------|---------------|---------|
| naive python loop | 80 ms | full | 
| flash-attn (连续 KV) | 0.5 ms | full |
| paged attn (16-block) | 0.6 ms | -30% |

paged 损失 < 20%，省 30%+ 显存 → **吞吐 +24x**（更多请求并发）。

## 6 · FlashInfer
2024 出现的 inference 专用 attn 库，集成 paged kv 的最优 kernel 实现。vLLM 0.7+ 集成。

## 7 · 实现：[paged_attention_triton.py](../src/paged_attention_triton.py)
- 教学 Triton kernel（简化 online softmax）
- 数值与朴素 paged 对照 < 1e-3

## 8 · 一句话
> 没有好 kernel，PagedAttention 只是个"漂亮的数据结构"。
