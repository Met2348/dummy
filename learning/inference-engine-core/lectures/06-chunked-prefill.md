# L06 · Chunked Prefill（DeepSpeed/Sarathi 2024）

## 1 · 痛点：prefill 阻塞 decode
- 一个 8k token 的 prompt prefill → 占用 GPU 100 ms
- 其他 batch 里所有 decode（每个 5 ms）等 100 ms
- **head-of-line blocking**

## 2 · 解：把 prefill 切块
- prefill 切成 `chunk_size = 512` token 小段
- 每个 iter 只算一个 chunk
- decode 与 prefill chunk **混合 batch**

```
iter 1: [decode A, decode B, prefill C chunk 0]
iter 2: [decode A, decode B, prefill C chunk 1]
...
iter k: [decode A, decode B, prefill C chunk k]   <- C 全 prefilled
iter k+1: [decode A, decode B, decode C]
```

## 3 · 数学：算力打包
- decode FLOPs: `B_d · d_model² · H` 很小
- prefill chunk FLOPs: `chunk · d_model²` 中等
- 一次 iter 总 = decode + chunk → GPU 不空闲

```
goal: chunk_size = T_decode_iter * H * d_model / ratio
```
经验值 `chunk = 512`。

## 4 · attention mask 处理
prefill 需要 causal mask，且 `chunk_k` 的 q 要 attend 到 `chunk_0..k` 的所有 K。
→ 复用 PagedAttention，q 走 chunk，K 从 block table fetch 累积。

## 5 · 收益
| 配置 | TTFT (s) | TPOT (ms) |
|------|---------|----------|
| 无 chunk | 0.05（短）/ 2.0（长 prompt）| 50（被阻塞）|
| chunk=512 | 0.1 | 8 |

长 prompt TTFT 多 2×，但 TPOT 降 6×。**总体 latency SLO 提升**。

## 6 · 注意：与 paged 共存
chunk prefill 写 KV 用 paged 的 `append_token`，已 chunk-friendly。

## 7 · vLLM 实现
`--enable-chunked-prefill --max-num-batched-tokens 512`

## 8 · 实现：[chunked_prefill.py](../src/chunked_prefill.py)
- `ChunkedPrefillScheduler.next_batch()` 决定本 iter 给哪个请求加 chunk
- 与 continuous batching 共存
