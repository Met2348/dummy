# L01 · LLM 推理服务全图（serving overview）

## 1 · 推理 vs 训练
| 维度 | Train | Inference |
|------|-------|-----------|
| 并行度 | 几千 token / step | 1 token / step (decode) |
| 显存峰值 | activation 主 | KV cache 主 |
| 瓶颈 | compute-bound | memory-bound (decode) |
| 优化目标 | step time | throughput + latency |

## 2 · 两阶段
- **prefill**: 一次性算完 prompt 的所有 K/V → 写入 KV cache → compute-bound
- **decode**: 每步 1 token，读旧 KV + 算新 K/V → memory-bound

## 3 · 关键变量
- `B` batch / `S` seq / `H` hidden / `L` layers / `d_h` head_dim
- KV cache size per token per layer = `2 · n_kv_heads · d_h · dtype_bytes`
- 7B fp16 单 token KV ≈ 256 KB · 32 层 = 8 MB → 8k seq = 64 GB ⚡

## 4 · 三大指标
- **throughput** total output token/s
- **latency** time to first token (TTFT) + inter-token latency (ITL)
- **goodput** 满足 SLO 的 throughput

## 5 · 朴素服务的三个浪费
1. **静态 batch** 短请求等长请求 → idle GPU
2. **静态 KV 分配** 按 max_len → fragmentation ≥ 60%
3. **prefill 阻塞 decode** 长 prompt 来一个，所有 decoder 全卡住

## 6 · 现代解
| 浪费 | 解 |
|------|-----|
| 静态 batch | **Continuous Batching** (Orca) |
| KV 碎片 | **PagedAttention** (vLLM) |
| prefill 阻塞 | **Chunked Prefill** (DeepSpeed) |

## 7 · 本专题路线图
- L02-L04：KV 管理（naive → paged → kernel）
- L05-L06：调度（continuous → chunked）
- L07：prefix cache
- L08-L09：scheduling + cuda graph
- L10-L12：attention/sampling/源码
- L13：手写 mini-vLLM

## 8 · 一句话总结
> 推理服务 = **怎么把 KV cache 喂得最满，把 GPU 算得最爽**。
