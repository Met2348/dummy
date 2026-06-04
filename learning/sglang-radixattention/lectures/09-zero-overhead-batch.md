# L09 · Zero-Overhead Batch

## 1 · 痛点
vLLM 即使有 PagedAttention，每个 iter 仍有 Python 调度开销 ~ms。SGLang 用三招把它降到 ~0.1ms。

## 2 · 三招
1. **CUDA Graph + bucketing**: 同 vLLM
2. **Overhead-free Scheduler**: 调度循环用 C++ 改写
3. **Prefill-Decode Co-batching**: chunked prefill 默认开

## 3 · CUDA Graph 配合 paged
难点：paged attention 形状不固定（block 数量变化）→ 不能直接 capture。
SGLang 解法：把 attention 拆出 graph，**只 capture FFN + LN + projection**，attention 单独走 FlashInfer。

## 4 · C++ 调度循环
Python 解释器 scheduler 开销：每 iter ~2ms。
SGLang 重写为 C++ 进程，IPC 通信，开销 ~0.1ms。

## 5 · 实测（A100, Qwen-7B, B=32）
| 引擎 | iter 时间 | overhead 占比 |
|------|---------|-------------|
| vLLM 0.5 | 12 ms | 17% |
| SGLang 0.4 | 8 ms | 3% |

## 6 · 何时收益最明显
- 小 batch / 短 output：overhead 占比高 → SGLang 强
- 大 batch / 长 output：compute 占主导 → 平手

## 7 · 一句话
> 当 batch 数和 decode token 多到 GPU 飞起，调度循环本身就成了瓶颈。
