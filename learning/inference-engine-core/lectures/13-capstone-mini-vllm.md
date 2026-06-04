# L13 · Capstone — mini-vLLM 100 行复刻

## 1 · 目标
- 200 行内（不计注释）实现：
  - PagedKvPool + BlockTable
  - Engine (admit + iter step + retire)
  - Sampler (top-k/p)
  - 5 case demo + vLLM 对照

## 2 · 架构图
```
+-------------------------+
| MiniEngine              |
|  · pool: PagedKvPool    |
|  · pending: deque       |
|  · running: list[Req]   |
|  · sampler: Sampler     |
|  · forward_fn: Callable | <- 真实 / mock model
+-------------------------+
       ↓ step()
+-------------------------+
| 1 admit (KV budget)     |
| 2 forward(running)      |
| 3 sample + append       |
| 4 retire finished       |
+-------------------------+
```

## 3 · 5 个对照 case
| case | prompt 长 | out 长 | batch | 看 |
|------|----------|--------|-------|-----|
| 1 短 in 短 out | 32 | 32 | 8 | warmup |
| 2 长 in 短 out | 1024 | 8 | 4 | prefill 主导 |
| 3 短 in 长 out | 16 | 256 | 8 | decode 主导 |
| 4 大 batch | 16 | 32 | 32 | KV 压力 |
| 5 流式 | 32 | 128 | 1 | TTFT/ITL |

## 4 · 指标
- throughput (tok/s) → mini vs vLLM
- 显存峰值（torch.cuda.max_memory_allocated）
- TTFT / ITL

## 5 · 退出条件
- throughput ≥ vLLM 50%（教学版可接受）
- 5 case 全跑通无 OOM
- 数值 sanity：固定 seed 下生成 token 序列与 reference 一致

## 6 · 实现：[mini_vllm.py](../src/mini_vllm.py)
入口 `python mini_vllm.py --case 1`

## 7 · 对照工具：[vllm_compare.py](../src/vllm_compare.py)
`python vllm_compare.py` 跑 5 case 在 vllm.LLM 上的 baseline，写 `bench.json`

## 8 · 一句话
> 100 行 mini-vLLM 不是要替代 vLLM — 是要让你**永远再不怕 vLLM 源码**。
