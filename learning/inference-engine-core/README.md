# Topic 1: Inference Engine Core（vLLM 骨架复刻）

> Module 5 「用大模型」第 1 专题 · 13 lectures · 13 notebooks · ~14h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 推理服务全图（prefill/decode/KV math）| `common.py` |
| L02 | Naive KV cache + fragmentation 度量 | `naive_kv.py` |
| L03 | PagedAttention (vLLM SOSP'23) | `paged_kv.py` |
| L04 | Paged attention kernel (Triton/torch) | `paged_attention_triton.py` |
| L05 | Continuous Batching (Orca OSDI'22) | `continuous_batching.py` |
| L06 | Chunked Prefill (DeepSpeed/Sarathi) | `chunked_prefill.py` |
| L07 | Prefix Caching (hash + LRU) | `prefix_cache.py` |
| L08 | Scheduling Policies (FCFS/SJF/priority) | `scheduling_policies.py` |
| L09 | CUDA Graphs (capture/replay/bucketing) | — |
| L10 | Attention Backends (FA v2/v3/FlashInfer) | `attention_naive.py` |
| L11 | Sampling (T/top-k/p/min-p/penalty) | `sampling.py` |
| L12 | vLLM 源码导读 (5 文件) | — |
| L13 | **Capstone: mini-vLLM 5 case** ⭐ | `mini_vllm.py` + `vllm_compare.py` |

## Tags

- `ie-paged` — L01-L04 PagedAttention 基础
- `ie-sched` — L05-L07 continuous batching + chunked + prefix
- `ie-policy-sample` — L08-L12 调度/sampling/backends/源码
- `infer-engine` — 最终（含 Capstone + README）

## 三轨代码策略

| 轨 | 工具 | 文件 |
|----|------|-----|
| minimal | 100-200 行手写 | `mini_vllm.py` |
| 库 | vLLM 对照 | `vllm_compare.py` |
| kernel | Triton 简化 | `paged_attention_triton.py` |

## Capstone 5 case 实测（mini-vLLM）

| case | prompt | out | batch | 吞吐 (tok/s) |
|------|--------|-----|-------|-------------|
| 1 short-in-short-out | 32 | 32 | 8 | ~17k |
| 2 long-in-short-out | 1024 | 8 | 4 | ~830 |
| 3 short-in-long-out | 16 | 256 | 8 | ~22k |
| 4 big-batch | 16 | 32 | 32 | ~22k |
| 5 streaming-1 | 32 | 128 | 1 | ~21k |

> 数据来源 CPU 单线程 stub forward（无真实模型）；与 vLLM 直接对比需用 `vllm_compare.py` 在 GPU 上跑同 5 case。

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试
python -m pytest src/tests/ -v

# 不装 pytest 也可以直接跑
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_paged_attention as T; [getattr(T,n)() for n in dir(T) if n.startswith('test_')]"

# mini-vLLM 全 5 case
python src/mini_vllm.py --case 0

# 单 case
python src/mini_vllm.py --case 3

# vLLM 对照（需安装 vllm）
python src/vllm_compare.py --model Qwen/Qwen2.5-0.5B
```

## 退出条件 checklist

- [x] 13 lecture + 13 notebook
- [x] 21 个测试通过（6 paged + 6 sched + 9 policy/sampling + 6 capstone）
- [x] mini-vLLM 5 case 全部完成
- [x] git tag `infer-engine` ✓

## 与系列其它 topic 的关系

```
Topic 1 (本) inference-engine-core
   ↓ provides scheduler + paged KV
Topic 2 sglang-radixattention         (radix 替 hash, agent 场景)
Topic 3 speculative-decoding          (draft + verify 替 sample)
Topic 4 quantization-deploy           (FP8/int4 KV)
Topic 5 distributed-inference         (TP/PP/EP 跨卡)
Topic 6 production-serving            (TRT-LLM/Triton 替 mini)
Topic 7 serving-graduation            (五线综合 用 ckpt)
```

## 关键文献

- vLLM (Kwon SOSP 2023)
- Orca Continuous Batching (Yu OSDI 2022)
- Chunked Prefill / Sarathi (2024)
- FlashAttention v2 (Tri Dao 2023)
- FlashAttention v3 (Tri Dao 2024)
- FlashInfer (UCB 2024)

## 一句话总结

> **推理引擎 = scheduler + paged KV + sampler + 3 套 backend + 4 套 policy**。
> 这五大件搞清 → vLLM/SGLang/TRT-LLM 全是变奏。
