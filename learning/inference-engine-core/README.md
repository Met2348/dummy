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

# vLLM 对照（需安装 vllm + 下载 Qwen2.5-0.5B；缺 vllm 会 fail-fast，非静默跳过）
python src/vllm_compare.py --model Qwen/Qwen2.5-0.5B
```

## 运行验证（Runbook）

文档入口命令集中在 [`runbook.yaml`](runbook.yaml)，一键验证（在 3080 Ti repo-local `.venv`）：

```bash
python scripts/eric_3080ti_env_audit.py --runbook --modules inference-engine-core \
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json \
  --md-out  docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

可跑入口（均 repo-root 相对；全部纯 CPU、秒级、无重型依赖）：

| 入口 | 命令（full 形态） | 说明 |
|------|------------------|------|
| naive attention | `python learning/inference-engine-core/src/attention_naive.py` | L10 基线（flash-attn/flashinfer 缺时回退） |
| naive KV | `python learning/inference-engine-core/src/naive_kv.py` | L02 静态 KV 碎片度量（~70% 浪费） |
| paged KV | `python learning/inference-engine-core/src/paged_kv.py` | L03 block table + 物理块池 |
| paged attention (kernel) | `python learning/inference-engine-core/src/paged_attention_triton.py` | L04 缺 triton 时 **torch 回退** |
| chunked prefill | `python learning/inference-engine-core/src/chunked_prefill.py` | L06 decode + 切片 prefill 混批 |
| continuous batching | `python learning/inference-engine-core/src/continuous_batching.py` | L05 迭代级调度骨架 |
| prefix cache | `python learning/inference-engine-core/src/prefix_cache.py` | L07 block-hash + LRU（hit_rate~0.9） |
| sampling | `python learning/inference-engine-core/src/sampling.py` | L11 T/top-k/top-p/min-p/penalty |
| **mini-vLLM (capstone)** | `python learning/inference-engine-core/src/mini_vllm.py --case 0` | L13 全 5 case；`--case 1..5` 跑单个 |

> 这些 demo 是 CPU stub forward 的纯数值演示，<1s 即出真实数字（paged 利用率 / 批吞吐 / 命中率），秒级 PASS 非 no-op。

**关键坑注记：**
- `paged_attention_triton.py`：顶层不 import triton；其 `paged_attention_triton()` 是占位，缺 triton 时**直接委派 `paged_attention_torch`**（bit-identical 的真 torch 回退，非假成功 print）。与 naive dense attention 数值对齐（max abs diff `0.0`，见 `test_paged_attention_matches_dense`）。
- `vllm_compare.py`：**`tier: skip`** —— 需装 `vllm`(Linux+CUDA) 且下载 Qwen2.5-0.5B(~1GB)，本机不跑。缺 vllm 时**fail-fast**（`SystemExit`，非原先的 `print + return [] → exit 0` 假成功）。在装好 vllm 的机上可跑 full 形态对照 `mini_vllm` 的同 5 case。
- `scheduling_policies.py`（L08）是纯库（无 `__main__`），由 `test_scheduling.py` 覆盖，不在 runbook 入口里，归测试（V2）。

**测试（V2）：**

```bash
python -m pytest learning/inference-engine-core/src/tests -q   # 27 passed
```

## 退出条件 checklist

- [x] 13 lecture + 13 notebook
- [x] 27 个测试通过（6 paged + 6 sched + 9 policy/sampling + 6 capstone）
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
