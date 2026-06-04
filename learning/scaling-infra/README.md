# Topic 6: Scaling Infra（训练与推理基础设施）

> Module 3 「造大模型」第 6 专题 · 14 lectures · 14 notebooks · ~20h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Scaling Laws (Kaplan / Chinchilla) | `scaling_laws.py` |
| L02 | 并行训练总览 (DP/TP/PP/SP/ZeRO) | `parallelism_demo.py` |
| L03 | **FSDP** ⭐⭐⭐⭐⭐ | `fsdp_demo.py` |
| L04 | DeepSpeed (ZeRO + 3D) | `deepspeed_config.py` |
| L05 | Megatron-LM TP | `megatron_tp_demo.py` |
| L06 | Pipeline Parallel (1F1B) | `pipeline_parallel_demo.py` |
| L07 | Sequence/Context Parallel | (Megatron-CP / Ulysses) |
| L08 | **vLLM + PagedAttention** ⭐⭐⭐⭐⭐ | `paged_attention_demo.py`, `vllm_demo.py` |
| L09 | SGLang (RadixAttention) | `sglang_demo.py` |
| L10 | Speculative Decoding (EAGLE / Medusa) | `speculative_decoding.py` |
| L11 | 量化推理 (GPTQ/AWQ/FP8/int4) | `quantization_demo.py` |
| L12 | Mixed Precision + Stability | `mixed_precision_demo.py` |
| L13 | 监控与故障 | `monitoring_demo.py` |
| L14 | **Capstone** ⭐⭐⭐⭐⭐ Training Estimator | `capstone_train_estimator.py` |

## Tags

- `si-fsdp-ds` — L01-L04 scaling laws + FSDP + DeepSpeed
- `si-megatron` — L05-L07 TP / PP / SP
- `si-inference` — L08-L11 vLLM / SGLang / spec / quant
- `scaling-infra` — 最终 (含 capstone)

## 核心收获

1. **Scaling laws**：Chinchilla 1:20 是基线，Llama-3 风格 1:200+ 是工业实践
2. **训练并行**：< 7B FSDP、7-70B FSDP/TP 节点内、> 70B 3D
3. **推理优化**：PagedAttention 解决 KV 碎片，SGLang Radix 进化 prefix 共享
4. **混精度**：bf16 是 LLM 训练首选；fp16 需 GradScaler
5. **量化**：fp8 (H100+) > AWQ int4 > GPTQ > bitsandbytes

## 关键论文/工具

- Chinchilla (Hoffmann 2022)
- ZeRO (Rajbhandari 2020)
- Megatron-LM (Shoeybi 2019)
- vLLM (Kwon 2023)
- SGLang (Zheng 2024)
- EAGLE-2 / Medusa
- GPTQ / AWQ / SmoothQuant

## 环境

```powershell
python environment/verify_env.py
```

Windows 可跑 L01-L07/L12-L14 的 demo；vLLM/SGLang 实跑需 WSL2。

## 运行

```powershell
python -m pytest src/tests/ -v
python src/capstone_train_estimator.py
```
