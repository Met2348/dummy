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

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（14/14，全 CPU 纯数值/结构 self-test，秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules scaling-infra
> ```

**Scaling Laws + 并行训练（L01-L06，显存账本 / 真 torch 模块，各 < 4s）**：

```powershell
python learning/scaling-infra/src/scaling_laws.py          # Chinchilla 1:20 vs Llama-3 1:200，IsoFLOP loss 表（谷底 ratio≈100）
python learning/scaling-infra/src/parallelism_demo.py      # DP/ZeRO-1/2/3/TP/PP 显存对比（70B@64GPU：DP 840→ZeRO-3 13GB）
python learning/scaling-infra/src/fsdp_demo.py             # 真建 TinyModel 计参 + Llama-8B FSDP 各阶段 shard 显存
python learning/scaling-infra/src/deepspeed_config.py      # ZeRO-3(+CPU offload) / Megatron-DS JSON 配置生成
python learning/scaling-infra/src/megatron_tp_demo.py      # ColumnLinear→RowLinear→all-reduce 真 forward 出 shape
python learning/scaling-infra/src/pipeline_parallel_demo.py # GPipe/1F1B vs Interleaved bubble (S-1)/(S+M-1)
```

**推理优化（L08-L11，模拟 / 模板）**：

```powershell
python learning/scaling-infra/src/paged_attention_demo.py  # BlockManager block 管理 + beam-search CoW refcount 共享
python learning/scaling-infra/src/vllm_demo.py             # vLLM 离线推理 + OpenAI server 模板（仅打印，真跑需 WSL2）
python learning/scaling-infra/src/sglang_demo.py           # SGLang DSL：chat / fork / schema-JSON / launch（仅打印）
python learning/scaling-infra/src/speculative_decoding.py # accept=min(1,p_t/p_d) 拒绝采样 + speedup 曲线
python learning/scaling-infra/src/quantization_demo.py     # 8B 各 dtype 显存表 + GPTQ/AWQ/FP8/bnb 模板
```

**混精度 + 监控 + Capstone（L12-L14）**：

```powershell
python learning/scaling-infra/src/mixed_precision_demo.py  # bf16 vs fp16 范围 + AMP 模板 + loss-spike/grad-norm 守卫
python learning/scaling-infra/src/monitoring_demo.py       # MFU=6N·tok/s÷peak + EMA tracker + GPU 健康检查
python learning/scaling-infra/src/capstone_train_estimator.py # ⭐ 训练估算器：spec → 并行策略 + 显存 + throughput + 时长 + 成本
```

> **关键坑注记**：
> - 14 个 demo **全部无 argparse**（`runbook.yaml` 标 `v0: false`，跳过 `--help` 探针直接 smoke 直跑）；`common.py` 是纯公共工具（被其他脚本 import 的显存账本 helper），非 lecture 入口，不列。
> - **秒级 PASS 是真的**：scaling laws/显存账本/MFU/speculative 是纯解析公式（Chinchilla `406.4·N^-0.34+410.7·D^-0.28+1.69`、`6ND` FLOPs、ZeRO/TP 显存、MFU）；fsdp/megatron-tp/pipeline 真建 torch 模块跑 forward；paged-attention 真跑 BlockManager 模拟。非 no-op/假成功。
> - `vllm_demo` / `sglang_demo` / `quantization_demo`(模板段) 是**诚实的"仅打印 setup 模板"**教学脚本（docstring 明示真跑需 WSL2）——它们不 import 重型库也不伪装 exit 0，区别于"捕获缺依赖假装成功"的反模式。
> - **Capstone 修过一处真 bug**：175B/batch=2048/8GPU 这一案例显存 INFEASIBLE（最省策略仍需 342GB），`estimate()` 正确返回 `cost_usd=None`，但 `report()` 原先无条件 `${cost_usd:.0f}` → `TypeError` 崩在最后一例。已改为可行性分支打印 `n/a (does not fit)`，4 例全跑通。
> - 全部 CPU（无 `.cuda()`/文件写出的运行路径；命中的 `cuda`/`ckpt.pt` 均在打印的代码模板字符串内），`gpu: false`，不污染 repo。

**测试（V2）**：

```powershell
python -m pytest learning/scaling-infra/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules scaling-infra --tests
```

> 7 个测试硬断言 Chinchilla 1:20 比、over-train 1:200 比、estimator 7B/70B 可行性、DP/ZeRO-3/TP 显存缩放——均真实断言（无空守卫）。
