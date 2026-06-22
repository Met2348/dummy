# Topic 7: Pretraining Recipe（预训练管线）

> Module 3 「造大模型」第 7 专题 · 16 lectures · 16 notebooks · ~24h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 预训练总流水线 | `common.py` |
| L02 | 数据配比 + 课程 | `data_mixture.py` |
| L03 | Tokenization 选型 | (依赖 data-curation) |
| L04 | 初始化 + LR schedule (WSD/μP) | `init_schedule.py` |
| L05 | 数据加载 + shard | `dataset_shards.py` |
| L06 | **Phi-tiny 270M 架构** ⭐⭐⭐⭐⭐ | `phi_tiny_model.py` |
| L07 | 训练 loop + 稳定性 | `training_loop.py` |
| L08 | 评测 (HellaSwag/MMLU/loss) | `eval_benchmarks.py` |
| L09 | Continual Pretraining | — |
| L10 | 蒸馏 | `distillation.py` |
| L11 | Emergence + Debug | — |
| L12 | 预训练陷阱 | — |
| L13 | 合成数据 (Phi 风格) | `synth_data_prompt.py` |
| L14 | Llama-3 recipe 拆解 | — |
| L15 | DeepSeek-V3 recipe 拆解 | — |
| L16 | **Capstone** ⭐⭐⭐⭐⭐ 训 Phi-tiny 270M | `capstone_train.py` |

## Tags

- `pr-pipeline` — L01-L04 总览 + data mix + tokenization + LR
- `pr-model-train` — L05-L09 dataloader + Phi-tiny + train loop + eval + CPT
- `pretraining-recipe` — 最终 (含 L10-L16 + capstone)

## 核心收获

1. **Pipeline 全图**：data → tokenize → mix → shard → init → train → eval → SFT → RLHF
2. **Phi-tiny 270M 标杆**：Pre-RMSNorm + GQA + RoPE + SwiGLU + tied embedding
3. **WSD lr 优于 cosine**：stable + annealing 集中
4. **数据配比是 secret sauce**：Llama-3 50% web + 30% code + 20% other
5. **合成数据可与 real 1:1 混**：Phi 路线
6. **DeepSeek-V3 是工程奇迹**：MoE + MLA + MTP + DualPipe + FP8 = $5.6M

## 关键论文/工具

- Phi-1/2/3 (Microsoft 2023-2024)
- Llama-3 tech report (Meta 2024)
- DeepSeek-V3 tech report (2024.12)
- TinyLlama (2024)
- nanoGPT (Karpathy)
- DCLM / Cosmopedia / FineWeb-Edu
- WSD (MiniCPM 2024)
- μP (Yang 2022)

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试
python -m pytest src/tests/ -v

# Capstone (dry-run：建模 + sanity check + WSD lr 表，不真训/不下载)
python src/capstone_train.py

# Capstone (真训, 5090 24G ~6h；3080 Ti 16GB 装得下但更慢)
python src/capstone_train.py --train --max_step 4000
```

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（10/10）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules pretraining-recipe `
>   --json-out docs/local-env/ERIC-3080Ti-runbook-results.json --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
> ```

**Lecture 组件 demo**（8 个直跑脚本，均 CPU、秒级、真实数值/结构输出）：

```powershell
python learning/pretraining-recipe/src/phi_tiny_model.py    # L06 Phi-tiny 270M 建模 + 前向 (315.7M / excl-embed 264.2M)
python learning/pretraining-recipe/src/init_schedule.py     # L04 cosine/WSD/inverse-sqrt/μP lr 曲线
python learning/pretraining-recipe/src/data_mixture.py      # L02 phi/llama3/qwen/deepseek 配比采样
python learning/pretraining-recipe/src/dataset_shards.py    # L05 nanoGPT memmap shard + ShardManager
python learning/pretraining-recipe/src/eval_benchmarks.py   # L08 val_loss/ppl/tiny-HellaSwag 库 self-test
python learning/pretraining-recipe/src/distillation.py      # L10 KD KL loss + CE+KD 组合
python learning/pretraining-recipe/src/synth_data_prompt.py # L13 Phi 风格合成数据 prompt + 过滤
python learning/pretraining-recipe/src/training_loop.py     # L07 训练 loop 库接口（真训在 capstone）
```

**Capstone：从零预训练 Phi-tiny 270M**（L16 ⭐）：

```powershell
# 真训（full）：4000 step、effective batch 128、WSD lr，5090 ~6h
python learning/pretraining-recipe/src/capstone_train.py --train --max_step 4000 --micro_batch 16 --grad_accum 8 --seq_len 1024
# 快速 smoke（验证真训路径可跑通；cuda bf16、step 0 loss~9.6、~16s）
python learning/pretraining-recipe/src/capstone_train.py --train --max_step 3 --micro_batch 2 --grad_accum 2 --seq_len 128
# 或仅 dry-run（建模 + sanity check，不真训）
python learning/pretraining-recipe/src/capstone_train.py
```

> ⚠️ **数据是 mock**：capstone 用 `mock_data_loader`（random int token）作教学占位，验证的是**流水线与训练循环**而非收敛质量。真正训出可用 270M 需接真实语料（lectures/16-capstone.md：Cosmopedia + TinyStories，GPT-2 BPE，~500M token）。
> **ckpt 落盘**：`--train` 仅在 `step%1000==0` 写 `ckpt_{step}.pt` 到当前目录；smoke（3 step）**不落任何文件**。full 训会在仓库根写 ckpt，按需清理。
> **离线安全**：src 里**无** `load_dataset`/HF 下载（唯一 `load_dataset("HuggingFaceFW/fineweb-edu")` 只在 L05 讲义代码块、且已用命名空间 id）；smoke 不依赖网络。
> **common.py** 是纯工具（被 tests import），不在 runbook 入口内。

**测试（V2）**：

```powershell
python -m pytest learning/pretraining-recipe/src/tests/ -v
# 或经审计 harness（注意 --json-out/--md-out 指向 /tmp，勿覆盖基线）：
# python scripts/eric_3080ti_env_audit.py --modules pretraining-recipe --tests --json-out /tmp/v2-pretraining.json --md-out /tmp/v2-pretraining.md
```
