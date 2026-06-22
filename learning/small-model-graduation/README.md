# Topic 8: Small Model Graduation（五部曲毕业 Capstone）

> Module 3 「造大模型」第 8 专题 — **系列毕业 capstone** · 14 lectures · 14 notebooks · ~18h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 毕业总览 — Module 3 五部曲 | `common.py` |
| L02 | Vanilla GPT-2 baseline (ckpt A) | `vanilla_gpt2.py` |
| L03 | 改数据 (ckpt B Cosmopedia) | — |
| L04 | 改架构 (ckpt C Phi-tiny) | (reuse Topic 7 model) |
| L05 | 长 ctx 扩展 (ckpt D YaRN) | (reuse Topic 5 YaRN) |
| L06 | 综合 (ckpt E curriculum) | `train_variant.py` |
| L07 | Bench matrix (6 metric × 5 ckpt) | `bench_matrix.py` |
| L08 | 可视化与报告 | `visualize.py` |
| L09 | 同 prompt 多 ckpt 生成对照 | `generations_compare.py` |
| L10 | 五部曲 trick 总结 (top 10) | — |
| L11 | Module 3 全程回顾 | — |
| L12 | 真实世界案例 (DeepSeek-V3, R1, etc.) | — |
| L13 | 下一程 Module 4 桥梁 | — |
| L14 | **Final Graduation** ⭐⭐⭐⭐⭐⭐ | `graduation_capstone.py` |

## Tags

- `grad-5ckpt` — L01-L06 5 ckpt 训练
- `grad-bench` — L07-L10 评测 + 可视化 + tricks
- `造改-graduation` — 最终 (含 L11-L14 + 报告) ⭐⭐⭐⭐⭐⭐

## 五部曲 ckpt 对照

| ckpt | model | data | training | 关键提升 |
|------|-------|------|----------|---------|
| **A** | GPT-2 124M | TinyStories + WebText | 3000 step | baseline |
| **B** | GPT-2 124M | **Cosmopedia + 高质** | 3000 step | +5pp HellaSwag |
| **C** | **Phi-tiny 270M** (GQA+RoPE+SwiGLU+RMSNorm) | 同 B | 4000 step | +8pp HellaSwag |
| **D** | C + **YaRN scale=4** | 长 doc | C+100 step LoRA | +75pp NIAH |
| **E** | C + **curriculum** | 全 | 4000 step + WSD annealing | 综合最强 |

## 期望 benchmark

| ckpt | val_loss | HellaSwag | PIQA | tinyMMLU | GSM8K | NIAH@8k |
|------|----------|-----------|------|----------|-------|---------|
| A | 3.50 | 0.35 | 0.65 | 0.25 | 0.02 | 0% |
| B | 3.20 | 0.40 | 0.68 | 0.30 | 0.05 | 0% |
| C | 2.90 | 0.48 | 0.72 | 0.33 | 0.08 | 5% |
| D | 2.90 | 0.48 | 0.72 | 0.33 | 0.08 | 80% |
| **E** | **2.80** | **0.50** | **0.74** | **0.35** | **0.10** | **80%** |

## 五部曲贡献拆解

- **数据** (A→B): +5pp HellaSwag, +5pp MMLU
- **架构 + size** (B→C): +8pp HellaSwag
- **长 ctx** (C→D): 0pp benchmark, +75pp NIAH
- **curriculum** (C→E): 综合 +15pp HellaSwag

## 报告产出

```
report/
  benchmarks.csv     主数据表
  report.md          5 节文字
  generations.md     5 ckpt × 5 prompt
  ablation.json      五部曲分解
  curve.png          5 loss 曲线
  radar.png          5 metric × 5 ckpt
  niah_*.png         NIAH heatmap
```

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试
python -m pytest src/tests/ -v

# Graduation capstone (dry-run)
python src/graduation_capstone.py

# Graduation capstone (真训, 5090 ~24h)
python src/graduation_capstone.py --train
```

## 运行验证（Runbook）

文档入口命令清单见 [`runbook.yaml`](runbook.yaml)。一键 smoke 验证（V0 静态 `--help` + V1 最小预算真跑）：

```bash
.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --runbook --modules small-model-graduation --timeout 180 \
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

可跑入口（`full` = 文档原形态；smoke 由 runbook 自动缩小预算）：

| 入口 | full 形态 | smoke（验证器实跑） | GPU |
|------|-----------|---------------------|:---:|
| `vanilla_gpt2.py` | 直跑 | 同（建 124.4M + 前向） | — |
| `tinystories_original_minimal.py` | 直跑 | 同（self-test 真 assert） | — |
| `bench_matrix.py` / `generations_compare.py` / `visualize.py` / `common.py` | 直跑 | 同（真写表/模板/描述） | — |
| `train_variant.py --variant A --max_step 3000 --train` | A 真训 3000 step | `--max_step 2 --seq_len 64 --micro_batch 2 --grad_accum 2 --train`（2 step 真训, ~12s, 不落 ckpt） | ✓ |
| `graduation_capstone.py` | dry-run 生成 5 报告产物 | 同, 写 `/tmp/grad_smoke`（`--train` 才 5 ckpt 真训, 5090 ~24h） | — |

关键坑注记：

- **`train_variant.py` 的覆盖 flag**：lectures 文档化的 `--max_step` / `--seq_len` 等本是脚本未实现的（原仅 `--variant`/`--train`，文档命令会 `unrecognized arguments` 崩）；现已补齐为可选覆盖项（默认回退各 variant 配置），使文档命令真能跑，并让 smoke 用极小预算走真训路径。`--max_step < 100` 的 smoke 不写 `ckpt_*.pt`（仓库零污染）。
- **`graduation_capstone.py` dry-run** 用 `EXPECTED` 参考 benchmark 生成报告（stdout 明示 `using EXPECTED for dry-run`），是诚实的报告骨架，非伪造训练结果；真实数字需 `--train`。
- 训练用 `mock_loader`（random ints）教学占位，**不下载真实语料**（src 内无 `load_dataset`）；真训质量需真数据（见 lectures L03 Cosmopedia）。
- variant C/D/E `--train` 依赖 Topic 7 `pretraining-recipe/src/phi_tiny_model.py` 且为重型（Phi-tiny 270M / 8k ctx），非本地 smoke 目标；smoke 只跑最轻的 variant A。

测试（V2）：

```bash
.venv/Scripts/python.exe scripts/eric_3080ti_env_audit.py --modules small-model-graduation --tests --timeout 600 \
  --json-out /tmp/v2-small-model.json --md-out /tmp/v2-small-model.md
# 或直接: python -m pytest src/tests/ -v   (14 passed)
```

## 毕业 checklist

- [ ] 5 ckpt 全部训完
- [ ] 6 metric × 5 = 30 benchmark
- [ ] 5 prompt × 5 ckpt = 25 generation
- [ ] 4 图表 (curve / bar / radar / NIAH heatmap)
- [ ] report.md 10 节完整
- [ ] git tag `造改-graduation` 打上
- [ ] (可选) 进入 Module 4 RL 系列

## 与系列其它 topic 的关系

```
Topic 1 data-curation   → 数据 pipeline (ckpt B 用)
Topic 2 transformer-deep → Phi-tiny 架构 (ckpt C)
Topic 5 long-context     → YaRN (ckpt D)
Topic 6 scaling-infra    → 显存账本 + WSD
Topic 7 pretraining-recipe → Phi-tiny model code + train loop
Topic 8 (本)              → 综合 + 对照
```

## 关键文献

- Phi-1/2/3 (Microsoft 2023-2024)
- TinyLlama (2024)
- Cosmopedia (HF 2024)
- Llama-3 / DeepSeek-V3 tech reports
- YaRN (Peng 2024)
- WSD lr schedule (MiniCPM 2024)
