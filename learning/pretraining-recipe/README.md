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

# Capstone (dry-run)
python src/capstone_train.py

# Capstone (真训, 5090 24G ~6h)
python src/capstone_train.py --train --max_step 4000
```
