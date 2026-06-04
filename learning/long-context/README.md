# Topic 5: 长上下文（Long Context）

> Module 3 「造大模型」第 5 专题 · 13 lectures · 13 notebooks · ~22h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | RoPE 原理 + 复数旋转 | `common.py` |
| L02 | Position Interpolation (PI) | `rope_pi.py` |
| L03 | NTK-aware RoPE | `rope_ntk.py` |
| L04 | **YaRN** ⭐⭐⭐⭐⭐ | `rope_yarn.py` |
| L05 | 3D-RoPE (多模态/MM) | `rope_3d.py` |
| L06 | Ring Attention naive | `ring_attention_naive.py` |
| L07 | Striped/Ring lib 对照 | `ring_attention_lib.py` |
| L08 | Infini-Attention | `infini_attention.py` |
| L09 | FlashAttention 长 ctx 优化 | (用 L02 之 fa) |
| L10 | **NIAH + RULER** 评测 | `niah_eval.py`, `ruler_eval.py` |
| L11 | 长 ctx 数据打包 + 课程 | `long_data_packing.py` |
| L12 | 长 ctx 陷阱 (Lost in middle 等) | — |
| L13 | **Capstone** ⭐⭐⭐⭐⭐ Llama-3.2-1B YaRN 32k | `capstone_yarn_llama32.py` |

## Tags 路径

- `lc-rope` — RoPE + PI + NTK + YaRN + 3D
- `lc-ring` — Ring/Striped Attention
- `lc-infini` — Infini-Attention
- `lc-eval` — NIAH/RULER + data packing + pitfalls
- `long-context` — 最终, 含 Capstone

## 核心收获

1. **YaRN 优于 NTK** — NTK by parts + attn temperature 是当前 RoPE 扩 ctx 默认选项
2. **NIAH ≠ 真用** — RULER 测试更接近真实长 ctx 需求
3. **KV cache 是显存大头** — 1B + 128k ctx 已 17 GB，比模型本身还大
4. **Lost in middle 真实存在** — 设计长 ctx 系统要考虑加 RAG / 中部强化训练

## 关键论文

- YaRN (Peng et al, 2024) — 当前 RoPE 扩 ctx SOTA
- NTK-by-parts (bloc97 GitHub 2023) — YaRN 的核心 idea
- Ring Attention (Liu, Yan, Abbeel 2023) — 序列并行
- Infini-Attention (Munkhdalai 2024) — 滑窗 + 压缩 memory
- NIAH (Kamradt 2024 GitHub) — 评测标准
- RULER (NVIDIA 2024) — NIAH 升级版
- Lost in the Middle (Liu 2023) — 警示

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 跑所有测试
python -m pytest src/tests/ -v

# Capstone 真训 (5090 24G + 5.5h)
python src/capstone_yarn_llama32.py --train --steps 500
```
