# L10 · VLM 评测 — MMMU / MathVista / OCRBench / MMBench

## MMMU (Yue 2024)

- **11500 多模态题** × 30 学科 (art / engineering / medicine / ...)
- 4-MCQ
- 图像 + 文本混合

```
Q: [chemistry structure image]
What is the IUPAC name of this molecule?
A. methane  B. ethanol  C. benzene  D. methanol
```

**分数**：
- GPT-4V: 56%
- GPT-4o: 69%
- Claude 3.7: 75%
- 人类专家: ~89%

## MathVista (Lu 2023)

- **6141 视觉数学题**
- 几何 / 图表 / 函数图 / 表格 / 教科书图
- 短答案（数字 / 多选）

```
Q: [scatterplot]
What is the correlation between X and Y?
A. positive  B. negative  C. none
```

## OCRBench (Liu 2023)

- **1000 测试**：text recognition, info extraction, formula, handwriting
- 真测 VLM 的 OCR 能力
- 中文 / 多语言部分

## MMBench (Liu 2023)

- **2.9k 题** × 20 能力维度（perception, reasoning, knowledge）
- circular eval：A/B/C/D 选项轮换避免 bias

## 4 bench 对比

| Bench | 题数 | 重点 |
|------|------|------|
| MMMU | 11.5k | 学科广度 |
| MathVista | 6.1k | 视觉数学 |
| OCRBench | 1k | OCR / 文档 |
| MMBench | 2.9k | 多维度能力 |

## VLM 评测特殊问题

1. **image tokenizer 差**：每家不同（CLIP / SigLIP / Qwen-VL）
2. **图分辨率**：224 vs 448 vs 高分辨率
3. **多图支持**：单 vs 多图
4. **视频**：扩展为 VideoMME / EgoSchema

## 实操

我们 src 没专门 VLM runner（依赖大）。
推荐：用 `lmms-eval` (lm-eval-harness 的多模态版)。

## 一句话

> VLM bench = 给"会看图"的 LLM 出题 — 2024 后必备。
