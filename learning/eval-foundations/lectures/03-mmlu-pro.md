# L03 · MMLU-Pro — 升级"高考"

**Wang et al. 2024** · arXiv 2406.01574

## 三个核心升级

| 维度 | MMLU | MMLU-Pro |
|------|------|---------|
| 选项数 | 4 | **10** |
| 随机 chance | 25% | **10%** |
| 难度 | 高中-本科 | **本科-PhD** |
| 题目数 | ~14k | 12032 |
| 推理要求 | 弱 | **强**（多步推理）|
| 去污染 | ❌ | ✅ |

## 为什么扩到 10 选项

数学计算：
- 4-MCQ：guess 25%
- 10-MCQ：guess 10%
- 顶模型 GPT-4 在 MMLU 86%, MMLU-Pro 73% → 分辨度回来了

**关键洞察**：扩大选项不仅降 chance，更是**测试模型的判别能力**。在 4 选项 MCQ 中可能蒙对，10 选项就是真知识。

## 题目升级

MMLU 例：
```
Q: 太阳是什么颜色？
A. 黄色 B. 橙色 C. 白色 D. 红色
```

MMLU-Pro 例：
```
Q: 在标准大气压下，太阳辐射到达地球表面的主要电磁波段是？
A. γ射线  B. X射线  C. 紫外  D. 可见光
E. 近红外 F. 远红外 G. 微波  H. 无线电
I. 视波    J. 引力波
```

需要物理知识 + 排除法。

## 训练污染防御

MMLU-Pro 用：
1. **新出版的考试**（CFA, USMLE 2023+）
2. **手工 augment**（一题改 10 个 paraphrase）
3. **Min-K%++ 检测**：扫 Pile / RefinedWeb 等公开 corpus

→ leakage rate < 0.5%（MMLU 据估 5-15%）

## 主要分数（2024-2025）

| 模型 | MMLU-Pro |
|------|----------|
| GPT-3.5 | 39.5% |
| GPT-4 | 73.0% |
| GPT-4o | 76.5% |
| Claude 3.7 Sonnet | 80.4% |
| Llama 3.1 405B | 73.3% |
| Qwen 2.5 72B | 71.1% |
| 随机 | 10% |

## 实操

src/mmlu_pro_runner.py 给 5 题 micro-MMLU-Pro 覆盖 5 类（math/physics/law/chemistry/economics）。

随机模型 ≈ 10%，oracle = 100%。

## 一句话

> MMLU 是地基，MMLU-Pro 是新的"高考"——分辨当代顶模型。
