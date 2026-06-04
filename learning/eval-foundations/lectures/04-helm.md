# L04 · HELM — Stanford 全息评测

**Liang et al. 2022** · Holistic Evaluation of Language Models · CRFM Stanford

## 核心理念

不要"一个分数"，要 **scenario × metric** 矩阵。

```
       knowledge | reasoning | summ | toxicity | ...
GPT-4      87        72         62      0.02      ...
Claude     86        75         60      0.01      ...
Llama-3    85        70         65      0.03      ...
```

## 7 大 metric 维度

| 维度 | 含义 |
|------|------|
| **accuracy** | 主分数（exact match / F1 / ROUGE）|
| **calibration** | predicted prob 是否与正确率对齐 |
| **robustness** | 加 typo / 改 case 后掉多少 |
| **fairness** | 不同人口子群分数差 |
| **bias** | stereotype association |
| **toxicity** | 生成有害内容比例 |
| **efficiency** | tokens/sec、$/query |

每个 scenario 都跑 7 维。

## 16 大 scenario

NaturalQuestions / NarrativeQA / QuAC / HellaSwag / OpenbookQA / MMLU / TruthfulQA / MS MARCO / CNN/DailyMail / XSum / IMDB / RAFT / CivilComments / WikiText-103 / The Pile / TwitterAAE …

覆盖知识/阅读/总结/分类/语言模型/有害性/方言。

## 为什么它重要

1. **2022 第一个 holistic eval**：之前都是单 bench 报告
2. **维度独立**：accuracy 高不代表 robust / fair / safe
3. **Stanford 持续维护**：HELM-Lite / HELM-MMLU / HELM-Safety 分支

## 真分数（HELM v1, 2022）

| 模型 | mean win-rate | 备注 |
|------|--------------|------|
| InstructGPT davinci-2 | 0.74 | 当年最强 |
| TNLG-530B | 0.70 | NVIDIA |
| Anthropic-LM (52B) | 0.69 | Claude 早期版 |
| OPT-175B | 0.61 | Meta open |

后续 v2/v3 + HELM-Lite 在 2023-2024 持续更新。

## 实操：mini-HELM

我们 src/helm_local.py 实现 4 scenario + 2 metric：

```python
from helm_local import run_helm_local, render_table
from common import make_random_model

model = make_random_model()
cells = run_helm_local(model)
print(render_table(cells))
```

输出：
```
| scenario      | metric         | value |
|---|---|---|
| knowledge     | exact_match    | 0.000 |
| reasoning     | exact_match    | 0.000 |
| summarization | rouge1_proxy   | 0.020 |
| robustness    | exact_match    | 0.000 |
```

## 与 MMLU 对比

| 维度 | MMLU | HELM |
|------|------|------|
| 范围 | 知识 | 多 (16) |
| 维度 | 1 (accuracy) | 7 |
| 数据规模 | 14k | 数百万 token |
| 跑一次成本 | $10-100 | $1000+ |
| 教学用 | ✅ | 看主要分数 |
| 生产用 | 决策片面 | ✅ |

## 一句话

> HELM = 全息 X-ray 片，不是单点温度。
