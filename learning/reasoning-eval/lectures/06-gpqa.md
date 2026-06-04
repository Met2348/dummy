# L06 · GPQA Diamond — google-proof Q&A

**Rein et al. 2023** · arXiv 2311.12022

## 设计动机

MMLU 的题 google 一下就有 → 已被 corpus 污染。
GPQA 设计成 **"PhD 出题、google 也搜不到"**。

## 数据

- **448 题** Diamond split（最难、最干净）
- 1192 题完整集
- **PhD 出题**：每题作者是该领域博士
- 验证：另一组 PhD 试题 → ≥80% 才入选

## 3 学科

| 学科 | 子领域 |
|------|------|
| Biology | molecular bio / genetics / ecology |
| Physics | QM / EM / thermodynamics / astrophysics |
| Chemistry | organic / inorganic / physical chem |

## 例题

```
Q: An electron is in the 1s state of a hydrogen atom. What is the most
probable distance from the nucleus (in units of Bohr radius a_0)?

A. 0
B. a_0           ← correct
C. 2 a_0
D. 0.5 a_0
```

需要 QM 知识（径向分布 max in 1s 在 r = a_0）。

## 分数

| 类型 | GPQA Diamond |
|------|-------------|
| 随机 baseline | 25% |
| 非专业领域博士 | 34% (open book + google) |
| 专业领域博士 | 81% |
| GPT-3.5 | 28% |
| GPT-4 | 36% |
| GPT-4o | 53.6% |
| **Claude 3.7 Sonnet** | **68%** |
| **R1** | **71.5%** |
| **o3** | **87.7%** |

→ 比 MMLU-Pro 更分辨当代顶模型。

## 与 MMLU 区别

| 维度 | MMLU | GPQA |
|------|------|------|
| 题数 | 14k | 448 (Diamond) |
| 出题人 | 网上抓 | PhD 手写 |
| 污染 | 严重 | 几乎无 |
| 难度 | 高中-本科 | 研究生 |
| 推理需要 | 弱 | 强 |

## Open LLM Leaderboard v2 用法

直接报 GPQA Diamond 0-shot acc，不允许 CoT。

## 实操

src/gpqa_runner.py 5 题 micro（覆盖 P/C/B 三领域）：

```python
from gpqa_runner import run_gpqa
from common import make_dummy_model

rs = run_gpqa(make_dummy_model("E"))  # invalid letter
# Expected: 0% acc
```

## 一句话

> GPQA = 让 LLM 体会"研究生水准"的 reality check。
