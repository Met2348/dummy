# L07 · Humanity's Last Exam — 2025 新王

**Scale AI + CAIS, 2025.01**

## 背景

到 2025 初，MMLU-Pro / GPQA / AIME 都被 top model 推到 60%+。
需要新"难考"。

## 数据

- **3000 题**（v1: 1900, v2: 3000）
- **约 1000 位 PhD/领域专家**出题
- **跨 100+ 学科**（数学/物理/化学/生物/语言/历史/法律/医学/...）
- **审核**：每题至少 2 位独立专家通过
- **多模态**：~10% 题含图像

## 出题原则

每题应满足：
1. **明确答案** (short string / number)
2. **对当前 LLM 难**（提交时至少 80% LLM 错）
3. **专家可解**（不超 PhD 水准）
4. **去污染**：cutoff 后才公开

## 例题（来自 paper sample）

```
Q: In the language of Tlingit, what is the proto-form of the word for
'water' in Proto-Athabaskan?
A: tuʔ
```

```
Q: A particle in 3D box L×L×L. What is the energy of the (1,2,2) state
in units of (h^2 / 8mL^2)?
A: 9
```

## 分数（2025.01-2025.06）

| 模型 | HLE |
|------|------|
| GPT-4o | 3.5% |
| Claude 3.5 Sonnet | 4.6% |
| **R1** | **9.4%** |
| o1 | 9.1% |
| **Claude 3.7 (extended thinking)** | **14.4%** |
| **o3** | **20.3%** |
| **o3 (high compute)** | **26.6%** |

→ 远低于 GPQA / AIME，是 2025 的真王。

## 评测协议

- **0-shot**
- **short answer**：free-form text，但通常 1-3 词
- **judge**：LLM judge 评 equivalence（如 "tuʔ" vs "tuh"）

## 实操

src/hle_mock.py 5 题 micro：
```python
from hle_mock import run_hle
from common import make_dummy_model
rs = run_hle(make_dummy_model("foo"))  # 0%
```

## 一句话

> HLE = 2025 推理界的"奥赛决赛"，o3 都只 26%，远未饱和。
