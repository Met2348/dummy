# L06 · Generative RM (GenRM) — RM 用 CoT 推理打分

> 14 slides | 40 min | RM 新范式

---

## Slide 1 · 动机

传统 RM:
```
Question + Answer → BERT-style encoder → scalar
```

问题：
- 不可解释（为什么这么打分？）
- 难处理复杂推理（数学步骤）
- 弱 base model 当 RM → 弱信号

GenRM 思路：**RM 也是 LLM，用 CoT 推理**。

---

## Slide 2 · GenRM 核心

```
prompt: "Question: ... Answer: ... Critique step-by-step and give a score from 1-10."

generate: "Step 1 is correct... Step 2 has error... Score: 5"

extract: 5/10 = 0.5
```

→ RM 自己写 critique 再打分。

---

## Slide 3 · 优势

| 维度 | scalar RM | GenRM |
|------|----------|-------|
| 可解释 | ✗ | **✓** |
| 利用 base 推理能力 | ✗ | **✓** |
| 数学/代码强 | 中 | **强** |
| 推理速度 | 快 | 慢 ~10x |
| 训练 | BT loss | SFT/DPO |

---

## Slide 4 · 训练方法

GenRM 训练有 2 种:
1. **SFT on critique data**: (q, a, gold_critique, gold_score)
2. **DPO on critique pairs**: (q, a, critique_chosen, critique_rejected)

→ 2025 主流是 DPO（不需要 gold critique 数据）。

---

## Slide 5 · 实测效果

数学任务（MATH）:
| RM 类型 | accuracy |
|---------|---------|
| scalar RM (BERT) | 68% |
| scalar RM (LLaMA-7B) | 72% |
| **GenRM (LLaMA-7B + CoT)** | **78%** ⭐ |

→ 同 base 提升 6pp。

---

## Slide 6 · 推理时使用

```python
def get_reward(q, answer):
    prompt = build_genrm_prompt(q, answer)
    critique = judge_llm.generate(prompt, max_new_tokens=200)
    score = parse_score(critique)
    return score
```

→ 一次 generate ~200 token。

---

## Slide 7 · 推理加速

GenRM 推理慢 → BoN 应用受限。
解决:
- batched generate
- shorter critique (max 100 token)
- early-exit (找到 "Score:" 即停)

仍比 scalar RM 慢 3-5x。

---

## Slide 8 · 多样本聚合

GenRM 输出有随机性，可：
```
N = 4 samples → 平均 score
```

减小 variance，但成本 ×4。trade-off。

---

## Slide 9 · 与 RLAIF 关系

RLAIF = LLM 当 judge 产 preference data → 训 RM。
GenRM = LLM 直接当 RM。

→ GenRM 是 RLAIF 的逻辑终点：跳过 RM 训练。

---

## Slide 10 · 缺点：判官能力的天花板

GenRM 的 score 上限 = 判官 LLM 的能力。
- 判官弱 → score noise 大
- 判官强 → score 信号强

→ 选大模型当 GenRM（GPT-4 / Claude）。

---

## Slide 11 · Self-Taught Evaluator (Meta 2024.08)

进阶: 用 LLM 自合成 GenRM 训练数据：
```
1. LLM 生成 N answers
2. LLM 当 judge 选 chosen/rejected
3. 训 GenRM on synthetic preferences
4. 训出的 GenRM 再当 judge
```

→ 自我增强循环。

---

## Slide 12 · JudgeBench (2024)

系统评估"LLM 当 judge"基准：
- 200 prompts
- ground-truth 对照
- 各 judge 模型 (GPT-4 / Claude / Gemini / Llama-70B) accuracy

Anthropic Claude 3 Opus 当 judge 准确率最高 (~85%)。

---

## Slide 13 · 工业实践

```
RLHF pipeline 演进:
2022: BERT-RM
2023: LLM-RM (scalar head)
2024: GenRM (CoT)
2025: Self-Taught GenRM (synthetic data)
```

→ RM 越来越像 mini-LLM。

---

## Slide 14 · 一句话总结

> GenRM = RM 用 CoT 推理打分。可解释 + 推理强 + 数学/代码 +6pp。代价：推理慢 10x。

下一讲 L07 — Skywork-Reward V2 (8B 击败 70B)。
