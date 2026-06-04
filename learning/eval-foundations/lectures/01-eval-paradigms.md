# L01 · 评测 4 大范式

## 一图概括

LLM 评测可以分 4 大范式，各自适用场景不同。

| 范式 | 输入 | 信号 | 例子 |
|------|------|------|------|
| **Log-prob** | 一题 + 4 选项 | per-option 似然 | MMLU 经典版 |
| **Generative** | prompt + 让模型自由生成 | answer extract + match | GSM8K、HumanEval |
| **Preference** | 同 prompt 两 response | judge / 真人选 | MT-Bench、Arena |
| **Programmatic** | code / 函数 | exec 跑过 unit test | HumanEval、SWE-Bench |

## 1. Log-prob

```
score(option) = log P(option | prompt)
pred = argmax_i score(option_i)
```

适合 MCQ。优点：无 sampling 噪声、确定性、快。
缺点：需访问 logits（黑盒模型不行）；prompt 模板敏感。

## 2. Generative

```
text = model.generate(prompt, max_new_tokens=N)
pred = extract_answer(text)  # 正则、规则、LLM 抽取
acc = (pred == gold)
```

适合 open-ended、数学、code。优点：贴近真实用法。
缺点：answer extraction 是错误源，需精心写正则。

## 3. Preference

让 judge / 真人在两个 response 中选哪个更好：

```
winner = judge(prompt, resp_A, resp_B)  # in {A, B, tie}
```

聚合：Bradley-Terry / Elo（专题 4 详讲）。

## 4. Programmatic

```python
exec(generated_code + "\n" + unit_test, sandbox)
passed = (no_exception and all_assertions_true)
```

特别适合 code/agent，但 sandbox 安全敏感。

## 选哪个？

```
question type → paradigm
─────────────────────────
MCQ knowledge   → log-prob
math/code/free  → generative
chatbot quality → preference
code/agent      → programmatic
```

## 本 Topic 主要覆盖

L02-L08：log-prob + generative 主流 bench。
L09：lm-evaluation-harness 标准工具。
L10：去污染。
L11：陷阱合集。
L12：Capstone 4 bench 联跑。

## 一句话

> 评测 = 给 LLM 出题 + 对照答案。范式选得对，分数才有意义。
