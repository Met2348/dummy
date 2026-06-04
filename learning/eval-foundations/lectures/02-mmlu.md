# L02 · MMLU — 57 学科金标准

**Hendrycks et al. 2020** · Massive Multitask Language Understanding

## 数字概览

- **57 学科**: 高校/职业领域（STEM/法律/医学/历史/...）
- **每学科 ~100 题**：共 ~14000 题
- **4 选项 MCQ**：A/B/C/D
- **5-shot 标准**：5 题 demo + 1 题主问

## 评测协议

```python
prompt = "\n\n".join([demo_1, demo_2, demo_3, demo_4, demo_5, question])
pred = argmax_letter_prob(model, prompt)  # log-prob 范式
```

或 generative：

```python
text = model.generate(prompt, max_new_tokens=4)
pred = extract_letter(text)
```

## 为什么它统治了 2020-2024 评测

1. **覆盖广**：57 学科 ≈ "通用知识 + 推理"
2. **难度梯度**：从高中到 PhD
3. **简单复现**：4-MCQ + exact match
4. **第一个 large-scale**：当年比 SuperGLUE 翻倍

## 著名分数

| 模型 | MMLU 5-shot |
|------|-------------|
| GPT-3 (175B, 2020) | 43.9% |
| InstructGPT (2022) | 41.5% |
| GPT-4 (2023) | 86.4% |
| Llama 3.1 405B | 88.6% |
| Claude 3.7 Sonnet | 86.8% |
| 随机 baseline | 25% |

## 5-shot demo 怎么写

```
Question: What is 2+2?
A. 3
B. 4
C. 5
D. 6
Answer: B

Question: What year did WWII end?
A. 1942
B. 1944
C. 1945
D. 1947
Answer: C

...（5 个）...

Question: 真正要问的问题
A. ...
B. ...
C. ...
D. ...
Answer:
```

## MMLU 的"七年之痒"

2024 起暴露问题：
1. **饱和**：top model 都 >85%，分不出高低
2. **污染**：网上到处是 MMLU 题目，训练集偷渡
3. **标注错**：~5% 题目有错答案（社区 audit）
4. **prompt 敏感**：5-shot 顺序换一下，分数变 3-5pp

→ MMLU-Pro（L03）+ Open LLM Leaderboard v2（L05）应运而生。

## 实操：micro-MMLU

我们 src/mmlu_runner.py 内置 12 题 micro-MMLU，覆盖 4 学科：
- high_school_math
- world_history
- computer_science
- biology

```bash
python -c "import sys; sys.path.insert(0,'src'); \
  from common import make_random_model; \
  from mmlu_runner import run_mmlu, summarize; \
  print(summarize(run_mmlu(make_random_model())))"
```

期望：random ≈ 25%，oracle = 100%。

## 一句话

> MMLU 是过去 5 年的"高考"，现在变成"中考"——会做不丢人，会做满分才说明问题。
