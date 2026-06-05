# L04 · Reflexion（Shinn 2023）⭐⭐⭐⭐

## 30 秒核心

> Reflexion = ReAct + **失败后写一段 verbal reflection 进 memory** → 下一次试用上。

NeurIPS 2023 best paper finalist。无需 fine-tune，纯 prompt 工程实现"verbal RL"。

## 流程

```
attempt 1
  ReAct loop → 失败（test 不过 / wrong answer）
       ↓
  Reflexion step
  「上一次错在 ___，下次我应该 ___」
       ↓
  写入 episodic memory
       ↓
attempt 2 (新 ReAct loop，加 memory)
  → 成功 / 继续 reflect
```

## 三角色

| 角色 | 任务 |
|------|------|
| Actor | 跑 ReAct (生成 thought-action-obs) |
| Evaluator | 给 trajectory 打分（test 通过 / RM / external check） |
| Self-Reflection | 给失败 trajectory 写一段 verbal lesson |

## 为什么 work

- LLM 自己看自己的失败 trace → 找 bug
- Reflection 用自然语言，不需要 gradient
- Memory 是 prompt 一部分，下次 actor 看得到

## HumanEval 数字（Reflexion 论文）

| Method | pass@1 |
|--------|-------:|
| ReAct only | 80% |
| ReAct + Reflexion | **91%** |

## 实现核心（`reflexion_demo.py` 预告）

```python
memory = []
for trial in range(max_trials):
    trace = react_loop(task, history=memory)
    score = evaluator(trace)
    if score == 1.0:
        return trace
    reflection = llm(f"Trace:{trace}\nReflect:")
    memory.append(reflection)
```

## 与 RLHF 区别

| 维度 | Reflexion | RLHF |
|------|-----------|------|
| 学习信号 | verbal | scalar reward |
| 持久化 | memory prompt | weight update |
| 数据量 | 单 trace | 万级 preference |
| 成本 | 推理时 | 训练时 |

Reflexion 是 **"inference-time RL"** 的最早形式。

## 局限

| 坑 | 缓解 |
|-----|------|
| Evaluator 必需 | 单测/规则/RM/LLM-judge |
| Memory 增长 | summary / 留 top-K |
| 反思错了 | self-consistency 多 reflect 取均 |

## 关键文献

- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn 2023, NeurIPS)

## 退出条件

- 能解释 actor/evaluator/reflect 三角色
- 知道 HumanEval 80% → 91% 提升来自 reflexion
- 能区分 Reflexion vs RLHF

## 一句话

> Reflexion = 让 agent 写"日记反思失败" —— inference-time RL 的最早形式，HumanEval +11%。
