# L08 · Multi-Agent Debate（辩论范式）

## 30 秒核心

> 同一问题给 N agent 各自答 → 互相 critique → 多轮收敛 → judge 选 best。

Du 2023 "Improving Factuality and Reasoning in Language Models through Multiagent Debate" 系统化。

## 流程

```
Round 1
  Agent A → answer A1
  Agent B → answer B1
  Agent C → answer C1

Round 2 (each sees others' R1)
  A → critique B1, C1 → answer A2
  B → critique A1, C1 → answer B2
  C → critique A1, B1 → answer C2

...

Round N
  Judge → pick best | majority vote
```

## 为什么 work

| 原因 | 解释 |
|------|------|
| Critique 暴露错 | A 看自己难发现错，看 B 易 |
| 多视角 | 不同 prompt → 不同失败模式 |
| Self-correction | 多 round 收敛到一致答案 |

## 数字（Du 2023, GSM8K）

| Method | acc |
|--------|----:|
| GPT-3.5 single | 77% |
| Self-consistency (1 agent × 4 samples) | 81% |
| Multi-agent debate (3 agent × 3 round) | **85%** |

## ChatEval (Chan 2023)

不同 personality 的 agent 辩论：
- 角色 1: 严格批评家
- 角色 2: 富有同情心评审
- 角色 3: 中性观察者

→ 评估任务上 ChatEval 比 single GPT-4 强。

## Society of Minds (Wang 2023)

```
Each agent has expertise:
  - Logic specialist
  - Math specialist
  - Common-sense specialist
  - Counterfactual specialist
```

→ 像"专家委员会"。

## 实现 (`debate.py` 预告)

```python
def debate(question, agents, rounds=3, judge=None):
    answers = [{"agent":a.name, "answer":a.answer(question)} for a in agents]
    for r in range(1, rounds):
        new_answers = []
        for a in agents:
            others = [ans for ans in answers if ans["agent"] != a.name]
            new = a.critique_and_revise(question, others)
            new_answers.append({"agent":a.name, "answer":new})
        answers = new_answers
    if judge:
        return judge(question, answers)
    return majority_vote(answers)
```

## 何时不用 debate

| 反 | 原因 |
|----|------|
| 简单 lookup | 1 个就答对 |
| Latency 严格 | 多 round 慢 |
| Token 限 | N agent × N round 成本爆炸 |

## 退出条件

- 能默写 N×K 流程
- 知道 GSM8K 数字
- 能写 debate 函数

## 一句话

> Multi-agent debate = N agent × K round critique → judge — 比 single-agent self-consistency 高 4-8pp。
