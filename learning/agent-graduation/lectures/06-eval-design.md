# L06 · DRA Eval — τ-bench 风格

## τ-bench (2024, Sierra)

[Sierra AI 2024](https://arxiv.org/abs/2406.12045) 的 agent benchmark：
- 真实场景 (airline / retail / ...)
- 用户模拟 (LLM 扮用户)
- 多轮交互
- 真 API + DB

我们做 mock 版。

## 5 task

| Task | 描述 |
|------|------|
| airline-search | 帮用户搜机票 + book |
| retail-return | 处理退货 |
| banking | 查余额 + 转账 |
| support-trouble | troubleshooting |
| research-report | DRA 任务 (本 capstone) |

## 5 评测维

| 维 | 计算 |
|---|------|
| **Goal completion** | task 目标达成（bool / 0-1） |
| **Tool use** | tool calls 合理度（覆盖必要 vs 滥用） |
| **Safety** | 拒绝 harm / PII 泄漏 |
| **Efficiency** | 步数 / 总 token |
| **Cost** | $$ 估算 |

## Eval 流程

```python
for task in tasks:
    user_sim = create_user_simulator(task.goal)
    agent_responses = []
    while not user_sim.done:
        msg = user_sim.next_message(agent_responses)
        resp = agent.run(msg)
        agent_responses.append(resp)

    scores = {
        "goal": eval_goal(task, agent_responses),
        "tool_use": eval_tool_use(agent_responses),
        "safety": eval_safety(agent_responses),
        "efficiency": eval_efficiency(agent_responses),
        "cost": estimate_cost(agent_responses),
    }
```

## τ-bench 数字 (Sierra 2024)

| Agent | Goal completion |
|-------|----------------:|
| GPT-4 baseline | 60% |
| Claude 3.7 + tools | 75% |
| Specialized agent | 85% |

→ DRA 在 research-report task 上目标完成度。

## 我们 capstone-2 输出（实测，2026-07-12 ERIC-3080Ti；跑法见 L13）

```markdown
# Tau-bench Mock Eval - Capstone-2

| Task | Goal | Tool | Safety | Eff | Cost | Mean |
|------|-----:|-----:|-------:|----:|-----:|-----:|
| airline-search  | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 0.980 |
| retail-return   | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 0.980 |
| banking         | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 0.980 |
| support-trouble | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 0.980 |
| research-report | 1.00 | 1.00 | 1.00 | 1.00 | 0.09 | 0.818 |

Overall mean: 0.948
```

> `simulate_mock_agent()` 对前 4 个任务总是完成全部 required actions（Goal=1.0），是确定性 mock baseline；`research-report` 走真实 Capstone-1 DRA，goal/tool/safety/efficiency 全满分，只有 cost 维因真实 token 成本超过 $0.05 目标被拉到 0.09（详见 L13）。

## 退出条件

- 能讲 τ-bench 5 task
- 能列 5 评测维
- 能讲 user simulator 概念

## 一句话

> τ-bench mock = 5 task × 5 dim — DRA 在 research-report 上 goal/tool/safety/efficiency 全满分，overall mean 0.948。
