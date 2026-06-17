# L09 · 光谱另一端 — Autonomous Agent

## 形状

不是固定工作流:模型在**开放循环**里,根据累积的观察自己选工具,直到它判断完成(或撞到护栏)。最大灵活,最大成本与风险。

```
goal → [policy: 模型选动作] → [tool] → observe ─┐
          ↑                                      │
          └──────────── 循环 ────────────────────┘
                            ↓ (模型说 finish / 撞 guard)
                         output
```

## 何时升级到 agent

只有当 L04-L08 的 workflow 都被证明**不够**时。判据(回顾 L02):
- 子任务的**数量和顺序数据驱动**,无法预先画图。
- 任务需要**与环境反复交互**(读结果→决定下一步)。
- 例子:SWE bug 修复、深度研究、计算机操作。

## src 走读

[autonomous_agent.py](../src/patterns/autonomous_agent.py):

```python
def run_agent(goal, policy, tools, max_steps=8, tracker=None):
    state = {"goal": goal, "observations": [], "done": False, "result": None}
    for step in range(max_steps):           # ← max_steps 是最重要的一行
        action = policy(state)
        if "finish" in action:
            state["done"] = True; state["result"] = action["finish"]; break
        obs = tools[action["tool"]](**action.get("args", {}))
        state["observations"].append((action["tool"], obs))
    else:
        trace.add("loop", "max-steps", "hit cap without finishing")
    return PatternResult(..., ok=state["done"])
```

demo:goal="算月度席位预算",policy 确定性地 read_config → multiply → finish。`_self_test` 还验证了一个永不 finish 的 policy 会在 `max_steps` 处停下且 `ok=False`。

## 设计要点(agent 比 workflow 多操心的事)

1. **停机条件(stopping condition)**:模型说 done、达成目标判据、或撞 `max_steps`。三者缺一不可。
2. **循环护栏**:`max_steps` 是底线;更进一步要检测"原地打转"(L12 的 progress guard)。
3. **可观测性是刚需**:轨迹是涌现的,出问题必须有完整 trace 才能复盘(Topic 9 [observability](../../agent-harness-design/lectures/13-observability.md))。
4. **环境反馈质量**:tool 返回得清晰,模型才走得对——错误要 surface 不要 swallow(L12)。

## 成本现实

capstone 里 agent 是**最贵的**(5 次调用)。因为每一步都要先让模型"想一下选什么"(一次 LLM 调用),再执行。开放性的代价就是不可预测的调用数。

## 退出条件
- [ ] 能列出 agent 必备的三类停机条件
- [ ] 理解 max_steps 为何是底线护栏
- [ ] 说清 agent 比 workflow 额外要操心的 4 件事
