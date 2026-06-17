# L07 · Sub-agents — 隔离 context 的派生

## 为什么需要 sub-agent

一个大而噪的子任务(深度搜索、读 20 个文件)会**污染父 agent 的 context**。解法:派一个 sub-agent 在**独立窗口**里干,父 agent 只收一份**精炼摘要**。

```
父 agent (干净窗口)
   │ spawn(goal)
   ▼
sub-agent (自己的窗口,爱多脏多脏)
   │ 干完
   ▼
返回 {result, model_calls, ...}  ← 父只看这个,不看 sub 的中间过程
```

## src 走读

[subagents.py](../src/harness/subagents.py):

```python
def run_subagent(goal, model, registry, budget=2000, max_turns=6, permissions=None):
    ctx = ContextWindow(budget=budget)        # ← 独立窗口(隔离!)
    ctx.add("user", goal)
    result = run_loop(model, registry, ctx, ...)
    return {"goal": goal, "result": result, "ok": result is not None,
            "model_calls": ..., "context_msgs": len(ctx.messages)}  # 留在 sub 内
```

`fan_out(goals, model_factory, registry)` 跑多个 sub-agent(逻辑并行),收集摘要。`_self_test` 验证单个与 fan-out 都成功,且每个 sub 的 `context_msgs` 留在自己窗口里。

## 隔离是关键收益

| | 不用 sub-agent | 用 sub-agent |
|---|---------------|-------------|
| 父窗口 | 被子任务噪声塞满 | 只多一条摘要 |
| 失败影响 | 脏数据进主线 | 困在 sub 内 |
| 并发 | 难 | 天然(各自窗口) |

## 这就是 orchestrator-workers 的运行时

design-patterns 专题 [orchestrator-workers](../../agent-design-patterns/lectures/07-orchestrator-workers.md) 是**设计模式**;sub-agent 是它的**harness 实现**:orchestrator 决定派几个 sub,每个 sub 是一次 `run_subagent`,结果 synthesize。

## 何时用 / 何时别用

- **用**:子任务上下文重、可独立、结果可摘要(研究、批量文件处理)。
- **别用**:子任务很轻(直接调工具就行,别为一次调用起一个 agent——那是 over-engineering)。
- **注意成本**:每个 sub-agent 是一整个 loop,model 调用会累加。fan-out 20 个 = 20 倍 loop 成本。

## 退出条件
- [ ] 说清 sub-agent 的核心收益是 context 隔离
- [ ] 把 sub-agent 对应到 orchestrator-workers 设计模式
- [ ] 知道 fan-out 的成本是线性叠加
