# L08 · Agent 模式（5 种典型）

## 1 · ReAct
Thought → Action → Observation → ... → Final
```python
@sgl.function
def react(s, query):
    s += SYSTEM + f"Q: {query}\n"
    for step in range(5):
        s += f"Thought {step}: "
        s += sgl.gen("t", stop=["\n"])
        s += f"\nAction {step}: "
        s += sgl.gen("a", stop=["\n"])
        obs = run_tool(s["a"])
        s += f"\nObservation: {obs}\n"
        if "Final Answer" in s["t"]:
            break
```

**关键**：每轮 KV cache 累积，下一轮直接 append（不重 prefill）→ 比裸 OpenAI API 快 5-10x。

## 2 · Tree of Thought (ToT)
- fork k 路并行思考
- select 最佳一路
- 沿最佳路径继续

## 3 · Self-Consistency
- 同 prompt 生成 N 次（fork(N)）
- 多数投票

## 4 · Multi-Step Tool Use
- 每步 sample tool name + args
- runtime 在外执行
- 结果灌回 prompt

## 5 · Conversational Agent (多轮 chat)
- 历史拼接成 system prompt + N 轮
- radix tree 命中所有先前轮 → KV 全复用
- 新轮只 prefill 新输入

## 6 · 共性：share prefix + branch
所有 agent 模式都符合：
- **share**：父 prompt 在所有子任务中重复使用
- **branch**：子任务独立 decode

正是 RadixAttention + fork 的设计目标。

## 7 · 实现：[agent_patterns.py](../src/agent_patterns.py)
- `react_loop()` mock 版本
- `tree_of_thought()` 用 fork + select
