# L12 · Capstone — 手写 ReAct loop ⭐

## 任务

> 实现 `react_loop(question, mock_llm, tools)`，跑通一个 4-工具任务，trace 完整可读。

## 4 工具

| 工具 | 输入 | 输出 |
|------|------|------|
| `calculator` | 数学表达式 | 数字 |
| `search_mock` | query string | mock 搜索结果 |
| `file_op` | {action, path} | file 内容/状态 |
| `web_mock` | url | mock 网页内容 |

## Mock LLM 实现

mock LLM 用 **pattern match** 模拟决策：
- 看到 "compute" 关键词 → `Action: calculator(...)`
- 看到 "find" 关键词 → `Action: search_mock(...)`
- ...

这样 capstone 可在无真 LLM 的情况下跑通。

## 任务样例

```python
Q = "Find the 2025 most popular LLM name, then compute name_length × 3."
```

期望 trace：

```
Thought 1: Need to search for popular LLM
Action 1: search_mock("2025 most popular LLM")
Obs 1: "Claude (most cited)"

Thought 2: Name is Claude, length = 6
Action 2: calculator("6 * 3")
Obs 2: 18

Thought 3: Final answer ready
Final Answer: 18
```

## 退出条件

- [ ] ReAct loop 5 步内完成
- [ ] 4 工具中至少 2 个被调用
- [ ] Trace 完整可读
- [ ] Final answer == 18

## 实现要点

```python
def react_loop(question, llm, tools, max_steps=8):
    trace = Trace()
    history = build_initial_prompt(question, tools)
    for step in range(max_steps):
        thought_action = llm(history)
        if "Final Answer:" in thought_action:
            trace.add({"final": parse_final(thought_action)})
            return trace
        action, args = parse_action(thought_action)
        if action not in tools:
            obs = f"ERROR: unknown tool {action}"
        else:
            obs = tools[action](args)
        history += f"\n{thought_action}\nObservation: {obs}"
        trace.add({"step": step, "thought_action": thought_action, "obs": obs})
    return trace
```

## 运行

```python
from capstone_react import run_capstone, to_md
result = run_capstone()
print(to_md(result))
```

## 输出预期

```markdown
# ReAct Capstone Result

Q: Find the 2025 most popular LLM name, then compute name_length × 3.

## Trace

| Step | Thought | Action | Observation |
|------|---------|--------|-------------|
| 1 | Need to search | search_mock(...) | Claude (most cited) |
| 2 | Name = Claude (6 chars) | calculator(6*3) | 18 |
| 3 | Final answer ready | (final) | 18 |

## Final: 18 ✓
```

## 一句话

> 60 行 Python 手写 ReAct loop + 4 mock tool → agent 内核祛魅。
