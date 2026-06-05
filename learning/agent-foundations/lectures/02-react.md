# L02 · ReAct（Yao 2022）⭐⭐⭐⭐⭐

## 30 秒核心

> ReAct = **Re**asoning + **Act**ing —— LLM 在同一序列里交替输出"想"和"做"。

ICML 2023 经典，几乎所有现代 agent 框架（LangChain/LangGraph/AutoGen/CrewAI）的内核。

## 模板

```
Question: <用户问>
Thought 1: 我应该先 ... 因为 ...
Action 1: search("query")
Observation 1: <tool 返回>
Thought 2: 看到了 ... 接下来 ...
Action 2: calculator("3*4")
Observation 2: 12
...
Thought N: 已有所有信息
Final Answer: <答>
```

## 为什么 work

| 原因 | 解释 |
|------|------|
| Thought 提供 working memory | LLM 推理路径写在 prompt 里 |
| Action 是结构化输出 | 易解析、易调 tool |
| Observation 闭环 | tool 返回写回 prompt，LLM 看得到 |
| 范式简单 | 一个 loop 搞定 |

## 与纯 CoT 区别

| 维度 | CoT | ReAct |
|------|-----|-------|
| 推理 | ✓ | ✓ |
| 工具 | ✗ | ✓ |
| 外部知识 | 静态 | 动态 retrieve |
| 失败可见 | 不 | observation 揭示错误 |

## 实现核心（`react_loop.py` 预告）

```python
def react_loop(question, llm, tools, max_steps=10):
    history = f"Question: {question}\n"
    for step in range(max_steps):
        thought_action = llm(history + f"Thought {step+1}:")
        if "Final Answer:" in thought_action:
            return parse_final(thought_action)
        action_name, action_args = parse_action(thought_action)
        obs = tools[action_name](action_args)
        history += f"{thought_action}\nObservation {step+1}: {obs}\n"
    return None  # 超时
```

## ReAct vs Tool-Use（OpenAI function calling）

| 维度 | ReAct | OpenAI tools |
|------|-------|--------------|
| Thought 显式 | ✓ 文本里 | ✗ 隐式（模型内部） |
| Action 格式 | 文本解析 | 强制 JSON |
| 鲁棒性 | 取决于 parser | 强（schema 验证） |
| 调试 | 看 thought | 看 tool args |

**现代实践**：OpenAI/Anthropic 都把 ReAct 内化，但 LangGraph 仍用显式 ReAct trace 调试。

## 工程注意

| 坑 | 缓解 |
|-----|------|
| LLM 输出格式漂移 | regex + few-shot |
| 死循环 | max_steps + 检测重复 |
| Tool 失败 | observation 写错误 |
| Token 爆炸 | history truncate / summary |

## 关键文献

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao 2022, ICML 2023)

## 退出条件

- 能默写 ReAct 模板（thought-action-obs）
- 能解释为什么 thought 提供 working memory
- 能说出 max_steps 防死循环

## 一句话

> ReAct = Thought-Action-Observation 三段轮转 —— 简单到不可思议，但是几乎所有 agent 框架的内核。
