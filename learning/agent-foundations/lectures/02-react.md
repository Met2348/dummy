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
我自己的详细注释版本
```python
def react_loop(question, llm, tools, max_steps):
    history = f"Question: {question}\n"   # 初始的历史就是最初用户输入的question
    for step in range(max_steps):          # 允许尝试多次
        thought_action = llm(history + f"Thought {step+1}:")    # 调用LLM，输入截止目前的完整历史 和 提示LLM 进行新一轮React Loop中第一步： thought 的提示文本。 获取LLM的输出
        if "Final Answer:" in thought_action:    # 获取输出后，首先看LLM是否判定自己获得了最终答案。 若检测到最终答案的出现， 则直接返回最终答案
            return parse_final(thought_action)   # 需要专门的parser来提取最终答案
        action_name, action_args = parse_action(thought_action)  # 若执行到这一步， 说明最终答案还没有出现，则LLM返还的是React第二步 Action相关的信息。包含工具的名称和工具的调用相关参数
        obs = tools[action_name](action_args)    # 在工具列表找到这个工具，传入参数，等待工具输出返回结果 作为 React 第三步 observation的 素材
        history += f"{thought_action}\nObservation {step+1}: {obs}\n"  # 获取新的observation， 改变了LLM对任务状态的认识，更新到表征这个状态的history当中。这里是简单的concat，其实还可以summarize
    return None  # 若执行到这一步返回，则说明用尽了尝试次数，仍没有答案
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
