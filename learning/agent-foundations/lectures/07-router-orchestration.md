# L07 · Router Pattern

## 30 秒核心

> Router = **LLM 当路由器**，根据输入分发到合适的 sub-agent / tool / chain。

最简单的 multi-agent 入门范式。

## 模板

```
User question
  ↓
[Router LLM]
  ↓
Output: "math_expert" / "code_expert" / "search_agent"
  ↓
Forward to matching agent
  ↓
Final Answer
```

## 实现核心（`router_pattern.py` 预告）

```python
ROUTER_PROMPT = """
Q: {question}
Pick best handler: math / code / search / chat
Output ONLY the name."""

def router(question, llm, sub_agents):
    handler = llm(ROUTER_PROMPT.format(question=question)).strip()
    return sub_agents[handler](question)
```

## Router 种类

| 类型 | 实现 | 适用 |
|------|------|------|
| LLM router | LLM 直接选 | 灵活，token 贵 |
| Embedding router | 用 embedding 找最近 | 快、便宜 |
| Rule router | regex/keyword | 100% 确定时 |
| Hybrid | rule 兜底 + LLM fallback | 生产推荐 |

## LangChain RouterChain

```python
from langchain.chains.router import MultiPromptChain

chain = MultiPromptChain.from_prompts(
    llm,
    {"math": "...", "code": "...", "general": "..."}
)
```

## Hierarchical orchestration

Router 是 multi-agent 的最简形式。进阶：
- **Supervisor pattern**：1 个 supervisor + N 个 worker，supervisor 多次分配
- **Hand-off (Swarm)**：worker 之间互相切换控制
- **State-machine (LangGraph)**：显式 graph 决定哪个节点跑

## Router vs Plan-Execute

| 维度 | Router | Plan-Execute |
|------|--------|--------------|
| 决策 | 1 次（选谁）| N 次（列步骤）|
| 输出 | 直接传给 sub-agent | 自己执行 |
| 失败处理 | sub-agent 自管 | planner 重规划 |

## Router 容易踩的坑

| 坑 | 解 |
|-----|---|
| LLM 输出"math agent" 而非 "math" | strict regex + retry |
| Sub-agent 名字冲突 | namespace |
| 边界 case (既数学又代码) | hybrid / multi-route |

## 退出条件

- 能讲 4 种 router 实现的速度/成本对比
- 知道 router 是 multi-agent 入门

## 一句话

> Router = LLM 当交通警 —— 最便宜的 multi-agent 入门款，4 种实现方式速度成本不同。
