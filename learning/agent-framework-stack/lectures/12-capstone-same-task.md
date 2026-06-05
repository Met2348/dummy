# L12 · Capstone — 同任务 3 Framework ⭐

## 任务

> 同一个 "search + summary" 任务，用 3 个 framework 各实现一遍，对照代码行数 / 抽象层级 / 上手难度。

## 任务：Search and summarize

```
Input:  query = "What is ReAct agent pattern?"
Steps:
  1. Search tool 拿 3 results
  2. Summarize results
Output: 1 段 summary + cite
```

## 3 框架实现

### A. LangChain LCEL

```python
prompt = PromptTemplate("Summarize: {context}")
chain = (
    {"context": retriever_fn, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke("What is ReAct?")
```

### B. CrewAI

```python
researcher = Agent(role="Researcher", goal="search", ...)
writer = Agent(role="Writer", goal="summarize", ...)

t1 = Task(description="Search ReAct", agent=researcher)
t2 = Task(description="Summarize results", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
result = crew.kickoff()
```

### C. Claude Agent SDK style

```typescript
const result = await query({
  prompt: 'Search "ReAct agent" then summarize',
  options: {
    allowedTools: ['WebSearch', 'WebFetch'],
  },
});
```

→ LLM 自己 ReAct loop，1 行调。

## 对照

| 维度 | LangChain | CrewAI | Claude Agent SDK |
|------|----------|--------|------------------|
| LoC | 15 | 12 | 1 |
| 抽象层 | LCEL chain | Role + Task | 直接 LLM call |
| 调试 | LangSmith | 内置 trace | streaming |
| 学曲线 | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| 灵活 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 锁 vendor | ✗ | ✗ | ✓ Anthropic |

## 退出条件

- [ ] 3 个 mock framework 实现
- [ ] 同 query 跑通
- [ ] 输出对照 markdown

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/agent-framework-stack/src'); from capstone_same_task import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 一句话

> 同 search+summary 任务 3 framework 对照 — LangChain 15 行 / CrewAI 12 行 / Claude SDK 1 行，3 种抽象层。
