# L02 · Deep Research Agent (DRA) 是什么

## 30 秒

> Deep Research Agent = **多步 plan + tool 调用 + cite 答** 的 agent 产品形态。

2024-2025 涌现：
- **ChatGPT Deep Research** (2024.12)
- **Perplexity Deep Research** (2025.02)
- **Claude Deep Research** (2025.04)
- **You.com / Bing AI**

## 与普通 search 区别

| 维度 | Search | Deep Research |
|------|--------|---------------|
| Query 数 | 1 | N (多步规划) |
| Output | 简要答 | 长报告 + cite |
| 引用 | 0-3 | 10-50 |
| 时间 | 秒 | 5-30 分钟 |
| 成本 | $0.001 | $0.05-0.5 |

## 典型流程

```
User: "Write a report on 2026 LLM inference optimization."
   ↓
[Planner] 拆 5-10 个子问题
   ↓
For each sub-question:
   - Search (web / KB)
   - Read top docs
   - Extract relevant
   ↓
[Synthesizer] 汇总成 markdown
   ↓
[Citer] 加引用 [1] [2] ...
   ↓
[Verifier] 检查claim是否被source支持
```

## 我们 DRA 4 模块

| 模块 | 职责 | src |
|------|------|-----|
| Planner | 拆步骤 | `dra/planner.py` |
| Retriever | search + read | `dra/retriever.py` |
| Writer | 综合写报告 | `dra/writer.py` |
| Verifier | claim → source 验 | `dra/verifier.py` |

## Orchestrator 串联

```python
plan = planner(query)         # list of sub-questions
for q in plan:
    docs = retriever(q)
    notes.append({"q":q, "docs":docs})
report = writer(notes)
verified = verifier(report, notes)
```

## DRA 评测维度（τ-bench 风格）

| 维 | 含义 |
|---|------|
| Goal completion | 报告覆盖原 query |
| Tool use | search 调用合理 |
| Safety | 不传播 misinfo |
| Efficiency | 步数 / token 不爆炸 |
| Cost | < budget |

## 我们 capstone-1 任务

> "Write a brief report on 2026 LLM inference optimization techniques."

期望：
- 5 步 plan
- 5 mock tool calls (search / cite / verify / file_write)
- markdown 报告 with [1]-[N] cite
- 3+ 个 citation

## 退出条件

- 能讲 DRA 4 模块
- 能列 5 评测维
- 知道 2024-2025 商业 DRA

## 一句话

> Deep Research Agent = planner + retriever + writer + verifier 4 模块 — search 是秒，DRA 是分钟。
