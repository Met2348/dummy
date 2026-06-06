# L11 · 用 Portfolio 求职

## 4 个 channel

| Channel | 内容 |
|---------|------|
| **GitHub** | Portfolio.md as README |
| **LinkedIn** | 一段总结 + 链接 |
| **Blog (Medium / Substack / personal)** | 每专题 detailed write-up |
| **Twitter / X** | 短 thread per topic |

## GitHub README 模板

```markdown
# 39-Topic LLM Engineering Portfolio

## TL;DR
Completed 39 deep-dive topics covering造-改-用-评-守-造 agent.

## What I built
- DRA: deep research agent (3 modules, GRPO-trained)
- R1-zero replication on GPT-2-M
- mini-HELM eval framework
- ...

## Skills
- Python / TypeScript / PyTorch
- vLLM / SGLang / LangGraph / Claude SDK
- DPO / GRPO / RLHF / R1-style training
- RAG (5 strategies) / Tools (MCP) / Multi-agent

## Selected projects
[Link to each capstone]

## Contact
- Email
- LinkedIn
- Blog
```

## LinkedIn post 模板

```
🚀 Just completed a 39-topic LLM engineering deep-dive!

From scratch:
✓ Trained GPT-2 small (1B tokens)
✓ Replicated R1-Zero on GPT-2-M
✓ Built deep research agent (planner + retriever + writer + verifier)
✓ Evaluated 6 RAG strategies (GraphRAG winner)
✓ Designed 4-layer safety defense

Tech: PyTorch / vLLM / LangGraph / Anthropic SDK / MCP

Looking for AI engineer roles in [scope].

Repo: [link]
Portfolio: [link]

#LLM #AIEngineering #OpenSource
```

## Blog post 模板（per topic）

```markdown
# How I built a deep research agent (from scratch)

> Part 36 of 39 in my LLM engineering deep-dive.

## TL;DR (2 sentences)
## Problem
## Approach
## Implementation (code + diagram)
## Results
## What I learned
## What's next
```

## Twitter thread 模板

```
1/N 🧵 Building a deep research agent from scratch.

2/N Problem: ChatGPT Deep Research costs $$$. Can we replicate the core idea?

3/N Architecture: planner → retriever → writer → verifier.

4/N [code snippet]

5/N Results: 80% τ-bench goal completion, $0.05 per run.

6/N Repo: [link]
```

## 面试讲解

每 capstone 准备 1 分钟讲解：
- 问题
- 我的方案
- 关键 trade-off
- 数字结果
- 失败教训

10 capstone × 1 min = 10 分钟 portfolio 讲解。

## 反例

| 反 | 不要 |
|----|-----|
| "Used LangChain" | 太抽象 |
| 只列 stars | 没意义 |
| 没 demo | 没说服力 |
| 没 cost analysis | 没工程感 |
| Copy-paste tutorial | 没原创 |

## 退出条件

- 能写 GitHub README
- 能写 LinkedIn post
- 能 1 分钟讲一个 capstone

## 一句话

> 4 channel (GitHub / LinkedIn / Blog / X) + 1 分钟 per capstone — Portfolio v2 是出门作品集。
