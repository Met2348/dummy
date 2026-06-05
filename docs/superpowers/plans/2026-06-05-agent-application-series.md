# Module 7「Agent 应用层」7 专题学习系列规划

> 设计日期：2026-06-05
> 学习仓库：`c:\Workspace\dummy`
> 前置：Module 1 PEFT (3) + Module 3 造 (8) + Module 4 改 (7) + Module 5 用 (7) + Module 6 评 (7) = **32 专题** 已完结
> 视角错位：Module 4 `multimodal-agent` 是 RL 视角训 agent，Module 6 `agent-code-eval` 是评测视角跑测试，**本 Module 7 是工程视角"造 agent 产品"**

---

## 一、为什么是 Module 7

### 已有覆盖
- **Module 4 multimodal-agent**：VLM-R1 / WebRL / SWE-Gym / Safe-RLHF / 五线综合 ✓ (RL 视角)
- **Module 6 agent-code-eval**：HumanEval / MBPP / SWE-Bench / WebArena / GAIA / BFCL / OSWorld ✓ (评测视角)
- **空白**：**怎么从零造一个 agent 产品**？RAG/Tool/Memory/Multi-Agent 框架横向选型，MCP/A2A 协议，工程化部署

### Module 7 目标
完成后用户能：
1. 选定一个真实 use case，从零设计 Agent 架构
2. 选 RAG 策略（naive / hybrid / GraphRAG / Hippo）+ 选 reranker
3. 选 Tool 协议（OpenAI tools / MCP / A2A）+ 写 server
4. 选 framework（LangGraph / CrewAI / AutoGen / Claude Agent SDK）
5. 设计 memory layer（vector / Letta / Mem0）
6. 跑 RAGAS / τ-bench 评测，做迭代

### 2025-2026 关键事件（Module 7 必须覆盖）
| 时间 | 事件 | 影响 |
|------|------|------|
| 2024.11 | Anthropic 发布 **MCP** (Model Context Protocol) | Tool 互操作事实标准 |
| 2025.04 | Google 发布 **A2A** (Agent-to-Agent) protocol | Multi-agent 互操作标准 |
| 2024.10 | Anthropic **Computer Use** | 直接控制屏幕 |
| 2024.11 | Microsoft **Magentic-One** | 通用 multi-agent 框架 |
| 2025 | **Letta** (前 MemGPT) GA | 持久 memory 工业级 |
| 2025 | **Pydantic AI** / **Vercel AI SDK v4** | type-safe agent |
| 2025 | **Claude Agent SDK** | 官方 agent 工具包 |
| 2024-2025 | **GraphRAG** / **HippoRAG** / **LightRAG** | RAG 范式升级 |

---

## 二、7 专题总览

| # | 专题代号 | 一句话定位 | 主题数 | Lecture | 时长 | git tag |
|---|---------|----------|--------|---------|------|---------|
| 1 | `agent-foundations` | Agent 基础范式：ReAct/Reflexion/Plan-Execute/AutoGPT | 12 | 12 | 13h | `agent-foundations` |
| 2 | `rag-essential` | RAG 全谱：naive/hybrid/Reranker/ColBERT/GraphRAG/HippoRAG/HyDE + RAGAS | 14 | 14 | 14h | `rag-essential` |
| 3 | `tool-use-mcp` | Tool use 协议化：function calling/MCP/A2A/Computer Use/Sandbox | 12 | 12 | 12h | `tool-use-mcp` |
| 4 | `multi-agent-orchestration` | Multi-agent 编排：AutoGen/CrewAI/LangGraph/MetaGPT/Magentic-One/Swarm | 13 | 13 | 14h | `multi-agent` |
| 5 | `agent-memory-context` | Memory 层：Letta/Mem0/Episodic/Semantic/Cache + Vector DB 选型 | 12 | 12 | 12h | `agent-memory` |
| 6 | `agent-framework-stack` | 框架战横评：LangChain/LangGraph/LlamaIndex/Pydantic AI/Claude Agent SDK/Vercel SDK | 12 | 12 | 11h | `agent-framework` |
| 7 | `agent-graduation` | **毕业 Capstone**：deep research agent 从零造 + 39-topic 终极 portfolio | 14 | 14 | 16h | `应用-graduation` + `module7-complete` |
| | | **合计** | **89** | **89** | **92h** | — |

对照 Module 6 (86 lecture / 47h)，Module 7 体量约 **2×**，但全部 stdlib + mock，无外部依赖。

### 依赖关系图

```
Topic 1: agent-foundations  (ReAct / 思考 → 工具 → 观察 循环)
        |
        ↓
        ├─→ Topic 2: rag-essential (RAG 工具链)  ──┐
        ├─→ Topic 3: tool-use-mcp (Tool 协议)  ────┤
                                                   ↓
                       Topic 4: multi-agent-orchestration (多 agent 编排)
                                                   ↓
                       Topic 5: agent-memory-context (memory 层)
                                                   ↓
                       Topic 6: agent-framework-stack (框架选型)
                                                   ↓
                       Topic 7: agent-graduation ⭐ (毕业 capstone)
```

---

## 三、Topic 1：Agent 基础范式（`agent-foundations`）

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-overview.md | Agent 是什么 | LLM + perception + action + memory loop |
| 02-react.md | **ReAct** (Yao 2022) | Thought-Action-Observation 范式开山 |
| 03-cot-vs-tot.md | CoT / ToT / GoT | 推理结构演化 |
| 04-reflexion.md | **Reflexion** (Shinn 2023) | 失败后 self-reflect 写入 memory |
| 05-autogpt.md | AutoGPT / BabyAGI | 自循环规划 + tool（早期玩具）|
| 06-plan-execute.md | **Plan-and-Execute** | 先规划全步骤再执行（LangChain）|
| 07-router-orchestration.md | Router pattern | LLM 当路由器 → 子 agent |
| 08-tool-foundations.md | Tool 抽象 | function signature / schema / 调用协议 |
| 09-prompt-patterns.md | System prompt patterns | role / constraints / examples / output schema |
| 10-state-machine.md | Agent as state machine | LangGraph 风格 |
| 11-debugging-agents.md | Agent 调试 | logging / tracing / replay / human-in-loop |
| 12-capstone-react-loop.md | Capstone：手写 ReAct loop | 4 工具 + calculator + search mock |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `common.py` | Tool / ActionResult / Trace dataclass |
| `react_loop.py` | 手写 ReAct loop（10 步 max） |
| `reflexion_demo.py` | self-reflect → 写入 episodic memory |
| `plan_execute.py` | Plan → 多步顺序执行 |
| `router_pattern.py` | LLM-as-router 分发 |
| `state_machine.py` | LangGraph 风格 StateGraph |
| `tools/calculator.py` `tools/search_mock.py` `tools/file_op.py` `tools/web_mock.py` | 4 mock 工具 |
| `tracing.py` | log trace + replay |
| `capstone_react.py` | 4 工具 ReAct loop 跑通 |
| `tests/test_*.py` | 单测 + capstone 退出条件 |

### Capstone：手写 ReAct loop
- 4 工具：calculator / search mock / file op / web mock
- 问题：「Search 2025 年最受欢迎 LLM 是哪个？算它名字字数 × 3」
- 期望：mock LLM → ReAct loop → 5 步内 final answer
- 退出条件：trace 完整 + 4 工具均被调用

---

## 四、Topic 2：RAG 全谱（`rag-essential`）

### 章节规划（14 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-rag-overview.md | RAG 是什么 | retrieve → augment → generate |
| 02-naive-rag.md | Naive RAG | chunk + embed + cosine + top-k |
| 03-chunking-strategies.md | Chunking 策略 | fixed / semantic / proposition / agentic |
| 04-embedding-models.md | Embedding 模型 | text-embedding-3 / BGE / E5 / Voyage / SFR |
| 05-hybrid-retrieval.md | **Hybrid retrieval** | BM25 + vector RRF fusion |
| 06-reranker.md | **Reranker** | Cohere / BGE / Cross-encoder / mxbai |
| 07-colbert.md | **ColBERT** late interaction | token-level MaxSim |
| 08-hyde.md | **HyDE** (Hypothetical) | 让 LLM 编理想答案再 retrieve |
| 09-graph-rag.md | **GraphRAG** (Microsoft 2024) | entity graph + community summary |
| 10-hipporag.md | **HippoRAG** (OSU 2024) | PageRank 启发的图 RAG |
| 11-rag-fusion.md | RAG-Fusion / Multi-Query | 多 query 并行 + RRF |
| 12-self-rag.md | Self-RAG / CRAG | 自评估检索 + 自适应纠正 |
| 13-ragas-eval.md | **RAGAS** + Ragas metrics | faithfulness / answer relevancy / context precision |
| 14-capstone-rag-pipeline.md | Capstone：5 策略对照 | naive vs hybrid vs reranker vs HyDE vs GraphRAG |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `common.py` | Doc / Chunk / Query / RetrievalResult |
| `naive_rag.py` | mock embedding (hash-based) + cosine top-k |
| `chunker.py` | 4 chunking 策略 |
| `bm25_minimal.py` + `hybrid.py` | BM25 + RRF fusion |
| `reranker_mock.py` | mock cross-encoder score |
| `colbert_minimal.py` | token-level MaxSim mock |
| `hyde_demo.py` | LLM hypothesis → retrieve |
| `graph_rag.py` | entity 抽取 + community detection (Louvain mock) |
| `hipporag.py` | PageRank-based |
| `rag_fusion.py` | multi-query + RRF |
| `self_rag.py` | retrieve + self-eval + retry |
| `ragas_metrics.py` | 4 RAGAS 指标 mock 实现 |
| `capstone_rag_compare.py` | 5 策略横向 benchmark |

### Capstone：5 策略横评
- 50 文档 + 20 query
- 5 策略：naive / hybrid / hybrid+rerank / HyDE / GraphRAG
- 指标：RAGAS 4 维 + latency
- 预期：GraphRAG faithfulness 最高，hybrid+rerank balanced

---

## 五、Topic 3：Tool Use & MCP（`tool-use-mcp`）

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-tool-overview.md | Tool calling 范式 | function signature / OpenAI tools |
| 02-function-calling.md | OpenAI function calling | JSON schema / tools array |
| 03-mcp-protocol.md | **MCP** (Anthropic 2024.11) ⭐ | tool/resource/prompt 三大 primitive |
| 04-mcp-server-impl.md | 实现 MCP server | stdio / SSE transport / capabilities |
| 05-mcp-client-impl.md | 实现 MCP client | discover + invoke |
| 06-a2a-protocol.md | **A2A** (Google 2025) ⭐ | agent-to-agent 互操作 |
| 07-computer-use.md | **Computer Use** (Anthropic 2024.10) | screenshot + action |
| 08-sandbox-exec.md | Sandbox / e2b / pyodide | 安全代码执行 |
| 09-streaming-tools.md | Streaming + interrupting tools | 支持中断/恢复 |
| 10-tool-error-handling.md | Tool error 重试 | retry / fallback / circuit-breaker |
| 11-tool-security.md | Tool 注入攻击 | indirect injection / TOCTOU |
| 12-capstone-mcp-stack.md | Capstone：手写 MCP server + 3 工具 | calc/search/file 全 MCP 协议化 |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `common.py` | ToolSchema / ToolCall / ToolResult |
| `openai_tools.py` | function calling JSON 解析 |
| `mcp_protocol.py` | MCP 消息 envelope (JSON-RPC 2.0) |
| `mcp_server.py` | 手写最小 MCP server (stdio) |
| `mcp_client.py` | discover tools + invoke |
| `a2a_minimal.py` | A2A skill exchange minimal |
| `computer_use_mock.py` | screenshot + click action mock |
| `sandbox_mock.py` | restricted exec 白名单 |
| `streaming_tools.py` | async generator tool |
| `tool_retry.py` | exponential backoff |
| `tool_injection_demo.py` | indirect injection 演示 (mock) |
| `capstone_mcp_stack.py` | 3 工具 MCP server 跑通 |

### Capstone：手写 MCP server + 3 工具
- 实现 minimal MCP server (stdio JSON-RPC)
- 暴露 3 工具：calculator / search / file
- 客户端 discover → call → 解析
- 退出条件：3 工具均通过 MCP 协议成功调用

---

## 六、Topic 4：Multi-Agent 编排（`multi-agent-orchestration`）

### 章节规划（13 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-multiagent-overview.md | 多 agent 三大范式 | hierarchical / debate / collaborative |
| 02-autogen.md | **AutoGen** (Microsoft 2023) | conversable agent + group chat |
| 03-crewai.md | **CrewAI** (2024) | role-based crew |
| 04-langgraph.md | **LangGraph** ⭐ | StateGraph + node/edge |
| 05-metagpt.md | **MetaGPT** (DeepWisdom 2023) | SOP-driven 软件公司 |
| 06-magentic-one.md | **Magentic-One** (MS 2024.11) | 通用多 agent 框架 |
| 07-swarm.md | **OpenAI Swarm** (2024.10) | minimal hand-off |
| 08-debate-pattern.md | Multi-agent debate | reflection + critique |
| 09-hierarchical.md | Hierarchical pattern | supervisor + worker |
| 10-agent-communication.md | Agent 通信 | message bus / pub-sub / A2A |
| 11-conflict-resolution.md | 冲突 resolution | voting / weighted / judge |
| 12-cost-multiagent.md | Multi-agent 成本 | token 爆炸 / 何时不该用 |
| 13-capstone-coding-crew.md | Capstone：3 agent coding crew | PM + dev + reviewer |

### Capstone：3-agent coding crew
- PM agent → 拆 spec
- Dev agent → 写代码
- Reviewer agent → 审 + 提改
- 任务：写一个 fizzbuzz + tests
- 退出条件：3 agent 全部参与 + 最终通过测试

---

## 七、Topic 5：Memory & Context（`agent-memory-context`）

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-memory-overview.md | Agent memory 分类 | working / episodic / semantic / procedural |
| 02-memgpt.md | **MemGPT → Letta** | OS-inspired memory paging |
| 03-mem0.md | **Mem0** (2024) | extract → update → retrieve |
| 04-vector-db.md | Vector DB 选型 | Pinecone / Weaviate / Chroma / Qdrant / Milvus / Postgres pgvector |
| 05-episodic-memory.md | Episodic memory | event-based recall |
| 06-semantic-memory.md | Semantic memory | concept extraction + KG |
| 07-context-mgmt.md | Context management | summarization / pruning / compression |
| 08-prompt-caching.md | Prompt caching | Anthropic / OpenAI prompt cache |
| 09-kv-cache-mgmt.md | KV-cache mgmt | PagedAttention / shared prefix |
| 10-long-conv-strategies.md | 长对话策略 | rolling summary / RAG over history |
| 11-personalization.md | 个性化 memory | user profile / preference learning |
| 12-capstone-memory-agent.md | Capstone：3 memory 层 chatbot | working + episodic + semantic |

### Capstone：3 memory 层 chatbot
- working memory（当前对话 turn）
- episodic memory（历次对话事件）
- semantic memory（用户喜好提取）
- 10 turn 对话后能正确 recall 第 1 turn 的用户偏好

---

## 八、Topic 6：Framework 选型横评（`agent-framework-stack`）

### 章节规划（12 lectures）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-framework-landscape.md | 2025-2026 agent framework 地图 | 10+ 家横评 |
| 02-langchain.md | **LangChain** | chain / Runnable / LCEL |
| 03-langgraph.md | **LangGraph** ⭐ | StateGraph + checkpoint + HITL |
| 04-llamaindex.md | **LlamaIndex** | data framework / index 优势 |
| 05-pydantic-ai.md | **Pydantic AI** | type-safe / 输出 schema 强制 |
| 06-vercel-ai-sdk.md | **Vercel AI SDK** | Edge-native / streaming / TS-first |
| 07-claude-agent-sdk.md | **Claude Agent SDK** ⭐ | Anthropic 官方 |
| 08-llamastack.md | LlamaStack (Meta) | Meta 推 |
| 09-haystack.md | Haystack | RAG-focused |
| 10-semantic-kernel.md | Semantic Kernel (MS) | C# 友好 |
| 11-selection-decision-tree.md | 选型决策树 | 4 use case × 6 framework 推荐 |
| 12-capstone-same-task-3-frameworks.md | Capstone：同任务 3 framework | LangGraph vs CrewAI vs Claude SDK |

### Capstone：同任务 3 framework
- 任务：search + summary agent
- 用 LangGraph / CrewAI / Claude Agent SDK 各实现一遍
- 对照：代码行数 / 抽象层级 / 上手难度

---

## 九、Topic 7：Agent 毕业 Capstone（`agent-graduation`）⭐⭐⭐⭐⭐⭐⭐

### 章节规划（14 lectures，毕业系列）

| Lecture | 主题 | 核心 idea |
|---------|------|---------|
| 01-grad-overview.md | Module 7 收官 + 39-topic 全谱 | 跨 7 module 总图 |
| 02-deep-research-agent.md | Deep research agent 是什么 | 多步 plan → tool 调用 → 引用答 |
| 03-architecture-design.md | DRA 架构设计 | planner / retriever / writer / verifier |
| 04-tool-stack-selection.md | Tool stack 选型 | MCP servers 编排 |
| 05-memory-design.md | Memory 设计 | episodic + semantic 各管什么 |
| 06-eval-design.md | Eval 设计 | τ-bench / Theseus 风格 |
| 07-deployment.md | 部署 | FastAPI / streaming / 后台 task |
| 08-cost-monitoring.md | 成本监控 | token + tool call 计费 |
| 09-39-topic-portfolio.md | 39-topic Portfolio v2 | 含 Module 7 |
| 10-career-paths.md | 5 career 路径 | LLM infra / app / research / safety / 产品 |
| 11-self-promotion.md | 如何用 portfolio 求职 | resume / blog / GitHub / talk |
| 12-Capstone-1-DRA.md | **Capstone-1**：deep research agent | 从零造 |
| 13-Capstone-2-eval-pack.md | **Capstone-2**：评测包 | τ-bench mock + 5 维评分 |
| 14-Capstone-3-portfolio-v2.md | **Capstone-3** ⭐⭐⭐⭐⭐⭐⭐ | 39-topic 总收尾 portfolio |

### src/ 规划

| 文件 | 实现 |
|------|------|
| `dra/planner.py` `dra/retriever.py` `dra/writer.py` `dra/verifier.py` | Deep research agent 4 模块 |
| `dra/orchestrator.py` | LangGraph 风格状态机 |
| `dra/tools/` | 5 MCP-style tool（search/cite/calc/file/python） |
| `eval/tau_bench_mock.py` | 5 任务 mock |
| `eval/dra_eval.py` | 5 维评分 + 完整跑 |
| `portfolio_v2.py` | 39-topic generator（含 Module 7） |
| `tests/test_grad.py` | 全部 capstone 退出条件 |

### Capstone-1：Deep research agent 从零造
- 输入："给我一份 2026 年 LLM 推理优化技术报告 (含引用)"
- 流程：planner → 5 步 search → cite → writer → verifier
- 输出：markdown 报告 + 引用列表
- 退出条件：5 步全跑通 + 引用 ≥ 3 条

### Capstone-2：τ-bench mock eval pack
- 5 任务：airline / retail / banking / 5G mock
- 5 维：goal completion / tool use / safety / efficiency / cost
- 输出：md 评分表

### Capstone-3 ⭐⭐⭐⭐⭐⭐⭐：39-topic Portfolio v2
- 39 topic enumerated
- 7 module 时间线
- 全部 capstone 结果汇总
- 选型决策树 v2（含 agent 维度）
- 求职用 5 段「我能做什么」

---

## 十、跨专题工程策略

### 三轨代码策略（同 Module 6 mock 路线）
| Topic | 主要实现方式 | 注意 |
|-------|------------|------|
| 1-7 | Stdlib only mock | 与 Module 6 一致 |

所有 Module 7 模块：
- ✅ **stdlib only** — 无 torch/transformers/langchain/langgraph 依赖
- ✅ **mock LLM**：基于 keyword pattern 模拟 LLM 决策
- ✅ **mock embedding**：hash-based deterministic
- ✅ **mock tools**：硬编码返回值（calculator 真算，search 返回 mock 结果）
- ✅ **CPU only**：无 GPU 需求
- 教育目标：理解协议/范式/数据流，而非真训/真推

### 一致性测试
| 测试 | 标准 |
|------|------|
| ReAct loop trace | 完整 thought-action-obs 三段记录 |
| RAG retrieve | top-k 正确 + RRF 分数算对 |
| MCP server | 3 工具 list + invoke 协议合规 |
| Multi-agent | 3 agent 全参与 + 终止条件 |
| Memory | 10 turn 后能 recall turn 1 |

### Git 里程碑
| Tag | 时机 |
|-----|------|
| `agent-foundations` | Topic 1 末 |
| `rag-essential` | Topic 2 末 |
| `tool-use-mcp` | Topic 3 末 |
| `multi-agent` | Topic 4 末 |
| `agent-memory` | Topic 5 末 |
| `agent-framework` | Topic 6 末 |
| `应用-graduation` + `module7-complete` | Topic 7 末 ⭐⭐⭐⭐⭐⭐⭐ |

---

## 十一、跨 Module 终极图（Module 7 完成后）

```
Module 1 PEFT (3)          ┐
Module 3 造大模型 (8)        │
Module 4 改大模型 (7)        ├─→ 39-topic LLM 全栈工程师 ID 卡 v2
Module 5 用大模型 (7)        │   (出门作品集 + 5 career 路径)
Module 6 评测/安全 (7)       │
Module 7 Agent 应用层 (7)   ┘
─────────────────────────
       39 专题 / ~200h
```

5 大画像 → 6 大画像：
- 造模型 / 改模型 / 用模型 / 评模型 / 守模型 / **造 agent 产品**

---

## 十二、下一步

立即启动 **Topic 1 `agent-foundations`**：
1. 创建 `learning/agent-foundations/` 目录骨架
2. 写 README + environment + 12 lecture + src + tests + notebooks
3. capstone ReAct loop 跑通
4. tag `agent-foundations`

预计 1 天完成 Topic 1，按 Module 5/6 节奏推进。
