# Topic 6: Agent Framework Stack（框架横评 + 选型）

> Module 7 第 6 专题 · 12 lectures · ~11h
>
> 10 个 framework 横评 + 决策树 + 同任务 3 framework 对照

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 2025-2026 framework 地图 | (intro) + `survey_taxonomy.py`（论文 taxonomy toy：memory read 公式 + agent design → framework hint，见 `paper/guide_01_survey_large_language_model_based_autonomous_agents.md` §14.1，总览表原漏列已补） |
| L02 | **LangChain** | `langchain_style.py` |
| L03 | **LangGraph** ⭐ deep | `langgraph_style.py` |
| L04 | **LlamaIndex** | `llamaindex_style.py` |
| L05 | **Pydantic AI** | `pydantic_ai_style.py` |
| L06 | **Vercel AI SDK** | `vercel_ai_style.py` |
| L07 | **Claude Agent SDK** ⭐ | `claude_agent_sdk_style.py` |
| L08 | LlamaStack (Meta) | (lecture) |
| L09 | Haystack | (lecture) |
| L10 | Semantic Kernel (MS) | (lecture) |
| L11 | 选型决策树 | `selection_tree.py` |
| L12 | **Capstone**: 同任务 3 framework | `capstone_same_task.py` |

## Tags

- `agent-framework` — Module 7 第 6 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（10/10，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules agent-framework-stack
> ```

10 个脚本全是**手写主流 agent 框架风格复现**（LangChain LCEL `|` 管道 / LangGraph StateGraph / LlamaIndex
VectorStoreIndex / Pydantic AI TypedSchema / Vercel AI SDK generateText / Claude Agent SDK query() 等，
零真实框架依赖、纯 stdlib、纯 CPU）。每个直跑都会执行内置 `_self_test()`（真断言，非 print-only）。
直接 `python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/` 加进 `PYTHONPATH`，且 Python 本身也会把脚本
所在目录插入 `sys.path[0]`，故脱离 harness 单独跑也不依赖 CWD）：

```powershell
# 共享后端：mock_search 关键词命中 / mock_summarize 摘要 / FrameworkRun
python learning/agent-framework-stack/src/common.py
# L01 补充：论文 taxonomy toy（memory_read_score 召回排序 + AgentDesign→framework_hint）
python learning/agent-framework-stack/src/survey_taxonomy.py
# L02 LangChain：LCEL `|` 管道（RunnableSequence/RunnableParallel/PromptTemplate）
python learning/agent-framework-stack/src/langchain_style.py
# L03 LangGraph：StateGraph + conditional edge + interrupt/resume + checkpoint 历史
python learning/agent-framework-stack/src/langgraph_style.py
# L04 LlamaIndex：Document→Node 切块 + VectorStoreIndex + QueryEngine
python learning/agent-framework-stack/src/llamaindex_style.py
# L05 Pydantic AI：TypedSchema 校验/类型强转 + run_sync 重试直到通过 schema
python learning/agent-framework-stack/src/pydantic_ai_style.py
# L06 Vercel AI SDK：generateText 多步 tool loop + streamText 流式 chunk
python learning/agent-framework-stack/src/vercel_ai_style.py
# L07 Claude Agent SDK：query() + permissionMode + preToolUse hook 拦截危险 Bash
python learning/agent-framework-stack/src/claude_agent_sdk_style.py
# L11 框架选型决策树：8 组 criteria→framework 场景断言
python learning/agent-framework-stack/src/selection_tree.py
```

**Capstone（L12）：同一 search+summary 任务用 LangChain LCEL / Vercel AI SDK / Claude Agent SDK 3 种风格实现**

```powershell
python learning/agent-framework-stack/src/capstone_same_task.py
```

> 直跑先打印 `_self_test` 断言（3 框架都产出含 "ReAct" 的摘要，且 Claude Agent SDK 的 LoC < LangChain LCEL），
> 再打印 3-framework 对照 markdown 表。
> （历史版本用过 CWD 依赖的 `python -c "import sys; sys.path.insert(0,'learning/agent-framework-stack/src'); ..."`
> 一行流；已改为直接脚本调用，效果等价但不再依赖"当前目录=repo-root"这个隐藏前提。`paper/guide_01_survey_...md`
> 里仍留着旧一行流写法，未同步改动，两种写法都能跑，直跑版更稳。）

**关键坑注记**

- 全模块**零真实框架 import**（逐文件核实：只 `from __future__`/`dataclasses`/`typing`/`re`，无
  `langchain`/`llama_index`/`pydantic_ai`/`ai`/`anthropic` 包依赖）；`*_style.py` 文件名对应的是"风格复现"而非
  真实第三方库，`environment/verify_env.py` 也明确打印 `[OK] stdlib only`。
- `capstone_same_task.py` 的 3-framework 对照是**真跑出来的**：三个 `run_*_style()` 各自调用真实的
  `langchain_style`/`vercel_ai_style`/`claude_agent_sdk_style` 实现，`to_md()` 的 verdict 由
  `all("ReAct" in r.output for r in runs)` 真实推出；LoC 大小关系（Claude SDK 3 < LangChain LCEL 8）
  由 `_self_test` 显式断言，不是摆样子的硬编码表格。
- `survey_taxonomy.memory_read_score` 是论文 §5.4.1 memory reading 公式
  `score = a·recency + b·relevance + g·importance` 的 toy 落地（`relevance` 用词重叠比例近似），
  与 `paper/guide_01_survey_...md` §5.4.1 描述的公式逐项对应，非拍脑袋实现。

**测试（V2）**

```powershell
python learning/agent-framework-stack/src/tests/test_frameworks.py    # 预期：=== 10/10 modules passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules agent-framework-stack --tests
```

> 注：`test_frameworks.py` 是脚本式聚合器（汇总 10 个模块的 `_self_test`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑。

## 6 framework 决策树

```
Quick PoC?        → CrewAI / Vercel AI SDK
Type-safe required? → Pydantic AI
Complex state machine? → LangGraph
RAG-heavy?        → LlamaIndex
TS-first / Edge?  → Vercel AI SDK
Anthropic stack?  → Claude Agent SDK
C# / .NET?        → Semantic Kernel
Meta stack?       → LlamaStack
```

## 一句话

> 10 framework 横评 + 同 search+summary 任务 3 实现对照 — 看清 framework 是什么。
