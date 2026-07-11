# Topic 1: Agent Foundations（Agent 基础范式）

> Module 7「Agent 应用层」第 1 专题 · 12 lectures · ~13h
>
> 工程视角：从 ReAct 到 LangGraph-style state machine，理解 agent 五大范式

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Agent 是什么 | (intro) |
| L02 | **ReAct** (Yao 2022) | `react_loop.py` |
| L03 | CoT / ToT / GoT | (lecture) |
| L04 | **Reflexion** | `reflexion_demo.py` |
| L05 | AutoGPT / BabyAGI | (lecture) |
| L06 | **Plan-and-Execute** | `plan_execute.py` |
| L07 | Router pattern | `router_pattern.py` |
| L08 | Tool 抽象基础 | `tools/` |
| L09 | System prompt patterns | (lecture) |
| L10 | Agent as state machine | `state_machine.py` |
| L11 | Agent 调试 | `tracing.py` |
| L12 | **Capstone**: 手写 ReAct loop | `capstone_react.py` |

## Tags

- `agent-foundations` — Module 7 第 1 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（12/12，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules agent-foundations
> ```

12 个脚本全是**手写 agent 范式 + mock LLM / mock 工具**（零外部依赖、纯 stdlib、纯 CPU）。每个直跑都会执行内置
`_self_test()`（真断言，非 print-only）。直接 `python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/`
加进 `PYTHONPATH`，且 Python 本身也会把脚本所在目录插入 `sys.path[0]`，故脱离 harness 单独跑也不依赖 CWD）：

```powershell
# 共享后端（Tool/Trace/parse_action/parse_final/mock pattern-llm）
python learning/agent-foundations/src/common.py
# L02 ReAct（Yao 2022 thought-action-observation 循环）
python learning/agent-foundations/src/react_loop.py
# L04 Reflexion（Shinn 2023 actor+evaluator+self-reflect 多轮试错）
python learning/agent-foundations/src/reflexion_demo.py
# L06 Plan-and-Execute（先规划一次，再顺序执行）
python learning/agent-foundations/src/plan_execute.py
# L07 Router pattern（LLM 路由 + embedding cosine 路由）
python learning/agent-foundations/src/router_pattern.py
# L08 4 个 mock 工具（也可单独直跑自检）
python learning/agent-foundations/src/tools/calculator.py
python learning/agent-foundations/src/tools/search_mock.py
python learning/agent-foundations/src/tools/file_op.py
python learning/agent-foundations/src/tools/web_mock.py
# L10 StateGraph（手写 LangGraph 风格节点/边/条件边）
python learning/agent-foundations/src/state_machine.py
# L11 Tracing/replay（trace<->json 往返 + cost_summary 估算）
python learning/agent-foundations/src/tracing.py
```

**Capstone（L12）：ReAct loop + 4 mock 工具**

```powershell
python learning/agent-foundations/src/capstone_react.py
```

> 直跑先打印 `_self_test` 断言（Final Answer==18，且至少调用 2 个工具），再打印完整 trace 的 markdown 报告。
> （历史版本用过 CWD 依赖的 `python -c "import sys; sys.path.insert(0,'learning/agent-foundations/src'); ..."`
> 一行流；已改为直接脚本调用，效果等价但不再依赖"当前目录=repo-root"这个隐藏前提。）

**关键坑注记**

- 全模块 mock LLM 都是**关键词 pattern-match**（`make_pattern_llm`），不是真模型调用——这是刻意设计
  （让 5 个范式的控制流可无 GPU / 无网络地跑通 self-test），不是"未完成"。
- 12 个脚本自检都是**秒级真断言**（非 no-op）：如 `react_loop` 断言 `trace.final=="5"` 且走 2 步、
  `reflexion_demo` 断言 3 次试错中第 2 次成功、`state_machine` 断言路径与终态、`capstone_react`
  断言最终答案 `18` 且至少调用 2 个工具——都不是硬编码打印。
- `src/tools/*.py` 各自在文件顶部手动 `sys.path.insert` 回 `src/`，所以既可以被 `tools/__init__.py`
  当包导入（`from tools import ALL_TOOLS`），也可以单独当脚本直跑自检，两条路径互不冲突。

**测试（V2）**

```powershell
python learning/agent-foundations/src/tests/test_agent_foundations.py    # 预期：=== 12/12 modules passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules agent-foundations --tests
```

> 注：`test_agent_foundations.py` 是脚本式聚合器（汇总 12 个模块的 `_self_test`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑。

## 关键文献

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao 2022)
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn 2023)
- LangGraph: stateful agent applications
- Plan-and-Solve Prompting (Wang 2023)
- Toolformer (Schick 2023)
- AutoGPT / BabyAGI (2023 玩具开源)

## 5 范式对照

| 范式 | 特征 | 适用 |
|------|------|------|
| ReAct | thought-action-obs loop | 通用，灵活 |
| Reflexion | 失败后 self-reflect | 多次试错 |
| Plan-and-Execute | 先全规划再执行 | 任务可预先分解 |
| Router | LLM 当路由分发 | 多 expert agent |
| State machine | StateGraph 明确节点 | 流程清晰可控 |

## 与 Module 6 / Module 4 区别

| Module | 视角 | 重点 |
|--------|------|------|
| Module 4 multimodal-agent | RL | 训练 VLM agent |
| Module 6 agent-code-eval | 评测 | 跑 SWE-Bench/WebArena |
| **Module 7 agent-foundations (本)** | **工程** | **手写 agent 范式** |

## 一句话

> 5 个 agent 范式 (ReAct / Reflexion / Plan-Execute / Router / State machine) — 用 stdlib 全部手写一遍，理解 agent 的"内核"。
