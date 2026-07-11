# Topic 4: Multi-Agent Orchestration（多 agent 编排）

> Module 7 第 4 专题 · 13 lectures · ~14h
>
> AutoGen / CrewAI / LangGraph / MetaGPT / Magentic-One / Swarm 6 大编排范式

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 多 agent 三大范式 | `camel_role_play.py`（前置：CAMEL role-play 双 agent 先驱，见 `paper/guide_01_camel.md`） |
| L02 | **AutoGen** (MS 2023) | `autogen_mock.py` |
| L03 | **CrewAI** (2024) | `crewai_mock.py` |
| L04 | **LangGraph** ⭐ | `langgraph_mock.py` |
| L05 | **MetaGPT** (DeepWisdom 2023) | `metagpt_mock.py` |
| L06 | **Magentic-One** (MS 2024.11) | `magentic_one_mock.py` |
| L07 | **OpenAI Swarm** (2024.10) | `swarm_mock.py` |
| L08 | Debate pattern | `debate.py` |
| L09 | Hierarchical (supervisor + worker) | `hierarchical.py` |
| L10 | Agent 通信 (msg bus / pub-sub) | `message_bus.py` |
| L11 | 冲突 resolution (vote/weighted/judge) | `conflict_resolution.py` |
| L12 | Multi-agent 成本 (token 爆炸) | `cost_analyzer.py` |
| L13 | **Capstone**: 3-agent coding crew | `capstone_coding_crew.py` |

## Tags

- `multi-agent` — Module 7 第 4 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V1 验证通过（14/14，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules multi-agent-orchestration
> ```

14 个脚本全是**手写多 agent 编排范式 mock**（零外部依赖、纯 stdlib、纯 CPU）。每个直跑都会执行内置
`_self_test()`（真断言，非 print-only）。直接 `python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/`
加进 `PYTHONPATH`，且 Python 本身也会把脚本所在目录插入 `sys.path[0]`，故脱离 harness 单独跑也不依赖 CWD）：

```powershell
# 共享后端（AgentMessage/AgentReply/CostTracker/mock pattern-agent）
python learning/multi-agent-orchestration/src/common.py
# L01 前置：CAMEL role-play 双 agent 先驱（inception prompting + 失败模式检测，见 paper/guide_01_camel.md）
python learning/multi-agent-orchestration/src/camel_role_play.py
# L02 AutoGen（ConversableAgent + GroupChat）
python learning/multi-agent-orchestration/src/autogen_mock.py
# L03 CrewAI（Agent/Task/Crew，sequential + hierarchical）
python learning/multi-agent-orchestration/src/crewai_mock.py
# L04 LangGraph（StateGraph：reducer + conditional edge + checkpoint）
python learning/multi-agent-orchestration/src/langgraph_mock.py
# L05 MetaGPT（SOP pipeline：PM→Architect→Engineer→QA）
python learning/multi-agent-orchestration/src/metagpt_mock.py
# L06 Magentic-One（Orchestrator + Task/Progress ledger）
python learning/multi-agent-orchestration/src/magentic_one_mock.py
# L07 OpenAI Swarm（hand-off：函数返回触发转移）
python learning/multi-agent-orchestration/src/swarm_mock.py
# L08 Debate pattern（多轮批评 + 多数投票裁决）
python learning/multi-agent-orchestration/src/debate.py
# L09 Hierarchical（supervisor 关键词路由 + N worker）
python learning/multi-agent-orchestration/src/hierarchical.py
# L10 Agent 通信（pub-sub 消息总线）
python learning/multi-agent-orchestration/src/message_bus.py
# L11 冲突 resolution（majority/weighted/tie-break/borda/judge）
python learning/multi-agent-orchestration/src/conflict_resolution.py
# L12 Multi-agent 成本（单 vs 多 agent token 爆炸对比）
python learning/multi-agent-orchestration/src/cost_analyzer.py
```

**Capstone（L13）：3-agent hierarchical coding crew（PM + Engineer + Reviewer）**

```powershell
python learning/multi-agent-orchestration/src/capstone_coding_crew.py
```

> 直跑先打印 `_self_test` 断言（verdict PASS、5/5 tests），再打印完整 crew 报告的 markdown（PM spec /
> Engineer code / Reviewer 逐条 PASS/FAIL / 最终 verdict / token 成本）。
> （历史版本用过 CWD 依赖的 `python -c "import sys; sys.path.insert(0,'learning/multi-agent-orchestration/src'); ..."`
> 一行流；已改为直接脚本调用，效果等价但不再依赖"当前目录=repo-root"这个隐藏前提。）

**关键坑注记**

- 全模块 mock agent/LLM 都是**关键词 pattern-match 或模板拼接**（`make_mock_agent`、各框架的 `execute_fn`），
  不是真模型调用——这是刻意设计（让 6 大编排范式 + CAMEL 的控制流可无 GPU / 无网络地跑通 self-test），不是"未完成"。
- 14 个脚本自检都是**秒级真断言**（非 no-op）：如 `langgraph_mock` 断言状态机在条件边下走满 4 步且 checkpoint
  续跑幂等、`camel_role_play` 断言正常轨迹 0 failure 而刻意构造的坏轨迹能查出 `role_flipping`/`flake_reply`/
  `infinite_loop`、`cost_analyzer` 断言多 agent 场景 token/成本比单 agent 高 >10×（实测 16.7×）、`capstone_coding_crew`
  断言 3 agent 全参与且 5/5 测试通过——都不是硬编码打印。
- `camel_role_play.py`（CAMEL, Li 2023.03）是本模块最早的论文，也是 AutoGen 等后续框架 role-play 设计的先驱；
  原 README 未列出此脚本（现已补），运行/测试链路本身一直正常。

**测试（V2）**

```powershell
python learning/multi-agent-orchestration/src/tests/test_multi_agent.py    # 预期：=== 14/14 modules passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules multi-agent-orchestration --tests
```

> 注：`test_multi_agent.py` 是脚本式聚合器（汇总 14 个模块的 `_self_test`，含 `camel_role_play`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑。

## 6 框架选型决策树

| 场景 | 推荐 |
|------|------|
| 快速 PoC | CrewAI (role-based, 简洁) |
| 复杂状态机 | LangGraph |
| 研究/实验 | AutoGen |
| SOP 化软件开发 | MetaGPT |
| 通用 supervisor | Magentic-One |
| 极简 hand-off | Swarm |

## 关键文献

- **CAMEL**: Communicative Agents for "Mind" Exploration of LLM Society (Li 2023.03, NeurIPS 2023) — 最早的双 agent role-play 先驱
- AutoGen (Microsoft 2023.10)
- CrewAI (joaomdmoura 2024)
- LangGraph (LangChain 2024)
- MetaGPT (Hong 2023, DeepWisdom)
- Magentic-One (Microsoft 2024.11)
- OpenAI Swarm (2024.10, experimental)

## 一句话

> 6 框架共一个内核：N agent + 消息总线 + 终止条件 — 手写一遍看清 framework 是什么。
