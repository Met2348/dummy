# Topic 7: Agent Graduation（Module 7 收官 + 39 专题 Portfolio v2）⭐⭐⭐⭐⭐⭐⭐

> Module 7 第 7 专题 · 14 lectures · ~16h
>
> **整个 39 专题学习马拉松最后一程** —— Deep Research Agent 从零造 + Portfolio v2

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 评测毕业 + 39-topic 全谱 | (intro) |
| L02 | Deep Research Agent 是什么 | `dra/orchestrator.py` |
| L03 | DRA 架构设计 | `dra/planner.py` |
| L04 | Tool stack 选型 | `dra/tools/` |
| L05 | Memory 设计 | (lecture) |
| L06 | Eval 设计 (τ-bench) | `eval/tau_bench_mock.py` |
| L07 | Deployment | (lecture) |
| L08 | 成本监控 | `dra/common.py`（`DRACost`） |
| L09 | 39-topic Portfolio v2 设计 | `portfolio_v2.py` |
| L10 | 5 career 路径 | (lecture) |
| L11 | 用 portfolio 求职 | (lecture) |
| L12 | **Capstone-1: Deep Research Agent** | `dra/*` |
| L13 | **Capstone-2: τ-bench eval pack** | `eval/dra_eval.py` |
| L14 | **Capstone-3: 39-topic Portfolio v2** ⭐⭐⭐⭐⭐⭐⭐ | `portfolio_v2.py` |

## Tags

- `应用-graduation` — Module 7 收官
- `module7-complete` — Module 7 整体完成
- `series-complete` — **整个 39 专题学习马拉松完成 ⭐⭐⭐⭐⭐⭐⭐**

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（11/11 PASS）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules agent-graduation
> ```

11 个脚本（`dra/` 7 个 + `eval/` 3 个 + `portfolio_v2.py`）全部无 argparse、纯 stdlib（无 torch/transformers/网络依赖）→ 直跑到完成即验证（runbook 里 `v0: false`，跳过 `--help` 探针）：

```powershell
# --- DRA 4 模块 + 共享库（L02-L05）---
python learning/agent-graduation/src/dra/common.py          # 共享数据类型 Plan/Note/Draft/Verified/DRACost（L08 成本模型实际所在文件）
python learning/agent-graduation/src/dra/tools/dra_tools.py # L04 5 mock tool：search/fetch/cite/file_write/python 沙箱 exec
python learning/agent-graduation/src/dra/planner.py         # L03 query → 5 条 sub-question
python learning/agent-graduation/src/dra/retriever.py       # per sub-q search+fetch → Note
python learning/agent-graduation/src/dra/writer.py          # notes → 带 [N] 引用的 markdown 报告
python learning/agent-graduation/src/dra/verifier.py        # claim → source 支持度判定

# --- Capstone-1: Deep Research Agent (L12) ⭐⭐⭐⭐⭐⭐ ---
python learning/agent-graduation/src/dra/orchestrator.py    # planner→retriever→writer→verifier 串联，打印完整报告

# --- eval/ 3 个（L06 + paper guide 01）---
python learning/agent-graduation/src/eval/tau_bench_mock.py     # L06 5 task + mock agent 模拟器
python learning/agent-graduation/src/eval/agent_eval_matter.py  # paper 01 "AI Agents That Matter" 落地：Pareto frontier / cost-controlled score / holdout-by-generality

# --- Capstone-2: τ-bench eval pack (L13) ⭐⭐⭐⭐⭐⭐ ---
python learning/agent-graduation/src/eval/dra_eval.py        # 5 task × 5 dim 评分表，research-report 任务复用 Capstone-1 DRA

# --- Capstone-3: 39-topic Portfolio v2 (L09/L14) ⭐⭐⭐⭐⭐⭐⭐ ---
python learning/agent-graduation/src/portfolio_v2.py         # self-test + 落盘到系统临时目录（不写入 repo）
```

> ℹ️ **无重型依赖，全 CPU**：`environment/verify_env.py` 只检查 Python 版本（"stdlib only"）。`dra_tools.file_write_tool` 只写内存 dict（`_IN_MEMORY_FS`），不落磁盘。
>
> ℹ️ **Capstone-1/2 直跑即打印完整报告**：`orchestrator.py`/`dra_eval.py` 的 `__main__` 在 `_self_test()` 后紧跟 `print(to_md(...))`，与讲义早期版本给出的 `python -c "sys.path.insert(...); ..."` 一行流等价，但不依赖 CWD（旧一行流从非 repo 根目录跑会 `ModuleNotFoundError`）。
>
> ℹ️ **Capstone-3 落盘到系统临时目录**：`portfolio_v2.py` 的 `__main__` 会真实生成 39-topic Portfolio v2 并写到 `%TEMP%/agent_graduation_portfolio_v2.md`（不写入 repo）；若要写到自定义路径，显式调用 `write_portfolio_v2(your_path)`——该函数本身没有默认路径保护，裸相对路径（如 `"portfolio_v2.md"`）会落在调用时的 CWD，从 repo 根直接调用会落进 repo 根，请显式传安全路径（见 L14）。

**测试（V2）**：

```powershell
python learning/agent-graduation/src/tests/test_graduation.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules agent-graduation --tests
```

预期：`=== 11/11 modules passed ===`（逐模块跑 `_self_test()`）。

> 注：`test_graduation.py` 是脚本式 runner（无 pytest `test_*` 函数），`pytest` 会 collect 到 0 个用例（exit 5，"no tests ran"）→ 审计 harness 自动回退直接跑脚本执行真断言。

## 39 专题总览

| Module | 专题 | 完成 |
|---|---|---|
| Module 1 PEFT | prompt / lora / adapter | 3 ✓ |
| Module 3 造大模型 | data / transformer / moe / ssm / long-ctx / scaling / pretraining / small-grad | 8 ✓ |
| Module 4 改大模型 | rl-found / rlhf / dpo / process / r1 / sota-2026 / multimodal | 7 ✓ |
| Module 5 用大模型 | engine / sglang / spec / quant / distrib / prod / grad | 7 ✓ |
| Module 6 评测/安全 | eval-found / reason / agent-code / judge / red-team / safety / grad | 7 ✓ |
| Module 7 Agent 应用层 | agent-found / rag / tool-mcp / multi-agent / memory / framework / grad | 7 ✓ |
| **总计** | **39 专题** | **✓** |

## 6 大画像（毕业 → 完整全栈）

```
1. 造模型 — 从 0 训 GPT-2 / Phi-tiny (Module 3)
2. 改模型 — LoRA / Adapter / DPO / R1-Zero (Modules 1+4)
3. 用模型 — vLLM / SGLang / 量化 / 分布式 (Module 5)
4. 评模型 — 25 bench × judge × Arena (Module 6)
5. 守模型 — 红队 + 4 层防御 + Constitutional Cls (Module 6)
6. 造 agent 产品 — ReAct/RAG/MCP/multi-agent/memory (Module 7)

= 2026 LLM 全栈工程师 ID 卡 v2
```

## 一句话

> Module 7 收官 + 39 专题终极 Portfolio = LLM 全栈工程师 ID 卡 v2。
