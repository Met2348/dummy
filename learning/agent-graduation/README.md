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
| L08 | 成本监控 | `eval/cost_tracker.py` |
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

## 跑测试

```powershell
$env:PYTHONIOENCODING="utf-8"; python learning/agent-graduation/src/tests/test_graduation.py
```

## 跑 3 Capstone

```powershell
# Capstone-1: Deep Research Agent
$env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'learning/agent-graduation/src'); from dra.orchestrator import run_capstone_1, to_md; print(to_md(run_capstone_1()))"

# Capstone-2: τ-bench eval pack
python -c "import sys; sys.path.insert(0,'learning/agent-graduation/src'); from eval.dra_eval import run_capstone_2, to_md; print(to_md(run_capstone_2()))"

# Capstone-3: 39-topic Portfolio v2
python -c "import sys; sys.path.insert(0,'learning/agent-graduation/src'); from portfolio_v2 import write_portfolio_v2; print(write_portfolio_v2('portfolio_v2.md'))"
```

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
