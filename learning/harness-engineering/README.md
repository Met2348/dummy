# harness-engineering — 生产级 Agent Harness 工程 + 研究入口

> **`agent-harness-design` 的进阶续作** · 14 讲 + Capstone · 混合重心: 生产工程 + PhD 研究桥
> 从「理解 harness 由哪些零件组成」(已有专题) → 「**工程一个能上生产、能跑长任务、可观测、可评测的 harness**, 并看见其中的研究 gap」。

---

## 为什么现在学这个 (2026 实况)

2026 年模型在商品化 (GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.1 Pro / DeepSeek-V4 数周内齐发, 能力都极强)。
竞争护城河**不在选哪个模型, 而在你给它套的 harness 有多强**:

- 高达 **88%** 的企业 agent 项目到不了生产;
- 其中 **65%** 的失败可追溯到 **harness 缺陷** (context drift / schema 错配 / state 退化), 而非模型推理。

「harness engineering」由此在 2026 成为被正名的工程学科 (Mitchell Hashimoto 2026-02 → OpenAI Codex → LangChain 同周跟进)。本专题就把这层亲手工程一遍。

## 与已有 `agent-harness-design` 的边界 (不重复)

| 维度 | agent-harness-design (理解层) | 本专题 (工程层 + 前沿) |
|---|---|---|
| 模型 | MockModel | provider 抽象 + 流式 + tool 协议 (默认仍 mock, 可接真) |
| context | 基础 compaction | **5 阶段渐进式 compaction** + context folding + state 落盘 |
| 长任务 | 单会话多步 | **long-horizon**: loop-with-hook + 文件系统状态 |
| 可观测 | trace + cost | **OpenTelemetry + LLM span 类型** |
| 评测 | 概念 | **真 harness eval** (同模型换 harness 的分差) |
| 安全 | auto/readonly/ask | 权限门 + destructive hooks + 预算护栏 + 企业 RBAC |
| 视野 | 自己的 mini-harness | Claude Code/Codex/deepagents/AgentCore 案例 + 研究前沿 |

## 14 讲总览

| Lecture | 主题 | 配套代码/notebook |
|---|---|---|
| **Part I 护城河与定义** |||
| L01 | 模型商品化 → harness 是护城河 (88%/65%) | (intro) |
| L02 | harness 的本构定义: 4 要素 inclusion test | (concept) |
| **Part II 升级到生产级** |||
| L03 | 接真模型: provider 抽象 / 流式 / tool 协议 | `src/provider.py` |
| L04 | 5 阶段渐进式 compaction | `src/compaction.py` · N1 |
| L05 | long-horizon 自治: loop-with-hook | `src/long_horizon.py` · N2 |
| L06 | subagent 作为 context firewall + debate | (concept + 复用已有 subagents) |
| L07 | tool/MCP 控制平面 | `src/provider.py` (tool dispatch) |
| L08 | 安全与控制: 权限门/hooks/预算/RBAC | (concept) |
| **Part III 成熟度三件套** |||
| L09 | 生产可观测性: OTel + LLM span | `src/otel_trace.py` |
| L10 | 评测 harness 本身 | `src/harness_eval.py` · N3 |
| L11 | 5 大架构模式 (70 系统实证) | (concept) |
| L12 | portable harness (NL harness 前沿) | (concept) |
| **Part IV 研究入口** |||
| L13 | 用 6 类 gap 雷达扫 harness 开放问题 | 接 `critical-reading-gap` |
| L14 | **Capstone**: 升级 mini-harness 跑长任务 | `src/` 全家桶 |

## 动手 (notebook, 默认无需 API key)

| notebook | 你会真的看到/做什么 |
|---|---|
| `N1-compaction-in-action.ipynb` | 填满上下文窗口, 看 5 阶段 compaction 逐级触发 + token 曲线 |
| `N2-long-horizon-task.ipynb` | loop-with-hook 跨 3 个上下文窗口完成一个长任务, state 走文件系统 |
| `N3-harness-eval.ipynb` | 同一任务, 开/关 compaction 两个 harness 配置的成功率+成本对照 |

## 研究桥 (Part IV — 给你 PhD 用)

harness engineering 是 2026 少数「系统工程 ⨯ NLP 研究」真交叉方向。L13 用你 `critical-reading-gap` 的 **6 类 gap 雷达**扫出 long-horizon / harness-eval / portable-harness / context-folding 等真 gap; Capstone 产出 2-3 张 harness 方向的 **idea 卡** (复用 `critical-reading-gap/templates/idea-card.md`)。

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 核心组件全部通过 ✅
```
Python 3.13 / Windows native; 核心运行只用 stdlib + MockProvider, **无需 GPU、无需 API key**。
接真模型: 装 `anthropic`/`openai` 并设环境变量, 把 `MockProvider` 换成 `AnthropicProvider`/`OpenAIProvider` (L03)。

## 设计/计划

- 设计: `docs/superpowers/specs/2026-06-22-harness-engineering-design.md`
- 计划: `docs/superpowers/plans/2026-06-22-harness-engineering.md`
