# harness-engineering 专题设计 spec

> Date: 2026-06-22 · 用户: 博0, NLP/LLM 方向
> 定位: `agent-harness-design` (Module 7 第9专题, 16讲 mini-harness 理解层) 的**进阶续作** —— 从「理解 harness 由哪些零件组成」升级到「生产级 harness engineering + 研究前沿」。
> 重心: **混合 (生产工程 + 研究入口)** —— 用户确认。

## 1. 动因: 为什么现在做 harness engineering

「harness engineering」在 2026 年成为被正名的工程学科 (Hashimoto 2026-02 博客 → OpenAI Codex → LangChain 同周跟进)。核心论点与硬数据:

- **模型商品化**: GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.1 Pro / DeepSeek-V4 数周内齐发, 能力都极强 → 护城河不在选哪个模型, 而在 harness。
- **生产鸿沟**: 高达 88% 企业 agent 项目到不了生产; 其中 65% 失败可追溯到 **harness 缺陷** (context drift / schema 错配 / state 退化), 而非模型推理。
- **本构定义**: harness = 4 必要充分要素 (agent loop / tool interface / context management / control mechanisms); 用它鉴定 Claude Code / Codex CLI / Aider / Cline / OpenHands / SWE-agent。
- **学术化**: 70 系统实证 (5 架构模式 + 统一调度框架); Claude Code 逆向出 5 阶段渐进式 compaction; long-horizon 的 loop-with-hook; Natural-Language Agent Harnesses (arXiv 2603.25723)。

### 对本用户的特殊价值
harness engineering 是 2026 少数「系统工程 ⨯ NLP 研究」真交叉方向, 其中 long-horizon 自治、harness eval、portable harness 都是可做 PhD 的真 gap。用户上周刚建好 `critical-reading-gap` 找-gap 工具 → 本专题 Part IV 直接用它把 harness 变成研究入口。

### 来源 (供 L01/L13 引用)
- The Agent Harness — NJ Raman (Medium, 2026-04)
- Harness Engineering: The Infrastructure Layer — Vishal Mysore (Medium, 2026-05)
- Agent Harness Engineering — O'Reilly Radar / Adnan Masood (Medium, 2026-04)
- awesome-harness-engineering (GitHub, ai-boost)
- AddyOsmani.com — Agent Harness Engineering
- Natural-Language Agent Harnesses — arXiv 2603.25723
- 70-system 实证研究 (2026-04)

## 2. 与已有专题的边界 (不重复造轮子)

| 维度 | agent-harness-design (已有) | harness-engineering (本专题) |
|---|---|---|
| 模型 | MockModel 替身 | provider 抽象 + 流式 + tool 协议 (默认仍 mock, 可接真) |
| context | 基础 compaction | 5 阶段渐进式 compaction + context folding + state 落盘 |
| 长任务 | 单会话多步 | long-horizon: loop-with-hook + 文件系统状态 |
| 可观测 | trace + cost | OpenTelemetry + LLM span 类型 |
| 评测 | 概念 | 真 harness eval (同模型换 harness 的分差) |
| 安全 | auto/readonly/ask | 权限门 + destructive hooks + 预算护栏 + 企业 RBAC |
| 视野 | 自己的 mini-harness | Claude Code/Codex/deepagents/AgentCore 案例 + 研究前沿 |

**Capstone 复用而非重写**: 直接 import / 升级用户已有 `agent-harness-design/src/harness` 的概念骨架, 接上本专题新增的生产级组件。

## 3. 课程结构 (14 讲 + Capstone, 4 Part)

**Part I 护城河与定义**: L01 模型商品化→护城河 (88%/65% 数据) · L02 本构定义 4 要素 + inclusion test
**Part II 升级到生产级**: L03 接真模型 (provider 抽象) · L04 5 阶段 compaction · L05 long-horizon (loop-with-hook) · L06 subagent context firewall + debate · L07 tool/MCP 控制平面 · L08 安全与控制
**Part III 成熟度三件套**: L09 OTel 可观测 · L10 harness eval · L11 5 大架构模式 (70 系统) · L12 portable harness (NL harness 前沿)
**Part IV 研究入口**: L13 用 6 类 gap 雷达扫 harness 开放问题 · L14 Capstone

## 4. 运行原则 (notebook 必须可跑)

- 默认 **MockProvider** (确定性, 模拟流式 + tool-call), 无需 API key, Windows native 可跑。
- 真 provider 适配为可选 (有 key 才用), 不阻塞教学。
- stdlib + 少量已验证依赖 (nbformat/pandas/matplotlib); 复用 critical-reading-gap 已装环境。
- 所有 .py 顶部 `sys.stdout.reconfigure(utf-8)` 防 Windows GBK 崩溃 (已知坑)。

## 5. Capstone 交付

升级版 mini-harness, 跑通一个**跨上下文窗口的长任务**, 包含:
1. provider 抽象 (mock 默认)
2. 5 阶段 compaction 真触发
3. loop-with-hook: 拦截 early-stop, 新窗口 + 文件系统读回 state, 直到 completion-goal
4. OTel 式 trace (reasoning span + tool child span)
5. 一个 harness eval: 同一任务, 关/开 compaction 两个 harness 配置的成功率/成本对照
6. **产出 2-3 张 harness 方向 idea 卡** (复用 `critical-reading-gap/templates/idea-card.md`)

## 6. 成功标准

- [ ] 14 讲齐全, 研究生级 (图 + 公式逐项 + 2026 实况与来源)
- [ ] src 生产级组件可 import 可测
- [ ] ≥3 notebook nbconvert 跑通 (compaction / long-horizon / harness-eval)
- [ ] Capstone 跨窗口长任务跑通 + 产出 idea 卡
- [ ] 明确标注「工程 → 研究」桥: Part IV 真的接上 critical-reading-gap
- [ ] portfolio 更新 (Module 7 进阶 / 工程⨯研究交叉)
