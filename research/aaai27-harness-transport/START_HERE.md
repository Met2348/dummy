# 从这里开始

## 当前主线

> **2026-07-13 执行状态：** Qwen3-4B/8B 本机 source pilot 已完成。17 对 `NONE/REPLAN` 全部为零 terminal utility，当前暂停 branch bank 与 target transport，先执行 Source Policy Gate v2。详见[最终实验报告](experiments/local-none-replan-source-pilot-final-20260713-zh.md)和 [DR-0003D](decisions/0003d-local-source-policy-pivot-zh.md)。这不是对 TRACE-H transport 的反证，因为 transport 尚未运行。

> **Source Policy v2 后续结果：** 8B competence 从0/3提高到2/3，4B仍为0/3；8B困难任务的6对 REPLAN branches 仍无正 advantage。当前应保留v2 executor、停止自然语言 REPLAN，并重新设计结构化 action mechanism。见[证据强度更新](experiments/local-source-policy-v2-evidence-assessment-20260713-zh.md)。

**TRACE-H：在目标模型零 harness-action feedback 的条件下，把来源模型上的运行时控制经验运输成未见 executor 的可执行 Harness policy。**

核心问题：

> 基础模型 dynamics 改变后，能否不在目标任务上试跑 CHECK/RETRY/REPLAN，就从来源模型的事件级 action responses 与目标 baseline states 编译出一个直接提高任务效用的 runtime controller？

核心式：

```text
source same-prefix action branches
  -> event-level response bank

target baseline states
  -> response-aware partial transport
  -> conservative action/NONE runtime policy
```

论文是 design method：输出 executable policy，第一主表是 target success/utility。Effect prediction、alignment、calibration 和 trigger statistics 只用于解释机制，不能作为投稿胜负。

## 先读什么

1. [约 600 字 Idea 完整陈述](notes/trace-h-idea-600zi-zh.md)
2. [术语与符号说明：导师版](notes/trace-h-terminology-guide-zh.md)
3. [Policy Transport 正式 Proposal](proposals/trace-h-policy-transport-proposal-zh.md)
4. [导师一页简报](notes/advisor-brief-trace-h-policy-zh.md)
5. [DR-0003B：Design-method pivot](decisions/0003b-trace-h-policy-transport-pivot-zh.md)
6. [扩张后的全文证据边界](foundations/notes/method-design-evidence-boundary-zh.md)
7. [方法 PK 矩阵](notes/trace-h-method-pk-matrix-zh.md)
8. [72 小时 Design-Method Kill Test](notes/trace-h-72-hour-method-kill-test-zh.md)
9. [本机 314-episode 具体实验计划](experiments/local-development-experiment-plan-zh.md)
10. [2026-07-12 本机实验启动状态](experiments/local-run-status-20260712-zh.md)
11. [2026-07-12 L2/L3 续跑状态](experiments/local-run-continuation-20260712-zh.md)
12. [2026-07-13 Qwen3-4B/8B source pilot 最终报告](experiments/local-none-replan-source-pilot-final-20260713-zh.md)
13. [DR-0003D：源策略停止与修复决策](decisions/0003d-local-source-policy-pivot-zh.md)
14. [本地工作站与 HiPerGator 分阶段执行方案](deployment/local-vs-hipergator-execution-plan-zh.md)
15. [Policy Transport Seal](protocol/policy-transport-seal.md)
16. [Claim Register](protocol/claim-register.md)
17. [MASA 全文证据卡](foundations/method-wave/notes/2605.30723-masa-fulltext-evidence-zh.md)
18. [旧 Patch-Effect Proposal](proposals/trace-h-formal-proposal-zh.md)
19. [第二轮全文碰撞审计](foundations/notes/second-wave-fulltext-collision-audit-zh.md)

## 当前判断

- Offline-RL Harness 已占据固定 executor 上的 learned Harness MDP controller；
- MASA 已占据 target-backbone-conditioned skill evolution/rewriter，并报告强 end-to-end gain；
- SkVM 已占据跨 model+harness 的 capability compiler/runtime；
- Adaptive Auto-Harness 已占据 harness tree 与 solve-time router；
- SkillAdaptor/Bayesian-Agent 已占据 target trajectory feedback adaptation；
- Metric Freedom 已占据 baseline-only generic headroom。

当前方法空间不是再发明 controller、router 或 compiler，而是：

**whole target executor held out、target action outcome 为零、source same-prefix branches 提供 action response、partial transport 处理 target-private states，最终部署 state-conditioned policy 并在 end-to-end utility 上胜现有方法。**

## 当前状态

- Idea 质量评估：约 8.1/10；
- 当前 AAAI-ready 程度：约 2/10；
- 项目自己的方法证据：尚无；本机 runtime、contract、replay 与 synthetic 工程门已通过；
- 当前无条件接受概率主观规划值：3-7%，72 小时 pilot 后重估；
- 工作量估计：155-230 聚焦人时；
- 决策：只先运行约 750-900 runs 的 design-method kill test；
- 算力边界：已确认 16 张 B200 配额；本机只做约 314-episode 工程微型实验，正式 source/target/pilot 使用统一 BF16 B200 runtime；
- AAAI-27 摘要截止：2026-07-21；正文截止：2026-07-28。

不要把“文献证明问题存在”写成“我们的方法已经有效”。

## 不可妥协的停止门

出现以下任一情况时停止当前主张：

- source branch/replay 无法稳定恢复同一 state；
- branch action effects 不可重复；
- TRACE-H 不胜 Source-AW、MF-Gated、kNN、balanced OT 中最强者；
- 只改善 effect MAE、OT alignment 或 process frequency，不改善 target success/utility；
- target action outcome 在 policy freeze 前泄漏；
- cross-family target 出现实质负迁移；
- 2026-07-14 前 pilot runner 与 branch bank 仍不稳定。

## 文档纪律

此后面向导师和研究决策的文档默认中文。论文原题、代码、schema fields、模型名和最终英文投稿稿件保留英文，详见 [中文文档规范](protocol/document-language-policy.md)。

重要结论必须写入 `decisions/`。聊天结论不是持久研究决策。
