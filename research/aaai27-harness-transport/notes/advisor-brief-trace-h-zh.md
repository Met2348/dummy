# 导师一页简报：TRACE-H 与 Harness Transportability

> **历史简报：** 当前 design-method 版本请读[TRACE-H Policy Transport 导师简报](advisor-brief-trace-h-policy-zh.md)与[约 600 字 Idea](trace-h-idea-600zi-zh.md)。本文以下 patch-effect prediction 内容保留作上一轮决策快照。

## 系统级研究问题

模型与 harness 共同决定 agent 系统行为，但基础模型升级后，现有控制逻辑通常只能整体照搬或重新试错。TRACE-H 提出 **Harness Transportability**：把 harness 从与单一模型绑定的经验工程，提升为可测量、可预测迁移、可审计和可进行风险控制的独立系统层。

> 模型可以替换时，哪些控制逻辑仍应保留，哪些必须删除；能否在任何目标补丁试跑之前，用来源经验和目标 baseline 轨迹作出可靠判断？

完整叙事见[约 600 字 Idea 陈述](trace-h-idea-600zi-zh.md)。

## 统一假说与可证伪落点

核心假说是“需求-响应运输律”：target baseline 暴露模型需要什么，source paired runs 刻画每个 patch 在何种失败状态下能够补救或造成伤害。原子 patch 不是贡献上限，而是让系统级运输规律可识别、可密封检验的实验单元。

```text
Generic headroom controls
  = baseline strength + Metric Freedom

Patch-specific forecast
  = target baseline 中该 patch 的触发机会
  x source models 上该 patch 的条件 rescue/harm

Decision
  = choose none / choose P1-P3 / abstain
```

Metric Freedom 对同一 task/metric 的所有 patch 给出同一个 headroom signal；TRACE-H 要进一步建立 patch-specific transport，并把预测转化为复用、拒绝或暂缓部署的风险决策。

## 最强近邻与精确区别

- **Metric Freedom：** 已占据 baseline-only a priori skill-utility prediction；13 个聚合点上报告总体相关性，并在 GPT-5.1 复核。它不做 patch-specific signed effect、真正封存的 held-out prediction、calibration 或 regret。
- **AHE：** 已在 edit 前写 predicted fixes/regressions，但同一模型/任务内迭代，回归预测 precision/recall 仅 11.8%/11.1%。
- **Self-Harness：** 所谓 held-out 被每个 candidate 反复查询作为接受门，不是 sealed final test。
- **HarnessFix：** GPT-5 mini 上修复的 frozen harness 迁移四个目标模型均提升；cross-model artifact transfer 已不是新颖性。
- **What Should a Skill Remember：** frozen selector 已有，但不预测 target-model-conditioned patch effect。
- **The Harness Effect：** 6-model frozen-loop matrix 表明 baseline strength 是强零假设。
- **A Framework for Evaluating Agentic Skills at Scale：** 约 500 skills、1,000 tasks、19 configurations 已测 individual skill delta，但没有 predictor。

## 最小实验

- 3 source models + 1 sealed target；
- Harness-Bench 固定 36 个离线 tasks；
- baseline + 3 个 event-gated atomic patches；
- 18 个预声明 Metric Freedom probe tasks，每题共 6 次 baseline diagnostic runs；
- 约 936 个完整 primary + diagnostic episodes；
- source leave-one-model-out；
- target 先跑 baseline diagnostics、提交各 patch effect/interval 与 choice，再打开 patch outcomes。

72 小时 pilot 使用 12 tasks、3 models 和约 234 episodes。只有 TRACE-H 在 sealed pseudo-target 上胜过 `MF + source prior` 才继续。

## 当前评价

| 维度 | 第二轮评价 |
|---|---|
| Neatness | 8.4/10；generic headroom 与 patch-specific choice 两层仍可一图讲清 |
| Excitement | 8.3/10；若同一 `F` 下不同 patch 反号且能提前选对，会很强 |
| 问题证据 | 9.5/10；跨模型异质性、负迁移和非单调 benefit 证据充分 |
| 机制证据 | 8.5/10；Metric Freedom、Harness Effect、SEAGym、Harness Updating 共同支撑 |
| Novelty | 7.1/10；generic predictor 已碰撞，但 Harness Transportability 问题、需求-响应运输律、密封评测和风险决策组成更大的系统级主张，须由实验建立 |
| Soundness potential | 8.2/10；强基线、contract、seal、regret，但模型数仍少 |
| 工作量可行性 | 4.8/10；约 100-155 聚焦人时，17 天高风险 |
| 当前项目证据 | 1/10；仍无自己的 pilot |
| Idea 总评 | 7.8/10，低于第一轮 8.2 |
| 当前 AAAI-ready | 2/10 |
| 接受概率 | 当前无条件主观估计 3-8%；若一个 sealed target 明显胜全部 MF/simple baselines，条件估计 20-32% |

## 最关键风险

最危险的不是“文献撞了所以不能做”，而是实验最后只证明 Metric Freedom 或 baseline strength 已经足够。那时 TRACE-H 的 patch-specific 条件分解没有增量，不能靠更长限定词保住主张。

## 当前建议

只批准 collision-aware 72 小时 kill test：必须实现忠实的 Metric Freedom baseline、完成 no-trigger contract、获得至少两个 patch 的足够 opportunity，并在 sealed pseudo-target 上显示 patch-specific choice/regret 增量。若 `MF + source prior` 持平或更好，停止当前 AAAI 主方法；若 opportunity-only 已足够，则主动简化；只有 typed response 再胜出时才保留完整 TRACE-H。

正式方案见 [Proposal](../proposals/trace-h-formal-proposal-zh.md)，最新决策见 [DR-0003](../decisions/0003-metric-freedom-collision-reassessment-zh.md)，逐篇证据见[第二轮全文碰撞审计](../foundations/notes/second-wave-fulltext-collision-audit-zh.md)。
