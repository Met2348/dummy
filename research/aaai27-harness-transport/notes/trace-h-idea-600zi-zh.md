# TRACE-H Idea：约 600 字完整陈述

当基础模型升级时，旧 agent harness 往往被直接照搬或在目标任务上重新搜索，但 harness 本质上是作用于执行轨迹的控制策略：同一个检查、重试或重规划动作，在不同模型上可能补救失败，也可能浪费预算甚至破坏本来正确的行为。现有工作已经能够在固定 executor 上学习 harness controller，也能利用目标模型反馈改写或编译 skills，却仍依赖原 executor 的离线数据或目标侧干预试错。TRACE-H 研究更困难的 Cross-Executor Harness Policy Transport：在看不到任何目标 harness-action outcome 时，把来源模型上的控制经验运输成一个可直接部署于未见模型的动态策略。

我们提出 Branch-and-Transport 机制。首先在来源 executor 的同一 trajectory prefix 上分支执行 NONE、CHECK、RETRY、REPLAN，比较各分支的最终任务效用，构造事件级反事实 response bank。其次只运行目标模型的无干预 baseline trajectories，利用错误类型、执行进度、验证状态、剩余预算和局部语义建立 state representation，并以 response-aware partial optimal transport 将目标 states 对齐到来源 branch states。部分传输允许目标独有状态保留为 unmatched mass，避免为了覆盖率强行套用错误经验。最后以运输后的 action lower confidence bound 减去调用成本，编译出轻量 runtime router；有可靠正收益时执行相应 action，证据不足时选择 NONE。

理论上，我们将目标策略遗憾分解为来源响应估计误差、状态对齐代价、有限样本不确定性和未匹配质量，使每个算法组件都对应可检验的性能项，而不是用通用表示学习掩盖机制。

TRACE-H 的产物不是诊断报告，而是改变目标 agent 行为的可执行 harness policy。实验在 ALFWorld 和 WebShop 上留出整个 Qwen 与跨家族 executor，目标适配阶段严格封存所有干预结果，主指标是成功率、成本调整效用、负干预率和相对 oracle 的 policy regret。方法必须正面超过 No Harness、固定规则、source-only Offline-RL controller、Metric-Freedom gate、nearest-executor、kNN、balanced OT，并与 MASA、SkillAdaptor、SkVM-style adaptation 按目标反馈预算比较。若成立，TRACE-H 将证明 harness control 可以像策略一样跨基础模型运输，在昂贵目标试错前完成可执行、风险受控的适配。
