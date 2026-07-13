# DR-0003E：从自然语言 REPLAN 转向结构化 action mechanism

- **日期：** 2026-07-13
- **负责人：** TRACE-H 项目组
- **状态：** accepted for local execution；不开放 target transport
- **复审时间：** structured branch runner 完成并跑完 Qwen3-8B failure-prefix probe 后

## 问题

Source Policy v2 已经让 Qwen3-8B 在三任务 gate 中达到 2/3 success，但旧 `NONE/REPLAN` 分支在 `pick_two_obj_and_place-Newspaper` 两个 failure prefix、三 seed 的 6 对比较中 terminal score 全为 0。是否继续扩大旧 REPLAN，还是重做 action 机制？

## 决策

选择 `PIVOT_TO_STRUCTURED_ACTION_MECHANISM`。

停止扩大旧自然语言 `REPLAN`。下一轮真实 Qwen/ALFWorld 分支只允许围绕结构化机制展开：

- `anti_loop_retry`
- `precondition_check`
- `subgoal_ledger`
- `bundle_conservative`
- 少量 `natural_replan` 仅作为旧机制对照

正式 target transport、Qwen3-14B development-target、Qwen3-32B/Gemma sealed target 继续暂停，直到结构化 action 在 source failure prefix 上产生可重复正 terminal advantage。

## 依据

1. 真实 Qwen3-8B Source Policy v2 gate：3 题中 2 题成功，parser failure 0，source scaffold 能力不再全退化。
2. 真实 Qwen3-8B `NONE/REPLAN` branch：2 个 Newspaper failure prefix x 3 seeds 全部 terminal score 0，旧 REPLAN 停止。
3. L4.2 synthetic mechanism PK：`bundle_conservative` mean utility 0.7200、accuracy 1.0000、negative intervention 0、private abstain 1.0000，排名第一。
4. L4.3 mechanism transport stress：private 25% 下 partial OT utility 0.7197，高于 best-fixed 0.6195；balanced OT private negative rate 为 1.0，而 partial OT private none rate 为 1.0。

机器证据见：

- `experiments/local-dev/reports/L3-source-policy-v2-gate-qwen3-8b.json`
- `experiments/local-dev/reports/L3-source-policy-v2-qwen3-8b-none-replan-branch.json`
- `experiments/local-dev/reports/L4-mechanism-design-synthetic-20260713.json`
- `experiments/local-dev/reports/L4-mechanism-transport-synthetic-20260713.json`
- `experiments/local-structured-mechanism-exploration-20260713-zh.md`

## 下一轮运行门

必须新建或扩展 branch runner，使每个 intervention 都记录：

- action mechanism id 与版本；
- 可见输入：task、observation、history、admissible commands；
- 机制输出：mask、structured guidance、selected support reason；
- 是否读取 oracle：必须为 false；
- continuation terminal score、success、parser failure、trajectory divergence。

真实 gate：

- failure-primary：Newspaper p003/p012 上至少两个 seed-0 prefix-action 组合 terminal advantage > 0；
- repeated sign stability >= 70%；
- non-regression：heat-egg p007、book-to-sofa p010/p015 不得由成功变失败；
- parser/infrastructure failure <= 2%；
- 只改变轨迹但 terminal score 仍全 0 时，判定该机制失败。

## Consequences

- 当前 TRACE-H 仍为 conditional-go，不进入 target PK。
- 方法 claim 从“自然语言 REPLAN 可迁移”改为“结构化可见状态 repair actions 可形成 response bank，并由 partial OT/LCB 保守迁移”。
- L4 synthetic 只能作为机制筛选和软件证据，不能写成主实验效果。
- B200 预算只在结构化 action source gate 通过后使用。
