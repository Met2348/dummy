# DR-0003H：29 个 no-progress candidates 不足后的样本量升级方案

- **日期**：2026-07-14
- **状态**：accepted for experiment planning
- **触发问题**：导师指出当前 29 个本地 candidates 太少，不能支撑 AAAI 主结果。
- **关联证据**：
  - `experiments/local-five-way-pk-coverage-20260713-zh.md`
  - `experiments/local-dev/reports/L3-five-way-pk-coverage-aware-summary-20260713.json`
  - `deployment/local-vs-hipergator-execution-plan-zh.md`
  - `decisions/0003g-expanded-source-first-evidence-and-next-mechanism-zh.md`

## 决策

导师的质疑成立。当前 29 个样本只能称为 **development evidence**，用于发现机制、暴露边界、决定下一轮是否值得上集群；不能作为 AAAI 主表或最终 claim。

正式实验必须采用三层样本设计：

1. **Development branch pool**：本地/小规模，用于机制迭代，不写主结论；
2. **Source validation branch bank**：集群上扩大 no-progress branch points，用于验证 state-repair 机制与 baselines；
3. **Sealed target task-level final test**：主论文结果必须以完整 task success / utility 为单位，而不是只以 prefix rescue rate 为单位。

也就是说，论文可以保留 prefix-level branch analysis，但 headline 必须是：

> 在 sealed target tasks 上，TRACE-H 的 end-to-end utility 胜过最强 same-budget baseline。

## 为什么 29 不够

当前 29 个候选存在四个问题：

1. **单位太窄**：它们是 no-progress prefixes，不是完整任务样本。prefix rescue 有价值，但不能直接等价为 task success。
2. **来源太集中**：主要来自 Qwen3-4B/8B 的少量 gate 与 expansion tasks，source-first 成功集中在 newspaper、watch、tomato 等少数任务族。
3. **baseline 覆盖不齐**：NONE、target-first、source-first 覆盖 29/29；natural REPLAN 只覆盖 4/29，bundle 只覆盖 3/29。
4. **没有 sealed target**：当前是 development pool，不是冻结 policy 后第一次在 held-out executor 上执行。

因此，当前正确表述是：

> 29-case development pool 显示 source-first affordance macro 相对 NONE 与 target-first 呈现 8:0 独有成功优势，足以支持继续扩量；但不足以支撑最终 performance claim。

## 样本单位重新定义

论文中必须区分三种 N：

| 符号 | 单位 | 用途 | 当前状态 |
|---|---|---|---|
| `N_task` | 完整 ALFWorld/WebShop task episode | 主结果、success rate、utility | 仍未完成 |
| `N_prefix` | 从 baseline trajectory 抽出的 branch point | 机制分析、branch advantage、failure taxonomy | 当前 29 |
| `N_branch` | prefix × action × seed continuation | REPLAN/bundle/source repair 的配对比较 | 当前只覆盖部分 baseline |

主论文表 1 使用 `N_task`；机制表和 appendix 使用 `N_prefix/N_branch`。不能把 `15/29 prefix success` 写成主任务成功率。

## 最小论文级目标

### A. 本地 development 扩量

目标：把当前 29 个 no-progress candidates 扩到 **120-180 个**，但仍标注为 development。

设计：

- source executor：Qwen3-4B、Qwen3-8B；
- task strata：pick-and-place、two-object、clean、heat、cool、look/open navigation failure；
- 每个 task 最多保留 2 个 no-progress prefixes，避免单条失败轨迹贡献过多；
- cheap baselines 全覆盖：NONE、target-first、source-first、source-first ablations；
- expensive baselines 分层覆盖：优先覆盖 source-first 独有成功、剩余预算短/长各一半、不同任务族。

用途：

- 冻结 mechanism variants；
- 估计 effect size；
- 做 task-family failure taxonomy；
- 决定哪些 baseline 必须上 B200 全量。

禁止：

- 用本地 development N 写最终成功率；
- 看完结果后改 sealed target task list；
- 只挑 source-first 有利任务族。

### B. Source validation branch bank

目标：在 HiPerGator 上建立 **600-900 个 unique source branch points**，形成真正可训练/可验证的 source response bank。

建议结构：

| 维度 | 规划 |
|---|---|
| source models | Qwen3-4B、Qwen3-8B、Qwen3-14B |
| tasks per source model | 100-150 |
| prefixes per task | 最多 2 |
| branch actions | NONE、natural REPLAN、bundle/structured repair、target-first、source-first、主要 ablations |
| seeds | cheap symbolic methods 1 seed；LLM branch 至少 priority strata 3 seeds |

这层回答：

- source-first 是否只在 29 个样本上偶然有效；
- 机制收益是否跨 model、task family、prefix depth；
- natural REPLAN / bundle 是否真的弱，还是当前覆盖太少；
- 哪些 failure 是 irrecoverable，哪些可由 state repair 挽救。

### C. Sealed target final test

目标：主结果以 **task-level sealed target** 为单位，推荐最小规模：

| 设置 | 最小 | 更稳妥 |
|---|---:|---:|
| target executors | 2 | 2-3 |
| final-test tasks / target | 100 | 150-200 |
| seeds / task-method | 3 | 3 |
| same-budget methods | 8-9 | 8-12 |

最小主表约为：

```text
2 targets × 100 tasks × 3 seeds × 9 methods = 5,400 final-test runs
```

更稳妥主表约为：

```text
2 targets × 150 tasks × 3 seeds × 9 methods = 8,100 final-test runs
```

这与已有 B200 资源规划一致；完整资源包络仍在 14,000-20,000 total runs 量级，包括 source bank、ablation、oracle 与公开系统 baseline。

## 分层 baseline 覆盖策略

不是所有 baseline 都必须在 development 阶段全量，但 final-test 必须同预算公平。

| baseline | development | source validation | sealed final |
|---|---:|---:|---:|
| NONE | 全覆盖 | 全覆盖 | 全覆盖 |
| target-first macro | 全覆盖 | 全覆盖 | 全覆盖 |
| source-first state repair | 全覆盖 | 全覆盖 | 全覆盖 |
| source-first ablations | 全覆盖或大覆盖 | 大覆盖 | 关键 ablation |
| natural REPLAN | priority + stratified | 至少覆盖所有 source-first unique-win strata | 同预算全覆盖或预声明子集 |
| bundle / structured action | priority + stratified | 至少覆盖所有 source-first unique-win strata | 同预算全覆盖或预声明子集 |
| kNN / nearest / balanced OT / MF-gated | 离线可全覆盖 | 全覆盖 | 全覆盖 |
| MASA / SkillAdaptor | smoke + feasibility | 成本前沿样本 | 作为外部系统 baseline / upper comparator |

若 natural REPLAN 或 bundle 因成本无法在 final-test 全量，必须预声明为“representative expensive baseline subset”，不能在主表里与全量方法直接画等号。更好的方案是用 B200 批量补齐 final-test 同预算覆盖。

## 统计与审稿口径

主结果不以单个比例取胜为准，必须报告：

1. paired task-level success / utility；
2. bootstrap 95% confidence interval；
3. McNemar 或 paired permutation test；
4. 多 baseline Holm correction；
5. per-task-family breakdown；
6. negative intervention rate；
7. cost-adjusted utility；
8. failure taxonomy：不可行动、预算不足、multi-instance partial delivery、parser/environment failure。

粗略 power 直觉：

- 若相对 strongest baseline 的 absolute gain 约 10 pp，`100 tasks × 2 targets × 3 seeds` 通常足以给出清晰方向和置信区间；
- 若 gain 只有 5 pp，则需要 200+ tasks/target 或更多 seeds，否则容易被审稿人认为噪声；
- 如果收益只来自一个任务族，即使 N 很大也不够，必须按 task family 显示不是单点技巧。

因此，正式样本量不是只追求大，而是要保证：

```text
stratified tasks + paired methods + sealed targets + strong baselines + uncertainty
```

## 立即执行方案

### 24 小时内

1. 停止把 29 写成强 performance claim；所有文档改为 development evidence。
2. 从现有 ALFWorld task manifest 中生成 stratified expansion manifest，目标 60-80 tasks。
3. 本地先扩 cheap macro baselines 到 120-180 prefixes，更新 failure taxonomy。
4. 补齐 source-first unique-win strata 上的 REPLAN / bundle 覆盖。

### 48-72 小时内

1. 在 B200 上跑 30-episode throughput block，估计真实 run time；
2. 冻结 source validation manifest：3 source models × 100-150 tasks；
3. 生成 target baseline calibration 与 final-test disjoint task lists；
4. freeze 前只允许 target NONE calibration，不允许 target intervention leakage。

### 集群主实验

1. 先跑 source validation branch bank；
2. 编译 TRACE-H / baseline policies；
3. freeze artifacts 与 task order；
4. blind target final-test 一次性运行全部同预算方法；
5. 解封后只做预注册统计，不追加样本直到显著。

## 对老师的直接回应

可以这样回答：

> 老师说得对，29 个不是论文级样本量。我们现在把它定位为 development evidence，只用于发现机制和筛选 baseline。正式论文会改成三层设计：本地把 no-progress prefix pool 扩到 120-180 做机制迭代；B200 上建立 600-900 个 source branch points 做 source validation；主表用 sealed target task-level final test，最低 2 个 target executors × 100 tasks × 3 seeds × 约 9 个同预算方法，也就是约 5,400 条 final-test runs。这样主 claim 不再依赖 29 个 prefix，而是依赖 paired task-level utility 和 strongest-baseline PK。

## 复审条件

若出现以下任一情况，需要重新开决策：

1. source validation 扩到 120+ prefixes 后 source-first 独有优势消失；
2. natural REPLAN 或 bundle 在扩大覆盖后接近 source-first；
3. B200 throughput 低到无法完成 5,400 final-test runs；
4. sealed target final-test 中 strongest same-budget baseline 与 TRACE-H 持平或更好；
5. 效果只来自单一任务族，无法支撑 general harness claim。
