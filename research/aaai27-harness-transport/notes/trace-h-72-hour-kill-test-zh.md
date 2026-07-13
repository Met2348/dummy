# TRACE-H 72 小时机制杀伤实验

> **历史协议：** 本文针对旧 patch-effect predictor。当前 design-method pilot 使用[Cross-Executor Policy Transport Kill Test](trace-h-72-hour-method-kill-test-zh.md)，主胜负是 end-to-end utility，而非 effect prediction。

- **目的：** 在 Metric Freedom 这一直接碰撞下，正面判别 generic headroom H-MF、SEAGym 启发的 H-FS、机会标准化 H-OPP 与条件响应 H-RESP 哪一层有继续投入 AAAI-27 的最小信号。
- **性质：** kill test，不是论文主实验，不用于事后挑任务、挑 patch 或宣布成功。
- **截止：** 启动后 72 小时，且最晚 2026-07-14 形成书面 go/pivot/stop 决策。

## 1. 冻结设计

| 维度 | 冻结选择 |
|---|---|
| Task suite | Harness-Bench 2.0 固定快照 |
| 任务 | 12 个，三个预声明类别各 4 个 |
| MF probes | 每类预声明 2 个，共 6 tasks；每题共 6 次 baseline diagnostic runs |
| 模型 | 两个 source + 一个预先指定的 sealed pseudo-target，优先覆盖三个不同家族 |
| 条件 | baseline、E-NORM、R-BOUND、F-GATE |
| Primary effect episodes | `3 x 12 x 4 = 144` |
| MF diagnostic repeats | 6 probe tasks 每题额外 5 次，共 `3 x 30 = 90` |
| 总 episodes | 约 `144 + 90 = 234` |
| 随机性 | 优先使用支持固定 seed/replay 的 endpoint；`temperature=0` 不视为充分条件 |
| 重复 | seedable 模型用同一 seed 配对；不可 seed 模型必须按冻结重复数估计分布效应 |
| 主要结果 | paired success delta |
| 次要结果 | cost、trigger、rescue、harm、contract violations |

任务必须在任何模型运行前按以下信息选择：平台可运行性、oracle 可执行性、类别覆盖、任务时长。禁止根据历史 leaderboard、模型难度、预估 trigger 或 patch 适配程度选择。

## 2. 启动前必须冻结的文件

- `experiments/manifests/pilot-tasks.json`
- `experiments/manifests/pilot-mf-probes.json`
- `experiments/manifests/pilot-models.json`
- `experiments/interventions/e-norm.json`
- `experiments/interventions/r-bound.json`
- `experiments/interventions/f-gate.json`
- `experiments/predictions/pilot-analysis-plan.md`
- 234 个 run slots、source/target 身份、primary/diagnostic 标记、执行阶段的随机化顺序与 hash
- episode schema、failure taxonomy 与 infrastructure failure 规则

每个 intervention contract 必须含：

- trigger predicate；
- first-trigger type；
- 可读取字段；
- intervention operator；
- 最大触发次数；
- 最大额外 step/token/time；
- no-trigger invariance test；
- 禁止访问的信息；
- known harm mechanism。

## 3. 实施顺序

### 0-12 小时：Reference Harness 与契约测试

1. 在两个排除于分析的 smoke tasks 上跑通 baseline；
2. 分别强制构造一次 P1-P3 trigger，确认 telemetry；
3. 对不触发 case 比较 request、tool state、workspace 和 resource counters；
4. 验证同一 episode 不能超过 contract 的触发上限；
5. 验证 patch 无法读取 oracle answer、最终 score 或其他 condition outcome；
6. 冻结 commit。

任何 patch 无法满足 dormant contract 时，先删除该 patch；不得通过放宽 atomicity 定义保留。若模型 endpoint 无法固定或重放随机性，必须在 pilot manifest 中标为 `distributional_only`，不得把两次独立输出的差异计为 no-trigger contract violation，也不得声称 task-level exact pairing。

### 12-36 小时：Target baseline、MF diagnostics 与 Source paired runs

- 先运行三个模型的 36 个 primary baseline；
- 对 6 个预声明 MF probe tasks，每个模型追加 5 个带预声明 diversity prior 的 diagnostic runs，共 90 个；这些 runs 不配对 patch，只用于忠实计算 `F_out`；
- 再只运行两个 source models 的 72 个 patch episodes；
- 按模型、任务和 condition block 随机化，避免时间漂移与 provider 波动集中在一个条件；
- 失败原样保存，不自动丢弃；
- API timeout、provider 5xx、sandbox 启动失败标为 infrastructure failure；
- agent 自己造成的超时、错误命令和预算耗尽均为有效 outcome；
- 重跑必须保留原失败记录并使用相同配置。

### 36-44 小时：密封伪目标预测

以模型 A、B 为 sources，模型 C 为预先指定的 pseudo-target：

1. 用 A、B 的 paired outcomes 估计每个 patch/trigger type 的 rescue、harm、unchanged；
2. 只用 C 的 baseline/diagnostic runs 估计 `MF-faithful`、`MF-all`、baseline strength 与 `pi_C`；若 mixed-rule 退化则标记 missing，禁止换 probe task；
3. 同时计算 zero-effect、source mean、baseline-strength、MF-headroom、MF-nearest、Source x MF、capability-nearest、failure-surface-nearest、opportunity-only 和 TRACE-H；
4. 写入带 hash 的预测与部署文件，记录代码 commit，随后禁止修改 estimator、taxonomy 和阈值。

其中 MF-headroom 严格按 Metric Freedom 的 repeated raw-run protocol 计算 `F_out`，由 A/B 内部验证冻结阈值；MF-nearest 用 `(F_out, baseline strength)` 选择来源；Source x MF 用预声明裁剪式缩放 source patch prior。Failure-surface-nearest 按 C 与 A/B 的 baseline first-trigger distribution 距离选择一个来源，复制该来源 aggregate patch effect。Opportunity-only 使用 C 的 opportunity distribution 与 A/B pooled response，但不区分非零 trigger type。这些都是必须认真击败的强基线，不能因为接近我们的叙事而被弱化。

### 44-56 小时：打开伪目标

只在 seal 完成后运行 C 的 36 个 patch episodes并评分。A、B 互换的 source LOO 可作为预冻结诊断，但不能冒充第二个 prospective target。

### 56-64 小时：机制与成本审计

- 列出每个 patch 的 opportunity rate；
- 计算 `P(rescue|trigger)`、`P(harm|trigger)`；
- 审计全部 rescue/harm 轨迹和最多 12 个 unchanged 轨迹；
- 核查 no-trigger task 是否真正一致；
- 报告 patch 新增 step、token、wall time；
- 判断 effect disagreement 是否来自预算、provider 或真实行为差异。

### 64-72 小时：书面决策

新建 `decisions/0004-trace-h-pilot-decision-zh.md`，逐条回答 gate。不能只发聊天结论。

## 4. 必须计算的表

### 表 A：激活与契约

| Model | Patch | Opportunity tasks | Activated runs | No-trigger mismatches | Contract violations |
|---|---|---:|---:|---:|---:|

### 表 B：Paired effect

| Model | Patch | Baseline success | Patch success | Delta | Rescue | Harm | Added cost |
|---|---|---:|---:|---:|---:|---:|---:|

### 表 C：密封伪目标预测

| Source set | Target | Patch | Source mean | MF-headroom | MF-nearest | Source x MF | Failure-surface-nearest | Opportunity-only | TRACE-H | Observed target delta | Better method |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

### 表 D：Trigger-conditioned response

| Patch | Trigger type | Source model | n | Rescue rate | Harm rate | Conditional delta |
|---|---|---|---:|---:|---:|---:|

### 表 E：Patch choice

| Target category | `F_out` | Best source-prior choice | Best MF-family choice | TRACE-H choice | Observed best action | Choice regret |
|---|---:|---|---|---|---|---:|

## 5. Go 条件

必须同时满足：

1. 至少两个 patch 在三个模型中的至少两个模型上各有不少于 4 个 opportunity tasks；
2. 对 seed/replay 配对的 runs，no-trigger condition 没有非基础设施 outcome mismatch；对 `distributional_only` 模型，重复量足以估计 baseline drift；
3. 至少一个 patch 在模型间出现以下任一信号：总体符号不同、effect 相差至少 `2/12`，或不少于 3 个 task-level discordant responses；
4. 至少一个 trigger type 的 opportunity rate 或 conditional response 能合理解释该差异；
5. `F_out` 能按冻结协议计算，且 MF diagnostic runs 与 primary effect runs 严格隔离；
6. sealed pseudo-target 上，TRACE-H 至少在一个 patch-specific 主要指标或 choice regret 上明确优于 `MF + source prior`、baseline strength 和 source-copy；若 TRACE-H 只与 failure-surface-nearest/opportunity-only 持平，则明确记录为“只支持 H-FS/H-OPP，不支持 H-RESP”；
7. success 结论在报告 added cost 后仍有实际意义；
8. 约 936 episodes 的完整设计预计可在截止前完成。

第 6 条在 12 任务小样本上只作为方向 gate，不宣称统计显著性。

## 6. Pivot 条件

- **Opportunity-only pivot：** `rho` 基本稳定，所有变化由 `pi` 解释；改写为 failure-opportunity standardization。
- **Metric Freedom pivot：** `F + source prior` 已解释 patch gain/choice；转为 Metric Freedom 独立复核或边界研究，停止当前 TRACE-H 主方法。
- **H-FS pivot：** failure-surface-nearest 与 TRACE-H 持平，但二者明显胜 capability/source-copy；改写为 failure-surface alignment 的 prospective sealed validation，不再声称 typed response 必要。
- **Single-patch pivot：** 只有一个 patch 有足够 trigger 与 heterogeneous response；主实验只保留该 family，增加任务数。
- **Verifier pivot：** 只有 F-GATE 有信号；聚焦 verifier opportunity、recovery 与 tax。
- **Active-probe pivot：** baseline failure surface 不足，但少量 target intervention probes 能快速校准；放弃纯 baseline-only claim。

Pivot 必须新建 decision record，并重写 claim register，不能在同一实验里静默换问题。

## 7. Stop 条件

- 三个 patch 都很少触发；
- no-trigger invariance 无法实现；
- 所有 model difference 都由 infrastructure 或不匹配预算造成；
- conditional response 比 raw source effect 更不稳定且无可解释 strata；
- source mean 已近乎完美，所有 target-aware 方法都没有预测或决策增益；
- Metric Freedom-informed family 与 TRACE-H 持平或更好；
- repeated baseline probes 无法稳定计算 `F_out`，或只能在看过 pseudo-target patch outcomes 后调节；
- failure-surface-nearest / opportunity-only 与 TRACE-H 持平，且 sealed protocol 本身也没有形成可独立陈述的新发现；
- reference harness 或三个 endpoints 在 36 小时内仍不稳定；
- 唯一结论只是重跑 Harness-Bench、SEAGym、LIFE-HARNESS、HASP、ToolBench-X 或 The Verifier Tax，既没有 prospective seal 增量也没有新 estimand 证据。

## 8. 禁止操作

- 根据 pilot 结果重新挑 36 个“更容易触发”的主任务；
- 根据 baseline mixed/degenerate 状态替换已冻结的 MF probe tasks；
- 查看 pseudo-target patch outcomes 后修改 trigger taxonomy；
- 把 prose prompt 改动与 executable patch 混为一个 atomic intervention；
- 删除 agent-caused timeout 或 budget failure；
- 用追加运行直到显著的方式改变样本量；
- 把 pilot 的单一伪目标称为对任意 unseen deployment model 的普适验证；
- pilot 未通过仍以 deadline 为理由进入完整矩阵。

## 9. 最终产物

- 234 个 append-only episode records；
- raw messages、tool traces、workspace diff 与 oracle outputs；
- 三份 intervention contracts；
- 一份 Metric Freedom fidelity report，含 probe manifest、distance、mixed/degenerate 与 `MF-faithful/MF-all`；
- 一份在 target patch run 前完成的 prediction seal；
- 五张规定表；
- 12-24 条人工 trajectory audit；
- 一份 DR-0004 go/pivot/stop 记录；
- 一条命令重建全部 pilot 表格。
