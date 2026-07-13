# DR-0003D：本机源策略全零后的停止与修复决策

- **日期：** 2026-07-13
- **负责人：** TRACE-H 项目组
- **状态：** accepted for execution；TRACE-H 方法主线仍为 conditional-go
- **复审日期：** Source Policy Gate v2 完成后，且不得晚于 2026-07-15
- **替代：** 不替代 DR-0003B；更新 DR-0003C 中“本机 L3 后直接进入 L5”的执行条件

## 问题

在 Qwen3-4B 与 Qwen3-8B 的 paired `NONE/REPLAN` 分支均未产生任何非零 terminal utility 后，是继续扩展 `CHECK/RETRY`、进入 transport 和 B200 扩量，还是停止当前 source runner 并修复 source policy？

## 决策

选择 `PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL`：立即停止当前 NF4 Qwen3-4B/8B source branch bank 的扩展，不补跑 `CHECK/RETRY`，不形成 response bank，不进入 Qwen3-14B development-target transport PK，也不在 B200 上原样放大。保留 TRACE-H policy transport 作为 conditional method hypothesis，因为本轮只测试了 source branch response，尚未测试 transport、partial OT、LCB 或 target utility。下一项唯一获准的 GPU 实验是 Source Policy Gate v2：在同一三任务、同一 parser/replay 协议下，改用带 subgoal memory 与 anti-loop recovery 的 benchmark-aligned executor scaffold；baseline 至少 1/3 成功后才允许重新开 branch，branch 至少两个 seed-0 正 advantage 且重复符号稳定率不低于 70% 后才恢复完整本机计划。

## 当前证据

### 项目直接观察

- 4B 与 8B 都完成 L0 NF4 probes，无 OOM；8B 4096-token peak allocated 为 6.66 GiB；
- 两个模型的 3-task、150-call baseline 均为 0/3 success，invalid action 与 parser failure 都为 0；
- 4B 的 6/6、8B 的 3/3 prefixes 精确恢复，deterministic NONE 整个 action suffix 与 terminal score 也全部一致；
- 4B 有 10 对、8B 有 7 对 NONE/REPLAN；17 对 raw terminal scores 全为 0；
- 4B 的 REPLAN 改变 10/10 action suffix，8B 改变 3/7，因此 4B 失败不是 guidance 完全未进入 policy；
- 所有 plans 非空，NONE/REPLAN extra-call 合同为 0/1，branch parser failure 为 0；
- 联合审计结论为 `PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL`，且 `transport_tested=false`。

完整实验解释见[本机源策略分支最终报告](../experiments/local-none-replan-source-pilot-final-20260713-zh.md)，机器证据见[跨模型联合审计](../experiments/local-dev/reports/L3-multimodel-source-pilot-audit.json)。

### 文献支持

现有 proposal 与全文证据边界支持“source same-prefix responses 可作为 transport 输入”这一研究问题，但没有文献能够替代本项目对 source response 非退化性的直接验证。本决策不新增文献 claim。

### 反证

- “模型再大一点自然会解决”没有在 4B -> 8B 上成立；
- “REPLAN 只是没有影响动作”与 4B 的 10/10 trajectory divergence 冲突；
- “runner/replay 坏了导致全零”与 9/9 prefix 和 NONE suffix integrity 冲突。

### 尚未知

- 更强 executor scaffold 能否在不增加 oracle 权限的情况下获得非零 source utility；
- 当前任务的 sparse terminal reward 是否足以形成稳定 response bank；
- BF16 是否会实质改变 NF4 下的 source-policy 结论；
- TRACE-H transport 在合法非退化 response bank 上是否优于 kNN、balanced OT 和 Source-AW。

## 评分

| 维度 | 分数 | 置信度 | 理由 |
|---|---:|---|---|
| Neatness | 8.0/10 | 中 | 研究问题与 source/target 隔离仍清楚，但 source executor 前置条件暴露 |
| Excitement | 7.5/10 | 中 | 成功时仍是 design method；当前没有 utility signal |
| Evidence strength | 4.5/10 | 高 | 工程与负结果证据强，方法有效性证据仍为 0 |
| Novelty | 7.5/10 | 中 | 本轮未出现新碰撞，也未增加 novelty 证据 |
| Soundness | 6.5/10 | 中高 | paired branches、repeats、完整 replay audit 提高了诊断可信度 |
| Feasibility | 4.5/10 | 中高 | 当前 scaffold 不可扩量，截止期压力显著上升 |
| Community fit | 7.0/10 | 中 | 若能给出 end-to-end utility 仍契合 agent/harness 社区，否则不足主会 |

## 备选方案

### 继续补跑 CHECK/RETRY

未选择。source terminal response 已在两个模型、17 对分支上全部退化；新增 action 只会增加成本，不能修复 baseline 规划与记忆。

### 直接把 14B 改成 source

未选择。14B 在本机协议中已经预声明为 `DEV_TARGET`；临时换角色会破坏 whole-executor holdout。正式 B200 协议可把 14B 作为 source，但不能与本机 target 结果混用。

### 直接上 B200 BF16 重跑

暂不选择。NF4 可能影响行为，但 8B 的动作退化与 guidance adoption 问题足以要求先做三任务 scaffold gate。B200 只能在 gate 通过后放大科学样本，不能用于放大已知退化 runner。

## 风险与反证

以下任一结果会要求重新评估本决策：

- 在完全相同 scaffold 下，B200 BF16 的预注册三任务 probe 显示稳定非零 utility，证明 NF4 是主因；
- benchmark-aligned scaffold 仍连续两版无法得到 1/3 baseline success，说明任务/模型组合不适合作为当前 response-bank substrate；
- source baseline 成功但 action advantages 仍全零，说明需要改 event support 或 reward signal，而不是继续加 agent memory；
- Source Policy Gate v2 通过，但 transport synthetic 或 target leak gate 失败，届时仍不得进入主实验。

## Go / Pivot / Stop 门

### Go

- 同一三任务 baseline 至少 1/3 success；
- parser failure <=2%，prefix 与 deterministic NONE continuation integrity 为 100%；
- 至少两个 seed-0 REPLAN advantages >0；
- 两个 repeated candidates 的 advantage-sign stability >=70%；
- 无 oracle、gold action、target outcome 或 silent retry。

### Pivot

- baseline 有成功但 branch 全零：转向更密集的合法 progress signal/event support，重新冻结 contract；
- NF4 与 BF16 三任务结果相反：把 source gate 移到 B200 BF16，并保留精度消融；
- 8B scaffold 仍失败但有明确可复现的 benchmark agent baseline：采用该 scaffold，保持相同 action budget 做 PK。

### Stop

- 到 2026-07-15 仍未通过 baseline 1/3 门；
- 通过 baseline 后仍无法得到两个可重复正 action effects；
- 为通过门需要 oracle、gold trajectory、target outcome 或改变任务集合；
- 只改善轨迹多样性或 plan adoption，不改善 terminal utility。

## 后果

- 新增本机源策略分支最终报告与跨模型机器审计；
- 当前 L3 branch bank 标记为 diagnostic failure，不作为 transport 输入；
- L4 synthetic tests 可继续做 CPU 工程验证，但不能被写成方法收益；
- L5、B200 72-hour kill test 和完整 PK 暂停，等待 Source Policy Gate v2；
- Qwen3-14B 继续保持本机 `DEV_TARGET`，Qwen3-32B/Gemma 仍保持 sealed；
- AAAI-27 readiness 下调，若 stop 门触发则本轮更换主投方向。

## 变更记录

- 2026-07-13：根据 4B/8B 共 17 对 NONE/REPLAN 全零结果建立决策；历史单模型报告不改写，由跨模型审计覆盖其 stale 8B 标签。
- 2026-07-13：Source Policy v2 在8B上达到2/3 success、在4B上仍为0/3；8B competence gate 因而通过。但困难任务两个prefix、三seed的6对 NONE/REPLAN 仍全部为0，虽然每对轨迹改变32-47步。最终执行决定更新为：保留v2作为合格8B source scaffold，停止当前自然语言 REPLAN，不进入target transport；下一轮必须重新设计结构化 action mechanism。详见[证据强度更新](../experiments/local-source-policy-v2-evidence-assessment-20260713-zh.md)。
