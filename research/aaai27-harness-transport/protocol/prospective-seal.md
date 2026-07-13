# TRACE-H Prospective Seal 规范

> **历史协议：** 本文封存旧 patch-effect predictions。当前 executable policy transport 使用[Policy Transport Seal](policy-transport-seal.md)，额外封存 target action feedback ledger、router artifact 与同预算 baselines。

## 目标

防止任何 target patch outcome 影响 feature、trigger taxonomy、estimator、threshold、baseline、claim 或部署决策。

本项目允许在目标同一批任务上观察 baseline 轨迹，因此设置应准确称为：

> baseline-observed、intervention-outcome-sealed prospective transductive evaluation。

禁止称为 zero-interaction 或完全零样本 target transfer。

## Target Patch 运行前必须冻结

1. 精确 model identifier、provider endpoint、access date 与 decoding/reasoning parameters；
2. source、target 与 final-audit task manifests 及 SHA-256；
3. baseline harness commit 与三个 one-change patch commits；
4. intervention contracts 和 no-trigger unit-test outputs；
5. 36 个 target primary baseline、90 个 Metric Freedom diagnostic repeats 的完整 manifest 与 raw records；
6. target `MF-faithful`、`MF-all`、baseline-strength 与 failure-surface tables；
7. Metric Freedom output distance、mixed-question rule、degenerate rule、threshold、nearest metric 与 Source x MF clipping；
8. trigger taxonomy、stratum fallback 与最小样本阈值；
9. fitted conditional rescue/harm profile 与软件环境；
10. 108 个 target task-patch predictions；
11. 9 个 category-patch effect intervals，以及 choose none/P1/P2/P3/reject/abstain decisions；
12. MF-headroom、MF-nearest、Source x MF、baseline strength、failure-surface-nearest、opportunity-only 与 TRACE-H 全部预测文件；
13. scoring code、metrics、`delta_min`、cost weight、choice-regret 定义与 exclusion rules；
14. `experiments/predictions/SEAL.md` 中的 commit hash 和时间戳。

## Seal 前允许的 Target 信息

- 截止日期前公开的 model documentation；
- model identifier、family、context limit 与 tool-call format；
- 冻结 target tasks 上的 baseline-only outcomes 和 telemetry；
- 预声明 probe tasks 上带冻结 diversity prior 的 baseline-only Metric Freedom diagnostics；
- 不含 benchmark answer 的预声明 synthetic interface probes；
- 排除于报告的 infrastructure smoke tests。

## Seal 前禁止的信息

- 任一 target task 上启用 P1、P2、P3 或其变体；
- modified prompt、tool、memory、verification、retry 或 context policy 下的 target outcome；
- target-specific patch 编辑或 prompt tuning；
- 从 target patch trajectory 计算的任何 feature；
- target patch aggregate table 或 individual outcome；
- 由目标结果驱动的 threshold、stratum 或 baseline 修改；
- 根据 target baseline mixed/degenerate 状态替换主任务或 MF probe tasks；
- 让人类先看 target patch trajectory 再决定是否计入。

## Seal 流程

1. 验证所有 manifest、contract、prediction 与 code hash；
2. 独立脚本检查 target patch output 目录为空；
3. 运行 schema validator；
4. 生成 human-readable prediction summary；
5. 提交 immutable commit/tag；
6. 在 `SEAL.md` 记录 UTC 与 Asia/Shanghai 时间；
7. 由第二人或独立脚本确认后才启动 target patch runs。

## Unseal 流程

1. 记录 target-run start timestamp；
2. 按冻结顺序运行，不编辑 patch；
3. raw outcomes 只写 append-only 目录；
4. 基础设施失败先标记，按冻结规则重跑；
5. 用冻结 scorer 生成第一份 score report；
6. commit 第一份 report 后才能做 retrospective analysis；
7. 所有 protocol deviations 在看 aggregate result 前写入日志。

## 失效条件

以下任一情况使相关 target cell 不再 prospective：

- prediction commit 前观察过 target patch outcome；
- target result 影响 trigger、patch、threshold 或 exclusion；
- prediction file 在 unseal 后被覆盖；
- 无法证明 patch output directory 在 seal 时为空；
- target baseline 中意外启用了 intervention logic。

失效 cell 只能标为 retrospective/exploratory，不能与 prospective 主结果混报。
