# TRACE-H 实验工作区

## 最新结论

### 2026-07-14 最终本机 paper-PK dashboard

推荐先看：

- [L9 审稿式 requirement audit](local-reviewer-requirement-audit-20260714-zh.md)
- [最终本机论文级 PK 记录](local-final-branch-race-paper-pk-20260714-zh.md)
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json`
- `local-dev/reports/L9-reviewer-requirement-audit-20260714.json`
- `local-dev/figures/L9-reviewer-requirement-audit-20260714.svg`
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-table-20260714.tsv`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-primary-20260714.svg`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-secondary-20260714.svg`

当前本机证据：

- evidence blocks: 6
- real data sources: 3, 包括 ALFWorld dense-prefix replay、WebShop-small interactive text、HarnessBench endpoint outcome subset
- unique baselines: 38
- unique ablations: 6
- ALFWorld 独立 macro baselines: 7
- Ours 在所有 primary metrics 上均为唯一 rank 1
- Ours 相对全部 paper baselines 的 paired sign tests 均显著为正
- WebShop static、WebShop interactive、HarnessBench endpoint 的 secondary metrics 均显著
- L9 requirement audit: local objective `satisfied=true`; submission final `satisfied=false`
- `paper_goal_satisfied=false` 仍然保留：HarnessBench 已升级到 endpoint oracle outcome subset，但仍是离线 deterministic subset 和手写本地 handler，不是真正冻结后的 LLM-agent sealed final；split audit 是 post-hoc sealed-style audit，不是真正预注册 sealed final

### 2026-07-14 ALFWorld 139-prefix adaptive branch-race

最终 ALFWorld 本机方法为 `TRACE-H adaptive branch-race ledger`：按 `reverse_container -> source_horizon_h32 -> source_exhaustive -> none_continuation -> h24 -> h80` 顺序 first-success early stop。它不是离线 oracle race，而是可实现的 adaptive race。

核心结果：

- 候选规模从 68 个 dense no-progress prefixes 扩展到 139 个 threshold-2 no-progress prefixes
- Ours: `80/139`, success rate `0.5755`
- 最强独立 baseline: `alphabetical_full=59/139`
- 最强单分支 ablation: `source_exhaustive_full=77/139`
- Ours selected secondary `success_per_10_steps=0.9732`, 也是 rank 1
- Ours branch-cost audit `branch_cost_success_per_100_steps=7.4961`, 也是 rank 1
- 75% holdout: primary 与 secondary 均 rank 1，且 Ours-vs-baselines 均显著
- 50% holdout: primary 因 ALFWorld holdout 仅 48 个 prefix，仍存在统计功效不足；secondary 已显著

主要产物：

- `local-dev/reports/L3-thr2-139-macro-pk-final-adaptive-reverse-h32-branch-race-ledger-summary-20260714.json`
- `local-dev/reports/L3-thr2-139-macro-traceh-adaptive-race-reverse-h32-exhaustive-20260714.json`
- `local-dev/reports/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-holdout75-20260714.json`
- `local-dev/reports/L7-alfworld-compute-cost-audit-final-adaptive-reverse-h32-branch-race-20260714.json`
- `local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-holdout75-primary-20260714.svg`
- `local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-holdout75-secondary-20260714.svg`

### 2026-07-14 WebShop-small interactive PK

[WebShop-small 交互式 PK 实验记录](local-webshop-interactive-pk-20260714-zh.md) 记录了本机 WebShop text environment 修复、`indexes_1k` Lucene 索引构建，以及 1000 个 WebShop synthetic goals 上 9 个 baseline 和 Ours 的真实交互 PK。

核心结果：

- Ours: `mean_webshop_reward=0.9510`, `exact_purchase_rate=0.932`
- 最强 reward baseline: `all_text_overlap_full_search=0.8127`
- 最强 exact baseline: `attribute_overlap_attr_search=0.610` / `rarest_attribute_anchor=0.610`
- Ours 对全部 9 个 baselines 在 reward 和 exact 两个指标上均显著

主要产物：

- `local-dev/reports/L4-webshop-interactive-pk-goals1000-20260714.json`

### 2026-07-14 HarnessBench routing projection PK

[HarnessBench routing projection PK 实验记录](local-harnessbench-routing-pk-20260714-zh.md) 记录了 106 个真实 HarnessBench workspace tasks 上的 capability routing 预测实验。该实验不是端到端 agent outcome，而是 routing projection。

核心结果：

- Ours: `capability_jaccard=0.999`, `exact_capability_set_rate=0.991`
- 最强 baseline: `all_text_flat`, `capability_jaccard=0.964`, `exact=0.755`
- Ours 相对最强 baseline 的 exact set delta 为 `+0.236`, paired sign test `p=4.17e-07`

主要产物：

- `local-dev/reports/L4-harnessbench-routing-pk-20260714.json`

### 2026-07-14 HarnessBench endpoint outcome PK

[HarnessBench endpoint outcome PK 本机实验记录](local-harnessbench-endpoint-pk-20260714-zh.md) 记录了 30 个离线 HarnessBench endpoint tasks 上的 oracle outcome PK。该实验直接创建 workspace、生成 `out/` 产物并调用原 HarnessBench oracle；Ours 在 outcome、endpoint success 与 check-pass 三个指标上均为 `1.0`，相对 6 个 baseline 均显著。最强 baseline 为 `narrow_file_tool`，mean outcome `0.3893`；Ours 相对它的 primary sign test 为 `wins=26, losses=0, ties=4, p=2.98e-08`。

主要产物：

- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-20260714.json`
- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-matrix-20260714.tsv`
- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-20260714.html`

## 历史记录

- [Local paper PK dashboard v0](local-paper-pk-dashboard-20260714-zh.md)
- [ALFWorld horizon / transform recovery PK](local-paper-pk-horizon-transform-recovery-20260714-zh.md)
- [Dense macro PK](local-dense-macro-pk-20260714-zh.md)
- [Expanded macro evidence](local-expanded-macro-evidence-20260713-zh.md)
- [Five-way PK coverage](local-five-way-pk-coverage-20260713-zh.md)
- [Qwen3-4B/8B NONE vs REPLAN final](local-none-replan-source-pilot-final-20260713-zh.md)
- [Source Policy v2 evidence assessment](local-source-policy-v2-evidence-assessment-20260713-zh.md)
- [Local development experiment plan](local-development-experiment-plan-zh.md)

## 目录约定

- `scripts/`: 可复现实验、汇总和可视化脚本
- `src/`: TRACE-H 本地实验核心代码
- `tests/`: 本地单元测试
- `local-dev/`: 本机 development reports、figures 与临时 handoff bundle
- `schemas/`: contract、episode、branch、router、ledger schemas

大模型权重、ALFWorld 数据和大型 raw traces 位于外部运行目录；本仓库只保存可审计 manifest、hash、小型报告和图表。

## 数据隔离

- `DEV_SOURCE_BRANCH`、`DEV_TARGET_BASELINE`、`DEV_TARGET_FINAL` 仅用于本机工程协议
- 正式实验应使用 `SOURCE_BRANCH`、`SOURCE_VALIDATION`、`TARGET_BASELINE_CALIBRATION`、`TARGET_FINAL_TEST`
- target baseline 与 target final 必须使用不同 output roots
- target final raw records 在全部方法冻结前不能进入调参或 artifact 生成流程
- infrastructure failure、parser failure 和预算超限都必须保留，不能静默重跑
