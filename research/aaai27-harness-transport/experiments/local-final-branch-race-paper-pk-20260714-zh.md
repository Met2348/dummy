# 最终本机论文级 PK 记录：TRACE-H Adaptive Branch-Race Ledger

日期：2026-07-14

## 一句话结论

本机阶段最有说服力的版本是 `TRACE-H adaptive branch-race ledger`：它不再是单一路径 `source_first`，也不是离线 oracle 式“跑完所有分支再挑最好”，而是一个可实现的 first-success 分支竞争机制。最终版本按 `reverse_container -> source_horizon_h32 -> source_exhaustive -> none_continuation -> h24 -> h80` 的顺序尝试分支，成功即停止；若全失败，则选择已评估分支中分数最好、步数更短的结果。

最终 dashboard 显示：Ours 在 6 个 evidence blocks 的 primary metrics 上均为唯一 rank 1，且对全部 paper baselines 的 paired sign tests 均显著为正。新增 compute-matched baseline 与成本审计后，Ours 在 ALFWorld 上同时满足 success rank 1、selected efficiency rank 1、branch-cost efficiency rank 1。

## 方法定义

最终 Ours 为 adaptive first-success race，而不是 oracle race：

1. `reverse_container_full`: 先测试一个强、短、但覆盖有限的 baseline 快路径。
2. `source_horizon_h32`: 再测试 TRACE-H source ledger 的 horizon-aware 分支。
3. `source_exhaustive`: 如果静态词表覆盖不足，执行 source-prioritized exhaustive sweep。
4. `none_continuation`: 若干预不必要，则允许回退到原 source policy。
5. `source_horizon_h24` / `source_horizon_h80`: 作为备用 horizon 分支。

这个顺序的意义是：先利用便宜快路径，再使用 TRACE-H ledger 扩大成功覆盖，最后保留 defer 以避免过度干预。它是设计类机制，不是诊断性分析。

## ALFWorld 139-prefix 扩展实验

候选从 68 个 dense no-progress prefixes 扩展到 139 个 threshold-2 no-progress prefixes。该扩展降低 stagnation detector 的 revisit threshold，用更早、更轻的 no-progress 状态测试机制是否稳健。

主要结果：

- Ours: `80/139`, success rate `0.5755`
- 最强独立单路径 baseline: `alphabetical_full=59/139`
- compute-matched baseline race: `63/139`
- 最强单分支 ablation: `source_exhaustive_full=77/139`
- Ours selected secondary: `success_per_10_steps=0.9732`
- Ours branch-cost audit: `branch_cost_success_per_100_steps=7.4961`
- Ours 平均选中分支步数: `14.37`
- Ours 平均实际评估分支总步数: `143.24`

关键文件：

- `local-dev/reports/L3-thr2-139-macro-pk-final-adaptive-reverse-h32-branch-race-ledger-summary-20260714.json`
- `local-dev/reports/L3-thr2-139-macro-traceh-adaptive-race-reverse-h32-exhaustive-20260714.json`
- `local-dev/reports/L3-thr2-139-macro-baseline-adaptive-race-with-defer-20260714.json`
- `local-dev/reports/L7-alfworld-compute-cost-audit-final-adaptive-reverse-h32-branch-race-20260714.json`

## 总 Dashboard

最终 dashboard：

- `local-dev/reports/L9-reviewer-requirement-audit-20260714.json`
- `local-dev/reports/L9-reviewer-requirement-audit-20260714.tsv`
- `local-dev/figures/L9-reviewer-requirement-audit-20260714.svg`
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json`
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-table-20260714.tsv`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-primary-20260714.svg`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-secondary-20260714.svg`

Dashboard 状态：

- local dataset count: 6
- real data source count: 3
- unique baseline count: 38
- unique ablation count: 6
- ALFWorld independent macro baseline count excluding NONE: 7
- Ours primary rank 1: true
- Ours unique primary rank 1: true
- Ours vs paper baselines significant: true
- WebShop static secondary significant: true
- WebShop interactive secondary significant: true
- HarnessBench endpoint secondary significant: true
- HarnessBench endpoint success rate: 1.0
- L9 local objective satisfied: true
- L9 paper submission final satisfied: false
- `paper_goal_satisfied=false`

`paper_goal_satisfied=false` 是刻意保留的审稿边界：HarnessBench 已升级到 endpoint oracle outcome 子集，但仍是离线 deterministic subset 和手写本地 handler，不是真正冻结后的 LLM-agent sealed final；sealed-style split 是 post-hoc audit，不是真正预注册 final。

## L9 Requirement Audit

新增 [L9 审稿式 requirement audit](local-reviewer-requirement-audit-20260714-zh.md)，将原始目标拆成 10 条本机硬条件和 1 条投稿边界：

- local paper-level PK objective: `satisfied=true`
- pass: 10
- warn: 1
- fail: 0
- 真实数据源：ALFWorld dense prefixes、WebShop-small interactive text、HarnessBench endpoint outcome subset
- 每个关键真实数据集均有两种 metrics、至少 6 个 baseline + Ours、Ours primary/secondary rank 1、primary paired tests 显著
- warn 项仅表示投稿最终 sealed LLM-agent result 未完成，不表示本机 PK 目标失败

## Split Audit

75% holdout 是当前最有用的本机 sealed-style audit：

- `local-dev/reports/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-20260714.json`
- `local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-primary-20260714.svg`
- `local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-secondary-20260714.svg`

结果：

- ALFWorld holdout: 84 / 139
- WebShop holdout: 750 / 1000
- HarnessBench endpoint holdout: 22 / 30
- holdout Ours primary rank 1: true
- holdout Ours secondary rank 1: true
- holdout Ours-vs-baselines primary significant: true
- holdout Ours-vs-baselines secondary significant: true

## 成本审计

成本审计专门回答“是不是只是多跑分支所以赢”的问题：

- `local-dev/reports/L7-alfworld-compute-cost-audit-final-adaptive-reverse-h32-branch-race-20260714.json`
- `local-dev/figures/L7-alfworld-selected-branch-efficiency-final-adaptive-reverse-h32-branch-race-20260714.svg`
- `local-dev/figures/L7-alfworld-branch-cost-efficiency-final-adaptive-reverse-h32-branch-race-20260714.svg`

结果：

- Ours success rank 1: true
- Ours selected efficiency rank 1: true
- Ours branch-cost efficiency rank 1: true
- Ours vs compute-matched baseline branch-race success significant: true

注意：branch-cost efficiency 是审计指标，不是主论文指标。它按实际评估过的分支总步数收费；adaptive race 可以 first-success early stop，因此比离线 oracle race 更适合写进正式方法。

## 当前证据边界

这套本机实验已经足够支持“adaptive branch-race ledger 是比单一路径 baseline 和 baseline-only branch-race 更强的 harness mechanism”这个 claim，但还不能直接当最终投稿实验：

- HarnessBench 已从 routing projection 升级到 endpoint oracle outcome subset，但还需要真实 LLM-agent sealed final。
- 需要预注册 split，并冻结方法后再跑 target final。
- ALFWorld 目前是 dense-prefix repair，不是完整 episode-level agent benchmark。
- 正式论文必须报告 branch budget，并保留 compute-matched baseline 与 cost audit。

## 下一步

1. 固化 `TRACE-H adaptive branch-race ledger` 作为本机主方法版本。
2. 在 HiPerGator 上跑预注册 target final：完整 ALFWorld episode / WebShop interactive / 第三个端到端 harness benchmark。
3. 增加真实端到端 baseline：例如 Reflexion/ReAct-style retry、best-of-N branch selection、tool-routing-only harness。
4. 把 `paper_goal_satisfied=false` 的三个原因逐项消掉，再写正式实验章节。

## 追加：HarnessBench endpoint outcome 升级

同日已新增并扩展 `L8-harnessbench-endpoint-pk-traceh-ledger-20260714`：在 30 个离线 HarnessBench endpoint tasks 上，6 个 baseline 与 `TRACE-H endpoint ledger` 均直接写 `out/` 产物，并由原 HarnessBench oracle 评分。Ours 的 `oracle_outcome_score=1.0`、`oracle_check_pass_rate=1.0`、`endpoint_success_rate=1.0`；相对 `demo_local` 与 `narrow_file_tool` 的 primary sign test 均为 `wins=26, losses=0, ties=4, p=2.98e-08`。新的 L5 dashboard 和 L6 75% fixed-split audit 已切到 endpoint 版 HarnessBench，endpoint holdout 为 22/30。该结果修复了“只有 routing projection”的硬伤，但仍应作为本机机制证据，而不是最终投稿 sealed agent 结果。
