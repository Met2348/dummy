# L9 审稿式 Requirement Audit

日期：2026-07-14

## 结论

本地 PK 目标的硬条件已满足：`10` 条 pass，`1` 条 warn，`0` 条 fail。warn 不是本机 PK 失败，而是投稿最终版仍需要 frozen method 后的 sealed LLM-agent final。

- 可视化矩阵：`research/aaai27-harness-transport/experiments/local-dev/figures/L9-reviewer-requirement-audit-20260714.svg`
- L5 dashboard `paper_goal_satisfied`: `false`
- HarnessBench endpoint tasks: `30`
- HarnessBench endpoint success rate: `1.0`
- L6 holdout rank/significance: primary `True`, secondary `True`

## Checklist

| status | requirement | evidence | detail |
| --- | --- | --- | --- |
| `pass` | 至少 3 个真实数据源 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | real_data_source_count=3; real_rows=3 |
| `pass` | 每个关键真实数据集至少 6 个 baseline + Ours | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | ALFWorld dense prefixes=8; HarnessBench endpoint outcome subset=6; WebShop-small interactive text=9 |
| `pass` | 每个关键真实数据集都有两种 metrics | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | ALFWorld dense prefixes:success_rate/success_per_10_steps; HarnessBench endpoint outcome subset:oracle_outcome_score/oracle_check_pass_rate; WebShop-small interactive text:mean_webshop_reward/exact_purchase_rate |
| `pass` | Ours 在所有 primary metrics 上 rank 1 且唯一 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | all_ours_primary_rank_1=True; unique=True |
| `pass` | Ours 相对全部 paper baselines 统计显著 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | significant=True |
| `pass` | 真实数据集 secondary metrics 也 rank 1 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | ALFWorld dense prefixes:secondary_rank=1; HarnessBench endpoint outcome subset:secondary_rank=1; WebShop-small interactive text:secondary_rank=1 |
| `pass` | 真实数据集 primary paired tests 均显著 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | ALFWorld dense prefixes=True; HarnessBench endpoint outcome subset=True; WebShop-small interactive text=True |
| `pass` | HarnessBench endpoint 子集达到 30 tasks 且 Ours 全成功 | `research/aaai27-harness-transport/experiments/local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-20260714.json` | task_count=30; success=1.0 |
| `pass` | 可视化产物存在 | `research/aaai27-harness-transport/experiments/local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-primary-20260714.svg, research/aaai27-harness-transport/experiments/local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-secondary-20260714.svg, research/aaai27-harness-transport/experiments/local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-primary-20260714.svg, research/aaai27-harness-transport/experiments/local-dev/figures/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-secondary-20260714.svg` | L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-primary-20260714.svg:True; L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-secondary-20260714.svg:True; L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-primary-20260714.svg:True; L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-secondary-20260714.svg:True |
| `pass` | 75% fixed-split audit 仍 rank 1 且显著 | `research/aaai27-harness-transport/experiments/local-dev/reports/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-20260714.json` | unit_counts={'ALFWorld dense prefixes': {'all': 139, 'holdout': 84, 'development': 55}, 'WebShop-small interactive text': {'all': 1000, 'holdout': 750, 'development': 250}, 'HarnessBench endpoint outcome subset': {'all': 30, 'holdout': 22, 'development': 8}} |
| `warn` | 投稿最终 sealed LLM-agent result 仍未完成 | `research/aaai27-harness-transport/experiments/local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json` | The dashboard now includes ALFWorld dense-prefix replay, WebShop-small interactive text, and HarnessBench endpoint outcome subset.; HarnessBench endpoint evidence is an offline deterministic subset with hand-written local handlers, not yet a sealed LLM-agent final.; Sealed target task-level results and visualizations across three real interactive datasets are still missing. |

## Dataset Audit

| dataset | real | units | baselines | metrics | ranks | significant |
| --- | --- | ---: | ---: | --- | --- | --- |
| ALFWorld dense prefixes | true | 139 | 8 | `success_rate` / `success_per_10_steps` | primary 1, secondary 1 | true |
| HarnessBench endpoint outcome subset | true | 30 | 6 | `oracle_outcome_score` / `oracle_check_pass_rate` | primary 1, secondary 1 | true |
| Synthetic mechanism design | false | 1000 | 4 | `mean_utility` / `accuracy` | primary 1, secondary 1 | true |
| Synthetic transport stress | false | 1000 | 4 | `mean_utility` / `safety_rate` | primary 1, secondary 1 | true |
| WebShop-small interactive text | true | 1000 | 9 | `mean_webshop_reward` / `exact_purchase_rate` | primary 1, secondary 1 | true |
| WebShop-small static product selection | false | 415 | 8 | `exact_target_rate` / `target_mrr` | primary 1, secondary 1 | true |

## Remaining Boundary

当前本机证据已经满足用户提出的 local paper-level PK 目标；但它仍不能被写成最终投稿 sealed result。主要原因是 HarnessBench endpoint 使用 deterministic local handlers，且 L6 split 是 post-hoc sealed-style audit。下一步需要在 HiPerGator 上冻结方法后跑预注册 target final。
