# HarnessBench endpoint outcome PK 本机实验记录

日期：2026-07-14

## 一句话结论

本轮把 HarnessBench 证据从 `routing projection` 往前推进到真实 endpoint oracle outcome：在 30 个可离线运行的 HarnessBench endpoint tasks 上，脚本创建隔离 workspace，复制 fixtures，让 6 个 baseline 和 `TRACE-H endpoint ledger` 直接写 `out/` 产物，再调用原 HarnessBench oracle 评分。Ours 在 `oracle_outcome_score`、`endpoint_success_rate`、`oracle_check_pass_rate` 三个指标上均为 `1.0`，相对全部 baseline 的 paired sign test 均显著。

## 任务与 baseline

任务子集覆盖文件与 shell、会议摘要、邮件分类、图像 answer 文件、办公文档、archive checksum、batch transform、calendar scheduling、PPT brief、合同风险、邮件线程归并、CRM follow-up、offline QA、messy sales cleaning、多表 revenue reconciliation、SQLite BI report、metric definition audit、交易异常检测、预算方差、漏斗掉点分析、库存预测、A/B test caveat、时序异常归因、财务 close reconciliation、schema drift、JSONL sessionization、metric migration diff、policy version conflict、insufficient-evidence QA，共 30 个离线 endpoint tasks。

Baseline 共 6 个：
- `no_output`
- `schema_only`
- `prompt_literal_stub`
- `fixture_copy`
- `demo_local`
- `narrow_file_tool`

Ours 为 `traceh_endpoint_ledger`，它是本机 endpoint handler 组合，用于验证真实 oracle outcome 链路是否能闭环；它还不是最终论文里的 sealed LLM-agent 方法。

## 结果

- Ours mean outcome: `1.0000`
- Ours endpoint success rate: `1.0000`
- Ours mean check pass rate: `1.0000`
- 最强 baseline: `narrow_file_tool`, mean outcome `0.3893`, endpoint success rate `0.1333`
- `demo_local`: mean outcome `0.3774`, endpoint success rate `0.1333`
- Ours vs `demo_local`: primary sign test `wins=26, losses=0, ties=4, p=2.98e-08`
- Ours vs `narrow_file_tool`: primary sign test `wins=26, losses=0, ties=4, p=2.98e-08`
- Ours vs 其他 baseline: primary / secondary 均显著

## 关键文件

- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-20260714.json`
- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-matrix-20260714.tsv`
- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-aggregate-20260714.tsv`
- `local-dev/reports/L8-harnessbench-endpoint-pk-traceh-ledger-20260714.html`

新的总 dashboard 使用 endpoint 版 HarnessBench：
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-20260714.json`
- `local-dev/reports/L5-local-paper-pk-dashboard-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-table-20260714.tsv`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-primary-20260714.svg`
- `local-dev/figures/L5-local-paper-pk-final-adaptive-reverse-h32-branch-race-webshop-harnessbench-endpoint-secondary-20260714.svg`

75% fixed-split audit 也已切到 endpoint：
- `local-dev/reports/L6-sealed-style-split-audit-final-adaptive-reverse-h32-branch-race-harnessbench-endpoint-holdout75-20260714.json`

## 证据边界

这一步显著强于 routing projection，因为它调用了真实 HarnessBench oracle，而不是只预测 capability label。但它仍然不是最终投稿级 sealed agent 结果：任务子集是离线 deterministic subset，Ours 使用手写本地 handlers，且 split 是 post-hoc fixed split。下一步应在 HiPerGator 上冻结方法后，用真实 LLM-agent baselines 跑预注册 target final。
