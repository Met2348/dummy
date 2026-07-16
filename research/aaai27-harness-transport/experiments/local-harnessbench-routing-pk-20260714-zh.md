# HarnessBench routing projection PK 实验记录（2026-07-14）

## 一句话结论

本机新增第三个真实任务来源：`harness-bench` 的 106 个真实 workspace benchmark 任务。当前实验不是端到端 agent outcome，而是一个**harness/tool capability routing projection**：给定任务 title、prompt、tags、fixtures 和 oracle 信号，预测该任务需要哪些 harness 能力。`TRACE-H multisource routing ledger` 相比 8 个单源或平铺 baseline，在 `capability_jaccard` 与 `exact_capability_set_rate` 两个指标上均 rank 1 且显著。

## 实验设置

脚本：

- `scripts/run_harnessbench_routing_pk.py`

报告：

- `local-dev/reports/L4-harnessbench-routing-pk-20260714.json`

任务数：

- 106 个 HarnessBench tasks。

预测标签：

- `file_io`
- `shell_exec`
- `browser`
- `memory`
- `vision`
- `git`
- `office`
- `code`
- `data_analysis`
- `security`
- `planning`
- `writing`

## Baseline

- `majority_prior`
- `title_only`
- `prompt_only`
- `tags_only`
- `fixtures_only`
- `oracle_only`
- `title_prompt`
- `all_text_flat`

Ours：

- `traceh_multisource_ledger`

## 主要结果

| method | capability_jaccard | exact_set |
|---|---:|---:|
| TRACE-H multisource routing ledger | 0.999 | 0.991 |
| flat all-text keywords | 0.964 | 0.755 |
| oracle-only keywords | 0.852 | 0.340 |
| title+prompt keywords | 0.793 | 0.255 |
| prompt-only keywords | 0.782 | 0.245 |
| fixture-only keywords | 0.444 | 0.019 |
| tags-only keywords | 0.338 | 0.019 |
| title-only keywords | 0.285 | 0.019 |
| majority prior | 0.231 | 0.000 |

相对最强 baseline `all_text_flat`：

- Jaccard delta: `+0.035`
- paired wins/losses/ties: `26 / 1 / 79`
- p-value: `4.17e-07`
- exact set delta: `+0.236`
- exact wins/losses/ties: `26 / 1 / 79`
- p-value: `4.17e-07`

## 解释

这块实验支持的不是“agent 已经能完成 HarnessBench 任务”，而是一个更窄但有用的机制点：harness 运行前需要先判断任务需要哪些能力模块。单源 baseline 容易漏掉跨文件、跨工具线索；`all_text_flat` 虽然很强，但把所有线索平铺后仍会在部分任务上过召回或漏召回。`TRACE-H multisource routing ledger` 把 title、prompt、tags、fixtures、oracle import/检查逻辑分层汇总，并对代码、数据、浏览器、视觉、记忆等信号做结构化合并，所以 exact set 明显更高。

## 证据边界

这是第三个真实任务来源，但仍是 projection，不是完整交互 benchmark。它可以作为论文补充实验或方法动机证据；主结果仍应优先放 ALFWorld 与 WebShop interactive。若要把 HarnessBench 变成主文级证据，需要后续接一个真实 agent adapter，跑 outcome oracle，而不是只跑 routing。
