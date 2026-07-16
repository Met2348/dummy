# TRACE-H 本地 dense prefix 多基线 PK 报告

- **日期**：2026-07-14
- **定位**：development evidence；不是 sealed target final result
- **目的**：回应“29 个太少”的问题，先在本机把已有真实 source-policy traces 的 no-progress prefix pool 放宽到 38 个；随后新增 Qwen3-4B expansion offsets 3-8 的真实 rollout，把 pool 扩到 58 个；本轮再新增 Qwen3-4B expansion offsets 9-14，把 pool 扩到 68 个，并把不调模型的强 cheap baselines 与 source-first 消融跑满。

## 一句话结论

在 68 个真实 no-progress prefixes 上，`source_first_full` 达到 **30/68**，而 `NONE` 是 **13/68**、`target_first_full` 是 **14/68**。paired discordance 为：

| 对比 | source-first 独有成功 | 对方独有成功 |
|---|---:|---:|
| source-first vs NONE | 18 | 1 |
| source-first vs target-first | 17 | 1 |

这比 29-case 报告更强，也更真实：dense extraction 与新增 4B expansion traces 后，source-first 仍明显领先，但不再是完美 0-loss。这个 18:1 / 17:1 的形状比小样本 8:0 更适合写成 development evidence，因为它同时支持优势和边界。

额外生成的 discordance audit queue 显示，`source_first_full` 同时击败 `NONE` 与 `target_first_full` 的核心样本为 **10** 个；`source_first_full` 成功但 `no_instance_ledger` 失败的样本为 **14** 个。这两类样本分别用于论文中的 qualitative mechanism examples 与 instance-ledger 必要性分析。

新增 offsets 9-14 的 10 个 dense prefixes 是边界压力样本：该 block 中 `NONE=1/10`、`target_first_full=1/10`、`source_first_full=1/10`，没有新增 source-only win，也没有新增反向 loss。因此它降低了总体比例，但保留了 paired advantage 的方向。

## 机器报告

- `local-dev/reports/L3-source-policy-v2-dense-merged-candidates-20260714.json`
- `local-dev/reports/L3-dense-macro-pk-summary-20260714.json`
- `local-dev/reports/L3-dense-macro-pk-matrix-20260714.tsv`
- `local-dev/reports/L3-source-policy-v2-dense-merged-candidates-58-20260714.json`
- `local-dev/reports/L3-dense58-macro-pk-summary-20260714.json`
- `local-dev/reports/L3-dense58-macro-pk-matrix-20260714.tsv`
- `local-dev/reports/L3-dense58-macro-discordance-audit-queue-20260714.json`
- `local-dev/reports/L3-dense58-macro-discordance-audit-queue-20260714.md`
- `local-dev/reports/L3-source-policy-v2-qwen3-4b-expansion-o009-o014-20260714.json`
- `local-dev/reports/L3-source-policy-v2-qwen3-4b-expansion-o009-o014-no-progress-candidates-dense-20260714.json`
- `local-dev/reports/L3-source-policy-v2-dense-merged-candidates-68-20260714.json`
- `local-dev/reports/L3-dense68-macro-pk-summary-20260714.json`
- `local-dev/reports/L3-dense68-macro-pk-matrix-20260714.tsv`
- `local-dev/reports/L3-dense68-macro-discordance-audit-queue-20260714.json`
- `local-dev/reports/L3-dense68-macro-discordance-audit-queue-20260714.md`

对应脚本：

- `scripts/merge_candidate_reports.py`
- `scripts/run_symbolic_search_macro_probe.py`
- `scripts/build_dense_macro_pk_summary.py`
- `scripts/build_dense_discordance_queue.py`

## Candidate pool

第一阶段没有新增模型 rollout，而是先把已有 source-policy traces 的 extraction 放宽为 `max_per_run=30, min_step_gap=1`：

| block | source runs | dense prefixes |
|---|---:|---:|
| Qwen3-8B gate | 3 | 6 |
| Qwen3-4B gate | 3 | 11 |
| Qwen3-8B expansion | 6 | 21 |
| subtotal | 12 | 38 |
| Qwen3-4B expansion 新增 | 6 | 20 |
| Qwen3-4B expansion offsets 9-14 新增 | 6 | 10 |
| **total** | **24** | **68** |

因此它仍是 development pool，但比原 29 更接近机制压力测试：包含更密集的同轨迹相邻失败点、更多 late-prefix budget pressure、4B/8B expansion 的交叉失败面，以及更多 partial-delivery/visible-but-unactionable 失败。

## 多基线结果

| method | success | mean macro steps | 解释 |
|---|---:|---:|---|
| NONE / original source policy | 13/68 | 原轨迹 | 原始 terminal success |
| target_first_full | 14/68 | 23.18 | 简单目标容器优先 |
| source_first_full | 30/68 | 15.94 | source-first + ledgers + inventory inference |
| source_first_no_history_ledger | 30/68 | 15.94 | 不读取 prefix 历史 ledger |
| source_first_no_deposit_lock | 26/68 | 17.91 | 不锁定 deposit target |
| source_first_no_inventory_inference | 30/68 | 15.94 | 不从 admissible commands 推断当前携带物 |
| source_first_no_instance_ledger | 16/68 | 22.37 | 不维护 delivered/transformed instance ledger |

最关键的是 `source_first_no_instance_ledger` 从 30/68 掉到 16/68，几乎退回 target-first/NONE 水平。这说明当前收益不是只来自“先搜 source object”的启发式，而主要依赖 instance-level ledger：哪些实例已经交付、哪些实例已经 clean/heat/cool、是否还需要第二个实例。

`source_first_no_deposit_lock` 从 30/68 掉到 26/68，说明 deposit lock 对 two-object / multi-instance 任务有明确贡献，但不是所有任务都依赖它。`no_history_ledger` 与 `no_inventory_inference` 暂时不掉，说明当前 dense pool 中多数成功不依赖 prefix 前已完成子目标或 carried-object inference；这也提示下一轮 expansion 应主动采样 partial-progress prefixes。

## 分 block 结果

| block | candidates | NONE | target-first | source-first full | no instance ledger |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B gate dense | 6 | 4 | 4 | 6 | 2 |
| Qwen3-4B gate dense | 11 | 0 | 5 | 7 | 3 |
| Qwen3-8B expansion dense | 21 | 6 | 1 | 10 | 5 |
| Qwen3-4B expansion dense | 20 | 2 | 3 | 6 | 5 |
| Qwen3-4B expansion offsets 9-14 dense | 10 | 1 | 1 | 1 | 1 |
| **total** | **68** | **13** | **14** | **30** | **16** |

## 对 claim 的影响

当前可以更有底气地说：

> 在本地 dense development pool 中，source-first state repair 相对原始 source policy 与 target-first baseline 分别保持 18:1 与 17:1 paired advantage；instance-level ledger 是主要必要组件，去掉后成功率从 30/68 降到 16/68。

但仍不能说：

- 不能说 68-prefix 是论文级主结果；
- 不能说 natural REPLAN / bundle 已在 68 个 prefixes 上全量失败；
- 不能说 source-first 是通用 ALFWorld solver；
- 不能把 prefix rescue rate 写成 task-level final-test success rate。

## 下一步

本轮已经完成“更多 cheap baselines PK”和两组真正的本机 rollout 扩量：

- `Qwen3-4B expansion offsets 3-8`
- raw root：`/home/wsl/traceh-local/raw/source_policy_v2_expansion_4b_o003_o008_20260714`
- report：`local-dev/reports/L3-source-policy-v2-qwen3-4b-expansion-o003-o008-20260714.json`
- result：6 episodes，1 success，5 failures，300 environment steps，600 model calls，0 invalid actions，protocol ok
- `Qwen3-4B expansion offsets 9-14`
- raw root：`/home/wsl/traceh-local/raw/source_policy_v2_expansion_4b_o009_o014_20260714`
- report：`local-dev/reports/L3-source-policy-v2-qwen3-4b-expansion-o009-o014-20260714.json`
- result：6 episodes，3 success，3 failures，193 environment steps，386 model calls，0 invalid actions，protocol ok

跑完后应立即：

1. 优先在 source-first unique-win strata 上补 natural REPLAN / bundle；
2. 将本地目标从 68 prefixes 推进到 80-120，优先新增不同 task family，而不是继续加密同一轨迹；
3. 对 source-only、target-only、none-only、以及新增 4B expansion 中的失败 case 做逐例 error audit；
4. B200 前冻结一个 60-80 task source-validation manifest，避免本地一直滚动加样本；
5. 继续强调：主论文结果必须来自 sealed target task-level final test，不来自这张 prefix 表。
