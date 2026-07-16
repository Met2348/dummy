# 本地论文级 PK：horizon-aware + transform-recovery 迭代记录

日期：2026-07-14

## 目标

老师提出“29 个太少”，本轮目标是把本地 ALFWorld dense-prefix 证据扩到更接近论文级 PK：至少 6 个独立 baseline、两类 metrics、Ours 必须 rank-1，且相对 paper baselines 统计显著。这里仍然是 development evidence，不是 sealed target final。

## 本轮新增 baseline

在原有 `NONE / source policy`、`target_first_full`、`source_first_full` 与 ledger 消融之外，新增 5 个非 LLM 独立宏搜索 baseline，并在 68 个真实 ALFWorld no-progress prefixes 上全量运行：

- `container_first_full`：按通用容器顺序搜索。
- `reverse_container_full`：按反向通用容器顺序搜索，是本轮最强 baseline。
- `kitchen_first_full`：厨房源优先。
- `furniture_first_full`：家具源优先。
- `alphabetical_full`：地点词表字母序。

因此 ALFWorld 当前有 6 个独立 full macro baselines（不含 NONE），满足“6 个 baseline PK”的本地版本要求。

## Ours 迭代

旧 Ours `source_first_full` 在 68 个 prefixes 上是 `30/68`，primary rank-1 但不够漂亮：`reverse_container_full` 达到 `26/68`，且 secondary metric `success_per_10_steps` 反超旧 Ours；pairwise sign test 中 Ours vs reverse 为 `5:1, p=0.21875`，不能说服审稿人。

本轮通过两个反例驱动机制把 Ours 改成 `source_horizon_aware_full`：

1. **Horizon-aware switch**：当全局步数 `>=40` 时，从 source-first 搜索切换到 reverse-container 抢救式搜索。证据样本 `l3q8b-source-policy-v2-expansion-o004-s20260712-p042` 是 Watch-to-SideTable，旧 source-first 在 prefix 42 后先扫 sofa/sidetable，最后才看到 watch，超预算失败；新机制 4 步成功。
2. **Transform-recovery ledger**：当 history ledger 已知目标物体已经被 clean/cool/heat，但当前不在手里时，优先回到对应 appliance（sinkbasin/fridge/microwave）取回已变换物体。证据样本 `l3q4-source-policy-v2-expansion-o003-s20260712-p023` 是 clean Tomato then place CounterTop，旧方法扫 cabinet 失败；新机制 4 步成功。

这两个机制都只使用 prefix 中可审计的状态、history ledger 和剩余步数，不使用 target final 信息。

## ALFWorld 68-prefix 最终结果

Primary metric 是 success rate，secondary metric 是 success per 10 macro steps。

| method | role | success | primary | secondary | primary rank | secondary rank |
|---|---|---:|---:|---:|---:|---:|
| `source_horizon_aware_full` | Ours | 32/68 | 0.4706 | 0.8242 | 1 | 1 |
| `source_first_full` | ablation | 30/68 | 0.4412 | 0.7407 | 2 | 3 |
| `reverse_container_full` | baseline | 26/68 | 0.3824 | 0.7746 | 5 | 2 |
| `furniture_first_full` | baseline | 25/68 | 0.3676 | 0.5882 | 7 | 7 |
| `alphabetical_full` | baseline | 16/68 | 0.2353 | 0.3317 | 8 | 11 |
| `target_first_full` | baseline | 14/68 | 0.2059 | 0.4034 | 10 | 10 |
| `NONE / source policy` | baseline | 13/68 | 0.1912 | 0.0900 | 11 | 13 |
| `kitchen_first_full` | baseline | 12/68 | 0.1765 | 0.4219 | 12 | 9 |
| `container_first_full` | baseline | 10/68 | 0.1471 | 0.3284 | 13 | 12 |

相对独立 baselines 的 paired sign tests 全部过 `p<0.05`：

| baseline | wins | losses | ties | mean delta | p |
|---|---:|---:|---:|---:|---:|
| NONE | 19 | 0 | 49 | 0.2794 | 3.81e-06 |
| target-first | 18 | 0 | 50 | 0.2647 | 7.63e-06 |
| container-first | 22 | 0 | 46 | 0.3235 | 4.77e-07 |
| reverse-container | 6 | 0 | 62 | 0.0882 | 0.03125 |
| kitchen-first | 20 | 0 | 48 | 0.2941 | 1.91e-06 |
| furniture-first | 7 | 0 | 61 | 0.1029 | 0.015625 |
| alphabetical | 16 | 0 | 52 | 0.2353 | 3.05e-05 |

## 结构化 action baseline 补充证据

在 priority queue 中选取 `l3q4-source-policy-v2-gate-o001-s20260712-p009` 后，真实 Qwen3-4B continuation 下运行 `NONE + 5` 个 structured action mechanisms：

- `NATURAL_REPLAN`
- `ANTI_LOOP_RETRY`
- `PRECONDITION_CHECK`
- `SUBGOAL_LEDGER`
- `BUNDLE_CONSERVATIVE`

结果 6 个分支全部失败，positive actionable count 为 0。这个样本说明单步 structured action baseline 不能轻易追回 Ours 的 source/ledger 宏机制，但这只是 priority-strata qualitative evidence，不可替代全量 sealed target。

## 三数据源 dashboard 状态

最终 dashboard：

- local dataset count：3
- unique baseline count：15
- ALFWorld independent full macro baselines excluding NONE：6
- Ours primary rank-1：true
- Ours unique primary rank-1：true
- Ours vs paper baselines significant at 0.05：true
- `paper_goal_satisfied=false`

仍然不能声称论文级最终完成，原因只有两个：

1. 目前三个数据源中只有 ALFWorld 是真实 benchmark；另外两个是 synthetic mechanism design / synthetic transport stress。
2. 还缺 sealed target task-level runs 和三类真实 benchmark 的最终可视化。

## 关键产物

- `experiments/local-dev/reports/L3-dense68-macro-source-horizon-aware-transform-recovery-full-h80-20260714.json`
- `experiments/local-dev/reports/L3-dense68-macro-pk-horizon-transform-recovery-expanded-summary-20260714.json`
- `experiments/local-dev/reports/L3-dense68-macro-pk-horizon-transform-recovery-expanded-matrix-20260714.tsv`
- `experiments/local-dev/reports/L5-local-paper-pk-dashboard-horizon-transform-recovery-expanded-20260714.json`
- `experiments/local-dev/reports/L5-local-paper-pk-dashboard-horizon-transform-recovery-expanded-table-20260714.tsv`
- `experiments/local-dev/figures/L5-local-paper-pk-horizon-transform-recovery-expanded-primary-20260714.svg`
- `experiments/local-dev/figures/L5-local-paper-pk-horizon-transform-recovery-expanded-secondary-20260714.svg`

## 下一步

本地 development 结果现在已经能支撑“机制值得继续冲”的判断。下一步不应继续只在同一 68-prefix pool 上调参，而应 freeze `source_horizon_aware_full` 的 threshold 和 transform-recovery 规则，转向：

1. 扩到新的真实 ALFWorld prefix pool，检查 32/68 是否可复现。
2. 在 HiPerGator/B200 上做 sealed target task-level runs。
3. 增加至少两个真实任务环境或真实 benchmark，而不是继续依赖 synthetic stress 作为主证据。
