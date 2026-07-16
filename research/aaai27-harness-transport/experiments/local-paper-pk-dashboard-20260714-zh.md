# TRACE-H 本地论文级 PK Dashboard v0

- **日期**：2026-07-14
- **定位**：local development dashboard；不是 sealed target final result
- **目标对应**：先把 6+ baselines、Ours、两类 metrics、3 个本机数据源、统计检验和可视化管线跑通，再决定 Ours 的下一轮迭代方向。

## 一句话结论

当前本机 dashboard 纳入 **3 个数据源**、**10 个独立 baseline 方法**、**4 个 mechanism ablations**。Ours 在三个数据源的 primary metric 上均为 rank 1，且相对所有 paper-baseline comparison 都达到 paired/sign-test `p < 0.05`。但这还不能声称论文目标完成，因为只有 ALFWorld dense prefixes 是真实 benchmark 数据源；另外两个是 synthetic stress datasets，而且 ALFWorld 目前只有 2 个独立 paper baselines 加 4 个 ablations。

## 机器产物

- `local-dev/reports/L5-local-paper-pk-dashboard-20260714.json`
- `local-dev/reports/L5-local-paper-pk-dashboard-table-20260714.tsv`
- `local-dev/figures/L5-local-paper-pk-primary-20260714.svg`
- `local-dev/figures/L5-local-paper-pk-secondary-20260714.svg`
- `scripts/build_local_paper_pk_dashboard.py`

## 数据源与两类 metrics

| 数据源 | 单位 | Ours | primary metric | secondary metric |
|---|---:|---|---|---|
| ALFWorld dense prefixes | 68 prefixes | TRACE-H source-first full | success rate | success per 10 steps |
| Synthetic mechanism design | 1000 seeds | bundle conservative | mean utility | accuracy |
| Synthetic transport stress | 1000 seeds | partial OT-LCB | mean utility | safety rate |

## 主结果

| 数据源 | Ours primary | 最强 paper baseline primary | Ours secondary | 最强 paper baseline secondary |
|---|---:|---:|---:|---:|
| ALFWorld dense prefixes | 0.441 | 0.206 (`target-first macro`) | 0.741 | 0.403 |
| Synthetic mechanism design | 0.720 | 0.531 (`precondition check`) | 1.000 | 0.792 |
| Synthetic transport stress | 0.823 | 0.760 (`balanced OT-LCB`) | 1.000 | 0.852 |

ALFWorld paired sign tests:

| 对比 | mean delta | wins | losses | ties | p-value |
|---|---:|---:|---:|---:|---:|
| Ours vs NONE/source policy | +0.250 | 18 | 1 | 49 | 7.63e-05 |
| Ours vs target-first macro | +0.235 | 17 | 1 | 50 | 1.45e-04 |
| Ours vs no instance ledger | +0.206 | 14 | 0 | 54 | 1.22e-04 |

这说明 ALFWorld 当前证据里，真正支撑机制贡献的是 `instance-level ledger`；`history ledger` 和 `inventory inference` 在当前 68-prefix pool 上没有产生额外收益，因为去掉它们后仍是 30/68。

## 当前不满意处

1. **还不是 3 个真实数据集。** 当前是 1 个真实 benchmark 数据源 + 2 个 synthetic stress 数据源。它能证明方法机制与统计管线，但不能替代 WebShop / MiniWoB / ScienceWorld / BabyAI 等真实环境。
2. **ALFWorld 还不是 6 个独立 paper baselines。** 现在有 `NONE` 与 `target-first macro` 两个独立 baseline，另有 4 个 ablations。下一轮必须补满 natural REPLAN、bundle、kNN/nearest response、balanced transport、partial transport 或其他可执行 baseline。
3. **Ours 在 ALFWorld 不是唯一第一。** `no_history_ledger` 和 `no_inventory_inference` 与 full Ours 并列 30/68，说明 full 方法中的这两个组件在当前采样分布上不是必要组件。
4. **还没有 sealed target task-level final。** 当前所有结论仍是 development evidence，不能写成 AAAI 主表。

## 下一轮迭代目标

本机下一轮应该优先做三件事：

1. 把 ALFWorld dense prefixes 从 68 推到 80-120，并补齐 full-coverage baselines：`NONE`、natural `REPLAN`、single-step `bundle`、`target-first macro`、`source-first no-instance`、`balanced/nearest response transfer`、Ours。
2. 主动采样 partial-progress prefixes，而不是继续采样纯 no-progress prefixes，用来检验 `history ledger` 与 `inventory inference` 是否真的有用。
3. 把 dashboard 升级为 sealed-ready：每个数据源固定 two metrics、paired test、CI、可视化和 failure audit queue，迁移到 B200 后直接换成三套真实 benchmark。
