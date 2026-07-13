# TRACE-H 本机深度实验：结构化 action 机制与 transport 变种探索

- **日期：** 2026-07-13
- **状态：** 本机 CPU/synthetic 证据已完成；真实 Qwen/ALFWorld 分支待新 runner 接入
- **结论一句话：** 不应继续扩展旧的自然语言 `REPLAN`；下一轮应把 action 机制改成“anti-loop + precondition check + object-instance ledger”的结构化 bundle，并保留专门机制作为 ablation。

## 1. 为什么做这轮

上一轮真实 Qwen3-8B Source Policy v2 已经证明：source scaffold 本身能在 3 个 ALFWorld 任务上达到 2/3 success，但在 `pick_two_obj_and_place-Newspaper` 的两个 no-progress prefix 上，`NONE/REPLAN` 共 6 对分支 terminal score 全为 0。也就是说，失败不在 parser、replay、显存或 source scaffold，而在 action 机制：一次自然语言 plan 会改变轨迹，却没有带来终局收益。

因此，本轮不再盲目扩大 `REPLAN` 样本，而是回答两个更关键的问题：

1. 哪些结构化 action 机制有清楚的适用边界？
2. 如果这些机制形成 response bank，partial OT + LCB 是否仍比 best-fixed、balanced OT、kNN 有意义？

## 2. 新增代码与实验

新增实现：

- `src/traceh_core/mechanisms.py`：统一实现 5 个机制变种。
- `tests/test_mechanisms.py`：覆盖 loop、precondition、two-object instance ledger、private/no-support abstain。
- `scripts/run_mechanism_design_synthetic.py`：L4.2 机制设计合成 PK。
- `scripts/run_mechanism_transport_synthetic.py`：L4.3 机制作为 action 的 transport 压力测试。

验证：

- `uv run pytest`：146 passed。
- `L4-synthetic-transport-refresh-20260713.json`：L4.1 transport 三件套刷新通过。
- `L4-mechanism-design-synthetic-20260713.json`：L4.2 通过。
- `L4-mechanism-transport-synthetic-20260713.json`：L4.3 通过。

## 3. 方法变种

| 变种 | 机制 | 预期作用 | 本轮边界 |
|---|---|---|---|
| `natural_replan` | 额外自然语言 plan | 作为旧 REPLAN 对照 | 在 precondition/two-object 场景负干预高 |
| `anti_loop_retry` | 避免同状态重复动作，排除 passive/no-op | loop escape | 只解决循环，不解决缺前置条件 |
| `precondition_check` | 根据可见 inventory/container 状态先修前置条件 | take/open/put 顺序 | 强，但不是完整记忆机制 |
| `subgoal_ledger` | 记录已交付实例，避免回头拿已完成对象 | two-object memory | 不会自动处理 closed-container |
| `bundle_conservative` | anti-loop + precondition + instance ledger，低支持 abstain | 下一轮主机制 | synthetic 中最好，但仍需真实 Qwen 分支验证 |

重要：这些机制只使用 visible task、observation、history、admissible commands，不读取 `expert_plan`、gold action 或 target outcome。

## 4. L4.2 机制设计结果

合成规模：20 seeds，480 cases，2400 method rows。

| 变种 | mean utility | accuracy | actionable success | negative intervention | private abstain |
|---|---:|---:|---:|---:|---:|
| `bundle_conservative` | 0.7200 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| `precondition_check` | 0.5308 | 0.7917 | 0.7222 | 0.0000 | 1.0000 |
| `subgoal_ledger` | 0.3088 | 0.6250 | 0.5000 | 0.1667 | 1.0000 |
| `anti_loop_retry` | 0.0308 | 0.4583 | 0.2778 | 0.5417 | 1.0000 |
| `natural_replan` | 0.0158 | 0.4583 | 0.2778 | 0.5417 | 1.0000 |

断言全部通过：

- bundle 排名第一；
- bundle private abstain >= 95%；
- anti-loop 在 loop escape 中 accuracy >= 95%；
- precondition 在 missing-take 中 accuracy >= 95%；
- ledger 在 two-object 中 accuracy >= 95%；
- natural REPLAN 负干预率高。

解释：单个专门机制在自己的小域内强，但会在别的故障类型上失效。bundle 的价值不是某个子规则更聪明，而是把“循环、前置条件、实例记忆、无支持 abstain”编成同一个 conservative action。

## 5. L4.3 mechanism transport 结果

合成规模：30 seeds，4 scenarios。action set 为 `NONE/natural/anti_loop/precondition/ledger/bundle`。

关键结果：

| 场景 | best fixed utility | fixed bundle utility | balanced OT utility | partial OT utility | partial private none |
|---|---:|---:|---:|---:|---:|
| matched known support | 0.9600 | 0.9600 | 0.9698 | 0.9800 | 1.0000 |
| private 25% | 0.6195 | 0.6195 | 0.6195 | 0.7197 | 1.0000 |
| source skew loop rare | 0.7148 | 0.7565 | 0.7150 | 0.7749 | 1.0000 |
| semantic conflict | 0.6913 | 0.6913 | 0.6913 | 0.7703 | 1.0000 |

断言全部通过：

- private shift 下 partial OT 比 best-fixed 至少高 0.10 utility；
- partial OT private none rate >= 90%；
- balanced OT private negative rate >= 20%；
- source skew 下 partial OT 比 fixed bundle 至少高 0.015 utility。

解释：L4.3 不支持“partial OT 大幅击败所有固定机制”的强 claim；它支持的是更窄但有价值的 claim：当 target 有 source 未覆盖状态或 source 分布偏斜时，partial/unmatched mass + LCB/NONE 能减少负迁移，并获得小到中等增量。

## 6. 对真实实验的直接影响

下一轮真实 Qwen/ALFWorld 不应再跑旧 `NONE/REPLAN`。建议新建 `STRUCTURED_REPAIR_BUNDLE v0.2` 分支 runner，比较：

1. `NONE`
2. `natural_replan`：旧机制对照，只保留少量样本
3. `anti_loop_retry`
4. `precondition_check`
5. `subgoal_ledger`
6. `bundle_conservative`

候选 prefix 来自 Qwen3-8B Source Policy v2：

- failure-primary：`pick_two_obj_and_place-Newspaper` 的 p003、p012；
- non-regression：heat-egg p007、book-to-sofa p010/p015。

真实 gate：

- failure-primary 中至少 2 个 seed-0 prefix-action 组合 terminal advantage > 0；
- repeated sign stability >= 70%；
- non-regression 中 bundle 不得把原本会成功的 continuation 变成失败；
- parser/infrastructure failure 仍为 0 或 <= 2%；
- 若只提高 trajectory diversity、不提高 terminal score，仍判为不通过。

## 7. 证据边界

本轮是机制设计和 transport 压力测试，不是论文主结果。它不能证明 TRACE-H 在 ALFWorld target 上有效，也不能替代 B200 BF16 source/target 分支。它能支持的决策只有：旧自然语言 REPLAN 不值得继续扩大；结构化 bundle 是下一轮真实 branch 的首选机制；partial OT 的主要卖点应写成“private/unmatched target state 下的保守不迁移”，而不是泛泛声称总能大幅超过 fixed controller。
