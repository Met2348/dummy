# TRACE-H 72 小时 Design-Method Kill Test

- **目的：** 判定 Branch-and-Transport policy 是否在 whole-executor pseudo-target 上产生真实 end-to-end 增量。
- **禁止替代：** effect MAE、OT 可视化、trigger rate 或 process maturity 不能作为 Go 依据。
- **决策输出：** DR-0004 Go/Pivot/Stop。
- **执行资源更新：** 导师组已确认 16 张 B200 配额；本文件的 750-900-run 正式 pilot 使用统一 BF16 B200 runtime。本机只执行独立的[约 314-episode development protocol](../experiments/local-development-experiment-plan-zh.md)，其结果不替代 DR-0004。

## 1. Pilot 设置

| 项 | 冻结值 |
|---|---|
| 环境 | ALFWorld |
| Source executors | Qwen3-4B、Qwen3-8B |
| Pseudo-target | Qwen3-14B，完全留出 |
| Harness actions | NONE、CHECK、RETRY、REPLAN |
| Source branch tasks | 30 |
| Target baseline calibration | 20 disjoint tasks |
| Target final test | 30 disjoint tasks |
| Max branch points | 每个 source episode 2 个 |
| Target action feedback before freeze | 0 |
| 预计 runs | 750-900 |

任务 manifest、模型 checkpoint、量化、prompt、action parser、event detector、max steps 和 seed policy 必须在第一条 source run 前写入 freeze file。

## 2. Action 契约

### NONE

不改变 executor 的默认下一步与上下文。

### CHECK

只允许获取或整理完成当前动作所需的 precondition/state evidence；不能直接给出任务答案。

### RETRY

仅在上一 action 被拒绝、无效或未改变环境时触发；规范化该 action 后允许一次重试。

### REPLAN

基于当前 observation 和剩余目标生成最多三步的短计划，随后立即回到 executor；不能替代后续所有控制。

每个非 NONE action 有固定 token/step budget。超过预算计为真实成本和失败，不静默裁剪后重跑。

## 3. Source Branch Bank

1. source baseline run 产生 event prefix；
2. 保存环境 snapshot 或 replay log；
3. 对 prefix 重放并校验 state hash；
4. 从同一 hash 分别运行四 actions；
5. 记录 terminal success、tokens、steps、invalid actions 和 utility；
6. 重复抽查至少 20% branch points，估计 stochastic variance；
7. 形成 append-only branch bank。

若 replay hash 一致率低于 95%，或同 action utility 的重测符号稳定率低于 70%，停止方法实验，优先修 runner。

## 4. Target Seal

Pseudo-target 只运行 20 个 baseline calibration tasks。完成后冻结：

- state extractor 与 scaler；
- event compatibility mask；
- OT cost weights；
- partial mass/regularization；
- uncertainty bootstrap；
- action cost；
- coverage threshold；
- compiled router；
- 所有 baseline configs；
- 30 个 final-test task IDs 和顺序。

冻结文件 hash 后，才允许任何 target final-test non-NONE action。

## 5. 必跑方法

1. No Harness；
2. Best Fixed；
3. Source-AW；
4. Nearest-AW；
5. MF-Gated AW；
6. kNN-Branch；
7. Balanced-OT；
8. TRACE-H Partial-OT + LCB。

若时间允许再加入 Category Router 与 PAR-style penalty，但前八项不可删。

## 6. Primary outcome

预声明：

```text
U = success
    - lambda_token * normalized_tokens
    - lambda_step * extra_steps
    - lambda_invalid * invalid_actions
```

`lambda` 只从 source LOMO 冻结。主报告同时展示 success，不允许通过成本权重把更低成功率包装成胜利。

Primary contrast：

```text
TRACE-H utility - best competing zero-target-action method utility
```

使用 task-paired bootstrap CI；success 用 paired test；30 tasks 只作 kill signal，不宣称最终统计充分。

## 7. Go 条件

必须全部满足：

1. TRACE-H 的 mean utility 高于最强同预算 baseline；
2. paired bootstrap 的优势方向稳定，至少 80% bootstrap replicates 为正；
3. success 不低于最强 baseline 超过 1 个 task；
4. negative intervention rate 低于 Source-AW 与 Balanced-OT；
5. partial OT 相比 balanced OT 或 kNN 至少有一个 end-to-end 增量；
6. branch response 在 source LOMO 中有可迁移信号；
7. 运行 900 次以内可完成且无严重 parser/replay 系统误差。

Pilot 不要求小样本显著性 `p<0.05`，但只改善 secondary diagnostics 不能 Go。

## 8. Pivot 条件

- kNN 最强且 TRACE-H 无增量：保留 branch bank，删除 OT；
- Balanced-OT 最强：删除 partial/unmatched mechanism；
- Source-AW 最强：转为 controller cross-executor benchmark；
- Best Fixed 最强：动态 state representation 失败，先简化 action/event schema；
- success 上升但成本过高：转 budget-aware policy；
- 只有同家族有效：将 claim 限定为 scale transfer，不承诺 cross-family。

## 9. Stop 条件

- TRACE-H utility 不胜最强 baseline；
- target negative intervention 高于 Source-AW；
- replay/state hash 不可靠；
- source branch effects 大多不可重复；
- policy freeze 前读取 target action outcome；
- method gain 来自 prompt/token budget 不公平；
- process frequency 改善但 final success/utility 不改善。

## 10. 72 小时时间盒

### 0-12 小时

- 固定 ALFWorld runner、模型与 task splits；
- 实现 action interface、snapshot/replay 与 state hash；
- 运行 10-task smoke test。

### 12-36 小时

- 收集 source branch bank；
- 训练 Source-AW；
- 运行 source LOMO；
- 冻结 state metric 与 OT hyperparameters。

### 36-48 小时

- 运行 pseudo-target baseline calibration；
- 编译 TRACE-H router 和全部 baseline artifacts；
- 生成 seal hash。

### 48-66 小时

- 按随机化 block 运行 target final-test methods；
- 写 append-only raw records；
- 不查看中间方法排名并改阈值。

### 66-72 小时

- 统一统计与 error audit；
- 输出主表、paired plot、feedback ledger；
- 建立 DR-0004。

## 11. 必交 artifacts

- frozen task/model/action manifest；
- source branch bank 与 state hashes；
- target baseline calibration records；
- policy/baseline artifacts 及 hashes；
- target information ledger；
- 750-900 条 append-only runs；
- end-to-end primary table；
- branch/partial/LCB ablation；
- DR-0004 Go/Pivot/Stop。
