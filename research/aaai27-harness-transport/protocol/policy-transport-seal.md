# TRACE-H Cross-Executor Policy Transport Seal

## 1. 目标

证明 target executor 的任何非 NONE harness-action outcome 都没有参与 state metric、transport plan、router、baseline 选择或阈值冻结。

## 2. 四个数据区

1. `SOURCE_BRANCH`：来源 executor 的 baseline prefixes 与全部 action branches；
2. `SOURCE_VALIDATION`：leave-one-source-executor-out 选择算法与超参数；
3. `TARGET_BASELINE_CALIBRATION`：目标 executor 只执行 NONE 的未标注 state trajectories；
4. `TARGET_FINAL_TEST`：policy freeze 后才允许执行非 NONE actions。

前三区可用于 policy freeze；第四区在 unseal 前不可读。

## 3. Freeze 前必须提交

- git commit 与环境 lockfile；
- model checkpoints/quantization/chat template；
- task manifests 与 split hashes；
- action semantics、masks、budgets；
- event detector 与 state extractor code hash；
- source branch bank hash；
- target baseline calibration raw hash；
- feature scaler/embedding version；
- transport cost、partial mass、regularization；
- uncertainty/bootstrap protocol；
- utility weights；
- router artifact；
- 每个 baseline artifact；
- primary contrast 与 statistics script hash；
- target final-test execution order。

## 4. Target Information Ledger

每个方法必须记录：

```text
method_id
source_action_outcomes_seen
target_baseline_episodes_seen
target_action_outcomes_seen_before_test
target_action_outcomes_seen_online
extra_test_time_calls
tokens
steps
wall_clock
```

same-budget 主表要求 `target_action_outcomes_seen_before_test = 0`。

## 5. 允许的信息

- 公开 model metadata；
- target baseline trajectories；
- environment/tool schema；
- action validity masks；
- source action outcomes；
- source LOMO validation results。

## 6. 禁止的信息

- target 上任何 CHECK/RETRY/REPLAN branch reward；
- target final-test task 的 method outcome；
- target method ranking；
- 根据 target non-NONE failure 调 event taxonomy、OT weight、coverage 或 LCB；
- 根据 target result 更换 quantization/prompt/parser；
- 从 MASA evolved target skills 或 SkillAdaptor target adaptations 向 TRACE-H 回流信息。

## 7. Baseline 隔离

MASA evolved、SkillAdaptor、target-AW 和 SkVM-JIT 可读取额外 target action outcomes，但必须：

1. 在 TRACE-H seal 之后运行；
2. 使用独立 artifact 目录；
3. 不回写 TRACE-H source/target calibration；
4. 在 extra-feedback 表中单独报告；
5. 明确 target outcome budget。

## 8. Unseal 顺序

1. 验证 freeze hashes；
2. 只读加载 task manifest；
3. 按预声明 block 顺序执行同预算方法；
4. append raw records；
5. 全部运行完成后一次性计算 primary table；
6. 再运行 extra-feedback methods；
7. 最后计算 oracle branch values 与 policy regret。

## 9. 泄漏处理

任何 target non-NONE outcome 在 freeze 前被查看，当前 target 立即失去 prospective 身份。不能删除日志后继续声称 sealed；必须更换从未运行的新 target executor/checkpoint，或将结果降为 retrospective development。
