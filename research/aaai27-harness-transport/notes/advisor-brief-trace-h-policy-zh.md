# 导师一页简报：TRACE-H Cross-Executor Harness Policy Transport

> `executor`、`branch`、`partial OT`、`unmatched mass`、`LCB` 等定义及完整例子见[导师版术语与符号说明](trace-h-terminology-guide-zh.md)。

## 一句话

> 不在目标模型上试跑任何 harness action，能否把来源模型学到的 CHECK/RETRY/REPLAN 控制策略运输到未见 LLM executor，并直接提高目标任务成功率与成本效用？

## 为什么这是方法问题

基础模型变化会改变 agent dynamics。同一个 harness action 在来源模型上有效，在目标模型上可能冗余或有害。已有方法分别能在固定 executor 上学习 controller、用目标反馈搜索/改写 skill、按 capability 编译 skill，或在目标失败后继续适配；但它们没有在 whole-executor holdout、零目标 action outcome 的条件下运输一个可执行 runtime policy。

## 新机制

```text
1. Branch
   在来源 trajectory 的同一 prefix 分支运行
   NONE / CHECK / RETRY / REPLAN
   -> event-level action advantage bank

2. Transport
   只运行目标 baseline trajectories
   -> partial OT 对齐到 source response states
   -> target-private states 保留 unmatched

3. Compile
   transported action LCB - execution cost
   -> runtime router: action or NONE
```

这不是静态 patch 选择。Router 在运行时根据当前 state 作控制决策，产物可直接执行。

## 与谁正面 PK

### 同预算第一主表

`No Harness, Best Fixed, Source-AW, Nearest-AW, MF-Gated AW, Category Router, kNN-Branch, Balanced-OT, PAR-style penalty, TRACE-H`。

TRACE-H 必须超过其中最强者。

### 直接论文系统

- **Offline-RL Harness：** 固定 executor learned controller；
- **MASA：** target model-aware static skill evolution/rewriter；
- **SkillAdaptor：** target failure + rerun qualification；
- **SkVM：** capability-based AOT/JIT skill compiler；
- **Partial Harnessing：** theory-driven static coverage。

所有方法分别报告 source episodes、target baseline episodes、target action outcomes 和 test-time cost，避免拿零反馈方法与大量 target search 方法假装同预算。

## 主实验

- 环境：ALFWorld 主实验，WebShop 第二环境；
- source executors：Qwen3-4B/8B/14B；
- sealed targets：Qwen3-32B 与一个 Gemma 跨家族模型；
- target adaptation：只允许 baseline trajectories；
- target final test：policy freeze 后第一次执行非 NONE action；
- primary：success、normalized utility、negative intervention、oracle policy regret；
- secondary：action-value error、alignment cost、matched mass。

## 72 小时 pilot

- ALFWorld；
- source Qwen3-4B/8B；
- pseudo-target Qwen3-14B 完整留出；
- 30 source branch tasks、20 target baseline calibration、30 final-test；
- 约 750-900 runs；
- 必跑 8 个同预算方法。

唯一 Go 条件：TRACE-H 的 **end-to-end utility** 胜最强同预算 baseline。预测更准、聚类更好、CHECK 更多但任务不提升，全部算失败。

## 新颖性边界

不能声称首个 controller、model-aware adaptation、skill compiler、dynamic router 或 baseline-only predictor。可争取的贡献是：

> counterfactual branch supervision + response-aware partial transport + conservative executable policy，在 zero-target-intervention whole-executor holdout 上成立。

## 当前判断

- Idea overall：8.1/10；
- 技术深度：8.8/10；
- excitement：8.9/10；
- 工作量可行性：3.9/10；
- 当前自有证据：1/10；
- 工作量：155-230 聚焦人时；
- 决策：只批准 72 小时 method kill test，结果通过后才投入完整矩阵。

正式方案见[Policy Transport Proposal](../proposals/trace-h-policy-transport-proposal-zh.md)，证据边界见[全文审计](../foundations/notes/method-design-evidence-boundary-zh.md)，公平比较见[PK 矩阵](trace-h-method-pk-matrix-zh.md)，决策见[DR-0003B](../decisions/0003b-trace-h-policy-transport-pivot-zh.md)。
