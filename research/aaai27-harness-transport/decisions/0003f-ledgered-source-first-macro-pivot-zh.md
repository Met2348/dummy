# DR-0003F：从单步 action scorer 转向 ledgered source-first macro

- **日期：** 2026-07-13
- **状态：** accepted for next local branch runner
- **取代范围：** 不废弃 `bundle_conservative`，但不再把它作为主方法形态；它降级为单步 ablation。

## 决策

TRACE-H 当前主线从“单步结构化 action scorer”升级为“带状态账本的 source-first macro transport”。核心机制包括：

- source-first search；
- ALFWorld action grammar 对齐，尤其是 `move X to Y`；
- delivered-instance ledger；
- deposit-target lock；
- transformation ledger；
- inferred inventory from admissible commands。

## 依据

真实 Qwen3-8B Source Policy v2 no-progress candidates 上，`ledgered_source_first_macro` 达到 5/5 success：

- heat egg -> countertop：6 steps，score 1.0；
- two newspaper -> drawer：两个 prefix 均 8 steps，score 1.0；
- book -> sofa：8 steps 与 1 step，score 1.0。

对照 `target_first` 只有 3/5。失败集中在 two-newspaper 任务：它先穷举 drawer，错过 source search 时机，50-step horizon 内无法找到 newspaper。

## 为什么这比单步 bundle 更 promising

单步 bundle 在真实 p003 smoke 中只选出 `go to drawer 1`，而 NONE 也做同样第一步，terminal score 仍为 0。它不能表达：

- 两个实例需要进入同一个具体 receptacle；
- 已交付实例不能再被拿出；
- heat/cool/clean 需要先完成 transformation；
- prefix 可能已经处在 carrying 状态。

这些都是“机制”问题，不是简单 prompt phrasing 问题。AAAI 角度上，主贡献应写成一种可审计的状态化 harness mechanism，而不是诊断性失败分析。

## 下一轮 gate

进入下一阶段前必须完成：

1. 新 branch runner：在同一前缀上比较 `NONE`、`natural_replan`、单步 bundle、`target_first macro`、`source_first ledger macro`。
2. 扩大 prefix pool：覆盖更多 task types 和更多 source failures。
3. 记录每一步 macro state：carried object、delivered objects、transformed objects、deposit target、chosen action reason。
4. 明确非 oracle 边界：只能读 task/observation/history/admissible commands。

只有当 source-side branch PK 继续保持 terminal advantage，才进入 B200 上的大规模 source validation 和后续 target transport。

