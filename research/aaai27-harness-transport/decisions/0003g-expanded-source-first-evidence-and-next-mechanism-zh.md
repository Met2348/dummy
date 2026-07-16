# DR-0003G：扩展 source-first 证据后，引入 affordance-aware state repair

- **日期**：2026-07-13
- **状态**：accepted for next mechanism iteration
- **取代范围**：不推翻 0003F；把 0003F 的 `ledgered_source_first_macro` 从 5-case 正证据升级为 29-case mixed evidence，并新增 `affordance ledger` 与 `remaining-budget router` 作为下一版必要机制。

## 决策

TRACE-H 当前主线继续推进 `source_first + ledger`，但正式机制从“ledgered source-first macro”升级为：

> **affordance-aware source-first state repair**：先恢复可行动源对象与携带状态，再做 transformation 与目标容器锁定；同时记录目标物是否真正出现在 admissible action space 中，并根据剩余步数决定继续修复、停止上报不可修复，或交给更高成本策略。

这意味着论文设计不能只写“搜索顺序从 target-first 改成 source-first”。新机制至少包含：

1. source-first object recovery；
2. delivered-instance ledger；
3. transformed-instance ledger；
4. deposit-target lock；
5. inferred inventory；
6. admissible-action affordance ledger；
7. remaining-budget router；
8. multi-instance completion checker。

## 证据

本轮把 evidence 从原始 8B gate 的 5 个 no-progress candidates 扩到 29 个 paired candidates：

| block | candidates | source-first success | target-first success | source-only | target-only |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B gate | 5 | 5 | 3 | 2 | 0 |
| Qwen3-4B gate | 7 | 4 | 3 | 1 | 0 |
| Qwen3-8B expansion | 17 | 6 | 1 | 5 | 0 |
| total | 29 | 15 | 7 | 8 | 0 |

机器可读证据见：

- `experiments/local-dev/reports/L3-symbolic-search-macro-expanded-summary-20260713.json`
- `experiments/local-dev/reports/L3-symbolic-search-macro-affordance-diagnostics-20260713.json`
- `experiments/local-expanded-macro-evidence-20260713-zh.md`

paired discordant cases 为 8:0，说明 source-first 不是偶然换个启发式，而是在可修前缀上捕捉到 target-first 不具备的状态恢复结构。尤其是 clean tomato 和 two-newspaper：target-first 会先消耗步数确认目标 receptacle，source-first 则先找目标物、做 transformation 或锁定首个 deposit target。

## 为什么需要新增 affordance-aware 机制

扩展结果也显示，继续声称“source-first macro 可以救大多数 no-progress”会过宽。失败 trace 暴露出一个更有论文价值的边界：**目标物可见不等于目标物可行动**。

例子：

- heat-egg p022/p030：`countertop 2` observation 中出现 `egg 1/2/3`，但 admissible commands 中没有任何 egg command；
- box p025/p038：`dresser/sofa` observation 中出现 `box 1/2/3`，但没有 `take box` command；
- spraybottle p003/p025/p030：能交付 `spraybottle 1`，但第二个 spraybottle 不在后续可行动空间中；
- cool-pot p012/p027：没有可行动 `pot`，只有 `potato` 等干扰项。

v4-affordance 重跑后，14 个 source-first 失败中有 5 个出现 visible-but-unactionable target step，4 个出现 partial delivery，5 个接管时剩余预算不超过 9 步，8 个全程没有任何目标物 actionable step。这些都不是“再多搜几步”能解决的问题，而是 router 和 affordance contract 应该显式建模的问题。

因此下一版不能只追求更长搜索或更激进搜索。正确机制是让 harness 判断：

- 当前 prefix 是否仍有可执行 repair path；
- 失败是因为模型迷路，还是因为 action space 已不支持目标实例；
- 剩余环境步数是否足够完成 multi-instance 或 transformation chain；
- 什么时候应该停止 local macro，转交给更高成本策略或标记为 irrecoverable。

## 对 AAAI 叙事的提升

0003F 的叙事是“source-first ledger macro 比单步 action scorer 更强”。0003G 后，叙事可以更像主会设计论文：

> LLM harness 不应只在失败时重新 prompt 或重新规划，而应维护一个可审计的任务状态账本，并基于 admissible action affordance 判断当前失败是否可修、应由哪类 repair controller 接管、何时停止无效搜索。TRACE-H 的贡献是把失败接管从自然语言重规划转为状态化、可验证、可路由的 policy transport。

这个叙事比单纯诊断更强，因为它提出了新的机制组合，并且有 paired development evidence 支持机制选择。

## 风险

1. 当前样本仍是 development evidence，不能作为最终 sealed target 结论。
2. source-first 成功集中在若干任务族，正式实验必须覆盖更多 task type。
3. affordance ledger 可能被审稿人认为接近 hand-engineered ALFWorld controller；论文需要强调它是 harness-level contract/routing evidence，不是要提交一个通用 ALFWorld symbolic solver。
4. multi-instance 任务的失败会拉低 aggregate success；必须把“不可行动第二实例”与“策略未找到第二实例”分开标注。

## 下一步 gate

进入下一阶段前必须完成：

1. 把 `affordance ledger` 写入 macro trace schema：每步记录 target object 是否 visible、是否 actionable、是否 carried、是否 already delivered。
2. 新增 `remaining_budget_estimate`：估计从当前 prefix 到完成至少需要多少步。
3. 在同一 branch runner 中做五方 paired PK：`NONE`、natural `REPLAN`、`bundle_conservative`、`target_first macro`、`source_first affordance-aware macro`。
4. 本地 development 只允许继续用于机制设计；正式 claim 等 HiPerGator 全量 source validation 和 sealed target split 后再写死。
