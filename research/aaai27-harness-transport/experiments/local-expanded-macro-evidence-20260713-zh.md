# TRACE-H 本地扩展实验证据：ledgered source-first macro

- **日期**：2026-07-13
- **问题**：在真实 ALFWorld no-progress 接管点上，`source_first + ledger` 是否只是少数样例有效，还是稳定优于 `target_first`？
- **结论一句话**：扩展到 29 个真实 no-progress 接管点后，`source_first + ledger` 为 15/29，`target_first` 为 7/29；paired discordant cases 为 source-only 8、target-only 0，方向性很强，说明贡献不是“写了一个 symbolic macro”，而是“先恢复源对象/携带状态，再锁定目标容器”的机制选择。

## 输入报告

本轮纳入三组来源，均为本地真实回放环境，不使用 expert action 或 outcome oracle：

| block | source-first report | target-first report | candidates |
|---|---|---:|
| Qwen3-8B gate | `local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-8b-all-candidates-h80-v4-affordance.json` | `local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-8b-all-candidates-h80-v1.json` | 5 |
| Qwen3-4B gate | `local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-4b-gate-candidates-h80-v4-affordance.json` | `local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-4b-gate-candidates-h80-v1.json` | 7 |
| Qwen3-8B expansion | `local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-8b-expansion-candidates-h80-v4-affordance.json` | `local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-8b-expansion-candidates-h80-v1.json` | 17 |

机器可读汇总固定在：

- `local-dev/reports/L3-symbolic-search-macro-expanded-summary-20260713.json`
- `local-dev/reports/L3-symbolic-search-macro-affordance-diagnostics-20260713.json`

v4-affordance 报告在每个 macro step 额外记录 `target_visible_before`、`target_actionable_commands_before`、`inventory_before`、`delivered_before`、`transformed_before` 和 `deposit_target_before`。这让“看见目标物但没有可行动命令”的失败边界可以直接从报告复现。

## 总体结果

| 指标 | source-first + ledger | target-first |
|---|---:|---:|
| success / candidates | 15/29 | 7/29 |
| positive terminal score | 15/29 | 7/29 |
| both success | 7 | 7 |
| source-only success | 8 | 0 |
| target-only success | 0 | 0 |
| both fail | 14 | 14 |

如果只看 paired discordant cases，8 个样例只被 source-first 解决，0 个样例只被 target-first 解决。把这当作小样本 sign test，双侧值为 0.0078125，单侧 source-first advantage 为 0.00390625。这里不应写成最终统计显著性主结论，因为样本是从本地失败池抽取的 development set；但它足以支持下一步把方法重心从 target-first/单步 action scorer 转向 source-first ledgered state repair。

## 分任务结果

| task goal | candidates | source-first success | target-first success | 解释 |
|---|---:|---:|---:|---|
| heat some egg and put it in countertop | 4 | 2 | 2 | 早接管可修，晚接管中 egg 可见但无 `take egg` admissible command |
| put two newspaper in drawer | 4 | 3 | 0 | source-first 的 deposit-target lock 是关键，target-first 先扫 drawer 会耗尽步数 |
| put a book in sofa | 4 | 4 | 4 | 简单搬运任务，两者都能修；source-first 步数更短 |
| clean some tomato and put it in countertop | 3 | 3 | 0 | transformation 任务强支持 source-first：target-first 先穷举 countertop/containers，错过 tomato source |
| put some watch on sidetable | 3 | 2 | 1 | source-first 在中等剩余预算下更稳；极晚接管仍失败 |
| put two spraybottle in cabinet | 5 | 0 | 0 | 只能找到并交付 `spraybottle 1`，第二实例在 admissible action space 中不可达 |
| put a newspaper in sofa | 1 | 1 | 0 | source-first 快速找到 newspaper，target-first 耗尽搜索预算 |
| put a cool pot in diningtable | 2 | 0 | 0 | 未出现可行动的 `pot`，只见 `potato` 等无关物 |
| put two box in coffeetable | 3 | 0 | 0 | observation 可见 box，但没有 `take box` admissible commands；p046 只能交付已携带的一件 |

## 机制证据

这轮最重要的结果不是 15/29 本身，而是 paired pattern：

1. **source-first 独有收益集中在需要先找物体/先做 transformation 的任务**。clean tomato 三个接管点全部 source-first 成功、target-first 失败；two-newspaper 三个可修接管点也只被 source-first 修复。
2. **ledger 是必要机制，不是装饰**。two-newspaper 任务需要把两个实例放入同一个具体 drawer，不能交付第一个后换 drawer，也不能把已交付实例重新拿出来。
3. **inferred inventory 有真实作用**。book/box 的部分接管点中，prefix 可能已经让 agent 携带目标物，macro 需要从 admissible move commands 推断当前 inventory，而不是从空 inventory 重新搜索。
4. **admissible-action 边界必须进入方法**。多个失败不是因为 observation 没有目标物，而是目标物可见但不可行动。例如 heat-egg p022/p030 里 observation 出现 `egg 1/2/3`，但 admissible commands 里没有任何 egg command；box p025 中 observation 出现 `box 1/2/3`，也没有 `take box` command。

## 失败分类

| 失败类型 | 典型样例 | 含义 |
|---|---|---|
| hard late prefix | p041 newspaper, p042 watch, p044/p048 spraybottle, p046 box | 接管时剩余 2-9 步，环境 50-step horizon 已经太紧；这类应由 router 早期触发，而不是靠 macro 硬救 |
| visible but non-actionable object | egg p022/p030, box p025/p038 | observation 中有目标物，但 admissible commands 没有 take/action；下一版必须维护 affordance ledger |
| multi-instance insufficiency | spraybottle p003/p025/p030, box p046 | 可以交付一件，但任务要求两件；若第二实例不可行动，macro 应停并上报不可修复，而不是 `look` 到终止 |
| target object absent in admissible space | cool pot p012/p027 | 没有可行动 `pot`，只有 `potato` 等干扰项；需要区分 substring false positive 与 action-space support |

affordance diagnostics 的聚合计数如下：

| diagnostic | count |
|---|---:|
| source-first failures | 14 |
| failures with visible-but-unactionable target step | 5 |
| failures with partial delivery | 4 |
| failures with remaining budget <= 9 | 5 |
| failures with no actionable target step at all | 8 |

## 对论文 claim 的影响

当前可以扩张的 claim：

> 在真实 LLM 失败接管点中，存在一类可由非 oracle、低步数、状态化 harness macro 修复的前缀；关键机制是 source-first recovery、delivered/transformed/deposit ledger、inferred inventory 与 admissible-action affordance filtering。与 target-first 搜索顺序相比，source-first 在 paired development evidence 上呈现 8:0 的独有成功优势。

当前不能写的 claim：

- 不能说 macro 是通用 ALFWorld solver。
- 不能说所有 no-progress 都可修。
- 不能说 observation 中看到目标物就一定可行动。
- 不能把 development set 的 sign test 当正式 target split 统计结论。

## 下一轮实验

1. 在同一 branch runner 中纳入 `NONE`、natural `REPLAN`、单步 `bundle_conservative`、`target_first macro`、`source_first ledger macro`，做统一 paired PK。
2. 增加 `affordance ledger`：记录“可见目标物但没有 admissible action”的状态，并在不可修复时停止 macro、交给 router 或更高算力 agent。
3. 增加 `remaining-budget router`：prefix 剩余步数小于估计 repair steps 时不再尝试完整 macro，而是标记为 late irrecoverable 或进入更激进策略。
4. 在 HiPerGator 上扩展到全 ALFWorld valid_seen/valid_unseen、多模型、多 seed，并把这 29 个本地 development cases 冻结为 design evidence，不再作为最终 target 调参集。
