# TRACE-H 本地五方 PK 覆盖报告

- **日期**：2026-07-13
- **目的**：把原始 source policy、自然语言 REPLAN、单步 bundle、target-first macro、source-first macro 放到同一证据表里，检查当前 claim 是否真的超过 baseline。
- **结论一句话**：在 29 个真实 no-progress 接管点上，全覆盖可比的三方结果为：原始 NONE/source policy 7/29，target-first macro 7/29，source-first affordance macro 15/29；source-first 相对 NONE 和 target-first 都是 8 个独有成功、0 个独有失败。自然语言 REPLAN 和 bundle 目前覆盖不足，但在已补跑的 priority-1 样例上都没有产生正收益。

## 报告文件

机器可读报告：

- `local-dev/reports/L3-five-way-pk-coverage-aware-summary-20260713.json`
- `local-dev/reports/L3-five-way-pk-coverage-aware-matrix-20260713.tsv`
- `local-dev/reports/L3-source-first-unique-win-rerun-queue-20260713.json`

汇总脚本：

- `scripts/build_five_way_pk_summary.py`

该脚本不调用模型，只读取已有 raw traces 和本地报告。以后补跑 REPLAN 或 bundle 后，重新运行脚本即可更新覆盖表。

## 五方定义

| 方法 | 覆盖 | 说明 |
|---|---:|---|
| `NONE / source policy original` | 29/29 | 候选所属原始 raw trace 的 terminal score，不重跑模型 |
| `target_first macro` | 29/29 | 同一 prefix 上由 target-first symbolic macro 接管 |
| `source_first affordance macro` | 29/29 | 同一 prefix 上由 source-first + ledger + affordance instrumentation 接管 |
| natural `REPLAN` | 4/29 | 已有昂贵 LLM branch 覆盖 8B gate 的两个 newspaper prefixes、4B gate 的 newspaper p009、8B expansion 的 watch p036，每个 3 seeds |
| `bundle_conservative` | 3/29 | 已有 structured action 覆盖 8B newspaper p003、4B newspaper p009 和 8B watch p036；p003 有两个 smoke 变体 |

因此，当前主结论必须基于前三方；REPLAN/bundle 只说明“早期覆盖样例中没有正收益”，不能写成 29-case 全量结论。

## 全覆盖三方结果

| block | candidates | NONE/source policy | target-first macro | source-first macro |
|---|---:|---:|---:|---:|
| Qwen3-8B gate | 5 | 3 | 3 | 5 |
| Qwen3-4B gate | 7 | 0 | 3 | 4 |
| Qwen3-8B expansion | 17 | 4 | 1 | 6 |
| **total** | **29** | **7** | **7** | **15** |

paired dominance：

| 对比 | source-first 独有成功 | 对方独有成功 |
|---|---:|---:|
| source-first vs NONE/source policy | 8 | 0 |
| source-first vs target-first | 8 | 0 |

这比上一轮只和 target-first 比更强：source-first 不仅超过 target-first，也没有破坏原始 source policy 已经能解决的 7 个接管点。也就是说，在这个 development pool 里它表现为一个 conservative improvement：保留原有成功，再额外修复 8 个失败。

## REPLAN / bundle 覆盖结果

| 方法 | 覆盖候选 | seeds/runs | success |
|---|---:|---:|---:|
| natural REPLAN | 4 | 12 seeds | 0 |
| bundle_conservative | 3 | 4 reports | 0 |

早期 REPLAN/bundle 负结果覆盖 newspaper 与 watch 两类 source-first 独有成功：

- natural REPLAN 在 p003/p012 上 6 个分支 raw terminal score 全为 0；
- natural REPLAN 在 4B newspaper p009 上 3 个分支 raw terminal score 全为 0，且 prefix replay 的 hash、suffix action、terminal score 全部匹配；
- natural REPLAN 在 watch p036 上 3 个分支 raw terminal score 全为 0；
- bundle_conservative 在 p003 上选择 `go to drawer 1`，但 terminal score 仍为 0；
- bundle_conservative 在 4B newspaper p009 上也选择 `go to drawer 1`，置信度 0.3，终局仍为 0，说明“走向 drawer”这种单步 repair hint 不足以维持 two-object subgoal；
- bundle_conservative 在 watch p036 上高置信度选择 `take creditcard 1 from sidetable 2`，但任务目标是 watch，说明单步 bundle 会被局部可取物体误导，terminal score 仍为 0；
- source-first macro 在两个 newspaper prefixes 上均 8 步成功，原因是先找 newspaper source，再锁定 `drawer 1` 作为 deposit target；在 watch p036 上 10 步成功，说明该优势不只存在于 two-object drawer 任务。

这给论文机制叙事提供了一个清晰对照：自然语言重规划和单步修补没有维护“目标实例、source receptacle、deposit receptacle、剩余步数”之间的状态账本，容易在局部可执行动作上短视；source-first ledger 则先把 repair path 的关键实体绑定起来，再执行多步一致修复。

## 对 claim 的更新

当前可以写得更强：

> 在本地 development evidence 中，source-first affordance macro 对原始 source policy 和 target-first macro 都呈现 8:0 的独有成功优势，并且没有牺牲任何原本成功的接管点。这支持把 TRACE-H 写成一种 conservative state-repair harness：只在可执行 repair path 存在时接管，并通过 ledger 保持多步子目标一致性。

当前仍不能写：

- 不能说 natural REPLAN 在 29 个 case 上失败，因为目前只覆盖 4 个候选；
- 不能说 bundle 在 29 个 case 上失败，因为目前只覆盖 3 个候选；
- 不能把 development evidence 写成 sealed target result；
- 不能把 source-first macro 写成通用 symbolic solver，它仍然受 admissible action space 和剩余步数约束。

## 下一步

1. 优先补跑 **source-only vs NONE 的 8 个候选** 上的 natural REPLAN 和 bundle，而不是盲目全量补跑 29 个；这些样例最能回答“LLM replan 是否能替代我们的机制”。其中 priority-1 为 4 个同时赢 NONE 和 target-first 的候选，priority-2 为 4 个赢 NONE 但 target-first 也能成功的候选。
2. 把 affordance diagnostics 接入 runner：当目标可见但不可行动、或剩余步数不足时，macro 应提前停止并输出 router reason，而不是继续 `look` 到 episode end。
3. 若补跑 REPLAN 仍为 0 或明显低于 source-first，则论文可以把 baseline PK 写成：natural language branch / single-step structured action / target-first macro / source-first state repair。
