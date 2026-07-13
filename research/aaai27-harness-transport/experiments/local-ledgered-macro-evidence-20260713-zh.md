# TRACE-H 本机实验证据升级：ledgered source-first macro

- **日期：** 2026-07-13
- **范围：** Qwen3-8B Source Policy v2 产生的真实 ALFWorld no-progress prefixes
- **结论一句话：** 单步结构化 action 还不够；带状态账本的 `source_first` macro 在 5 个真实失败/停滞前缀上达到 5/5 success，而 `target_first` 对照只有 3/5。

## 1. 为什么继续扩展

上一轮真实 Qwen3-8B branch smoke 说明，单步 `bundle_conservative` 只能把 Newspaper p003 的第一步从空转引向 `go to drawer 1`，但这一步与 NONE continuation 的第一步重合，terminal score 仍为 0。因此，问题不是“能否选出一个看似合理的动作”，而是“机制能否跨多步保持任务状态、动作语法和已完成子目标的一致性”。

这轮实验将候选机制从单步 command scorer 扩成一个可复现实验用 macro controller。它只读取 visible task、observation、history 和 admissible commands，不读取 expert action、gold state 或 target outcome。

## 2. 新增机制

当前有效机制可以概括为 `ledgered_source_first_macro`：

1. **Goal parser：** 从 `Your task is to:` 后抽取目标物体、目标 receptacle、数量，以及 `heat/cool/clean` 这类 transformation。
2. **Action grammar ledger：** 把 ALFWorld 原生命令 `move X to Y` 识别为交付动作，而不是只识别自然语言中的 `put`。
3. **Source-first search：** 未携带目标物体时，优先搜索 source surfaces/containers；拿到物体后再转入目标 receptacle。
4. **Deposit-target lock：** 第一次交付到 `drawer 1` 后，第二个同类实例必须回到同一个具体 receptacle，而不是换到另一个 drawer。
5. **Delivered-instance ledger：** 已交付的 `newspaper 2` 不再被重新拿出，避免“放入-拿出-放入”循环。
6. **Transformation ledger：** 对 `heat egg` 任务，先执行 `heat egg 1 with microwave 1`，再允许交付到 countertop。
7. **Inferred inventory：** 如果 prefix 已经携带物体，且 admissible commands 中出现 `move book 1 to sofa 1`，无需本地 inventory 初始化也可直接交付。

代码位置：

- `experiments/scripts/run_symbolic_search_macro_probe.py`
- `experiments/src/traceh_core/mechanisms.py`
- `experiments/tests/test_mechanisms.py`

## 3. 主结果

报告文件：`experiments/local-dev/reports/L3-symbolic-search-macro-source-first-qwen3-8b-all-candidates-h80-v3.json`

| candidate | 任务类型 | macro steps | score | success | 关键动作 |
|---|---|---:|---:|---|---|
| p007 | heat egg -> countertop | 6 | 1.0 | yes | find egg, heat in microwave, move to countertop |
| p003 | two newspaper -> drawer | 8 | 1.0 | yes | deposit target locked to drawer 1 |
| p012 | two newspaper -> drawer | 8 | 1.0 | yes | same as p003 |
| p010 | book -> sofa | 8 | 1.0 | yes | find book on sidetable, move to sofa |
| p015 | book -> sofa, already carrying | 1 | 1.0 | yes | direct inferred-inventory delivery |

总计：5 candidates，5 successes，5 positive terminal scores。

## 4. 对照结果

报告文件：`experiments/local-dev/reports/L3-symbolic-search-macro-target-first-qwen3-8b-all-candidates-h80-v1.json`

| variant | success | 失败点 |
|---|---:|---|
| `target_first` | 3/5 | 两个 Newspaper prefix 先穷举 drawer，50-step horizon 内来不及找到 newspaper |
| `source_first + ledger` | 5/5 | 无失败 |

这个对照很关键：成功不是来自“写了一个 symbolic macro”本身，而是来自明确的机制设计选择：先找 source，再锁定 deposit target，并维护 delivered/transformed ledger。

## 5. 和单步 bundle 的差距

真实单步 smoke 报告：`experiments/local-dev/reports/L3-structured-action-smoke-v2-qwen3-8b.json`

单步 `bundle_conservative` 在 Newspaper p003 上选择 `go to drawer 1`，但 NONE continuation 也选择同一步，后续仍然反复搜索 drawer/sidetable，terminal score 为 0。它缺少三件事：

- 不知道第二个 newspaper 也必须进入同一个具体 drawer；
- 不记录已交付实例，容易把已放下的物体重新拿出；
- 不处理 `heat/cool/clean` transformation 子目标。

因此正式方法不应继续写成“单步诊断/单步修复”，而应写成“状态化 harness action macro + conservative transport/routing”。

## 6. 证据边界

这不是最终论文主实验。它是 source-side counterfactual upper-bound 和机制设计证据：

- 使用真实 Qwen3-8B 失败前缀和真实 ALFWorld 环境；
- 使用真实 admissible command space；
- 不使用 expert plan 或 outcome oracle；
- 但当前 macro 是规则实现，不是 LLM 自主生成的 harness policy；
- 还没有在 sealed target、更多 task types、更多 seeds 上验证。

因此它支持的 claim 是：失败前缀中存在可执行、低步数、非 oracle 的 repair trajectory；关键贡献应转向“ledgered macro mechanism + transport/routing”，而不是自然语言 REPLAN。

## 7. 下一轮实验

1. 将 `ledgered_source_first_macro` 接入 branch runner，与 `NONE`、`natural_replan`、单步 `bundle_conservative`、`target_first macro` 做同前缀 PK。
2. 扩展候选集：从当前 5 个 no-progress prefixes 扩到更多 source failures，不只依赖 Qwen3-8B gate 的少量停滞点。
3. 把 macro 输出改写成可被 LLM/harness 调用的结构化 contract：`SEARCH_SOURCE`、`TRANSFORM_OBJECT`、`LOCK_DEPOSIT_TARGET`、`DELIVER_TO_LOCKED_TARGET`。
4. 在 B200 上运行更大 source validation 后，再决定是否进入 target transport。

