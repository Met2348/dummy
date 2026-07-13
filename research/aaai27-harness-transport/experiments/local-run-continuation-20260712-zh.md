# TRACE-H 本机实验续跑：Qwen-ALFWorld Baseline 与 Branch 候选

> **后续结果（2026-07-13）：** 本文提出的 4B NONE/REPLAN kill test 与后续 8B escalation 均已完成。17 对 branches 全为零 terminal utility，执行状态已转为 source-policy pivot。见[最终报告](local-none-replan-source-pilot-final-20260713-zh.md)；本文保留为当时的过程记录。

- **日期：** 2026-07-12
- **续跑范围：** Qwen3-4B baseline adapter、动作解码、50-step ALFWorld micro、NO_PROGRESS 检测、prefix replay
- **当前结论：** L2 工程链路与 parser/replay 门通过；3-task baseline success 为 0，尚不能进入 20-episode 扩量，但已有 6 个可精确重放的 L3 branch candidates

## 1. 新增并实跑的闭环

当前真实数据流已经接通：

```text
ALFWorld observation + admissible commands
  -> Qwen3-4B NF4
  -> constrained command-trie decoder
  -> strict exact-text parser
  -> environment step
  -> state hash + raw output + token/cost record
  -> append-only trace/episode stores
```

每个环境步只有一次模型生成，没有 parser retry。episode summary 通过当前 `traceh-episode` schema；trace 与 summary 分别使用 append-only store，相同 run ID 不能覆盖。最终自动测试为 **119 passed**。

## 2. Decoder 开发审计

所有失败版本均保留，不能只报告最终成功配置。

| 版本 | 设计 | Calls | Invalid | Parser rate | 判断 |
|---|---|---:|---:|---:|---|
| v1 | 自由生成 `ACTION: command` | 36 | 4 | 11.1% | 模型会生成动态 inadmissible action |
| v2 | 严格 `ACTION_ID: n` | 36 | 36 | 100% | 模型常只输出裸编号，协议不匹配 |
| v3 | normalized numeric ID | 36 | 7 | 19.4% | 出现编号+文本与冲突编号，不能继续放宽 parser |
| v4 | 单 token label logits | 36 | 0 | 0% | 合法但有严重 label-position bias，不作为可信 baseline |
| v5 | 完整命令 token-trie | 150 | 0 | 0% | 当前采用；保留命令语义且无重试 |

v4 的偏置证据很明显：其中一个 50-step episode 有 47 次选择标签 `a`，标签随状态映射到不同命令。它解决了合法性，却可能没有在比较命令语义。v5 直接约束完整 `ACTION: <admissible command>` token paths；其 2-step smoke 与原自由生成都选择 `go to fridge 1 -> open fridge 1`，因此当前选择 v5。

## 3. 可信 50-step Baseline 结果

冻结配置：Qwen3-4B、NF4 double quant、BF16 compute、seed `20260712`、history 4、4096 input tokens、50-step horizon、`exact-text-v1 + command-trie-v1`。

| Task | Steps | Success | Invalid | Input tokens | Output tokens | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Heat egg -> countertop | 50 | 0 | 0 | 45,117 | 360 | 36.10 s |
| Two newspapers -> drawer | 50 | 0 | 0 | 34,299 | 342 | 30.54 s |
| Book -> sofa | 50 | 0 | 0 | 25,454 | 360 | 28.74 s |
| **Total** | **150** | **0/3** | **0/150** | **104,870** | **1,062** | **95.38 s** |

原始 summary 见 [command-trie H50 JSON](local-dev/reports/L2-qwen3-4b-alfworld-baseline-command-trie-h50.json)，完整 traces 位于 `/home/wsl/traceh-local/raw/source_baseline/`。

### 失败结构

- egg task：21 个 unique observations，29 次 period-2 revisit，末段在 fridge/countertop 间循环；
- newspaper task：23 个 unique observations，后段重新扫描 drawers；
- book task：15 个 unique observations，27 次 period-2 revisit，末段在 sofa/sidetable 间循环；
- 三条 trajectory 的 score 始终为 0。

因此 `0/3` 不是 parser 或 infrastructure failure，而是当前 Qwen3-4B baseline policy 的规划/记忆失败。样本只有三个任务，不能估计总体成功率；但已经足够否决“直接扩到 20 episodes”的当前配置。

## 4. NO_PROGRESS 与 Prefix Replay

detector 定义：在最近 12 个 prefixes 内，相同 environment observation 与相同 score 第三次出现时触发 `NO_PROGRESS`；同一状态键只保留第一次触发。然后每条 baseline 最多选择 8 个候选，候选间隔至少 4 步。

从三条 baseline 检出 7 个事件，选择 6 个 branch candidates：

| Baseline task offset | Selected steps |
|---:|---|
| 0 | 22 |
| 1 | 25, 34 |
| 2 | 11, 23, 28 |

候选清单见 [L3 candidates](local-dev/reports/L3-no-progress-candidates.json)。在三个全新 ALFWorld 实例中重放 baseline actions 到对应步骤，结果为：

- matching prefixes：6/6；
- match ratio：1.0；
- 所有 field-level diff：空。

逐候选证据见 [L3 prefix replay audit](local-dev/reports/L3-prefix-replay-audit.json)。这六个 prefixes 可以进入 branch pilot，但还没有任何 action outcome。

## 5. 当前证据边界

已经成立：

1. 本机 Qwen-ALFWorld runner 可以持续执行并记录完整 50-step episodes；
2. command-trie decoder 在不增加信息和不重试的条件下将 invalid action 降为 0；
3. baseline failure 中存在清晰、可机器检测的 NO_PROGRESS support；
4. 第一批 6 个 branch 起点能够 100% 精确恢复。

仍未成立：

1. `CHECK/RETRY/REPLAN` 在这些 prefixes 上能改善 terminal utility；
2. action advantage 的符号能够重复；
3. 4B source response bank 不会因为全部 terminal utility 为 0 而退化；
4. TRACE-H 能胜过 NONE、kNN、balanced OT 或其他 baseline；
5. 当前 dirty harness 能作为正式冻结 artifact。

## 6. 下一决策

当前应执行一个极小的 branch kill test，而不是马上跑 20 条 baseline：

1. 在 6 个 prefixes 上先运行 `NONE` 与 `REPLAN`，共 12 个 continuations；
2. 对其中两个 prefixes 重复两次，检查 outcome 与 advantage 符号稳定性；
3. 若 `REPLAN` 在 6 个点全部仍为零 utility，再测试 8B executor，而不是扩展 4B；
4. 若至少出现两个非零、可重复 action effects，再补 `CHECK/RETRY` 并形成四动作 branch bank；
5. 在扩量或 B200 handoff 前，必须将当前项目做 scoped commit 或生成独立 source-tree hash，去掉 `harness_commit=...-dirty`。

这一步的核心问题已经从“runner 能不能跑”转变为：**在可精确恢复的真实 no-progress prefixes 上，新的 harness action 是否能产生非退化、可重复的 terminal advantage。**
