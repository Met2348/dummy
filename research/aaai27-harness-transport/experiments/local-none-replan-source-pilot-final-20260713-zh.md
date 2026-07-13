# TRACE-H 本机源策略分支实验最终报告：Qwen3-4B/8B NONE vs REPLAN

- **日期：** 2026-07-13
- **实验层级：** 本机 development pilot，不是论文主实验
- **模型：** Qwen3-4B、Qwen3-8B，NF4 double quant，BF16 compute
- **环境：** ALFWorld eval-in-distribution，50-step horizon
- **结论：** 当前 4B/8B source policy 与现有 prompt/scaffold 不能形成非退化 response bank；停止扩展 `CHECK/RETRY`，转向修复 source policy 或 task signal。该结果没有测试、也没有否定 TRACE-H transport 算法。

## 1. 本轮到底完成了什么

本轮把此前停在 prefix replay 的工作继续到了可判定状态，完成了以下闭环：

```text
4B/8B 本机量化可运行性
  -> 3-task x 50-step baseline
  -> NO_PROGRESS candidate 抽取
  -> prefix 精确重放
  -> deterministic NONE 全后缀一致性
  -> paired NONE/REPLAN + repeated seeds
  -> 原始轨迹级跨模型联合审计
  -> Go/Pivot/Stop 决策
```

`REPLAN` 的语义保持为一次额外模型调用，最多 192 tokens，不立即执行环境动作；后续动作仍由同一 executor 在 command trie 中选择。`NONE` 不增加调用和环境步。所有 branch 起点先从 baseline actions 恢复，不读取 oracle action 或 target outcome。

## 2. Runtime 与 checkpoint

| 模型 | 权重文件 | 权重字节 | 加载时间 | 模型 footprint | 4096-token 峰值显存 | L0 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-4B | 3 | 8,044,982,000 | 8.34 s | 2.42 GiB | 3.38 GiB | 通过 |
| Qwen3-8B | 5 | 16,381,516,776 | 15.17 s | 5.55 GiB | 6.66 GiB | 通过 |

8B checkpoint 的 10 项 SHA-256 manifest 校验全部通过，manifest 见 [Qwen3-8B.sha256](local-dev/models/Qwen3-8B.sha256)。量化结果只证明本机工程可运行性，不能与后续 B200 BF16 主结果混表。

## 3. Baseline 结果

| 模型 | Tasks | 总步数/调用 | 成功 | Invalid | Parser failure | 判断 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-4B | 3 | 150 | 0/3 | 0 | 0/150 | runner 合法，但策略循环且无任务收益 |
| Qwen3-8B | 3 | 150 | 0/3 | 0 | 0/150 | 扩大模型未修复 source policy 退化 |

两组 baseline 都通过 parser/infrastructure gate，却没有任何成功或非零 terminal score。因此问题不在 action legality，而在当前 executor 的规划、记忆和循环恢复能力。原始汇总见 [4B baseline](local-dev/reports/L2-qwen3-4b-alfworld-baseline-command-trie-h50.json) 与 [8B baseline](local-dev/reports/L2-qwen3-8b-alfworld-baseline-command-trie-h50.json)。

## 4. Replay 与 continuation integrity

| 模型 | NO_PROGRESS candidates | Prefix hash 精确恢复 | deterministic NONE 全后缀一致 | Terminal score 一致 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 6 | 6/6 | 6/6 | 6/6 |
| Qwen3-8B | 3 | 3/3 | 3/3 | 3/3 |

这里验证的不只是 branch 起点 hash。我们从每个候选 prefix 继续运行 deterministic `NONE` 到 episode 结束，并逐步比较整个 action suffix；9 个候选全部完全一致。因此后面的 paired branch 差异不能归因于恢复到了不同环境状态。

证据文件：

- [4B prefix replay](local-dev/reports/L3-prefix-replay-audit.json)
- [8B prefix replay](local-dev/reports/L3-qwen3-8b-prefix-replay-audit.json)
- [4B NONE continuation](local-dev/reports/L3-none-continuation-integrity.json)
- [8B NONE continuation](local-dev/reports/L3-qwen3-8b-none-continuation-integrity.json)

## 5. Paired NONE/REPLAN 主结果

前两个 candidate 使用 seeds 0/1/2，其余 candidate 使用 seed 0。4B 共 10 对、20 条 branches；8B 共 7 对、14 条 branches。

| 模型 | Pairs | REPLAN 改变 action suffix | REPLAN 未改变 suffix | 正 utility | 负 utility | 所有 raw scores 为 0 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 10 | 10/10 | 0/10 | 0 | 0 | 是 |
| Qwen3-8B | 7 | 3/7 | 4/7 | 0 | 0 | 是 |
| **合计** | **17** | **13/17** | **4/17** | **0** | **0** | **是** |

所有 17 个 REPLAN plans 均为非空；每个 NONE 的 extra calls 为 0，每个 REPLAN 为 1；branch parser failure 为 0。4B 的每对轨迹有 16-32 个 action 不同，首个分歧都出现在 continuation 的第 0 或第 1 步，说明 plan 确实进入并改变了 policy。8B 只有 3 对发生改变，差异 action 数分别为 28、24、31，另外 4 对整个 suffix 不变。

因此不能把结果简化为“REPLAN 没生效”：

1. 对 4B，机制干预明显改变行为，但没有产生 terminal utility；
2. 对 8B，机制既存在 guidance adoption 不稳定，也没有 terminal utility；
3. 两个模型都没有产生可用于估计 action advantage 的非零 source response。

机器可复算的联合结果见 [L3 multimodel audit](local-dev/reports/L3-multimodel-source-pilot-audit.json)。它由 [分析脚本](scripts/analyze_multimodel_source_pilot.py)直接读取两份 branch report 和 34 条 append-only trace records 生成。

## 6. 决策与证据边界

联合决策是：`PIVOT_SOURCE_POLICY_OR_TASK_SIGNAL`。

立即停止以下动作：

- 不在当前 4B/8B branch candidates 上补跑 `CHECK/RETRY`；
- 不把 17 对全零结果包装成 response bank；
- 不进入 14B development-target transport PK；
- 不把 B200 用于原样放大当前 source runner。

同时保留以下主张：

- prefix replay、NONE continuation、append-only records 与 action contract 已通过工程门；
- 当前失败定位在 source policy/scaffold 与 sparse task signal，而不是 replay；
- transport、partial OT、LCB、target policy 和跨 executor utility 尚未被实验测试；
- 所以本轮既不是 TRACE-H 的正证据，也不是其方法反证。

8B 单模型历史报告中的 `MOVE_TO_QWEN3_8B` 是复用了 4B 的固定标签；原始报告不改写。联合审计将其显式标为 stale，并由当前跨模型决策覆盖。

## 7. 下一轮只做 Source Policy Gate v2

下一轮不是扩样，而是做一个三任务 source-policy 修复门：

1. 保留同一 ALFWorld tasks、50-step horizon、command trie、exact parser、seed 和 raw-record schema；
2. 只改 executor scaffold：加入 benchmark-aligned 的短期 subgoal、已访问对象/容器记忆和显式 anti-loop recovery；
3. 允许使用 development/train demonstrations 构造 prompt，但运行时禁止 oracle state、gold action 和额外环境权限；
4. 先跑 3 个 deterministic baselines；至少 1/3 成功，才重新抽取 NO_PROGRESS branches；
5. 再跑最小 NONE/REPLAN paired test；至少 2 个 seed-0 candidates 为正 advantage，且 repeated sign stability >=70%，才恢复四 action branch bank；
6. 未过上述门，不运行 14B target，也不提交 B200 扩量。

Qwen3-14B 继续保持本机 `DEV_TARGET` 身份，不能因为 4B/8B 失败就临时改为 source 后再冒充 held-out target。正式 B200 设计仍可预声明 14B source，但必须与本机 development split 分开记录。

## 8. 对 AAAI-27 的现实影响

本轮把一个潜在的万级错误扩量提前拦住了，但也说明当前项目还没有方法收益证据。距离项目记录中的摘要截止日很近，下一步必须把时间集中到 Source Policy Gate v2；若该门仍失败，应停止把 TRACE-H 作为本轮 AAAI-27 主投方向，而不是靠更多全零 branches 维持表面进度。

