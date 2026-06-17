# L15 · 评测 Harness

## 核心命题:评的是 harness,不只是 model

回到 L01 的实证:**同模型、换 harness,分数差几十分**。所以 agent benchmark 测的是 **model × harness** 的联合体。优化 harness 是提分的独立杠杆。

## 主流 agent benchmark(回顾 agent-code-eval 专题)

| Benchmark | 测什么 | harness 影响点 |
|-----------|--------|---------------|
| **SWE-bench** | 真实 GitHub issue 修复 | 文件检索、编辑工具、测试反馈回灌 |
| **Terminal-Bench** | 终端任务 | 命令工具、输出裁剪、错误恢复 |
| **WebArena / OSWorld** | Web / OS 操作 | 观察-动作循环、截图处理 |
| **τ-bench** | 工具+对话 | 工具设计、多轮、安全 |

`agent-code-eval` 专题从**评测视角**跑这些;本节从**harness 视角**看"哪些 harness 设计决定了分数"。

## harness 的哪些设计影响分数

| 设计 | 影响 |
|------|------|
| 工具粒度/描述(L04) | 模型选对工具的概率 |
| context 管理(L05-06) | 长任务不丢关键信息 |
| 错误回灌(L03/L12) | 模型能否从失败中纠正 |
| loop guard / max_turns(L12) | 不卡死、不烧爆 |
| 权限(L09) | 既安全又不过度限制 |

## 怎么用 A/B 对照评 harness

固定 model + 固定任务集,只改 harness 的一个变量,比较:

```
harness_v1 (无错误回灌) vs harness_v2 (错误 surface 回灌)
          ↓ 同模型同任务
       成功率 / 平均回合数 / 平均成本 对照
```

本仓库的 capstone 已经演示了这种对照思想:同 model 同工具,只改 **permission mode**,结果(是否写成功、最终回答)就不同——这正是"harness 变量 A/B"的微缩版。

## 评测信号来自可观测(L13)

trace + cost tracker 就是评测指标的来源:

| 指标 | 来自 |
|------|------|
| 成功率 | 最终结果是否达标 |
| 效率 | `tracker.model_calls` / 回合数 |
| 成本 | `tracker.usd()` |
| 安全 | 权限 deny 次数、是否碰危险操作 |
| 稳健 | loop guard 是否触发、重试次数 |

## 设计要点

1. **优化 harness 是独立提分手段**,别只盯着换模型。
2. **A/B 要控变量**:一次只改一个 harness 设计,否则归因不清。
3. **多维度评**:不只成功率,还要成本/安全/稳健(呼应 τ-bench 五维)。

## 退出条件
- [ ] 理解 benchmark 测的是 model × harness
- [ ] 说出至少 3 个影响分数的 harness 设计
- [ ] 知道怎么用 A/B + trace/cost 评一个 harness 改动
