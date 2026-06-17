# L13 · 设计决策树 + 生产 Checklist

## 决策树:这任务该用什么?

```
                    ┌─ 任务只有一步? ──是──→ 单次 LLM 调用
                    │
   任务来了 ────────┼─ 步骤固定可枚举? ─是─→ 能画固定流程图?
                    │                          ├─是,顺序依赖 → Prompt Chaining (L04)
                    │                          ├─是,分类分流 → Routing (L05)
                    │                          ├─是,独立多面 → Parallelization (L06)
                    │                          └─是,有清晰评判 → Evaluator-Optimizer (L08)
                    │
                    ├─ 子任务运行时才知道数量? ─是→ Orchestrator-Workers (L07)
                    │
                    └─ 需与环境反复交互、轨迹不可预测? ─是→ Autonomous Agent (L09)
                                                             (并加满下面的护栏)
```

**默认从树的顶端开始,越往下越贵越不可控,能停早就停早。**

## 生产 Checklist(上线前过一遍)

### 控制与安全
- [ ] **停机条件**齐全(完成判据 + max_steps + 无进度检测)
- [ ] **危险操作有护栏**(写/删/转账 → 人审或权限门,见 Topic 9 [permission-system](../../agent-harness-design/lectures/09-permission-system.md))
- [ ] **成本上限**(token/调用/$ 预算,超了硬停)

### 上下文与工具
- [ ] context 有预算核算 + compaction 策略(L10)
- [ ] 工具集**精选**,描述清晰,无名字撞车(L12 tool sprawl)
- [ ] 工具错误**surface 成结构化结果**,不 swallow(L12)
- [ ] 大工具输出**裁剪**后再进 context

### 可靠性
- [ ] 关键判断考虑 voting / evaluator 复核(L06/L08)
- [ ] evaluator 与 generator **不同视角**(防自我确认)
- [ ] 链/循环不过长(累积失败率)

### 可观测
- [ ] 全程 **trace**(每步 kind/label),可复盘(Topic 9 [observability](../../agent-harness-design/lectures/13-observability.md))
- [ ] **cost tracking**(调用数/token/$)
- [ ] 失败可定位、可重放

### 人机协同
- [ ] 高风险步骤 **human-in-the-loop**
- [ ] 用户可**中途打断/纠偏**(Topic 9 [streaming-steering](../../agent-harness-design/lectures/11-streaming-steering.md))

## 一句话总纲

> **能用确定性代码控制的,别交给模型;必须交给模型的,加满护栏和观测。**

## 退出条件
- [ ] 能用决策树给一个真实任务定方案
- [ ] 把生产 checklist 当上线前的硬门槛
- [ ] 复述总纲并理解它统一了整个专题
