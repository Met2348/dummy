# L14 · Capstone：升级版 harness 跑通长任务 + 产出 idea 卡

> Part IV · 40-min lecture + 动手 · 配套 `notebooks/N4-capstone.ipynb` + `src/*` 全家桶 · 目标: 把全课串成一个能跑的升级版 harness, 并把工程落成研究。

---

## 0. Capstone 要交付什么

把本专题所有 src 组件**串成一个 harness**, 跑通一个**跨上下文窗口的长任务**, 全程可观测、可评测, 最后产出你自己的研究 idea 卡。

```
升级版 harness = provider(L03) + compaction(L04) + long_horizon/hook(L05)
                 + otel_trace(L09) + harness_eval(L10)
                          │
                 跑通一个跨窗口长任务 (会 early-stop, 被 hook 救回)
                          │
                 ├─ 全程 OTel trace (window/reasoning/tool span 树)
                 ├─ 开/关 compaction 的成本对照 (harness_eval)
                 └─ 产出 2-3 张 harness 方向 idea 卡 (复用 critical-reading-gap 模板)
```

---

## 1. 组件如何咬合（一张装配图）

```
   ┌─────────────────────── run_long_horizon (L05 外循环) ───────────────────────┐
   │                                                                              │
   │   window 循环:                                                               │
   │     _seed_messages = system + goal + FileStateStore.summary()  ← 跨窗口记忆   │
   │     ┌── step 循环 ──────────────────────────────────────────────┐           │
   │     │  Tracer.span("reason", ...)              ← L09 可观测       │           │
   │     │    provider.stream(messages, tools)      ← L03 模型抽象     │           │
   │     │    Tracer.span("tool:...")  → dispatch   ← L07 工具         │           │
   │     │    Compactor.compact(messages)           ← L04 窗口内压缩   │           │
   │     │  hook: stop 且 goal 未达成 → 拦截, 换窗   ← L05 长任务       │           │
   │     └──────────────────────────────────────────────────────────┘           │
   │   goal_met(state) → success                                                  │
   └──────────────────────────────────────────────────────────────────────────────┘
        harness_eval (L10): 把上面整体当一个配置, 和别的配置对照成功率/成本
```

每一根线都对应一讲。**Capstone 的意义不是写新代码, 而是看清这些组件如何咬合成一个整体**——这正是 L12 说的「harness 作为一个连贯工件」, 你亲手把散落的零件装成了那个工件。

---

## 2. 动手步骤（在 N4-capstone.ipynb 里）

1. **装配**: 用 `demo_setup` 起一个长任务 (6 步, 第 2 步会 early-stop), 配 `Compactor` + `Tracer`。
2. **跑通**: `run_long_horizon(..., compactor, tracer, hook=True)` —— 观察它跨 2 个窗口、被 hook 救回、成功。
3. **可观测**: `print(tracer.render())` 看 window/reasoning/tool 的 span 树; `tracer.stats()` 看时间花在哪。
4. **评测**: `evaluate(default_configs(), ...)` 跑三配置对照, 确认 hook 决定成败、compaction 决定成本。
5. **落成研究**: 用 `critical-reading-gap/templates/idea-card.md`, 把 L13 的 G9/G10 (或你自己扫出的 gap) 写成 2-3 张 idea 卡, 过三筛。

---

## 3. 你完成本专题后应该能（产出 checklist）

- [ ] 说清 harness 的 4 本构要素, 并用它鉴定任意 agent 系统是不是真 harness (L02)
- [ ] 把一个 harness 从 MockProvider 换到真 provider, 主体不改 (L03)
- [ ] 解释并实现 5 阶段渐进式 compaction, 说清每阶段意图与不变量 (L04)
- [ ] 用 loop-with-hook + 文件系统 state 跑通一个跨窗口长任务 (L05)
- [ ] 给 harness 装 OTel 式 trace, 出事能定位到 span (L09)
- [ ] 设计对照实验证明「同模型换 harness 改变成败/成本」(L10)
- [ ] 说清五大架构模式 + 统一调度框架, 按任务选型 (L11)
- [ ] 用 6 类 gap 雷达扫出 harness 的研究题目, 产出 idea 卡 (L13)
- [ ] **完整跑通一次「工程 → 找 gap → idea 卡」的循环**

---

## 4. 收口：这门课在你 portfolio 里的位置

```
Module 7 agent-harness-design (理解层)  ──► harness-engineering (工程层 + 前沿)
                                                    │
                                            Part IV 研究桥
                                                    │
                                                    ▼
                                    Module 9 critical-reading-gap (研究技能)
                                    用 gap 雷达 + idea 卡, 把 harness 变成你的 PhD 候选方向
```

这是你 portfolio 里**第一个「工程 ⨯ 研究」双栖专题**: 它既是一块硬工程肌肉, 又是一个真实的研究入口。**如果你将来 PhD 想做 agent/harness 方向, 这门课就是你的起点; 即便不做, 它也让你成为「能 systems engineer 地思考 agent」的少数人。**

> 下一步 (留给你, 也可让我继续): 选 L13 的 G9 或 G10, 用 `experiment-design` (Module 9.4, 待建) 把那张 idea 卡的「最小验证实验」做严谨, 跑出第一个信号。
