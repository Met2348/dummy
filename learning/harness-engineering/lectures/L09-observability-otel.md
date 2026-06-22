# L09 · 生产可观测性：OpenTelemetry 式 trace

> Part III · 40-min lecture · 配套代码 `src/otel_trace.py` · 目标: 把 agent 运行做成结构化、可追溯的 trace, 而不是一团 print 日志。

---

## 0. 8 小时任务在第 7 小时挂了，你怎么办

这是可观测性的灵魂拷问:

> 一个 8 小时的自治工作流, 在第 7 小时失败。「我们精确知道哪一步、哪个工具、什么输入出的事」和「我们得把整段会话重放一遍」之间的差距, 是**以工程师工时和客户信任来计量的**。

没有结构化 trace, 你只能重放 (慢、贵、可能复现不出)。有了它, 你直接跳到出事的那个 span。

---

## 1. 2026 标准：OpenTelemetry + LLM span 类型

> **生产标准**: 用 **OpenTelemetry** 描述 agent 运行, 并加上 LLM 特有的 span 类型。**每个 reasoning step 是一个 span; 每个 tool call 是它的 child span。**

> **span**: 一段有名字、有起止、有属性的「工作单元」。span 可以**嵌套** (child span), 形成一棵树, 精确刻画「谁在谁里面、各花了多久」。OpenTelemetry 是分布式系统可观测的事实标准, 这里把它借来描述 agent。

agent 的 span 层级:

```
▭ window-0                         一个上下文窗口 = 一个 span
  ◆ reason@w0.s1                   一步推理 = 一个 span
    → tool:search                  这步里的工具调用 = child span
    → tool:read_file
  ◆ reason@w0.s2
    → tool:write
▭ window-1                         换窗口 (L05) = 新的顶层 span
  ▣ subagent:analyze               子 agent (L06) = 套一层 span
    ◆ reason ...
```

对应 `src/otel_trace.py` 的 `Tracer`:

```python
tr = Tracer()
with tr.span("window-0", "window"):
    with tr.span("reason@w0.s1", "reasoning"):
        with tr.span("tool:search", "tool", args="x"):
            ...
print(tr.render())     # 文本树
tr.stats()             # {reasoning: {count, duration}, tool: {...}}
tr.to_dict()           # 可导出成 OTel 风格 JSON
```

> 注意一个工程细节: `otel_trace.py` 用**逻辑时钟** (单调计数器) 而非墙钟时间。为什么? 为了让 notebook / 测试的 trace 输出**可复现** (墙钟每次都不同, 没法做确定性断言)。生产里换成真实时间戳即可——这又是一处「机制对了, 换个时钟源就上生产」。

---

## 2. trace 不只是调试，它是四件事的基础

```
结构化 trace  ──┬──►  调试   (出事跳到那个 span, 不重放)
                ├──►  成本归因 (每个 tool/step/window 花了多少 token/钱 → CostTracker)
                ├──►  评测   (L10: 比较两个 harness 配置的 trace)
                └──►  审计   (L08: 谁/何时/做了什么, 可追溯)
```

- 你 Module 7 `tracing.py` 的 `Trace + CostTracker` 已有雏形; 本专题把它对齐到 **OTel span 树**的生产形态。
- `stats()` 这种聚合 (各 kind 的 span 数与时长) 是「这个 harness 把时间花在哪了」的第一手数据——是优化的起点。

---

## 3. 可观测性的成熟度阶梯

```
Level 0  print 大法              出事靠肉眼翻日志           (玩具)
Level 1  结构化日志              JSON 行, 可 grep           (能用)
Level 2  span 树 (本讲)          嵌套、可定位、可聚合         (生产入门)
Level 3  OTel + 后端             接 Jaeger/Honeycomb 等      (生产)
         + 成本/质量指标          每 span 带 token/cost/eval
```

本专题 `otel_trace.py` 把你带到 Level 2 (span 树的机制与导出)。Level 3 是把 `to_dict()` 的输出喂给真实 OTel 后端——接口已经对齐, 是工程接线问题。

---

## 4. 本讲小结 + 通往 L10

- 可观测性的价值在「长任务出事时能精确定位, 而非重放」。
- 2026 标准: OTel + LLM span——reasoning step = span, tool call = child span, window/subagent 各套一层。
- trace 是调试/成本归因/评测/审计四件事的共同基础。
- 用逻辑时钟保证可复现; 生产换真实时间戳。

> **下一讲 L10**: 你能看见 harness 了, 但它**好不好**? 怎么量化「这个 harness 比那个强」? 关键事实——**同一个模型, 换 harness, SWE-bench 差几十分**。`src/harness_eval.py` + N3 notebook 让「harness 是自变量」变得可测量。
