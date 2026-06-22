# L04 · 5 阶段渐进式 compaction

> Part II · 40-min lecture · 配套代码 `src/compaction.py` · notebook `N1` · 目标: 把「上下文满了怎么办」从一刀切 summarize, 升级成生产级的**逐级压缩**。

---

## 0. 上下文窗口是预算，不是仓库

> **context engineering**: 决定每一步把哪些指令、证据、中间产物、state 放进上下文。它是 prompt engineering 的升维——不只是「写好一句 prompt」, 而是「在一个会被反复填满的窗口里, 持续地策展信息」。

长任务跑着跑着, 上下文会被工具输出、历史对话塞满。一旦逼近窗口上限, 你必须**腾地方**。最笨的做法是「满了就把前面全 summarize 成一段」——但这会**一次性丢掉大量信息**。

生产级做法 (Claude Code 逆向揭示): **渐进式 (progressive)**——能用轻手段就不用重手段, 逐级升级, 每级只丢「当前最该丢的」。

---

## 1. 五个阶段（逐级加重）

```
上下文逼近预算
      │
  ┌───▼─────────────────────────────────────────────┐
  │ Stage 1  budget reduction  截断单条超大 tool 输出   │  最轻, 几乎不丢语义
  │ Stage 2  snip              丢最老的低价值消息        │  丢冗余历史
  │ Stage 3  microcompact      把一段较老消息压成短摘要   │  有损压缩一部分
  │ Stage 4  context collapse  仅留最近 K 条, 余者塌缩    │  大幅压缩
  │ Stage 5  auto-compact      整窗重置: sys+全量摘要+最近 │  最重, 近乎重开
  └──────────────────────────────────────────────────┘
   每一级都先试; 不够再升级。system / pinned 永不丢。
```

对应 `src/compaction.py` 的 `Compactor.compact()`:

```python
comp = Compactor(max_tokens=600)
new_msgs, events = comp.compact(messages)   # 逐级施用直到 <= 600
for e in events:
    print(e.stage, e.name, e.tokens_before, "→", e.tokens_after)
```

`compact()` 的核心逻辑: 按 Stage 1→5 顺序, **每个阶段反复施用直到本阶段无效**, 还不达标就升到下一阶段, 直到落进预算或手段用尽。每次施用记一条 `CompactionEvent` (哪一级、压缩前后 token), 这就是 N1 里能画出「token 曲线 + 阶段标记」的原料。

---

## 2. 逐个阶段讲（设计意图）

### Stage 1 · budget reduction（截断超大单条）
一个 tool 读了个 10MB 日志, 单条就撑爆窗口。先**截断这种异常大的单条** (`SINGLE_MSG_CAP`), 几乎不影响别的。**最便宜、最该先做。**

### Stage 2 · snip（丢最老的低价值消息）
保护 system 和**最近 N 条** (`RECENT_PROTECT`), 从最老的开始丢「低价值」消息 (旧 tool_result / 旧对话)。一次丢一条, 丢到达标为止。**丢的是冗余, 不是信息密集区。**

### Stage 3 · microcompact（局部摘要）
把「较老的一段」压成一条短摘要 (约原 token 的 25%), 插回原位。**有损, 但只损一部分历史, 近期上下文完好。**

### Stage 4 · context collapse（大塌缩）
只保留最近 K 条 + pinned, 其余**全部塌缩成一条运行摘要** (约 12%)。窗口已经很紧时用。

### Stage 5 · auto-compact（整窗重置）
最后手段: `[system, 全量摘要(~8%), 最近两条]`——近乎重开一个窗口。这也是 **L05 long-horizon 跨窗口续跑**的衔接点: 当一个窗口实在压不下去, 就该换新窗口了。

> 关键不变量 (本专题反复强调):
> - **system / pinned 消息永不丢** (角色定位、目标、关键约束必须常驻)。
> - **最近 N 条受保护** (当前正在做的事不能被压掉)。
> - 压缩**单调减少 token** (每个 event 的 `freed = before - after > 0`)。

---

## 3. 占位摘要 vs 真实摘要（诚实说明）

`src/compaction.py` 里的 `_summarize()` 是**确定性占位**——它不调 LLM, 只把若干消息替换成一条更短的标记串, 目的是让你**看清「信息被有损压缩」这件事本身**和 token 曲线。

> 真实 harness 在 Stage 3/4/5 会**调一次模型**做语义摘要 (把旧上下文喂给模型, 让它写一段浓缩)。接真 provider (L03) 后, 把 `_summarize` 换成一次 `provider.stream` 调用即可——**机制不变, 只是摘要质量从「占位」变「语义」**。这正是为什么我们先用 Mock 把机制跑通: 机制对了, 换真模型只是替换一个函数。

这是一处典型的「策展性诚实」: 我明确告诉你哪部分是占位、真实怎么做, 而不是假装占位摘要等于真摘要。

---

## 4. 一个常被忽视的点：compaction 会影响成本与正确性

- **成本**: 不压缩 → 每步喂给模型的上下文越来越大 → token 花费暴涨。压缩把它**摁在预算内**。N3 (harness eval) 会量化: 同任务, 开/关 compaction, `context_tokens` 差多少。
- **正确性**: 压缩是有损的。压错了 (把关键信息当冗余丢了) → 模型「失忆」→ 任务失败。**什么该 pin、什么该压、压多狠, 是 harness engineering 的核心手艺, 也是 L13 的研究 gap 之一** (context folding 的理论目前并不成熟)。

---

## 5. 本讲小结 + 通往 L05

- 上下文是预算不是仓库; 满了要渐进式压缩, 不是一刀切。
- 5 阶段 (budget→snip→microcompact→collapse→auto-compact), 逐级加重, 轻手段优先。
- 不变量: system/pinned 不丢、最近受保护、单调减 token。
- 占位摘要演示机制, 接真模型时换成语义摘要。

> **下一讲 L05**: compaction 是「在一个窗口内腾地方」。但有些任务大到**一个窗口装不下、要跨多个上下文窗口连续干几小时**——这是 2026 公认最难的问题。`src/long_horizon.py` 的 **loop-with-hook + 文件系统 state** 怎么解, N2 notebook 怎么跑通一个跨窗口长任务。

**动手**: 打开 `notebooks/N1-compaction-in-action.ipynb`, 填满一个窗口, 看 5 阶段逐级触发 + token 曲线。
