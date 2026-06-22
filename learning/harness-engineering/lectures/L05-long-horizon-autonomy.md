# L05 · long-horizon 自治：loop-with-hook + 文件系统 state

> Part II · 40-min lecture · 配套代码 `src/long_horizon.py` · notebook `N2` · 目标: 让 agent 跨多个上下文窗口连续工作——2026 公认最难、最值钱的问题。

---

## 0. 为什么这是「圣杯」

短任务 (几步内完成) 现在的模型 + 基础 harness 就能干。难的是**长时程自治 (long-horizon autonomy)**: 一个任务大到要跨多个上下文窗口、连续工作几小时 (重构一个大型代码库、做一次完整研究)。今天的模型在长任务上有**三个病**:

```
① early stopping     没干完就想收工 ("我觉得差不多了")
② 分解差            复杂任务拆不好, 步骤遗漏/乱序
③ 跨窗口失忆        上下文一换 (compaction/新窗口), 前面做的就忘了
```

harness 必须**围绕这三个病设计**——因为短期内模型不会自己治好它们。

---

## 1. 核心洞察：把「真相」放到上下文之外

上下文窗口是**会变、会忘**的 (compaction 会压、换窗口会重置)。所以**不能把任务的真相只存在上下文里**。

> **关键设计: 文件系统是真相之源 (filesystem as source of truth)。**
> todo 清单、进度、笔记、中间产物——**落盘**。上下文会忘, 文件不会。每个新窗口**从干净状态起步**, 但开头**从文件系统读回 state**。

`src/long_horizon.py` 的 `FileStateStore`:

```python
store = FileStateStore("work/")        # state.json: {progress, notes, todo}
store.save({...}); store.load()
store.summary()   # "进度=3/6; 最近笔记: ..."  ← 给下一个窗口的"读回"摘要
```

这就把「跨窗口记忆」从「指望模型记住」变成「从磁盘读回」——一个**确定性、不会失忆**的机制。

---

## 2. loop-with-hook：拦截 early-stop，强制续跑

光有文件系统还不够——模型会 **early-stop** (没干完就发 `stop`)。解法是一个 **hook**:

```
        ┌──────────────────────────────────────────────┐
        │  窗口 w: 从 state 读回 → 干几步 → 模型发 stop    │
        └───────────────────┬──────────────────────────┘
                            │ 模型想收工
                   ┌────────▼─────────┐
                   │ hook: goal 达成?  │
                   └───┬──────────┬───┘
                  达成 │          │ 未达成
                ┌──────▼──┐   ┌───▼────────────────────────┐
                │ 真正结束 │   │ 拦截! 不让停。               │
                │ success │   │ 存 state → 开新窗口 w+1       │
                └─────────┘   │ → 从 state 读回 → 继续        │
                              └────────────────────────────┘
              每个窗口干净起步, 靠文件系统接力, 直到 completion-goal 真达成
```

对应 `run_long_horizon(...)` 的逻辑:

```python
res = run_long_horizon(provider, goal, tools, store, goal_met,
                       compactor=Compactor(600), max_windows=6, hook=True)
# 模型发 stop 时: goal_met(state)? 是→success; 否且 hook→换窗续跑(stop_intercepted)
```

- **每个窗口** `_seed_messages()` 重建干净上下文: `system + 目标 + store.summary()`。
- 模型发 `stop` 且 `goal_met` 为假 → **hook 拦截**, 标记 `stop_intercepted`, 跳到下一窗口。
- `completion-goal` 由 `goal_met(state)` 判定 (读文件系统里的进度), 而不是「问模型做完没」——**判定权在 harness, 不在模型**。

---

## 3. 看它真的救回一个 early-stop（N2 的剧本）

`demo_setup()` 故意构造一个「6 步任务, 但模型在第 2 步就错误地收工」的确定性场景:

```
窗口0: do_step(1) → do_step(2) → 模型发 stop  (early-stopping 病!)
       hook: goal_met? progress=2 < 6 → 拦截! 存盘, 换窗口
窗口1: 读回 progress=2 → do_step(3,4,5,6) → 第6步 stop
       hook: goal_met? progress=6 == 6 → 真正完成 ✅
结果: success=True, 2 个窗口, 6 步, 有 1 次 stop_intercepted
```

对照实验 (`hook=False`, N3 会跑): 模型第 2 步收工就**直接放它走** → 任务**失败** (`aborted_early=True`)。

> 这就把一个抽象论断「hook 能救长任务」变成了**可运行、可测量**的对照。`src/tests/test_all.py` 里 `test_hook_rescues_early_stop` 和 `test_no_hook_fails_long_task` 把这两条都钉死成单测。

---

## 4. 与 compaction、subagent 的关系

- **compaction (L04) 管窗口内**: 一个窗口里上下文涨太大, 渐进压缩。
- **loop-with-hook (本讲) 管窗口间**: 一个窗口装不下, 换窗口、靠文件系统接力。
- 两者**叠加**: 窗口内压缩到压不动了 → 触发换窗 → 新窗口从 state 读回。这正是 Stage 5 auto-compact 和换窗的衔接。
- **subagent (L06)**: 另一种「给上下文减负」的方式——把一个子任务**整个外包**给一个隔离的子 agent, 它自己的脏上下文不污染主线, 只把结论带回。

---

## 5. 这一讲里藏着的研究 gap（预告 L13）

long-horizon 是研究的富矿, 因为现状远未解决:
- **何时该换窗、压缩到什么程度才不丢关键信息**——目前是工程直觉, 缺理论 (context folding 的形式化)。
- **completion-goal 怎么定义才既不 early-stop 也不 over-run**——可验证任务 (有标准答案) 好办, 开放任务很难。
- **跨窗口的 state 表示**——文件系统是当前最优工程解, 但什么是「最小充分 state」是开放问题。

> L13 会用你 `critical-reading-gap` 的 6 类 gap 雷达, 把这些系统地变成候选研究题目。

---

## 6. 本讲小结 + 通往 L06

- long-horizon 三病: early-stopping / 分解差 / 跨窗口失忆。
- 核心解: **文件系统是真相之源** + **loop-with-hook 拦截 early-stop 强制续跑**。
- 判定权在 harness (`goal_met` 读文件系统), 不在模型。
- 窗口内 compaction + 窗口间 hook 叠加; subagent 是第三种减负方式。

> **下一讲 L06**: subagent。不只是「并行干活」, 更是一个 **context firewall (上下文防火墙)**——把子任务的脏上下文隔离, 重建权限, 只把结论带回主线。以及多 agent 的 **debate 模式**为什么比单 agent 更可靠。

**动手**: 打开 `notebooks/N2-long-horizon-task.ipynb`, 看 loop-with-hook 跨 2 个窗口救回一个 early-stop 的长任务。
