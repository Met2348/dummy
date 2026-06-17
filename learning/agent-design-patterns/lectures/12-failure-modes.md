# L12 · Agent 反模式与失败模式

## 为什么单列一节

agent 最危险的失败**不抛异常**——它们悄悄降低质量,日志一片绿,结果是错的。认识它们的**形状**是 agent 设计的一半功力。[failure_modes.py](../src/failure_modes.py) 把每个反模式做成"坏 demo + 修复"成对。

## 1. Runaway Loop(失控循环)

**症状**:agent 原地打转,反复做同一动作,烧光预算/时间。

```python
# 坏:无进度检查,跑到硬上限
runaway_without_guard(...) -> 撞 max_steps

# 修:检测"最近 N 步动作相同 = 无进度",提前停
loop_with_progress_guard(actions, no_progress_limit=2)
```

修复关键:不只设 `max_steps`,还要**检测无进度**(动作/状态不再变化)。

## 2. Context Rot(上下文腐烂)

**症状**:往 context 塞太多无关内容,信号被噪声稀释,模型表现下降。

```python
retrieve_naive("python bug", docs)        # 4 篇全塞 → 噪声
retrieve_relevant("python bug", docs, k=2) # 按相关性留 top-2 → 信号
```

修复:**少而精 > 多而杂**。检索/记忆都要排序截断,不是越多越好(呼应 L10)。

## 3. Tool Sprawl(工具泛滥)

**症状**:给 agent 几十个名字相近的工具,它选错。

```python
pick_tool("please email the team", tools)         # 抓到 'search' → 错!
pick_tool_curated(query, tools, aliases={...})    # 按意图别名 → 'send_email'
```

修复:**精选小工具集 + 意图别名**;宁可返回"无合适工具"也别硬选一个错的。

## 4. Silent Failure(静默失败)

**症状**:catch-all 吞掉工具错误,返回假值,agent 在垃圾上继续推理。

```python
run_tool_swallowing(boom)   # 返回 ""  ← 模型毫不知情
run_tool_surfacing(boom)    # 返回 {"ok": False, "error": "..."} ← 看得见
```

修复:**错误要 surface 成结构化结果**,让 agent 和 trace 都能看见并反应。这是整个 PR 评审里最常抓的一类问题。

## 其它要警惕的

| 反模式 | 一句话 |
|--------|--------|
| Over-engineering | 单次调用能解决的事上了 agent(违反 L02) |
| 错误累积 | 长链/长循环里每步小误差滚雪球 |
| Sycophantic eval | 模型自评只会自我确认(L08 换视角) |
| 无观测 | 出了问题没 trace,无法复盘(Topic 9) |

## 退出条件
- [ ] 能说出 4 大静默失败的症状与修复
- [ ] 理解"少而精 > 多而杂"贯穿检索/工具/context
- [ ] 记住:错误要 surface,不要 swallow
