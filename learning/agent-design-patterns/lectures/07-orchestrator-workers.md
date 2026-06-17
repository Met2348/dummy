# L07 · Workflow 模式 4 — Orchestrator-Workers

## 形状

像 sectioning,但**子任务事先不知道**:一个 orchestrator(LLM)**动态拆解**输入,为每个子任务派一个 worker,最后 synthesize。

```
input → [orchestrator: 动态拆出 N 个子任务]
            ├→ [worker 1] ┐
            ├→ [worker 2] ├→ [synthesize] → output
            └→ [worker N] ┘     (N 在运行时才确定)
```

## 与 Sectioning 的关键差异

| | Sectioning | Orchestrator-Workers |
|---|-----------|---------------------|
| 子任务集 | **写死**(代码里列好) | **运行时由 LLM 决定** |
| 子任务数 | 固定 | 数据驱动,可变 |
| 适用 | 你知道有哪几面 | 你**不知道**会拆出几块 |

经典例子:"修改所有 import 了 X 的文件"——改几个文件取决于代码库,事先数不出来。

## src 走读

[orchestrator_workers.py](../src/patterns/orchestrator_workers.py):

```python
def run_orchestrator(text, planner, worker, synthesize, tracker=None):
    subtasks = planner(text)              # ← 动态!数量取决于输入
    trace.add("plan", "decompose", f"{len(subtasks)} subtasks")
    results = [(st, worker(st)) for st in subtasks]
    return PatternResult(..., synthesize(text, results), ...)
```

demo 的 planner 扫描请求里提到的侧面("pricing/competitors/market"),**只拆出被提到的那几个**——`_self_test` 验证了它没有凭空造出没提的 "risks"。换个输入,子任务数就变。

## 设计要点

1. **planner 是这模式的大脑**:它的拆解质量决定一切。要给它清楚的拆解标准,否则会拆得过细(成本爆)或过粗(漏覆盖)。
2. **worker 应无状态、可并发**:每个 worker 只管自己那块,彼此不依赖。
3. **synthesize 不是简单拼接**:常需要去重、消解冲突、按重要性排序。

## 边界:它快变成 agent 了

orchestrator-workers 已经有"LLM 决定做什么"的味道,但它仍是 **workflow**——planner 拆完一次就执行,不回头。一旦 orchestrator 要根据 worker 结果**反复重新规划**,你就跨进了 autonomous agent(L09)。

## 退出条件
- [ ] 能说清它和 sectioning 的唯一关键差异(动态 vs 固定)
- [ ] 理解 planner 质量是瓶颈
- [ ] 知道"反复重规划"是它升级成 agent 的临界点
