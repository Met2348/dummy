# L04 · Workflow 模式 1 — Prompt Chaining

## 形状

把任务拆成**固定顺序**的几次 LLM 调用,每次吃上一步的输出。步与步之间可插入**程序化 gate**(纯代码检查),不合格就 fail-fast 或分支。

```
input → [LLM: 抽提纲] ─(gate: ≥2 要点?)─→ [LLM: 扩写] → [LLM: 润色] → output
                          │ 不合格
                          └──→ 提前中止(省下后续调用)
```

## 何时用

- 任务能**干净地拆成有序子任务**。
- 你愿意用一点延迟,换每步更高的准确率(每步 prompt 更聚焦)。
- 经典例子:生成营销文案→翻译→本地化校验;先列大纲→再扩写。

## src 走读

[prompt_chaining.py](../src/patterns/prompt_chaining.py) 的核心就是一个循环:

```python
def run_chain(text, steps, gates=None, tracker=None):
    cur = text
    for name, fn in steps:
        cur = fn(cur)                      # 一次 LLM 调用
        trace.add("llm", name, preview(cur))
        if name in gates:
            ok, reason = gates[name](cur)  # 程序化 gate
            if not ok:
                return PatternResult(..., ok=False)  # fail-fast
    return PatternResult(..., ok=True)
```

demo:`一个生产力小工具` → 提纲 → 扩写 → 润色,提纲后有个"≥2 要点"的 gate。

## 设计要点

1. **gate 是纯代码,不是 LLM**——长度检查、JSON schema 校验、关键字必含。便宜、确定、可单测。
2. **fail-fast 省钱**:gate 不过就别往下调。`_self_test` 里验证了"gate 失败时只发生 1 次 LLM 调用"。
3. **链不要太长**:每多一环,累积失败率上升(0.9⁵≈0.59)。超过 4-5 环考虑换 orchestrator。

## 和其它模式的边界

| 对比 | 区别 |
|------|------|
| vs Routing | chaining 是"全做且按顺序";routing 是"二选一分流" |
| vs Orchestrator | chaining 步骤**写死**;orchestrator 步骤**运行时才定** |

## 退出条件
- [ ] 能说清 gate 为什么必须是程序化的
- [ ] 理解 fail-fast 的省钱逻辑
- [ ] 知道链过长的累积失败率问题
