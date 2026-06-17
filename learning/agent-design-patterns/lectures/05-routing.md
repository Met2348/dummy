# L05 · Workflow 模式 2 — Routing

## 形状

先用一次调用把输入**分类**,再分流到**专门化的 handler**。每个 handler 可以有自己的 prompt、模型、工具,互不干扰。

```
                 ┌─→ [billing handler]
input → [classify] ─→ [technical handler] → output
                 └─→ [general handler]
```

## 何时用

- 输入天然落入**几个差异明显的类别**,分开处理各自更好。
- 想避免一个"什么都管"的巨型 prompt(既臃肿又互相干扰)。
- 例子:客服分流(账单/技术/通用);难易分流(简单问题用小模型,难题用大模型——省钱)。

## src 走读

[routing.py](../src/patterns/routing.py):

```python
def run_router(text, classifier, handlers, tracker=None):
    category = classifier(text)               # 1 次:分类
    handler = handlers.get(category) or handlers["default"]
    out = handler(text)                        # 1 次:专门处理
    return PatternResult("routing", out, trace, tracker, ok=category in handlers)
```

demo:把"登录坏了报错"分类到 `technical`,分流到 debug handler。

## 设计要点

1. **分类与处理解耦**:分类器可以很轻(甚至关键词/小模型),把贵的大模型留给真正需要的 handler。
2. **永远要有 default 分支**:分类不确定时,落到兜底,而不是崩。`_self_test` 验证了未知类别落 default 且被标记 `ok=False`(可观测)。
3. **类别要互斥且穷尽**:重叠类别 → 分类抖动;遗漏类别 → 全落 default。

## 成本视角(对照 capstone)

routing 在 capstone 里是**最省的**(2 次调用 vs agent 的 5 次):因为它"只做需要做的那一支",不像 chaining/sectioning 把所有步骤都跑一遍。

## 退出条件
- [ ] 理解"分类便宜、处理可贵"的省钱结构
- [ ] 知道为什么必须有 default 分支
- [ ] 能解释 routing 为何比全跑型模式省
