# L08 · Workflow 模式 5 — Evaluator-Optimizer

## 形状

一个 generator 出草稿,一个 evaluator 打分并给**具体反馈**,generator 据此修改。循环到通过或耗尽预算。

```
task → [generate] → [evaluate] ──pass──→ output
          ↑              │
          └──feedback────┘ (fail)
```

## 何时用

两个条件**同时**满足才值得:
1. 有**清晰的评价标准**(能机器判 / 能让 LLM 稳定判);
2. **迭代确实能改进**(不是一次到位的任务)。

经典例子:翻译(评估流畅度/忠实度再改)、写必须过测试的代码、结构化抽取(字段补全)。

## src 走读

[evaluator_optimizer.py](../src/patterns/evaluator_optimizer.py):

```python
def run_eval_opt(task, generator, evaluator, max_iters=4, tracker=None):
    feedback, output = "", ""
    for i in range(max_iters):
        output = generator(task, feedback)     # 带着上轮反馈重生成
        ok, fb = evaluator(output)
        if ok:
            return PatternResult(..., ok=True)
        feedback = fb
    return PatternResult(..., ok=False)         # 预算耗尽
```

demo:生成 bug 工单,evaluator 检查 `title/priority/labels` 是否齐全,generator 每轮补一个缺失字段——3 轮收敛,trace 里能看到 3 次 generate + 3 次 evaluate 交替。

## 设计要点

1. **反馈要具体、可执行**:"不够好"没用;"缺 priority 字段"才能让 generator 知道改哪。反馈粒度直接决定收敛轮数。
2. **必须有预算上限** `max_iters`:否则评估永不通过时会无限循环(`_self_test` 验证了耗尽预算返回 `ok=False`)。
3. **evaluator 和 generator 最好是不同视角**:同一个模型自评容易自我确认(sycophancy)。可换 prompt、换模型,甚至用程序化检查(过测试、schema)。

## 和 Autonomous Agent 的区别

evaluator-optimizer 的循环是**结构固定的**(永远 generate→evaluate),只是次数可变。agent 的循环里**每一步做什么都由模型自由决定**。前者是"受控迭代",后者是"开放探索"。

## 退出条件
- [ ] 记住"清晰标准 + 迭代有用"两个前提
- [ ] 理解反馈粒度 ↔ 收敛轮数
- [ ] 知道为什么 evaluator 该换视角
