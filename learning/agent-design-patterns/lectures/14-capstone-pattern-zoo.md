# L14 · Capstone — Pattern Zoo

## 目标

**一个任务,六种设计,一张对照表**。证明本专题的核心命题:

> 模型固定时,**架构本身**就是成本/可靠性杠杆。

## 任务

把一段自由文本需求 → 结构化工单 `{title, priority, labels, acceptance}`:

> "Users can't log in on mobile after the latest update — the login button does nothing. It's blocking many customers, needs a fix ASAP."

同一组确定性"抽取器"(`extract_title` / `infer_priority` / `derive_labels` / `write_acceptance`),用六种设计各做一遍。

## src 走读

[capstone/pattern_zoo.py](../src/capstone/pattern_zoo.py) 把六种模式都套在同一任务上:

```python
APPROACHES = [
    ("prompt_chaining",      via_chaining,     "ordered, predictable subtasks"),
    ("routing",              via_routing,      "distinct input categories"),
    ("parallelization",      via_sectioning,   "independent facets, want speed"),
    ("orchestrator_workers", via_orchestrator, "subtask set unknown up front"),
    ("evaluator_optimizer",  via_eval_opt,     "clear criteria + iteration helps"),
    ("autonomous_agent",     via_agent,        "open-ended, needs flexibility"),
]
```

## 运行

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/agent-design-patterns/src/capstone/pattern_zoo.py
```

## 输出对照表(实测)

| Design | LLM calls | tokens | $ | steps | ticket ok | best when |
|--------|----------:|-------:|--:|------:|:---------:|-----------|
| prompt_chaining | 4 | 18 | 0.00010 | 4 | PASS | ordered, predictable subtasks |
| routing | **2** | 11 | **0.00006** | 2 | PASS | distinct input categories |
| parallelization | 4 | 18 | 0.00010 | 5 | PASS | independent facets, want speed |
| orchestrator_workers | 5 | 17 | 0.00011 | 6 | PASS | subtask set unknown up front |
| evaluator_optimizer | 4 | 24 | 0.00012 | 8 | PASS | clear criteria + iteration helps |
| autonomous_agent | **5** | 25 | **0.00014** | 5 | PASS | open-ended (costliest) |

## 读这张表

- **六种都能产出合格工单**——但代价天差地别。
- routing 最省(**2 次调用**):只走需要的那一支。
- autonomous_agent 最贵(**5 次调用**):每步都要先"想一下"再做。
- 成本极差 **2.3×**(0.00014 / 0.00006),全因架构不同,**模型一字没换**。

## 毕业要义

> 拿到一个 agent 需求,先爬 L13 的决策树,**从最省的设计开始**,被任务逼着才往下走。这就是 agent design 的全部纪律。

## 退出条件
- [ ] 跑通 capstone,看到六行对照表
- [ ] 能解释成本差异来自哪
- [ ] 把"从最便宜的设计开始"内化为默认习惯
