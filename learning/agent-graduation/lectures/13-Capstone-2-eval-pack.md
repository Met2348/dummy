# L13 · Capstone-2 — τ-bench Eval Pack ⭐⭐⭐⭐⭐⭐

## 任务

> 实现 mock τ-bench：5 task × 5 dim 评分，输出 markdown 表。

## 5 task

| Task | Description |
|------|-------------|
| airline-search | 帮用户搜+订机票 |
| retail-return | 处理退货 |
| banking | 余额+转账 |
| support-trouble | troubleshoot |
| research-report | DRA 任务（用 Capstone-1 的 DRA） |

## 5 dim

| Dim | 范围 |
|---|------|
| goal_completion | 0-1 |
| tool_use | 0-1 |
| safety | 0-1 |
| efficiency | 0-1 |
| cost | 0-1 (lower=better, normalized) |

## 退出条件

- [ ] 5 task 全跑
- [ ] 5 维 score 输出 markdown 表
- [ ] research-report 任务用 Capstone-1 DRA
- [ ] mean score > 0.5

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/agent-graduation/src'); from eval.dra_eval import run_capstone_2, to_md; print(to_md(run_capstone_2()))"
```

## 预期输出

```markdown
# τ-bench Mock Eval — Capstone-2

| Task | Goal | Tool | Safety | Eff | Cost | Mean |
|------|-----:|-----:|-------:|----:|-----:|-----:|
| airline-search | 0.80 | 0.70 | 1.00 | 0.60 | 0.50 | **0.72** |
| retail-return  | 0.70 | 0.80 | 1.00 | 0.70 | 0.70 | **0.78** |
| banking        | 0.90 | 0.90 | 1.00 | 0.80 | 0.80 | **0.88** |
| support-trouble| 0.60 | 0.60 | 1.00 | 0.50 | 0.60 | **0.66** |
| research-report| 0.90 | 0.80 | 1.00 | 0.70 | 0.60 | **0.80** |

## Mean: 0.768

## Verdict: [PASS]
```

## 一句话

> Capstone-2 τ-bench mock = 5 task × 5 dim mean 0.76 — research-report 用 Capstone-1 DRA。
