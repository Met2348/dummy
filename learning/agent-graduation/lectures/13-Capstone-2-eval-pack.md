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
$env:PYTHONIOENCODING="utf-8"; python learning/agent-graduation/src/eval/dra_eval.py
```

（等价于讲义旧版 `python -c "sys.path.insert(...); from eval.dra_eval import run_capstone_2, to_md; print(to_md(run_capstone_2()))"` 一行流，但不依赖 CWD——脚本 `__main__` 自带 `_self_test()` + `print(to_md(run_capstone_2()))`。经审计 harness 用 `PYTHONPATH=learning/agent-graduation/src` 隔离运行验证：`python scripts/eric_3080ti_env_audit.py --runbook --modules agent-graduation`。）

## 预期输出（实测，2026-07-12 ERIC-3080Ti）

```
[OK] eval.dra_eval._self_test passed (overall mean 0.948)

# Tau-bench Mock Eval - Capstone-2

| Task | Goal | Tool | Safety | Eff | Cost | Mean |
|------|-----:|-----:|-------:|----:|-----:|-----:|
| airline-search  | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | **0.980** |
| retail-return   | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | **0.980** |
| banking         | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | **0.980** |
| support-trouble | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | **0.980** |
| research-report | 1.00 | 1.00 | 1.00 | 1.00 | 0.09 | **0.818** |

## Overall mean: 0.948

## Verdict: [PASS] (target overall mean > 0.5)
```

> 注：本讲义早期版本的分数表是撰写时的设计草图（数字未跑代码验证过）；`simulate_mock_agent()` 对 4 个非 research 任务总是完成全部 required actions + 1 个无害 extra action，是确定性(非随机)的"接近满分" mock baseline，所以四行分数稳定在 0.98；`research-report` 走真实 Capstone-1 DRA（cost_usd≈0.095 > $0.05 目标 → cost 维仅 0.09）拉低到 0.818。上面已替换为脚本真实 stdout，退出条件"mean score > 0.5"实测 0.948 达标。

## 一句话

> Capstone-2 τ-bench mock = 5 task × 5 dim mean 0.76 — research-report 用 Capstone-1 DRA。
