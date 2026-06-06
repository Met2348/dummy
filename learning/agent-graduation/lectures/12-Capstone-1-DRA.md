# L12 · Capstone-1 — Deep Research Agent ⭐⭐⭐⭐⭐⭐

## 任务

> 从零造一个 DRA, 跑同一个 query 生成 markdown 报告。

## Query

```
"Write a brief report on 2026 LLM inference optimization techniques."
```

## DRA 流程（4 模块）

```
1. Planner: 5 sub-questions
2. Retriever: per sub-q → search + fetch
3. Writer: 综合成 markdown
4. Verifier: 检查 claim → source
```

## 退出条件

- [ ] 4 模块全跑通
- [ ] 5 步 plan
- [ ] 至少 3 citation
- [ ] markdown 报告 ≥ 300 字
- [ ] 100% claims 都有 source

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/agent-graduation/src'); from dra.orchestrator import run_capstone_1, to_md; print(to_md(run_capstone_1()))"
```

## 预期输出

```markdown
# Deep Research Agent Capstone-1

## Plan (5 sub-questions)
1. What are major 2026 LLM inference frameworks?
2. ...
5. ...

## Notes (5 sub-q × 3 docs)
[table]

## Draft (markdown report)
[5-paragraph report with [1]-[N]]

## Verification
- 12 claims, 12 supported, 0 unsupported
- 5 citations valid

## Final report
[final md]

## Cost
- LLM calls: 7
- Tool calls: 15
- ~cost_usd: 0.012

## Verdict: [PASS]
```

## 一句话

> Capstone-1 DRA = planner + retriever + writer + verifier 4 模块跑通 + markdown 报告 + cite 全 supported。
