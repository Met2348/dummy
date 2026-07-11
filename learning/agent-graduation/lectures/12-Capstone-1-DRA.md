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
$env:PYTHONIOENCODING="utf-8"; python learning/agent-graduation/src/dra/orchestrator.py
```

（等价于讲义旧版 `python -c "sys.path.insert(...); from dra.orchestrator import run_capstone_1, to_md; print(to_md(run_capstone_1()))"` 一行流，但不依赖 CWD——脚本 `__main__` 自带 `_self_test()` + `print(to_md(run_capstone_1()))`。经审计 harness 用 `PYTHONPATH=learning/agent-graduation/src` 隔离运行验证：`python scripts/eric_3080ti_env_audit.py --runbook --modules agent-graduation`。）

## 预期输出（实测，2026-07-12 ERIC-3080Ti）

````
[OK] dra.orchestrator._self_test passed (5 cites, 10 supported)

# Deep Research Agent - Capstone-1

**Query:** Write a brief report on modern LLM inference optimization techniques.

## Plan
- 5 sub-questions
- Rationale: Decomposed into 5 sub-questions on 'llm inference'.

### Sub-questions
1. What are major LLM inference engines used in modern serving stacks?
2. How do PagedAttention and RadixAttention compare?
3. What is speculative decoding and how much speedup does it give?
4. What quantization methods are most popular?
5. How does FlashAttention v3 affect inference?

## Retrieval: 10 doc snippets across 5 sub-q

## Draft preview (first 400 chars)
```
# Report: Write a brief report on modern LLM inference optimization techniques.
...
```
- Citations: 5
- Claims: 10

## Verification
- Supported: 10
- Unsupported: 0

## Cost
- LLM calls: 3
- Tool calls: 16
- Tokens in: 334
- Tokens out: 959
- ~cost_usd: 0.095387

## Verdict: [PASS]
````

> 注：本讲义早期版本的"预期输出"是撰写时的设计草图（`## Notes`/`## Final report` 等小节名），与 `to_md()` 实际输出的小节名（`## Retrieval`/`## Draft preview`/`## Verification`/`## Cost`）不完全一致；上面已替换为脚本真实 stdout。退出条件里的"5 步 plan / 至少 3 citation / 100% claims 都有 source"三条在实测里都成立（5 sub-questions、5 citations、10/10 supported）。

## 一句话

> Capstone-1 DRA = planner + retriever + writer + verifier 4 模块跑通 + markdown 报告 + cite 全 supported。
