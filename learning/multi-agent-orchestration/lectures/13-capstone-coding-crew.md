# L13 · Capstone — 3-Agent Coding Crew ⭐

## 任务

> 3 agent 协作完成一个编码 task：写 FizzBuzz + tests。

## 3 角色

| Agent | role | Output |
|-------|------|--------|
| PM | 分析需求 → 写 spec | spec.md |
| Engineer | 看 spec 写代码 | fizzbuzz.py |
| Reviewer | 审代码 + tests | review.md + pass/fail |

## 工作流（hierarchical）

```
User: "Write a FizzBuzz program with tests"
   ↓
PM → spec:
  - 输入: int 1-N
  - 规则: 3 → "Fizz", 5 → "Buzz", 15 → "FizzBuzz"
  - 输出 list
   ↓
Engineer → code:
  def fizzbuzz(n): ...
  def test_fizzbuzz(): assert fizzbuzz(15) == "FizzBuzz"
   ↓
Reviewer → review:
  - 检 5 example
  - 跑 test
  - verdict: PASS / FAIL with feedback
   ↓
(loop until PASS)
```

## 退出条件

- [ ] 3 agent 全部参与
- [ ] spec / code / review 三段输出
- [ ] reviewer 最终 verdict = PASS
- [ ] 5 个 example 全对

## 跑

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import sys; sys.path.insert(0,'learning/multi-agent-orchestration/src'); from capstone_coding_crew import run_capstone, to_md; print(to_md(run_capstone()))"
```

## 预期输出

```markdown
# 3-Agent Coding Crew Capstone

## Round 1
### PM spec
- Input: int 1-N
- Rules: 3→Fizz, 5→Buzz, 15→FizzBuzz
- Output: List

### Engineer code
def fizzbuzz(n): ...

### Reviewer
- Verdict: PASS
- Test count: 5 / 5

## Verdict: [PASS]
```

## 一句话

> 3 agent hierarchical crew (PM → Engineer → Reviewer) 协作完成 FizzBuzz — 范式可推广到任意 dev pipeline。
