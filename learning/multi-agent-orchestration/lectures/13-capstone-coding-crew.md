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
python learning/multi-agent-orchestration/src/capstone_coding_crew.py
```

> 直接跑脚本即可（`if __name__ == "__main__"` 已经先跑 `_self_test()` 再打印完整报告）；
> 不需要 `python -c "...sys.path.insert..."` 这种依赖"当前目录=repo-root"的一行流写法。

## 预期输出

先打印 self-test 断言通过，再打印完整 markdown 报告（真实 stdout，非手写示例）：

````markdown
[OK] capstone_coding_crew._self_test passed (3 agents, 5/5 tests)

# 3-Agent Coding Crew Capstone

## PM spec
```
Spec for: Implement FizzBuzz with tests
- Inputs: int n (1..N)
- Rules:
  - i % 3 == 0 -> 'Fizz'
  - i % 5 == 0 -> 'Buzz'
  - i % 15 == 0 -> 'FizzBuzz'
  - else -> str(i)
- Output: list[str]
- Edge cases: n=0 returns [], n<0 returns []
```

## Engineer code
```python
def fizzbuzz(n):
    out = []
    for i in range(1, n+1):
        if i % 15 == 0: out.append('FizzBuzz')
        elif i % 3 == 0: out.append('Fizz')
        elif i % 5 == 0: out.append('Buzz')
        else: out.append(str(i))
    return out
```

## Reviewer
Reviewer findings:
- fb(3): PASS
- fb(5): PASS
- fb(15) ends FizzBuzz: PASS
- fb(0): PASS
- fb(1): PASS

## Verdict: [PASS] (5/5 tests)
## Cost
- tokens_in: 680
- tokens_out: 400
- llm_calls: 3
- ~cost_usd: 0.00804
````

## 一句话

> 3 agent hierarchical crew (PM → Engineer → Reviewer) 协作完成 FizzBuzz — 范式可推广到任意 dev pipeline。
