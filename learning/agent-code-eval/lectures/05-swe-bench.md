# L05 · SWE-Bench / SWE-Bench Verified

**Jimenez et al. 2023** · Princeton

## 数据

- **2294 真实 GitHub issue**（django/sklearn/sympy/flask/pytest/...）
- 每题：issue 描述 + 当时 codebase 快照 + 修复 commit + 测试 patch
- 评判：**模型修改 → 运行隐藏测试是否过**

## 例题

```
Issue #11815 (Django):
"Models with class Meta: ordering = ['-pk'] are sorted wrongly when
combined with __or."

Files to potentially modify:
- django/db/models/sql/compiler.py
- django/db/models/sql/query.py

Hidden tests: 5 unit tests added in fix commit
```

模型必须 navigate 1000-file 代码库，找 root cause，写 patch。

## SWE-Bench Verified (OpenAI 2024.08)

OpenAI 人工 audit → 挑出 500 题"标注质量高"的子集。
现在大家报的都是 Verified。

## 评测协议

1. 提供：repo URL + issue + 当时 commit hash
2. agent：clone + 探索 + 改文件
3. 跑：repo 自带的 hidden test suite
4. 评：是否全过

## 分数

| Agent | SWE-Bench Verified |
|-------|---------------------|
| GPT-4 单调用 | <2% |
| AutoCodeRover (2024) | 19% |
| SWE-Agent | 22.7% |
| Aider | 29% |
| **Claude 3.5 Sonnet + Cogtor** | **49%** |
| **Claude 3.7 Sonnet** | **50%+** |
| OpenHands (2025) | 53% |
| 人类工程师 | ~85% |

## 难在哪

1. **navigate 大 repo**：1000+ files，工具调用为主
2. **root cause analysis**：issue 描述模糊
3. **多文件 patch**：改 A 才能让 B 通过
4. **隐藏测试不可见**：debug 全靠 stdout

## Agent 架构

```
Loop:
  observe (file listing / file content)
    ↓
  think (CoT reasoning)
    ↓
  act (read_file / write_file / run_test / shell)
    ↓
  goto observe
Until: test pass or max_steps
```

## 实操

src/swebench_mock.py 1 题（divide by zero）：

```python
from swebench_mock import run_swebench_mock, REPO_FILE_FIXED
from common import make_mock_model

m = make_mock_model({"swe_1": f"```python\n{REPO_FILE_FIXED}```"})
rs = run_swebench_mock(m)
print(rs[0]["passed"])  # True
```

## 一句话

> SWE-Bench = LLM 的"真实软件工程师面试"，2024 起是 agent benchmark 之王。
