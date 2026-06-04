# Topic 3: Agent / Code Eval（Agent & 代码评测）

> Module 6「评」第 3 专题 · 12 lectures · 12 notebooks · ~13h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Agent 评测全景 | `common.py` |
| L02 | HumanEval + MBPP | `humaneval_runner.py`, `mbpp_runner.py` |
| L03 | BigCodeBench | (lecture only) |
| L04 | LiveCodeBench | `livecodebench_mock.py` |
| L05 | SWE-Bench Verified | `swebench_mock.py` |
| L06 | WebArena | `webarena_mock.py` |
| L07 | GAIA | (lecture only) |
| L08 | OSWorld | (lecture only) |
| L09 | BFCL Function Calling | `bfcl_runner.py` |
| L10 | MMMU/MathVista (VLM) | (lecture only) |
| L11 | Agent 评测陷阱 | (lecture only) |
| L12 | **Capstone: mini-agent 5-bench** | `mini_agent.py` |

## Tag

- `agent-code-eval` — Topic 3 完结

## 跑测试

```powershell
python learning/agent-code-eval/src/tests/test_agent.py
```

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/agent-code-eval/src'); from mini_agent import run_all, to_md; from common import make_mock_model; print(to_md(run_all(make_mock_model({}, default='')), 'empty'))"
```

## 关键文献

- Chen et al. 2021 HumanEval (OpenAI Codex)
- Austin et al. 2021 MBPP (Google)
- Zhuo et al. 2024 BigCodeBench
- Jain et al. 2024 LiveCodeBench
- Jimenez et al. 2023 SWE-Bench, SWE-Bench Verified (OpenAI 2024.08)
- Zhou et al. 2023 WebArena (CMU)
- Mialon et al. 2023 GAIA (Meta)
- Xie et al. 2024 OSWorld (THU)
- BFCL v3 (Berkeley 2025)
- Yue et al. 2024 MMMU

## 一句话

> Agent bench = LLM 的"实习考评"，测交付能力。
