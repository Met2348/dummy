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

## 运行验证（Runbook）

入口命令清单见 [`runbook.yaml`](runbook.yaml)。一键验证全部 8 个文档入口（V0 静态 + V1 smoke 直跑）：

```powershell
python scripts/eric_3080ti_env_audit.py --runbook --modules agent-code-eval `
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json `
  --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

预期：`8/8 pass`。全部脚本无 argparse → 无参直跑（`v0: false`）；纯 CPU 手写 mock + 真 `exec()` 评测，秒级完成，无需 GPU/下载/重型栈。

**可跑入口**（均 repo-root 相对，直接 `python ...` 即可）：

| 脚本 | 跑完打印什么 |
|------|------|
| `src/common.py` | 沙箱 `safe_exec` demo：correct→PASS / buggy→AssertionError / forbidden→BLOCKED |
| `src/humaneval_runner.py` | HumanEval：empty pass@1=0.00 vs oracle=1.00 + pass@k 估计量（真 exec，非硬编码） |
| `src/mbpp_runner.py` | MBPP：empty pass@1=0.00 vs oracle=1.00（真 exec 对拍测试用例） |
| `src/livecodebench_mock.py` | LiveCodeBench：3 算法题 empty 0.00 vs oracle 1.00（逐题真 exec） |
| `src/swebench_mock.py` | SWE-Bench：提交 buggy→fail / fixed→pass（真跑 hidden ValueError 测试） |
| `src/webarena_mock.py` | WebArena：无动作→fail / 4 步脚本→pass（真状态机，打印终态 cart/confirmed） |
| `src/bfcl_runner.py` | BFCL：correct-JSON acc=1.00 / wrong-city→fail（真 JSON 解析 + 选函数&参数判分） |
| `src/mini_agent.py` | Capstone：empty 全 0.00 / smart 全 1.00 的 5-bench 评分表 |

每个脚本先跑 `_self_test()`（empty→0 / oracle→1 守卫），再 `_demo()` 打印**真实逐样本评分**。

**关键坑注记**：
- 这些 runner 是**真评测器**，不是返回硬编码分数：`humaneval`/`mbpp`/`livecodebench`/`swebench` 把候选代码 `common.safe_exec` 真 `exec()` 后对拍 hidden tests（受限 `__builtins__` + 禁止模式黑名单），empty 模型恒 0、oracle 模型恒 1 即证。
- `safe_exec` 是**教学级沙箱**：黑名单挡 `import os`/`open`/`eval` 等 + 受限内建命名空间；**不做** AST 解析、子进程隔离、或真实超时（`timeout` 形参目前未强制）——docstring 已诚实标注"Real sandbox needs AST + subprocess"。仅供本地 mock 候选代码用，**勿**拿去跑不可信代码。
- `swebench_mock`/`webarena_mock`/`livecodebench_mock` 三个 mock 是**机制驱动**的：SWE 由"修复后文件能否过 hidden test"判定、WebArena 由"动作序列驱动状态机后的终态"判定、LCB 真 exec——展示的通过/失败由展示的输入真实推出，无魔法常数。
- `mini_agent.py` 是 5-bench **评测聚合器**（每 bench 单次 generate→score），**不是**迭代式 read→生成→执行→反馈→改 的 agent 闭环；docstring/`_demo()` 已注明。

## 测试（V2）

```powershell
python learning/agent-code-eval/src/tests/test_agent.py
```

预期：`8/8 modules passed`（逐模块跑 `_self_test()`）。

> 注：该测试是脚本式 runner（非 `test_*` 函数），用 `python <file>` 直跑才执行断言；`pytest` 收集到 0 个用例（"no tests ran"，exit 0）。

## 跑 capstone（单独）

```powershell
python learning/agent-code-eval/src/mini_agent.py
```

打印 mock 模型 × 5 bench 对照表（empty 全 0.00、smart 全 1.00）。

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
