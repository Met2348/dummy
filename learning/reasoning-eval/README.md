# Topic 2: Reasoning Eval（推理 benchmark 深化）

> Module 6「评」第 2 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 推理 bench 全景 | `common.py` |
| L02 | GSM8K | `gsm8k_runner.py` |
| L03 | MATH | `math_runner.py` |
| L04 | AIME 2024/2025 | `aime_runner.py` |
| L05 | math-verify / sympy | `math_verify_demo.py` |
| L06 | GPQA Diamond | `gpqa_runner.py` |
| L07 | Humanity's Last Exam | `hle_mock.py` |
| L08 | ARC-AGI | (lecture only) |
| L09 | ZebraLogic / BBH-logic | `zebra_logic.py` |
| L10 | 工具增强数学 | `tool_aug_math.py` |
| L11 | 推理评测陷阱 | (lecture only) |
| L12 | **Capstone: 5-bench 2-model 对照** | `capstone_reasoning_compare.py` |

## Tag

- `reasoning-eval` — Topic 2 完结

## 运行验证（Runbook）

入口命令清单见 [`runbook.yaml`](runbook.yaml)。一键验证全部 10 个文档入口（V0 静态 + V1 smoke 直跑）：

```powershell
python scripts/eric_3080ti_env_audit.py --runbook --modules reasoning-eval `
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json `
  --md-out docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

预期：`10/10 pass`。全部脚本无 argparse → 无参直跑（`v0: false`）；纯 CPU 手写 mock，秒级完成，无需 GPU/下载。

**可跑入口**（均 repo-root 相对，直接 `python ...` 即可）：

| 脚本 | 跑完打印什么 |
|------|------|
| `src/common.py` | 抽取/数值等价/pass@k 工具 demo |
| `src/gsm8k_runner.py` | GSM8K：dummy acc=0.00 vs oracle acc=1.00（真算，非硬编码） |
| `src/math_runner.py` | MATH：acc + 按难度 L1–L5 分桶 |
| `src/aime_runner.py` | AIME：整数精确匹配 acc + pass@k |
| `src/math_verify_demo.py` | 数学验证：`1/2`==`0.5`==`\frac{1}{2}`==`50%` 真等价判断 |
| `src/gpqa_runner.py` | GPQA：多选字母抽取 acc |
| `src/hle_mock.py` | HLE：短答 gold-substring 判分 acc |
| `src/zebra_logic.py` | ZebraLogic：约束谜题判分 acc |
| `src/tool_aug_math.py` | 工具增强 vs 纯 CoT：tool acc=1.00 vs CoT acc=0.00 |
| `src/capstone_reasoning_compare.py` | Capstone：2 模型 × 5 bench 对照表 |

每个脚本先跑 `_self_test()`（dummy→0 / oracle→1 守卫），再 `_demo()` 打印**真实算出的 acc / 验证结果**。

**关键坑注记**：
- 这些 runner 是**评测器**——`run_X(model)` 真做 predict→抽答案→比 gold→统计 acc（非占位分数；dummy 模型恒 0、oracle 模型恒 1 即证）。
- `zebra_logic.py` 是 benchmark **runner**（判模型给的答案），**不内嵌约束求解器**；谜题的"解"由 mock 模型提供，脚本只判分。
- `math_verify_demo.py` 是**教学版**验证器（regex + `Fraction`，**不依赖 sympy**）——覆盖 `a/b`/`%`/`\frac`/`\boxed` 数值等价，不做符号代数（`x+y==y+x`）；docstring 已诚实标注。
- `tool_aug_math.py` 真在受限沙箱里 `exec` 模型产出的代码（import/dunder/open 均拦截），演示 PoT 工具调用 vs 纯 CoT 的差异。

## 测试（V2）

```powershell
python learning/reasoning-eval/src/tests/test_reasoning.py
```

预期：`10/10 modules passed`（逐模块跑 `_self_test()`）。

> 注：该测试是脚本式 runner（非 `test_*` 函数），用 `python <file>` 直跑才执行断言；`pytest` 收集到 0 个用例（"no tests ran"，exit 0）。

## 跑 capstone（单独）

```powershell
python learning/reasoning-eval/src/capstone_reasoning_compare.py
```

打印 2 模型 × 5 bench 对照表（baseline 全 0.00、r1_tiny 全 1.00）。

## 关键文献

- Cobbe et al. 2021 GSM8K
- Hendrycks et al. 2021 MATH
- Rein et al. 2023 GPQA
- Scale AI + CAIS 2025 Humanity's Last Exam
- AIME 2024/2025 (USAMO)
- Chollet 2019 ARC; 2025 ARC-AGI-2
- DeepSeek-R1 paper (cons@64 / AIME)

## 一句话

> 推理 bench = 当今 LLM 的 IQ 测试 — 数学/科学/逻辑全覆盖。
