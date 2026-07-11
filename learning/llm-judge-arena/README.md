# Topic 4: LLM-Judge / Arena（裁判 + Arena 评测）

> Module 6「评」第 4 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | LLM-as-Judge 4 类 | `common.py` |
| L02 | MT-Bench | `mt_bench_runner.py` |
| L03 | Arena-Hard | `arena_hard_runner.py` |
| L04 | Chatbot Arena Elo | `bradley_terry.py` |
| L05 | AlpacaEval 2 LC | `alpaca_eval.py` |
| L06 | Prometheus 2 | `prometheus2_judge.py` |
| L07 | G-Eval | (lecture only) |
| L08 | Judge 4 大 bias | `judge_bias_demo.py` |
| L09 | JudgeBench | (lecture only) |
| L10 | Pairwise vs Pointwise | (lecture only) |
| L11 | 成本工程 | (lecture only) |
| L12 | **Capstone: mini-Arena 5 ckpt** | `mini_arena.py` |

## Tag

- `llm-judge-arena` — Topic 4 完结

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V1 验证通过（**纯 stdlib，无需改代码**）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules llm-judge-arena
> ```

7 个 lecture 直跑 demo（均无需传参，纯 stdlib 数值/字符串计算，CPU 秒级出结果）：

```powershell
python learning/llm-judge-arena/src/common.py             # L01 judge 工具箱（length/keyword/position-bias）
python learning/llm-judge-arena/src/mt_bench_runner.py     # L02 MT-Bench toy pointwise
python learning/llm-judge-arena/src/arena_hard_runner.py   # L03 Arena-Hard toy pairwise + win rate
python learning/llm-judge-arena/src/bradley_terry.py       # L04 Chatbot Arena BT 拟合 + Elo
python learning/llm-judge-arena/src/alpaca_eval.py         # L05 AlpacaEval 2 length-controlled win rate
python learning/llm-judge-arena/src/prometheus2_judge.py   # L06 Prometheus 2 rubric 打分 mock
python learning/llm-judge-arena/src/judge_bias_demo.py     # L08 4 大 judge bias 复现
```

**Capstone：mini-Arena（5 mock 模型 round-robin battle → BT 拟合 → Elo 排行榜）**：

```powershell
python learning/llm-judge-arena/src/mini_arena.py
```

> 注（demo 性质，非 bug）：8 个脚本全部无 argparse（runbook 内标 `v0: false`，跳过 `--help` 探针，直接 smoke 直跑到完成）；均为 stdlib-only toy 实现，未接真实 judge API（`environment/requirements.txt` 里的 openai/anthropic/torch/transformers 仅真实 judge 场景才需要，`verify_env.py` 缺失时只 `[INFO]` 不阻塞）。`mini_arena.py` 的 Elo 数值会很极端，因为 toy 数据接近完全可分——真实系统需要更多噪声、更多 votes 和置信区间。

**测试（V2）**：`test_judge.py` 聚合跑 8 个模块的 `_self_test()`：

```powershell
python learning/llm-judge-arena/src/tests/test_judge.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules llm-judge-arena --tests
```

预期：`8/8 modules passed`。（`test_judge.py` 无 pytest `test_*` 函数，`pytest` 会 collect-0 → 审计 harness 自动回退直接跑脚本。）

## 关键文献

- Zheng et al. 2023 MT-Bench / LLM-as-Judge (LMSYS)
- Li et al. 2024 Arena-Hard (LMSYS)
- Chiang et al. 2024 Chatbot Arena
- Dubois et al. 2024 AlpacaEval 2 LC
- Kim et al. 2024 Prometheus 2 (KAIST)
- Liu et al. 2023 G-Eval (Microsoft)
- Tan et al. 2024 JudgeBench
- Bradley & Terry 1952 BT-model

## 一句话

> Judge = 自动评开放生成；Arena = pairwise BT-Elo 排行。
