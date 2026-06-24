# Topic 1: Eval Foundations（评测基础）

> Module 6「评」第 1 专题 · 12 lectures · 12 notebooks · ~12h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 评测 4 大范式 | `common.py` |
| L02 | MMLU | `mmlu_runner.py` |
| L03 | MMLU-Pro | `mmlu_pro_runner.py` |
| L04 | HELM | `helm_local.py` |
| L05 | Open LLM Leaderboard v2 | (lecture only) |
| L06 | BBH | `bbh_runner.py` |
| L07 | TruthfulQA | `truthfulqa_runner.py` |
| L08 | HellaSwag/ARC/Winogrande | `commonsense_runner.py` |
| L09 | lm-evaluation-harness | `lm_eval_adapter.py` |
| L10 | 污染检测 | `contamination_check.py` |
| L11 | 评测陷阱合集 | (lecture only) |
| L12 | **Capstone: 4-bench 联跑** | `eval_pipeline.py` |

## Tag

- `eval-foundations` — Topic 1 完结

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（10/10，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules eval-foundations
> ```

10 个脚本全是**手写 mock 评测 runner**（零外部依赖、纯 CPU）。每个直跑都会：① 跑内置
`_self_test`（含 oracle=100% / random∈[0,1] 真实断言）② 打印一个 random baseline demo。
直接 `python <脚本>` 即可（脚本无 argparse；harness 会自动把 `src/` 加进 `PYTHONPATH`，故 `from common import ...` 裸导入可解析）：

```powershell
# L01 共享后端（mock model + 答案抽取 + 聚合）
python learning/eval-foundations/src/common.py
# L02 MMLU（12 题 4 学科，真算逐题 acc + by_subject）
python learning/eval-foundations/src/mmlu_runner.py
# L03 MMLU-Pro（10 选项 A-J，chance~10%）
python learning/eval-foundations/src/mmlu_pro_runner.py
# L04 mini-HELM（4 scenario × metric 整体性表格）
python learning/eval-foundations/src/helm_local.py
# L06 BBH（tracking/date/logic 3 task）
python learning/eval-foundations/src/bbh_runner.py
# L07 TruthfulQA（MC1 抗常见误解；higher acc != truer）
python learning/eval-foundations/src/truthfulqa_runner.py
# L08 HellaSwag/ARC/Winogrande
python learning/eval-foundations/src/commonsense_runner.py
# L09 lm-eval-harness 风格 Task API 适配器
python learning/eval-foundations/src/lm_eval_adapter.py
# L10 污染检测（13-gram 重叠 + canary + Min-K%++ 草图）
python learning/eval-foundations/src/contamination_check.py
```

**Capstone（L12）：4-bench 联跑 → markdown 报告 + ASCII 条**

```powershell
python learning/eval-foundations/src/eval_pipeline.py
```

> 直跑即输出 random baseline 报告（mmlu/bbh/truthfulqa/commonsense 各一行 + Overall average）。
> 想换成 oracle（全 100%）或真模型：见 `lectures/12-capstone-eval-pipeline.md`——
> `run_all_benches` 接收任意 `ModelFn = Callable[[str, int], str]`，把 mock 换成 HF `generate` 闭包即可。

**关键坑注记**

- **mock 是真算的，不是硬编码分数**：每个 runner 都是 `predict → 抽取字母 → 比对 gold → 统计 acc`；
  oracle 必为 100%、random 在 chance 附近（self-test 已断言）。污染检测真算 n-gram 重叠率
  （干净语料 0.000、泄漏语料 1.000、canary 命中 flagged=True）。
- **TruthfulQA**：gold 标的是"抵抗常见误解"的选项（如口香糖正常排出 / 太阳从太空看是白色）；
  分数高 ≠ 更真，要看错误模式（demo 打印 `by_category`）。
- 真模型评测是**可选**路径（需 HF 权重 + GPU），非默认；默认全是秒级 mock，方便教学/CI。

**测试（V2）**

```powershell
python learning/eval-foundations/src/tests/test_eval.py    # 预期：10/10 modules passed
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules eval-foundations --tests
```

> 注：`test_eval.py` 是脚本式聚合器（汇总 10 个模块的 `_self_test`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑。

## 与 Module 6 关系

```
Topic 1 (本) → 2 reasoning → 3 agent-code → 4 judge-arena
                                                ↓
                       5 red-team → 6 safety-defense
                                                ↓
                            7 eval-graduation (毕业)
```

## 与其它 module 关系

```
Module 3 造的 ckpt    ┐
Module 4 改的 ckpt    ├─→ Topic 7 五线综合 mini-HELM 跑分
Module 5 部署的 ckpt  ┘
```

## 关键文献

- Hendrycks et al. 2020 MMLU
- Wang et al. 2024 MMLU-Pro
- Liang et al. 2022 HELM (Stanford)
- Suzgun et al. 2022 BBH
- Lin et al. 2021 TruthfulQA
- HuggingFace Open LLM Leaderboard v2 (2024.06)

## 一句话

> 评测 = 给 LLM 出题 + 对照答案。本 topic 教你跑通主流"高考"。
