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

## 跑测试

```powershell
python learning/eval-foundations/src/tests/test_eval.py
```

预期：`10/10 modules passed`。

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/eval-foundations/src'); from common import make_random_model; from eval_pipeline import run_all_benches, to_md; print(to_md(run_all_benches(make_random_model()), 'random_baseline'))"
```

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
