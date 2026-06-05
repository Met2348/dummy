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

## 跑测试

```powershell
python learning/llm-judge-arena/src/tests/test_judge.py
```

预期：`8/8 modules passed`。

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/llm-judge-arena/src'); from mini_arena import make_arena_models, run_round_robin, make_leaderboard, to_md; from common import make_length_judge; models=make_arena_models(); j=make_length_judge(prefer_longer=True); print(to_md(make_leaderboard(run_round_robin(models, j))))"
```

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
