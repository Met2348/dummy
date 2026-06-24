# Topic 5: Red-Team / Jailbreak（红队与越狱攻击）

> Module 6「评」第 5 专题 · 12 lectures · 12 notebooks · ~13h
>
> ⚠️ **安全声明**：所有代码均为**教学 mock**，
> 不针对真实生产模型，不输出有效 jailbreak prompt。
> 学习目的：理解攻击 → 才能更好防御（Topic 6）。

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 红队历史 + Anthropic 框架 | `common.py` |
| L02 | Jailbreak 4 大分类 | (lecture only) |
| L03 | GCG (Zou 2023) | `gcg_minimal.py` |
| L04 | PAIR (Princeton 2024) | `pair_minimal.py` |
| L05 | AutoDAN (UMD 2023) | `autodan_minimal.py` |
| L06 | Crescendo (Microsoft 2024) | `crescendo_demo.py` |
| L07 | Many-shot (Anthropic 2024) | (lecture only) |
| L08 | Prompt Injection (OWASP #1) | `prompt_injection_demo.py` |
| L09 | Prefilling attack | (lecture only) |
| L10 | 多模态 jailbreak | (lecture only) |
| L11 | JailbreakBench / HarmBench | `jailbench_runner.py` |
| L12 | **Capstone: 4×3 ASR 矩阵** | `red_team_matrix.py` |

## Tag

- `red-team-jailbreak` — Topic 5 完结

## 跑测试

```powershell
python learning/red-team-jailbreak/src/tests/test_redteam.py
```

预期：`8/8 modules passed`。

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/red-team-jailbreak/src'); from red_team_matrix import run_matrix, to_md; print(to_md(run_matrix()))"
```

## 关键文献

- Anthropic 2022 manual red-team
- Zou et al. 2023 GCG (CMU + CAIS)
- Chao et al. 2024 PAIR (Princeton)
- Liu et al. 2023 AutoDAN (UMD)
- Russinovich et al. 2024 Crescendo (Microsoft)
- Anil et al. 2024 Many-shot (Anthropic)
- OWASP LLM Top 10 (2024)
- Chao et al. 2024 JailbreakBench
- Mazeika et al. 2024 HarmBench (CMU)

## 与 Topic 6 衔接

Topic 6 (`safety-defense`) 用同样攻击方法 + 加防御，对比 ASR 降低效果。

## 一句话

> 红队 = 找洞，懂攻击才能造好防御（Topic 6）。


---
## 🔬 小而真 · 真实模型例子
> 除 toy 外, 本专题附一个**真实小模型** notebook (本地 gpt2/TinyLlama, CPU 离线):
> - [`notebooks/N13-real-refusal.ipynb`](notebooks/N13-real-refusal.ipynb) — 真实 TinyLlama-Chat 拒答行为 (防御视角: 正常答/有害拒/边界脆弱)
> 共享工具见 [`learning/_shared/realmodels.py`](../_shared/realmodels.py)。
