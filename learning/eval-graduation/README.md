# Topic 7: Eval Graduation（评测毕业 ⭐⭐⭐⭐⭐⭐）

> Module 6「评」第 7 专题 — **系列毕业** · 14 lectures · ~14h
>
> 全部 25 prior + 7 Module 6 = **32 专题学习马拉松** 收官 capstone

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 评测毕业总览 | (intro) |
| L02 | 25 专题 ckpt zoo | `ckpt_zoo/` |
| L03 | bench × ckpt 矩阵设计 | (lecture) |
| L04 | mini-HELM | `mini_helm.py` |
| L05 | mini-Arena | `mini_arena.py` |
| L06 | mini 红队 | `mini_red_team.py` |
| L07 | mini 防御 | `mini_defense.py` |
| L08 | 成本工程 | (lecture) |
| L09 | Portfolio 设计 | `portfolio.py` |
| L10 | blog 风格 README | (lecture) |
| L11 | 选型决策树 | (lecture) |
| L12 | **Capstone-1: mini-HELM** | `mini_helm.py` |
| L13 | **Capstone-2: Arena + 红队 + 防御** | 3 个 src |
| L14 | **Capstone-3: Portfolio ⭐⭐⭐⭐⭐⭐** | `portfolio.py` |

## Tags

- `评-graduation` — 系列收官（3 Capstone + Portfolio）⭐⭐⭐⭐⭐⭐
- `module6-complete` — Module 6 整体完成

## 跑测试

```powershell
python learning/eval-graduation/src/tests/test_graduation.py
```

预期：`6/6 modules passed`。

## 跑 3 Capstone

```powershell
# Capstone-1: mini-HELM
python -c "import sys; sys.path.insert(0,'learning/eval-graduation/src'); from mini_helm import run_mini_helm, to_md; print(to_md(run_mini_helm()))"

# Capstone-2A: mini-Arena
python -c "import sys; sys.path.insert(0,'learning/eval-graduation/src'); from mini_arena import run_capstone_arena, to_md; print(to_md(run_capstone_arena()))"

# Capstone-2B: 红队
python -c "import sys; sys.path.insert(0,'learning/eval-graduation/src'); from mini_red_team import run_red_team, to_md; print(to_md(run_red_team()))"

# Capstone-2C: 防御加固
python -c "import sys; sys.path.insert(0,'learning/eval-graduation/src'); from mini_defense import compare_defense, to_md; print(to_md(compare_defense()))"

# Capstone-3: Portfolio (写文件)
python -c "import sys; sys.path.insert(0,'learning/eval-graduation/src'); from portfolio import write_portfolio; print(write_portfolio('portfolio.md'))"
```

## 5 个 ckpt 对应

| ckpt | 出处 |
|------|------|
| `vanilla` | Module 3 baseline GPT-2 base |
| `lora` | Module 1 PEFT 后 |
| `dpo` | Module 4 改大模型 (RLHF) |
| `r1_tiny` | Module 4 reasoning-r1 |
| `phi_tiny` | Module 3 pretraining-recipe |

## 跨 Module 闭环图

```
Module 1 PEFT (3)        ┐
Module 3 造大模型 (8)     ├─→ Module 6 评测/安全 (本 capstone)
Module 4 改大模型 (7)     │       ↓
Module 5 用大模型 (7)     ┘  32-topic Portfolio ⭐⭐⭐⭐⭐⭐
```

## 关键文献

- HELM (Stanford 2022)
- Chatbot Arena Elo (LMSYS 2024)
- HarmBench (CMU 2024)
- Constitutional Classifiers (Anthropic 2025)
- Llama Guard 3 (Meta 2024)
- 整个 Module 6 引用文献

## 一句话

> Module 6 收官 = Module 1+3+4+5 的 25 专题学完后，加 1 份 portfolio.md ——
> 你的"LLM 全栈工程师 ID 卡"。
