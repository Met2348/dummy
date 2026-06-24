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

## 跑测试

```powershell
python learning/reasoning-eval/src/tests/test_reasoning.py
```

预期：`10/10 modules passed`。

## 跑 capstone

```powershell
python -c "import sys; sys.path.insert(0,'learning/reasoning-eval/src'); from capstone_reasoning_compare import run_capstone; print(run_capstone()['table'])"
```

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


---
## 🔬 小而真 · 真实模型例子
> 除 toy 外, 本专题附一个**真实小模型** notebook (本地 gpt2/TinyLlama, CPU 离线):
> - [`notebooks/N13-real-cot.ipynb`](notebooks/N13-real-cot.ipynb) — 真实 TinyLlama CoT vs 直接答 (小模型真实算错 = 评推理为何要看过程)
> 共享工具见 [`learning/_shared/realmodels.py`](../_shared/realmodels.py)。
