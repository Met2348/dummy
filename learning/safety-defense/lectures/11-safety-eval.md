# L11 · 安全评测 bench

## 1. WildBench (2024)

- **1024 真实用户 prompt** 来自 WildChat (Anthropic)
- pairwise judge 评 "helpful + harmless"
- 含**hard subset** 测对齐性

## 2. HHH (Helpful / Honest / Harmless) — Anthropic

- 经典 3 维评对齐
- 数据集：anthropic/hh-rlhf

## 3. TrustLLM (2024.01)

- 8 维：truthfulness / safety / fairness / privacy / robustness / ethics / accountability / transparency
- 多模型横评

## 4. SALAD-Bench (2024)

- 21k harmful queries × 6 分类
- 攻击 / 防御双角度
- "Safety AnaLysis And Defense Bench"

## 5. WildGuardMix / WildJailbreak

- WildGuard 的训练 + eval 集
- 92k adversarial 样本
- 用作 OOD safety eval

## 6. ToxiGen

- 生成式有毒文本
- 隐式 (implicit) hate detection
- 274k 训练 + 940 测试

## 业界使用

| Lab | 主用 bench |
|-----|----------|
| OpenAI | RealToxicityPrompts + 内部 |
| Anthropic | HHH + Constitutional AI eval + 内部 |
| Meta | HarmBench + Llama Guard eval |
| Google | SALAD-Bench + 内部 |
| 中国 | C-Eval + SafetyBench |

## Bench × Defense 表（toy 示例）

| Defense | HarmBench ASR ↓ | WildBench ↑ | TrustLLM |
|---------|------------------|------------|----------|
| baseline | 80% | 60 | 0.55 |
| + Llama Guard | 25% | 58 | 0.68 |
| + Constitutional Cls | 4% | 60 | 0.83 |
| + NeMo rules | 3% | 55 | 0.80 |

## 实操（自家 eval）

src/safety_eval_runner.py 评 precision/recall/F1：

```python
from safety_eval_runner import evaluate_guard, to_md
from llama_guard_mock import classify_input

metrics = evaluate_guard(classify_input)
print(to_md("llama_guard", metrics))
```

输出：
```
- precision: ...
- recall:    ...
- F1:        ...
```

## 一句话

> 安全评测 bench = 标准化报"我多安全"的语言；6 个主流，业界混用。
