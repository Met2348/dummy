# L11 · 推理评测陷阱

## 1. Answer extraction 错误

```
Model: "The answer is approximately 3.14 (specifically π/1)."
Gold:  "3.14"
```

正则 `\d+\.\d+` 取到 "3.14" ✓ 
但 `\d+\.\d+|\d+/\d+` 可能取到 "π/1" 后续 → 错。

**对策**：
- 优先匹配 `\boxed{...}` 或 `#### N`
- 多 fallback 链
- LLM judge 抽答案

## 2. Pass@1 vs Pass@k 混报

```
Paper A: GPT-4 GSM8K 92% (pass@1, greedy)
Paper B: GPT-4 GSM8K 96% (cons@8)
```

直接对比 **错**。

**对策**：报告分数时必须注明 metric 名 + N。

## 3. Temperature 误用

```
GSM8K pass@1 用 T=0.0 (greedy)  ✓
AIME pass@k  用 T=0.7 (sample)  ✓
```

混用：T=0.7 测 GSM8K pass@1 → 抖太凶。

## 4. 多 sample 投票方式

| 方法 | 公式 | 用法 |
|------|------|------|
| **pass@k** | any(samples[0:k]) | 信息论上限 |
| **maj@k** | most_common(samples[0:k]) | R1 主报 |
| **cons@k** | maj 但 tie 算错 | 严格版 |
| **best-of-N** | argmax PRM(sample) | + verifier |

## 5. Verifier 不一致

A 用 sympy，B 用 string match，C 用 LLM judge。
同 GSM8K 题，A 算对 B 算错。

R1 paper 用 `math-verify` → 比 string match +2-5pp。

**对策**：报 verifier 名 + version。

## 6. 训练污染

```
模型见过 GSM8K test → 高分 ≠ 推理强
```

不写 verifier 也会"作弊"。
**对策**：用 LiveCodeBench / AIME 新年份 / HLE 这种"训练 cutoff 后"题。

## 7. CoT prompt 差异

```
"Let's think step by step."     → 强 CoT
"Show your work."               → 中 CoT
"Final answer is:"              → 几乎无 CoT
```

同模型不同 prompt，GSM8K 差 20pp。
**对策**：lock prompt template。

## 8. Few-shot demo 偷渡

```
5-shot demo 选自 train split 但 train 和 test 有重叠
```

不可能完全避免，需检查 split。

## 9. 长输出截断

```
max_new_tokens = 256
R1-style 思考链 ~2000 token → 截断后 \boxed{} 缺失
```

→ pred="" → 算错。

**对策**：R1-style eval `max_new_tokens >= 4096`。

## 10. AIME 答案越界

```
Gold: 994
Model: "1995"  (>999)
```

技术上无效，但许多 runner 当错。

**对策**：要么限制 generation，要么 post-filter [0, 999]。

## 一句话

> "GPT-4 GSM8K 92%" 后面藏着 10 个细节，没说清就是耍流氓。
