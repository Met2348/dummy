# L09 · 同 prompt 多 ckpt 生成对照

> 8 slides | 25 min ⭐⭐⭐

## Slide 1 · 5 个 ckpt 同 prompt

```python
prompts = [
    "The quick brown fox jumped over",
    "In a small village, there lived a",
    "Once upon a time, there was a",
    "The chef cracked an egg into",
    "The robot turned on its lights and",
]
```

## Slide 2 · 生成 config

```python
gen_cfg = {
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "seed": 42,
}
```

## Slide 3 · 期望差异 (qualitative)

```
A: 简短, 多语法错, 重复
B: 流畅度提高, 仍语义浅
C: 上下文连贯, 词汇丰富
D: 同 C, 但能处理长 ctx prompt
E: 最佳, 一致 + 连贯 + 丰富
```

## Slide 4 · 实际比较 (mock)

```
Prompt: "The chef cracked an egg into"

A: "the pan and the the the and..."
   ↑ 重复退化

B: "the pan and started cooking dinner."
   ↑ 短 + 直接

C: "the sizzling pan. The aroma of butter filled
    the kitchen as he reached for the spatula."
   ↑ 上下文丰富

D: 同 C

E: "the hot skillet. The yolk burst open and
    spread across the buttered surface as he
    deftly flipped it with practiced ease."
   ↑ 细节 + 连贯
```

## Slide 5 · markdown 记录

```markdown
## Prompt 1: "The chef cracked..."

### ckpt A
> the pan and the the the and...

### ckpt B
> the pan and started cooking dinner.

...
```

## Slide 6 · 主观评分 (LLM-as-judge)

```python
def judge(prompt, response, criteria="fluency"):
    rubric = "Rate fluency 1-5"
    judge_response = call_gpt4(rubric, prompt, response)
    return parse_score(judge_response)
```

可选 GPT-4 自动打分.

## Slide 7 · 注意 sampling 随机

```
同 prompt 不同 seed 输出不同
→ 5 次平均 / 多 sample
```

## Slide 8 · 总结

```
生成对照是定性评测
配合 quantitative benchmark
让 reader 看见模型差异
```

## 参考
- HELM evaluation
- AlpacaEval
