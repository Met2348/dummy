# L13 · 合成数据 (Phi 方式)

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 合成数据动机

```
"Textbooks Are All You Need" (Microsoft 2023)
50% 合成 + 30% 高质 web = Phi-1
真实 web 噪声太大
```

## Slide 2 · 合成 pipeline

```
1. 定 spec (topic/genre/age)
2. seed prompt 给 GPT-4 / Claude
3. 生成 K 倍样本
4. dedup + quality filter
5. tokenize → 加入训练集
```

## Slide 3 · Phi-1 合成 (CodeT5)

```
spec: Python 教科书
prompt: "Write a Python tutorial chapter on X"
6.6B token synthesized
+ filter (Stack Exchange 风格)
```

## Slide 4 · Phi-2 合成 (GPT-3.5)

```
扩展到 NL + code
1.4T 合成 token
比 CodeT5 时代精细
```

## Slide 5 · Cosmopedia (HuggingFace 2024)

```
开源合成数据集
30B token + 25 个 categories
Mistral-Large 生成
Phi-1.5 重训复刻
```

## Slide 6 · 数据生成 prompt template

```
You are an expert {{role}}. Write a clear, accurate
explanation of {{topic}} suitable for a {{audience}}.
Length: {{n_words}} words. Style: textbook.
Output only the explanation, no preamble.
```

## Slide 7 · 多样性管理

```
seed pool: 1k topic × 10 audience × 5 style = 50k 组合
温度 0.8 + top-p 0.95
逐 topic 调用 K 次 → 减少近重复
```

## Slide 8 · 质量过滤

```
- perplexity model 过滤极端高 ppl
- educational classifier (FineWeb-Edu)
- n-gram dedup vs eval set
- toxic filter
```

## Slide 9 · 风险

```
- model collapse: 多代合成数据 → 分布塌
  (Shumailov 2023 警告)
- bias 放大: teacher model bias 传递
- 法律: API ToS 限制
```

## Slide 10 · 实际比例

```
Phi-1.5: 30% 合成 + 70% real
Phi-2: 40% 合成
Phi-3: 不公开比例 (但仍是 mix)
保留 real 数据 → 防 model collapse
```

## Slide 11 · 自蒸馏 (Self-improve)

```
- model 自己生成
- 比 teacher 模式简单
- "self-improvement" 路线 (SPIN / Self-Rewarding)
```

## Slide 12 · 总结

```
合成数据是 Phi 的 secret sauce
不 替代 real data, 而是补充
prompt 工程 + 多样性 + 过滤 都关键
```

## 参考
- Phi-1 (Microsoft 2023)
- Cosmopedia (HF 2024)
- Shumailov 2023 (model collapse)
