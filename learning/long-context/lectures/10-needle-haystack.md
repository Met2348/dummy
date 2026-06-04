# L10 · NIAH + RULER 评测

> 18 slides | 55 min ⭐⭐⭐⭐

## Slide 1 · NIAH

```
长文档插入一个"针" (随机句子)
问 LLM: 那个句子是什么？
```

测试 LLM 在长 ctx 中找特定信息。

## Slide 2 · 实现

```python
def make_niah_query(haystack_text, needle, depth, length):
    # haystack 撑长度
    haystack = haystack_text * (length // len(haystack_text))
    # 在 depth × length 位置插 needle
    pos = int(depth * length / 100)
    text = haystack[:pos] + needle + haystack[pos:]
    return text + "\nQ: What's the special sentence?"
```

## Slide 3 · NIAH 评测矩阵

```
context length: 4k, 8k, 16k, 32k, 64k, 128k
needle depth: 0%, 25%, 50%, 75%, 100%
↓
5 × 6 = 30 测试点
```

## Slide 4 · 通过率

```
Llama-3.1 8B 128k:    NIAH > 95% 各点
Mistral 7B 32k:       早期 64%, "lost in middle"
Claude 3.5 200k:       几近 100%
```

## Slide 5 · RULER

NIAH 太简单（重复 needle 1 次）。RULER 加 13 子任务：
- Multi-key NIAH
- Multi-value NIAH
- Multi-query NIAH
- 变量追踪
- 聚合

## Slide 6 · "Lost in the Middle"

Liu 2023 现象：
```
context 长 → LLM attention 偏向首尾
中间信息易丢失
```

可视化：accuracy 在 depth 50% 显著低。

## Slide 7 · 现代模型

```
Llama-3.1: 训了长 ctx + RoPE scaling → middle 不太差
DeepSeek-V3: middle OK
GPT-4o: 几乎完美
```

## Slide 8 · 评测代码 niah_eval.py

```python
def niah_eval(model, tokenizer, context_lengths, depths,
               n_samples=10):
    results = {}
    for L in context_lengths:
        for d in depths:
            acc = 0
            for _ in range(n_samples):
                q = make_niah(L, d)
                pred = model.generate(q)
                acc += check_correct(pred)
            results[(L, d)] = acc / n_samples
    return results
```

## Slide 9-18 · 评测细节（略）

## 参考
- NIAH (Kamradt 2024)
- RULER (NVIDIA 2024)
- Lost in the Middle (Liu 2023)
