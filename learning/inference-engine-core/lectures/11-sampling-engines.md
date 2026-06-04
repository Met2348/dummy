# L11 · Sampling 引擎

## 1 · 5 类 sampler
| 类 | 参数 | 用途 |
|----|-----|------|
| greedy | — | 确定性 / 评测 |
| temperature | T | 随机性调节 |
| top-k | k | 截断 |
| top-p (nucleus) | p | 自适应截断 |
| min-p | min_p | 2024 新增，混合截断 |
| Mirostat | tau | 控制 surprise |

## 2 · 实现要点（fused 单 kernel）
```python
logits = logits / T
mask = build_top_p_mask(logits, p) | build_top_k_mask(logits, k)
logits = logits.masked_fill(~mask, -inf)
probs = softmax(logits)
token = torch.multinomial(probs, 1)
```

## 3 · Guided / constrained
- regex / json schema / grammar → token mask（详见专题 2 SGLang）
- vLLM `--guided-decoding-backend outlines`

## 4 · 重复惩罚
- presence_penalty / frequency_penalty (OpenAI 兼容)
- 在 sampler 前对 logits 减/加

## 5 · 多请求一次性
- 每个请求各自 temperature/top-p
- 批 sample：要 broadcast 参数，避免循环

## 6 · 实现：[sampling.py](../src/sampling.py)
- `Sampler` class
- `sample_batch(logits, configs)` 一次性出
- top-k + top-p 合并 mask
- repetition penalty
