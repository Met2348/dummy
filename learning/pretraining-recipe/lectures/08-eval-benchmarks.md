# L08 · 评测 (HellaSwag / MMLU / Loss-based)

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 评测层次

```
内部: validation loss (语言建模 loss)
能力: HellaSwag / WinoGrande / PIQA (commonsense)
知识: MMLU (多学科 multi-choice)
推理: GSM8K / MATH
代码: HumanEval / MBPP
长 ctx: NIAH / RULER
对话: MT-Bench / AlpacaEval
```

## Slide 2 · validation loss

```python
@torch.no_grad()
def val_loss(model, val_data, n_batch=50):
    model.eval()
    total = 0
    for _ in range(n_batch):
        x, y = sample_batch(val_data)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
        total += loss.item()
    return total / n_batch
```

## Slide 3 · perplexity

```
ppl = exp(loss)
GPT-2 124M: ppl on Wikitext 30
Phi-1.5 1.3B: ppl 9.7
```

## Slide 4 · multiple choice

```python
def eval_mc(model, prompt, choices):
    losses = []
    for c in choices:
        ids = tokenize(prompt + c)
        with torch.no_grad():
            logits = model(ids[:-1])
            loss = F.cross_entropy(logits, ids[1:])
        losses.append(-loss * len(tokenize(c)))
    return np.argmax(losses)
```

## Slide 5 · HellaSwag

```
任务: 给定开头, 选最合理的句子续 (4 choice)
training: SWAG 改进
size: 70k val
acc:
  GPT-2 124M: 31%
  Phi-1.5:    53%
  Llama-2 7B: 76%
```

## Slide 6 · MMLU

```
57 学科 × 平均 200 题 (multi-choice 4)
0-shot / 5-shot
size: 14k 总
acc:
  random:     25%
  GPT-2:      26% (no signal)
  Phi-2:      58%
  GPT-4:      87%
```

## Slide 7 · BIG-Bench Hard (BBH)

```
23 hard tasks, mix 推理/代码/常识
Llama-2 7B BBH: 32
Llama-3 8B BBH: 65
```

## Slide 8 · 长 ctx 评测

```
NIAH: 见 long-context L10
RULER: NIAH 增强 (multi-key/value/var)
```

## Slide 9 · lm-evaluation-harness

```bash
pip install lm-eval
lm-eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks hellaswag,mmlu --device cuda
```

业界标准库, 大部分 benchmark 都有。

## Slide 10 · tinyMMLU

```
MMLU 14k → tiny-MMLU 100 题
快速 sanity check
< 5 min on 5090
```

## Slide 11 · 与 loss 关系

```
loss 低 ≠ benchmark 高
emergence 阶跃: 某 N 之前 benchmark 完全 random
loss 是 lower bound 必要不充分
```

## Slide 12 · 总结

```
评测分层: loss → commonsense → 知识 → 推理
工具: lm-eval-harness 一站
本 capstone 测: val_loss + tinyHellaSwag + tinyMMLU
```

## 参考
- HellaSwag (Zellers 2019)
- MMLU (Hendrycks 2020)
- lm-eval-harness GitHub
