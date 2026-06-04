# L12 · 量化评测

## 1 · 维度
| 维度 | 量 | 方法 |
|------|---|------|
| 精度 | PPL / MMLU / HellaSwag / 任务 acc | 评测库 |
| 显存 | weight + KV + activation | `torch.cuda.max_memory_allocated` |
| 速度 | TTFT, ITL, tok/s | 标准 benchmark |
| 稳定性 | extreme prompts 测试 | 红队 |

## 2 · PPL 评测细节
- 用 Wikitext-2 或 c4 子集
- 严格 stride 1 滑动
- 计算 `exp(mean(NLL))`

## 3 · 任务 acc
- MMLU 5-shot（4 选 1，57 学科）
- HellaSwag 0-shot
- GSM8K 0-shot CoT
- 注意 prompt 格式（zero/few-shot）一致

## 4 · 显存测量
```python
torch.cuda.reset_peak_memory_stats()
out = model(input_ids)
peak = torch.cuda.max_memory_allocated() / 1e9
```

## 5 · 速度测量
- 先 warmup 5 iter
- 测 100 iter 取 median
- 报告 ITL (inter-token latency)
- 注意 batch=1 / batch=N 差别

## 6 · A/B 对比表（模板）
| 方案 | PPL | MMLU | 显存 | tok/s |
|------|-----|------|------|-------|
| fp16 | 5.68 | 45.3 | 14 GB | 130 |
| GPTQ-4 | 5.85 | 44.5 | 3.5 GB | 180 |
| AWQ-4 | 5.81 | 44.9 | 3.5 GB | 200 |
| FP8 | 5.70 | 45.0 | 7 GB | 220 |
| W4A8 | 5.95 | 43.0 | 3.5 GB | 280 |

## 7 · 实现：[quant_eval.py](../src/quant_eval.py)
- `eval_ppl_mock(model_fn, data)`
- `eval_mmlu_mock(model_fn)`
- `memory_table` 生成 markdown
