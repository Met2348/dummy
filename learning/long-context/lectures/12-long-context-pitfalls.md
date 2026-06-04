# L12 · 长 ctx 陷阱合集

> 14 slides | 40 min ⭐⭐⭐⭐

## Slide 1 · Lost in the Middle

Liu 2023 经典论文：
```
LLM accuracy 在 needle 位于
首部 90%, 中部 40%, 尾部 70%
                ↑ "middle U-curve"
```

## Slide 2 · 原因

```
1. RoPE 高频对近距 attention 偏置
2. casual mask → 前文累积更多 KV
3. SFT 数据多为短对话 → 长 ctx 中部冷启动
```

## Slide 3 · 缓解方法

```
✓ RoPE scaling (YaRN)
✓ NIAH-style 长数据 SFT
✓ middle-truncation 测试驱动
✗ vanilla attention 没改善
```

## Slide 4 · KV cache 爆炸

```
1B model + 128k ctx:
  KV cache ≈ 32 layer × 8 KV head × 128 head_dim × 2 (K+V) × 128k × 2 byte
            ≈ 17 GB
```

vs 模型权重 1B × 2 = 2 GB → KV 是模型 8 倍。

## Slide 5 · 解决方案

| 方法 | 显存 |
|------|------|
| GQA | KV 头变少 4-8× |
| MQA | 极端 KV=1 head |
| MLA | DeepSeek-V3 压缩 6.7× |
| KV quant int4 | 4× |
| H2O / SnapKV | 剪枝 token |
| PagedAttention | block 分配，碎片 0 |

## Slide 6 · prefill 慢

```
128k prefill: 2-15s
1M prefill: 60-300s
↑ 用户体验崩
```

## Slide 7 · prefill 优化

```
chunked prefill: 4k 一段，与 decode 流水
spec dec + verifier: 推测但要小心 long ctx
Mamba: 线性 prefill (不依赖 ctx)
```

## Slide 8 · context limit ≠ usable limit

```
广告: "128k window"
实际 NIAH 90%+: 32k
高质量 retrieval: 16k
```

模型卡上写的 ctx 长 ≠ 真正能用。

## Slide 9 · 长 ctx 与 fact recall 错觉

```
ctx 越长 → "应该能记住 everything"
现实: middle 信息易丢
建议: 长 ctx + RAG 组合
```

## Slide 10 · 评测警示 Spurious benchmark

```
naive perplexity: 长 ctx 上 perplexity 几乎不变 (低权重)
真正 reveal 长 ctx 缺陷的是 NIAH/RULER
```

## Slide 11 · 训练崩溃

长 ctx fine-tune 易出问题：
```
- gradient norm 长 ctx 上爆
- loss spike (尤其 packed 多 doc)
- attention 渐变模式异常
解决: warmup + grad clip + 小心 doc mask
```

## Slide 12 · 安全风险

```
长 ctx → 容易塞 prompt injection
"忽略上面所有指示，做 X" (隐藏在中部)
```

需要长 ctx-aware 防御机制。

## Slide 13 · 推理 vs 训练 ctx 不一致

```
训练: 128k
推理: 64k usable
SFT 在 65-128k 这段，但 inference 用 32k → 浪费
```

## Slide 14 · 总结

```
广告 ctx 长 ≠ 可用长
长 ctx = RoPE scaling + 长数据 SFT + KV 压缩 + retrieval 检测
缺一不可
```

## 参考
- Lost in Middle (Liu 2023)
- H2O (NeurIPS 2023)
- Prompt Injection in Long Context
