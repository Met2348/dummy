# L11 · KV Cache 量化

## 1 · 痛点
长 ctx 服务，KV cache 比 weight 还占显存：
- 7B fp16 weight = 14 GB
- 8k ctx KV = 8 GB（per batch=1）
- 32k ctx KV = 32 GB → 5090 24GB 装不下

## 2 · 解：KV int8 / int4
- int8 KV → 显存减半（14 GB → 7 GB）
- int4 KV → 显存四分之一

## 3 · 精度
KV 比 weight 更敏感（attention softmax 放大误差）：
| 方案 | PPL |
|------|-----|
| fp16 KV | 5.68 |
| int8 KV (per-token) | 5.75 |
| int4 KV (per-token) | 6.20 |
| fp8 KV | 5.70 |

**fp8 KV 接近无损**，是最优选项（Hopper+ 硬件支持）。

## 4 · per-token vs per-channel
- per-token: 每 token 一个 scale → 精度高
- per-channel: 一列一个 scale → 适合 attention scoring

## 5 · 部署
```
# vLLM
--kv-cache-dtype fp8

# SGLang
--kv-cache-dtype int8
```

## 6 · 与 PagedAttention 集成
- KV 量化时存进 paged block
- 读取时 fused dequant + attention 算（FlashInfer kernel 支持）
- 不额外开销

## 7 · 实现：[kv_quant.py](../src/kv_quant.py)
- per-token int8 KV quant
- mock attention with quantized KV
- 误差对照 vs fp16 KV
