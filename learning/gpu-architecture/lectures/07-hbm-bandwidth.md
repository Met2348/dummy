# L07 — HBM 带宽是 LLM 推理的天花板

## HBM 演进

| Gen | 速率 | 容量 stack | 旗舰 GPU | 典型 GPU 总 BW |
|----|-----|-----------|---------|---------------|
| HBM2 | 1.6 GT/s | 16 GB | V100 | 0.9 TB/s |
| HBM2e | 2.4 GT/s | 16 GB | A100 80G | 2.0 TB/s |
| HBM3 | 6.4 GT/s | 24 GB | H100 80G | 3.35 TB/s |
| HBM3e | 9.2 GT/s | 36 GB | H200 / B200 | 4.8 / 8.0 TB/s |
| HBM4 (2025+) | 8.5 GT/s × 2× pin | 48 GB | Rubin (rumor) | 16+ TB/s |

## 推理算式

decode 阶段，每生成 1 token 必须读完整个 weight + KV cache：

`tokens_per_sec ≤ BW / (sizeof_weights + sizeof_KV)`

- Llama-70B BF16 = 140 GB → H100 80G 装不下 (需 TP=2)
- Llama-70B FP8 = 70 GB → H100 单卡可装 → BW 3.35 TB/s / 70 GB ≈ 47 tok/s 上限
- Llama-70B FP4 = 35 GB → B200 8 TB/s / 35 GB ≈ 228 tok/s 上限

## 关键洞见

**推理 BW 受限 → 量化是最大杠杆**。这就是为什么 Blackwell 上 FP4 推理价值远超 FP4 训练。
