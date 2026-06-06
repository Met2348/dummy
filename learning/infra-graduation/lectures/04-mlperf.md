# L04 — MLPerf Training & Inference

## MLPerf Training v4.0 (2025)

| Task | 关注 metric | 当前 SOTA |
|------|-----------|----------|
| llm-pretrain | time-to-target loss | Llama-2-70B/Mixtral |
| llm-finetune | time-to-target perplexity | Llama-3.1 |
| image_classification | ResNet-50 to 75.9% | <2 min on 1024 H100 |
| object_detection | RetinaNet | retired |
| medical_imaging | 3D U-Net | retired |

## MLPerf Inference v4.1 (2025)

| Scenario | 用途 |
|---------|------|
| Server (SLA 99% latency) | 生产推理 |
| Offline (max throughput) | batch 离线 |
| Edge (mobile/embedded) | 端侧 |

LLM specific:
- Llama-2-70B-99% (MLPerf 推理 4.0+)
- Mixtral-8x7B-99%
- Stable Diffusion XL

## Capstone-2: 5 tasks × H100 vs B200

```
B200 vs H100 平均 2.28× 加速 (5 任务)
```

跟 MLPerf 实际报告 (B200 vs H100 2-4×) 吻合。

## 教学要点

- MLPerf 是 vendor 竞速场 — 数字光鲜但反映真实
- 提交分两类：closed (规则严格，可比) / open (任意优化)
- 训练 vs 推理 SOTA 演进速度不同 — 推理过去 1 年 4×，训练 1 年 1.5×
