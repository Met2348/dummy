# L10 · Hybrid 设计原则

> 20 slides | 60 min ⭐⭐⭐⭐

## Slide 1 · 何时混

```
- 长 context (32k+):       要 mamba 减 KV cache
- retrieval task:          要 attention
- 流式 streaming:           要 mamba
- 通用 chat:               attention 仍主
```

## Slide 2 · 比例

```
Jamba: 1 attn : 7 mamba
Zamba: 1 shared attn : N mamba
Pure Mamba (Codestral-Mamba): 0 attn
```

任务定 → 比例定。

## Slide 3 · 选哪一层 attention

```
均匀:   每 N 层 1 个 (Jamba)
首末:   第 0 和最后一层 attention
关键:   学习选 (NAS)
```

## Slide 4 · 实务建议

```
预算紧 (< 7B):     纯 Mamba 或 Zamba (省参)
中等 (7B-30B):    Jamba 风格 1:7
大模型 (>30B):    仍多用 MoE+attn (DS-V3)
```

## Slide 5 · 训练时

```
混合 forward:
  - mamba 用 SSM scan
  - attention 用 SDPA / FA
两种 kernel 都需调优
```

## Slide 6 · 推理时

```
KV cache + SSM state 并存
batch 时管理两种 cache
```

vLLM 支持 hybrid model（0.6+）。

## Slide 7 · attention sink + Mamba

attention layer 仍有 sink，mamba 无。
混合时 sink 处理与全 attention 一致。

## Slide 8 · 长 context scaling

```
纯 attention:     不可推外 (无 RoPE scaling)
混合:             attention 部分用 RoPE+YaRN
                 mamba 天然长
```

## Slide 9 · MoE + Mamba

```
Mamba layer + MoE FFN:
  - 用 routing 选 mamba expert? 几乎无人做
  - 通常 MoE 加在 attention 层 FFN
```

## Slide 10 · 实务 best practices

```
1. 先全 attention 训
2. ablate 改成 hybrid
3. 监控 long context 性能
4. 推理时验证 KV cache + SSM state 配合
```

## Slide 11-20 · 详细案例（略）

## 参考
- Jamba paper
- Zamba paper
- Hybrid trends 2024-2025
