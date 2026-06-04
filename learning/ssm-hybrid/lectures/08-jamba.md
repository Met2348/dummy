# L08 · Jamba — Mamba + Attention + MoE Hybrid

> 24 slides | 70 min ⭐⭐⭐⭐

> AI21 2024.03 / 首个生产级 hybrid

## Slide 1 · Jamba 架构

```
Block 类型:
  Mamba layer
  Attention layer
  MoE layer
↓
按比例 1:7 配比 (1 attention : 7 mamba)
```

每层选不同类型。

## Slide 2 · 配比哲学

```
Mamba:    流式 / 长 context (主体)
Attention: 全局 retrieval (少量但关键)
MoE:      容量扩展
```

→ 各取所长。

## Slide 3 · 参数

```
Jamba-base: 52B 总, 12B 激活
hidden 4096
attn:mamba = 1:7
```

## Slide 4 · 性能

```
Jamba 52B vs Mixtral 47B:
  ppl 同
  long context (256k): Jamba 强
  推理速度: Jamba 快 (mamba 占多数)
```

## Slide 5 · 实现 jamba_block.py

```python
class JambaLayer(nn.Module):
    def __init__(self, cfg, layer_type="mamba"):
        if layer_type == "attention":
            self.mixer = MHA(...)
        elif layer_type == "mamba":
            self.mixer = MambaBlock(...)
        # 可选 MoE FFN
        if cfg.use_moe:
            self.ffn = MoELayer(...)
        else:
            self.ffn = SwiGLU(...)
```

## Slide 6 · attention 层选哪一层

```
Jamba: 第 4 / 12 / 20 / 28 层 是 attention
其他都是 Mamba
```

均匀分布 OR 重要位置。

## Slide 7 · KV cache

attention 层有 KV cache，Mamba 层有 SSM state。
混合时两种 cache 并存。

## Slide 8 · 推理速度

```
attention 层: O(L²) → 但只 1/8 层
Mamba 层:     O(L) → 主体
↓
整体接近 O(L · L/8) = O(L²/8)
```

比纯 transformer 快 ~5×（长序列）。

## Slide 9 · 训练成本

```
Jamba 52B 训练 ~ 3T token
比 Mixtral 47B 训练成本接近
```

## Slide 10 · MoE 加在何处

```
Jamba 把 MoE 加在某些 attention 层
不是 Mamba 层
```

设计选择，未必最优。

## Slide 11 · 实务何时用

```
长 context 服务:     Jamba 友好
低延迟推理:           Jamba 友好
通用 chat:           Mixtral / Llama 仍主流
```

## Slide 12 · 加载

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

## Slide 13 · 与 Phi-Hybrid

Phi-3.5-vision 也部分用 hybrid 思路。
未来更多 hybrid 模型出现。

## Slide 14-24 · 详细（略 - 见 Jamba paper）

## 参考
- Jamba (AI21 2024.03)
- AI21 blog 详解
