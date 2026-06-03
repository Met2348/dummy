# L05 · Mixtral 8x7B — 首个主流开源 MoE

> 18 slides | 55 min | MoE Architecture 第 5 讲 ⭐⭐⭐⭐

---

## Slide 1 · Mixtral 简介

Mistral 2024.01：
- 8 expert × 7B base
- top-2 routing
- 总参 47B，激活 13B
- 性能 ≈ Llama-2 70B

---

## Slide 2 · 架构

```
n_layer = 32
hidden = 4096
8 expert per FFN layer
top_k = 2
each expert: SwiGLU MLP d_ff=14336
```

---

## Slide 3 · 与 Llama-2 7B 关系

```
Mixtral = Llama-2 7B 架构 + 把 FFN 替换为 MoE
```

attention 等其他组件完全沿用 Llama-2 7B。

---

## Slide 4 · 训练数据

未公开具体数据。猜测：
- web 主体 + code + math
- 多语言（fr/de/es/it 强）

---

## Slide 5 · 性能

| Bench | Mixtral 8x7B | Llama-2 70B |
|-------|--------------|-------------|
| MMLU | 70.6 | 69.9 |
| HellaSwag | 87.4 | 87.3 |
| GSM8K | 60.4 | 57 |
| HumanEval | 40 | 30 |

→ 13B 激活 ≈ 70B 激活，效率高。

---

## Slide 6 · 推理速度

```
Mixtral 13B 激活：~  Llama-2 13B 速度
Llama-2 70B:        慢 5×
```

→ MoE 推理优势明显。

---

## Slide 7 · KV cache

```
Mixtral: GQA 32/8
KV cache 与 Llama-2 7B 一致（13B 激活但 attention 共享 KV）
```

---

## Slide 8 · 加载 Mixtral

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,    # 24GB GPU 必须 4bit
)
```

5090 24GB：4bit 加载 ~ 24GB，刚好够。

---

## Slide 9 · 推理 token/s

```
5090 4bit:        ~ 25 tok/s
H100 bf16:        ~ 70 tok/s
+ vLLM PA:        ~ 200 tok/s
```

---

## Slide 10 · Mixtral 与 GShard 路由对比

```
GShard top-2 + 1.25 capacity + aux 0.01
Mixtral top-2 + 1.0 capacity + aux 0.02
```

非常相似的配方。

---

## Slide 11 · expert 学到什么

Mistral 论文 ablation：
- expert 0-3 学语法 / 常识
- expert 4-7 学专业（code / math）

但实测 expert 行为高度交叠。

---

## Slide 12 · Mixtral 8x22B

Mistral 2024.04 升级版：
- 8 expert × 22B base
- 总参 141B，激活 39B
- 性能 ≈ Llama-3 70B

---

## Slide 13 · 训练成本估算

```
Mixtral 8x7B ~  Llama-2 7B 训练 × 5
              = 7M GPU 小时（A100）
```

每 expert 几乎独立训练。

---

## Slide 14 · open weights, open license

Apache 2.0 license → 完全商用。

→ 开启"开源 MoE"时代。

---

## Slide 15 · 推理优化

```
expert offload (CPU/disk)
grouped GEMM (megablocks)
vLLM 集成
```

详见 L12 inference 章节。

---

## Slide 16 · DeepSeekMoE 之前的最强 MoE

Mixtral 2024.01 - DeepSeek-V2 2024.05 期间，Mixtral 是 SOTA 开源 MoE。

---

## Slide 17 · 实务建议

```
新项目用 Mixtral 还是 DeepSeek?
- 兼容性优先 → Mixtral (主流 lib 支持)
- 极致效率 → DeepSeek-V3 (MLA + Aux-Free + MTP)
```

---

## Slide 18 · 课后思考

1. Mixtral 8 expert 真的够吗？为什么不像 GShard 2k？
2. 4bit Mixtral ppl 损失多少？
3. 8x7B 与 8x22B 各 expert 大小对模型的影响？

---

## 参考

- Mixtral 8x7B paper 2024.01
- Mixtral 8x22B 2024.04
