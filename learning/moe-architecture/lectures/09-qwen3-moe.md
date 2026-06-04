# L09 · Qwen3-MoE 系列

> 16 slides | 50 min | MoE Architecture 第 9 讲 ⭐⭐⭐⭐

---

## Slide 1 · Qwen3 系列概览

阿里 2024.12-2025.01:

```
Qwen3-A3B:   30B 总, 3B 激活, MoE
Qwen3-235B:  235B 总, 22B 激活, 高端
Qwen3-Coder: 代码专项
Qwen3-Math:  数学专项
```

---

## Slide 2 · Qwen3-A3B 架构

```
n_layer = 48
hidden = 2048
n_head = 32, n_kv = 4 (GQA)
n_experts = 128 routed, 0 shared
top_k = 8
```

类 DeepSeek-V2 的细粒度路线。

---

## Slide 3 · Qwen3-235B 架构

```
n_layer = 94
hidden = 4096
n_head = 64, n_kv = 8 (GQA)
n_experts = 128 routed
top_k = 8
```

→ 与 V3 接近，但 expert 数稍少 (128 vs 256)。

---

## Slide 4 · routing 算法

Qwen3 用 aux loss + GShard 风格。没切到 Aux-Free（2024.12 时 V3 才出）。

---

## Slide 5 · 性能

| Model | MMLU | GSM8K | HumanEval |
|-------|------|-------|-----------|
| Qwen3-A3B | 72 | 80 | 68 |
| Qwen3-235B | 80 | 90 | 78 |

A3B 性价比突出。

---

## Slide 6 · 多语言

Qwen 系列强中文、日文、韩文。

→ tokenizer 152k（vs Llama 128k），中文压缩比好。

---

## Slide 7 · 加载 Qwen3

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-A3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

A3B 4bit ~ 18GB → 5090 OK。

---

## Slide 8 · 推理速度

```
5090 4bit:        ~ 30 tok/s
+ vLLM:           ~ 100 tok/s
```

A3B 在边缘 GPU 上推理友好。

---

## Slide 9 · 训练数据

```
预训练 ~ 18T token
en/zh ~ 50/50
code ~ 15%
math ~ 8%
```

中文比例比 Llama / DeepSeek 高很多。

---

## Slide 10 · 与 Mixtral 对比

```
Qwen3-A3B: 128 expert × 220M ≈ 30B
Mixtral 8x7B: 8 expert × 7B = 56B
↓
Qwen 细粒度路线，参数效率高
```

---

## Slide 11 · 推理优化

阿里自家 lib：
- vLLM 支持
- TensorRT-LLM 支持
- AWQ 量化兼容

---

## Slide 12 · 与 DeepSeek 哲学差异

```
DeepSeek: 极致压缩 (MLA + MTP + FP8 + Aux-Free)
Qwen:     工程稳定 (标准 GQA + 标准 MoE)
```

Qwen 更易上手，DeepSeek 极致效率。

---

## Slide 13 · 应用

Qwen3 / Qwen3-Code / Qwen3-Math 是中国市场主流：
- 阿里云
- 多家企业 fine-tune

---

## Slide 14 · 训练成本

未具体公开，但估计与 DeepSeek 同段。

---

## Slide 15 · 工程亮点

```
1. 中文压缩率高
2. 多 expert 大模型在中等 GPU 可用
3. 推理生态完善
```

---

## Slide 16 · 课后思考

1. Qwen3 与 DeepSeek-V3 哪个推理省？
2. 中文 SFT 数据从哪来？
3. Qwen3 是否会引入 Aux-Free？
4. 中国市场对 MoE 的应用 trend？

---

## 参考

- Qwen2.5 / Qwen3 技术报告 2024
- Qwen GitHub
