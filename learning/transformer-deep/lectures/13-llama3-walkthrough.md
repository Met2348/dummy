# L13 · Llama-3 精读

> 22 slides | 65 min | Transformer Deep 第 13 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 熟悉 Llama-3 8B / 70B / 405B 架构差异
2. 了解 RoPE scaling 与 128k context
3. GQA 8/64 的工程考量
4. 知道 Llama 系列演化（1 → 2 → 3 → 3.1 → 3.2）

---

## Slide 1 · Llama-3 总览

| | 8B | 70B | 405B |
|---|----|-----|------|
| 层 | 32 | 80 | 126 |
| Hidden | 4096 | 8192 | 16384 |
| n_head | 32 | 64 | 128 |
| n_kv_head | 8 | 8 | 16 (估) |
| d_head | 128 | 128 | 128 |
| d_ff | 14336 | 28672 | 53248 |
| vocab | 128k (tiktoken-style) | 128k | 128k |
| context | 128k | 128k | 128k |

---

## Slide 2 · Llama-3 关键改动

```
1. tokenizer: SP → tiktoken-style BBPE 128k
2. RoPE base: 10000 → 500000 (+ scaling)
3. 训练 token: 2T → 15T
4. 数学/code 专门 oversample + token split
```

vs Llama-2，每点都是显著工程。

---

## Slide 3 · Llama-3 architecture 公式

```
Q, K, V:  GQA 64/8
RoPE:     base 500000 + scaling
Norm:     RMSNorm Pre
MLP:      SwiGLU (d_ff = 28672 = 3.5 × d_model)
```

注意 Llama-2 d_ff 是 11008 = 2.69 × d_model, Llama-3 调高到 3.5×。

---

## Slide 4 · GQA 8/64 — 为什么 8

```
KV cache for 70B at 32k:
  MHA h=64:        5 GB
  GQA g=8:         0.6 GB    ← 选这个
  MQA g=1:         0.08 GB (太激进，质量损失)
```

g=8 是质量与显存的甜点。

---

## Slide 5 · RoPE scaling 三段

Llama-3.1 引入 "RoPE scaling type llama3"：

```
freq_low  = 1 / (factor * inv_freq)        # 长波段
freq_mid  = smooth interp
freq_high = 1 / inv_freq                    # 短波段不动
factor = 8.0 (扩 8×)
```

→ 长波段缓和 + 短波段保持 → 8k 训 32k+ 推。

详见专题 5 long-context。

---

## Slide 6 · 128k context 训练

```
training:    8k → mid-training 扩到 128k (~ 0.5T token)
RoPE scale 因子: 8
Sliding window: 不用
```

直接全 attn + RoPE scaling 训。

---

## Slide 7 · 训练 token 配比

```
en web         ~ 50%
multilingual   ~ 5%
math           ~ 5%
code           ~ 17%
other          ~ 23%
```

数学 / code 各专门 oversample。

---

## Slide 8 · vocab 128k

```
gpt2 50k → cl100k 100k → llama3 128k → o200k 200k
```

Llama-3 借鉴 cl100k 思路，自训 128k tiktoken-style vocab。中文压缩率比 Llama-2 +30%。

---

## Slide 9 · Llama-3 vs DeepSeek-V3

| | Llama-3 70B | DeepSeek-V3 |
|---|-------------|-------------|
| 类型 | dense | MoE |
| 总参 | 70B | 671B |
| Attn | GQA | MLA |
| KV cache | 中 | 小 |
| 训练成本 | 高 | 低 |
| 推理 | 标准 | MLA 省 |

→ 两条路线之争。Llama 大众化，DeepSeek 极致工程。

---

## Slide 10 · Llama-3 子系列

```
3.0    8B / 70B
3.1    8B / 70B / 405B (+ 128k)
3.2    1B / 3B (small) + 11B / 90B (vision)
3.3    70B (instruct 优化)
```

3.2 进入"小模型"领域；3.3 主推指令对齐。

---

## Slide 11 · Llama-3.2 小模型

```
1B: 16 层 / 2048 hidden / 32 head / 8 kv
3B: 28 层 / 3072 hidden / 24 head / 8 kv
```

为手机端 / 边缘部署设计。
专题 8 small-model-graduation 会深入。

---

## Slide 12 · Llama-3.2 vision

```
vision encoder + cross-attention → text decoder
11B / 90B vision variant
```

decoder 用 Llama-3 base，加 vision adapter。

---

## Slide 13 · 加载 Llama-3 (transformers)

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

8B 在 5090 24GB 上 bf16 加载 + 推理 OK。

---

## Slide 14 · 推理速度对比

```
Llama-3 8B on 5090 (32k context):
  prefill 8k: ~ 2s
  decode tok/s: ~ 40
  + vLLM PagedAttention: ~ 100
```

---

## Slide 15 · Llama-3 vs Llama-2

```
Llama-2 7B vs Llama-3 8B:
  MMLU:    46 → 66
  HumanEval: 13 → 62
  GSM8K:   15 → 80
```

7B 至 8B 体量微增，性能爆涨。证明数据 + 训练比 architecture 更重要。

---

## Slide 16 · Llama-3 训练 infra

```
24576 H100 GPUs
预训练 70B 用 ~ 6.4M GPU 小时
30 天连续训练
```

Meta 公开的 infrastructure paper 详述细节。

---

## Slide 17 · 关于"开源 + 商业 friendly"

Llama-3 license: 700M MAU 以下免费商用。
→ 中小公司商用首选。

vs DeepSeek MIT license: 完全自由。

---

## Slide 18 · 架构组件统计代码

```python
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
print(f"layers: {model.config.num_hidden_layers}")
print(f"hidden: {model.config.hidden_size}")
print(f"head: {model.config.num_attention_heads}")
print(f"kv:   {model.config.num_key_value_heads}")
print(f"d_ff:  {model.config.intermediate_size}")
print(f"vocab: {model.config.vocab_size}")
```

详见 `src/llama3_summary.py`。

---

## Slide 19 · 工程亮点 take-away

```
1. RoPE scaling type llama3 处理长上下文
2. GQA 8 head 是 70B 的甜点
3. 数学 / code 数据 oversample
4. tokenizer 切到 128k tiktoken-style
5. 训练规模 15T token + 大规模 GPU 集群
```

---

## Slide 20 · Llama-3 没做的

```
- MoE: 没用 (Mixtral / DeepSeek 用)
- MLA: 没用 (DeepSeek 用)
- MTP: 没用
- FP8: 没全栈用
```

Meta 保守路线 — 选成熟技术。

---

## Slide 21 · Llama-4 (未来)

预计 2025-2026：
- MoE 引入
- 多模态深度整合
- Embedding 大 (代号 Behemoth)
- 不可避免会借鉴 DeepSeek-V3

---

## Slide 22 · 课后思考

1. GQA 8 vs 4 vs 16 对 70B 影响？
2. Llama-3 不用 SWA 的原因？
3. RoPE 500000 base 在 4k context 是否有损？
4. Llama-3 8B → 405B 的参数膨胀曲线？

---

## 参考

- Llama-3 technical report 2024.07
- Llama-3.2 release notes 2024.09
- The Llama 3 Herd of Models paper
