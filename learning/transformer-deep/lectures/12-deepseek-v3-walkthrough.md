# L12 · DeepSeek-V3 全栈精读

> 28 slides | 80 min | Transformer Deep 第 12 讲 ⭐⭐⭐⭐⭐

> 2024 年最具系统创新的开源大模型架构

---

## 学习目标

1. 看懂 DeepSeek-V3 完整 architecture
2. MLA + DeepSeekMoE + Aux-Free + MTP 四件套
3. 知道哪些是工程亮点、哪些是研究创新

---

## Slide 1 · 总览

| 项 | 值 |
|----|-----|
| 总参数 | 671B |
| 激活参数 | 37B (5.5%) |
| 层数 | 61 |
| Hidden | 7168 |
| Attention | MLA, h=128, d_low=512 |
| MoE | 1 shared + 256 routed, top-8 |
| Activation | SwiGLU |
| Norm | RMSNorm Pre |
| Context | 128k |
| Train tokens | 14.8T |

---

## Slide 2 · 架构图（简化）

```
Embed
   ↓
[Block × 61]
   ├─ MLA (RoPE + 压缩)
   ├─ RMSNorm
   ├─ FFN:
   │   - 前 3 层: dense SwiGLU
   │   - 后 58 层: DeepSeekMoE (1 shared + 256 routed)
   └─ RMSNorm
   ↓
LM head
```

前几层 dense 增强通用知识，后 MoE 增加容量。

---

## Slide 3 · MLA — Multi-head Latent Attention

```
压缩矩阵: W_DKV: d → d_low (512)
K, V: 共享 c_kv (d_low)
推理 K_full = c W_UK
推理 V_full = c W_UV
```

KV cache: 仅 c (d_low=512) 而非完整 K, V → **100× 压缩**。

---

## Slide 4 · MLA 关键 trick — RoPE 分离

```
K = [K_rope (d_rope=64) | K_nope (d_nope=128)]
                                  ↑
                          这部分被低秩压缩
K_rope 单独走 MHA path（小 d_rope=64）
```

→ 压缩与 RoPE 互不冲突。

---

## Slide 5 · DeepSeekMoE

```
1 个 shared expert (共享给所有 token)
256 个 routed expert (每 token 选 top-8)
top-k = 8 router
```

shared expert 学通用知识；routed expert 学专业知识。

---

## Slide 6 · routed expert 计算

```python
for each token:
    scores = router(token)        # (n_routed,)
    top_k_idx = topk(scores, 8)
    output = sum(expert_i(token) * scores[i] for i in top_k_idx)
output += shared_expert(token)
```

---

## Slide 7 · Aux-Free Routing ⭐⭐⭐⭐⭐

**最大创新**。传统 MoE 用 aux loss 平衡 expert 负载（DeepSeek-V2 还用）。

DeepSeek-V3：
```
bias_i 在每 step 后微调:
  if expert_i 被频繁选 → bias_i -= update_rate
  if expert_i 被冷落 → bias_i += update_rate

router score 用于排序的: s_i + bias_i
但梯度只过 s_i (不影响主 loss)
```

→ **不引入 aux loss，但负载自动均衡**。专题 3 详讲。

---

## Slide 8 · update_rate 经验

```
update_rate = 1e-3
bias init = 0
```

DeepSeek-V3 论文严格按此。改 update_rate 会显著影响均衡。

---

## Slide 9 · MTP — Multi-Token Prediction

```
正常: 预测 token t+1
MTP:  额外预测 token t+2, t+3, ... 多步骤
```

辅助 head 预测未来 token。训练时使用，推理时可选 speculative decoding。

→ 训练效率 +1-2 pp，推理理论上可 2× speedup。

---

## Slide 10 · MTP 实现

```
extra_head_2 = LM_head(layer_60_out + extra_proj_2(...))
extra_head_3 = ...
loss = ce(next_token) + 0.3 * ce(next_2) + 0.3 * ce(next_3)
```

---

## Slide 11 · FP8 训练

DeepSeek-V3 在 H800 上 FP8 端到端训练：

```
- 大部分 matmul: FP8
- 累加: BF16
- 关键路径 (softmax, gate): FP32
```

省 50% 显存 + 30% 速度。

---

## Slide 12 · NSA (Native Sparse Attention)

DeepSeek-V3 实验性使用了 NSA：
- 长上下文中只对部分 K 做 attention
- 三层 hierarchical: token / segment / page

效果待商榷，未完全用上。

---

## Slide 13 · 训练配方

```
pretrain:    14.8T token (en/zh/code/math 均衡)
mid-train:   长上下文扩展到 128k
SFT:         多源高质量
RLHF:        GRPO + 多 reward source
```

---

## Slide 14 · 与 Llama-3 对比

| | Llama-3.1 70B | DeepSeek-V3 |
|---|---------------|-------------|
| 架构 | Dense | MoE |
| 总参 | 70B | 671B |
| 激活 | 70B | 37B |
| Attn | GQA 64/8 | MLA |
| MoE | — | 1+256 / top-8 |
| Context | 128k | 128k |

→ DeepSeek-V3 总参 10×, 但激活 ~ 2×。

---

## Slide 15 · 训练成本

```
Llama-3.1 70B: ~ 7M H100 hours
DeepSeek-V3:   ~ 2.8M H800 hours (论文报数)
```

DeepSeek-V3 训练成本只有 Llama-3 ~40%。MoE + FP8 是核心。

---

## Slide 16 · 推理成本

```
Llama-3.1 70B: 7 GB KV cache @ 32k
DeepSeek-V3:   0.6 GB (MLA)
                ↓
        相同 batch 可服务 10× 更多请求
```

---

## Slide 17 · 各组件贡献

DeepSeek-V3 论文 ablation:

```
+ MLA:          训练快 25%, 推理快 8×
+ MoE 256+1:    +1.5 pp MMLU
+ Aux-Free:     +0.5 pp (省 loss + 更均衡)
+ MTP:          +0.7 pp + 推理 2× speed
+ FP8:           -50% 显存
```

---

## Slide 18 · DeepSeek-V3 → R1 路径

DeepSeek-V3 作为 R1-Zero 的基座：
```
V3 base
  ↓ GRPO + outcome reward
R1-Zero
  ↓ cold-start SFT + 4 阶段
R1
```

R1 训练在 V3 之上，复用所有架构 + MoE expert。

→ 专题 5 reasoning-r1 详讲。

---

## Slide 19 · 工程 take-away

```
1. MLA 是中大模型推理的最佳 KV cache 方案
2. Aux-Free MoE 路由是无 aux loss 但平衡的简洁解
3. MTP 训推都受益，几乎无成本
4. FP8 训练在 Hopper 上必须
```

---

## Slide 20 · 加载 DeepSeek-V3 (transformers)

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

670B 总参 + MoE → 需 多张 H100 (≥ 8 张 80GB)。

---

## Slide 21 · 架构组件统计

```python
def count_components(model):
    counts = {"linear": 0, "rmsnorm": 0, "embed": 0}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear): counts["linear"] += 1
        elif "RMSNorm" in type(m).__name__: counts["rmsnorm"] += 1
        ...
    return counts
```

详见 `src/deepseek_v3_summary.py`。

---

## Slide 22 · DeepSeek-V3 vs V2

```
V2: 已有 MLA, DeepSeekMoE, aux loss
V3: + Aux-Free, MTP, FP8, 训练成本下降
```

V2 是基础，V3 是工程精修。

---

## Slide 23 · 关于"开源"

DeepSeek-V3 完整开源（含权重 + 配置 + 训练 recipe）。
对比 GPT-4: 完全闭源。

→ 中国团队对开源贡献巨大。

---

## Slide 24 · DeepSeek 经验对其他模型的启发

```
Mistral / Qwen / Llama 后续版本会借鉴：
  - MLA 替代 GQA (推理省)
  - Aux-Free 替代 aux loss
  - MTP 加进训练
  - FP8 训练
```

预计 2026 大模型都会向 DeepSeek-V3 范式收敛。

---

## Slide 25 · Architecture 图示（概念）

```
Token in
  ↓
Embed (d=7168)
  ↓
[Layer 1-3: dense SwiGLU MLP]
  ↓
[Layer 4-61: MLA + DeepSeekMoE]
  ├─ attn: MLA(h=128, d_rope=64, d_nope=128, d_low=512)
  ├─ RMSNorm
  ├─ FFN: shared(d_ff=2048) + 256 expert × top-8
  └─ RMSNorm
  ↓
MTP head (+ token at t+1, t+2, t+3)
  ↓
LM head
```

---

## Slide 26 · 主要"读源码" hot spots

```
modeling_deepseek_v3.py
  ├─ DeepseekV3DecoderLayer
  │   ├─ DeepseekV3Attention (MLA)
  │   └─ DeepseekV3MoE
  ├─ Aux-Free router with bias
  └─ MTP heads
```

---

## Slide 27 · 复现路径

1. 8 × H100 + Mistral 7B base
2. 加 MLA layer 替换 GQA
3. 加 256-expert MoE
4. 训 100B token ablation

→ 教学复现路径。专题 3 / 7 联动。

---

## Slide 28 · 课后思考

1. MLA 比 GQA 多多少 FLOPs？
2. Aux-Free 在 1 expert 总是空时怎么救？
3. MTP 推理时怎么做 speculative？
4. FP8 训练失败模式有哪些？

---

## 参考

- DeepSeek-V3 技术报告 2024.12
- DeepSeek-V2 技术报告 2024.05
- MTP: Gloeckle 2024
