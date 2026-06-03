# L08 · FlashAttention v2 / v3

> 24 slides | 70 min | Transformer Deep 第 8 讲 ⭐⭐⭐⭐

> v2 (2023) / v3 (2024 H100 FP8 TMA)

---

## 学习目标

1. 理解 FA2 比 FA1 提升的 3 个工程点
2. 了解 FA3 在 H100 上的 FP8 + TMA 加速
3. 知道何时该升级

---

## Slide 1 · FA1 局限

```
- causal mask 时未对角线块单独优化
- backward 不充分并行
- 仅 fp16 / bf16，无 FP8
- 单 SM 内部 warp 协同弱
```

→ FA2 (2023.07) 主要解决这些。

---

## Slide 2 · FA2 改进 1 — 减重算

```
FA1 backward 重算 Q,K,V → 浪费 25%
FA2: 重算 attention matrix 但只算需要的
   → 更精细的 dQ, dK, dV 计算路径
```

backward 提速 ~30%。

---

## Slide 3 · FA2 改进 2 — warp 重排

```
FA1: 每 warp 同一行
FA2: 各 warp 同一列
↓
更少 sync barrier，吞吐 +50%
```

具体 CUDA warp specialization 细节略。

---

## Slide 4 · FA2 改进 3 — causal mask

```
FA1: 完全 mask 块跳过
FA2: 三角对角块单独 kernel，零 mask check
↓
causal context 额外 +30%
```

---

## Slide 5 · FA2 综合速度

A100 长上下文：

```
FA1:  6× vs vanilla
FA2:  9× vs vanilla     (+50%)
```

实测 forward + backward 全网训练 ~1.6× over FA1。

---

## Slide 6 · FA3 (2024 H100)

主要为 NVIDIA Hopper (H100/H200):

```
1. FP8 输入 / 累加
2. TMA (Tensor Memory Accelerator) 异步加载
3. Warpgroup 协同 (新 PTX 指令)
```

需要 SM90+ (Hopper)。Ada Lovelace (4090/5090 sm_8x/12x) 部分受益。

---

## Slide 7 · TMA 异步加载

```
FA2:  加载 K, V 占 ~ 1/3 时间
FA3 TMA:  加载与 compute overlap
        → forward ~30% 更快
```

---

## Slide 8 · FP8 + 累加 BF16

```
Q, K, V    in FP8 (E4M3 / E5M2)
Q @ K^T    accumulate in BF16
softmax    in FP32
output     in BF16
```

数值挑战：FP8 范围窄，需要 scaling factor 校准。

---

## Slide 9 · FA3 数值精度

FP8 路径误差 ~ 0.5pp ppl over BF16 → 仍可接受。

Llama-3 / Mistral 8x22B 训练实际用 FA3 + FP8。

---

## Slide 10 · FA3 性能

H100 长上下文：

```
FA2:  ~ 50% of theoretical peak (BF16)
FA3:  ~ 75% of theoretical peak (BF16)
FA3 + FP8: ~ 1.5x → 接近 sm_90 极限
```

---

## Slide 11 · 5090 (sm_120) 当前

2026 现状：
- PyTorch 2.5 SDPA backend 自动选择
- flash-attn 2.6+ 支持 sm_120
- FA3 仅 H100/H200 (sm_90)

5090 受益于 FA2 完整功能 + 部分 FP8 (Ada Lovelace)。

---

## Slide 12 · API 对比

```python
# FA1/2
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)

# FA3 (sm_90)
from flash_attn import flash_attn_with_kvcache_v3
out = flash_attn_with_kvcache_v3(q, k, v, ...)
```

接口几乎透明，version 切换是 kernel 内部事。

---

## Slide 13 · benchmark 数字总览

| | A100 (FA2) | H100 (FA2) | H100 (FA3) | H100 (FA3 FP8) |
|---|-----------|-----------|-----------|----------------|
| 4k forward | 100% | 200% | 240% | 360% |
| 32k forward | 100% | 200% | 250% | 400% |

→ FA3 在 Hopper 上的 FP8 是 LLM 训练成本下降的关键。

---

## Slide 14 · 何时升级

```
A100 用户   → FA2 已极致
4090/5090   → FA2 + bf16
H100 用户   → FA3 + FP8 必装
```

---

## Slide 15 · 长上下文 + FA3

```
Llama-3.1 128k:  pretrain 全靠 FA3
                 inference 用 vLLM PagedAttention + FA kernel
```

FA 是长上下文的硬件基础设施。

---

## Slide 16 · FA 之外的 attention 工程

```
StreamingLLM     attention sinks 处理
RingAttention     多卡序列并行
PagedAttention    KV cache 分页
xFormers          通用 attention kernel
Mistral SWA       sliding window
```

各专攻一面，FA 是单卡 attention 内存效率基础。

---

## Slide 17 · attention 是否还能更快

理论上限：HBM bandwidth × 算法效率。FA3 已经 75%。

→ 更多收益来自：
1. 算法（lower rank Q/K）
2. 不同模型（SSM / Mamba 直接绕过 attention）

---

## Slide 18 · FA 不能做的事

- 不能算非二次 attention（MLA partially）
- 不能处理非常稀疏 mask
- 不优化超短序列 (< 256)

→ 短上下文 vanilla 仍可能更快。

---

## Slide 19 · 代码实务

```python
# 推荐：用 PyTorch SDPA 自动 backend
from torch.nn.functional import scaled_dot_product_attention as sdpa
out = sdpa(q, k, v, is_causal=True)
```

PyTorch 2.0+ 自动选 FA / cuDNN / math，无需手动选。

---

## Slide 20 · 显式用 FA 的理由

```
1. 想要 attn bias (ALiBi / 自定义)
2. 想要 dropout
3. 想要 KV cache 高效接口
4. 训练 LLM 在 Hopper 想 FP8
```

否则 PyTorch SDPA 已够用。

---

## Slide 21 · 与 LoRA / PEFT 配合

LoRA forward 改写 Q, K, V，但 attention 还是 standard FA：

```python
q = base_q(x) + lora_q(x)
out = flash_attn_func(q, k, v, ...)
```

PEFT 训练完全兼容 FA。

---

## Slide 22 · 与 KV cache 配合

vLLM PagedAttention + FA：
```
prefill 阶段:  FA full attention
decode 阶段:   PagedAttention (1 Q × N K, V)
```

下一讲 L09 详细。

---

## Slide 23 · "FA 的下个 5 年"

- Blackwell B200 sm_100 → 进一步 FP4
- Multi-GPU sequence parallel 集成
- Sparse + structured attention 支持

→ "attention 的硬件优化"远未完成。

---

## Slide 24 · 课后思考

1. FA2 比 FA1 快 1.5×，主要来自哪三处？
2. FA3 在 5090 上能不能用？
3. 自己写 Triton kernel 能比 FA2 快吗？
4. PyTorch SDPA 与 flash-attn lib 何时不一样？

---

## 参考

- Dao 2023 (FlashAttention-2)
- Dao 2024 (FlashAttention-3)
- NVIDIA H100 white paper
- vLLM PagedAttention 2023
