# L12 · MoE 推理优化

> 24 slides | 70 min | MoE Architecture 第 12 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 理解 MoE 推理瓶颈（不同于训练）
2. 知道 grouped GEMM (megablocks) 加速
3. expert offload (CPU/disk)
4. 与 vLLM / TensorRT-LLM 集成

---

## Slide 1 · MoE 推理瓶颈

```
1. memory: 256 expert 全在 GPU 上 → 显存高
2. compute: 每 token 选 8 expert → 稀疏 GEMM 慢
3. dispatch: token 路由 + 收集 → 通信
```

---

## Slide 2 · grouped GEMM

```
直接循环 256 expert:
  for i in 256: out = expert_i(tokens_i)
↓ 慢，每个 expert kernel 启动
grouped GEMM:
  one fused kernel handle all expert
```

megablocks 库实现，~ 3× 加速。

---

## Slide 3 · megablocks

GitHub stanford-futuredata/megablocks。

```python
from megablocks.layers.moe import MoE
moe = MoE(num_experts=256, expert_capacity=1.25, top_k=8)
```

CUDA + Triton 写的高效 sparse kernel。

---

## Slide 4 · grouped GEMM 数学

```
传统:  Y = X · W  (dense)
grouped: 
  for each group i:
    Y_i = X_i · W_i
  → 把所有 group 拼成 block-diagonal
```

block-diagonal 形式 GPU 友好。

---

## Slide 5 · expert offload

```
hot expert (top-32) GPU
warm (32-128)        CPU pinned
cold (> 128)         disk
```

按 expert 使用频率分层。

---

## Slide 6 · CPU offload 实现

```python
class OffloadedExpert(nn.Module):
    def __init__(self, expert):
        self.expert_cpu = expert.cpu()
    def forward(self, x):
        expert_gpu = self.expert_cpu.cuda()
        out = expert_gpu(x)
        del expert_gpu  # GC
        return out
```

简化版，性能差但教学清晰。

---

## Slide 7 · vLLM + MoE

vLLM 0.5+ 支持 MoE 模型：
- Mixtral / DeepSeek-V3 / Phi-MoE
- PagedAttention + grouped GEMM
- 推理 200 tok/s 量级

---

## Slide 8 · TensorRT-LLM + MoE

NVIDIA 提供 MoE kernel：
- INT8 / FP8 量化
- Expert parallel 多 GPU
- 性能 vLLM × 1.5

---

## Slide 9 · 量化

```
Mixtral 8x7B:
  fp16: 90GB
  int8: 45GB    
  int4 (AWQ): 24GB → 5090 OK
```

MoE 比 dense 更容易压缩（expert 间冗余）。

---

## Slide 10 · sparse compile

```
torch.compile 对 MoE 支持有限
megablocks 提供 Triton 编译版
```

未来 PyTorch 2.6+ 改善 sparse 编译。

---

## Slide 11 · Routing 优化

```
原: gate(x) → softmax → topk → all_to_all → expert → all_to_all
快: 1. gate 用 fp16 (而非 fp32)
    2. topk fused with all_to_all
    3. expert 推理 fused
```

DeepSeek-V3 内部 kernel 这么干。

---

## Slide 12 · capacity unconstrained 推理

推理时不需 drop：
```
inference: capacity = ∞
training:  capacity = 1.0
```

无 drop 模型质量略升。

---

## Slide 13 · batch + MoE

batch 越大，routing overhead 越摊薄。

```
batch=1: gate 占 30% time
batch=32: gate 占 < 5%
```

vLLM continuous batching → 推理友好。

---

## Slide 14 · expert parallel inference

```
GPU 0: expert 0-31
GPU 1: expert 32-63
...
8 GPU × 32 expert = 256 expert
每 token all-to-all 到对应 GPU
```

DeepSeek-V3 推理标准。

---

## Slide 15 · expert sharing

```
跨 layer 共享 expert (理论)
→ 参数 ×2 但激活同
```

实务未广泛采用。

---

## Slide 16 · KV cache

MoE attention 与 dense 同：
- Mixtral GQA 32/8: KV cache 标准
- DeepSeek MLA: 进一步 100× 压缩

attention 与 routing 解耦。

---

## Slide 17 · 实测推理速度

```
5090:
  Mixtral 4bit:     25 tok/s
  Phi-3.5-MoE 4bit:  20 tok/s
H100 + vLLM:
  Mixtral fp16:     150 tok/s
  DeepSeek-V3:      80 tok/s (671B)
```

---

## Slide 18 · 推理硬件 ROI

```
MoE 671B 总:    需 8× H100
等效 dense 70B: 1-2× H100
↓
MoE 容量大但 deployment 贵
```

→ MoE 适合大规模 SaaS，不适合单卡 demo。

---

## Slide 19 · expert prefetch

```
预测 token routing pattern → expert prefetch from CPU
减少 stall
```

类似 OS prefetch。

---

## Slide 20 · 推理框架对照

| | Mixtral | DeepSeek-V3 |
|---|---------|-------------|
| transformers | yes | yes (新版) |
| vLLM | yes (主流) | yes |
| TensorRT-LLM | yes | yes |
| SGLang | yes | yes |
| llama.cpp | yes (gguf) | partial |

---

## Slide 21 · 自训 MoE 怎么 deploy

```
1. 保存 SafeTensors
2. 用 transformers 加载
3. 用 vLLM 推理
4. 4bit AWQ 量化
```

完整流程 (Phi-MoE 风格)。

---

## Slide 22 · grouped GEMM demo

`src/grouped_gemm_demo.py`:

```python
import torch

def grouped_gemm_naive(xs, ws):
    """naive: 循环."""
    return [x @ w for x, w in zip(xs, ws)]

def grouped_gemm_fused(xs, ws):
    """fused: cat + block diag (toy)."""
    X = torch.block_diag(*xs)
    W = torch.block_diag(*ws)
    return X @ W
```

megablocks 优化版用 Triton。

---

## Slide 23 · 工程 take-away

```
1. 推理用 vLLM / TensorRT-LLM 首选
2. 4bit 量化是 MoE 标配
3. expert offload 在 24GB 必需
4. continuous batching + routing overhead 减小
```

---

## Slide 24 · 课后思考

1. grouped GEMM 在小 batch 是否反而慢？
2. expert offload 与 PagedAttention 兼容吗？
3. INT4 vs INT8 MoE 性能差？
4. 推理时 routing 能 batch 化吗？

---

## 参考

- megablocks paper 2023
- vLLM 文档
- DeepSeek-V3 inference 章节
