# L06 · DeepSeekMoE — 细粒度 + 共享专家

> 22 slides | 65 min | MoE Architecture 第 6 讲 ⭐⭐⭐⭐⭐

> DeepSeek-V2 引入 / V3 完善

---

## Slide 1 · DeepSeekMoE 创新

1. **细粒度**：256 个 small expert（vs Mixtral 8 large）
2. **共享专家**：1 shared expert + 256 routed
3. top-8 routing（vs Mixtral top-2）

---

## Slide 2 · 细粒度的好处

```
8 大 expert × top-2 = 25% 容量激活
256 小 expert × top-8 = 3% 容量激活
```

→ 容量 × 总参数大，激活相同时更精细。

---

## Slide 3 · 共享专家

```
output = shared_expert(x) + Σ_i top_k_gate_i × routed_expert_i(x)
```

shared 学通用知识，每 token 必过。routed 学专业。

---

## Slide 4 · DeepSeek-V2 vs V3

| | V2 | V3 |
|---|----|----|
| expert | 160 routed + 2 shared | 256 routed + 1 shared |
| top-k | 6 | 8 |
| aux | yes | **Aux-Free** ⭐ |

V3 引入 Aux-Free 是核心创新（L07 详）。

---

## Slide 5 · 细粒度路由实现

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, d, n_routed=256, n_shared=1, top_k=8):
        self.shared = SwiGLUMLP(d, d_ff)
        self.routed = ModuleList(SwiGLUMLP(d, d_ff_small) for _ in range(n_routed))
        self.gate = Linear(d, n_routed)
    def forward(self, x):
        out = self.shared(x)
        gates = self.gate(x).softmax(-1)
        top_k_gates, top_k_idx = gates.topk(8)
        for k in range(8):
            for token_i, expert_i in enumerate(top_k_idx[:, k]):
                out[token_i] += top_k_gates[token_i, k] * self.routed[expert_i](x[token_i])
        return out
```

---

## Slide 6 · 细粒度的"小 expert" 设计

```
routed expert: d_ff = d_model × 1.5 (vs Mixtral 3.5)
shared expert: d_ff = d_model × 8
```

每 small expert 更专业，shared 较大补全能力。

---

## Slide 7 · 总参数

```
DeepSeek-V3 671B:
  shared FFN: 1 × 25M ~ 25M
  routed FFN: 256 × 2.6M ~ 660M
  attention: ~ 11M / layer
```

主体参数在 routed expert。

---

## Slide 8 · 激活参数

```
每 token 激活:
  shared:  25M
  top-8 routed: 8 × 2.6M = 20M
  attn: 11M
Total: ~ 60M / layer × 61 layer = 37B
```

总 671B 中 ~ 5.5% 激活。

---

## Slide 9 · 细粒度 ablation

DeepSeek-V2 论文 ablation：
```
8 expert top-2:    baseline
64 expert top-6:   +1.5 pp MMLU
256 expert top-8:  +2.5 pp
```

→ 细粒度持续提升，至少到 256。

---

## Slide 10 · 共享专家的妙处

```
without shared:    每 token 由 8 routed expert 拼
with shared:       通用 + 专业组合
                  ↓
              router 不必学"通用"，专心学专业
```

→ 训练稳定性 + 知识完整性。

---

## Slide 11 · 与 Mixtral 速度

```
Mixtral 8x7B: 13B 激活, 5090 ~ 25 tok/s
DeepSeek-V3: 37B 激活, 5090 OOM (需多卡)
```

DeepSeek-V3 671B 需 8× H100。

---

## Slide 12 · 训练成本

DeepSeek-V3 报告：~ 2.8M GPU 小时（H800）。

```
比 Llama-3 70B 训练：
  Llama: 7M H100 = 性能 X
  DeepSeek-V3: 2.8M H800 = 性能 X × 1.05
```

每 GPU 小时性价比 ~ 2.5× Llama。

---

## Slide 13 · 与 Mixtral 路由对比

| | Mixtral | DeepSeek |
|---|---------|----------|
| 粒度 | 8 大 | 256 小 |
| top_k | 2 | 8 |
| shared | 无 | 有 |
| aux | yes | Aux-Free (V3) |

---

## Slide 14 · DeepSeek-V2 / V3 layer 模板

```python
class DeepSeekBlock(nn.Module):
    def __init__(self, cfg):
        self.attn = MLA(cfg)        # MLA (专题 2 讲过)
        self.norm1 = RMSNorm(cfg.d)
        self.moe = DeepSeekMoE(cfg)  # 本课
        self.norm2 = RMSNorm(cfg.d)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x
```

---

## Slide 15 · 训练实务

```
aux loss α=0.001 (very small)
expert dropout 0.0
routing softmax temperature 1.0
```

V3 中 aux loss 已**完全去掉**（Aux-Free）。

---

## Slide 16 · 推理实务

```
4bit 加载: 671B → ~340GB → 仍需 4× H100
GGUF / AWQ 进一步: 200GB → 2× H100
```

---

## Slide 17 · 与 Mixtral 选型

```
中等任务 / 24GB 单卡 → Mixtral 8x7B 4bit
大规模 SAAS         → DeepSeek-V3
研究复现             → DeepSeek-V2-Lite (16B 缩水版)
```

---

## Slide 18 · DeepSeekMoE 缺点

```
1. 路由 W 大 (d × 256 = 1.8M)
2. expert offload 复杂（256 个）
3. 训练时 all-to-all 通信压力
```

DeepSeek-V3 专门优化 all-to-all。

---

## Slide 19 · 其他细粒度 MoE

```
JetMoE (2024):    8 expert, 但每 layer 不同 (动态)
OpenMoE:           open-source 细粒度复现
LLM4Compiler:      code 细粒度
```

→ 细粒度成为新趋势。

---

## Slide 20 · 与 MoE 数学

```
output = Σ G_i · expert_i + shared_expert
```

shared 项无 gate，相当于"权重 1"。

---

## Slide 21 · 训练注意

```
aux loss 调度:
  warmup 1k step → 启
  α = 0.001
```

V2 用 aux，V3 用 Aux-Free（下一讲）。

---

## Slide 22 · 课后思考

1. 256 expert vs 8 expert 的参数效率？
2. shared expert 的 d_ff 该多大？
3. 推理时 shared expert 占多少 FLOPs？
4. routed expert 256 → 1024 是否更好？

---

## 参考

- DeepSeek-V2 技术报告 2024.05
- DeepSeek-V3 技术报告 2024.12
- DeepSeekMoE 论文
