# L07 · Aux-Loss-Free Routing ⭐⭐⭐⭐⭐

> 32 slides | 90 min | MoE Architecture 第 7 讲 ⭐⭐⭐⭐⭐ 必修

> DeepSeek-V3 (2024.12) 最大算法创新 / 2026 起主流

---

## 学习目标

1. 理解 Aux-Loss-Free 的核心思想
2. 推导 偏置项更新算法
3. 与传统 aux loss 对比
4. 写出 50 行实现

---

## Slide 1 · 问题：aux loss 的副作用

```
total_loss = ce_loss + α · aux_loss
```

aux loss 的副作用：
1. **干扰 ce_loss 优化**（α 难调）
2. 早期训练阶段冲突
3. router 偏向"均衡"而非"最优"

---

## Slide 2 · Aux-Loss-Free 思路

**不用 aux loss，而用 bias**：

```
router_score_i = softmax(logits)_i + bias_i
top_k = argmax of routed scores
```

只用 bias_i 控制 expert 负载，不进梯度。

---

## Slide 3 · bias 更新公式 ⭐⭐⭐⭐⭐

**每步**根据上一 step expert 负载更新 bias：

```
if expert_i 接收 token 数 > average:
    bias_i -= γ   # 推开 (降低被选概率)
if expert_i 接收 token 数 < average:
    bias_i += γ   # 拉近
```

`γ = update_rate = 1e-3`。

---

## Slide 4 · 关键 detail — bias 不进 gate

```python
# 用于排序
sort_scores = softmax_scores + bias
top_k_idx = sort_scores.topk(k)

# 用于加权（不含 bias）
top_k_gates = softmax_scores[top_k_idx]
```

bias 只影响"谁被选"，不影响"权重多少"。

---

## Slide 5 · 完整算法

```python
class AuxFreeRouter(nn.Module):
    def __init__(self, d, n_expert, top_k=8, update_rate=1e-3):
        self.W = Linear(d, n_expert, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_expert), requires_grad=False)
        self.update_rate = update_rate
    
    def forward(self, x):
        logits = self.W(x)
        scores = F.softmax(logits, dim=-1)
        sort_scores = scores + self.bias              # bias 影响排序
        top_k_gates, top_k_idx = self._topk(sort_scores, scores)
        if self.training:
            self._update_bias(top_k_idx)
        return top_k_gates, top_k_idx, None           # no aux loss!
```

---

## Slide 6 · bias 更新细节

```python
def _update_bias(self, top_k_idx):
    n_tok = top_k_idx.shape[0]
    load = bincount(top_k_idx.flatten(), minlength=n_expert).float()
    target = top_k * n_tok / n_expert
    # 高于 target 的 expert 降，低的升
    delta = torch.zeros_like(self.bias)
    delta[load > target] = -self.update_rate
    delta[load < target] = +self.update_rate
    self.bias.data.add_(delta)
```

---

## Slide 7 · 与 aux loss 对比

| | aux loss | Aux-Free |
|---|---------|----------|
| 优化路径 | 进梯度 | 进 bias（无梯度）|
| 干扰 ce | yes | **no** |
| 实现复杂 | 中 | 简 |
| 调参 | α 难调 | update_rate 稳 |
| DeepSeek 性能 | -0.5pp | baseline ⭐ |

---

## Slide 8 · 为什么 bias 不影响 gate

```
top_k_gates = softmax_scores (raw, no bias)
output = Σ gates · expert_out
```

bias 只决定 top-k 选谁；选完后用 raw softmax 加权。

→ 模型最终输出与"无 bias 训练"等价（同 router weights 下）。

---

## Slide 9 · 数学理解

```
最优配置: 各 expert 收到 ≈ k × n_tok / n 个 token
↓
expert_i 收到 c_i = ?
↓
bias_i 修正使 c_i 收敛到 ideal
```

类似 PID 控制器，但只 P 项（无 I, D）。

---

## Slide 10 · update_rate 选择

```
γ = 1e-3 (论文)
↓
每 step bias 改 0.001
→ 1000 step ~ ±1.0 bias 范围
```

太大 → bias 震荡；太小 → 收敛慢。1e-3 是甜点。

---

## Slide 11 · 训练过程可视化

```
step 1:     bias = [0, 0, 0, 0]    load: [40, 10, 10, 10]
step 100:   bias = [-0.1, +0.03, +0.03, +0.03]  load: [25, 25, 25, 25]
step 1000:  bias = [-0.3, +0.1, +0.1, +0.1]
```

→ 自动收敛到平衡。

---

## Slide 12 · 与传统 aux loss 的精确对比

DeepSeek-V3 ablation：

```
DeepSeek-V2 (aux loss):   MMLU 78.5
DeepSeek-V3 (Aux-Free):   MMLU 79.0  (+0.5pp)
```

→ 看似小，但**完全免费**（无额外计算）。

---

## Slide 13 · expert imbalance 检测

实际负载是否均衡：

```python
load = bincount(top_k_idx.flatten())
max_load / min_load    # < 2.0 视为 OK
```

DeepSeek-V3 训练后 ratio ~ 1.3，非常平衡。

---

## Slide 14 · 与 shared expert 配合

DeepSeek-V3：
```
shared expert: 始终参与（无 bias）
routed expert: Aux-Free bias 路由
```

shared 学通用，routed 用 Aux-Free 均衡 + 学专业。

---

## Slide 15 · 推理时 bias 怎么处理

```
推理: 用训练末态的 bias（不变）
保存: 仅 W (gate matrix) + bias（小 vector）
```

bias 是 expert 数大小（256 维），存储微小。

---

## Slide 16 · 与 z-loss 的关系

z-loss 防 logits 爆炸：

```
z_loss = (log(Σ exp(logits)))²
```

Aux-Free 仍可与 z-loss 共用（不冲突）：
```
total = ce + 1e-3 * z_loss   # 无 aux loss!
```

---

## Slide 17 · 实务超参（论文）

```
update_rate = 1e-3        ⭐
bias_init = 0
z_loss α = 1e-3 (可选)
```

DeepSeek-V3 严格用这套，不做改动。

---

## Slide 18 · 失败模式 1 — update_rate 过大

```
γ = 0.01:
  bias 震荡 → load 不稳
  ↓
  expert 0 步步偏移 → 仍偏 router
```

→ 实测 γ > 5e-3 后失败。

---

## Slide 19 · 失败模式 2 — bias init ≠ 0

```
bias_init = [1, 0, 0, 0]:
  step 0: expert 0 偏向, bias 0 高
  ↓
  系统需 1000+ step 才纠回
```

→ 必须 init = 0。

---

## Slide 20 · 与 Mixtral 接口对比

| | Mixtral router | Aux-Free router |
|---|---------------|-----------------|
| forward 输入 | x | x |
| forward 输出 | gates, idx, aux | gates, idx, None |
| backward | ce + α·aux | ce only |
| bias state | 无 | required |

接口替换简单（只需删 aux）。

---

## Slide 21 · 适用范围

```
适用:
  - DeepSeek-V3 sized MoE (256+ expert)
  - GShard / Switch / Mixtral 风格
不适用:
  - expert-choice (本身均衡)
  - extreme expert (>1024)
```

---

## Slide 22 · 与 batch size 关系

```
batch 大 → bias 更新更准
batch 小 → 更新噪声大
```

DeepSeek-V3 训用 4096 batch，bias 极稳。

---

## Slide 23 · "为什么 simple is better"

Aux-Free 简单胜在：
1. 一个超参（update_rate）
2. 不进 gradient
3. 与 ce 解耦
4. 实现 < 50 行

ML 历史上"少即是多"的又一案例。

---

## Slide 24 · 与未来 routing 方向

Aux-Free 是 routing 的"现代解"。未来可能：
- bias 用更智能更新（动量 / Adam-like）
- bias 跨层共享
- bias 与 capacity factor 联动

---

## Slide 25 · 代码 src/aux_loss_free.py

```python
class AuxFreeRouter(nn.Module):
    def __init__(self, d, n_expert, top_k=2, update_rate=1e-3):
        super().__init__()
        self.W = nn.Linear(d, n_expert, bias=False)
        self.register_buffer("bias", torch.zeros(n_expert))
        self.n_expert = n_expert
        self.top_k = top_k
        self.update_rate = update_rate

    def forward(self, x):
        logits = self.W(x)
        scores = F.softmax(logits, dim=-1)
        sort_scores = scores + self.bias
        _, top_k_idx = sort_scores.topk(self.top_k, dim=-1)
        top_k_gates = scores.gather(-1, top_k_idx)
        top_k_gates = top_k_gates / top_k_gates.sum(-1, keepdim=True)
        if self.training:
            with torch.no_grad():
                load = torch.bincount(top_k_idx.flatten(),
                                      minlength=self.n_expert).float()
                target = self.top_k * x.shape[0] / self.n_expert
                delta = torch.where(load > target,
                                    -self.update_rate,
                                    torch.where(load < target,
                                                 self.update_rate, 0.0))
                self.bias.add_(delta)
        return top_k_gates, top_k_idx
```

---

## Slide 26 · 训练监控

实务监控：
```
1. bias 值范围（应稳定 ±0.5 内）
2. expert load max/min ratio（应 < 2.0）
3. update_rate 自动调（如检测震荡）
```

---

## Slide 27 · 与其他 MoE 方法的兼容

```
GShard router    →  替换路由 → Aux-Free
Switch top-1     →  替换路由 → Aux-Free
Mixtral top-2    →  替换路由 → Aux-Free
DeepSeek-V2     →  替换路由 → Aux-Free
```

→ 通用 plug-in。

---

## Slide 28 · 已被采用

```
2024.12 DeepSeek-V3 首推
2025.01 各家复现实现 (open-r1 / TinyZero)
2025 后期成为新主流
```

---

## Slide 29 · "为什么不早出"

理论上 PID 思路并不新。但：
1. 需要充分训练规模才能稳定
2. ML 社区"看 loss 调超参"惯性
3. 实证才能说服人

DeepSeek-V3 凭借 14.8T token 规模证明 Aux-Free 工作。

---

## Slide 30 · 与 capacity factor 协作

```
Aux-Free 已经强迫均衡，capacity 可放宽:
  factor = 1.25 → 1.5 (训) / ∞ (推)
```

由 Aux-Free 自我管理，capacity 兜底。

---

## Slide 31 · 缺陷与未解

```
- 静态 update_rate (不随 batch / step 变)
- 跨 layer 独立 (没用 cross-layer 信息)
- 长序列下 bias 累积
```

→ 未来 routing 改进的方向。

---

## Slide 32 · 课后思考

1. update_rate 改自适应（如 Adam）会更好吗？
2. bias 跨 layer 共享是否合理？
3. Aux-Free 在 expert-choice 上能用吗？
4. 推理时是否 bias 需要 finetune？

---

## 参考

- DeepSeek-V3 技术报告 2024.12 (Appendix B 算法细节)
- DeepSeekMoE V2 2024.05
- 各开源复现 (Open-R1 2025.01)
