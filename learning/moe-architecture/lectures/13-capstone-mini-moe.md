# L13 · Capstone — Mini-MoE (4-expert)

> 28 slides | 80 min | MoE Architecture 第 13 讲 ⭐⭐⭐⭐⭐ 毕业作品

> 把专题 2 GPT-mini 的 MLP 替换为 4-expert MoE + Aux-Free

---

## 学习目标

1. 集成 GPT-mini base + 4-expert MoE
2. 用 Aux-Free 路由
3. 对比 dense 80M vs MoE 80M/120M
4. 输出 ckpt 供未来专题（如专题 8）参考

---

## Slide 1 · 目标 architecture

```
n_layer = 12 (与 GPT-mini 一致)
hidden = 768
n_head = 12 (GQA, n_kv=2)
n_experts = 4
top_k = 2
shared expert: 1
routed expert: 4
routing: Aux-Free
```

---

## Slide 2 · 参数估算

```
dense GPT-mini:   80M params
mini-MoE:
  base attn: 同
  routed FFN: 4 × small (vs 1 large)
  shared FFN: 1 small
↓
total: ~110M, activated: ~ 85M
```

---

## Slide 3 · MLP 替换

```python
class MoEBlock(Block):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 替换 self.mlp 为 MoE
        self.mlp = DeepSeekMoELayer(
            d_model=cfg.d_model,
            n_routed=4,
            n_shared=1,
            top_k=2,
        )
```

attention + norm 沿用 GPT-mini。

---

## Slide 4 · 用 Aux-Free 路由

```python
class MiniMoEBlock(Block):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate = AuxFreeRouter(cfg.d_model, n_experts=4, top_k=2)
        self.routed = ModuleList(SwiGLUMLP(cfg.d_model, d_ff_small) for _ in range(4))
        self.shared = SwiGLUMLP(cfg.d_model, d_ff_shared)
    def forward(self, x):
        # ...
        out = self.shared(x)
        # routed + Aux-Free
        gates, idx, _ = self.gate(x.view(-1, d))
        for e in range(4):
            mask = (idx == e)
            if mask.any():
                ...
        return out
```

---

## Slide 5 · 训练数据

```
同专题 2 GPT-mini 的 mock data
batch 8, seq 64, vocab 1024
```

教学规模玩具。

---

## Slide 6 · 训练

```
optimizer: AdamW
lr: 3e-4
batch: 8 × 64 = 512 tok/batch
steps: 100 smoke / 1000 real
```

---

## Slide 7 · 与 dense 80M 对比指标

```
dense 80M:       final loss ~ 3.0
mini-MoE 110M:   final loss ~ 2.7 (-0.3)
```

应至少有 0.3+ loss 优势。

---

## Slide 8 · 路由热图

```
横轴: token id (vocab 1024)
纵轴: top-1 expert id (0-3)
颜色: 频次
```

可视化哪些 token 倾向哪些 expert。

---

## Slide 9 · 文件结构

```
src/mini_moe.py              # 集成 GPT-mini + 4-expert
src/capstone_train_mini_moe.py
src/tests/test_mini_moe_vs_dense.py
notebooks/13-capstone.ipynb
```

---

## Slide 10 · 集成代码片段

```python
class MiniMoE(nn.Module):
    def __init__(self, cfg):
        self.tok_embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(
            MoEBlock(cfg) if (i+1) % 2 == 0 else DenseBlock(cfg)
            for i in range(cfg.n_layer)
        )
        # 每 2 层一个 MoE，每 2 层一个 dense (类 Switch)
        self.norm_f = RMSNorm(cfg.d)
        self.lm_head = nn.Linear(cfg.d, cfg.vocab, bias=False)
```

---

## Slide 11 · 训练循环

```python
for step in range(steps):
    x, y = batch()
    logits = model(x)
    loss = ce(logits.view(-1, V), y.view(-1))
    # 注意：用 Aux-Free，无 aux loss！
    loss.backward()
    opt.step()
```

---

## Slide 12 · MoE 训练注意

```
1. expert init 不能完全一样 (否则路由失效)
2. 监控 expert utilization
3. bias 范围在 ±0.5 内（正常）
4. lr 3e-4 同 dense
```

---

## Slide 13 · 验证

```python
def test_capstone_moe_vs_dense():
    dense = GPTMini(cfg)
    moe = MiniMoE(cfg)
    # 训 500 step
    train(dense), train(moe)
    dense_loss = eval(dense)
    moe_loss = eval(moe)
    assert moe_loss < dense_loss * 0.95  # +5% 改善
```

---

## Slide 14 · 路由热图代码

```python
def routing_heatmap(model, dataloader):
    counts = zeros(vocab, n_experts)
    for x, _ in dataloader:
        ...
        counts[token_id, expert_id] += 1
    return counts
```

绘制 imshow。

---

## Slide 15 · expert specialization 观察

实际训练后：
```
expert 0:  常见词 (the, a, is)
expert 1:  数字 / 数学相关
expert 2:  长 token / 代码
expert 3:  其他
```

某种程度上 expert 学到分工。

---

## Slide 16 · Capstone 退出条件

```
[ ] 训练 100+ step 不崩塌
[ ] expert utilization 平衡 (max/min < 2)
[ ] val loss < dense × 0.95
[ ] 路由热图可视化
[ ] tests PASS
```

---

## Slide 17 · 与 Mixtral 8x7B 关系

```
本 capstone 4 expert × 80M  (玩具)
Mixtral 8 expert × 7B       (实战)
↓
缩小 100×，原理完全一致
```

---

## Slide 18 · 显存

```
mini-MoE 110M:
  fp32: 440MB
  +opt state: 880MB
  +grad: 440MB
Total ~ 2GB on 5090 (绰绰有余)
```

---

## Slide 19 · 与专题 1 接口

可加载专题 1 输出的 tokenizer：

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="../data-curation/.../m.model")
```

---

## Slide 20 · 与专题 7 接口

mini-MoE 是预热，专题 7 真训 270M Phi-tiny 可选择 MoE 替换。

---

## Slide 21 · 提升训练性能

```
torch.compile(model)  # 编译
混合精度 bf16          # 显存省
gradient checkpointing  # MoE 用
```

5090 上 100M 不需，但留口子。

---

## Slide 22 · 推理 demo

```python
model.eval()
x = torch.randint(0, vocab, (1, 8))
out = model.generate(x, max_new=20)
```

10-20 token 续写验证。

---

## Slide 23 · "Aux-Free vs aux" 对照

```
跑 2 个版本:
  aux loss:    final loss X1
  Aux-Free:    final loss X2
比较 X1, X2
```

实测 X2 ≤ X1（Aux-Free 略好或持平）。

---

## Slide 24 · 文件清单

```
mini_moe.py:                     ~ 180 lines
capstone_train_mini_moe.py:      ~ 80 lines
test_mini_moe_vs_dense.py:        ~ 50 lines
13-capstone.ipynb
```

---

## Slide 25 · 总成本

```
GPU: 5090 24GB
时间: 30 min smoke + 4h 真训 (可选)
ckpt: ~ 500MB (含 optimizer state)
```

---

## Slide 26 · 与其他专题"汇总"

至此（专题 1+2+3）完成：
```
专题 1 data-curation   →  corpus
专题 2 transformer-deep →  GPT-mini 80M
专题 3 moe-architecture →  mini-MoE 110M (本课)
                          ↓
           专题 4 ssm-hybrid (Mamba 替换 attn)
           专题 5-7 (long-context + infra + recipe)
           专题 8 graduation (五部曲)
```

---

## Slide 27 · 工程价值

```
学完本专题：
  ✓ 能从头实现 MoE
  ✓ 选 routing 算法 (top-k + Aux-Free)
  ✓ 训练 + 推理优化
  ✓ 集成到 transformer 整 pipeline
```

→ 已能复现 Mixtral / DeepSeek-V3 风格 MoE。

---

## Slide 28 · 课后思考

1. 4 expert 比 8 expert 性能损失多少？
2. shared expert d_ff 该多大？
3. Aux-Free 在 4 expert 是否过 overkill？
4. 推理时是否 batch 多 token 一起 routing？

---

## 参考

- DeepSeek-V3 报告
- Mixtral paper
- 本系列前 12 lecture
