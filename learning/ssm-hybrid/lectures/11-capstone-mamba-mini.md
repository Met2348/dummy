# L11 · Capstone — 130M mini-Mamba

> 24 slides | 70 min ⭐⭐⭐⭐⭐ 毕业

## Slide 1 · 目标

```
24 层 Mamba (d_model 512, d_state 16)
~ 130M 参数
训练: 专题 1 输出的 1B-token + Phi 合成
对照: 同算力 GPT-mini 80M
```

## Slide 2 · 架构

```python
class MiniMamba(nn.Module):
    def __init__(self, cfg):
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = ModuleList(MambaBlock(cfg.d_model, cfg.d_state)
                                for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
```

## Slide 3 · 与 GPT-mini 对比

| | GPT-mini 80M | mini-Mamba 130M |
|---|--------------|------------------|
| 架构 | Transformer | Mamba |
| KV cache 32k | ~ 600MB | ~ 1MB |
| ppl 4k | ≈ | ≈ (-0.1) |
| ppl 32k | OOM | OK |

## Slide 4 · 训练

```
batch 8, seq 1024
optimizer: AdamW (3e-4)
1 epoch 50M token → ~ 30 min on 5090
```

## Slide 5 · 实现

```python
class MiniMamba(nn.Module):
    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h) + h
        return self.lm_head(self.norm_f(h))
```

## Slide 6 · 训 100 step smoke

```python
m = MiniMamba(cfg)
opt = AdamW(m.parameters(), lr=3e-4)
for step in range(100):
    x, y = batch()
    loss = ce(m(x), y)
    loss.backward()
    opt.step()
```

应 loss 持续下降。

## Slide 7 · 长上下文外推

```
训 1k context → 推 4k context
ppl 不爆 → 成功
```

Mamba 天然可外推 (state 无 position 依赖)。

## Slide 8 · 退出条件

```
[ ] 100 step 训练 loss 下降
[ ] 1k→4k context perplexity 不爆
[ ] forward / backward 测试 PASS
```

## Slide 9-24 · 详细代码（src/mini_mamba.py）

## 参考
- Mamba paper
- 本系列前 10 lecture
