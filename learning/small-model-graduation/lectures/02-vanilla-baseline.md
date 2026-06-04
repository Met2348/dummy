# L02 · Vanilla baseline (ckpt A)

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · ckpt A 设计

```
GPT-2-small (124M)
- 12 layer
- 768 hidden
- 12 head
- absolute PE
- MHA (no GQA)
- GELU MLP
- post-norm 风格
- 50257 vocab GPT-2 BPE
```

## Slide 2 · 数据 (baseline)

```
TinyStories 100M token (50%)
OpenWebText 100M token (50%)
合计 200M token
```

不挑数据。

## Slide 3 · 训练 config

```
seq_len: 512
batch: 32 × 4 grad_accum = 128
lr: 6e-4 cosine
warmup: 1000 step
max_step: 200M / (128 × 512) ≈ 3000
~ 4h on 5090
```

## Slide 4 · 期望

```
val_loss: ~ 3.5
HellaSwag: ~ 35%
tinyMMLU: ~ 25% (近 random)
NIAH @ 2k: 0% (基本不能)
```

## Slide 5 · 模型代码 (vanilla)

```python
class VanillaGPT2(nn.Module):
    def __init__(self, vocab=50257, hidden=768, n_head=12, n_layer=12):
        self.embed = nn.Embedding(vocab, hidden)
        self.pos_embed = nn.Embedding(2048, hidden)
        self.blocks = nn.ModuleList([
            GPT2Block(hidden, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
    def forward(self, x):
        T = x.shape[1]
        h = self.embed(x) + self.pos_embed(torch.arange(T))
        for b in self.blocks: h = b(h)
        return self.lm_head(self.ln_f(h))
```

## Slide 6 · GPT-2 Block (post-norm)

```python
class GPT2Block(nn.Module):
    def __init__(self, d, h):
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(),
            nn.Linear(4*d, d),
        )
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x, is_causal=True)[0])
        x = self.ln2(x + self.mlp(x))
        return x
```

## Slide 7 · 对照点

```
比 ckpt C/E 缺少:
  - GQA → 4× KV cache
  - SwiGLU → 1pp benchmark
  - RoPE → 长 ctx 不可扩
  - 高质数据 → 数 pp 减
合计 ~ -8pp
```

## Slide 8 · 启动

```bash
python src/train_variant.py --variant A --max_step 3000
```

## Slide 9 · 实际 loss 曲线 (预期)

```
step    loss
0       11.0
500     5.5
1000    4.3
1500    3.8
2000    3.6
3000    3.5
```

## Slide 10 · 评测

```
$ python eval_one.py ckpt_A.pt
val_loss: 3.50
HellaSwag: 0.35
tinyMMLU: 0.25
```

## Slide 11 · 与 ckpt B 区别只在数据

```
A: TinyStories + OpenWebText (低质)
B: Cosmopedia + OpenWebText filtered (高质)
其余 architecture / training 一致
```

## Slide 12 · 总结

```
ckpt A 是参考点
不应追求高 benchmark
目的: 让后续 B/C/D/E 的提升可量化
```

## 参考
- GPT-2 (Radford 2019)
- nanoGPT
