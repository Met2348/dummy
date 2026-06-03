# L14 · Capstone — 80M GPT-mini

> 32 slides | 90 min | Transformer Deep 第 14 讲 ⭐⭐⭐⭐⭐ 毕业作品

> RoPE + RMSNorm + GQA + SwiGLU 的 80M model

---

## 学习目标

1. 集成前 13 讲组件成完整 80M GPT
2. 写 forward / backward / KV cache 增量解码
3. 训 1 epoch 验证 ppl 下降
4. 输出 ckpt 供专题 8 毕业作品复用

---

## Slide 1 · 目标 architecture

```
n_layer = 12
hidden  = 768
n_head  = 12 (Q)
n_kv    = 2  (GQA group=6)
d_head  = 64
d_ff    = 2048 (SwiGLU 2.67 × hidden)
vocab   = 8192 (玩具)
context = 1024
```

参数估算 ≈ 80M。

---

## Slide 2 · GPT-mini Block

```
class Block:
    def __init__(self, cfg):
        self.norm1 = RMSNorm(d_model)
        self.attn  = GQA(d_model, n_head, n_kv_head)
        self.norm2 = RMSNorm(d_model)
        self.mlp   = SwiGLUMLP(d_model, d_ff)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x
```

Pre-LN residual style (Llama 风格)。

---

## Slide 3 · 完整 model

```python
class GPTMini:
    def __init__(self, cfg):
        self.tok_embed = Embedding(vocab, d_model)
        self.blocks = ModuleList(Block(cfg) for _ in range(n_layer))
        self.norm_f = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab, bias=False)
        # weight tie embedding & lm_head
        self.lm_head.weight = self.tok_embed.weight
    def forward(self, x):
        h = self.tok_embed(x)
        mask = causal_mask(x.shape[1])
        for blk in self.blocks: h = blk(h, mask)
        return self.lm_head(self.norm_f(h))
```

---

## Slide 4 · RoPE 集成进 GQA

```python
class GQAWithRoPE(GQA):
    def __init__(self, cfg):
        super().__init__(...)
        self.rope = RoPE(self.d_head, base=10000)
    def forward(self, x):
        q, k, v = ... 
        q, k = self.rope(q, k)
        # attention as usual
```

---

## Slide 5 · weight tying

```
embedding.weight = lm_head.weight
```

省 64M 参数（vocab × d）。GPT-2 起的标配。

---

## Slide 6 · init 策略

```
Linear.weight ~ N(0, 0.02)
RMSNorm.gamma ~ 1
embedding     ~ N(0, 0.02)
```

GPT-2 风格 init。已在 src/common.py 实现。

---

## Slide 7 · 训练数据

```
专题 1 输出: corpus.jsonl.gz (50M token + 8k SP tokenizer)
↓
read jsonl → SP encode → input_ids 序列
↓
slice 1024-token chunk
↓
DataLoader (batch=16)
```

教学版用 mock 数据，真正训用专题 1 输出。

---

## Slide 8 · loss

```python
def forward_loss(model, x, y):
    logits = model(x)            # (b, t, vocab)
    return F.cross_entropy(
        logits.view(-1, vocab),
        y.view(-1),
    )
```

teacher forcing: x[:-1], y[1:]。

---

## Slide 9 · optimizer

```
AdamW
lr  = 3e-4
betas = (0.9, 0.95)
weight_decay = 0.1
warmup = 1000 step
cosine decay to 3e-5
```

Llama-2 / GPT-3 风格。

---

## Slide 10 · 训练循环（简化）

```python
for step, (x, y) in enumerate(loader):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    loss.backward()
    grad_clip(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## Slide 11 · KV cache 增量解码

```python
def generate(model, prefix, max_new=50):
    cache = []
    # prefill
    logits, cache = model(prefix, use_cache=True)
    out = [prefix[-1].item()]
    for _ in range(max_new):
        next_tok = sample(logits[:, -1])
        out.append(next_tok)
        x_in = tensor([next_tok])
        logits, cache = model(x_in, cache=cache)
    return out
```

KV cache 让 incremental decoding O(t)。

---

## Slide 12 · 显存预算

```
80M params × 4 byte (fp32) = 320 MB
+ optimizer state (AdamW 2×) = 640 MB
+ gradient = 320 MB
+ activations (batch 16 × 1024 × 768 × 12 layer)
≈ 1.5 GB on 5090
```

完全可以单卡训。

---

## Slide 13 · 训练耗时估算

```
50M token × 1 epoch
batch 16, seq 1024 → 25k step per epoch
5090 ~ 50 step/s (含 attention)
→ ~ 10 分钟 / epoch
```

3 epoch ~ 30 分钟 → val ppl < 30 可达。

---

## Slide 14 · 验证

```
val_loader: 5% held-out
val_ppl = exp(val_loss)
```

目标：val ppl < 30 (玩具数据集)。

---

## Slide 15 · 与专题 8 对接

```
本 capstone ckpt → 专题 8 五部曲毕业作品 ckpt 之一
                 ↓
              "Vanilla GPT-2" baseline
```

→ 五部曲完整链路的第一根支柱。

---

## Slide 16 · 完整 src 文件

```
src/gpt_mini.py            # 80M model
src/capstone_train.py      # 训练 main
src/tests/test_gpt_mini_forward.py
src/tests/test_kv_cache.py
```

---

## Slide 17 · 实现 gpt_mini.py 提示

主要组件已有：
```
common.causal_mask
rmsnorm.RMSNorm
gqa.GQA (扩展加 RoPE)
swiglu.SwiGLUMLP
rope.RoPE
```

集成进 Block + GPT 即可，~150 行。

---

## Slide 18 · 测试 — forward

```python
def test_forward_shape():
    cfg = GPTMiniConfig(vocab=512, n_layer=2, n_head=4, n_kv=2,
                        d_model=64, max_seq=128)
    model = GPTMini(cfg)
    x = torch.randint(0, 512, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, 512)
```

---

## Slide 19 · 测试 — backward

```python
def test_backward_flows():
    ...
    y.sum().backward()
    for p in model.parameters():
        assert p.grad is not None
```

---

## Slide 20 · 测试 — KV cache

```python
def test_kv_cache_consistency():
    """无 cache full forward vs cache incremental 应一致."""
    x = randint(...)
    out_full = model(x)
    # cache 增量: 每 token 单独 forward
    cache = None
    out_cached = []
    for t in range(x.shape[1]):
        o, cache = model(x[:, t:t+1], cache=cache)
        out_cached.append(o)
    out_cached = cat(out_cached, dim=1)
    diff = (out_full - out_cached).abs().max()
    assert diff < 1e-4
```

---

## Slide 21 · gradient checkpointing

```python
from torch.utils.checkpoint import checkpoint
def block_forward(x, mask):
    return checkpoint(self.blocks[i], x, mask)
```

省显存换算力。对 80M 不需要，对 7B+ 必需。

---

## Slide 22 · 数据 loader 增量加载

```python
class JsonlDataset(IterableDataset):
    def __init__(self, path, seq_len, tokenizer):
        self.path = path
        ...
    def __iter__(self):
        for doc in read_jsonl_gz(self.path):
            ids = tokenizer.encode(doc["text"])
            for i in range(0, len(ids) - seq_len, seq_len):
                yield ids[i:i+seq_len], ids[i+1:i+seq_len+1]
```

流式读取，省内存。

---

## Slide 23 · capstone 退出条件

```
[ ] forward / backward smoke pass
[ ] KV cache 一致性 pass
[ ] 训 1 epoch 完成
[ ] val ppl < 30 (玩具数据)
```

---

## Slide 24 · 与其他专题的"接口"

```
data-curation → corpus.jsonl + tokenizer.model
   ↓
本 capstone  → 80M_gpt-mini.ckpt + 训练 log
   ↓
专题 7 pretraining-recipe → 真训 270M
   ↓
专题 8 graduation → 五部曲毕业 (本 ckpt 是 baseline 之一)
```

---

## Slide 25 · 与 Llama-3 8B 的距离

```
80M  → 8B  ≈ 100×
12 → 32 layer ≈ 3×
768 → 4096   ≈ 5×
50M → 15T    ≈ 300000× token
```

→ 1 个超级集群、6 个月、几百万美金。

---

## Slide 26 · 80M 的真实能力

```
val ppl ~ 30 (普通文本)
能：补完简单句子
不能：复杂推理 / 代码 / 数学
       (太小了)
```

但 forward / backward / KV cache 全栈学完。

---

## Slide 27 · 加载验证

```python
model = GPTMini.from_checkpoint("...")
text = "The cat is "
ids = tokenizer.encode(text)
out = model.generate(ids, max_new=20)
print(tokenizer.decode(out))
# "The cat is sleeping on the mat ..."
```

至少能续写连贯短文。

---

## Slide 28 · 性能数字

```
forward (batch=8, seq=1024): ~ 30 ms on 5090
                              ~ 100 ms on 4090
KV cache decode/token: ~ 5 ms
```

---

## Slide 29 · model.json (配置)

```json
{
  "vocab_size": 8192,
  "n_layer": 12,
  "n_head": 12,
  "n_kv_head": 2,
  "d_model": 768,
  "d_ff": 2048,
  "max_seq": 1024,
  "rope_base": 10000.0,
  "tie_embeddings": true,
  "norm_eps": 1e-6
}
```

---

## Slide 30 · 与 HF transformers 兼容

如果想兼容 HF API：

```python
from transformers import PreTrainedModel
class GPTMiniHF(PreTrainedModel):
    config_class = GPTMiniConfig
    ...
```

可选，不必做。

---

## Slide 31 · README 应有

```
- architecture 说明
- 训练命令
- 配置文件
- forward / KV cache 用法
- 与专题 1 / 8 的接口
```

---

## Slide 32 · 课后思考

1. weight tying 对 ppl 影响？
2. RoPE 与 KV cache 集成时旧 K 要不要 re-rotate？
3. 12 层 vs 24 层在同 80M 总参数下哪个 better？
4. SwiGLU d_ff = 2048 与 GELU d_ff = 3072 总参数等价吗？

---

## 参考

- 本系列前 13 lecture
- nanoGPT (Karpathy)
- Llama-3 architecture
