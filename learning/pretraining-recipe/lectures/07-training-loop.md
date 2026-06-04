# L07 · 训练 loop 与稳定性

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · 总骨架

```python
for step in range(max_step):
    x, y = get_batch(loader)
    with autocast(bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    lr = wsd_lr(step, max_step, base_lr)
    for g in optimizer.param_groups: g["lr"] = lr
```

## Slide 2 · gradient accumulation

```python
for micro in range(accum):
    x, y = next(loader)
    loss = model(x).cross_entropy(y) / accum
    loss.backward()
optimizer.step()
optimizer.zero_grad()
```

实际 batch = micro_batch × accum × n_gpu.

## Slide 3 · AdamW 配置

```python
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
    weight_decay=0.1,
)
```

## Slide 4 · param group: no decay for LN/bias

```python
decay = [p for n,p in model.named_parameters()
         if p.dim()>=2 and "norm" not in n]
no_decay = [p for n,p in model.named_parameters()
            if p.dim()<2 or "norm" in n]
opt = AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=6e-4)
```

## Slide 5 · 监控 loss & grad_norm

```python
if step % 100 == 0:
    print(f"step {step} loss {loss.item():.4f} "
          f"grad_norm {grad_norm:.3f} lr {lr:.2e}")
    wandb.log({"loss": loss.item(), "grad_norm": grad_norm})
```

## Slide 6 · loss spike 检测

```python
ema.update(loss.item())
if loss.item() > 3 * ema.ema:
    print(f"SPIKE at step {step}, skipping")
    opt.zero_grad()
    continue
```

## Slide 7 · ckpt

```python
if step % 500 == 0:
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "rng": torch.get_rng_state(),
        "shard_state": loader.state(),
    }, f"ckpt_{step}.pt")
```

## Slide 8 · resume

```python
ckpt = torch.load("ckpt_5000.pt")
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["opt"])
torch.set_rng_state(ckpt["rng"])
loader.restore(ckpt["shard_state"])
start_step = ckpt["step"] + 1
```

## Slide 9 · 训练 vs 评测开关

```python
model.train()  # dropout / BN 训
model.eval()   # 评测
torch.no_grad() in eval
```

## Slide 10 · 加 eval 间隙

```python
if step % 1000 == 0:
    model.eval()
    val_loss = run_eval()
    wandb.log({"val_loss": val_loss})
    model.train()
```

## Slide 11 · throughput log

```python
import time
t0 = time.time()
# ... train step ...
tok = batch * seq
dt = time.time() - t0
tok_per_s = tok / dt
mfu = 6 * N * tok_per_s / (1500e12)
```

## Slide 12 · 完整伪代码

```python
opt = AdamW(...)
ema = EmaLossTracker()
for step in range(max_step):
    for _ in range(accum):
        x, y = sample_batch(data, seq_len, micro_batch)
        with autocast(bfloat16):
            loss = compute_loss(model, x, y) / accum
        loss.backward()
    gn = clip_grad_norm(model, 1.0)
    if ema.is_spike(loss.item()*accum):
        opt.zero_grad(); continue
    lr = wsd_lr(step, max_step, base_lr)
    for g in opt.param_groups: g["lr"] = lr
    opt.step()
    opt.zero_grad()
    ema.update(loss.item()*accum)
    if step % 100 == 0: log(step, loss, gn, lr)
    if step % 500 == 0: save_ckpt(...)
```

## Slide 13 · 5090 5h 训练 270M 计算

```
batch 32 × seq 2048 = 65k tok / step
5090 ~ 35k tok/s → 1.9 step/s
500 step / batch = 16M tok
1B token / 16M = 60 hour
                ↑ realistic
```

减半: 0.5B token → 30h.

## Slide 14 · 总结

```
训练 loop = batch + amp + clip + step + log + spike guard + ckpt
本 capstone 30h 跑 0.5B token on 5090
```

## 参考
- nanoGPT train.py
- Phi-1 tech report
