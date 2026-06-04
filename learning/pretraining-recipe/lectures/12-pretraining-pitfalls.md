# L12 · 陷阱合集

> 10 slides | 25 min ⭐⭐⭐⭐

## Slide 1 · 数据陷阱

```
- 重复 doc (CommonCrawl 一份 PDF 多 URL)
- 评测数据泄漏 (MMLU 出现在 web)
- license 违规 (Books3 / Pile)
- 私人信息 (PII)
```

## Slide 2 · 模型陷阱

```
- 未 tie embedding → 参数浪费
- LN 位置错 (post-norm 不稳)
- ReLU 替 SwiGLU → 性能差 2pp
- 未 init residual → 深网络炸
```

## Slide 3 · 训练陷阱

```
- lr 太大 → spike
- batch 太大 → 收敛慢
- warmup 太短 → loss 早期跳
- ckpt 不存 RNG → resume 数据顺序变
```

## Slide 4 · 评测陷阱

```
- 数据泄漏 (MMLU 在 train set)
- prompt 格式不一致 → score 差
- few-shot 选不同例 → 大波动
- 误用 perplexity (与 accuracy 不直接相关)
```

## Slide 5 · scaling 陷阱

```
- 信 Kaplan, 实际 Chinchilla → token 不够
- 信 Chinchilla, 实际 over-train → token 不够
- 数据质量差 → 多 token 也无用
```

## Slide 6 · "训练损失低 ≠ 模型好"

```
val_loss 1.8 不一定比 2.1 好
benchmark 才是 ground truth
```

## Slide 7 · 系统陷阱

```
- shard 顺序固定 (resume 后跳 shard) → 数据偏移
- NCCL timeout 假死
- IB 故障 → all-reduce 慢
- 多卡 RNG seed 没同步
```

## Slide 8 · 别忘了 EMA

```
某些 paper (DeepSeek-V3) 用 weight EMA
ema_weights = 0.9 * ema + 0.1 * cur
推理时用 EMA
loss 更平滑
```

## Slide 9 · 注意 vocab 倍数

```
vocab 应 64 / 128 对齐 (GPU friendly)
50257 是 GPT-2 倒霉数, padding 到 50304
影响速度 5-10%
```

## Slide 10 · 总结

```
预训练雷区多
监控 + sanity check + ckpt 是三件套
"看起来没问题" ≠ 没问题
```

## 参考
- Karpathy nanoGPT notes
- Llama-3 tech report (lessons)
