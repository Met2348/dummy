# L11 · Emergence + Debug 

> 12 slides | 30 min ⭐⭐⭐⭐

## Slide 1 · Emergent abilities

```
Wei 2022: 某 N 之前 task accuracy ≈ random
        N 之后跳跃式上升
example: 3-digit add @ ~3B param
```

## Slide 2 · 例 (Wei 2022)

```
emoji 推理:
  < 7B: 0% acc
  >= 7B: 70% acc
  阶跃!
```

后 Schaeffer 2023 反驳: emergence 是 evaluation 引起的 illusion。

## Slide 3 · 训练异常: NaN / Inf

```
原因:
  - lr 太大
  - bf16 attention softmax 溢出
  - 数据有 outlier
  - init 太大
检测: 每 step 监控 loss/grad_norm
```

## Slide 4 · 处理 NaN

```python
if not torch.isfinite(loss):
    print(f"NaN at step {step}")
    opt.zero_grad(); continue
```

简单跳过 = ok 当低频时。

## Slide 5 · loss 飘升 (slow spike)

```
现象: ema loss 缓慢上升 (非 sudden spike)
原因: data distribution drift / lr 余热
处理: 减 lr, 检查数据
```

## Slide 6 · 训练异常: 完全发散

```
loss 跳到 10+ 永不回
通常 lr 配置错
退回 ckpt + lr / 2 重启
```

## Slide 7 · 调 batch / grad accum

```
显存紧 → 减 micro_batch + 加 grad_accum
保持 effective batch (例 0.5M token)
```

## Slide 8 · 调 ctx len

```
显存紧 → 减 seq_len
但 long ctx 训练能力差
trade-off
```

## Slide 9 · debugging 工具

```
print(loss, grad_norm) every 100 step
tensorboard / wandb 看 loss curve
torch.autograd.detect_anomaly() 找 NaN 源
nvidia-smi 看显存 / GPU util
```

## Slide 10 · sanity check 必要

```
1. forward 一次 → shape 对吗
2. backward 一次 → grad 不 NaN
3. 5 step → loss 下降
4. 100 step → loss 持续下降
5. 1000 step → log 写 ckpt
```

## Slide 11 · 常见 bug

```
- attention mask shape 错 → 跨 doc attend
- LR / batch 不匹配
- 数据未 shuffle → 学习只见同 source
- ckpt RNG state 没保存 → resume 不一致
- LN gamma 没 init 1
```

## Slide 12 · 总结

```
emergence 真假之争未定
debug = sanity check + log + 备 ckpt
"训不动" 通常是数据 / lr / batch 问题
```

## 参考
- Wei 2022 Emergent abilities
- Schaeffer 2023 反驳
