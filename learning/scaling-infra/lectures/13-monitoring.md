# L13 · 监控 + 大规模训练故障

> 12 slides | 35 min ⭐⭐⭐

## Slide 1 · 必监控指标

```
loss / grad_norm / param_norm
lr / 实际 batch (after grad accum)
throughput tok/s
MFU (Model FLOPs Utilization)
GPU util / mem / power
NCCL 通信延迟
```

## Slide 2 · MFU

```
MFU = actual TFLOPS / theoretical TFLOPS
A100: 312 TFLOPS bf16
H100: 990 TFLOPS bf16
5090: ~1500 TFLOPS bf16

实测 30%+ 是好, 50%+ 是 SOTA (GPT-3 NVIDIA 报告)
```

## Slide 3 · TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter("runs/exp1")
w.add_scalar("loss", loss, step)
w.add_scalar("grad_norm", gn, step)
```

## Slide 4 · WandB

```python
import wandb
wandb.init(project="my-llm")
wandb.log({"loss": loss, "lr": lr, "step": step})
```

## Slide 5 · GPU 监控

```
nvidia-smi          # 实时
nvitop              # htop-like
nvidia-dcgm         # 集群级
prometheus + grafana # 持久化
```

## Slide 6 · NCCL 调试

```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

帮助 debug 多卡通信失败。

## Slide 7 · ckpt 策略

```
small model: every 1k step + last 3 keep
big model: every 500 step + async upload S3
RLHF: 每 phase 保存 (SFT/RM/PPO)
```

## Slide 8 · resume 训练

```
保存:
  model state_dict (FSDP 用 FULL_STATE_DICT)
  optimizer state_dict
  scheduler state_dict
  rng_state (torch.cuda.get_rng_state_all)
  step + epoch
```

## Slide 9 · 常见故障

```
NCCL timeout: 某卡慢 → all-reduce 卡住
OOM 偶发: activation peak > 平均
slow node: 一卡硬件慢 → 拖整集群
权重炸 (nan): 数值不稳 / lr 太大
```

## Slide 10 · OOM 排查

```
- 减 batch_size → 还 OOM? 减 ctx_len
- 开 gradient checkpoint → 还 OOM?
- ZeRO stage ↑
- 量化 (训练 LoRA)
```

## Slide 11 · slow node

```
prometheus 上看每卡 step time
> mean + 2σ → 标记
跳过该 batch 让其他卡继续
长期: 替换硬件
```

## Slide 12 · 总结

```
监控 = 训练成功保障
nvidia-smi/wandb/grafana 三件套
ckpt + resume 必备
NCCL 调试技能必备
```

## 参考
- WandB docs
- NVIDIA DCGM
- PyTorch FSDP ckpt
