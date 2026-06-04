# L16 · Capstone — 训 Phi-tiny 270M

> 20 slides | 70 min ⭐⭐⭐⭐⭐

## Slide 1 · 目标

```
模型: Phi-tiny 270M (本 topic L06)
基座: TinyStoriesV2 + Cosmopedia 子集
token: 0.5B (5090 24G 24-36h)
评测: tinyHellaSwag + val_loss
```

## Slide 2 · 数据准备

```
HF datasets:
  HuggingFaceTB/cosmopedia-100k (100k 合成教科书)
  roneneldan/TinyStoriesV2-GPT4-train (2.7M 短故事)
混合: 70% Cosmopedia + 30% TinyStories
tokenize GPT-2 BPE
合计 token: ~ 500M
shard 大小: 100MB / shard
```

## Slide 3 · shard 数

```
500M token × 2 byte = 1 GB total
100 MB / shard = ~ 10 shards
存 ./data/shards/shard_NNN.bin
```

## Slide 4 · 训练 config

```python
class TrainConfig:
    seq_len = 1024
    micro_batch = 16
    grad_accum = 8        # effective batch 128
    max_step = 4000
    base_lr = 6e-4
    weight_decay = 0.1
    grad_clip = 1.0
    schedule = "wsd"
    warmup_pct = 0.05
    decay_pct = 0.2
```

effective tokens / step = 128 × 1024 = 131k
total token = 4000 × 131k = 524M ✓

## Slide 5 · 显存账本 (5090 24G)

```
weights bf16: 270 × 2 = 540 MB
grad bf16: 540 MB
AdamW fp32 state: 270 × 8 = 2.16 GB
activations: ~ 4-6 GB (seq 1024, micro 16, grad ckpt)
peak: ~ 10-12 GB ✓ 余 12+ GB
```

## Slide 6 · 时长估算

```
tok/s 实测 (5090, 270M, bf16, no fa-cuda): ~ 25k
131k / step → 5.2 s / step
4000 step → 21000 s ≈ 5.8 h
                    ↑ 还好
```

注: 真实需 30h+ (因 activation memory pressure 减慢).

## Slide 7 · 启动命令

```bash
python src/capstone_train.py \
  --model phi_tiny \
  --max_step 4000 \
  --micro_batch 16 \
  --grad_accum 8 \
  --base_lr 6e-4 \
  --ckpt_dir ./ckpt
```

## Slide 8 · 实验日志期望

```
step 0:    loss 11.0  (random init)
step 100:  loss 7.5   (vocab 学到)
step 500:  loss 4.5
step 1000: loss 3.8
step 2000: loss 3.3
step 3500: loss 3.0   (annealing 开始)
step 4000: loss 2.8   (final)
```

## Slide 9 · val_loss 监控

```
每 500 step 跑 50 batch val
expected: train_loss ≈ val_loss (玩具规模无 overfit)
```

## Slide 10 · 评测 tinyHellaSwag

```
3 题 → 期望:
  random: 0.33
  step 0: 0.33
  step 4000: 0.5-0.7
```

## Slide 11 · 生成示例

```python
prompts = ["Once upon a time, ", "The chef cracked an egg"]
out = generate(model, prompts, max_new_tokens=100)
# expected: 流畅简单英语, 不必正确事实
```

## Slide 12 · 完整 capstone script 结构

```python
def main():
    # 1. data
    prepare_shards(...)
    loader = make_loader(...)
    # 2. model
    config = PhiTinyConfig()
    model = PhiTiny(config).cuda().bfloat16()
    # 3. train
    train(model, loader, max_step=4000, ...)
    # 4. eval
    val_loss = validation_loss(...)
    hs = run_tiny_hellaswag(...)
    # 5. generate
    show_generations(...)
```

## Slide 13 · 真训前 sanity

```
1. forward 一次  → shape OK
2. 5 step       → loss 11 → 9
3. 100 step     → loss < 5
4. ckpt save/load 测试
5. RNG state 一致
```

## Slide 14 · 加速 trick

```
- torch.compile(model)   +20% 速度
- F.scaled_dot_product_attention (built-in flash)
- tf32 enabled (torch.set_float32_matmul_precision('high'))
- mem_format = channels_last (NHWC) — 不适用 LLM
```

## Slide 15 · 真训 vs 玩具

```
本 capstone: 500M token, 6h, val_loss 2.8
Phi-1.5 (官方): 30B token, 600x GPU-hour, val_loss < 2.5
量级差 60×
```

## Slide 16 · 输出物

```
ckpt/
  final.pt (model weights)
  step_2000.pt (intermediate)
report.md (loss curve + hellaswag + 生成示例)
notebook 13-capstone.ipynb (展示)
```

## Slide 17 · 系列收尾

```
本 capstone = pretraining-recipe 终点
也是 Module 3 「造大模型」最大单 capstone

下一 topic small-model-graduation 是综合毕业
将本 270M ckpt 作为基座
```

## Slide 18 · 完成 checklist

```
[ ] 数据 prepare 完成
[ ] sanity check 通过 (loss 下降 100 step)
[ ] 训完 4000 step
[ ] val_loss < 3.0
[ ] tinyHellaSwag > 0.4
[ ] 生成 5 个流畅样本
[ ] README 写报告
[ ] tag pretraining-recipe
```

## Slide 19-20 · 资源 / 参考

```
- nanoGPT karpathy/nanoGPT GitHub
- Phi-1.5 paper (Microsoft 2023)
- TinyStories paper (Eldan 2023)
- Cosmopedia HF
```

## 参考
- nanoGPT
- Phi-1.5 / Phi-2 reports
