# L05 · 长 ctx 扩展 (ckpt D)

> 10 slides | 30 min ⭐⭐⭐⭐

## Slide 1 · ckpt D vs C

```
D = C + 长 ctx 扩展
方法: YaRN scale=4, max_len 2048 → 8192
不重新训, 在 C 的 ckpt 上继续 fine-tune 100 step
```

## Slide 2 · 复用 Topic 5 的 YaRN

```python
from long_context.src.rope_yarn import yarn_inv_freq, yarn_attention_scaling

inject_yarn(model, scale=4.0, new_max_pos=8192)
```

## Slide 3 · LoRA 锁住主权重

```python
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, target_modules=["q_proj", "k_proj"])
model = get_peft_model(model, cfg)
```

只训 LoRA, 节省显存 + 时间.

## Slide 4 · 数据

```
长文档 (FineWeb-Edu 8k+ chunk)
50M token
```

## Slide 5 · 训练时长

```
100 step × seq 8192 × batch 4 = 3.3M token
~2-4h on 5090
```

## Slide 6 · 期望

```
ctx     NIAH @ before  NIAH @ after
2k      90%            95%   (持平)
4k      40%            85%
8k      5%             80%
```

## Slide 7 · benchmark 不变

```
val_loss: 同 C (~ 2.9)
HellaSwag: 同 C (~ 0.48)
tinyMMLU: 同 C (~ 0.33)
NIAH @ 8k: 5% → 80%  ⭐ 关键提升
```

长 ctx 训练只改 NIAH, 不改 short ctx benchmark.

## Slide 8 · 启动

```bash
python src/train_variant.py --variant D \
  --resume ckpt_C.pt \
  --inject_yarn --scale 4.0 \
  --max_step 100 --seq_len 8192
```

## Slide 9 · ckpt D 是 long ctx 阶段

```
教学意义: 长 ctx 是独立阶段, 不与短 ctx 同时训
工业实践: 同 (Llama-3 / Qwen-2.5 / DeepSeek-V3 都分阶段)
```

## Slide 10 · 总结

```
ckpt D = C + YaRN + 长数据 SFT
NIAH 提升 75pp, benchmark 0pp
长 ctx 不是免费午餐
```

## 参考
- Topic 5 long-context (YaRN)
- Topic 7 pretraining-recipe (CPT)
