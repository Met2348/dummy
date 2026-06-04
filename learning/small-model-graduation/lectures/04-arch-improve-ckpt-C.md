# L04 · 改架构 (ckpt C, Phi-tiny)

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · ckpt C 与 B 的差别

```
data: 同 B (Cosmopedia + filtered)
training: 同 B
model: 改 Phi-tiny 270M
  + RoPE
  + RMSNorm
  + SwiGLU
  + GQA
  + tied embedding
```

## Slide 2 · 参数变化

```
ckpt A/B: 124M (vanilla GPT-2)
ckpt C:   270M (Phi-tiny)
```

变大 2.2× → 更长训:
- A/B: 3000 step × 64k tok = 200M token
- C: 4000 step × 128k tok = 500M token

## Slide 3 · 复用 Topic 7 model

```python
from pretraining_recipe.src.phi_tiny_model import PhiTinyConfig, PhiTiny

cfg = PhiTinyConfig()
model = PhiTiny(cfg)
```

完全 reuse.

## Slide 4 · 期望 (vs ckpt B)

```
val_loss: 3.2 → 2.9  (-0.3)
HellaSwag: 0.40 → 0.48  (+8pp)
tinyMMLU: 0.30 → 0.33  (+3pp)
```

## Slide 5 · 提升来源拆解

```
+ size 124M → 270M: ~ +5pp HellaSwag
+ SwiGLU vs GELU: ~ +1pp
+ GQA vs MHA: 0pp (perf 持平, 节省 KV)
+ RoPE vs absolute PE: 0pp 短 ctx 持平
+ 更多 token: ~ +2pp
```

合计 +8pp HellaSwag.

## Slide 6 · 训练时长

```
B: 4h
C: 8h (大 model + 多 token)
```

仍可 5090 24G 跑.

## Slide 7 · 启动

```bash
python src/train_variant.py --variant C \
  --model phi_tiny --max_step 4000
```

## Slide 8 · 内存 (5090)

```
weights bf16: 540 MB
grad bf16: 540 MB
opt fp32: 2.16 GB
activation: 4-6 GB
total: ~ 12 GB ✓
```

## Slide 9 · ckpt 比较码

```python
def eval_compare(ckpts):
    for label, path in ckpts.items():
        m = load(path)
        print(f"{label}: val_loss={val_loss(m):.3f}")
        print(f"  HellaSwag={hellaswag(m):.3f}")
        print(f"  tinyMMLU={tinymmlu(m):.3f}")
```

## Slide 10 · 与 ckpt B 一致性测试

```
启动相同 seed
forward A 一次, 期望 logits 完全一致 (除架构差)
```

## Slide 11 · 总结

```
ckpt C 是"改架构"实验
+8pp 来自 size + 现代架构
之后 D 加长 ctx, E 是综合最终
```

## Slide 12 · 参考

- Topic 2 transformer-deep (架构基础)
- Topic 7 pretraining-recipe (Phi-tiny model code)
