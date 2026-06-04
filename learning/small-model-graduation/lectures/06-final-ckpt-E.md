# L06 · 综合 (ckpt E)

> 10 slides | 30 min ⭐⭐⭐⭐⭐

## Slide 1 · ckpt E = C + 长 ctx 一起训

```
不分阶段, 一开始就 RoPE base × 4
长 doc + 短 doc 混合训练
直接 8k seq_len
```

## Slide 2 · 复刻 Llama-3 / DeepSeek 思路 (微缩)

```
phase 1: 80% step, 1024 ctx, general data
phase 2: 15% step, 4096 ctx, code + math
phase 3: 5% step, 8192 ctx, long doc
WSD lr final 20% annealing 高质数据
```

## Slide 3 · 资源

```
total step: 4000 (与 C 相同)
但 phase 3 慢 (seq 8×)
实际 ~ 10h
```

## Slide 4 · model

```
与 ckpt C / D 相同 Phi-tiny 270M
但 max_pos 一开始就 8k
RoPE base = 40000 (= 10000 × 4)
```

## Slide 5 · 数据

```
phase 1: Cosmopedia 50% + filtered web 30% + code 20%
phase 2: + math (10%)
phase 3: long-doc only (FineWeb-Edu 长文档)
```

## Slide 6 · 期望 (vs B)

```
val_loss: 3.2 → 2.8  (-0.4)
HellaSwag: 0.40 → 0.50  (+10pp)
tinyMMLU: 0.30 → 0.35  (+5pp)
NIAH @ 8k: 0% → 80%   ⭐
```

## Slide 7 · 与 ckpt C+D 串联 vs E 直接

```
C+D: 8h + 4h = 12h, 两阶段, NIAH 80%
E: 10h 一气, NIAH 80%
两者 benchmark 接近
工业: E 路线 (省阶段切换)
教学: C/D 分阶段更清晰
```

## Slide 8 · 启动

```bash
python src/train_variant.py --variant E \
  --max_step 4000 --curriculum
```

## Slide 9 · 五部曲贡献分解

```
total +15pp HellaSwag:
  + data quality:  +5pp
  + architecture:  +5pp
  + scale (270M):  +3pp
  + long ctx + curriculum: +2pp
```

## Slide 10 · 总结

```
ckpt E 是 Module 3 最终 capstone
五部曲所有 trick 集成
本系列 graduation
```

## 参考
- Llama-3 tech report (curriculum)
- DeepSeek-V3 tech report
