# L03 · 改数据 (ckpt B)

> 10 slides | 30 min ⭐⭐⭐⭐

## Slide 1 · ckpt B 与 A 唯一差别

```
model: 与 A 相同 (vanilla GPT-2 124M)
training: 与 A 相同 (3000 step)
data: 改了
  A: TinyStories + OpenWebText
  B: Cosmopedia + filtered OpenWebText (high quality)
```

## Slide 2 · Cosmopedia 简介

```
HuggingFaceTB/cosmopedia
30B synthetic educational tokens
Mistral-Large 生成
有 25 个 topics × 教科书 / 故事 / Q&A
```

本 capstone 用 100k 子集 (= ~ 50M token).

## Slide 3 · 数据过滤 (从 Topic 1)

```python
# 复用 Topic 1 的 quality filter
from data_curation.src.quality_filter import filter_doc

clean = [d for d in webtext if filter_doc(d, min_words=50,
                                            max_perplexity=100)]
```

## Slide 4 · 配比

```
A: 50% TinyStories + 50% OpenWebText
B: 60% Cosmopedia + 40% OpenWebText (filtered)
   ~ 200M token (与 A 一致)
```

## Slide 5 · 预期改善

```
val_loss: 3.5 → 3.2  (-0.3)
HellaSwag: 0.35 → 0.40  (+5pp)
tinyMMLU: 0.25 → 0.30  (+5pp)
```

数据贡献 ~5pp.

## Slide 6 · 训练时长

```
A 4h
B 4h (无任何变更)
对比 0 额外成本, 提升显著
```

## Slide 7 · 启动

```bash
python src/train_variant.py --variant B --max_step 3000
```

## Slide 8 · 关键: 数据 quality > quantity

```
Phi-1: 7B synthetic ≈ Llama-1 7B 1.5T web
量 200×, 质 hi 仍能赢
本 capstone 50M 高质 ≈ A 200M 普通
```

## Slide 9 · 注意

```
合成 + real 都要保留 (防 model collapse)
60/40 比 100% 合成更稳
```

## Slide 10 · 总结

```
ckpt B 是"换数据"实验
+5pp 是数据 alone 的贡献
之后 ckpt C 加架构, 再 +1pp
```

## 参考
- Cosmopedia (HF 2024)
- Phi-1.5 paper
