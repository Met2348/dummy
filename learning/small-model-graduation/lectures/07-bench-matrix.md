# L07 · 评测矩阵 (5 × M)

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 评测项目

```
val_loss    内部
HellaSwag   常识
PIQA        物理常识
tinyMMLU    多学科知识
GSM8K-tiny  数学推理
NIAH @ 8k   长 ctx
```

合 6 metric × 5 ckpt = 30 数据点.

## Slide 2 · benchmark 矩阵预期

```
           val_loss  HellaSwag  PIQA   tinyMMLU  GSM8K  NIAH@8k
ckpt A     3.50      0.35       0.65   0.25     0.02   0%
ckpt B     3.20      0.40       0.68   0.30     0.05   0%
ckpt C     2.90      0.48       0.72   0.33     0.08   5%
ckpt D     2.90      0.48       0.72   0.33     0.08   80%
ckpt E     2.80      0.50       0.74   0.35     0.10   80%
```

## Slide 3 · GSM8K-tiny

```
原 GSM8K 8.5k 题
tiny: 50 题精选 (难度梯度)
答案 regex: "#### \d+"
exact match
```

## Slide 4 · PIQA

```
Physical Interaction QA
2 choice
size: 16k val
acc:
  random: 0.5
  GPT-2: 0.65
  Llama-2 7B: 0.79
```

## Slide 5 · tinyHellaSwag (本 capstone)

```
3-30 manual examples (复用 Topic 7)
不够 stat sig, 仅 sanity check
真实评测用 lm-eval-harness HellaSwag full
```

## Slide 6 · 评测脚本架构

```python
def run_all_benchmarks(model, tokenizer):
    return {
        "val_loss": validation_loss(model, val_data),
        "hellaswag": tinyhellaswag(model, tokenizer),
        "piqa": tinypiqa(model, tokenizer),
        "tinymmlu": tinymmlu(model, tokenizer),
        "gsm8k": gsm8k_tiny(model, tokenizer),
        "niah_8k": niah_pass_rate(model, tokenizer, ctx=8192),
    }
```

## Slide 7 · 报告 CSV

```csv
variant,val_loss,hellaswag,piqa,tinymmlu,gsm8k,niah_8k
A,3.50,0.35,0.65,0.25,0.02,0
B,3.20,0.40,0.68,0.30,0.05,0
C,2.90,0.48,0.72,0.33,0.08,0.05
D,2.90,0.48,0.72,0.33,0.08,0.80
E,2.80,0.50,0.74,0.35,0.10,0.80
```

## Slide 8 · 可视化

```python
import matplotlib.pyplot as plt

# loss curve overlay (5 lines)
# bar chart per metric (5 vs)
# spider chart (5 metrics × 5 ckpt)
```

## Slide 9 · 报告 markdown

```markdown
# 五部曲对照实验报告

## 实验设计
[5 ckpt 表]

## 结果
[CSV 表]
[3 张图]

## 拆解
+ data:    +5pp HellaSwag
+ arch:    +8pp
+ long_ctx: +75pp NIAH
+ all:     +15pp HellaSwag, +5pp MMLU
```

## Slide 10 · benchmark 用 lm-eval

```bash
lm-eval --model hf \
  --model_args pretrained=./ckpt_E \
  --tasks hellaswag,piqa,arc_easy \
  --batch_size 8
```

工业标准, 100+ task.

## Slide 11 · 注意

```
小模型 (124M-270M):
  GSM8K: 几乎 0 (推理太难)
  MATH: 0
  HumanEval: < 5
不要期望超出能力的指标
```

## Slide 12 · 总结

```
评测矩阵是毕业 report 主要呈现
五部曲贡献用 ablation 清晰展示
```

## 参考
- lm-evaluation-harness
- HellaSwag / PIQA / MMLU papers
