# L10 · 五部曲 trick 总结

> 14 slides | 40 min ⭐⭐⭐⭐⭐

## Slide 1 · 5 部曲 × 5 关键 trick

```
data:        高质 + 配比 + dedup + 长 doc
arch:        RMSNorm + GQA + SwiGLU + RoPE + tied
ctx:         YaRN + 课程 + packing + NIAH 训
infra:       FSDP + bf16 + grad ckpt + WSD
recipe:      μP + warmup + clip + spike skip + ckpt
```

## Slide 2 · top-10 高 ROI trick

```
1. 数据 dedup (MinHash)               easy + 3pp
2. 高质数据 mix (Cosmopedia)           easy + 5pp
3. 改 GELU → SwiGLU                   easy + 1pp
4. MHA → GQA                           easy + 0pp + 4× KV
5. abs PE → RoPE                        easy + 长 ctx 可扩
6. cosine → WSD                         easy + 1pp + 易 resume
7. 加 RMSNorm vs LN                     easy + 0pp + 速度
8. tied embedding                       easy + 50M 参数
9. grad checkpoint                       easy + 5× ctx
10. bf16 + AMP                          must, 否则 OOM
```

## Slide 3 · 数据 tricks

```
- MinHash dedup (Topic 1 L03)
- 不同 source 比例 (DCLM/DoReMi)
- 课程: 通用 → code → math
- annealing 阶段加高质
- 合成 + real 混合
- chat 模板 (SFT 阶段)
```

## Slide 4 · architecture tricks

```
- Pre-Norm > Post-Norm (稳)
- RMSNorm > LayerNorm (省 1/2 参数, 同效果)
- SwiGLU > GELU (+1pp)
- GQA > MHA (减 KV cache)
- RoPE > absolute/relative PE (长 ctx 友好)
- tied embedding (省参数)
- 4*hidden d_ff dense, 8/3*hidden SwiGLU
```

## Slide 5 · long-ctx tricks

```
- YaRN > NTK > PI (RoPE scaling)
- 课程 max_len 8k → 16k → 32k
- packing + doc mask (训练效率)
- attention temperature
- annealing 长 doc
```

## Slide 6 · infra tricks

```
- FSDP > DDP (省显存)
- bf16 > fp16 (不需 GradScaler)
- grad checkpoint (5× ctx)
- WSD > cosine
- AdamW betas (0.9, 0.95)
- weight_decay 0.1
- LN/bias 不 decay
- grad clip 1.0
```

## Slide 7 · recipe tricks

```
- μP scaling (小 ckpt 调超参)
- warmup 5% (避免 spike)
- spike skip (EMA threshold)
- ckpt every 500 step
- RNG state 保存
- log every 100 step
- val_loss every 1000 step
- resume 友好 (shard cursor)
```

## Slide 8 · "不要做"清单

```
✗ post-norm 风格新 model (LLaMA-1 后弃)
✗ MHA on 大 model (memory 浪费)
✗ ReLU MLP (1pp 差)
✗ absolute PE (不能扩 ctx)
✗ untied embedding (浪费 50M)
✗ 量化训练 (qLoRA 用 LoRA 微调可, 全训不可)
✗ 100% 合成数据 (model collapse)
```

## Slide 9 · "时间紧"必备 (5 件)

```
1. RoPE + GQA + RMSNorm + SwiGLU + tied (架构)
2. bf16 + grad ckpt + FSDP (infra)
3. 高质数据 (Cosmopedia 路线)
4. WSD lr schedule
5. ckpt + log every step (debug)
```

## Slide 10 · 推理时 trick

```
- vLLM / SGLang
- PagedAttention
- 量化 (AWQ/FP8)
- speculative decode (EAGLE)
- KV cache quant
```

## Slide 11 · benchmarking tricks

```
- lm-eval-harness
- few-shot 5 vs 0 (公平)
- prompt 一致 (不要换)
- 多 seed 平均
- 长 ctx 用 NIAH/RULER (不用 ppl)
```

## Slide 12 · 总结表

```
data     +5pp    | dedup + 高质 mix + 课程
arch     +5pp    | RoPE + GQA + SwiGLU + RMSNorm
ctx      +75pp NIAH | YaRN + 长 doc + annealing
infra    -       | bf16 + FSDP + grad ckpt
recipe   稳定    | WSD + warmup + clip + spike skip
合计     +15pp HellaSwag, +5pp MMLU, NIAH 0→80%
```

## Slide 13 · Module 3 学完意味着

```
- 能跑通从零开始的预训练
- 理解 Llama-3 / DeepSeek-V3 全图
- 知道每个 trick 的 ROI
- 能改进现有 ckpt (CPT/长 ctx 扩)
```

## Slide 14 · 下一程 Module 4

```
有了 ckpt E → 做 SFT
SFT 后 → 做 DPO / GRPO
进入对齐 + 推理 RL 系列
```

## 参考
- Llama-3 tech report
- DeepSeek-V3 tech report
- Phi-3 tech report
- Topic 1-7 of this series
