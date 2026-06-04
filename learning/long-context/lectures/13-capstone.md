# L13 · Capstone — Llama-3.2-1B YaRN 扩到 32k

> 18 slides | 60 min ⭐⭐⭐⭐⭐

## Slide 1 · 目标

```
基座: Llama-3.2-1B-Instruct (原 ctx 8k)
目标: 扩到 32k 并 NIAH > 80%
方法: YaRN scale=4 + LoRA fine-tune 500 step
设备: 5090 24G + bf16 + grad checkpoint
```

## Slide 2 · 总流程

```
① 加载 Llama-3.2-1B (原 8k)
② 注入 YaRN RoPE (s=4)
③ NIAH baseline @ 8k/16k/32k
④ LoRA r=16 SFT on 长 corpus (8k→32k 课程)
⑤ NIAH after @ 8k/16k/32k
⑥ 对照表 + 报告
```

## Slide 3 · 数据准备

```
SlimPajama-6B 子集 + arxiv-papers
打包到 32k:
  ~ 200 step × batch 1 × 32k = 6.5M token
```

## Slide 4 · YaRN 注入

```python
def yarn_rope(rope, scale=4.0, max_pos=32768):
    base = rope.base
    dim = rope.dim
    new_base = base * scale ** (dim / (dim - 2))
    inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq
```

## Slide 5 · attention temperature

```python
attn_temp = math.sqrt(0.1 * math.log(scale) + 1.0)
attn_weights = (q @ k.transpose(-2, -1)) / (math.sqrt(d) * attn_temp)
```

YaRN 独有：缓解 scale 后 softmax 锐化。

## Slide 6 · LoRA 配置

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Slide 7 · 训练 hyperparams

```
lr: 5e-5
batch_size: 1 (32k 一条已挤满)
grad_accum: 8
max_steps: 500
warmup: 50
gradient_checkpointing: True
bf16: True
```

## Slide 8 · 课程

```
step 0-100: max_len 8k
step 100-300: max_len 16k
step 300-500: max_len 32k
```

## Slide 9 · NIAH 评测网格

```
ctx: 4k, 8k, 16k, 24k, 32k
depth: 10%, 30%, 50%, 70%, 90%
n_sample: 5
共 5 × 5 × 5 = 125 样本
```

## Slide 10 · 显存账本

```
Llama-3.2-1B:  2.4 GB (bf16)
KV cache @32k: 1.3 GB
activation:    8-12 GB (grad ckpt)
LoRA grad:     0.2 GB
total: ~ 16 GB ✓ on 24G card
```

## Slide 11 · 预期对照

```
@ 8k:
  baseline: 95%
  YaRN-only: 92%
  YaRN+LoRA: 95%
@ 32k:
  baseline: 5% (OOR position id)
  YaRN-only: 60%
  YaRN+LoRA: 82%
```

## Slide 12 · 训练时长

```
5090 24G, bf16, grad ckpt:
  500 step × ~40s/step ≈ 5.5 hour
```

## Slide 13 · 评测时长

```
125 sample × 3 setting = 375 inference
@ 32k 单次 1-3s
total ≈ 15-30 min
```

## Slide 14 · capstone_yarn_llama32.py 框架

```python
def train_yarn_llama_32k():
    model = load_llama_3_2_1b()
    inject_yarn(model, scale=4.0)
    add_lora(model, r=16)
    
    dataset = curriculum_dataset(stages=[8k, 16k, 32k])
    
    train(model, dataset, max_steps=500)
    save_lora(model, 'ckpts/yarn-32k')
```

## Slide 15 · 评测脚本

```python
def eval_niah(model, ctx_len, depth, n=5):
    correct = 0
    for _ in range(n):
        q, ans = make_niah_query(ctx_len, depth)
        pred = model.generate(q, max_new_tokens=20)
        if ans in pred:
            correct += 1
    return correct / n
```

## Slide 16 · 可视化

```
heatmap: x=ctx, y=depth, color=accuracy
3 张图: baseline / YaRN-only / YaRN+LoRA
明显看到 middle U 缓解
```

## Slide 17 · 真实性 caveat

```
本 capstone 5090 单卡只是教学规模
真实 Llama-3.1-8B-128k:
  8×H100 + 1k step + 30B token
  PI → YaRN → LongRoPE → DPO 多段
```

## Slide 18 · 系列收尾

至此完成 5 个 topic 的 Module 3 「造大模型」核心：
- data-curation
- transformer-deep
- moe-architecture
- ssm-hybrid
- **long-context (本)**

接下来：scaling-infra → pretraining-recipe → graduation.
