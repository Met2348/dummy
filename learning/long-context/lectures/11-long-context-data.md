# L11 · 长上下文数据：打包与课程

> 16 slides | 45 min ⭐⭐⭐⭐

## Slide 1 · 长 ctx 训练数据从哪来

```
预训练语料平均 1-4k token
真实长文档:
  books (Gutenberg, Books3): 50-200k
  GitHub repo concat: 10-100k
  arxiv full paper + ref: 20-50k
  对话日志: 1-30k
```

需混合上述源 → "long-data mix"。

## Slide 2 · 打包 (packing) 原因

GPU 不喜欢 padding：

```
naive: [doc1 1k] + [pad to 32k] → 31k 浪费
packed: [doc1 1k][doc2 8k][doc3 23k] → 0 浪费
```

## Slide 3 · attention mask 隔离

打包后多文档共享 sequence → 必须用 **document attention mask** 阻断跨 doc attention：

```
mask[i,j] = 1 只在 i,j 同 doc 内
```

否则 doc2 能看 doc1 的 KV → 错误学习。

## Slide 4 · 实现

```python
def make_packed_batch(docs, max_len=32768):
    packed = []
    cur = []
    cur_len = 0
    for d in docs:
        if cur_len + len(d) > max_len:
            packed.append(cur)
            cur = [d]
            cur_len = len(d)
        else:
            cur.append(d)
            cur_len += len(d)
    if cur: packed.append(cur)
    return packed
```

## Slide 5 · 课程学习 (curriculum)

```
stage 1: 0-4k        训 base 知识
stage 2: 4k-16k      引入中等长度
stage 3: 16k-128k    长 ctx 强化
```

逐步增长 max_seq_len，避免一开始就 OOM。

## Slide 6 · 数据配比

| 阶段 | 短<4k | 4-16k | 16-128k |
|------|-------|-------|---------|
| Base | 80% | 15% | 5% |
| 长 ctx | 30% | 30% | 40% |

## Slide 7 · 合成长数据

短文档 + 拼接 + 改写 → 假长文档：

```
DocPack: 5-10 个相关 short doc 串联
RepoPack: 同一 GitHub repo files 串联
WikiPack: 主题相关 wiki 串联
```

DeepSeek-V3 与 Qwen-2.5 都用此方法。

## Slide 8 · "针在草堆"训练

```
长文档 + 插入 reference 句子 (训练时也有 needle 任务)
→ 强化"长 ctx retrieval"能力
```

## Slide 9 · 评测题混入训练

risky but 有效：把 NIAH 类似 task 加入训练（不重叠 test set），accuracy 明显涨。

## Slide 10 · YaRN-style 训练

YaRN 论文方法：
1. 先 PI 调到 32k，全量 fine-tune 1B token
2. 再扩到 128k，再 fine-tune 0.5B token
3. attn temp `1/√t` 自动加入

## Slide 11 · LongRoPE

```
搜索 RoPE per-dim scale factor (而非均匀)
→ 推到 2M+
```

Microsoft Phi-3-mini-128k 即用此法。

## Slide 12 · Position interpolation 复习

```
ctx 8k → 32k:
  PI:   pos / 4
  NTK:  base × 4^(d/(d-2))
  YaRN: NTK by parts + attn temp
```

## Slide 13 · 长 ctx fine-tune 资源

```
8x H100 80G:
  Qwen-7B 32k: 1d
  Qwen-7B 128k: 3-5d
  full Llama-3-70B 128k: 7-15d (需 LoRA)
```

5090 24G：仅 32k LoRA 可行。

## Slide 14 · 训练 trick

- **gradient checkpointing**：必开
- **packed FlashAttention 2/3**：必开
- **bf16 + ZeRO-3**：多卡省显存

## Slide 15 · long_data_packing.py 实现

```python
def pack_documents(docs, max_len):
    cur, cur_len, batches = [], 0, []
    for d in docs:
        if cur_len + len(d) > max_len:
            if cur: batches.append(cur)
            cur, cur_len = [d], len(d)
        else:
            cur.append(d); cur_len += len(d)
    if cur: batches.append(cur)
    return batches

def make_doc_mask(doc_lens):
    """block-diagonal mask"""
    total = sum(doc_lens)
    mask = torch.zeros(total, total, dtype=torch.bool)
    s = 0
    for L in doc_lens:
        mask[s:s+L, s:s+L] = True
        s += L
    return mask
```

## Slide 16 · 总结

```
长 ctx 训练 = 数据准备 + 课程 + packing + RoPE 扩
缺一不可
```

## 参考
- YaRN 2024 (Peng et al)
- DeepSeek-V3 tech report
- Qwen-2.5-1M tech report
