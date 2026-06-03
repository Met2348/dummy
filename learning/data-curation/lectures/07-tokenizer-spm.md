# L07 · SentencePiece + Unigram — Llama 系标准

> 22 slides | 65 min | Data Curation 第 7 讲 ⭐⭐⭐⭐

---

## 学习目标

1. 理解 SentencePiece 与 BPE 的关系
2. 掌握 Unigram LM 子词算法
3. 比较 BPE / Unigram 的优劣
4. 跑通 sentencepiece 训练 + 推理 demo

---

## Slide 1 · SentencePiece 是什么

Google 2018 提出，是**框架**而非单算法：
- 支持 BPE / Unigram / Char / Word
- 不要求语料预 tokenize（处理空格）
- 输出 protobuf 文件，跨语言/平台
- Llama-1/2/3 / Mistral / Gemma / Phi 全用

```bash
spm_train --input=raw.txt --model_prefix=m --vocab_size=32000 --model_type=unigram
```

---

## Slide 2 · 与 BPE 的关键差异

| 维度 | GPT-2 BBPE | SentencePiece |
|------|-----------|---------------|
| 预 tokenize | regex 必需 | **不需要** |
| 空白处理 | 单独 | 视为 `▁` |
| 算法 | 仅 BPE | BPE 或 Unigram |
| 输出 | merge 列表 | protobuf |

SP 的"无需预 tokenize"是它能跨语言（中日韩没明显空格）的关键。

---

## Slide 3 · `▁` 符号

```
"Hello world" → "▁Hello▁world"
"中文没有空格" → "▁中文没有空格"
```

`▁` (U+2581) 标记前导空格。这样多语言统一处理。

---

## Slide 4 · BPE 模式（SP-BPE）

与 GPT-2 BBPE 类似，但 char-level（基于 unicode）而非 byte-level：

```
init  = set of unique characters in corpus
merge = same greedy frequency merging
```

Llama-1 用此模式。

---

## Slide 5 · Unigram 算法概述

```
1. 初始大词表 (1M 候选)
2. EM 算法估每个 token 概率 p(w_i)
3. 计算每个 token 移除后的 loss 增量
4. 移除 loss 增量最小的 10%
5. 回到 2，直到目标词表
```

Unigram 是"自顶向下"，BPE 是"自底向上"。

---

## Slide 6 · Unigram 的概率模型

句子 S 切分为 token 序列 (t_1, t_2, ..., t_k)：

```
P(S) = ∏ p(t_i)
```

最优切分 = Viterbi 找最大概率。SP 用 EM + Viterbi 联合训练。

---

## Slide 7 · BPE vs Unigram

| | BPE | Unigram |
|---|-----|---------|
| 切分唯一性 | 唯一 | 多解（最优 Viterbi）|
| subword regularization | 不支持 | 支持 |
| 速度 | 快 | 中 |
| 实务 | Llama-1 | Llama-2/3/Mistral |

Unigram 因支持 subword reg（训练时随机选次优切分）→ 模型抗 typo 更强。

---

## Slide 8 · subword regularization

```python
spm.encode_as_pieces("hello", enable_sampling=True, alpha=0.1)
# 多次调用得不同切分
# → "hello" / "hel|lo" / "h|el|lo" / ...
```

训练时随机采样，相当于 token-level 数据增强。Llama-3 仅在 SFT 启用。

---

## Slide 9 · sentencepiece 训练

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="m",
    vocab_size=32000,
    model_type="unigram",     # 或 "bpe"
    character_coverage=0.9995,
    num_threads=8,
)
```

10MB 语料 ~1 min。100GB 需分片预处理。

---

## Slide 10 · 关键超参

| 参数 | 推荐 | 含义 |
|------|------|------|
| `vocab_size` | 32k-128k | 词表大小 |
| `model_type` | unigram | 算法 |
| `character_coverage` | 0.9995 (en) / 1.0 (zh) | 字符覆盖率 |
| `byte_fallback` | True | 罕见 char → byte |
| `split_digits` | True | 每数字独立 token |
| `add_dummy_prefix` | True | 句首加 `▁` |

---

## Slide 11 · byte_fallback 的妙处

```
原 vocab 不含 emoji "🦄"
↓ byte_fallback=True
encode: ["<0xF0>", "<0x9F>", "<0xA6>", "<0x84>"]   ← 4 byte token
```

256 个 byte token 保底，任何 unicode 都能 round-trip。Llama-2/3 启用。

---

## Slide 12 · sentencepiece 推理

```python
sp = spm.SentencePieceProcessor(model_file="m.model")
ids = sp.encode("Hello world", out_type=int)
pieces = sp.encode("Hello world", out_type=str)
text = sp.decode(ids)
```

C++ 后端，~500k token/s 单核。

---

## Slide 13 · Llama-1 vs Llama-2/3 词表

| | Llama-1 | Llama-2 | Llama-3 |
|---|---------|---------|---------|
| 算法 | BPE | BPE | **tiktoken-BBPE** |
| 词表 | 32k | 32k | **128k** |
| 中文 token | 多 | 多 | **少** |

Llama-3 切到 tiktoken-style BBPE！抛弃 SP，跟随 GPT-4。

---

## Slide 14 · Mistral / Mixtral 词表

Mistral 用 SP-BPE 32k，与 Llama-1 相似。Mixtral 8x7B 沿用。

Phi-3 也是 SP，词表 32k。

→ SP 仍是当前**最广**使用方案，尤其小模型。

---

## Slide 15 · 训 SP 用什么语料

实务：

```
- 多源平衡（web 60% / books 20% / code 10% / wiki 10%）
- 过滤后高质量子集 ~ 1B token 足够
- 避免单一语言主导（80% en + 20% 其他）
```

训 tokenizer 比训模型省 1000×。

---

## Slide 16 · 评估指标

```
压缩率：char / token
词表利用率：>0.9 (训练后)
未覆盖率：byte_fallback 触发率 < 0.1%
```

定期 sanity check：用真实测试集 encode → decode，比较 BLEU / 精确匹配。

---

## Slide 17 · SentencePiece protobuf 内部

```
m.model   = 序列化的 SentencePieceModel proto
含：
  - vocab list with scores (Unigram)
  - merges list (BPE)
  - normalizer rules (NFKC 等)
  - byte_fallback flag
```

可读：`spm_export_vocab` 输出文本词表。

---

## Slide 18 · 多语言挑战

不同字符集 unicode 范围差异大：

```
en        : 0x20-0x7E (95 char)
zh CJK     : 0x4E00-0x9FFF (20k+)
emoji      : 0x1F000-0x1FFFF (~3k)
```

`character_coverage=0.9995` 在英语足够；中文必须 1.0 否则字典丢字。

---

## Slide 19 · 数字 split 的重要性

```
"3.14159" 不 split  → 1 token
"3.14159"   split  → 7 token "3" "." "1" "4" "1" "5" "9"
```

数学训练：split 后模型学"逐位运算"，无 split 学不会大数加减。

Llama-3 显式 split_digits=True。

---

## Slide 20 · 训完之后

```
m.model      → 推理用
m.vocab      → 文本词表（可读）
```

把 `m.model` 放入 `transformers` 模型 repo，配置 `tokenizer.model`，HF API 直接加载。

---

## Slide 21 · 实务陷阱

1. 训 tokenizer 时多语言比例 ≠ 训模型时 → 部分语言压缩差
2. `character_coverage` 太低 → 罕见字直接 byte_fallback 全是 4 token
3. 训完没启 `byte_fallback` → emoji / 古文 无法 encode
4. 与下游推理框架 (vllm) tokenizer 不一致 → 出错

---

## Slide 22 · 课后思考

1. Llama-3 为什么从 SP 切到 tiktoken BBPE？
2. Unigram subword regularization 真能减少 overfit 吗？
3. 1M 候选 → 32k 词表 = 30× 压缩。EM 步骤有几次？
4. 若只训中文模型，词表大小应选多少？

---

## 参考

- Kudo 2018 (SentencePiece)
- Kudo 2018 (Unigram + subword reg)
- SentencePiece github
- Llama-3 tokenizer report 2024
