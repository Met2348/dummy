# L06 · BPE Tokenizer — 从手写到 tiktoken

> 28 slides | 75 min | Data Curation 第 6 讲 ⭐⭐⭐⭐⭐

> Karpathy 风格 minbpe 教学 + 工业 tiktoken 对照

---

## 学习目标

1. 理解 BPE 算法核心：合并最频繁字节对
2. 手写 100 行 BPE 训练 + 推理
3. 用 tiktoken 加载 cl100k / o200k / 自训
4. 分析压缩率、对模型训练的影响

---

## Slide 1 · 为什么需要 tokenizer？

LLM 处理 token id，而不是 char。两个极端：

```
char-level   |H|e|l|l|o|     8 token
word-level   |Hello|         1 token
BPE          |Hel|lo|        2 token  ← 折中
```

BPE 让"频繁的词"成 1 token，"罕见词"按 sub-word 拼。

---

## Slide 2 · BPE 算法（训练）

```
1. 初始词表 = {byte 0-255}
2. 把语料拆成 byte 序列
3. 统计相邻 byte pair 频率
4. 选最频繁的 pair → 合并为新 token
5. 词表 +1，回到 3
6. 直到目标词表大小
```

经典版本：Sennrich 2015。GPT-2 用 byte-level 变种。

---

## Slide 3 · 数学定义

```
T = set of tokens
V = vocab size target

while |T| < V:
    pair, count = argmax over (a,b) in T×T of freq(ab)
    T.add(new_token = a + b)
    apply_merge(corpus, (a,b))
```

每次合并增加 1 token，需 V-256 次合并。

---

## Slide 4 · 例子

语料：`aaabdaaabac`（字符示意）

```
init   pairs: (aa, 4) (ab, 2) (bd, 1) (da, 1) (ac, 1)
merge1 (aa → Z):    ZabdZabac
merge2 (Za → Y):    YbdYbac   pairs:(Yb,2)(bd,1)(c,1)
merge3 (Yb → X):    XdXac
```

3 次合并后压缩比 11 → 5。

---

## Slide 5 · Byte-level BPE (GPT-2)

直接在 byte 上 BPE 而非 char：
- 词表初始 = 0-255（256 byte 必有覆盖）
- UTF-8 多字节字符天然支持
- 罕见 unicode 自动 fallback

**优**：无 UNK，全语言覆盖
**劣**：中文 1 char = 3 byte，token 多

---

## Slide 6 · BBPE 算法（GPT-2）

```python
def bbpe_train(corpus_bytes, vocab_size):
    vocab = list(range(256))           # init bytes
    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for word in corpus_bytes:
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += 1
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        new_id = len(vocab); vocab.append(new_id)
        # apply merge to all words
        ...
    return vocab, merges
```

详见 `src/bpe_trainer.py`。

---

## Slide 7 · 编码（推理）

```
text → bytes → 反复应用 merges（按训练顺序）→ token ids
```

实务：trie / cache 加速，1M token/s 单核可达。

---

## Slide 8 · 关于"pre-tokenization"

GPT-2 在 BPE 前先用 regex 切：

```python
pre = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|
                    [\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+""", re.X)
```

把语料切成"词样"块，BPE 只在块内合并。
→ 防止 BPE 学到跨词的合并（如 "apple_OF" → 1 token）。

---

## Slide 9 · 词表大小怎么选

| 词表 | 例 | 备 |
|------|-----|---|
| 32k | Llama-1 / 2 | 早期 |
| 50k | GPT-2 | byte-level |
| 100k | cl100k (ChatGPT) | 多语言加强 |
| 128k | Llama-3 | 多语言 + 数学符 |
| 200k | o200k (GPT-4o) | 进一步 |

大词表 → 推理省（每 token 多 char）；训练慢（embedding 大）。

---

## Slide 10 · 词表对压缩率影响

GPT-4 实测压缩：
```
English (32k vs 100k vs 200k):  1.5 / 1.8 / 1.9 char/token
Chinese                        :  0.8 / 1.5 / 1.7
Code                           :  3.0 / 3.5 / 3.8
Math (LaTeX)                   :  1.2 / 1.6 / 1.9
```

大词表对非英语 / 数学 / 代码增益最大。

---

## Slide 11 · 中文 tokenization 痛点

| tokenizer | "今天天气真好" 编为 |
|-----------|-------------------|
| GPT-2 50k | 16 token (byte fallback) |
| cl100k | 6 token |
| Qwen 152k | 3 token |
| Yi 64k | 4 token |

→ GPT-2 中文 token 是 cl100k 的 2-3 倍。中文模型必须自训 tokenizer。

---

## Slide 12 · tiktoken 简介

OpenAI 出品，Rust 实现：

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("Hello, world!")
text = enc.decode(ids)
```

- 加载快（mmap）
- 编码快（O(n)，pyrust）
- 兼容 GPT-2/3/4 全系列

---

## Slide 13 · tiktoken vs HuggingFace

| | tiktoken | HF tokenizers |
|---|---------|---------------|
| 后端 | Rust | Rust |
| 速度 | 极快 (mmap) | 快 |
| 训练 | 不支持 | 支持 |
| 部署 | OpenAI 家系列 | 通用 |

实务：训用 HF tokenizers（或 SP），推理用 tiktoken。

---

## Slide 14 · 自训 BBPE 到底要多少语料

```
10MB  → 5k merge (玩具)
100MB → 32k merge (小模型)
1GB   → 50k+ merge (Llama 类)
10GB  → 128k merge (多语言)
```

更多语料 → 合并次数饱和（边际收益递减）。

---

## Slide 15 · special tokens

```
<|endoftext|>   →  document 边界
<|im_start|>    →  chat role 标记 (ChatML)
<|im_end|>
<thinking>       →  推理标记
<answer>
```

训 tokenizer 时**预留** unused id，后期附加。GPT-4 cl100k 预留 1k+ slots。

---

## Slide 16 · 数字怎么 tokenize

```
"123456" → ?
```

GPT-2 / cl100k: 多种切法都行（看频率）。Llama-3 显式 split：每个数字 1 token。

```python
re.compile(r"\d{1,3}")
```

→ 数学训练效果改善。

---

## Slide 17 · 关于"训练-推理 tokenizer 不一致"

实务里常见：
- 训用 BBPE
- 推理用 SentencePiece

→ 模型必须用"训练时同款 tokenizer"，否则乱码。

**结论**：tokenizer 选定后不能换。

---

## Slide 18 · 训练-tokenizer 数据要不要 dedup

dedup 前 vs 后？经验：**两个时机都行**，效果差异 < 1%。但：

- 全 web 训 tokenizer → vocab 偏 web
- 多源平衡训 → 更通用

Llama-3 训 tokenizer 用 30% 高质量代理子集。

---

## Slide 19 · GPT-2 BPE 教学版（伪）

```python
import re, collections
def get_pairs(word_freqs):
    pairs = collections.Counter()
    for word, f in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += f
    return pairs
```

完整版 `src/bpe_trainer.py`。

---

## Slide 20 · merge 列表的"递归性"

```
merges = [
    ("h","i"),       → hi
    ("hi","gh"),     → high
    ("high","er"),   → higher
]
```

每行依赖之前的合并。推理时**必须按顺序**应用。

---

## Slide 21 · 性能优化（生产）

- mmap 词表
- trie 树查 longest match
- cache 频繁单词
- 多核 batch encode

tiktoken 1.4M token/s 单核（Rust）。

---

## Slide 22 · 压缩率评估

```python
def compression(tokenizer, text):
    return len(text) / len(tokenizer.encode(text))
```

一般：
- en 4-5 char/token
- code 3-4
- zh 1-2 (依 tokenizer)

LLM 训练成本 ~ token 数 → 压缩率 ↑ → 成本 ↓。

---

## Slide 23 · tokenizer 与多模态

VLM tokenizer 通常含：
- 文本 BPE
- image patch token (16-256/图)
- audio token (codec)

例：Qwen2-VL `<vision_start> ... <vision_end>`。

---

## Slide 24 · 不要混用 tokenizer

混的代价：
- 训练用 A，推理用 B → 输出乱码
- 蒸馏时 teacher / student 不同 → logit 错配
- 继续训练时词表不兼容 → 必须扩词表 init

**纪律**：项目第一天定 tokenizer，半年内不变。

---

## Slide 25 · 关于"扩词表"

```
原 32k → 加 5k 中文 token → 37k
```

新 token embedding 用 mean(known sub-pieces) init，再训 1k step warmup。否则 loss 爆炸。

实务：Chinese-Llama / Open-Chinese-LLaMA 都做过。

---

## Slide 26 · 词表的"对齐"问题

```
"GPT-2 token 0  ≠ Llama token 0"
```

模型间 token id 不可迁移。词嵌入也无法直接复用。

→ 模型不通用，必须重训 embedding。

---

## Slide 27 · 实务清单

```
[ ] 选 tokenizer 类型 (BBPE / Unigram / WordPiece)
[ ] 选词表大小（32k / 100k / 128k）
[ ] 选预 split regex（数字 / 空白 处理）
[ ] 留 1k unused for special tokens
[ ] 训练数据 dedup + 多源平衡
[ ] 验证：压缩率 + 多语言均衡
[ ] 冻结，永不变
```

---

## Slide 28 · 课后思考

1. byte-level vs char-level BPE 实务区别？
2. 词表 64k → 128k 训练成本变化？
3. 数字必须独立 token 的两个理由？
4. tiktoken cl100k 为什么对中文友好？

---

## 参考

- Sennrich 2015 (BPE for NMT)
- Karpathy minbpe (github.com/karpathy/minbpe)
- tiktoken (github.com/openai/tiktoken)
- Llama-3 tokenizer report 2024
