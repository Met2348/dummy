# L08 · Byte-level / 多语言 tokenizer

> 18 slides | 55 min | Data Curation 第 8 讲 ⭐⭐⭐

---

## 学习目标

1. 理解 byte-level vs char-level 的 trade-off
2. 比较 GPT-2 / cl100k / Llama-3 / Qwen 多语言压缩率
3. 选 tokenizer 的工程考量

---

## Slide 1 · char vs byte vs subword

```
char:     [H][e][l][l][o]       5 token
byte:     [72][101][108][108][111]   5 token
subword:  [Hel][lo]              2 token
```

LLM 几乎都用 subword（BPE / Unigram），底层是 char 或 byte。

---

## Slide 2 · byte-level 的优势

```
中文「中」UTF-8 = 0xE4 0xB8 0xAD (3 byte)
BPE byte-level:
  ─ 词表必含 0x00-0xFF
  ─ 任何 unicode 均可编码（无 UNK）
  ─ 中文 1 char ≈ 3 token (未合并时)
```

GPT-2 BBPE 是先驱。

---

## Slide 3 · char-level 的优势

```
中文「中」char-level 仅 1 token
词表覆盖率高 → 中文压缩好
但词表大（UTF-8 全字符 ~13万）
```

SentencePiece 默认 char-level + byte_fallback。

---

## Slide 4 · GPT-2 vs cl100k vs o200k

| Tokenizer | 词表 | 算法 | 用 |
|-----------|------|------|---|
| gpt2 | 50k | BBPE | GPT-2/3 |
| cl100k_base | 100k | BBPE | GPT-3.5/4 |
| o200k_base | 200k | BBPE | GPT-4o |
| Llama-3 | 128k | tiktoken-style BBPE | Llama-3 |
| Qwen-2.5 | 152k | tiktoken-style | Qwen |

后期模型词表暴涨，主要为非英语 + 数学。

---

## Slide 5 · 压缩率实测

```text
sample (English, 100 chars):
  gpt2          → 28 token (3.6 char/tok)
  cl100k_base   → 24 token (4.2)
  o200k_base    → 22 token (4.5)

sample (Chinese, 100 chars):
  gpt2          → 240 token (0.4)
  cl100k_base   → 70 token (1.4)
  o200k_base    → 55 token (1.8)
  Qwen-2.5     → 45 token (2.2)
```

→ 中文 GPT-2 是 Qwen 的 5× token 数。

---

## Slide 6 · 编程语言

| | gpt2 | cl100k | Code Llama |
|---|------|--------|-----------|
| Python | 4.2 | 4.7 | 4.9 |
| C++ | 3.8 | 4.3 | 4.5 |
| Rust | 3.5 | 4.0 | 4.2 |

Code Llama 单独训了 code tokenizer，压缩 +5%。

---

## Slide 7 · 数学符号

```
\int_0^\infty e^{-x^2} dx

gpt2:   ['\\', 'int', '_', '0', '^', '\\', 'inf', 'ty', ...]  ~16 token
cl100k: ['\\int', '_', '0', '^', '\\infty', ...]              ~11 token
Llama-3: 类似 cl100k
```

新词表对 LaTeX 友好，数学训练效益高。

---

## Slide 8 · split_digits 实务

```
"3.14159"
  ── 不 split → 1 token (单一 "3.14159")
  ── split    → 7 token (1 数字 / token)
```

Llama-3 / GPT-4 split → 学会逐位算。
GPT-3.5 不 split → 大数加法弱。

---

## Slide 9 · special tokens

```
<|endoftext|>     end of document
<|startoftext|>   start (一些早期模型)
<|im_start|>      ChatML 角色起始
<|im_end|>        ChatML 角色结束
<thinking>         R1 推理
<answer>          R1 答案
<unused_001>      预留 1-1000 slot
```

GPT-4 cl100k 预留 1k unused 用于未来扩展。

---

## Slide 10 · tokenizer 的"对齐"

不同模型 tokenizer 不通用，意味着：
- 跨模型蒸馏需要 token-by-token re-map
- 双模型联合训练需统一词表（很难）
- 评测 token 数与"等价 char/word 数"不同（perplexity 不可直接比）

---

## Slide 11 · 词表扩展（Continual）

```
原 Llama-3 128k → 加 Chinese 5k token = 133k
new_embedding[t] = mean(old_embedding[sub_tokens(t)])
warmup 1k step → 全网络解冻继续训
```

Chinese-LLaMA / 阿里 Qwen 中文扩展都这么做。

---

## Slide 12 · multilingual coverage

理想：每种语言独立词表，但内存太大。

折中：
```
gpt2 → 英语 95%、其他差
cl100k → 中日韩 加强
o200k → 5+ 主要语言全覆盖
Bloomz → 46 自然语言均衡（vocab 250k）
```

vocab 越大 → 多语言越均衡，模型 embedding 越大。

---

## Slide 13 · 词表 vs embedding 维度

```
vocab 50k × hidden 4096 = 200 M 参数
vocab 200k × hidden 4096 = 800 M 参数 (4×)
```

embedding 占总参数的 5-10%（小模型更高）。词表大对小模型成本影响大。

---

## Slide 14 · GPT-4o 词表的"代价"

```
o200k vs cl100k
+ 100k token
+ 600 M embedding
+ 慢 lookup
-  压缩率 +10-15%
```

是否值得？OpenAI 的 ROI 计算说是值得（多语言 / 代码 / 数学训练成本省了更多）。

---

## Slide 15 · 词表选择决策树

```
英语 only         → 32k (gpt2 / Llama-1)
英语 + code        → 50k (gpt2)
多语言 + 数学      → 128k (Llama-3) / 152k (Qwen)
极致多语言        → 200k (o200k) / 250k (Bloomz)
```

---

## Slide 16 · 实务对比脚本

```python
import tiktoken
for name in ["gpt2","cl100k_base","o200k_base"]:
    enc = tiktoken.get_encoding(name)
    ids = enc.encode(text)
    print(name, len(ids), f"{len(text)/len(ids):.2f}")
```

详见 `src/vocab_compare.py`。

---

## Slide 17 · 训练时间影响

```
词表 = 32k → 200k:
embedding fwd/back 慢 2-3×
training token 总数省 ≈ 15%
                ↓
        基本抵消
```

但**推理**永远是省的，长期看大词表赢。

---

## Slide 18 · 课后思考

1. 为什么 byte_fallback 几乎是标配？
2. 大词表的存储 / 加载有什么坑？
3. 跨模型蒸馏的 token 对齐有哪些办法？
4. Llama-3 从 SP → tiktoken 的真实驱动是什么？

---

## 参考

- GPT-2 paper (Radford 2019)
- tiktoken docs
- Llama-3 tokenizer report
- Qwen-2.5 technical report
