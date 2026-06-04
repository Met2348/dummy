# L03 · Tokenization 选型

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 选型矩阵

| 名 | 算法 | vocab | 训练 | 推荐场景 |
|----|------|-------|------|---------|
| GPT-2 BPE | BPE | 50257 | tiktoken | 英文 |
| Llama-2 | SentencePiece BPE | 32000 | spm | 通用 |
| Llama-3 | tiktoken | 128k | tiktoken | 多语 |
| Qwen-2.5 | tiktoken | 151k | tiktoken | 中文 |
| DeepSeek-V3 | tiktoken | 129k | tiktoken | 多语 |
| Mistral | SentencePiece | 32k | spm | 通用 |

## Slide 2 · vocab size 选择

```
小 vocab (32k): 短词被拆分多, seq 长
大 vocab (128k+): 中文/code 高效, embedding 占显存大

trade-off:
  embedding 占 N_token × hidden × 4 byte
  Llama-3 128k × 4096 × 2 = 1 GB
  seq 长度: 128k vocab → 通常少 30% seq
```

## Slide 3 · vocab 与 OOV

```
BPE/SentencePiece 无 OOV (回落到 byte)
但稀有词被拆很碎
专业领域 (代码、医学) → 扩 vocab
```

## Slide 4 · 中文 token 效率

```
GPT-2 vocab: "你好" → 6 tokens (按 byte)
Qwen vocab: "你好" → 1 token
↑ 训练 / 推理效率差 6×
```

## Slide 5 · 实操：BPE 训练 (HF tokenizers)

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
)
tk.train(["corpus.txt"], trainer)
tk.save("tokenizer.json")
```

## Slide 6 · 实操: SentencePiece

```bash
spm_train --input=corpus.txt --model_prefix=spm \
  --vocab_size=32000 --model_type=bpe \
  --character_coverage=0.9995 \
  --byte_fallback=true
```

byte_fallback 必开！

## Slide 7 · special tokens

```
<|endoftext|>      序列结束
<|pad|>            padding
<|im_start|> / <|im_end|>   chat 模板用
<|reserved_{i}|>   预留 (Llama-3 留了 256 个)
```

## Slide 8 · chat 模板

```
Llama-3 format:
  <|begin_of_text|>
  <|start_header_id|>user<|end_header_id|>
  你好<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  ...
```

base model 训练不加, instruct 阶段加。

## Slide 9 · 工程: tokenize 一致性

```
训练 / 推理用同 tokenizer
add_special_tokens=True for instruct
注意 chat_template (Jinja)
```

## Slide 10 · token-to-byte 比

```
GPT-2 (英文): 4 byte / token (压缩 4×)
Qwen (中文): 2 byte / token (UTF-8 byte ≈ 3-4/字)
```

数据规模换算需注意单位。

## Slide 11 · trade-off 总结

```
+ 大 vocab: 中文/code 友好, seq 短, 推理快
- 大 vocab: embedding 显存大, 小 vocab item underrepresented
妥协: 100k-150k 是当前最优 window
```

## Slide 12 · 推荐

```
玩具 (本 capstone): 复用 GPT-2 50257 BPE
正式: 自训 128k tiktoken style
中文重: 扩 zh 字符 + 用 Qwen tokenizer
```

## 参考
- Llama-3 tech report
- Qwen-2.5 tech report
