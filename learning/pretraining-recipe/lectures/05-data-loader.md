# L05 · 数据加载与 shard

> 12 slides | 35 min ⭐⭐⭐⭐

## Slide 1 · 数据规模

```
1B token 训练: ~ 4 GB tokenized
100B: ~ 400 GB
1T: ~ 4 TB
```

需 memmap + shard 化。

## Slide 2 · shard 格式

```
shards/
  shard_000.bin   # uint16 / uint32 numpy
  shard_001.bin
  ...
shards.idx       # 索引: shard 内 token 数
```

## Slide 3 · numpy memmap

```python
import numpy as np
data = np.memmap("shard_000.bin", dtype=np.uint16, mode="r")
# 不实际加载, OS 按需 mmap
ids = data[100:200]
```

## Slide 4 · seq sampling

```python
def get_batch(data, seq_len, batch):
    n = len(data) - seq_len - 1
    idx = np.random.randint(0, n, size=batch)
    x = np.stack([data[i:i+seq_len] for i in idx])
    y = np.stack([data[i+1:i+seq_len+1] for i in idx])
    return torch.from_numpy(x), torch.from_numpy(y)
```

## Slide 5 · packing

```
长 doc 拼接, eos token 隔开
单 seq 多 doc, attention mask 隔离 (见 long-context L11)
```

## Slide 6 · resume 友好

```
保存:
  - epoch / step
  - random seed (rng state)
  - shard cursor (当前 shard + offset)
load 后从同位置继续, 数据顺序 deterministic
```

## Slide 7 · 多 worker DataLoader

```python
loader = DataLoader(
    dataset, batch_size=64, num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
```

## Slide 8 · stream from disk vs cache

```
小数据 (< 50 GB): pre-load RAM
中 (50-500 GB): memmap from SSD ✓
大 (> 500 GB): stream from object store (S3/GCS)
```

## Slide 9 · HF datasets

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu",
                   split="train", streaming=True)
for ex in ds:
    yield tokenize(ex["text"])
```

streaming=True 避免下载全部。

## Slide 10 · nanoGPT 数据格式

```python
# prepare.py:
import tiktoken
enc = tiktoken.get_encoding("gpt2")
all_ids = []
for text in corpus:
    all_ids.extend(enc.encode_ordinary(text))
    all_ids.append(enc.eot_token)
arr = np.array(all_ids, dtype=np.uint16)
arr.tofile("train.bin")
```

## Slide 11 · 大规模工程

```
Megatron-LM IndexedDataset:
  .bin + .idx 文件
  支持多文档采样 + bos/eos
DeepSpeed-Megatron: 一致
```

## Slide 12 · 总结

```
小项目: numpy memmap + tiktoken
中规模: HF datasets streaming
大规模: Megatron IndexedDataset
```

## 参考
- nanoGPT data pipeline
- Megatron IndexedDataset
