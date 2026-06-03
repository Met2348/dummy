# L03 · 去重三方法 — MinHash / SimHash / SemDeDup

> 26 slides | 75 min | Data Curation 第 3 讲 ⭐⭐⭐⭐⭐

> 大模型训练里"去重"= 提质 + 防记忆 + 省算力，三鸟一石

---

## 学习目标

1. 理解为什么 dedup 对 LLM 至关重要
2. 掌握 MinHash + LSH 的数学原理
3. 推导 SimHash 与 cosine 的关系
4. 知道 SemDeDup（语义去重）何时用
5. 写出 3 种 dedup 的 Python 玩具实现

---

## Slide 1 · 为什么要去重

| 后果 | 例子 |
|------|------|
| **记忆放大** | 重复 100 次的诗 → 模型直接背 |
| **训练浪费** | C4 原始有 ~25% near-duplicate |
| **benchmark 泄漏** | GSM8K 题在 web 上重复成千次 |
| **ppl 虚低** | 测试集与训练集重叠 |

**目标**：保留多样性，剪掉冗余。

---

## Slide 2 · 三种粒度

```
exact dedup  →  字符完全相同
near dedup   →  少量字符差异（typo / 标点）
semantic     →  改写但意思相同（paraphrase）
```

| 方法 | 粒度 |
|------|------|
| `set([hash(text)])` | exact |
| **MinHash + LSH** | near |
| **SimHash** | near |
| **SemDeDup** | semantic |

---

## Slide 3 · Jaccard 相似度

两个文档 = 各自 n-gram 集合 A, B。

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

- J=1: 完全相同 set
- J=0: 无任何 n-gram 重叠
- 实务阈值：J ≥ 0.7 视为 near-duplicate

---

## Slide 4 · MinHash 直觉

**问题**：直接算 J(A,B) 需要遍历集合，1M doc 两两比较 → 5 × 10¹¹ 次

**MinHash 神奇性质**：
```
E[ min_h(A) == min_h(B) ] = J(A, B)
```

用 k 个独立哈希函数 → 得到 k 个最小值组成的"签名" → 比较签名的命中率 ≈ Jaccard。

---

## Slide 5 · MinHash 算法

```python
def minhash_sig(tokens, k=128):
    hashes = [hash_fn(i) for i in range(k)]  # k 个哈希
    sig = [min(h(t) for t in tokens) for h in hashes]
    return sig    # 长度 k 的向量
```

**实际**：用 `datasketch.MinHash`（C 加速）。

**签名估计**：
```python
agreement = sum(s1[i] == s2[i] for i in range(k)) / k
# agreement ≈ J(A, B)
```

---

## Slide 6 · LSH（Locality-Sensitive Hashing）

签名比较仍是 O(N²) — 1M 文档要 5 × 10¹¹ 次比较。

**LSH 思路**：把签名切成 b 个 band，每个 band 含 r 个值（k = b × r）。
```
两文档至少有 1 个 band 完全相同 → 候选 pair
```

只对候选 pair 算精确 Jaccard。

---

## Slide 7 · LSH 概率分析

文档相似度为 s（Jaccard）：

```
P(某 band 全等)        = s^r
P(某 band 全不等)      = 1 - s^r
P(所有 band 不等)      = (1 - s^r)^b
P(候选对) = 1 - (1 - s^r)^b   ← LSH 命中曲线
```

调 b, r 控制"陡门槛"。

---

## Slide 8 · LSH 命中曲线 (b=20, r=6)

```
s = 0.3 → P ≈ 1.5%      (低相似不命中)
s = 0.5 → P ≈ 30%
s = 0.7 → P ≈ 95%       ← 接近完美
s = 0.9 → P ≈ 100%
```

**threshold 大致** `(1/b)^(1/r) ≈ 0.7`。FineWeb 用 b=20 r=4。

---

## Slide 9 · datasketch API

```python
from datasketch import MinHash, MinHashLSH

mh = MinHash(num_perm=128)
for token in shingles:
    mh.update(token.encode())

lsh = MinHashLSH(threshold=0.7, num_perm=128)
lsh.insert("doc_id", mh)
duplicates = lsh.query(mh)
```

只 3 行就完成 ~1M 文档 dedup。

---

## Slide 10 · MinHash 实务细节

| 细节 | 推荐 |
|------|------|
| shingle | char 5-gram 或 word 1-gram |
| num_perm | 128 (FineWeb) / 256 (DCLM) |
| threshold | 0.7 (web) / 0.8 (code) |
| 跨 dump | 全局 index 大但能跨月去重 |

---

## Slide 11 · 数据规模 vs 内存

| 文档数 | num_perm=128 | LSH index 内存 |
|--------|-------------|---------------|
| 1M | 4 GB sig | ~6 GB |
| 100M | 400 GB | 600 GB |
| 1B | 4 TB | 多机分片 |

实务：FineWeb 用 datatrove + slurm 分片去重。

---

## Slide 12 · SimHash — 另一思路

Charikar 2002，Google 用了 10 年。

**思路**：
1. 给每个 token 算一个 hash bit vector
2. 用 token 的权重（tf-idf 或 1）加权求和 → 实值向量
3. 取符号 → 64-bit fingerprint

```
sim_hash(A) XOR sim_hash(B) 的 popcount = Hamming 距离
```

阈值通常 ≤ 3 视为 duplicate。

---

## Slide 13 · SimHash 算法

```python
def simhash(tokens, dim=64):
    v = [0.0] * dim
    for t in tokens:
        h = hash(t) & ((1 << dim) - 1)
        for i in range(dim):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    return sum((1 if v[i] > 0 else 0) << i for i in range(dim))
```

64-bit 整数 → 紧凑、比较快。

---

## Slide 14 · Hamming vs Jaccard

| | MinHash | SimHash |
|---|---------|---------|
| 距离 | Jaccard | cosine ≈ |
| 签名 | 128 × 64-bit | 1 × 64-bit |
| 阈值 | s ≥ 0.7 | hamming ≤ 3 |
| 内存 | 大 | 小 |
| 实务 | FineWeb | Google search 早期 |

LLM 时代 MinHash 更主流（参数可调、阈值清晰）。

---

## Slide 15 · SemDeDup（语义去重）

MinHash/SimHash 是**字面**层面。如果两文档表述不同但意思相同？

```
A: "The cat is on the mat."
B: "A feline rests upon the rug."
```

→ Jaccard ≈ 0，但语义相同。

**SemDeDup**：用 sentence-transformer 算 embedding，cosine ≥ 0.95 视为 dup。

---

## Slide 16 · SemDeDup 流程

```
1. 每文档 embedding 384-dim (MiniLM)
2. K-means 聚 1000 个簇
3. 每簇内两两 cosine
4. cosine ≥ 0.95 → 保留 1 个（短的优先）
```

DCLM 用 SemDeDup 在 MinHash 之后再做一遍。

---

## Slide 17 · SemDeDup 实务

| | 内存 | 速度 | 召回 |
|---|---|------|------|
| 384-dim float16 | 0.8 GB / 1M doc | GPU 必要 | 高（找回 paraphrase）|
| K-means 加速 | 5 min / 1M | Faiss-GPU | — |

适合：质量极高的场景；不适合：1B+ doc 全量（太贵）。

---

## Slide 18 · 三方法对比

| 方法 | 召回 | 误检 | 速度 | 内存 |
|------|------|------|------|------|
| exact hash | 低 | 0 | 极快 | 极小 |
| MinHash + LSH | 中-高 | 低 | 快 | 中 |
| SimHash | 中 | 低 | 极快 | 极小 |
| SemDeDup | **极高** | 中 | 慢 | 大 |

**业界配方**：MinHash 全量 → SemDeDup 在高质量子集。

---

## Slide 19 · Llama-3 dedup 配方

- 全 web：MinHash sig=128，threshold 0.8
- 13-gram exact match (benchmark contamination)
- 跨 dump 去重
- Code：SimHash + 文件 hash

未公开 SemDeDup，但 Llama-3 内部 ablation 表明 SemDeDup 给小模型 +1 pt。

---

## Slide 20 · FineWeb dedup 配方

- 全局 MinHash 跨 96 dumps
- num_perm=128, threshold 0.7
- 每 dump 内 + 跨 dump
- 最终保留 ~6% 的 doc（94% 是 dup！）

体现了 web 的"信息熵"实际很低。

---

## Slide 21 · "保留哪个"的策略

dedup 后保留 cluster 中的：

```
A) 最早出现        (按 ts)
B) 最长           (字符多)
C) 最高 quality   (FineWeb-Edu 分数)
D) 随机
```

FineWeb 用 A（最早），原因：最早往往是"原文"，后面是 scrape。

---

## Slide 22 · "近"到什么程度算重复？

| 任务 | 阈值 |
|------|------|
| web | J ≥ 0.7 |
| code | J ≥ 0.8（保留改写）|
| 数学 | J ≥ 0.9（保留同题不同解）|
| 对话 | J ≥ 0.6（更宽容）|

无统一答案，需 ablation 调出最佳。

---

## Slide 23 · 失败模式

| 失败 | 原因 | 补救 |
|------|------|------|
| 召回不够 | num_perm 太小 | 升到 256 |
| 误删多 | threshold 太低 | 升到 0.8 |
| 长文档漏 | shingle 太短 | 用 word 5-gram |
| 跨语言漏 | hash 函数不一 | 各语种独立 dedup |

---

## Slide 24 · 工程坑

```
1. shingle 决定一切 → 经验 char 5-gram 比 word 强
2. hash 函数固定 seed → 跨节点可复现
3. LSH 用 redis 后端 → 多机
4. 比对前 normalize（unicode NFKC + 小写）
```

---

## Slide 25 · 代码三轨

```
src/minhash_dedup.py     # datasketch 库
src/simhash_dedup.py     # 手写 64-bit
src/semdedup_demo.py     # sentence-transformers + cosine
```

教学版均 < 100 行。

---

## Slide 26 · 课后思考

1. 如果 num_perm 越大 → 召回越高、内存越大。临界点在哪？
2. SemDeDup 的 0.95 阈值是怎么定的？做一个 ablation。
3. 跨语言去重为什么难？
4. dedup 后语料缩了 90% → 模型效果会差吗？为什么？

---

## 参考

- Broder 1997 (MinHash)
- Charikar 2002 (SimHash)
- Abbas 2023 (SemDeDup)
- FineWeb tech report 2024
- DCLM 2024 (Apple)
