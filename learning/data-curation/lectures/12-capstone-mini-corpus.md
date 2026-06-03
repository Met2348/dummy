# L12 · Capstone — 1B token 自制语料

> 24 slides | 70 min | Data Curation 第 12 讲 ⭐⭐⭐⭐⭐ 毕业作品

> 从 WARC 到 jsonl + SP tokenizer 的端到端 pipeline

---

## 学习目标

1. 串联本系列前 11 讲的所有方法
2. 输出 1B token jsonl 高质量语料
3. 训出 32k SentencePiece tokenizer
4. 为后续专题 7 pretraining-recipe 提供输入

---

## Slide 1 · Capstone 目标

```
输入：1 CC segment (~1GB WARC.gz) 或者预下载子集
输出：
  - mini_corpus.jsonl.gz     ~50M token 高质量
  - mini_tokenizer.model     32k SentencePiece
  - report.md                各阶段缩水率
```

注：1B token 需 10-20GB WARC，本课用 50M token 玩具量级（保证可重复）。

---

## Slide 2 · Pipeline 全图

```
WARC.gz
  ↓ trafilatura + lang filter
extracted.jsonl
  ↓ MinHash dedup
deduped.jsonl
  ↓ heuristic + FineWeb-Edu classifier
quality.jsonl
  ↓ toxicity + PII anonymize
clean.jsonl
  ↓ SentencePiece 训词表 + encode
final.jsonl + tokenizer.model
```

每一步丢 30-80%，最终 ~2-5% 通过率。

---

## Slide 3 · 阶段产出与缩水

预期阶段缩水率：

```
WARC raw           1 GB  100%
extracted          80 MB  8%   (trafilatura)
deduped            65 MB  6.5% (MinHash 0.7)
quality           30 MB  3%   (heuristic + edu)
clean              28 MB  2.8% (PII 替换 ~7% doc)
tokenized          ~50 M token
```

→ 1B token 需 ~20GB WARC。

---

## Slide 4 · 各阶段时间预算（5090）

```
trafilatura          ~30 min / GB warc
MinHash             ~15 min / 1M docs
classifier          ~10 min / 1M docs (GPU)
PII detect          ~20 min / 1M docs
SP train (32k)      ~5 min  / 1B token
SP encode           ~30 min / 1B token
```

20GB WARC → 端到端 1-2 天单卡。

---

## Slide 5 · 代码组织

```
src/capstone_mini_corpus.py
    └── stage_1_extract()
    └── stage_2_dedup()
    └── stage_3_quality()
    └── stage_4_pii()
    └── stage_5_tokenize()
    └── main(args)
```

每 stage 输入/输出文件可独立 inspect。

---

## Slide 6 · 配置文件

```yaml
input:
  warc: data/seg.warc.gz
  out_dir: out/

extract:
  min_text_len: 200
  lang_filter: [en]

dedup:
  num_perm: 128
  threshold: 0.7

quality:
  use_fineweb_edu: true
  edu_threshold: 2.5    # 玩具放宽，prod 3.0

pii:
  use_presidio: false  # 玩具用 regex 兜底

tokenizer:
  vocab_size: 32000
  model_type: unigram
```

---

## Slide 7 · 各阶段单元测试

```
test_capstone_smoke.py
  ├── test_extract_smoke
  ├── test_dedup_smoke
  ├── test_quality_smoke
  ├── test_pii_smoke
  └── test_tokenize_smoke
```

1k 文档跑通 → 升级到 1M 文档。

---

## Slide 8 · 实务节选 — extract

```python
def stage_1_extract(warc_path, out_path, lang_filter, min_len):
    docs = []
    for doc in iter_warc(warc_path):
        if doc["lang"] not in lang_filter: continue
        if len(doc["text"]) < min_len: continue
        docs.append(doc)
    write_jsonl_gz(docs, out_path)
    return len(docs)
```

---

## Slide 9 · 实务节选 — dedup

```python
def stage_2_dedup(in_path, out_path, num_perm=128, threshold=0.7):
    docs = list(read_jsonl_gz(in_path))
    pairs = [(d["url"], d["text"]) for d in docs]
    kept_ids, dup_map = minhash_dedup.dedup(pairs, threshold, num_perm)
    kept = [d for d in docs if d["url"] in kept_ids]
    write_jsonl_gz(kept, out_path)
    return len(kept)
```

---

## Slide 10 · 实务节选 — quality

```python
def stage_3_quality(in_path, out_path, edu_threshold):
    out = []
    for doc in read_jsonl_gz(in_path):
        h = heuristic_score(doc["text"])
        if not h["passed"]: continue
        if edu_threshold:
            score = fineweb_edu_score(doc["text"][:2048])
            if score is None or score < edu_threshold: continue
        out.append(doc)
    write_jsonl_gz(out, out_path)
    return len(out)
```

---

## Slide 11 · 实务节选 — PII

```python
def stage_4_pii(in_path, out_path):
    out = []
    for doc in read_jsonl_gz(in_path):
        flagged, _ = is_toxic(doc["text"], 0.5)
        if flagged: continue
        cleaned, _ = anonymize(doc["text"])
        doc["text"] = cleaned
        out.append(doc)
    write_jsonl_gz(out, out_path)
    return len(out)
```

---

## Slide 12 · 实务节选 — tokenizer

```python
def stage_5_tokenize(in_path, out_path, model_path,
                     vocab_size=32000, model_type="unigram"):
    text_path = in_path.replace(".jsonl.gz", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for doc in read_jsonl_gz(in_path):
            f.write(doc["text"] + "\n")
    train_spm(text_path, vocab_size=vocab_size,
              model_type=model_type, out_path=model_path)
    sp = spm.SentencePieceProcessor(model_file=model_path)
    out_docs = []
    for doc in read_jsonl_gz(in_path):
        ids = sp.encode(doc["text"])
        doc["n_tokens"] = len(ids)
        out_docs.append(doc)
    write_jsonl_gz(out_docs, out_path)
```

---

## Slide 13 · report.md 生成

```
| Stage | n_docs | n_tokens | size_MB | ratio |
|-------|--------|----------|---------|-------|
| extract |  82,000 | 250M |  80 | 100% |
| dedup   |  65,000 | 200M |  65 | 80%  |
| quality |  30,000 |  85M |  30 | 37%  |
| clean   |  28,000 |  80M |  28 | 34%  |
| final   |  28,000 |  50M |  28 | -    |  ← 加 token 列
```

每阶段缩水率为优化 ablation 提供数据。

---

## Slide 14 · 单测试 vs 完整运行

```bash
# 玩具：1k 文档 smoke
python capstone_mini_corpus.py --smoke

# 完整：从 WARC 训练 1B
python capstone_mini_corpus.py \
    --warc data/seg.warc.gz \
    --out_dir out/mini-corpus \
    --vocab 32000
```

---

## Slide 15 · 验证 tokenizer 质量

```python
sp = spm.SentencePieceProcessor(model_file="m.model")
for text in test_set:
    ids = sp.encode(text)
    rec = sp.decode(ids)
    assert rec == text, "round-trip 失败"
```

实务：~98% round-trip（byte_fallback 兜底）。

---

## Slide 16 · Capstone 与下游对接

```
capstone 输出  →  专题 7 pretraining-recipe
                     ↓
              tokenize + dataloader
                     ↓
              训 270M Phi-tiny
                     ↓
              评测 MMLU-tiny / GSM8K-tiny
```

本课的 50M token 玩具量级训 Phi-tiny 不够；真训需 5B+。

---

## Slide 17 · 真训规模估算

```
50M token  → 训 80M-mini-GPT  →  ~30 min on 5090
500M token → 训 270M Phi-tiny →  ~5h
5B token  → 训 270M Phi-tiny  →  ~20h
```

Capstone 教 pipeline；真训成本在专题 7。

---

## Slide 18 · 报告卡片样例

```
== Capstone-1 Report ==
Input warc: data/seg.warc.gz (978 MB)
Pipeline:
  extract: 82k docs (75 MB)        | 78s
  dedup:   65k docs (65 MB) -20%   | 31s
  quality: 30k docs (28 MB) -57%   | 240s
  PII:     28k docs (27 MB) -7%    | 95s
  tokenizer: 32k vocab               | 280s
  encode:  50M tokens                | 380s
Total: 1104s (~18 min)
Output: mini_corpus.jsonl.gz + mini_tokenizer.model
```

---

## Slide 19 · 退出条件（Capstone PASS）

```
[ ] env verify_env.py 三段 PASS
[ ] 6 个 tests PASS
[ ] 12 notebook 均 jupyter nbconvert 跑通
[ ] capstone_mini_corpus.py 跑通 1k 文档 smoke
[ ] 输出 mini_corpus.jsonl.gz + tokenizer.model
[ ] report.md 列各阶段缩水率
```

完成 → tag `data-curation`。

---

## Slide 20 · 接下来

```
专题 1: data-curation     ✓ 本讲
专题 2: transformer-deep   ← 下一步
专题 3: moe-architecture
专题 4: ssm-hybrid
专题 5: long-context
专题 6: scaling-infra
专题 7: pretraining-recipe ← 用本课输出
专题 8: small-model-graduation ← 综合毕业
```

---

## Slide 21 · 真实工程的几个细节差异

```
单机 vs 集群
  ── ray / dask / spark 分片
  ── 单机也能跑 1B token，需 1-2 天

存储
  ── 中间结果尽量 jsonl.gz 流式
  ── 最终训练用 mds (mosaic streaming dataset)

监控
  ── 各 stage 缩水率应稳定
  ── 异常缩水 → 检查 lang_filter / 编码
```

---

## Slide 22 · 数据"经济学"

```
拿到原始 CC ~ 免费
trafilatura 抽 ~ $0.005 / GB
classifier 过滤 ~ $0.02 / 1M doc (GPU 时)
SP 训 ~ $5 单次（CPU 几小时）
Magpie 合成 ~ $50 / 1M instruction（GPU 时）
```

数据准备成本通常 < 1% 训练成本。但决定模型上限。

---

## Slide 23 · 经验法则

```
1. 早期：手工探索 → 不要被自动化绊住
2. 中期：建 ablation 框架 → 系统对比
3. 后期：固定 pipeline → 大规模执行
4. 永远：保留 metadata 供回溯
```

---

## Slide 24 · 致谢与 Next

至此 12 讲完成，专题 1 data-curation 收口 ✓

下一专题 2 **transformer-deep** 进入"造大模型"的核心 — RoPE / GQA / MLA / FlashAttention。

---

## 参考

- FineWeb tech report 2024
- DCLM technical report 2024
- Phi-4 technical report 2024
- Llama-3 technical report 2024
- DeepSeek-V3 technical report 2024
