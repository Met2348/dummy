# Data Curation Implementation Plan

**Goal**: 完整实现数据 curation 学习包（12 lecture + 9 minimal/lib src + 6 测试 + 12 notebook + Capstone 1B-token 自制语料）

**Architecture**: 12 lectures = 11 主线 + 1 capstone。三轨代码：手写 minimal + 库（datasketch/sentencepiece/tiktoken）+ 工业级（FineWeb-Edu classifier 加载）。Windows native，复用 Module 1+2 cu130 环境。

**Tech Stack**: warcio + trafilatura + datasketch + simhash + sentencepiece + tiktoken + detoxify + presidio + fasttext

**Design 文档**: `docs/superpowers/specs/2026-06-04-data-curation-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/data-curation/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- warcio + trafilatura + fasttext-langdetect + datasketch + simhash
- sentencepiece + tiktoken + detoxify + presidio-{analyzer,anonymizer}
- datasets + huggingface-hub

### Task 1.3: environment/verify_env.py
- Part A: datasketch + sentencepiece + tiktoken import
- Part B: FineWeb-Edu classifier 模型加载
- Part C: BPE 训练 smoke（100 行训 1k merge）

### Task 1.4: src/common.py
- text normalize / url strip / language detect helpers
- jsonl writer / reader 流式接口

### Task 1.5: papers/ 12 个占位 + README index

### Commit: `chore: data-curation scaffold`

---

## Phase 2: L01-L02 CommonCrawl + 抽取

### Task 2.1: lectures/01-data-overview.md (22 slides)
- C4 / The Pile / RedPajama / FineWeb / DCLM 演化时间线
- 数据规模与质量 trade-off

### Task 2.2: lectures/02-commoncrawl.md (24 slides)
- WARC 格式 / trafilatura 主内容抽取 / fasttext language detect

### Task 2.3: src/cc_extract.py
- WARC 流式读取 + trafilatura 抽取 + jsonl 输出
- 1 GB segment ≈ 50k 文档处理

### Task 2.4: src/tests/test_warc_extract.py
- 注入 known doc，验证 title/content 抽取正确

### Task 2.5: notebooks/01-data-overview.ipynb + 02-commoncrawl.ipynb
- 时间线可视化 + 小型 WARC 1 文档抽取演示

### Commit + Tag: `data-cc-extract`

---

## Phase 3: L03 去重三方法

### Task 3.1: lectures/03-dedup.md (26 slides)
- MinHash / SimHash / SemDeDup 三方法对比
- Jaccard 相似度 vs Hamming 距离 vs cosine

### Task 3.2: src/minhash_dedup.py
- datasketch MinHash + LSH index
- 100k 文档 dedup 演示

### Task 3.3: src/simhash_dedup.py
- 手写 SimHash 算法
- 64-bit fingerprint

### Task 3.4: src/semdedup_demo.py
- sentence-transformers embedding + clustering
- 教学版（不调用 Faiss）

### Task 3.5: src/tests/test_dedup_recall.py
- 注入 100 重复文档，MinHash 召回 > 95%
- SimHash 假阳率 < 5%

### Task 3.6: notebooks/03-dedup.ipynb
- 3 方法去重前后对比 + 速度 benchmark

### Commit + Tag: `data-dedup`

---

## Phase 4: L04-L05 质量过滤 + PII

### Task 4.1: lectures/04-quality-filter.md (24 slides)
- C4 启发式（长度/标点/词典）
- FineWeb-Edu classifier 训练流程

### Task 4.2: src/quality_filter.py
- C4 启发式规则集
- FineWeb-Edu classifier 加载 + 打分

### Task 4.3: lectures/05-toxicity-pii.md (20 slides)
- Detoxify multi-label toxicity
- Presidio PII recognition

### Task 4.4: src/toxicity_pii_filter.py
- Detoxify 接口 + threshold
- Presidio analyzer + anonymizer pipeline

### Task 4.5: src/tests/test_classifier_acc.py + test_pii_coverage.py
- FineWeb-Edu 测试集 acc > 70%
- PII 替换率 > 90%

### Task 4.6: notebooks/04-quality-filter.ipynb + 05-toxicity-pii.ipynb

### Commit + Tag: `data-quality`

---

## Phase 5: L06-L08 tokenizer 三方法

### Task 5.1: lectures/06-tokenizer-bpe.md (28 slides)
- BPE 算法推导 + Karpathy minbpe 风格实现
- merge 规则可视化

### Task 5.2: src/bpe_trainer.py
- 手写 BPE 训练（参考 minbpe）
- 1M 字符训 1k merge

### Task 5.3: src/bpe_tiktoken.py
- tiktoken 加载 + encode/decode
- 与手写 BPE merge sequence 对比

### Task 5.4: lectures/07-tokenizer-spm.md (22 slides)
- SentencePiece + Unigram 算法
- 与 BPE 对比

### Task 5.5: src/spm_trainer.py
- sentencepiece 训练 + Unigram model

### Task 5.6: lectures/08-tokenizer-byte.md (18 slides)
- Byte-level / 多语言 tokenizer
- cl100k / Llama-3 128k 词表分析

### Task 5.7: src/vocab_compare.py
- 不同 tokenizer 在 5 种语言（en/zh/ja/code/math）压缩率对照

### Task 5.8: src/tests/test_bpe_consistency.py
- 手写 BPE vs tiktoken merge sequence 长度 < 5% 差异

### Task 5.9: notebooks/06-tokenizer-bpe.ipynb + 07-tokenizer-spm.ipynb + 08-tokenizer-byte.ipynb

### Commit + Tag: `data-tokenizer`

---

## Phase 6: L09-L10 数据配比 + 指令合成

### Task 6.1: lectures/09-data-mix.md (22 slides)
- DataComp 比赛 + Doremi 自动权重学习

### Task 6.2: src/data_mix_ablation.py
- 不同配比下小模型 perplexity 对照（mock 小模型）

### Task 6.3: lectures/10-instruction-data.md (24 slides)
- Alpaca / Self-Instruct / Magpie 流程

### Task 6.4: src/magpie_synthesis.py
- Magpie 模板自动生成 instruction 数据

### Task 6.5: notebooks/09-data-mix.ipynb + 10-instruction-data.ipynb

### Commit + Tag: `data-mix`

---

## Phase 7: L11-L12 陷阱 + Capstone + README

### Task 7.1: lectures/11-curation-pitfalls.md (20 slides)
- 污染 / 重复 / benchmark 泄漏 案例
- Llama-3 数据清洗细节

### Task 7.2: lectures/12-capstone-mini-corpus.md (24 slides)
- 完整 pipeline 设计 + 各阶段输出

### Task 7.3: src/capstone_mini_corpus.py
- 串联：CC 抽取 → MinHash 去重 → FineWeb-Edu 过滤 → SP tokenizer 训练
- 输入 1 CC segment，输出 1B-token jsonl + 32k SP tokenizer

### Task 7.4: src/tests/test_capstone_smoke.py
- 跑 1k 文档 mini pipeline，验证各阶段输出

### Task 7.5: notebooks/11-pitfalls.ipynb + 12-capstone.ipynb

### Task 7.6: learning/data-curation/README.md
- 标准结构（同 Module 2 系列）

### Commit + Tag: `data-curation`

---

## 验证清单（专题完成）

```powershell
python learning/data-curation/environment/verify_env.py
python -m pytest learning/data-curation/src/tests/ -v
jupyter nbconvert --execute --inplace learning/data-curation/notebooks/*.ipynb
python learning/data-curation/src/capstone_mini_corpus.py
```

预期：
- env 三段全 PASS
- 6 tests PASS
- 12 notebook 跑通
- Capstone 输出 1B-token jsonl + 32k SP tokenizer

---

## 总览

- 12 lectures × 平均 75 min = 15h slides
- 9 src 实现（不含 capstone）
- 6 tests
- 12 notebooks
- 1 capstone
- 预计 6 commit + 6 tag
- 总时长估计 14h（lecture 学习 + 7h notebook + 4h Capstone）
