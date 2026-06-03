# Data Curation 学习包

> Module 3「造大模型」**专题 1 / 8** — 数据准备：从 CommonCrawl 到 1B-token 高质量语料

| 元数据 | 值 |
|--------|----|
| 方法数 | 14 |
| Lecture | 12 |
| 估时 | 14h |
| 环境 | Windows native，复用 Module 1+2 cu130 |
| Design | [`docs/superpowers/specs/2026-06-04-data-curation-design.md`](../../docs/superpowers/specs/2026-06-04-data-curation-design.md) |
| Plan | [`docs/superpowers/plans/2026-06-04-data-curation.md`](../../docs/superpowers/plans/2026-06-04-data-curation.md) |

---

## 专题概览

12 Lecture，覆盖现代 LLM 语料的**完整 ETL 流水**：

| # | Lecture | 关键方法 | 时长 |
|---|---------|---------|------|
| L01 | 数据时代鸟瞰 | C4 → Pile → RedPajama → FineWeb → DCLM | 1h |
| L02 | CommonCrawl + 抽取 | WARC / trafilatura / fasttext | 1h |
| L03 | 去重三方法 | **MinHash + LSH** / SimHash / SemDeDup | 1.5h |
| L04 | 质量过滤 | C4 + Gopher 启发式 + **FineWeb-Edu classifier** | 1.5h |
| L05 | 毒性 + PII | Detoxify + Presidio | 1h |
| L06 | BPE Tokenizer | 手写 + **tiktoken** | 1.5h |
| L07 | SentencePiece + Unigram | spm + subword reg | 1h |
| L08 | Byte-level / 多语言 | gpt2 / cl100k / o200k 对照 | 1h |
| L09 | 数据配比 | **Doremi** / DCLM ablation | 1.5h |
| L10 | 指令数据合成 | Alpaca / Self-Instruct / **Magpie** | 1.5h |
| L11 | 陷阱合集 | benchmark contamination / 重复 / PII / 合成近亲 | 1h |
| L12 | **Capstone — 1B token 自制语料** | 端到端 pipeline | 1h |

---

## 学习路径

```
L01 时代鸟瞰
   │
   ↓
L02 抽取 ─► L03 去重 ─► L04 质量 ─► L05 毒性/PII
                                          │
                                          ↓
L06 BPE → L07 SP → L08 byte-level
                                          │
                                          ↓
                              L09 配比 + L10 合成
                                          │
                                          ↓
                              L11 陷阱 → L12 Capstone
```

---

## 目录结构

```
learning/data-curation/
├── README.md (本文件)
├── environment/
│   ├── requirements.txt
│   └── verify_env.py          # 三段式自检
├── papers/
│   └── README.md              # 12 论文 index
├── lectures/                  # 12 个 PPT-style markdown
│   ├── 01-data-overview.md
│   ├── 02-commoncrawl.md
│   ├── 03-dedup.md
│   ├── 04-quality-filter.md
│   ├── 05-toxicity-pii.md
│   ├── 06-tokenizer-bpe.md
│   ├── 07-tokenizer-spm.md
│   ├── 08-tokenizer-byte.md
│   ├── 09-data-mix.md
│   ├── 10-instruction-data.md
│   ├── 11-curation-pitfalls.md
│   └── 12-capstone-mini-corpus.md
├── src/
│   ├── common.py
│   ├── cc_extract.py
│   ├── minhash_dedup.py        # datasketch
│   ├── simhash_dedup.py        # 手写 64-bit
│   ├── semdedup_demo.py        # sentence-transformers
│   ├── quality_filter.py       # C4 + Gopher + FineWeb-Edu
│   ├── toxicity_pii_filter.py  # Detoxify + Presidio
│   ├── bpe_trainer.py          # 手写 BBPE
│   ├── bpe_tiktoken.py         # tiktoken 对照
│   ├── spm_trainer.py          # SentencePiece
│   ├── vocab_compare.py        # 多语言压缩率
│   ├── data_mix_ablation.py    # 配比 ablation 玩具
│   ├── magpie_synthesis.py     # Magpie 合成 (mock + real)
│   ├── capstone_mini_corpus.py # 端到端 pipeline
│   └── tests/
│       ├── test_warc_extract.py
│       ├── test_dedup_recall.py
│       ├── test_classifier_acc.py
│       ├── test_bpe_consistency.py
│       └── test_capstone_smoke.py
└── notebooks/                  # 12 个 ipynb 对应 lecture
```

---

## 环境配置

> ⚠️ 复用 Module 1+2 的 cu130 torch 环境；额外加 trafilatura / datasketch / sentencepiece 等。

```powershell
# 进入对应 venv
& C:\Workspace\dummy\learning\reasoning-r1\.venv\Scripts\Activate.ps1
pip install -r learning/data-curation/environment/requirements.txt

# 验证
python learning/data-curation/environment/verify_env.py
```

预期：Part A/B/C 三段 PASS（Part B 首次会下载 FineWeb-Edu classifier ~500MB，可跳过）。

---

## 横向对比表

### Dedup 方法

| 方法 | 粒度 | 召回 | 误检 | 内存 | 速度 |
|------|------|------|------|------|------|
| exact hash | character | 低 | 0 | 极小 | 极快 |
| **MinHash + LSH** | token-set | 高 | 低 | 中 | 快 |
| SimHash | token-set | 中 | 低 | 极小 | 极快 |
| SemDeDup | semantic | 极高 | 中 | 大 | 慢 |

**业界配方**：MinHash 全量 → SemDeDup 在高质量子集再过一遍。

### Tokenizer 对照

| Tokenizer | 词表 | 算法 | 用 | 英语 c/t | 中文 c/t |
|-----------|------|------|---|---------|---------|
| gpt2 | 50k | BBPE | GPT-2 | 4.0 | 0.4 |
| cl100k_base | 100k | BBPE | GPT-3.5/4 | 4.2 | 1.4 |
| o200k_base | 200k | BBPE | GPT-4o | 4.5 | 1.8 |
| Llama-3 | 128k | tiktoken-style | Llama-3 | 4.3 | 1.6 |
| Qwen-2.5 | 152k | tiktoken-style | Qwen | 4.3 | 2.2 |

---

## 学习目标自测

1. C4 / Pile / FineWeb / DCLM 各属于哪个阶段？
2. 写出 MinHash + LSH 找候选对的概率公式 `P(候选) = 1 - (1 - s^r)^b`。
3. 为什么 SimHash 用 Hamming 而 MinHash 用 Jaccard？
4. FineWeb-Edu 阈值 3.0 留多少比例数据？
5. C4 启发式 10 条至少能背出 6 条？
6. Detoxify 与 Presidio 各自解决什么问题？
7. Byte-level BPE 与 char-level BPE 在中文场景的根本差异？
8. Llama-3 为什么从 SentencePiece 切到 tiktoken-style BBPE？
9. Doremi 算法的目标函数？
10. Magpie 与 Self-Instruct 的核心区别？
11. benchmark contamination 用 n-gram = ?
12. 数据"经济学"中各阶段成本量级？

---

## Git 里程碑

| Tag | 时机 | 内容 |
|-----|------|------|
| `data-cc-extract` | Phase 2 | L01-L02 + cc_extract |
| `data-dedup` | Phase 3 | L03 + 3 个 dedup 实现 |
| `data-quality` | Phase 4 | L04-L05 + classifier + PII |
| `data-tokenizer` | Phase 5 | L06-L08 + 3 tokenizer 实现 |
| `data-mix` | Phase 6 | L09-L10 + mix + Magpie |
| `data-curation` | Phase 7 | L11-L12 + Capstone 收口 ⭐ |

---

## 验证命令

```powershell
# 1. 环境
python learning/data-curation/environment/verify_env.py

# 2. 测试
python -m pytest learning/data-curation/src/tests/ -v

# 3. 全部 notebook
jupyter nbconvert --execute --inplace learning/data-curation/notebooks/*.ipynb

# 4. Capstone smoke
python learning/data-curation/src/capstone_mini_corpus.py --smoke
```

---

## 实用入口

- 想跑 BBPE 手写: `python src/bpe_trainer.py --demo --vocab-size 400`
- 想看 MinHash 召回: `python src/minhash_dedup.py --demo`
- 想看 tokenizer 压缩率: `python src/vocab_compare.py`
- 想跑全流水: `python src/capstone_mini_corpus.py --smoke`
- 想看真实 Magpie: `python src/magpie_synthesis.py --real --n 5`

---

## 与其他专题的关系

```
本专题 data-curation
    │
    ├─→ 输出 ── corpus.jsonl + SP tokenizer
    │            │
    │            ↓
    │       专题 7 pretraining-recipe  (从 0 训 270M Phi-tiny)
    │            │
    │            ↓
    │       专题 8 small-model-graduation (五部曲毕业)
    │
    └─→ 概念 ── benchmark contamination / 13-gram
                 │
                 ↓
            专题 5 long-context (NIAH 评测时也要考虑)
```

完成本专题后可继续 **专题 2 transformer-deep** — 学 RoPE / GQA / FlashAttention。
