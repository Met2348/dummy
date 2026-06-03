# Data Curation 学习专题 — 设计文档

> **承接**: Module 1 PEFT + Module 2 RL/对齐+推理（已完成）
> **本专题**: Module 3 造大模型系列第 1 站 — 数据 + tokenizer
> **战略地位**: 全系列地基，"无数据无大模型"
> **总体规划**: `docs/superpowers/plans/2026-06-04-pretraining-architecture-series.md`

---

## 1. 专题定位

LLM 第一步从来不是模型——是**数据**。本专题覆盖 2024-2026 工业级数据 curation 全流程：从 CommonCrawl 抽取、去重、质量过滤、毒性/PII 过滤，到 BPE/SentencePiece tokenizer 训练。

### 1.1 为什么 PEFT/RL 出身者需要数据课

1. **PEFT/RL 全是用别人的数据**：FineWeb / Anthropic-HH 已 ready
2. **造大模型必须自己 curate**：data quality 直接决定 ceiling
3. **2024-2026 共识**: 「data > model」（Phi-4 / Llama-3.3 / FineWeb-Edu 全是证据）
4. **小模型时代尤其依赖**：Phi 系列「教科书数据」是核心 trick

### 1.2 本专题的"双面性"

- **工程 70%**：CommonCrawl 抽取 / MinHash dedup / 数据 pipeline
- **算法 30%**：BPE 推导 / Quality classifier / Doremi 数据配比

---

## 2. 方法清单（14 种）

| # | 方法 | 年份 | 论文/出处 | 核心 idea |
|---|------|------|---------|----------|
| 1 | **CommonCrawl WARC 抽取** | 2008+ | CC | web 原始数据获取 |
| 2 | **trafilatura HTML 抽取** | 2021 | — | 主内容 vs 噪声 |
| 3 | **fasttext language detection** | 2017 | Joulin | 语言识别 |
| 4 | **C4 启发式过滤** | 2019 | T5 | 长度/标点/词典启发 |
| 5 | **MinHash + LSH dedup** | 1998+ | Broder | 集合相似度近邻 |
| 6 | **SimHash dedup** | 2002 | Charikar | 局部敏感哈希 |
| 7 | **SemDeDup** | 2024 | Meta | embedding 去重 |
| 8 | **FineWeb-Edu classifier** | 2024 | HF | 教育值打分 |
| 9 | **Detoxify / 毒性过滤** | 2020 | Unitary | toxicity scoring |
| 10 | **Presidio / PII redaction** | 2020 | Microsoft | PII 识别 + 脱敏 |
| 11 | **BPE 算法** | 2016 | Sennrich | byte-pair merge |
| 12 | **SentencePiece + Unigram** | 2018 | Kudo | 语言无关分词 |
| 13 | **DataComp / Doremi** | 2024 | — | 数据配比学习 |
| 14 | **Magpie 指令数据合成** | 2024 | — | 模板自动生成 |

---

## 3. Lecture 结构（12 篇 = 11 主线 + 1 capstone）

| Lecture | 主题 | 主方法 | 时长 |
|---------|------|--------|------|
| **L01** LLM 数据全景 | C4 → Pile → FineWeb 演化 | 60 min |
| **L02** CommonCrawl 抽取 | WARC + trafilatura | 75 min |
| **L03** 去重策略 | MinHash + SimHash + SemDeDup | 90 min |
| **L04** 质量过滤 | C4 启发 + FineWeb-Edu classifier | 75 min |
| **L05** 毒性 / PII | Detoxify + Presidio | 60 min |
| **L06** BPE 算法 | tiktoken / minbpe 推导 | 90 min |
| **L07** SentencePiece + Unigram | spm 训练 + 与 BPE 对比 | 75 min |
| **L08** Byte-level / 多语言 | cl100k / Llama-3 128k 词表分析 | 60 min |
| **L09** 数据配比 | DataComp / Doremi 自动权重 | 75 min |
| **L10** 指令数据合成 | Alpaca / Self-Instruct / Magpie | 75 min |
| **L11** 数据陷阱合集 | 污染 / 重复 / benchmark 泄漏 | 60 min |
| **L12** Capstone：1B-token 自制语料 | 完整 pipeline | 120 min |

**总学时**: 12 lecture × 平均 75 min + 6h notebook ≈ 14 hours

---

## 4. Lecture 模板（PPT-style，每篇 18-26 slides）

```markdown
# Lecture N: {方法名}

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机（这个方法解决什么问题）
## Slide 3-4: 核心算法（MinHash / BPE merge 等）
## Slide 5-7: 直觉解释 / 几何意义（图解）
## Slide 8-12: 工程细节（pipeline / 速度 / 显存）
## Slide 13-16: 代码逐行（手写 vs 库）
## Slide 17-19: 实战数据（C4 vs FineWeb 比较）
## Slide 20-22: 陷阱与警示（这个方法什么时候不 work）
## Slide 23-25: 思考题 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | 手写 minimal | 库 1 | 库 2 / 工业 |
|------|-------------|------|-----------|
| WARC 抽取 | ✅ raw 解析 | ✅ warcio | — |
| trafilatura | — | ✅ trafilatura | — |
| MinHash dedup | ✅ 手写 hash | ✅ datasketch | — |
| SimHash | ✅ 手写 | — | — |
| BPE | ✅ minbpe 风格 | ✅ tiktoken | — |
| SentencePiece | — | ✅ sentencepiece | — |
| Quality classifier | ✅ fasttext 训 | — | ✅ FineWeb-Edu 加载 |
| 数据 mix | ✅ Doremi 简化 | — | — |
| 指令合成 | ✅ Magpie 模板 | — | — |

**目录约定**:
- `{method}_minimal.py` — 手写
- `{method}_lib.py` — 库对照
- `{method}_industry.py` — 工业级（仅复杂方法）

---

## 6. 一致性测试

```python
def test_warc_extract_correctness():  # 抽 known doc 验证 title/content
def test_minhash_dedup_recall():       # 注入 100 重复，召回 > 95%
def test_simhash_collision_rate():     # 假阳率 < 5%
def test_bpe_vs_tiktoken():            # 自训 BPE merge sequence 一致
def test_classifier_accuracy():        # FineWeb-Edu 测试集 acc > 70%
def test_pii_redaction_coverage():     # PII 替换率 > 90%
```

---

## 7. Notebook 结构（12 个）

每个 lecture 一个 ipynb：
1. import + 数据加载
2. 算法核心可视化（MinHash 哈希分布 / BPE merge tree）
3. minimal 实现 step-by-step
4. mini pipeline（10MB 子集）
5. 库对照（datasketch / tiktoken / sentencepiece）
6. 关键指标（去重前后大小 / quality 分布 / 词表压缩率）
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt (Windows native)
warcio>=1.7
trafilatura>=1.12
fasttext-langdetect>=1.0
datasketch>=1.6
simhash>=2.1
sentencepiece>=0.2
tiktoken>=0.7
detoxify>=0.5
presidio-analyzer>=2.2 presidio-anonymizer>=2.2
datasets>=2.20
huggingface-hub>=0.24
```

**verify_env.py 三段式**:
- Part A: 基础（datasketch + sentencepiece + tiktoken）
- Part B: classifier 模型加载（FineWeb-Edu）
- Part C: BPE 训练 smoke test（100 行训 1k merge）

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `data-cc-extract` | L01-L02: CC + trafilatura | 3 |
| `data-dedup` | L03: MinHash + SimHash + SemDeDup | 3 |
| `data-quality` | L04-L05: classifier + PII | 3 |
| `data-tokenizer` | L06-L08: BPE + spm + multilingual | 4 |
| `data-mix` | L09-L10: Doremi + 指令合成 | 3 |
| `data-curation` | L11-L12: Capstone + README | 3 |

---

## 10. 跨专题衔接

### 上游
- Module 1+2 完成（无技术依赖，仅认知前置）

### 下游
- 专题 2 Transformer：用本专题 32k SP tokenizer
- 专题 7 Pretraining：用本专题 1B-token 自制语料
- 专题 8 Graduation：用本专题语料训 student

### 跨专题对照表预留位
| 数据集 | size | 用途 |
|--------|------|------|
| 自制 1B | 1B token | 专题 2 GPT-mini |
| 自制 + Phi 合成 5B | 5B token | 专题 7 Phi-tiny |

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| CC 单 dump 80TB 太大 | 高 | 高 | 只下 1 segment ~200GB |
| 高速 SSD IO 瓶颈 | 中 | 中 | 流式处理 + 多线程 |
| FineWeb-Edu classifier 准确率 | 中 | 中 | 期望管理 70% |
| BPE 与 tiktoken 不严格一致 | 低 | 低 | 用 sequence 长度对比代替 byte-exact |
| 法律 / license | 中 | 高 | 注明 license + 不公开 raw 输出 |
| Presidio PII 漏检 | 中 | 中 | 多 entity 配 + 人工抽检 |

---

## 12. 论文 / 资料占位

```
papers/
├── 01-c4-2019.md                    # Raffel T5 数据 ablation
├── 02-the-pile-2020.md              # EleutherAI Pile
├── 03-fineweb-2024.md               # HF FineWeb 15T
├── 04-fineweb-edu-2024.md           # 教育数据子集
├── 05-broder-1998-minhash.md
├── 06-charikar-2002-simhash.md
├── 07-semdedup-2024.md              # Meta embedding dedup
├── 08-sennrich-2016-bpe.md
├── 09-kudo-2018-sentencepiece.md
├── 10-doremi-2023.md                # 数据配比
├── 11-self-instruct-2022.md
├── 12-magpie-2024.md
└── README.md
```

---

## 13. 实施方案

按 plan 文件 `2026-06-04-data-curation.md` 的 7 个 Phase 推进：

- Phase 1: 基础设施（目录 + env + tests 骨架）
- Phase 2: L01-L02 CC + trafilatura（tag `data-cc-extract`）
- Phase 3: L03 去重三方法（tag `data-dedup`）
- Phase 4: L04-L05 质量 + PII（tag `data-quality`）
- Phase 5: L06-L08 tokenizer 三方法（tag `data-tokenizer`）
- Phase 6: L09-L10 配比 + 指令合成（tag `data-mix`）
- Phase 7: L11-L12 陷阱 + Capstone + README（tag `data-curation`）

---

## 设计签字

- **设计日期**: 2026-06-04
- **设计者**: Claude Opus 4.7
- **审阅者**: 用户（待）
