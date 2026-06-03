# L04 · 质量过滤 — C4 启发式 + FineWeb-Edu classifier

> 24 slides | 70 min | Data Curation 第 4 讲 ⭐⭐⭐⭐⭐

---

## 学习目标

1. 写出 10 条 C4 启发式规则
2. 理解 FineWeb-Edu classifier 的训练流程
3. 知道 perplexity-based / classifier-based / score-based 三种过滤策略
4. 能为自己的语料定制一套过滤管线

---

## Slide 1 · 过滤的两种范式

```
启发式 (rule-based)   ─── C4 / Gopher rules
分类器 (model-based) ─── FineWeb-Edu / DCLM-baseline
```

实务：先启发式过 90%，再 classifier 精筛 → FineWeb-Edu 配方。

---

## Slide 2 · C4 启发式 10 条

| # | 规则 | 目的 |
|---|------|------|
| 1 | 段落 ≥ 5 句 | 去 SEO short blob |
| 2 | 末尾标点 [.?!"] | 去残缺段 |
| 3 | 删 `lorem ipsum` | 去 placeholder |
| 4 | 删 `{` | 去残留 JS |
| 5 | 删 boilerplate (terms / privacy / cookies) | 去导航 |
| 6 | 词数 ≥ 50 | 太短信息低 |
| 7 | 平均字长 5-15 | 去乱码 |
| 8 | 符号 / 字符 ≤ 0.1 | 去 ascii art |
| 9 | bullet 占段 ≤ 0.9 | 去列表页 |
| 10 | 重复 n-gram ≤ 0.2 | 去复读机 |

---

## Slide 3 · Gopher 规则补充

DeepMind 2022：
- mean word length 3-10
- 重复 4-gram ≤ 0.16
- 重复 5-10 char-gram ≤ 0.15
- 词典词占比 ≥ 0.8（防 token soup）

C4 + Gopher 是事实标准 baseline。

---

## Slide 4 · 启发式问题

| 问题 | 例 |
|------|-----|
| 误删高质量 | 短诗 / 公式 |
| 漏检低质量 | 自然语言的 SEO 文 |
| 跨语言 | 中文标点 ≠ 英文 |

→ 需要 classifier 补足。

---

## Slide 5 · perplexity 过滤

```
ppl(doc) = exp(NLL(doc))    # 小 LM 给的
```

- 高 ppl → 看着像乱码或低质 → 丢弃
- 低 ppl → 内容平庸但通顺 → 保留

CC-Net (FB 2019) 用 KenLM；2024+ 用小 LM (Pythia-160M 等)。

---

## Slide 6 · classifier 过滤

把"质量"当二分类（高 / 低）训：
- 用人工 / LLM 标注 ~50k 文档
- DistilBERT 蒸出 fast classifier
- 全量过滤

代表：FineWeb-Edu / DCLM-Baseline。

---

## Slide 7 · FineWeb-Edu 流程

```
1. Llama-3-70B-Instruct 给 500k 文档打 0-5 教育性分
2. 训 DistilBERT 回归
3. 阈值 ≥ 3.0 留
4. 全量过滤 15T → 1.3T tokens
```

效果：1.3T 训出来的模型 ≈ 15T 全量。

---

## Slide 8 · FineWeb-Edu 评分准则

Llama-3-70B prompt 节选：
> "Rate this web text on its educational value (0-5):
> - 0: gibberish / spam
> - 2: typical commercial / news
> - 4: structured learning material (textbook, well-written tutorial)
> - 5: highly educational, comparable to academic content"

prompt 设计是 classifier 质量的天花板。

---

## Slide 9 · 用 classifier 的代码

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceFW/fineweb-edu-classifier"
)
inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
score = model(**inputs).logits.item()
keep = score >= 3.0
```

batch 推理 GPU 上 ~1k doc/s。

---

## Slide 10 · DCLM-Baseline 配方

Apple 2024：

1. **fastText classifier** (高质量 / 低质量)
2. 高质量种子 = ELI5 + StackExchange + textbook
3. 低质量种子 = CC 随机
4. 全量过滤 240T → 4T tokens

fastText 比 BERT 快 100×，规模过滤的好选。

---

## Slide 11 · 三种 classifier 对比

| | fastText | DistilBERT | Llama-3-70B |
|---|---------|-----------|-------------|
| 速度 | 极快 | 中 | 慢 |
| 准确 | 中 | 高 | 极高 |
| 用法 | 一线全量 | 二线精筛 | 标注 |

DCLM = fastText 全量；FineWeb-Edu = DistilBERT 全量。

---

## Slide 12 · "教育性"vs "通用质量"

| 任务 | classifier 应训为 |
|------|---------------|
| 通用 LLM | 通用质量（流畅、信息密度）|
| **数学/推理** | 教育性（FineWeb-Edu 选这个）|
| 对话 | 对话风格 |
| 代码 | 编译通过 / 风格清晰 |

任务不同 → classifier 不同。

---

## Slide 13 · "数据"vs "模型"性价比

```
数据筛 4× → 模型小一档省 10×

15T web 训 70B vs 1.3T edu 训 70B
        ↓
   后者更优 + 算力少 10×
```

数据筛选投入与算力节省 ROI 极高。

---

## Slide 14 · 阈值的选择

```
threshold = 3.0   → 留 8.7%
threshold = 2.5   → 留 15%
threshold = 2.0   → 留 25%
```

太高 → 损失多样性；太低 → 噪声多。FineWeb-Edu 在 1.3T 选 3.0。

---

## Slide 15 · 多语言怎么办

FineWeb-Edu 主要英文。多语言：

```
1. 各语种独立 classifier
2. 用 Llama-3 多语版打分
3. 中文 classifier (Yi-1.5 / Qwen)
```

非英语高质量 classifier 是 2026 仍待解决的问题。

---

## Slide 16 · 数据筛 vs 模型大小

C/R 关系（Chinchilla）：

```
高质量数据 1B token ≈ 通用数据 5B token
```

数据每提质一档，可减小模型 0.3-0.5 倍体量。

---

## Slide 17 · 文本"修复"vs 删除

```
找到低质量段 → 删 OR 改 OR 重写？
```

- 简单删（FineWeb）
- 用 LLM 改写后保留（少数尝试，2025 H1 看到）
- 不可逆，FineWeb 倾向 conservative

---

## Slide 18 · 长度分布

```
< 50 tokens   → 删（信息量低）
50-1000      → 留（典型段落）
1000-4000    → 留（长文）
> 4000       → 截断或拆段
```

length filter 是最便宜也最有效的过滤。

---

## Slide 19 · 重复 n-gram

```python
def repeated_ngram_ratio(text, n=5):
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    return 1 - len(set(grams)) / max(len(grams), 1)
```

- 0.2 通常表示"复读机"
- C4 阈值 0.2，Gopher 0.16

---

## Slide 20 · 平均句长

```
< 5 词    → 列表 / 标题堆叠
5-30 词   → 正常段落
> 50 词   → 学术 / 法律
```

LLM 训练样本 5-30 是甜区。

---

## Slide 21 · benchmark contamination 过滤

dedup 章节讲了 13-gram exact match。但这里属于 **质量过滤** 的一环：

```
for benchmark_text in benchmarks:
    for 13gram in benchmark_text:
        if 13gram in doc:
            mark_contaminated(doc)
```

Llama-3 / Qwen 系列均做。

---

## Slide 22 · 域名 whitelist / blacklist

- whitelist: wikipedia / arxiv / stackoverflow → keep all
- blacklist: spam / adult / 已知低质量 → drop all

简单粗暴但很有效，每个 web crawl 都有自己的黑/白名单。

---

## Slide 23 · 过滤指标怎么评

```
1. 训小模型 (Pythia-160M) 10% 子集
2. 跑 MMLU-tiny / HellaSwag-tiny
3. 对比 baseline 子集
4. 选 acc 高的过滤方案
```

DCLM 比赛核心评测就是这个。

---

## Slide 24 · 实务管线总结

```
WARC → trafilatura → langfilter (en only)
     → dedupe (MinHash)
     → length filter (50-4000)
     → C4 + Gopher 启发式
     → FineWeb-Edu classifier (score ≥ 3.0)
     → benchmark contamination filter
     → tokenize
```

每一步丢 30-90% → 最终 ~3% 通过率。

---

## 参考

- C4: Raffel 2019
- Gopher: Rae 2022
- CC-Net: Wenzek 2019
- FineWeb-Edu: HuggingFace 2024
- DCLM-Baseline: Apple 2024
