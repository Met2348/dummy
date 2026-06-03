# L01 · 数据为王 — 现代 LLM 语料演化全景

> 22 slides | 65 min | Data Curation 第 1 讲 ⭐⭐⭐⭐⭐

> "Garbage in, garbage out" 在 LLM 时代被放大 10000 倍

---

## 学习目标

1. 掌握 2019-2026 主流公开语料的演化时间线
2. 理解"数据规模 vs 质量"的本质 trade-off
3. 知道每个里程碑语料解决了什么问题、留下什么坑
4. 能为自己的训练任务选合适的开源语料起点

---

## Slide 1 · 时代分水岭

```
2019  → C4 (T5, ~750GB 清洗后)              规模启蒙
2020  → The Pile (~800GB 多源拼盘)          多样性启蒙
2022  → CommonCrawl 主流 + ROOTS (BLOOM)    多语言启蒙
2023  → RedPajama (Llama 配方复刻 ~1.2T tk) 开放再现启蒙
2024  → FineWeb 15T + FineWeb-Edu 1.3T      质量分类启蒙
2024  → DCLM (Apple, 11T)                   ablation 启蒙
2025  → 各家私有 + 合成数据为主              合成时代
```

**主线**：从"多就好" → "质量优先" → "可解释 ablation" → "合成补缺"。

---

## Slide 2 · C4 (2019) — 第一次"清洗"

- 起源：Google T5 论文
- 来源：CommonCrawl 单月 dump
- 清洗：~10 条启发式规则（删非英语 / 去 boilerplate / 长度过滤）
- 后果：750GB（原始 6TB 缩到 ~12%）
- 历史地位：第一次系统讲清洗

```text
"keep paragraphs ≥ 5 sentences, end with punctuation"
"drop pages with `lorem ipsum`"
"drop pages with curse words list"
```

---

## Slide 3 · The Pile (2020) — 多源拼盘

EleutherAI 出品，825GB，22 个 sub-corpus：

| 类型 | 子集示例 | 权重 |
|------|----------|------|
| Web | CC-derived (Pile-CC) | 18% |
| Academic | PubMed Central / ArXiv | 18% |
| Code | GitHub | 7.5% |
| Books | Books3 (后下架) | 12% |
| Dialog | StackExchange / HN | 5% |

**贡献**：多样性 > 单源量。

**遗产**：Books3 版权问题 → 2023 撤下，今天复现需替换。

---

## Slide 4 · RedPajama (2023) — 开放复刻 Llama

Together AI 出品，1.2T tokens，复刻 Llama-1 数据配方：

```
CommonCrawl  878 B   (73%)
C4           175 B
GitHub        59 B
Books         26 B
ArXiv         28 B
Wikipedia     24 B
StackEx       20 B
```

**意义**：第一次完整公开"商业模型量级"的数据。后来催生 LLM360 / OpenLLaMA。

---

## Slide 5 · FineWeb (2024) — 质量飞跃

HuggingFace 出品：

- **FineWeb**: 15T tokens，CC 96 dumps 全 ETL
- **FineWeb-Edu**: 1.3T tokens 子集，**教育价值 classifier 筛**

关键创新：用 Llama-3-70B 给样本打分（0-5 教育性）→ 蒸馏小 classifier → 全量过滤。

> 1.3T tokens 训练效果 ≈ 15T 全量 ⇒ **质量真比数量重要**。

---

## Slide 6 · DCLM (2024) — Ablation 启蒙

Apple + UW 出品，DataComp-LM 比赛：

- 提供 240T raw web → 给参赛者
- 鼓励"清洗策略"的 leaderboard
- 推出 **DCLM-Baseline 4T tokens**，7B 模型 64% MMLU（与 Llama-3 持平）

**意义**：让"清洗策略本身"成为可比较、可消融的研究对象。

---

## Slide 7 · 数据规模 — 法则与现实

Chinchilla (2022)：**每 1 参数应当配 ~20 token**。

| 模型 | 参数 | 推荐 token | 实训 token |
|------|------|-----------|-----------|
| Llama-3 8B | 8B | 160B | **15T**（94× 过训）|
| Llama-3 70B | 70B | 1.4T | **15T**（10× 过训）|

**2024 后**：人人都 over-train（小模型成本更香）。

---

## Slide 8 · 质量 vs 规模 — 两种极端

| 极端 | 例 | 后果 |
|------|-----|------|
| **规模派** | C4 / The Pile (无强过滤) | 训练快但 ppl 高 |
| **质量派** | Phi-1 (textbook only ~7B tk) | 小模型也能强 |

**实务**：先质量保底，再用规模 push。FineWeb-Edu 是当前 sweet spot。

---

## Slide 9 · 一份语料的 5 个生命周期

```
1. Crawl       (WARC, 原始 HTML)
2. Extract     (trafilatura, 去 boilerplate)
3. Dedupe      (MinHash/SimHash, 跨文档)
4. Filter      (语言/质量/PII/toxic)
5. Tokenize    (BPE/SP, 输出 token id)
```

本系列的 12 个 lecture 就按这条线展开。

---

## Slide 10 · 时间线总表

| 年 | 名 | 体量 | 关键词 |
|---|----|------|--------|
| 2019 | C4 | 750GB | 启发式清洗 |
| 2020 | Pile | 825GB | 多源 |
| 2022 | ROOTS | 1.6TB | 多语言 |
| 2023 | RedPajama | 1.2T tk | 复刻 |
| 2023 | SlimPajama | 627B tk | dedup 优化 |
| 2024 | FineWeb | 15T tk | 全 dump |
| 2024 | FineWeb-Edu | 1.3T tk | classifier |
| 2024 | DCLM | 4T tk | ablation |
| 2024 | Dolma (AI2) | 3T tk | 透明 |
| 2025 | 私有/合成 | — | Phi-4/Llama-4 |

---

## Slide 11 · 中文 / 多语言

- **ROOTS (2022)**: BLOOM 训练用，46 自然语言 + 13 编程语言
- **CCI / CCI3 (智源 2024)**: 数百 GB 中文高质量
- **OpenDataLab WuDao**: 早期 3TB 中文，质量参差
- **MAP-NEO**: 4.5T 中英多源

**结论**：中文公开语料质量普遍低于英文，Llama/Qwen 私有数据是主要差距来源。

---

## Slide 12 · 代码语料

| 名 | 量 | 来源 |
|----|----|------|
| The Stack v2 | 67TB | GitHub (BigCode) |
| StarCoder data | 3TB | The Stack 过滤 |
| GitHub Code (Pile) | 7.5% Pile | 早期 |

**清洗**：去重 + 去秘密 (API key) + 去 license 不兼容 + 编译可过滤。

---

## Slide 13 · 学术 / 数学语料

- **PeS2o (S2ORC)**: 40M 论文，~40B tk
- **ArXiv 全量**: 2.5M 论文，LaTeX 源码（数学训练核心）
- **OpenWebMath**: 14.7B tk，专门数学
- **AutoMathText**: 200GB

**Phi-3 / Llama-3**: 数学 token 占比 5-10% 是数学能力涨的关键。

---

## Slide 14 · 指令数据演化

```
2022  FLAN-T5         手工 1.8k 任务模板
2023  Alpaca           175 seed → GPT-3.5 自扩 52k
2023  ShareGPT         人机真实对话
2024  Magpie           LLM 自问自答合成（无 seed）
2024  Tülu 3           Allen AI 高质量混搭
2024  Open-Hermes      多源精选
2025  各家私有为主
```

**趋势**：从"人工模板" → "GPT 蒸馏" → "无 seed 全合成"。

---

## Slide 15 · License 与版权

| 类型 | 案例 | 风险 |
|------|------|------|
| 开放 CC | Wikipedia | 低 |
| 学术 | ArXiv | 中（个别期刊禁爬）|
| 用户生成 | Reddit / Twitter | 高（2024 后陆续禁）|
| 书籍 | Books3 | 极高（2023 被 takedown）|

> 商业训练 — 律师在场是基本要求。教学复现可用 CC-BY 子集。

---

## Slide 16 · 数据 "污染" — benchmark 泄漏

定义：**测试集** 出现在训练集 → 评分虚高。

经典案例：
- C4 含 SuperGLUE 测试样本 → T5 评分有水分
- The Pile 含 LAMBADA → 早期 GPT 评分难比较
- GSM8K 整题文本被收录于多个 web crawl

**防御**：训练前对常用 benchmark 做 n-gram exact match 删除（Llama-3 做了 13-gram dedup）。

---

## Slide 17 · 数据 "新鲜度"

- LLM 的"知识截止"由 crawl date 决定
- CC 每月 1 dump → 工程上选最近 2-3 dump 增量
- 2024 后："continual pretraining"成为常规：每月用新数据增训

---

## Slide 18 · 数据 "毒性" 与 "PII"

| 类别 | 例 | 处理 |
|------|-----|------|
| Toxic | 仇恨 / 暴力 / 性 | Detoxify multi-label |
| PII | 邮箱 / 电话 / 身份证 | Presidio + regex |
| Bias | 种族 / 性别刻板印象 | RLHF 后阶段处理 |

清洗时 **要 不要 删 vs 要 不要 mask**？保留 mask 后版本更利于模型理解"什么是 PII"。

---

## Slide 19 · 数据配比 — 简单到复杂

```
naive       1:1:1 平均
manual      Llama-1: web 73% / code 5% / wiki 2% ...
Doremi      用小代理模型自动学权重
ODM         Mixtral 路径，动态采样
```

L09 会专门讲 Doremi 推导。

---

## Slide 20 · 合成数据时代（2024-）

```
Phi-1   textbook + GPT-3.5 合成 7B tk          → 1.3B 模型超 7B
Phi-2   GPT-4 合成 + filtered web ~250B tk
Phi-3   持续合成扩展
Phi-4   2024 重申: "高质量合成 > web 1 order"
Llama-3 数学/code 部分合成
Qwen-2.5 合成 instruction 数千万
```

**核心法则**：合成 = teacher LLM × 任务模板 × 多样性 reward。

---

## Slide 21 · 本系列的 12 lecture 路线

```
L01 · 数据时代鸟瞰 (本讲)
L02 · CommonCrawl + 抽取
L03 · 去重 (MinHash/SimHash/SemDeDup)
L04 · 质量过滤 (C4 启发式 + FineWeb-Edu)
L05 · 毒性 + PII
L06 · BPE tokenizer (含手写)
L07 · SentencePiece + Unigram
L08 · Byte-level + 多语言 tokenizer
L09 · 数据配比 (Doremi / DCLM)
L10 · 指令数据合成 (Magpie)
L11 · 陷阱合集
L12 · Capstone: 1B token 自制语料
```

---

## Slide 22 · 课后思考

1. 为什么 FineWeb-Edu 1.3T 训练效果 ≈ FineWeb 15T？
2. Chinchilla 的 20:1 法则在 2026 还适用吗？
3. 如果你只能选一份开源语料从 0 训 1B 模型，选哪个？为什么？
4. 合成数据有理论上限吗？

---

## 参考

- C4: Raffel et al. 2019 (JMLR 21)
- The Pile: Gao et al. 2020
- RedPajama: Together AI 2023
- FineWeb / FineWeb-Edu: Penedo et al. 2024
- DCLM: Li et al. 2024 (Apple)
- Chinchilla: Hoffmann et al. 2022
