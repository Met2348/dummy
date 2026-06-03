# L11 · 数据清洗陷阱合集

> 20 slides | 60 min | Data Curation 第 11 讲 ⭐⭐⭐⭐

> 那些让你浪费 GPU 周的真实"鬼故事"

---

## 学习目标

1. 识别 LLM 训练数据中 8 类典型陷阱
2. 知道 Llama-3 / DeepSeek-V3 / Phi-4 报告里"血与泪"
3. 写防御 checklist

---

## Slide 1 · 陷阱 1：benchmark 污染

```
GSM8K 训练集 51 题 出现在 CC 中 → exact match
GSM8K 测试集 31 题 出现在 CC 中 → 模型背了答案
```

模型 GSM8K 9X% 是因为背了，不是会推理。

**防御**：13-gram exact match 删除（Llama-3 配方）。

---

## Slide 2 · 陷阱 2：重复

```
The Pile 内部 60% 重复（高频名言 / 法律条款 / 维基镜像）
CC 内部 25% 跨网站近重复
跨 dump 不去重 → 重复率翻倍
```

不 dedup 直接训：ppl 虚低、记忆放大。

防御：MinHash 跨 dump 去重，threshold 0.7 起步。

---

## Slide 3 · 陷阱 3：长尾低质

```
domain-squatting 页    "this domain is for sale"
search engine results  "showing 1-10 of 1,000"
error pages            "404 not found"
广告内容               "Sponsored content"
```

trafilatura 不能完全去除，需 classifier 补刀。

---

## Slide 4 · 陷阱 4：编码混乱

```
- mojibake (双重编码: UTF-8 in CP1252)
- 全角 / 半角混杂
- Unicode replacement char (U+FFFD) 大量出现
```

防御：normalize NFKC + 删 U+FFFD > 1% 的文档。

---

## Slide 5 · 陷阱 5：少数语言被淹

LLM 训练 80% 英文 → 中文 / 日文 / 韩文表现差。

实务：
- 每语言独立 dedup（不同文化下 "重复" 定义不同）
- 多语言 classifier 训练（FineWeb-Edu 只英文）

DeepSeek-V3 报告：中文专门 oversample 5×。

---

## Slide 6 · 陷阱 6：代码语料的"秘密泄漏"

GitHub 上有：
- API key / token (10万+条)
- 私钥 / 证书
- 数据库连接字符串

防御：detect-secrets / trufflehog 工具 + regex 兜底。

Phi-3 报告：删了 ~1.2M 文件含 secrets。

---

## Slide 7 · 陷阱 7：版权 / license

```
Books3 (2021)  ← 2023 被 takedown
StackOverflow (2024)  ← 改 license 限 LLM
Reddit (2024)  ← 收费、限爬
Twitter/X       ← 完全关闭
```

随时间 license 收紧 → 旧训练集可能违法。

---

## Slide 8 · 陷阱 8：JS / SPA 抓不到

```
现代 web 50% 是 SPA → CC 抓初始 HTML 没内容
后果：内容多样性下降
```

无完美对策；FineWeb 接受这个损失。Google 用 Chromium render 但成本高。

---

## Slide 9 · Llama-3 报告血泪

```
1. 13-gram exact match → 删 ~2% 数据
2. 跨语言 dedup → 中文降 60%
3. URL filter → 删 ~100 万 domain
4. 数学专项 oversample 5%
5. 数学 token 切分 → 每数字 1 token
6. Annealing 末 5% 用 textbook 子集
```

每条都是踩坑得来。

---

## Slide 10 · Phi-4 报告血泪

```
1. 合成数据 50% → web 30% → 其他 20%
2. 合成数据由 Phi-3 + GPT-4 联合生成
3. 多样性 prompt 设计是核心
4. instruction → response 长度 control
5. 多轮对话 30%
6. textbook 子集压在 mid-training
```

Phi-4 把"合成"做到极致。

---

## Slide 11 · DeepSeek-V3 报告血泪

```
1. 14.8T 训练 token
2. 中文 4T / 英文 4T / code 2T / 数学 1T / 其他
3. FineWeb-Edu 风格 classifier
4. ROOTS dedup 跨语言
5. mid-training 注入数学 / code 加强
```

---

## Slide 12 · 陷阱 9：tokenizer 训练与训练数据不匹配

```
tokenizer 在 web 训 → 100% web
模型在 mid-training 切 textbook → 词表不匹配
↓
textbook 压缩率 0.8 (vs web 4.0) → 训练成本暴涨
```

防御：tokenizer 训练用最终的 mix 比例。

---

## Slide 13 · 陷阱 10：分词后 leak

```
training_text  "the cat <think> wait </think> answer"
tokenizer 看到 <think> → 分给 special token
推理时如果 user 输入 "<think>" → 触发模型 internal state
```

防御：special token 严格保留，user input 中过滤。

---

## Slide 14 · 陷阱 11："annealing" 数据选错

末段切高质量 → 模型偏向高质量分布。

```
错例：annealing 用 100% wiki → 模型答非 wiki 风格降
对例：annealing 用 多源精选 ~10 不同源
```

---

## Slide 15 · 陷阱 12：合成数据"近亲繁殖"

```
GPT-4 合成 → Phi-3 训
Phi-3 合成 → Phi-4 训
Phi-4 合成 → 下一代训
```

→ 模式越来越窄，多样性塌缩。

防御：每代刷一遍人写数据 / 真实 WildChat。

---

## Slide 16 · 陷阱 13："对齐税"

SFT / RLHF 训得"安全" → 部分能力下降：
- 创造性写作
- 数学推理（refusal 多）
- 多样性

实务：SFT 数据要有"宽松" 例子，不能全是 safe template。

---

## Slide 17 · 陷阱 14：数据-模型规模错配

```
1B 模型 + 10T 数据 → 80% 数据未学
70B 模型 + 100B 数据 → 数据不足，模型欠拟合
```

Chinchilla：N × 20 ≈ D。但小模型可以 over-train（数据多没关系）。

---

## Slide 18 · 陷阱 15：忽略 metadata

```
{"text": "...", "url": "..."}      ← ts 缺失
↓
无法做 cross-dump dedup
无法做 contamination 追溯
无法做 license 审计
```

实务：保留 url / ts / source 字段，存储成本可忽略。

---

## Slide 19 · 防御 checklist

```
[ ] 13-gram benchmark match removal
[ ] Cross-dump dedup
[ ] URL blacklist + whitelist
[ ] Heuristic + classifier filter
[ ] Toxicity + PII
[ ] License audit
[ ] Tokenizer mix 与训练 mix 匹配
[ ] 合成 + 真实数据混
[ ] mid-training high quality 子集
[ ] 13-gram annealing 数据再 dedup
```

每条都对应一种"鬼故事"。

---

## Slide 20 · 课后思考

1. benchmark 污染是不是无解？
2. 合成数据走到哪一代会"塌缩"？
3. 如果完全无 dedup，会发生什么定量影响？
4. 你的训练 pipeline 还缺什么 checklist 项？

---

## 参考

- Llama-3 technical report 2024
- Phi-4 technical report 2024
- DeepSeek-V3 technical report 2024
- "Detecting Pretraining Data" Shi 2023
