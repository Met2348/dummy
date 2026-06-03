# L02 · CommonCrawl + 抽取 — 从 WARC 到 jsonl

> 24 slides | 70 min | Data Curation 第 2 讲 ⭐⭐⭐⭐

> 一切 LLM 语料的起点：CommonCrawl

---

## 学习目标

1. 了解 CC 的发布节奏、文件格式、规模
2. 掌握 WARC / WAT / WET 三种格式与适用场景
3. 用 trafilatura 抽取主内容（不是 HTML）
4. 用 fasttext 做轻量级语种识别
5. 写出能扛 1GB segment 的流式处理器

---

## Slide 1 · CommonCrawl 简介

- 非营利组织（2007 至今），月度爬虫全 web
- 每月 1 dump（≈ 200-300TB 压缩 WARC）
- 数据公开免费，AWS S3 上 `commoncrawl` bucket
- LLM 训练数据 70% 以上由 CC 衍生

```
crawl-data/CC-MAIN-2024-51/
  segments/
    1733...-001.warc.gz     ~1 GB each
    1733...-002.warc.gz
    ...
```

---

## Slide 2 · 三种格式

| 后缀 | 内容 | 大小 | 何时用 |
|------|------|------|--------|
| `.warc.gz` | **原始 HTTP 响应** (HTML/JS/CSS) | 大 | 重抽取 |
| `.wat.gz` | metadata + 链接 | 中 | 网络分析 |
| `.wet.gz` | **已抽 plain text** | 小 | 快速训练 |

教学一般用 WET 起步（已抽文本），生产用 WARC 重新抽取（trafilatura 比 CC 自带的强）。

---

## Slide 3 · WARC 记录结构

```
WARC/1.0
WARC-Type: response
WARC-Target-URI: http://example.com/article
WARC-Date: 2024-12-15T10:30:00Z
Content-Length: 53892

HTTP/1.1 200 OK
Content-Type: text/html

<html>...
```

一个 WARC 文件由多条 record 拼接而成，每条独立可流式读。

---

## Slide 4 · warcio 流式 API

```python
from warcio.archiveiterator import ArchiveIterator

with open("seg.warc.gz", "rb") as stream:
    for record in ArchiveIterator(stream):
        if record.rec_type == "response":
            url = record.rec_headers.get_header("WARC-Target-URI")
            html = record.content_stream().read()
```

**特点**：内存 O(1)，不一次性加载。1GB segment 处理 ~30s（取决于抽取器）。

---

## Slide 5 · 为什么不直接用 WET？

WET 是 CC 自带的 `text/plain` 版本。问题：

- 抽取算法粗暴（基于 HTML tag 启发式）
- 含大量 navbar / footer / 广告文本残留
- 段落边界粗糙

trafilatura / readability-lxml 比 CC 自带强 20-30%（主内容比例提升）。

---

## Slide 6 · trafilatura 入门

```python
import trafilatura

html = "<html>...</html>"
text = trafilatura.extract(
    html,
    output_format="txt",
    include_comments=False,
    include_tables=False,
    deduplicate=True,
)
```

返回值是已去 boilerplate 的 plain text，或 None（如果是非内容页）。

---

## Slide 7 · trafilatura 算法概要

1. **lxml** 解析 DOM
2. **启发式打分** 每个块（密度 / 链接占比 / 标点占比）
3. **正文检测** 取最大密度块
4. 兜底：调用 readability-lxml 作为 fallback

学术界版本：CETD（content extraction via text density）。

---

## Slide 8 · 抽取效果对照

| 方法 | 主内容召回 | 噪声率 |
|------|----------|--------|
| CC WET | 60-70% | 高 |
| readability | 75-80% | 中 |
| **trafilatura** | **85-92%** | **低** |
| boilerpipe | 80% | 中 |

实务：FineWeb / DCLM 均用 trafilatura。

---

## Slide 9 · 语种识别 — fasttext

```python
from ftlangdetect import detect
detect(text="Hello world")
# → {"lang": "en", "score": 0.99}
```

fasttext lid.176.bin (~120MB) 支持 176 种语言。

**注意**：短文本 < 50 字符识别率会暴跌，做完 trafilatura 抽取后再做语言识别。

---

## Slide 10 · 处理 pipeline (mental model)

```
warc.gz ─► warcio ─► html ─► trafilatura ─► text
                                              │
                              fasttext ◄──────┤
                                              │
                              normalize ◄─────┤
                                              ▼
                                          jsonl line
```

每行 jsonl 含：`{"url": "...", "text": "...", "lang": "en", "ts": "..."}`

---

## Slide 11 · 流式 vs 批处理

| 模式 | 内存 | 速度 | 适用 |
|------|------|------|------|
| 流式 (yield) | O(1) | 中 | 教学 / 单机 |
| pyspark / beam | O(N) | 高 | 多机 |
| ray data | O(分片) | 高 | 中等 |

本课程用 Python 流式（教学清晰）；生产 FineWeb 用 datatrove + slurm。

---

## Slide 12 · "我只想 1GB demo"

公开的 CC 1GB segment 示例：

```
https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/
segments/1733...001.warc.gz
```

`curl -O` 下载，本课 capstone 用此规模。

---

## Slide 13 · 抽取后规模感

| 阶段 | 体量 (1GB WARC 起) |
|------|---------------|
| WARC 原始 | 1.0 GB |
| HTML 解 | ~600 MB |
| trafilatura 抽 plain text | ~50-80 MB |
| 语言过滤 (只 en) | ~30-50 MB |
| 后续 dedup / 质量 | -50% |

≈ 50k - 100k 文档 / GB WARC。1B token ≈ 200-500 GB WARC。

---

## Slide 14 · 常见 robots.txt 误读

CC 本身已经遵守 robots，所以 CC 数据集中的内容默认是"允许爬取"。

⚠️ 但商业再分发要看每个网站的 ToS。教学 / 研究通常 fair use。

---

## Slide 15 · 抽取陷阱 1 — 编码

```python
# CC 的 WARC 可能 mixed encoding
charset = chardet.detect(html_bytes)["encoding"]
text = html_bytes.decode(charset, errors="replace")
```

实务：trafilatura 自动检测；手写时记得 `errors="replace"`。

---

## Slide 16 · 抽取陷阱 2 — JS 渲染

CC 抓的是初始 HTML，不执行 JS。后果：

- SPA 网站（React/Vue 早期）抽到的几乎为空
- 动态内容缺失
- 广告/cookie banner 大量残留

**对策**：trafilatura 会 fallback 到 readability，但仍有 5-10% 损失。

---

## Slide 17 · 抽取陷阱 3 — boilerplate 残留

navbar / footer / "click here" / 隐私声明等。

trafilatura 已经处理 80%，剩 20% 留给后续：
- C4 启发式（段落长度过滤）
- FineWeb-Edu classifier 拦截

---

## Slide 18 · 抽取陷阱 4 — 长尾低质

CC 中有大量：
- domain-squatting 页（"buy this domain"）
- search engine results page
- error pages (404 / 500)

→ trafilatura 输出 < 100 字符的丢掉即可，能去掉 80% 噪声。

---

## Slide 19 · 流式 jsonl 输出

```python
import json, gzip

with gzip.open("out.jsonl.gz", "wt", encoding="utf-8") as f:
    for doc in extract(warc_path):
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
```

`.jsonl.gz` 是事实标准（每行 1 doc，gzip 压缩节省 70%）。

---

## Slide 20 · 元数据保留建议

最少字段：
```
{
  "url": "...",
  "ts": "2024-12-15T10:30:00Z",
  "lang": "en",
  "text": "...",
  "n_tokens": 1234   # 后续填
}
```

⚠️ 保留 url + ts 利于：dedup 跨 dump、benchmark contamination 追溯、license 审计。

---

## Slide 21 · 工具链总览

| 任务 | 工具 |
|------|------|
| WARC 读 | warcio |
| HTML 抽 | trafilatura |
| 语种 | fasttext-langdetect / langid |
| 编码 | chardet |
| 输出 | json / jsonlines / gzip |
| 大规模 | datatrove (HuggingFace) |

---

## Slide 22 · 1 文档 demo（伪代码）

```python
from warcio.archiveiterator import ArchiveIterator
import trafilatura
from ftlangdetect import detect

with open("seg.warc.gz", "rb") as f:
    for rec in ArchiveIterator(f):
        if rec.rec_type != "response": continue
        url = rec.rec_headers.get_header("WARC-Target-URI")
        html = rec.content_stream().read()
        text = trafilatura.extract(html)
        if not text or len(text) < 200: continue
        lang = detect(text[:512])["lang"]
        yield {"url": url, "text": text, "lang": lang}
```

完整版见 `src/cc_extract.py`。

---

## Slide 23 · 评估抽取质量

人工抽样 50 doc 看：
- 主内容召回率（缺段落比例）
- boilerplate 残留率（导航/广告文本）
- 编码错误率（乱码）

实务 trafilatura 通常：主内容 90%+ / boilerplate < 10% / 编码 0-2%。

---

## Slide 24 · 与 Notebook 对应

`notebooks/02-commoncrawl.ipynb`:
1. 加载内置 1-doc mock WARC
2. 跑 trafilatura
3. 语种识别
4. 输出 jsonl 验证

---

## 参考

- CommonCrawl: https://commoncrawl.org
- WARC 格式: ISO 28500
- trafilatura: Barbaresi 2021
- FineWeb extraction recipe: HuggingFace blog 2024
