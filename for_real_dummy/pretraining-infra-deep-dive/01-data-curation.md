# 01 · 数据处理全流水深挖(Data Curation)

> 总览见 [00-roadmap.md](00-roadmap.md)

预训练模型的天花板是数据,不是架构——同样的 Transformer 骨架,喂 The Pile 的原始转储和喂 FineWeb-Edu 精选语料,下游 benchmark 能差出 10 个点以上。本文对应 `learning/data-curation/`(Module 3《造大模型》第 1 专题,12 lecture + 13 个 src 源文件),从"一段 CommonCrawl 抓下来的 HTML 到底要经过几道关卡才能进训练集"讲起,覆盖抽取→去重→质量过滤→毒性/PII→分词→配比→合成→陷阱→端到端 capstone 全链路。12 个知识点。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13,torch 2.11.0+cu128)下用 `.venv/Scripts/python.exe` 实际跑通验证。`datasketch`/`trafilatura`/`sentencepiece`/`tiktoken`/`warcio`/`sentence-transformers`/`transformers` 均已安装,只有 `detoxify`/`presidio_analyzer`/`ftlangdetect` 三个包缺失——这三个缺失恰好触发了源码自带的优雅降级路径(下文知识点 2/5/6 会具体展示缺失时的真实行为,不是靠猜测转述)。Windows 终端跑这些脚本时如果不设 `PYTHONIOENCODING=utf-8`,中文/箭头符号会在 GBK 控制台下变成乱码(不是代码 bug,是终端编码问题),下文命令统一带上这个环境变量。

---

## 1. 数据时代鸟瞰(概念点)—— 从 C4 到 DCLM,每一代解决上一代的什么问题

**是什么:**
LLM 预训练语料经历了五代范式转移,每一代都是对上一代"数据质量成为瓶颈"的回应:

| 代际 | 语料 | 核心改进 | 年份 |
|---|---|---|---|
| 1 | C4 | 第一个大规模启发式清洗的 CommonCrawl 子集 | 2019 |
| 2 | The Pile | 22 个精选来源混合(非纯网页),首次系统化"配比" | 2020 |
| 3 | RedPajama | 复刻 Llama-1 论文配方,开源社区首次可复现工业级配比 | 2023 |
| 4 | FineWeb / FineWeb-Edu | 用可训练分类器(而非人工规则)打分,教育性过滤 | 2024 |
| 5 | DCLM | 系统化 data ablation 框架,把"哪种过滤器更好"变成可实验问题 | 2024 |

**一句话:** 数据处理的历史就是"人工启发式规则 → 可学习分类器 → 系统化实验框架"的三级跳,和整个机器学习领域"从手工特征到端到端学习"的演化路径是同一个故事在数据侧的重演。

**底层机制/为什么这样设计:** 早期(C4)用固定规则(去 HTML 标签、去脏词表、保留有标点的段落)是因为没有更好的办法量化"质量";FineWeb-Edu 能用分类器,是因为 LLM(GPT-4 等)已经强到可以给"这段文本教育价值几分"打标,才有了训练数据可用;DCLM 能做系统化 ablation,是因为算力便宜到可以为了"验证一个过滤规则"专门训一个 1B 代理模型再废弃。**代际更替的驱动力不是"想到了更好的点子",而是"上一代工具成熟到可以用来生产这一代的训练信号"。**

**AI 研究场景:** 面试/研究讨论"为什么不直接用更大的模型解决数据质量问题"时,这条脉络是标准答案框架——数据质量提升和模型规模提升不是替代关系,是接力关系。

**可运行例子:** 本知识点是历史脉络综述,不对应独立可执行脚本;下文知识点 5(质量过滤)会给出 C4 启发式规则和 FineWeb-Edu classifier 的真实并排运行结果,是这条历史脉络第 1 代与第 4 代的直接对比。

**面试怎么问 + 追问链:**
- **Q:** "FineWeb-Edu 和 C4 的质量过滤方式有什么本质区别?"—— 期望:C4 是固定阈值规则(词数、标点、符号占比),FineWeb-Edu 是一个训练出来的回归分类器,能捕捉规则覆盖不到的"教育价值"这种模糊语义信号。
- **追问1:** "分类器打分会不会引入新的系统性偏差?"—— 期望:会,分类器的训练数据本身是 LLM 打标的,继承了打标模型的偏好(比如偏好教科书式行文,可能低估口语化但信息量大的文本);这是"用模型清洗模型训练数据"的循环依赖问题,DCLM 这类系统化 ablation 框架存在的意义之一就是量化不同过滤器的实际下游效果差异,而不是假设"分类器一定比规则好"。

**常见坑:** 把"新一代语料"理解成"完全替代旧一代"是常见误解——工业界配方(如 Llama-3)通常是多代技术的**组合**:C4 式启发式规则做第一道粗筛(便宜、无需模型推理),FineWeb-Edu 式分类器做第二道精筛(贵、但精度高),而不是只用最新一代技术。

---

## 2. CommonCrawl 抽取(`cc_extract.py`)—— WARC 流式解析 + trafilatura 去 boilerplate

**是什么:**
```python
def extract_from_html(html: str | bytes, url: str = "", ts: str = "") -> dict | None:
    """单文档抽取 + 语种识别 → dict 或 None."""
    import trafilatura
    text = trafilatura.extract(html, include_comments=False, include_tables=False,
                               deduplicate=True)
    if not text or len(text) < MIN_TEXT_LEN:
        return None
    try:
        from ftlangdetect import detect
        lang = detect(text=text[:512].replace("\n", " "), low_memory=True)["lang"]
    except Exception:
        lang = "unk"
    return {"url": url, "ts": ts, "lang": lang, "text": text}
```
(`cc_extract.py:27-39`)

```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值(下同)

def iter_warc(warc_path: str | Path) -> Iterator[dict]:
    from warcio.archiveiterator import ArchiveIterator
    with open(warc_path, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "response":
                continue
            # ... 节选,完整版见 cc_extract.py:42-56
```
(`cc_extract.py:42-56`,节选,展示 warcio 怎么用生成器流式读 WARC.gz 而不是一次性加载进内存)

**一句话:** WARC(Web ARChive)是 CommonCrawl 存放原始抓取结果的标准容器格式——把成千上万次 HTTP 请求/响应连同时间戳、URL 等元数据原封不动打包进一个大文件,一条 WARC record 对应一次完整的网页抓取;这些记录里 90% 以上字节是 HTML 标签/JS/CSS/导航栏,`trafilatura` 用启发式规则(文本密度、DOM 结构、标点密度)定位"主内容区",丢掉剩下的 boilerplate。

**底层机制/为什么这样设计:** `iter_warc` 用 `ArchiveIterator` 逐条 yield 而不是先读全部记录再处理,是因为单个 WARC 分片可以有几 GB、包含数万个 HTTP 响应——O(1) 内存的流式处理是处理 PB 级 CommonCrawl 转储的唯一可行方式,一次性加载会直接 OOM。`extract_from_html` 里语种识别包在 `try/except` 里、失败回退 `"unk"`,而不是让整个抽取失败——这是"单点故障不能拖垮整条流水线"的工程原则,10亿文档里哪怕 0.1% 触发语种识别失败,硬失败也会导致上百万文档的抽取工作报废。

**AI 研究场景:** 任何"从头做预训练"的项目第一步都是这个环节;工业界 Llama-3/DeepSeek-V3 的技术报告都专门用一节讲"抽取器选型"(trafilatura vs jusText vs 自研),因为抽取质量的天花板决定了后续所有清洗步骤能处理的原料上限。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from cc_extract import extract_from_html, DEMO_HTML

doc = extract_from_html(DEMO_HTML, url="http://demo.test/cats")
assert doc is not None
assert "Cats sleep" in doc["text"]
assert "nav" not in doc["text"].lower()  # 导航栏 "Home | About | Login" 应被剔除
assert "Copyright" not in doc["text"]     # 页脚同样应被剔除
assert len(doc["text"]) >= 200            # MIN_TEXT_LEN 门槛
print(f"lang={doc['lang']}  text_len={len(doc['text'])}")
```

**实测(`.venv` 真跑):** 内置 mock HTML(含 `<nav>Home | About | Login</nav>` 导航栏和 `<footer>Copyright 2024</footer>` 页脚)真实抽取后,输出 412 字符正文,`trafilatura` 确认把导航栏和页脚**完全剔除**、只保留 `<article>` 里的两段正文。`lang` 字段实测输出 `unk`——因为本机 `ftlangdetect` 未安装,触发了 `except Exception: lang = "unk"` 这条回退路径,不是抽取失败,是语种识别这一个子功能优雅降级,抽取本身(trafilatura 部分)完全正常。这正是知识点开头"环境声明"提到的三个缺失依赖之一在真实运行中的具体表现。

**面试怎么问 + 追问链:**
- **Q:** "为什么不直接用正则表达式去掉 HTML 标签就行,还要用专门的 boilerplate removal 工具?"—— 期望:正则去标签只能去掉"标签语法",去不掉"标签包裹但语义上是噪声"的内容(导航栏、广告位、页脚版权声明的文本本身没有 HTML 特殊字符);trafilatura 靠文本密度和 DOM 结构位置判断"这块区域是不是主内容",这是结构性判断,正则做不到。
- **追问1:** "如果 trafilatura 判断错了怎么办(比如把广告当正文保留)?"—— 期望:这就是为什么抽取之后还需要知识点 5 的质量过滤和知识点 3/4 的去重作为下游防线——没有任何单一环节能做到 100% 准确,pipeline 设计的核心思路是"多层防御,每层抓不同类型的噪声",而不是指望某一层做到完美。

**常见坑:** `MIN_TEXT_LEN = 200` 这个硬编码阈值容易被忽视——如果传入的 HTML 提取出来的正文长度在 200 字符以下(哪怕内容本身是干净的,比如一条短新闻快讯),函数会直接返回 `None` 而不是保留短文档,批量处理时容易被误判成"抽取失败率异常高",实际上是正常的长度过滤在生效。

---

## 3. MinHash + LSH 去重(`minhash_dedup.py`)—— 用局部敏感哈希把 O(n²) 两两比较降到近似 O(n)

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

def shingles(text: str, n: int = 5) -> Iterable[str]:
    text = text.lower().replace("\n", " ")
    for i in range(len(text) - n + 1):
        yield text[i:i + n]

def minhash_sig(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m
```
(`minhash_dedup.py:19-29`)

**一句话:** 把文档表示成 5-gram 字符集合,用 128 个哈希函数的最小值组合(MinHash 签名)以概率方式估计两个集合的 Jaccard 相似度,再用 LSH(Locality-Sensitive Hashing)把"可能相似"的文档分到同一个桶里,避免真的对全部文档做两两比较。

**底层机制/为什么这样设计:** 先补上"为什么取哈希值最小的元素就能估计 Jaccard 相似度"这一步——`minhash_sig` 里 `MinHash(num_perm=128)` 内部对每个 shingle 算 128 个不同哈希函数的值,只留每个哈希函数下的**最小值**,这个操作看起来随意,背后是一个不难验证的概率论事实:给定一个理想的随机哈希函数 `h`,把两个集合 `A`、`B` 的并集 `A∪B` 里所有元素都过一遍 `h`,想象按 `h` 值从小到大把它们排成一队——"谁排第一"完全随机,每个元素等概率排到队首;而"排第一的元素同时属于 `A` 和 `B`"(即属于 `A∩B`)这件事发生的概率,恰好等于 `|A∩B|/|A∪B|`——因为 `A∪B` 里一共 `|A∪B|` 个元素等概率排第一,其中只有 `|A∩B|` 个"两边都有"的元素满足条件,这个比例正是 Jaccard 相似度的定义。一个哈希函数只能抽一次"谁排第一",样本量太小,所以用 128 个相互独立的哈希函数各跑一遍这个实验,数一下"两个文档的最小哈希元素相同"发生了多少次、除以 128,就是 Jaccard 相似度的一个无偏估计——这就是 MinHash 签名(128 个最小哈希值组成的向量)把"精确算交并集"变成"比较两个 128 维向量有多少位相同"这件便宜得多的事的原理。

有了这个估计器,下一步问题是:即使把"精确算 Jaccard"降级成"比较 128 个数字",千亿文档两两比较依然是 O(n²)——LSH 再往前一步,把"查邻居"降到平均 O(1):把 128 个 MinHash 值切成 `b` 个 band、每个 band 里 `r` 个值(`r×b≈num_perm`,`datasketch` 会在给定 `num_perm=128` 和 `threshold=0.7` 时自动求解一组具体的 `(b, r)`——`.venv` 实测求解出 `b=14, r=9`,`14×9=126`,不必精确等于 128,`datasketch` 允许略有富余不用满);只要两个文档在**至少一个** band 里的全部 `r` 个值都相同,就标记为候选相似对,送去做更精确的判断:

```
MinHash 签名(128 个数字,示意)
[ v1  v2 ⋯ v9 | v10 v11 ⋯ v18 | ⋯⋯ | v118 ⋯ v126 ]  (b=14 个 band,每个 band r=9 个值)
  └── band 1 ──┘ └──  band 2  ──┘        └── band 14 ──┘
两文档只要在某一个 band 内的 9 个值全部相同 → 判定为候选相似对
```

这个"至少一个 band 全部命中"的判定标准,决定了候选概率随真实相似度 `s` 变化的公式怎么来:两个文档在某一个 band 里 `r` 个哈希值全部相同的概率是 `s^r`(每个值独立命中的概率是 `s`,`r` 个都命中就是 `s` 的 `r` 次方);因此这个 band **没有**全部命中的概率是 `1-s^r`;`b` 个 band 相互独立,全部都没命中的概率是 `(1-s^r)^b`;于是"至少有一个 band 命中"(即被标记为候选对)的概率就是 **candidate pair 概率公式** `P(候选) = 1 - (1 - s^r)^b`。这个 S 型曲线在阈值附近极陡——相似度略超阈值,候选概率会跳跃式上升,这就是"阈值 0.7"能同时做到"高召回、低误检"的数学原因,不是拍脑袋定的数字。

**AI 研究场景:** MinHash+LSH 是 FineWeb/RedPajama/DCLM 等主流开源语料公开配方里去重环节的标配算法(通常配合 SemDeDup 做二次语义去重,见知识点 4),千亿级文档的近重复检测在工业界几乎只用这一套方法族,因为它是目前唯一能在合理时间内跑完超大规模数据的近似最近邻方案。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from minhash_dedup import dedup, shingles

base = "the mitochondria is the powerhouse of the cell producing atp"
docs = []
for i in range(50):
    docs.append((f"orig_{i}", base + f" -- variant {i}"))
    docs.append((f"copy_{i}", base + f" -- variant {i} ."))
independent = [
    "machine learning is a subset of artificial intelligence systems today.",
    "quantum computing relies on superposition of qubits and entanglement.",
    "the great wall of china is over thirteen thousand miles long total.",
    "shakespeare wrote thirty seven plays during the elizabethan era era.",
    "renewable energy includes solar wind hydro and geothermal sources alike.",
]
for i, t in enumerate(independent):
    docs.append((f"ind_{i}", t * 4))

kept, dup_map = dedup(docs, threshold=0.7)
assert len(docs) == 105
assert len(kept) == 6            # 5 个独立文档 + 1 个 orig/copy 代表(见下方实测解释)
assert "ind_0" in kept and "ind_4" in kept

# 独立验证:为什么 orig_0..orig_49/copy_0..copy_49 这 100 篇几乎全部塌缩成 1 篇存活
s0 = set(shingles(base + " -- variant 0"))
s1 = set(shingles(base + " -- variant 1"))
jaccard_01 = len(s0 & s1) / len(s0 | s1)
assert jaccard_01 > 0.7   # 远超阈值,MinHash 判定它们是"重复"
print(f"kept={len(kept)}  orig_0 vs orig_1 精确jaccard={jaccard_01:.3f}")
```

**实测(`.venv` 真跑):** `kept=6`,而不是"50 对近重复各留 1 篇、独立文档全留"这种直觉上更合理的结果(直觉预期 kept≈55)。独立验证发现原因:`orig_0` 和 `orig_1` 只在"variant 0"vs"variant 1"这一个数字上不同,其余 60+ 字符完全相同,5-gram shingle 集合的精确 Jaccard 相似度高达 **0.971**,远超 0.7 阈值——不只是 `orig_i` 和它自己的 `copy_i` 相似,`orig_0..orig_49` 这 50 篇彼此之间也两两相似(因为共享同一个长前缀模板),LSH 查询时后插入的文档几乎全部命中"已插入"的某篇作为邻居而被标记重复,最终整个"模板化"簇只剩 1 篇代表(`orig_0`)存活,5 篇内容独立的文档(`ind_0..ind_4`)则全部保留。**这是一条比 docstring 注释更准确的发现**:阈值 0.7 对"共享长模板、仅局部数字/短语不同"的文本(公司模板邮件、法律条款样板、生成式 spam)是极度激进的去重,几乎会把整簇打包坍缩成 1 篇——这正是知识点 11 陷阱合集里"The Pile 内部 60% 重复(高频名言/法律条款)"这类问题的算法侧真实行为写照。

**面试怎么问 + 追问链:**
- **Q:** "MinHash 的 128 个 permutation 数字是怎么选的,越多越好吗?"—— 期望:num_perm 越大,Jaccard 估计方差越小(标准误差 ∝ 1/√num_perm),但签名存储和计算开销线性增长;128 是精度和成本之间的工业界常用折衷点,FineWeb/RedPajama 等配方也用类似量级。
- **追问1:** "如果两篇文档只是数据集划分不同(比如一个是训练集一个是测试集)但内容近乎相同,MinHash 会不会把它们也去重掉?"—— 期望:会,MinHash 不区分"数据来源标签",只看内容相似度——这正是"benchmark contamination"问题的另一面:如果不对 train/test 分开处理直接全局去重,可能会把测试集里的某道题当成训练集里已存在的重复直接抹掉,反而需要专门的跨集合污染检测(13-gram exact match,见知识点 11),而不能指望通用去重算法顺带解决。

**常见坑:** 把"threshold=0.7"理解成"只有相似度精确大于等于 0.7 的文档对才会被两两标记"是不准确的——本模块 `dedup()` 的实现是"贪心增量插入,查询到任意一个已保留邻居就整体归并",不是精确的全局最优聚类,对高度模板化的语料集群效应会比想象中激进得多(如上方实测所示),线上生产环境需要对这种"雪崩式塌缩"做抽样审计,不能只看"去重前后文档数变化"这一个宏观指标。

---

## 4. SimHash 与 SemDeDup 去重对照(`simhash_dedup.py` + `semdedup_demo.py`)—— 三种去重方法的粒度/召回/代价光谱

**是什么:**
```python
def simhash(text: str, dim: int = 64) -> int:
    """64-bit SimHash fingerprint."""
    counts = [0] * dim
    for t in _tokens(text):
        h = _hash_token(t, dim)
        for i in range(dim):
            counts[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i in range(dim):
        if counts[i] > 0:
            fp |= 1 << i
    return fp

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
```
(`simhash_dedup.py:30-45`)

```python
def dedup(texts: list[str], thresh: float = 0.95) -> tuple[list[int], list[tuple[int, int, float]]]:
    emb = embed(texts)              # SentenceTransformer("all-MiniLM-L6-v2")
    sim = cosine_matrix(emb)        # 归一化后点积 == cosine
    ...
```
(`semdedup_demo.py:31-47`,节选)

**一句话:** MinHash 抓"token 集合层面的重叠"、SimHash 用位向量投票抓同一件事的更轻量版本、SemDeDup 用句向量 cosine 相似度抓"意思相近但用词完全不同"的语义重复——三者是同一个"去重"目标下,粒度从"字面"到"语义"、代价从"秒级"到"需要模型推理"的一条光谱,不是互相替代关系。

**底层机制/为什么这样设计:** SimHash 的核心技巧是"每个 token 的哈希值逐位投票"——如果两篇文档的 token 集合高度重合,大部分位上的多数投票结果会相同,Hamming 距离就小;这让它可以完全不用 shingle 集合(比 MinHash 更省内存),代价是对"局部编辑"(插入/删除几个词导致 token 集合小范围变化)比 MinHash 更敏感。SemDeDup 走完全不同的路径:先把文档编码成稠密向量(捕捉语义而非字面 token),再用 cosine 相似度判重——这是唯一能识别"用词完全不同但语义等价"这类重复的方法,代价是必须过一次 embedding 模型的前向推理,无法做到 MinHash/SimHash 那种纯 CPU 哈希计算的速度。

**AI 研究场景:** 工业界配方(如 DCLM)的标准做法是"MinHash 全量粗筛 → SemDeDup 在幸存的高质量子集上再筛一遍"——先用便宜的方法处理万亿 token 级别的原始体量,把数据量降到可以承受 GPU 推理开销的规模,再用贵但精确的方法做第二遍精筛,这是"多级过滤器,越贵的方法处理越小的数据量"这一通用系统设计模式在数据清洗领域的具体应用。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from simhash_dedup import simhash, hamming, is_duplicate
from semdedup_demo import embed, cosine_matrix

# SimHash: 局部编辑(改一两个词)的 Hamming 距离,以及默认阈值实际上抓不住这个案例
a, b = "the cat sits on the mat", "the cat sat on the mat"
d = hamming(simhash(a), simhash(b))
assert d == 5
assert is_duplicate(simhash(a), simhash(b), thresh=3) is False   # 默认阈值3抓不住,需要更松的阈值
assert is_duplicate(simhash(a), simhash(b), thresh=5) is True     # 阈值放宽到5才能判定为重复

# SemDeDup: 用词完全不同的语义等价句子 vs 局部编辑句子,相似度差异巨大
texts = [
    "The cat is on the mat.",
    "A feline rests upon the rug.",           # 语义等价,用词几乎不重叠
    "Python is a programming language.",
    "Python is a coding language.",           # 只改一个词
]
emb = embed(texts)
sim = cosine_matrix(emb)
cat_feline_sim = float(sim[0][1])
python_sim = float(sim[2][3])
assert python_sim > cat_feline_sim   # 局部编辑的相似度分数远高于纯语义等价
print(f"cat/feline(语义等价,用词不同) sim={cat_feline_sim:.3f}")
print(f"python(局部编辑,用词几乎相同) sim={python_sim:.3f}")
```

**实测(`.venv` 真跑):** SimHash 对 `"the cat sits on the mat"` vs `"the cat sat on the mat"`(只改一个词)算出 Hamming 距离 5,阈值 3 下不算重复,但 4 组测试对里全部 Hamming 距离都在 5-29 区间、没有一对落在阈值 3 以内(demo 自己用阈值 3 但 4 个样例全部判定 `dup=False`)——这说明源码 `run_demo` 演示的这几个样例的措辞差异对 64-bit SimHash 来说都不够"轻微"。SemDeDup 侧独立验证出一个更值得记录的发现:`"A feline rests upon the rug."` 和 `"The cat is on the mat."`(源码注释标注为"paraphrase of 0",即认定是同义句)实际 cosine 相似度只有 **0.576**,远低于源码 demo 用的 0.85 阈值(更远低于 README 里生产环境建议的 0.95);而 `"Python is a coding language."` vs `"Python is a programming language."`(只换一个形容词)相似度高达 **0.957**。这个反差是本次验证独立发现、比 demo 自身的注释更准确的结论:通用句向量模型(all-MiniLM-L6-v2)对"表层措辞变化"的敏感度,系统性地高于对"深层语义等价但用词迥异"的敏感度——SemDeDup 并不是营销叙事里"能理解语义"的万能去重器,它本质上仍然主要在吃词汇/句法层面的相似性信号,真正的同义改写(尤其换了主语名词、动词、宾语三个内容词)很容易漏网。

**面试怎么问 + 追问链:**
- **Q:** "SemDeDup 用的 embedding 模型选型会不会影响去重效果?"—— 期望:会,而且影响很大——上面的实测已经证明同一个 all-MiniLM-L6-v2 模型对不同类型的"重复"(局部编辑 vs 深层语义等价)判断力差异悬殊,换一个专门针对 paraphrase 任务微调过的句向量模型(如 `all-mpnet-base-v2` 或专门的 paraphrase 模型),cat/feline 这类样例的分数很可能显著提高。
- **追问1:** "如果预训练语料里真的存在大量'深层语义等价但用词不同'的重复(比如同一新闻事件被不同媒体各自撰写报道),现有三种方法都抓不住怎么办?"—— 期望:这类"事实性重复但表达不同"的内容,理论上对语言建模的伤害本来就比"字面重复"小得多(模型学到的是不同的表达方式,不是纯粹的记忆),所以工业界通常不特别针对这类重复做专项处理,三种方法的组合已经覆盖了对训练动态危害最大的"字面/近字面重复"这个主要矛盾。

**常见坑:** 不要用 demo 脚本里的阈值(SimHash thresh=3、SemDeDup demo 里的 0.85)当作生产环境的可靠默认值——这两个数字都是"教学放宽"过的演示参数(SemDeDup 源码注释自己写明"教学放宽到 0.85",生产参考值是 0.95),实际项目里这类阈值都需要在有标注的近重复对数据集上做验证后再定,不能照抄教学代码里的数字。

---

## 5. 质量过滤(`quality_filter.py`)—— C4/Gopher 十条启发式 + FineWeb-Edu 学习型分类器

**是什么:**
```python
def heuristic_score(text: str) -> dict:
    """C4 + Gopher 启发式，返回 (passed: bool, reasons: list, stats: dict)."""
    ...
    if n_sent < 3: reasons.append("too_few_sentences")
    if last and not _END_PUNCT.search(last): reasons.append("no_end_punct")
    if "lorem ipsum" in text.lower(): reasons.append("lorem_ipsum")
    ...
    if n_words < 30: reasons.append("too_few_words")
    ...
```
(`quality_filter.py:29-95`,节选,完整版共 10 条规则)

```python
def fineweb_edu_score(text: str) -> float | None:
    """返回 0-5 教育性 score；模型未装/未下载时返回 None."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ...
```
(`quality_filter.py:102-120`,节选)

**一句话:** 启发式规则是一组"如果满足以下任意条件就丢弃"的硬性阈值(词数、标点、符号占比、重复 n-gram 比例),FineWeb-Edu classifier 是一个训练过的回归模型,对文本打 0-5 分的连续"教育价值"分数,两者经常在同一份文本上给出不同判断。

**底层机制/为什么这样设计:** 启发式规则的判据全部是"表层统计特征"(词数、标点、符号比例),计算成本几乎为零,可以在抽取阶段以极低成本筛掉明显的垃圾(导航栏残留、乱码、SEO 灌水页);FineWeb-Edu classifier 是在几万条人工/LLM 标注的"教育价值"样本上训练出的小模型(基于 BERT 类架构),能捕捉"这段话虽然语法通顺、词数达标,但内容空洞"这种规则覆盖不到的语义判断——两者是互补而非替代关系,规则做第一道粗筛(便宜、召回优先),分类器做第二道精筛(贵、精度优先)。

**AI 研究场景:** FineWeb-Edu 这个数据集本身(基于此分类器筛选自 FineWeb)在 2024 年发布后成为开源社区训练小模型的热门底料,因为实测证明"用分类器精选的高教育价值网页数据训出的模型",在同等 token 预算下 benchmark 表现显著优于未经此过滤的原始 FineWeb,这也是本知识点排在"数据处理"专题里权重较高的原因(源 README 标注为该模块高权重内容)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from quality_filter import heuristic_score, fineweb_edu_score

good_edu = ("The mitochondria is the powerhouse of the cell. It produces "
            "adenosine triphosphate through oxidative phosphorylation, which fuels "
            "all cellular metabolism. Mitochondria have their own DNA, inherited "
            "maternally, supporting the endosymbiotic theory of their origin.")
spam = "Click here! Click NOW! Buy this domain. Click here. Click NOW."

h_good = heuristic_score(good_edu)
h_spam = heuristic_score(spam)
assert h_good["passed"] is True
assert h_spam["passed"] is False
assert "too_few_words" in h_spam["reasons"] or "high_repeat_5gram" in h_spam["reasons"]

edu_good = fineweb_edu_score(good_edu)
edu_spam = fineweb_edu_score(spam)
if edu_good is not None:   # 模型可下载时才能比较分数
    assert edu_good > edu_spam
    print(f"good_edu fineweb-edu score={edu_good:.2f}  spam score={edu_spam:.2f}")
```

**实测(`.venv` 真跑):** `HuggingFaceFW/fineweb-edu-classifier`(约 125M 参数的 BERT 系分类器)本机真实下载并加载成功(无需 `--use-edu` 手动开关,`fineweb_edu_score` 首次调用时自动懒加载)。4 个样本的真实分数:线粒体科普段落 **2.76**,"Click here!"式 spam **-0.005**,lorem ipsum **-0.138**,可再生能源段落 **2.68**——分类器对内容质量的区分度符合直觉。**但独立验证发现一处启发式规则与分类器判断相冲突的真实案例**:可再生能源那段文字 FineWeb-Edu 打分 2.68(和线粒体段落的 2.76 几乎一样高,应视为高质量),却被启发式规则判定 `passed=False`,原因仅仅是 `too_few_words`(全文只有约 28 个词,低于 `n_words < 30` 这条硬阈值)。这是两种过滤机制在真实运行中意见分歧的第一手证据:**启发式规则完全不理解语义,只要文本"太短"就会被拒绝,即使内容本身教育价值很高**;工业界配方通常不会只依赖单一层过滤器,原因正在于此——任何一层单独看都会有假阳性/假阴性。

**面试怎么问 + 追问链:**
- **Q:** "如果启发式规则和 FineWeb-Edu 分类器的判断冲突了(比如上面这个真实案例),生产 pipeline 应该听谁的?"—— 期望:没有统一答案,取决于两层过滤器在 pipeline 里的角色分工——如果启发式规则的作用是"抽取阶段的低成本粗筛,只挡最明显的垃圾",可以把 `n_words < 30` 这类阈值调低或改成 warning 而非硬拒绝,把精细判断都交给分类器;这属于 pipeline 设计的取舍,不是算法本身的对错问题。
- **追问1:** "分类器只有 0-5 的教育性一个维度,如果我想同时筛'教育价值'和'事实准确性'两个不同的质量维度怎么办?"—— 期望:需要训练/使用多个独立分类器(或多任务模型),FineWeb-Edu 分类器只覆盖"教育价值"这一个训练目标定义的维度,不能指望它顺带解决"内容是否有事实错误"这类完全不同的判定问题。

**常见坑:** `fineweb_edu_score` 首次调用会从 HuggingFace Hub 下载模型(本次实测过程中出现 `Warning: unauthenticated requests...` 提示,未设 `HF_TOKEN` 时下载速度会变慢但仍可用),批量处理数百万文档前应该先手动触发一次下载并确认模型缓存到本地,否则第一次跑大批量任务时会在下载阶段耗费不可预期的时间;此外 `heuristic_score` 的 10 条规则里"词数"这类硬阈值对短小但优质的文本(如精炼的问答对、诗歌、代码片段的自然语言描述)天然不友好,把这套规则原封不动应用到非"长网页正文"类型的数据源上需要重新校准阈值。

---

## 6. 毒性过滤 + PII 脱敏(`toxicity_pii_filter.py`)—— Detoxify 多标签分类 + Presidio 实体识别,双降级路径

**是什么:**
```python
def is_toxic(text: str, threshold: float = 0.5) -> tuple[bool, dict | None]:
    s = detoxify_score(text)
    if s is None:
        return False, None
    flagged = (s.get("toxicity", 0) >= threshold or
               s.get("severe_toxicity", 0) >= threshold * 0.5)
    return flagged, s
```
(`toxicity_pii_filter.py:32-38`)

```python
import re

_REGEX_PII = {
    "EMAIL": re.compile(r"\b[\w\.\-_]+@[\w\.\-]+\.[a-zA-Z]{2,}\b"),
    "PHONE_US": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # 完整版还有 PHONE_CN / CREDIT_CARD 两条,节选到此
}
```
(`toxicity_pii_filter.py:45-52`,节选,Presidio 不可用时的 regex fallback 表)

**一句话:** 毒性检测用 Detoxify(多标签神经网络分类器,判断 toxicity/severe_toxicity 等 6 个维度)、PII 识别优先用 Presidio(基于 NER 的实体识别引擎),两者都在缺失时有各自的降级路径,但降级后的**能力边界完全不同**——PII 降级成 regex(能力打折但仍然工作),毒性检测降级成**完全不检测**(能力归零)。

**底层机制/为什么这样设计:** `anonymize()` 的 Presidio 分支失败时会 `except: pass` 后落到 `_regex_anon()`,这是一个真正的"次优但可用"的兜底方案——regex 抓不到 NER 能抓到的实体类型(比如人名、地址这类没有固定格式的 PII),但对邮箱/电话/SSN/IP 这些格式固定的 PII 类型仍然有效。但 `is_toxic()` 的降级路径不同:`detoxify_score` 失败时直接返回 `None`,`is_toxic` 里 `if s is None: return False, None`——**这不是"用更弱的方法继续检测",而是"完全不检测,一律判定为不毒"**。这个设计差异的根本原因是:regex 对结构化 PII 天然适用(格式本身就是检测信号),但毒性判断本质上需要理解语义,没有对应的"弱化版规则引擎"可以退回,所以源码作者选择了"宁可漏检,不做假阳性拦截"的保守策略。

**AI 研究场景:** 任何面向公开发布的预训练语料都必须过毒性和 PII 两道关,这不只是"质量"问题,是法律合规问题(欧盟 GDPR、加州 CCPA 等隐私法规对训练数据中的 PII 有明确限制)——Phi-3 技术报告明确提到删除了约 120 万份含密钥/凭证的代码文件(见知识点 11),真实工业界确实把这当作严肃的数据治理环节,不是可选的锦上添花步骤。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from toxicity_pii_filter import process, is_toxic, anonymize

pii_sample = "SSN 123-45-6789 leaked from the database server at IP 192.168.1.1."
out = process(pii_sample)
assert "<SSN>" in out["text_clean"]
assert "<IP>" in out["text_clean"]
assert ("SSN", "123-45-6789") in out["pii_found"]

toxic_sample = "I hate you idiot, you are the worst."
flagged, tox = is_toxic(toxic_sample)
# 本机 detoxify 未安装 → 优雅降级为"完全不检测"，flagged 恒为 False，不是脚本 bug
assert flagged is False
assert tox is None
print(f"toxic_sample flagged={flagged}  (detoxify 不可用时的真实行为)")
```

**实测(`.venv` 真跑):** PII 侧完全符合预期:`"SSN 123-45-6789 leaked from the database server at IP 192.168.1.1."` 被 regex fallback 正确识别并替换成 `"SSN <SSN> leaked from the database server at IP <IP>."`,邮箱+电话样本同样正确脱敏成 `<EMAIL>`/`<PHONE_US>`。**毒性侧独立验证出一个需要特别记录的真实行为**:`"I hate you idiot, you are the worst."` 这句明显带侮辱性的文本,`process()` 返回 `drop: False`——不是因为规则判定它"不够毒",而是因为本机 `detoxify` 包未安装,`is_toxic()` 直接短路成"不检测",4 个 demo 样本里唯一应该被 flag 的这一条,在这台机器上**完全不会被拦截**。这个发现直接呼应上面"底层机制"分析的能力边界差异——同一份"依赖缺失时优雅降级"的设计,PII 那一半仍然提供实质防护,毒性那一半实质上等于**关闭了这道防线**,如果不读源码只看"脚本能跑、没有报错",很容易误以为毒性过滤在正常工作。

**面试怎么问 + 追问链:**
- **Q:** "一个数据清洗 pipeline 的某个环节因为依赖缺失而静默降级、且降级后完全不生效,这是好的工程实践吗?"—— 期望:不是——本知识点的真实案例恰好示范了这个反模式的危害:代码不报错、不崩溃,pipeline 看起来"正常运行完毕",但实际上完全没有做毒性过滤,如果不主动读源码或写针对性的回归测试断言"高毒性样本必须被 flag",这个能力缺口可能长期不被发现。
- **追问1:** "怎么改进这个设计,让这种能力缺口更容易被发现?"—— 期望:至少应该在依赖缺失时打印一条明确的 WARNING 日志(而不是纯 `except: pass` 沉默处理),更严格的做法是给关键安全类过滤步骤配一条"fail loud"的健康检查——批处理开始前先跑一个已知应该被 flag 的毒性样本,如果没被 flag 就直接终止 pipeline 并报错,而不是让整批数据带着虚假的"已清洗"标签流入下一环节。

**常见坑:** 不要用"脚本跑起来没报错"作为"某个数据治理环节生效"的证据——本知识点是这条原则最直接的反例:`toxicity_pii_filter.py --demo` 完整跑完、4 个样本全部输出、没有任何异常或崩溃,但其中最应该被拦截的一条毒性样本实际上完全没有被检测到,必须显式断言具体的检测结果(如上方可运行例子里的 `assert flagged is False` + 附带说明),才能把"能力缺失"和"能力正常"两种情况从表面行为上区分开。

---

## 7. 手写 BPE vs tiktoken(`bpe_trainer.py` + `bpe_tiktoken.py`)—— 100 行复刻工业级分词算法核心逻辑

**是什么:**
```python
class BPE:
    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)     # 贪心:每轮合并最高频的相邻 pair
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges.append((pair, new_id))
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
```
(`bpe_trainer.py:43-57`,节选)

**一句话:** Byte-level BPE 训练是一个纯粹的贪心算法——从 256 个字节起步,每一轮统计所有相邻 token pair 的出现频次,把频次最高的 pair 合并成一个新 token,重复直到达到目标词表大小;tiktoken(OpenAI 工业级实现)在算法上和这个 100 行版本完全一致,差异只在于用 Rust 重写以获得速度、以及针对不同模型代际(gpt2/cl100k/o200k)训练出的具体合并规则不同。

**底层机制/为什么这样设计:** 从字节(而非字符)起步是 byte-level BPE 的关键设计——UTF-8 编码保证任何 Unicode 字符串都能表示成字节序列,256 个初始 token 覆盖全部单字节值,这意味着**词表永远不会遇到"未知字符"**(不需要 `<UNK>` 兜底,任何输入都能编码);贪心合并最高频 pair 的直觉是"把出现次数最多的模式压缩成一个符号能最大化压缩率",这和霍夫曼编码"高频符号用短码"的思路同源,但 BPE 合并的是**可变长度的连续 token 组合**而不是给固定符号分配变长比特串。

**AI 研究场景:** 训练自定义 tokenizer(而不是直接复用现成的 GPT-2/Llama 词表)是任何从零做预训练的团队必经步骤——词表决定了同样字节数的文本会被切成多少个 token,直接影响训练/推理成本和长上下文场景下的有效窗口大小(知识点 9 会展示不同语言下这个差异有多大)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from bpe_trainer import BPE

text = ("the quick brown fox jumps over the lazy dog. "
        "the quick brown fox is quick and brown. "
        "machine learning is the subset of artificial intelligence. "
        "machine learning models learn patterns from data. ") * 50
bpe = BPE()
bpe.train(text, vocab_size=300)
assert len(bpe.vocab) == 300
assert len(bpe.merges) == 44   # 300 - 256

sample = "the quick brown fox"
ids = bpe.encode(sample)
decoded = bpe.decode(ids)
assert decoded == sample                    # encode/decode 精确 roundtrip
assert len(ids) == 1                         # 见下方"实测"——这个数字本身就是一处独立验证过的真实发现
print(f"'{sample}' ({len(sample.encode('utf-8'))} bytes) -> {len(ids)} token(s): {ids}")

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
n_tok_industrial = len(enc.encode(sample))
print(f"同一句话,cl100k_base(工业级词表)编码成 {n_tok_industrial} 个 token")
```

**实测(`.venv` 真跑):** `bpe.train()` 在这份约 10KB(50× 重复)的高度重复语料上跑完 44 次合并,`"the quick brown fox"`(19 字节,该语料开头出现的原句)独立验证确认**精确压缩成 1 个 token**(id=280),encode→decode 精确 roundtrip 回原字符串。这不是偶然——该短语是语料里逐字重复出现 50 次的完整句子前缀,44 次贪心合并足以把它逐步链式压缩成单个 token,这是"训练语料高度重复 → 分词器对该模式压缩率极端偏高"的一个极端但真实的示范案例,和工业界训练语料"重复率过高会导致 tokenizer 在重复片段上过度特化"这一告诫(知识点 11 陷阱 2)是同一个现象的两个角度。作为对照,cl100k_base(GPT-3.5/4 使用的工业级词表,在数万亿 token 上训练)对同一句话给出 5 个 token——这个数字差异恰恰说明:**手写实现在小型重复语料上的"压缩率"不能和工业级 tokenizer 类比**,后者的词表是在覆盖面极广的语料上训练的通用词表,不会对任何单一来源的重复模式过拟合。

**面试怎么问 + 追问链:**
- **Q:** "手写实现和 tiktoken 算法上完全一致,为什么工业界不直接用 Python 手写版本训练大词表?"—— 期望:这里的 `_get_pair_counts`/`_merge` 是 O(n) 全量重新扫描的朴素实现,在千亿字节级语料上训练会慢到不可行;tiktoken/SentencePiece 等工业实现用了更高效的数据结构(优先队列维护 pair 频次、链表结构做 O(1) 局部更新)并且用 Rust/C++ 重写,同样的算法思路,常数因子和渐进复杂度都被大幅优化。
- **追问1:** "词表大小(vocab_size)怎么选,越大越好吗?"—— 期望:不是,词表越大平均压缩率越高(每 token 承载更多信息),但 embedding 层和输出层参数量随词表大小线性增长,且极长尾的稀有 token 训练样本不足会学不好;工业界词表大小(32k-256k)是压缩率、模型参数量、稀有 token 训练充分度三者的折衷,近年趋势是随模型/数据规模增长而增大(GPT-2 的 50k → GPT-4o 的 200k)。

**常见坑:** 上面"1 个 token"的极端压缩结果容易被误读成"手写 BPE 实现比工业级 tokenizer 更强"——实际上这只反映了"训练语料高度重复、测试样本恰好是语料本身的子串"这一构造性场景下的行为,换一句不在训练语料里出现过的句子,手写版本的压缩率会立刻回落到和词表大小匹配的正常水平,不能用这一个特例外推到"手写 BPE 训练质量优于工业实现"的结论。

---

## 8. SentencePiece Unigram(`spm_trainer.py`)—— 概率式子词分割 + "vocab_size 两头夹"真实踩坑

**是什么:**
```python
def train_spm(text: str, vocab_size: int = 1000, model_type: str = "unigram",
              out_dir: str | None = None) -> str:
    import sentencepiece as spm
    ...
    # SentencePiece 对玩具级小语料有「两头夹」约束：
    #   下界 vocab_size >= required_chars（byte_fallback 注入 256 字节 + 单字 + 元 token），
    #   上界 vocab_size <= 可切出的 piece 数，否则在 hard 模式报 "Vocabulary size too high"。
    distinct_chars = len(set(text))
    floor = 256 + distinct_chars + 16
    vocab_size = max(vocab_size, floor)

    spm.SentencePieceTrainer.train(
        ..., byte_fallback=True, hard_vocab_limit=False,   # 上界变软上限
    )
```
(`spm_trainer.py:18-46`,节选)

**一句话:** Unigram 语言模型分词法(SentencePiece 默认模式)和 BPE 的贪心合并思路完全不同——它从一个大候选子词表出发,用 EM 算法迭代地删掉"删掉后整体语料概率损失最小"的子词,直到收敛到目标词表大小,天然支持"同一句话有多种合法切分方式"的概率采样(subword regularization)。

**底层机制/为什么这样设计:** `byte_fallback=True` 保证任何字符(包括训练语料里从未出现过的字符)都有兜底编码路径,代价是词表必须先预留 256 个字节 token 的位置,这就产生了下界约束;`hard_vocab_limit=False` 把 `vocab_size` 从"必须精确达到"的硬约束变成"不超过"的软上限,这是应对小语料训练时"根本切不出那么多不同子词"这一实际情况的正确工程解——如果不加这个参数,玩具级语料几乎必然在两个方向的某一个触发报错。

**AI 研究场景:** SentencePiece 是 Llama-1/2、T5、多数非 OpenAI 系模型的默认分词器实现,Unigram 模式下的 subword regularization(训练时对同一输入随机采样不同切分方式)是一种数据增强手段,能让模型对分词边界的具体位置不过度敏感,这对形态丰富的语言(如日语、芬兰语)或代码这类切分方式本身就有歧义的领域尤其有用。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
import sentencepiece as spm
from spm_trainer import train_spm

text = "\n".join([
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subset of artificial intelligence",
] * 50)
model_path = train_spm(text, vocab_size=512, model_type="unigram")
sp = spm.SentencePieceProcessor(model_file=model_path)

sample = "the cat learns quickly"
ids = sp.encode(sample, out_type=int)
decoded = sp.decode(ids)
assert decoded == sample   # 即使 "cat"/"learns"/"quickly" 不在训练语料里，byte_fallback 保证精确还原

# subword regularization: 同一句话，多次采样应出现不止一种切分
variants = {tuple(sp.encode(sample, out_type=str, enable_sampling=True, alpha=0.3))
            for _ in range(20)}
assert len(variants) > 1
print(f"20 次采样得到 {len(variants)} 种不同切分方式")
```

**实测(`.venv` 真跑):** 训练语料只有 2 个句型 × 50 次重复(300 个"句子"、Alphabet size=27),`train_spm` 的自适应下界逻辑把请求的 `vocab_size=512` 保持不变通过(因为 512 本身已经高于 `floor = 256+27+16=299`,不需要抬升),真实训练收敛到 41 条有效句子输入、EM 两轮迭代后 seed 89 个候选片段收敛。测试句 `"the cat learns quickly"` 含有"cat"/"learns"/"quickly"这些从未在训练语料中出现过的词,`sp.encode`/`sp.decode` 精确 roundtrip 成功(`byte_fallback` 生效,切成 `['▁', 'the', '▁c', 'at', '▁le', 'ar', 'n', 's', ...]` 这种字符级兜底片段拼接);20 次 subword regularization 采样(`alpha=0.3`)独立验证得到**至少 4 种不同的切分方式**,包括把 `"cat"` 切成 `['c','at']`、`['at']` 前带独立空格标记等变体,证实了"同一句话可以有多种合法子词序列"这一 Unigram 模型的核心特性在小语料下依然真实成立,不是只在大规模训练下才会体现的效果。

**面试怎么问 + 追问链:**
- **Q:** "为什么小语料训练 SentencePiece 容易报 vocab_size 相关的错误,这对生产环境训练有什么启示?"—— 期望:`byte_fallback` 模式下词表有硬性下界(必须容纳 256 字节 token+去重字符数+若干元 token),小语料的字符集本身通常不大,如果 `vocab_size` 设得比这个下界还小就会报错;生产环境训练语料通常是 TB 级,天然远高于这个下界,所以这个坑主要出现在"用小样本快速验证 pipeline"这类场景,恰恰是本知识点这种教学/单测代码最容易踩中的地方。
- **追问1:** "`hard_vocab_limit=False` 会不会导致实际训出来的词表大小和预期的不一致,进而影响下游模型的 embedding 层维度?"—— 期望:会,这是需要注意的副作用——训练脚本如果假设"vocab_size 参数=最终词表大小"来预先分配 embedding 层参数,遇到软上限生效(实际词表小于请求值)的情况需要读取训练后模型的真实词表大小,而不能直接复用传入参数,生产 pipeline 里这个数字通常要在分词器训练完成后重新读取再传给建模代码。

**常见坑:** `vocab_size` 参数在软上限模式下是"不超过"而非"精确等于"——如果下游代码硬编码假设分词器词表就是传入的 `vocab_size`(比如据此固定模型 embedding 矩阵形状),在小语料/测试环境下会因为真实词表更小而出现维度不匹配,必须以训练产出物(`.model`/`.vocab` 文件)里的实际大小为准。

---

## 9. 多语言 Tokenizer 压缩率对照(`vocab_compare.py`)—— 同一段意思,不同语言的"每 token 承载信息量"天差地别

**是什么:**
```python
def main() -> None:
    import tiktoken
    models = ["gpt2", "cl100k_base", "o200k_base"]
    for name, text in SAMPLES.items():   # english/chinese/japanese/code/math
        n_chars = len(text)
        for m in models:
            enc = tiktoken.get_encoding(m)
            n_tok = len(enc.encode(text))
            ratio = n_chars / n_tok
```
(`vocab_compare.py:33-50`,节选)

**一句话:** 同一个语义内容,用不同语言表达、喂给同一个 tokenizer,压缩率(字符数/token 数)可以相差 5 倍以上——这个差异直接决定了非英语用户实际能用的"有效上下文长度"和推理成本,是一个经常被低估的公平性问题。

**底层机制/为什么这样设计:** BPE/Unigram 类 tokenizer 的词表是在训练语料上"学习"出来的,如果训练语料里英语占比远高于其他语言(GPT 系列训练语料曾长期以英语为主),词表里的高频合并结果自然会大量偏向英语常见词根/词缀组合,中日文这类语言的字符在训练语料中出现频率相对更分散,难以形成同样比例的长 token 合并,单个汉字/假名往往需要更多字节(UTF-8 下一个汉字 3 字节)且不容易与相邻字符合并成长 token。

**AI 研究场景:** 这是一个真实影响产品体验和成本的工程问题——同样按"token 数"计费的 API,处理同等信息量的中文输入比英文贵数倍;同样声称"128k 上下文"的模型,中文用户实际可用的"有效字符数"远小于英文用户;DeepSeek-V3 技术报告明确提到"中文专门 oversample 5倍"来缓解类似的语言不平衡问题(知识点 11 有更完整的讨论)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
import tiktoken

samples = {
    "english": "Machine learning is a subset of artificial intelligence.",
    "chinese": "机器学习是人工智能的一个分支。",
}
for model in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(model)
    ratios = {name: len(text) / len(enc.encode(text)) for name, text in samples.items()}
    assert ratios["chinese"] < ratios["english"]   # 中文压缩率全部模型下都更低
    print(f"{model}: english={ratios['english']:.2f} c/t  chinese={ratios['chinese']:.2f} c/t")
```

**实测(`.venv` 真跑):** 5 类文本(english/chinese/japanese/code/math)× 3 个 tokenizer 的完整实测表格:

| 语言 | gpt2 | cl100k_base | o200k_base |
|---|---|---|---|
| english | 6.52 c/t | 6.52 c/t | 6.52 c/t |
| chinese | **0.48 c/t** | 1.00 c/t | 1.54 c/t |
| japanese | 0.75 c/t | 0.85 c/t | 1.30 c/t |
| code | 2.29 c/t | 3.21 c/t | 3.21 c/t |
| math(LaTeX) | 1.70 c/t | 1.95 c/t | 1.97 c/t |

最悬殊的对比:gpt2 词表下英文压缩率(6.52)是中文的 **13.6 倍**——同样一段中文意思,gpt2 词表编码出的 token 数是英文的十几倍;新一代 o200k_base(GPT-4o 使用)显著改善了这个差距(压缩率提高到 1.54),但英文仍然领先 4.2 倍。三代 tokenizer(gpt2→cl100k→o200k)在中日文上的压缩率单调递增,说明这确实是训练语料多语言占比和词表设计主动改进的结果,不是随机波动。

**面试怎么问 + 追问链:**
- **Q:** "为什么不干脆给每种语言训练一个独立的 tokenizer,再按需切换?"—— 期望:多语言模型需要在同一个 embedding 空间里处理所有语言,共享词表才能让模型学到跨语言的语义关联(比如"cat"和"猫"如果切分方式/embedding 维度体系不统一,很难做好翻译类任务);独立 tokenizer 会牺牲这种跨语言泛化能力,工业界通用大模型基本都用统一的多语言共享词表,只是通过调整训练语料的语言配比、词表训练时的语言采样权重来缓解不平衡,而不是切换成多套词表。
- **追问1:** "如果一个模型对中文的压缩率只有英文的 1/4,这对训练成本(而不只是推理成本)有什么影响?"—— 期望:训练时如果不针对语言做配比调整,同样的"文档数"配额下中文实际消耗的 token 预算(=训练计算量)会远低于表面文档数占比暗示的规模,配比设计如果只按"文档数"或"字节数"配平,而不按"token 数"配平,容易在不知不觉中让中文获得的有效训练信号比预期少得多——这也是为什么"数据配比"(知识点 10)必须结合分词特性一起设计,不能脱节。

**常见坑:** 拿"压缩率(c/t)"直接当作"模型对该语言理解能力"的代理指标是错误类比——压缩率只反映 tokenizer 词表设计对某种语言的字节效率高低,和模型在该语言上的实际理解/生成质量是两个不同维度的问题(压缩率低只意味着"更贵、更占上下文窗口",不直接等价于"理解得差",尽管两者在训练语料占比不均衡时经常同时发生、容易被混为一谈)。

---

## 10. 数据配比 Ablation 与 Magpie 指令数据合成(`data_mix_ablation.py` + `magpie_synthesis.py`)—— 从"配比怎么选"到"指令数据从哪来"

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

class MixSampler:
    """按权重从 domain 抽样的迭代器."""
    def sample(self, n: int) -> Iterable[tuple[str, str]]:
        keys = list(self.weights.keys())
        probs = list(self.weights.values())
        for _ in range(n):
            d = self.rng.choices(keys, probs)[0]
            yield d, self.rng.choice(self.domains[d])
```
(`data_mix_ablation.py:19-33`,节选)

```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值
import random

def _mock_instruction(rng: random.Random) -> str:
    t = rng.choice(INSTRUCTION_TEMPLATES)
    # 完整版还会代入 a/b/book/x/y/z/sentence/task/role 等占位符,节选到此
    return t.format(topic=rng.choice(TOPICS), action=rng.choice(ACTIONS))
```
(`magpie_synthesis.py:44-55`,节选,mock 模式;真实 Magpie 见下方"底层机制")

**一句话:** 数据配比(不同来源域各占多大比例)和指令数据合成(SFT 数据从哪里来)是预训练/后训练两个阶段各自最容易被低估的"隐藏超参数"——同样的模型架构和总 token 量,换一个配比或合成策略,下游 benchmark 表现可以有数个百分点的差异。

**底层机制/为什么这样设计:** `MixSampler` 用带权重的多项分布抽样模拟"训练时每个 batch 里各个 domain 出现的比例",这是 Doremi 等配比优化算法的简化教学版——真实 Doremi 算法会训练一个小型代理模型,动态调整各 domain 权重使得"最难学的 domain"获得更多样本(minimax 思路,不是固定配比)。`magpie_real()`(需要真实 instruct 模型)展示了 Magpie 论文的核心 trick:把 chat template 截断到"user 角色刚开始、内容为空"的位置,让 instruct 模型在没有真实指令的情况下自由续写——因为模型被训练成"user 说完后我来回答",把它截在 user 位置反而会让模型自己"生成一个像是 user 会问的问题",这是一种利用模型自身对话格式训练痕迹的技巧,不需要任何种子指令或人工设计的 prompt。

**AI 研究场景:** Llama-3 技术报告的配比(50% web+30% code+20% other)、DeepSeek-V3 的"中文/英文/代码/数学"四分,都是本知识点 `MixSampler` 概念的工业级实例;Magpie 是 2024 年后开源 SFT 数据集(尤其是无需人工标注、大规模合成对话数据)的主流生成方法之一,因为它完全不需要种子问题库,理论上可以无限规模合成。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/data-curation/src")
from data_mix_ablation import MixSampler, mock_ppl
from magpie_synthesis import magpie_mock

domains = {"web": ["w"+str(i) for i in range(100)], "code": ["c"+str(i) for i in range(100)]}
difficulty = {"web": 2.5, "code": 3.0}
uniform = MixSampler({"web": 0.5, "code": 0.5}, domains, seed=42)
code_heavy = MixSampler({"web": 0.2, "code": 0.8}, domains, seed=42)
ppl_uniform = mock_ppl(list(uniform.sample(1000)), difficulty)
ppl_code_heavy = mock_ppl(list(code_heavy.sample(1000)), difficulty)
assert ppl_code_heavy > ppl_uniform   # code 权重更高、code 更难 -> 玩具ppl应更高
print(f"uniform mock_ppl={ppl_uniform:.2f}  code_heavy mock_ppl={ppl_code_heavy:.2f}")

items = list(magpie_mock(n=5, seed=42))
assert len(items) == 5
assert all(item["mode"] == "mock" for item in items)
assert all(len(item["instruction"]) > 0 for item in items)
print(f"[样例] {items[0]['instruction']}")
```

**实测(`.venv` 真跑):** 数据配比 4 组真实运行结果:`uniform`(web/code/math/wiki 各 25%)mock_ppl=15.70,`Llama-like`(web 67%+wiki 25%,code/math 各仅 3-5%)mock_ppl=**11.32**(最低,因为高比例配置在两个"简单" domain 上),`Math-heavy`(math 占 50%)mock_ppl=**19.90**(最高,math 是玩具难度表里最难的 domain)——数字关系符合"配比权重 × domain 难度"的预期设计,四组 `actual_freq` 抽样结果(1000 次采样)也和配置权重高度吻合(如 Llama-like 配置 web=0.67,实测抽样频率 0.69)。Magpie mock 模式 5 条样例全部通过模板真实生成,指令内容覆盖"解释某概念"/"写 Python 函数"/"翻译句子"等 8 类模板,确认这条路径不依赖任何模型推理,是纯 Python 字符串模板合成(README 已声明,本次验证确认符合)。

**面试怎么问 + 追问链:**
- **Q:** "`mock_ppl` 只是把 domain 难度做加权平均,这和真实训练中的 loss 有什么本质区别?"—— 期望:本质区别很大——真实训练里各 domain 之间会有**迁移效应**(在 code 上学到的逻辑结构可能提升 math 推理能力,反之亦然),`mock_ppl` 完全没有建模这种交互,只是"各 domain 独立计数加权求和",源码注释也诚实标注"玩具用,不真训";真实做配比 ablation 必须真的训练代理模型观察实际 loss 曲线,不能靠这种简化公式外推。
- **追问1:** "Magpie 合成数据和真人标注的 SFT 数据相比有什么风险?"—— 期望:Magpie 数据的多样性和质量完全依赖底层 instruct 模型本身的能力和"偏好"——如果底层模型本身有某些话题上的知识盲区或风格偏好,合成出的指令分布会继承这些偏差;而且连续用同一个模型的输出训练下一代模型存在"合成数据近亲繁殖"的多样性塌缩风险(知识点 11 陷阱 12 有专门讨论),这是纯合成 SFT 数据集相对真人标注的核心权衡。

**常见坑:** `magpie_synthesis.py --real` 走真实 GPU 推理路径(需要下载 instruct 模型),源码和 README 都明确标注这是**可选路径,不进日常 smoke 测试**,只有 `--demo`(mock 模板)进常规验证流程——批量复验本系列时不应该误跑 `--real` 路径,否则会引入不必要的下载和 GPU 依赖。

---

## 11. 数据清洗陷阱合集(概念点,源:`lectures/11-curation-pitfalls.md`)—— 15 类真实工程"鬼故事"

**是什么:** 本讲不对应独立可执行脚本,是对 Llama-3/Phi-4/DeepSeek-V3 三份技术报告里数据处理"血泪教训"的系统化整理,共 15 类陷阱,前 11 类见下表,后 4 类(tokenizer 与训练数据不匹配、分词后 special token leak、annealing 数据选错、合成数据近亲繁殖、对齐税、数据-模型规模错配)在讲义原文均有独立 slide,此处按危害类型归并展示:

| 类别 | 典型案例 | 防御 |
|---|---|---|
| 数据污染 | GSM8K 测试集 31 题混入 CommonCrawl,模型"背题"而非"推理" | 13-gram exact match 删除(Llama-3 配方) |
| 重复 | The Pile 内部重复率 60%(高频名言/法律条款) | 跨 dump MinHash 去重,阈值 0.7 起步 |
| 编码问题 | mojibake(UTF-8 被二次按 CP1252 解码)、全角半角混杂 | NFKC 归一化 + 剔除 U+FFFD 占比过高的文档 |
| 语言不平衡 | 训练语料 80% 英文 → 非英语能力系统性偏弱 | 分语言独立去重 + oversample(DeepSeek-V3 中文 5×) |
| 秘密泄漏 | GitHub 代码语料含 10 万+ API key/私钥 | detect-secrets/trufflehog + regex 兜底(Phi-3 删除约 120 万文件) |
| 版权风险 | Books3(2023 下架)、Reddit(2024 限爬) | 持续追踪 license 变化,而非"抓取时合法就永久合法" |
| 合成数据塌缩 | GPT-4→Phi-3→Phi-4→下一代,逐代模式收窄 | 每代混入人写数据/真实对话日志 |

**一句话:** 这 15 类陷阱里,超过一半不是"算法不够先进"导致的,而是"流程/工程纪律"缺失导致的(没留 metadata、annealing 阶段选错子集、tokenizer 训练用的语料和最终训练配比不一致)——纯技术能力和数据治理纪律是两件不同的事,后者更容易被低估。

**底层机制/为什么这样设计:** 以陷阱"tokenizer 训练与训练数据不匹配"为例:如果 tokenizer 只在 web 文本上训练,词表天然偏向 web 常见词汇;后续 mid-training 阶段切换到 textbook 类高质量子集时,textbook 里的学术词汇/专业术语在 tokenizer 词表里可能没有对应的长 token,只能被切成很多短 token(压缩率骤降,知识点 9 已展示不同语料压缩率能差 5 倍以上),同样字节数的 textbook 内容会消耗远超预期的 token 预算,直接推高这个阶段的训练成本——这是知识点 9(多语言压缩率)揭示的现象在"训练语料 vs 分词器训练语料不一致"这个更一般场景下的重现,不是孤立的新问题。

**AI 研究场景:** 这份清单本质上是"任何独立复现预训练 pipeline 的团队都会踩的坑目录",面试/研究讨论"从零构建预训练数据 pipeline 需要注意什么"时,能对照具体技术报告(而不是泛泛而谈)举出 2-3 个具体陷阱,是候选人真的做过相关工作(或深入研究过公开报告)的强信号。

**可运行例子:** 本知识点为工程实践清单,不提供独立代码验证;15 类陷阱中和本文其他知识点直接呼应、已用真实运行结果验证过的有:陷阱"重复"对应知识点 3(MinHash 塌缩发现)、陷阱"语言不平衡"对应知识点 9(实测压缩率表)、陷阱"数据污染"对应知识点 3 追问链讨论的 train/test 混淆风险。

**面试怎么问 + 追问链:**
- **Q:** "如果你的预训练模型在某个 benchmark 上分数高得可疑,你会怎么排查是不是背了题?"—— 期望:第一步做 n-gram(常用 13-gram)精确匹配,检查训练语料里是否包含 benchmark 测试集的原文片段;这是行业标准做法(Llama-3 配方明确写入,"Detecting Pretraining Data" Shi 2023 这篇论文专门研究此问题),不是猜测性的怀疑,而是有具体可执行的检测方法。
- **追问1:** "13-gram 匹配会不会有漏网之鱼(比如题目被轻微改写后混入训练集)?"—— 期望:会,精确 n-gram 匹配只能抓字面重复,轻微改写(换同义词、调整句式)可以绕过;这是精确匹配方法的已知局限,更彻底的方案需要结合语义相似度检测(类似知识点 4 的 SemDeDup 思路),但计算成本高得多,工业界目前仍以 n-gram 精确匹配为主力,把语义检测留给更小规模的抽检审计。

**常见坑:** 把这份清单当作"读完就等于会做数据治理"是最大的坑——每一条防御措施在真实项目里都需要具体的阈值选择、工具链搭建、人工抽检验证,本讲的价值是"知道有哪些类型的问题需要主动防御",而不是提供可以直接套用的现成方案。

---

## 12. Capstone:1B Token 自制语料端到端流水线(`capstone_mini_corpus.py`)—— 五阶段真实串联,一次运行看清整条数据处理链路

**是什么:**
```python
def run_pipeline(out_dir: str, warc_path: str | None = None, use_edu: bool = False):
    report.append(stage_1_extract(warc_path, paths["extract"]))
    report.append(stage_2_dedup(paths["extract"], paths["dedup"]))
    report.append(stage_3_quality(paths["dedup"], paths["quality"], use_edu=use_edu))
    report.append(stage_4_pii(paths["quality"], paths["pii"]))
    report.append(stage_5_tokenize(paths["pii"], paths["tokenize"], model_dir=out_dir / "tokenizer"))
```
(`capstone_mini_corpus.py:221-238`,节选)

**一句话:** 这是本文知识点 2/3/5/6/8 五个独立组件的真实串联(不是重新实现,直接 `import` 前面几个知识点用到的同一份源码),用 1000 篇 mock 文档一次跑通"抓取→去重→质量→PII→分词"完整链路,数字全部真算而非硬编码,是验证"各组件独立测试通过"和"组件真的能拼成一条流水线"这两件不同事情的最后一步。

**底层机制/为什么这样设计:** `_mock_docs()` 构造的 1000 篇文档刻意分成三类(1/3 高质量多样化主题文档、1/3 高度重复的垃圾模板、1/3 明显 spam),且高质量文档特意设计成"不同主题+轮换句子顺序+不同结尾句"——源码注释明确写出这个设计动机:如果 mock 数据全部共享同一个长前缀且不以句号结尾,会被知识点 3 揭示的 MinHash"模板塌缩"效应和知识点 5 的启发式规则同时判定失败,导致整条流水线产出空语料却仍然 `exit 0`(静默假成功,是本模块历史上真实修复过的一个 bug)。这个精心设计的 mock 数据分布,本身就是"知识点 3 的发现"反过来指导"知识点 12 的测试数据设计"的具体例证。

**AI 研究场景:** 端到端 pipeline smoke test 是任何真实数据处理系统上线前的标准检查——单元测试能保证每个 stage 函数本身逻辑正确,但只有串联起来的集成测试才能发现"stage A 的输出格式 stage B 读不了"这类接口不匹配问题,或者更隐蔽的"每个 stage 单独看都对,但级联下来最终产出为空"这类系统性问题。

**可运行例子:**
```python
import sys, tempfile
sys.path.insert(0, "learning/data-curation/src")
from capstone_mini_corpus import run_pipeline

with tempfile.TemporaryDirectory() as tmp:
    report = run_pipeline(tmp, warc_path=None, use_edu=False)
    stages = {r["stage"]: r for r in report}

    assert stages["extract"]["n"] > 0     # 1000 mock docs 里,长度>=200的高质量docs幸存
    assert stages["dedup"]["n"] > 0       # 去重后不应清空(历史bug:曾经会清空)
    assert stages["dedup"]["n"] <= stages["extract"]["n"]
    assert stages["quality"]["n"] > 0
    assert stages["pii"]["n"] > 0
    assert stages["tokenize"]["n_tokens"] > 0
    print({s: r["n"] for s, r in stages.items()})
```

**实测(`.venv` 真跑):** 完整五阶段真实运行结果:`extract: 1000→334`(只有"高质量主题文档"这 1/3 通过了 `MIN_TEXT_LEN>=200` 长度过滤,另外两类 mock 文档本身就短,在抽取阶段已被筛掉,不是在后续 dedup/quality 阶段才被处理)→ `dedup: 334→15`(6 个主题模板轮换出的 334 篇文档,MinHash 判定后只剩 15 篇代表存活,和知识点 3 揭示的"模板化文本塌缩"效应同源)→ `quality: 15→15`(幸存文档本来就是刻意设计成通过启发式规则的,无进一步损失)→ `pii: 15→15`(mock 内容不含 PII/毒性)→ `tokenize: 15 docs,5397 tokens`(真实训练 SentencePiece 并编码全部文档,非硬编码估算)。全流程 3.1 秒完成,最终语料非空,验证了"多样化+句号结尾"的 mock 数据设计确实规避了历史上"空语料仍 exit 0"的 bug。

**面试怎么问 + 追问链:**
- **Q:** "这条 capstone pipeline 的各阶段 n_docs 变化(1000→334→15→15→15)说明了什么工程教训?"—— 期望:绝大部分"淘汰"发生在最早的 extract 阶段(长度过滤)和 dedup 阶段(模板塌缩),quality/pii 两个阶段几乎没有进一步淘汰——这提示在真实大规模 pipeline 里,**越早的阶段过滤强度往往越大**,越到后面的阶段处理的是已经被前面几轮筛过的"幸存者",各阶段的计算资源分配应该考虑这个漏斗形状(早期阶段面对的数据量最大,用最便宜的方法;晚期阶段数据量已经大幅收窄,可以用更贵但精细的方法)。
- **追问1:** "如果把 dedup 和 quality 两个阶段的顺序对调,结果会不同吗?"—— 期望:对本例的具体输出数字会不同(因为不同阶段的过滤条件互相独立、判定集合不同),但工程上通常把去重放在质量过滤之前,因为去重能显著减少后续阶段要处理的数据量,而质量过滤的判断(尤其调用 FineWeb-Edu classifier 这类需要模型推理的判断)单位成本更高,让"更便宜的判据"先跑、缩小规模,再让"更贵的判据"处理规模更小的幸存集合,是控制端到端处理成本的常见顺序设计。

**常见坑:** 这个 capstone 的 `--smoke` 模式全部用内置 mock 数据(`_mock_docs`),不接触任何真实 WARC 文件或网络请求——如果需要验证"真实 CommonCrawl 数据下这条流水线的表现",必须额外准备 `.warc.gz` 文件走 `--warc` 参数,mock 模式的产出数字(334/15/15/15/5397)只反映这批特意设计的合成测试数据的行为,不能直接当作"真实网页语料下大约会有这个淘汰比例"的经验估计。

---

*下一篇:[02-scaling-infra.md](02-scaling-infra.md) —— 从"数据处理好了"到"怎么把千卡集群喂饱",训练规模化的并行策略与显存账本。*
