# 21 · 手把手实战:从零搭一个迷你搜索引擎

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 21 个"知识点",不计入"140 个知识点"的统计——和 [19 类](19-mock-interview-capstone.md)、[20 类](20-advanced-interview-depth.md)是同一挂,但风格不一样:19/20 号文件里,你是**旁观者**,跟着面试官和候选人的对话把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个完整能用的小工具。

## 为什么是"搜索引擎"

不是要发明新知识点,是把三个你已经学过的知识点串成一个真实有用的东西:

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 输入几个词,瞬间找出哪些文档同时包含这几个词 | [01 类](01-complexity-and-python-builtins.md) dict 均摊 O(1)、[02 类](02-arrays-and-strings.md) 字符串处理 |
| 阶段 2 | 输入前缀,自动补全出完整的词(搜索框那种下拉提示) | [12 类](12-trie-and-string-matching.md) Trie 树 |
| 阶段 3 | 多个文档都命中时,先给出最相关的几个,而不是随便排序 | [07 类](07-heaps-and-priority-queues.md) 堆 / TopK |
| 阶段 4 | 把前三步拼成一个完整的类,跑一次端到端 demo | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行(本文件用仓库统一的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的代码时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:倒排索引——为什么不能每次都线性扫描全部文档

最笨的搜索方式:来一个查询词,就把所有文档从头到尾扫一遍,看哪些包含这个词。文档少的时候没问题,文档一多,这个笨办法的代价有多大?先测出来,再决定要不要优化——这是全系列"复杂度不是断言出来的,是真的测出来的"([00-roadmap.md](00-roadmap.md) 复杂度验证方法论)在这篇教程里的第一次应用。

```python
import re, time
from collections import defaultdict

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def make_corpus(n):
    docs = {}
    for i in range(n):
        docs[i] = f"document number {i} talks about topic {i % 50} and some filler words here"
    docs[n // 2] += " uniqueneedle appears only here"
    return docs

def linear_scan_search(docs, word):
    return {doc_id for doc_id, text in docs.items() if word in tokenize(text)}

def build_index(docs):
    index = defaultdict(set)
    for doc_id, text in docs.items():
        for w in set(tokenize(text)):
            index[w].add(doc_id)
    return index

def index_search(index, word):
    return index.get(word, set())

docs = make_corpus(20000)
index = build_index(docs)

# 两种做法必须先对齐结果，再比较速度——结果都不对，比速度没有意义
assert linear_scan_search(docs, "uniqueneedle") == index_search(index, "uniqueneedle") == {10000}

def best_of(fn, trials=5):
    best = None
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best

t_linear = best_of(lambda: linear_scan_search(docs, "uniqueneedle"))
t_index = best_of(lambda: index_search(index, "uniqueneedle"))

print(f"linear scan best-of-5: {t_linear * 1000:.4f} ms")
print(f"index lookup best-of-5: {t_index * 1000:.6f} ms")

# 容差放得很宽（只要求快 50 倍以上），因为这里要验证的是数量级差距，
# 不是精确系数——真实测出来的差距通常有几十万倍，50 倍只是一个绝对安全的下限
assert t_index < t_linear / 50
print("stage1 perf check ok")
```

两万篇文档里,线性扫描要挨个把每篇文档重新分词再判断,倒排索引只是一次哈希查找——这就是为什么几乎所有真实搜索系统(不管是 Elasticsearch 还是 SQLite 的 FTS5)都要先"建索引",而不是来一个查询就现扫一遍。

有了"要建索引"这个动机,接下来实现真正的功能:输入多个词,返回同时包含这些词的文档(搜索引擎里叫 AND 查询)。

```python
import re
from collections import defaultdict

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_index(docs):
    index = defaultdict(set)
    for doc_id, text in docs.items():
        for word in set(tokenize(text)):   # set() 去重：同一篇文档里这个词出现几次不重要，这一步只关心"有没有"
            index[word].add(doc_id)
    return index

def search_and(index, query):
    words = tokenize(query)
    if not words:
        return set()
    result = index.get(words[0], set()).copy()   # .copy() 很关键：不加的话下面 &= 会直接改到 index 内部的 set
    for w in words[1:]:
        result &= index.get(w, set())
    return result

docs = {
    1: "Python is a great language for data science",
    2: "Data science relies heavily on statistics and python",
    3: "Cats are great pets and very independent",
    4: "Dogs are great pets and very loyal",
}
index = build_index(docs)

assert search_and(index, "python") == {1, 2}
assert search_and(index, "great pets") == {3, 4}
assert search_and(index, "python data") == {1, 2}
assert search_and(index, "nonexistentword") == set()
assert search_and(index, "great") == {1, 3, 4}   # doc1 里也有 "great language"，容易漏看
print("stage1 correctness ok")
```

**注意 `search_and` 里的 `.copy()`**:`result &= index.get(w, set())` 如果 `result` 直接是 `index[words[0]]` 本身(没有 `.copy()`),`&=` 会原地修改这个 set——而这个 set 正是索引内部存的那个 set,一改就把索引改坏了,下次查询就会得到错误结果。这个坑不是理论上的,是可以现场复现的:

```python
from collections import defaultdict

index = defaultdict(set)
index["python"] = {1, 2, 3}
index["fast"] = {2, 3}

# 错误版本：不 copy，直接在 index["python"] 对应的 set 上做 &=
buggy_result = index["python"]
buggy_result &= index["fast"]

# index 内部的 "python" 这一项被污染了，后续任何单独查 "python" 的请求都会得到错误结果
assert index["python"] == {2, 3}          # 期望应该还是 {1, 2, 3}，但已经被改坏
assert buggy_result is index["python"]    # 两者是同一个对象，这就是问题根源

print("mutation bug reproduced:", dict(index))
```

## 阶段 2:前缀自动补全——给倒排索引配一个 Trie

阶段 1 解决的是"给完整的词,找文档"。搜索框还需要另一种能力:用户还没打完一个词,就要提示"你是不是想搜这个"——这需要"给一个前缀,找出所有以它开头的词",[12 类](12-trie-and-string-matching.md)的 Trie 树正是为这个场景设计的。

```python
class TrieNode:
    __slots__ = ('children', 'is_word')
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def _collect(self, node, prefix, out):
        if node.is_word:
            out.append(prefix)
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, out)

    def autocomplete(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []          # 前缀在树里都走不通，说明没有任何词以它开头
            node = node.children[ch]
        out = []
        self._collect(node, prefix, out)
        return sorted(out)

trie = Trie()
for w in ["data", "database", "date", "dark", "cat", "car", "care"]:
    trie.insert(w)

assert trie.autocomplete("da") == ["dark", "data", "database", "date"]
assert trie.autocomplete("ca") == ["car", "care", "cat"]
assert trie.autocomplete("zz") == []
assert trie.autocomplete("car") == ["car", "care"]   # "car" 自己也是一个完整词，要包含自己
print("stage2 ok")
```

`_collect` 是一次从当前节点出发的 DFS:先看当前节点是不是某个完整词的结尾(`is_word`),再往所有子节点递归——这一步和 [08 类](08-trees.md)树的前序遍历是同一个模式,只是这里的"树"每条边上都挂着一个字符。

## 阶段 3:相关度排序——多个文档都命中时,先给最相关的

阶段 1 的 `search_and` 只回答"哪些文档命中",不区分"命中得有多好"。真实场景里,一个词在文档里出现得越多,通常越能说明这篇文档和查询更相关——文档一多,不能全部返回,得挑出最相关的几个,[07 类](07-heaps-and-priority-queues.md)的堆/TopK 正是干这个的。

```python
import heapq, re
from collections import defaultdict, Counter

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_index_and_tokens(raw_docs):
    index = defaultdict(set)
    doc_tokens = {}
    for doc_id, text in raw_docs.items():
        tokens = tokenize(text)
        doc_tokens[doc_id] = tokens
        for word in set(tokens):
            index[word].add(doc_id)
    return index, doc_tokens

def search_ranked(index, doc_tokens, query, top_k):
    words = tokenize(query)
    if not words:
        return []
    candidates = index.get(words[0], set()).copy()
    for w in words[1:]:
        candidates &= index.get(w, set())
    scored = []
    for doc_id in candidates:
        counts = Counter(doc_tokens[doc_id])
        score = sum(counts[w] for w in words)   # 简化的相关度分数：查询词在文档里一共出现了几次
        scored.append((score, doc_id))
    return [doc_id for score, doc_id in heapq.nlargest(top_k, scored)]

raw_docs = {
    1: "python python python is great for data science",
    2: "python is okay but not my favorite for data science",
    3: "i love data science and python a little bit",
}
index, doc_tokens = build_index_and_tokens(raw_docs)

assert search_ranked(index, doc_tokens, "python", top_k=3) == [1, 3, 2]
assert search_ranked(index, doc_tokens, "python", top_k=1) == [1]
print("stage3 ok")
```

`heapq.nlargest(top_k, scored)` 内部维护一个大小为 `top_k` 的堆,不需要把全部候选文档排序完再截取前 K 个——这是 [07 类"TopK 不用完整排序"知识点](07-heaps-and-priority-queues.md) 的直接应用:文档数量是 N、每次比较 O(log K),比"排序全部 N 篇再切片"的 O(N log N) 更省,N 远大于 K 时差距明显。

**`scored` 是 `(score, doc_id)` 元组,这里有一个不能忽略的细节:元组比较是先比第一个元素,分数相同时会继续比第二个元素。** 也就是说分数打平的两篇文档,`doc_id` 更大的会排在前面——这和"更大的 ID 更相关"完全没有关系,纯粹是元组比较规则的副作用。上面例子里文档 2 和文档 3 分数都是 1,`search_ranked(..., "python", top_k=3)` 返回的 `[1, 3, 2]` 里 3 排在 2 前面,就是这个原因,不是 3 真的比 2 更相关。真实系统要么在分数打平时按时间戳这类有意义的次要指标继续排序,要么显式说明"打平的文档相对顺序不保证有意义",不能让读者误以为这个顺序有额外含义。

## 阶段 4:组装成一个完整的 `MiniSearchEngine`

把前三阶段拼进一个类,加一份真实的小型文档集合,跑一次完整的端到端 demo。

```python
import re, heapq
from collections import defaultdict, Counter

class _TrieNode:
    __slots__ = ('children', 'is_word')
    def __init__(self):
        self.children = {}
        self.is_word = False

class MiniSearchEngine:
    def __init__(self):
        self.index = defaultdict(set)     # 阶段1：word -> set(doc_id)
        self.doc_tokens = {}               # 阶段3 算相关度要用到每篇文档的完整分词结果
        self.trie_root = _TrieNode()       # 阶段2

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def add_document(self, doc_id, text):
        tokens = self._tokenize(text)
        self.doc_tokens[doc_id] = tokens
        for word in set(tokens):
            self.index[word].add(doc_id)
            self._trie_insert(word)

    def _trie_insert(self, word):
        node = self.trie_root
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
        node.is_word = True

    def _collect(self, node, prefix, out):
        if node.is_word:
            out.append(prefix)
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, out)

    def autocomplete(self, prefix):
        node = self.trie_root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        out = []
        self._collect(node, prefix, out)
        return sorted(out)

    def search(self, query, top_k=5):
        words = self._tokenize(query)
        if not words:
            return []
        candidates = self.index.get(words[0], set()).copy()
        for w in words[1:]:
            candidates &= self.index.get(w, set())
        scored = []
        for doc_id in candidates:
            counts = Counter(self.doc_tokens[doc_id])
            score = sum(counts[w] for w in words)
            scored.append((score, doc_id))
        return [doc_id for score, doc_id in heapq.nlargest(top_k, scored)]

engine = MiniSearchEngine()
corpus = {
    1: "Python python python is a popular language for data science and machine learning",
    2: "Deep learning models are trained using large amounts of labeled data",
    3: "Cats are independent pets that enjoy sleeping most of the day",
    4: "Dogs are loyal pets and need regular exercise every single day",
    5: "Python data science libraries include numpy pandas and scikit-learn",
}
for doc_id, text in corpus.items():
    engine.add_document(doc_id, text)

# doc1 提到 "python" 三次，doc5 只提到一次，分数不打平，排序结果没有歧义
assert engine.search("python") == [1, 5]
assert engine.search("learning") == [2, 1]
assert engine.search("data science") == [5, 1]
assert engine.search("nonexistent") == []

# 自动补全接在同一个引擎上
assert engine.autocomplete("py") == ["python"]
assert engine.autocomplete("da") == ["data", "day"]
assert engine.autocomplete("xyz") == []

print("stage4 end-to-end ok:", engine.search("python"), engine.autocomplete("da"))
```

到这里,`MiniSearchEngine` 已经是一个真实能用的小工具:喂给它任意一批文档,它能做多词 AND 查询、按相关度排序、支持前缀自动补全——三个能力分别来自三个你已经学过的知识点,拼起来就是一个真实系统的核心骨架(不是完整的 Elasticsearch,但内核思路是一样的)。

## 可以怎么继续扩展(只指方向,不在本文实现)

- **拼写纠错**:用户敲错一个字母也要能搜到——需要计算查询词和索引里的词之间的编辑距离,[10 类](10-dynamic-programming-basics.md)的编辑距离 DP 正好能用上。
- **更精细的相关度打分(TF-IDF)**:阶段 3 的打分只看"这个词在这篇文档里出现几次",没考虑"这个词本身是不是在所有文档里都很常见"(常见词权重应该更低)——这是信息检索里的标准做法,超出本文范围,只指出方向。
- **持久化存储**:现在的索引全部存在内存里,进程一退出就没了。怎么把倒排索引可靠地存到磁盘上、崩溃后还能恢复,属于 [database-deep-dive](../database-deep-dive/06-storage-engine-internals.md) 存储引擎内部机制的范畴。

这三个方向都不实现,是为了让这篇教程聚焦在"三个已学知识点怎么拼成一个真实工具"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这篇是试点,验证格式是否成立;其余系列要不要配套同类文件,是后续单独决定的问题,这里不展开。

---

*创建:2026-07-24*
