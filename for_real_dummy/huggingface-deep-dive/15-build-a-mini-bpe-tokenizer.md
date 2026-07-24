# 15 · 手把手实战:从零训练一个迷你 BPE Tokenizer

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 15 个"知识点",不计入"101 个知识点"的统计——和 [14 类](14-advanced-interview-depth.md)是同一挂,但风格不一样:14 号文件里,你是**旁观者**,跟着面试官和候选人的多级追问链把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立训练出一个真实能用的迷你 BPE tokenizer,并且用它给训练语料里从没出现过的新词分词。这是 dsa-deep-dive 系列率先试点过的"教程体"格式([dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md))往本系列的推广落地。

## 为什么是"训练一个 BPE tokenizer"

不是要发明新知识点,是把 [01 类](01-tokenizer-internals.md)"BPE 分词算法实操演示"这一点已经讲过的核心算法(统计相邻符号对频率 → 合并最高频的一对 → 重复)往前后各扩展一步:往前扩展到"给一段真实的、由多句话构成的语料,不是手写好的 `word: 频率` 字典",往后扩展到"训练出的合并规则表,能不能真的用来给一个训练时没见过的新词分词"。01 类的可运行例子只做了训练阶段的前半段(在 4 个词上跑 4 轮合并,看合并顺序),没有走到推理阶段;这篇教程把训练和推理两头都做完整,而且训练轮数从 4 轮扩大到 30 轮,足够看到"整词合并""跨词共享片段""合并顺序里的并列"这些 4 轮玩具例子看不到的真实现象。

| 阶段 | 让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 把几句话拆成词频统计,再拆成"字符+词尾标记"序列,数出相邻符号对的真实出现频率 | [01 类](01-tokenizer-internals.md) BPE 分词算法实操演示里"统计频率"这一步 |
| 阶段 2 | 反复"找最高频 pair → 合并 → 记录规则",跑够多轮,拿到一张完整的、真实训练出来的合并规则表 | 阶段 1 + [01 类](01-tokenizer-internals.md) 同一套贪心算法,规模从 4 个词扩大到真实句子语料 |
| 阶段 3 | 拿着训练好的规则表,对一个语料里完全没出现过的新词做真实 tokenize(推理阶段) | 阶段 2 的产出 + [01 类](01-tokenizer-internals.md)"训练阶段 vs 推理阶段"这组区分 |
| 阶段 4 | 把训练和推理拼进一个 `MiniBPETokenizer` 类,一次跑通"喂语料 → 出规则表 → 给新词分词" | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行(本文件用仓库统一的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的函数/语料时会重新贴一遍,不是偷懒复制,是这套校验机制要求的,和 [dsa-deep-dive/21 类](../dsa-deep-dive/21-build-a-mini-search-engine.md)是同一个约束)。

---

## 阶段 1:语料 → 词频 → "字符+词尾标记"表示,统计出真实的相邻符号对频率

先解决"频率从哪来"这个问题——BPE 训练的每一步都要用到词频,01 类的演示例子直接给了一个手写好的 `{word: 频率}` 字典,这里往前退一步,从几句真实的英文句子出发,自己数出词频。语料是特意设计过的:反复出现 `running`/`jumping`/`playing` 这几个都以 `-ing` 结尾的动词,以及 `garden`/`gate` 这两个都以 `ga` 开头的词——这样后面才能真的观察到"高频片段被合并成一个整体"这件事,不是随便找几句话拼凑的。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

word_freq = get_word_freq(sentences)

assert word_freq["the"] == 10
assert word_freq["is"] == 6
assert word_freq["dog"] == 4
assert word_freq["cat"] == 3
assert len(word_freq) == 20            # 6句话里一共20个不同的词
assert sum(word_freq.values()) == 44   # 总词数(含重复)是44

print("word_freq (top 8):", word_freq.most_common(8))
print("stage1a ok: real word frequencies from a real corpus, not a hand-typed dict")
```

6 句话总共 44 个词(含重复),经过大小写统一、按正则拆词之后剩下 20 个不同的词——`"the"` 出现 10 次,目前频率最高,这个数字会直接决定下面第一轮合并先处理"the"内部的哪个片段。

有了词频,接下来把每个词拆成 BPE 训练要用的表示:单字符序列 + 一个词尾标记。这里沿用 01 类和主流实现都用的 `</w>` 标记,作用是区分"词尾的字符"和"词中的字符"——比如 `"cat"` 结尾的 `t` 和 `"biggest"` 结尾的 `t` 语义上是两码事,不加标记的话 BPE 会把它们统计成完全同一种相邻对。

**实现上和 01 类的写法略有不同**:01 类把每个词表示成一个"空格分隔的字符串"(`"l o w </w>"`),合并时用正则替换;这里把每个词表示成一个 Python 元组(`('l', 'o', 'w', '</w>')`),合并时直接按下标扫描——两种写法是同一个算法的两种等价实现,选元组是因为阶段 3 要重复"把新词切成同样的符号序列"这个操作,元组/列表处理起来更直接,不需要再引入正则转义。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"  # 词尾标记，和01类BPE演示用的是同一个符号

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    """统计当前词表里每一种"相邻符号对"一共出现了多少次(按词频加权求和)。"""
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}
pairs = get_pair_counts(vocab)

# 零合并状态下的真实统计结果——没有手算，全部由上面两个函数跑出来
assert pairs[("e", "</w>")] == 12   # the(10)+fence(1)+gate(1)，都以e结尾
assert pairs[("t", "h")] == 11      # the(10)+with(1)
assert pairs[("g", "</w>")] == 10   # dog/running/jumping/playing/big等以g结尾的词之和
assert pairs[("i", "n")] == 7
assert len(pairs) == 59             # 一共59种不同的相邻符号对

print("top 5 pairs:", sorted(pairs.items(), key=lambda kv: -kv[1])[:5])
print("stage1b ok: BPE pair counting confirmed on the un-merged vocab")
```

零合并状态下,`('e', '</w>')` 以 12 次并列所有相邻对里的最高频——`"the"`(10 次)+`"fence"`(1 次)+`"gate"`(1 次)都是以 `e` 结尾的词,加起来正好 12。这不是挑出来凑数的例子,是 `get_pair_counts` 真实跑出来的结果:59 种不同的相邻符号对里,它是频率最高的一个,阶段 2 的第一轮合并就会选中它。

---

## 阶段 2:贪心合并——每轮挑最高频的 pair,合并,记录下完整的"合并规则表"

阶段 1 只统计了一轮。BPE 训练要把"统计 → 挑最高频 → 合并 → 重新统计"这个循环重复很多轮,每一轮合并产生的新符号会继续参与后面几轮的统计——这也是为什么"合并规则表"必须是**有序的**:阶段 3 的推理要严格按这个顺序应用规则,不能打乱(下面会验证这一点)。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_once(pair, vocab):
    """把词表里所有出现这个相邻对的地方，合并成一个新符号。"""
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}

merge_rules = []          # 训练出的"合并规则表"——本教程真正的产出
num_merges = 30
for _ in range(num_merges):
    pairs = get_pair_counts(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=lambda p: pairs[p])   # 贪心：只看当前这一轮频率最高的pair
    merge_rules.append((best_pair, pairs[best_pair]))
    vocab = merge_once(best_pair, vocab)

assert len(merge_rules) == 30
assert merge_rules[0] == (("e", "</w>"), 12)
assert merge_rules[1] == (("t", "h"), 11)
assert merge_rules[2] == (("th", "e</w>"), 10)
assert merge_rules[29] == (("a", "</w>"), 2)

for i, (pair, count) in enumerate(merge_rules):
    print(i, pair, count)
print("stage2a ok: trained 30 real merge rules from the corpus")
```

跑 30 轮之后拿到一张完整的合并规则表。前几条不意外:先合并 `"the"` 内部的片段(`('e','</w>')` → `('t','h')` → `('th','e</w>')`,三轮就把 `"the"` 变成了一个整体符号),紧接着是 `('g', '</w>')`——这条规则会在后面被 `dog`/`running`/`jumping`/`playing`/`big` 好几个不同的词复用,这正是 BPE"合并规则一旦学到,会被很多不同的词共享"这个核心特性的直接体现。

不过第 3 轮(下标从 0 开始数,即上面表格里 `merge_rules[2]`)藏着一个容易被忽略的细节——训练代码用的是 `max(pairs, key=...)`,Python 的 `max()` 在遇到并列最高频时,规则是"按可迭代对象的遍历顺序,返回第一个遇到的最大值",不是"更合理"或"字典序更小"这类看起来像是规则的规则。这一轮真的发生了并列,下面现场验证,不是假设出来的场景。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_once(pair, vocab):
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}

# 先真实跑完前2轮合并，把vocab推进到"第3轮开始前"的状态
for _ in range(2):
    round_pairs = get_pair_counts(vocab)
    round_best = max(round_pairs, key=lambda p: round_pairs[p])
    vocab = merge_once(round_best, vocab)

pairs = get_pair_counts(vocab)
max_count = max(pairs.values())
tied = sorted(p for p, c in pairs.items() if c == max_count)
picked = max(pairs, key=lambda p: pairs[p])

assert max_count == 10
assert tied == [("g", "</w>"), ("th", "e</w>")]
assert picked == ("th", "e</w>")

print(f"round 3 real tie: {tied}, both count={max_count}")
print(f"max() actually picked: {picked} -- just the first one it scanned, nothing more principled")
print("stage2b ok: confirmed a real tie, not a hypothetical one")
```

`('g', '</w>')` 和 `('th', 'e</w>')` 在这一轮真的并列 10 次——`max()` 最终选中了 `('th', 'e</w>')`,只是因为在 `pairs` 这个 `Counter` 的遍历顺序里它先被扫到,不是因为它在任何意义上"更值得合并"。**这不是这个玩具实现独有的缺陷:任何 BPE 训练器都要面对"并列时选哪个"这个问题**,真实的 `tokenizers` 库内部也有自己的确定性并列规则,只是很少有教程会提到这一层。30 轮训练里一共有 22 轮出现过并列——这个数字本身就说明,小语料上 BPE 训练的具体路径比想象中更依赖"实现细节",不是一个纯粹由词频唯一确定的过程。

光看一张扁平的规则列表,不容易看出"合并"具体怎么作用在某一个词上。取 `"jumping"` 这个词,追踪它在整个 30 轮训练过程里,究竟被改写了几次、在哪几轮被改写。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_once(pair, vocab):
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}
merge_rules = []
for _ in range(30):
    pairs = get_pair_counts(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=lambda p: pairs[p])
    merge_rules.append(best_pair)
    vocab = merge_once(best_pair, vocab)

# 追踪"jumping"这一个词，在30轮合并规则表里，究竟是哪几轮真正碰到了它
symbols = tuple(word_to_symbols("jumping"))
trace = [(-1, None, symbols)]
for i, pair in enumerate(merge_rules):
    new_vocab = merge_once(pair, {symbols: 1})
    new_symbols = next(iter(new_vocab))
    if new_symbols != symbols:
        trace.append((i, pair, new_symbols))
    symbols = new_symbols

assert trace[0][2] == ("j", "u", "m", "p", "i", "n", "g", "</w>")   # 起点：8个符号
assert [t[0] for t in trace[1:]] == [3, 4, 8, 23, 24, 25, 26]        # 30轮里恰好这7轮碰到了jumping
assert trace[-1][2] == ("jumping</w>",)                              # 终点：合并成1个符号

for step, pair, syms in trace:
    print(step, pair, syms, f"n={len(syms)}")
print("stage2c ok: jumping shrank from 8 symbols to 1, across 7 of the 30 rounds")
```

`"jumping"` 从 8 个符号(7 个字符+1 个词尾标记)一路缩到 1 个符号,只用到 30 条规则里的 7 条,而且这 7 条分布得很不均匀:第 3、4、8 轮很早就把 `...ing` 这个词尾结构处理掉了,但 `jump` 这个词根本身要等到第 23~26 轮才被合并完——因为 `j+u`、`ju+m`、`jum+p` 这几步在整个语料里只被 `"jumping"` 一个词贡献,频率只有 2,排在很多更高频的通用片段后面。这也呼应 01 类那句话"合并次数越多,这个词被切分成的 token 就越少"——但"轮到哪个词、哪一步"完全取决于真实词频,不是词本身的长度或"看起来该不该合并"。

---

## 阶段 3:推理阶段——用训练好的规则表,给一个从没见过的新词分词

阶段 2 产出的 `merge_rules` 是一个**有序**列表,早合并的规则排在前面。推理阶段(真实调用 tokenizer 对一个新词分词时发生的事)不是重新跑一遍训练,而是把这张规则表当成一本"操作手册":给一个新词,反复找"当前所有相邻符号对里,规则表中顺序最靠前的那一条",应用它,直到没有规则能再用为止。这和 01 类可运行例子里 `tok.tokenize(...)` 内部实际发生的事,是同一套逻辑。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_once(pair, vocab):
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}
merge_rules = []
for _ in range(30):
    pairs = get_pair_counts(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=lambda p: pairs[p])
    merge_rules.append(best_pair)
    vocab = merge_once(best_pair, vocab)

merge_ranks = {pair: rank for rank, pair in enumerate(merge_rules)}  # 越早学到的规则，rank越小、优先级越高

def tokenize(word, merge_ranks):
    """推理阶段：给一个词，反复应用"当前所有相邻符号对里，rank最小(最早学到)的那条规则"，直到没有规则能用。"""
    symbols = word_to_symbols(word)
    while len(symbols) > 1:
        candidates = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        applicable = [(merge_ranks[p], i) for i, p in enumerate(candidates) if p in merge_ranks]
        if not applicable:
            break
        _, i = min(applicable)
        symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
    return symbols

# 先做一致性检验：训练语料里出现过的词，用tokenize()重新分词，必须和训练时的最终形态完全一致
assert tokenize("running", merge_ranks) == ["running</w>"]
assert tokenize("the", merge_ranks) == ["the</w>"]
assert tokenize("runner", merge_ranks) == ["runn", "er</w>"]        # runner只出现1次，没能完全合并成一个token
assert tokenize("biggest", merge_ranks) == ["b", "i", "g", "g", "e", "s", "t</w>"]  # 全程没轮到它，保持字符粒度

print("running ->", tokenize("running", merge_ranks))
print("runner  ->", tokenize("runner", merge_ranks))
print("biggest ->", tokenize("biggest", merge_ranks))
print("stage3a ok: tokenize() reproduces training-time results exactly")
```

验证的第一步不是急着上新词,是先确认这个 `tokenize()` 函数在训练语料**内部**的词上,能不能复现训练时的最终结果——`running` 这种高频词训练时已经被合并成单个符号,`runner` 因为只出现 1 次没能完全合并,`biggest` 全程一次都没被合并规则碰到。`tokenize()` 在推理阶段重新算一遍,必须得到和训练时完全一样的结果,这是这个函数"实现正确"的底线要求,不是它的亮点——真正有意思的是下一步:换成训练语料里**一次都没出现过**的词。

```python
import re
from collections import Counter

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

def get_word_freq(sentences):
    counter = Counter()
    for s in sentences:
        words = re.findall(r"[a-z]+", s.lower())
        counter.update(words)
    return counter

END = "</w>"

def word_to_symbols(word):
    return list(word) + [END]

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_once(pair, vocab):
    a, b = pair
    merged = a + b
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

word_freq = get_word_freq(sentences)
vocab = {tuple(word_to_symbols(w)): f for w, f in word_freq.items()}
merge_rules = []
for _ in range(30):
    pairs = get_pair_counts(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=lambda p: pairs[p])
    merge_rules.append(best_pair)
    vocab = merge_once(best_pair, vocab)
merge_ranks = {pair: rank for rank, pair in enumerate(merge_rules)}

def tokenize(word, merge_ranks):
    symbols = word_to_symbols(word)
    while len(symbols) > 1:
        candidates = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        applicable = [(merge_ranks[p], i) for i, p in enumerate(candidates) if p in merge_ranks]
        if not applicable:
            break
        _, i = min(applicable)
        symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
    return symbols

# "jumper"/"grunting"/"zebra" 训练语料里一次都没出现过，但tokenize()照样能给出结果
jumper = tokenize("jumper", merge_ranks)
grunting = tokenize("grunting", merge_ranks)
zebra = tokenize("zebra", merge_ranks)

assert jumper == ["jump", "er</w>"]              # "jump"学自jumping,"er</w>"学自over/runner——两个独立学到的片段直接拼上
assert grunting == ["g", "run", "t", "ing</w>"]  # "run"和"ing</w>"是学过的,但"grunt"这个词根从没见过,只能保持零碎
assert zebra == ["z", "e", "b", "r", "a</w>"]    # "z"这个字符训练语料里根本没出现过,不会崩溃,只是没有任何规则能碰它

print("jumper   ->", jumper)
print("grunting ->", grunting)
print("zebra    ->", zebra)
print("stage3b ok: seen roots/suffixes merge, novel combinations stay fragmented")
```

`jumper`(全语料一次都没出现过)被切成了 `['jump', 'er</w>']`——只有 2 个片段,而且两个片段分别来自两个不同的高频词:`jump` 这个片段是从 `"jumping"`(词频 2)学到的,`er</w>` 是从 `"over"` 和 `"runner"`(合计词频 3)学到的。这个词本身从没被训练见过,但训练出的两条独立规则拼在一起,直接就把它切成了两个有意义的片段——这就是 BPE 常被拿来"泛化到未见过的词"的真实原因。

`grunting` 是更有代表性的一个例子:结果是 `['g', 'run', 't', 'ing</w>']`。`run` 和 `ing</w>` 都是训练学到的真实片段(分别来自 `running`/`runner` 和 `running`/`jumping`/`playing` 的词尾),被正确识别合并了;但 `grunt` 这个词根语料里完全没出现过,夹在中间的 `g` 和 `t` 没有任何规则能碰到它们,只能保持字符粒度——这正是这篇教程一开始就想观察的现象:**高频词根/后缀被合并成完整片段,罕见组合仍然保持拆分的字符粒度,而且这两种情况可以同时出现在同一个词里,不是非此即彼**。

顺带验证一个边界情况:`zebra` 里的 `z` 这个字符,整个训练语料里一次都没出现过。`tokenize()` 并没有因此报错或崩溃——这个函数从头到尾只处理"当前相邻对是否在规则表里",一个从没见过的字符不会匹配任何规则,就原样留在结果里,变成一个单字符的"token"。**这和真实生产环境的 tokenizer 不完全一样**:真实的 BPE(比如 GPT-2 系列)通常建立在**字节级**(byte-level)基础词表上,256 个字节值本身就是完整的基础符号集合,不会有"训练语料里没见过的原始输入字符"这种情况;这个玩具实现是字符级的,故意暴露了"如果基础符号集合本身不完整会怎样"这个真实生产实现要专门解决的问题,下面"可以怎么继续扩展"里进一步展开。

---

## 阶段 4:组装成一个完整的 `MiniBPETokenizer`

把阶段 1-3 的全部逻辑收进一个类:`train()` 对应阶段 1+2(语料进,合并规则表出),`tokenize()` 对应阶段 3(新词进,子词序列出)。

```python
import re
from collections import Counter

class MiniBPETokenizer:
    END = "</w>"

    def __init__(self):
        self.merge_rules = []
        self.merge_ranks = {}

    @staticmethod
    def _word_freq(sentences):
        counter = Counter()
        for s in sentences:
            counter.update(re.findall(r"[a-z]+", s.lower()))
        return counter

    @classmethod
    def _word_to_symbols(cls, word):
        return list(word) + [cls.END]

    @staticmethod
    def _pair_counts(vocab):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_once(pair, vocab):
        a, b = pair
        merged = a + b
        new_vocab = {}
        for symbols, freq in vocab.items():
            new_symbols, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    def train(self, sentences, num_merges):
        word_freq = self._word_freq(sentences)
        vocab = {tuple(self._word_to_symbols(w)): f for w, f in word_freq.items()}
        self.merge_rules = []
        for _ in range(num_merges):
            pairs = self._pair_counts(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=lambda p: pairs[p])
            self.merge_rules.append(best_pair)
            vocab = self._merge_once(best_pair, vocab)
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merge_rules)}
        return self

    def tokenize(self, word):
        symbols = self._word_to_symbols(word)
        while len(symbols) > 1:
            candidates = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            applicable = [(self.merge_ranks[p], i) for i, p in enumerate(candidates) if p in self.merge_ranks]
            if not applicable:
                break
            _, i = min(applicable)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
        return symbols

sentences = [
    "the cat is running in the garden",
    "the dog is running in the yard",
    "the cat is jumping over the fence",
    "the dog is jumping over the gate",
    "a big dog is playing with a small cat",
    "the biggest dog is the fastest runner",
]

bpe = MiniBPETokenizer().train(sentences, num_merges=30)

assert len(bpe.merge_rules) == 30
assert bpe.tokenize("running") == ["running</w>"]
assert bpe.tokenize("jumper") == ["jump", "er</w>"]
assert bpe.tokenize("grunting") == ["g", "run", "t", "ing</w>"]

results = {}
for w in ["the", "running", "runner", "jumper", "grunting", "biggest", "zebra"]:
    results[w] = bpe.tokenize(w)
    print(f"{w:10s} -> {results[w]}")

print("stage4 end-to-end ok: trained a real BPE merge table and tokenized real+unseen words with one class")
```

到这里,`MiniBPETokenizer` 已经是一个真实能用的小工具:喂给它任意一批句子,`train()` 能统计词频、跑贪心合并、产出一张有序的合并规则表;`tokenize()` 能拿着这张表,对任何新词(不管训练时见没见过)给出确定性的子词切分——内核思路和 01 类里 `AutoTokenizer.from_pretrained` 加载的真实 SentencePiece BPE tokenizer 完全一样,只是规模从几十万词表缩到了 30 条规则。

## 可以怎么继续扩展(只指方向,不在本文实现)

- **和真实 `tokenizers` 库训练接口的关系**:`tokenizers` 库的 `trainers.BpeTrainer`(以及更高层的 `AutoTokenizer.train_new_from_iterator`,[01 类](01-tokenizer-internals.md)/[02 类](02-model-loading-and-autoclass.md)目前都只演示过 `from_pretrained` 加载别人训练好的结果,没有现场训练过)内部是 Rust 实现的同一套"统计 → 合并"算法,只是:
  1. 停止条件不是这里用的"固定轮数",而是目标词表大小(`vocab_size`)——基础符号集合大小 + 合并次数 = 最终词表大小,真实训练前要先决定这个数字。
  2. 基础符号集合通常是**字节级**而不是这里的字符级——GPT-2/RoBERTa 那一支 BPE 用 256 个字节值打底,天然不会有本文 `zebra` 例子里"某个字符训练语料没见过"的情况。
  3. 真实语料是几十万到几十亿字符规模,`get_pair_counts` 这种"每轮全量重新扫描整个词表"的写法跑不动——Rust 实现会用更高效的数据结构(比如维护一个按频率排序的堆,每次合并只增量更新受影响的部分),这里为了教学时读者能一眼看懂算法本身,选的是最直白但最慢的写法。
  4. 还有 special tokens 处理、pre-tokenization 规则(比如 GPT-2 用正则先把文本切成不跨越单词边界的片段,再分别对每个片段跑 BPE)等等,都是训练主循环之外的工程细节,这篇没有涉及。
- **建议对照的真实例子**:[01 类第 2 点](01-tokenizer-internals.md)已经验证过真实 SentencePiece tokenizer 的例子(`"unbelievability"` → `['▁un', 'bel', 'iev', 'ability']` 4 个子词)——同样的"统计+合并"算法,规模从这里的 20 个词、30 轮合并,放大到几十万词表、真实语料,产出"常见词/词根整体保留、生僻词被拆碎"这个现象在直觉上是完全一致的。
- **词表大小怎么选是一个真实的工程权衡**:01 类"面试怎么问"部分已经提到过——词表越大,罕见 token 的 embedding 训练不充分;词表越小,序列越长,计算量越大。这篇教程用"跑 30 轮"代替了这个决策,真实训练需要按语料规模和下游任务认真选。

## 这篇教程展示的方法论

这篇教程复用的是 dsa-deep-dive 系列率先验证过的"教程体"模式([dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md)):挑一个已经讲过的知识点(这里是 [01 类](01-tokenizer-internals.md)的 BPE 训练算法演示),往前后各补一段它原本没走到的路(真实语料的词频统计、训练完之后真正用来给新词推理),分阶段增量实现,每一步都跑出真实输出再往下写——包括"22 轮出现并列""grunting 只合并了一半"这类不完美但真实的结果,不为了叙事干净而回避或手工修饰。这是把该格式从 dsa-deep-dive 推广到本系列的一次落地;是否/如何继续推广到仓库其余系列,留给后续统一决定,这里不展开。

---

*创建:2026-07-24*
