# 01 · 推导式与函数式基础(Comprehensions & Functional Basics)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 7 个知识点:列表推导式语法与嵌套、字典推导式 / 集合推导式、`map`/`filter` vs 推导式、`functools.reduce`、三元表达式、海象运算符 `:=`,以及收尾的 one-liner 取舍。本系列是 [python-advanced/](../python-advanced/00-roadmap.md) 的姊妹篇——那边讲"语言特性"(装饰器、生成器、OOP),这里讲"表达习惯"(同一件事,用什么写法更 pythonic、什么时候一行更清晰、什么时候是硬凑)。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9)下实际跑通验证,不是凭空写的。"AI 研究代码里的真实例子"部分优先引用/复现 `learning/` 目录下博士学长自己写的真实代码(标注了具体文件路径和行号,不包括 vendor 进来的第三方仓库 `official/repos/`);其中 `functools.reduce` 在本仓库自己写的代码里暂未挖到直接用例,已如实标注为示例性质,没有编造假的仓库引用。

---

## 1. 列表推导式语法与嵌套

**是什么:** `[expr for item in iterable]` 用一行生成一个新列表;把方括号里再套一层方括号,或者在同一层写两个 `for`,就能写出嵌套推导式——用来生成二维结构(比如矩阵),或者把嵌套结构"拍平"成一维。

**为什么课堂不教但很重要:** 课堂教的是 `for` 循环 + `.append()` 建列表,一步一步很直白;但 AI 研究代码里为了避免"先声明一个空列表、再写循环体、再 append"这三步样板代码,凡是能一行写完的构造过程基本都会写成推导式,嵌套矩阵/拍平批数据是其中最常见的两种场景。**这一节只讲推导式的语法本身**(嵌套怎么写、多个 for 怎么写)——推导式和生成器表达式的惰性求值/内存对比,[python-advanced/02-iterators-and-generators.md](../python-advanced/02-iterators-and-generators.md) 第 3 节已经用 `sys.getsizeof` 和耗时实测讲过了,这里不重复。

**从最笨的写法讲起:**

先看"构造二维结构"这个场景——手写双重 `for` 循环搭一张乘法表,和嵌套推导式的一行版本对比:
```python
# 笨办法:双重 for 循环 + append,一层一层攒出一张乘法表
table_dumb = []
for i in range(1, 4):
    row = []
    for j in range(1, 4):
        row.append(i * j)
    table_dumb.append(row)

# 正式写法:嵌套列表推导式——外层 for 对应外层循环,内层 for 对应内层循环,顺序完全一致
table_comp = [[i * j for j in range(1, 4)] for i in range(1, 4)]

assert table_dumb == table_comp == [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
print(table_comp)
```

再看"拍平"这个场景——这次不是"推导式套推导式",而是**同一层写两个 `for`**:
```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 笨办法:嵌套 for + append
flat_dumb = []
for row in matrix:
    for x in row:
        flat_dumb.append(x)

# 正式写法:一个推导式里连写两个 for——读的顺序、求值的顺序都和嵌套 for 循环完全一致:
# 先写外层的 `for row in matrix`,再写内层的 `for x in row`,不能颠倒
flat_comp = [x for row in matrix for x in row]

assert flat_dumb == flat_comp == [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(flat_comp)
```

**AI 研究代码里的真实例子:** `learning/kernel-engineering/src/fused_mlp.py` 第 16-18 行,`mlp_unfused` 函数用三层嵌套的列表推导式手写了一个"未融合版" MLP(matmul → GeLU → matmul),整个函数不依赖 numpy/torch,纯 Python 嵌套推导式实现,专门用作数值参考基准(oracle),去验证同文件里手写的"融合版"实现(`mlp_fused`,省掉了中间结果写回显存的开销)数值上是否完全一致:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from fused_mlp import mlp_unfused, _self_test

# 真实写法(节选自 fused_mlp.py:16-18):
#     h   = [[sum(x[i][d] * W1[d][k] for d in range(D)) for k in range(H)] for i in range(N)]
#     h2  = [[gelu(v) for v in row] for row in h]
#     out = [[sum(h2[i][k] * W2[k][d] for k in range(H)) for d in range(D)] for i in range(N)]
x = [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]]        # N=2 个 token, D=4 维
W1 = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]  # D=4 -> H=3
W2 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]    # H=3 -> D=4(投影回原维度)

out = mlp_unfused(x, W1, W2)
assert len(out) == 2 and len(out[0]) == 4
print(out)   # 本机实测: [[0.8411919906082768, 1.954597694087775, 2.996362607918227, 0.0], [0.1158..., 0.2621..., 0.4354..., 0.0]]

_self_test()   # 跑一遍仓库自带的 self test:确认 mlp_unfused 和 mlp_fused 数值一致(本机实测打印 "[OK] fused_mlp (saved 30.8% HBM)")
```

**可运行例子:** 嵌套推导式还有一个常见用途——不用 numpy,纯 Python 转置一个矩阵:
```python
matrix2 = [[1, 2, 3], [4, 5, 6]]     # 2 行 3 列

transposed = [[row[i] for row in matrix2] for i in range(len(matrix2[0]))]
assert transposed == [[1, 4], [2, 5], [3, 6]]                       # 变成 3 行 2 列
assert len(transposed) == len(matrix2[0]) and len(transposed[0]) == len(matrix2)
print(transposed)
```

**常见坑:**

**1. 条件表达式的两种用法极易混淆:`if` 当过滤器,和 `if/else` 当三元表达式,语法位置完全不同。** `[x for x in lst if cond]` 里的 `if` 写在**末尾**,是过滤器——不满足条件的元素直接丢弃,结果长度可能变短;`[a if cond else b for x in lst]` 里的 `if/else` 写在**表达式(最前面)位置**,是三元表达式(第 5 节详细讲)——每个元素都会产出一个结果,只是产出的值不同,结果长度永远不变:
```python
nums = [-2, -1, 0, 1, 2]

# if 在末尾:过滤器,只保留满足条件的元素
filtered = [x for x in nums if x > 0]
assert filtered == [1, 2]              # 长度从 5 变成了 2,元素被丢弃了

# if/else 在前面:三元表达式,对每个元素都产出一个值(这里相当于取绝对值)
signed = [x if x >= 0 else -x for x in nums]
assert signed == [2, 1, 0, 1, 2]       # 长度还是 5,一个不少,只是部分值被替换了

assert len(filtered) != len(nums)
assert len(signed) == len(nums)
```

**2. 嵌套推导式里 `for` 子句的求值顺序和书写顺序完全一致(从左到右,也就是从外到内),写反会直接报错,而不是得到"转置"之类的意外结果:**
```python
matrix3 = [[1, 2], [3, 4]]
try:
    # 顺序写反:内层 for x in row 被写到了外层 for row in matrix3 前面,
    # 而 row 在被用到的这一刻根本还没被定义
    bad = [x for x in row for row in matrix3]
except NameError as e:
    print(e)   # name 'row' is not defined
```

**3. 嵌套层数一多,可读性会直线下降**——超过两层嵌套推导式基本就没法一眼看懂了,这个坑留到第 7 节详细展开。

---

## 2. 字典推导式 / 集合推导式

**是什么:** `{key: value for item in iterable}` 一行生成字典;`{expr for item in iterable}` 一行生成集合(去重、无序)。语法上和列表推导式几乎一样,只是把方括号换成花括号,字典推导式多一个 `key: value` 的冒号。

**为什么课堂不教但很重要:** 课堂通常只教到列表推导式就结束了。但 AI 研究代码里两个高频场景天然对应字典/集合推导式:**建立 id→对象的索引表**(词表、分词表、按名字查配置)天然是字典推导式的活;**从一堆可能重复的数据里提取"出现过的唯一值"**(唯一 token、唯一类别、唯一实验名)天然是集合推导式的活。

**从最笨的写法讲起:**

先看"建索引表"场景:
```python
names = ["alpha", "beta", "gamma"]

# 笨办法:先建一个空字典,再逐个塞进去
by_id_dumb = {}
for i, name in enumerate(names):
    by_id_dumb[i] = name

# 正式写法:字典推导式
by_id_comp = {i: name for i, name in enumerate(names)}

assert by_id_dumb == by_id_comp == {0: "alpha", 1: "beta", 2: "gamma"}
print(by_id_comp)
```

再看"去重"场景:
```python
tags = ["gpu", "cpu", "gpu", "tpu", "cpu"]

# 笨办法:先建一个空集合,再逐个 add
unique_dumb = set()
for t in tags:
    unique_dumb.add(t)

# 正式写法:集合推导式
unique_comp = {t for t in tags}

assert unique_dumb == unique_comp == {"gpu", "cpu", "tpu"}
print(sorted(unique_comp))
```

**AI 研究代码里的真实例子:** 字典推导式的例子来自 `learning/data-curation/src/bpe_trainer.py` 第 41 行——手写 byte-level BPE 分词器训练时,初始词表就是"256 个字节值各自映射到自己",一行字典推导式写完:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\data-curation\src")
from bpe_trainer import BPE

# 真实写法(bpe_trainer.py:41): self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
bpe = BPE()
assert len(bpe.vocab) == 256
assert bpe.vocab[65] == b"A"      # ASCII 65 就是大写字母 A
assert bpe.vocab[0] == b"\x00"
print(bpe.vocab[65], len(bpe.vocab))
```

集合推导式的例子来自研究 agent 工具层 `learning/auto-research-frontier/m9.2-research-agent-core/src/research_agent/corpus.py` 第 54 行的 `_tokens` 函数——检索前把标题/关键词切成一个"去重后的小写词集合",用来计算查询和论文之间的关键词重叠度:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\auto-research-frontier\m9.2-research-agent-core\src\research_agent")
from corpus import _tokens

# 真实写法(corpus.py:54): return {t.strip(".,?!").lower() for t in text.replace("-", " ").split() if t}
toks = _tokens("Multi-Agent Debate, roles!")
assert toks == {"multi", "agent", "debate", "roles"}
print(toks)
```

**可运行例子:** 把上面两个真实场景缩成一个小例子——同时建词表、同时提取语料里的唯一词:
```python
vocab = {i: chr(97 + i) for i in range(5)}     # 0->'a', 1->'b', ..., 模仿"id -> 符号"的词表初始化
assert vocab == {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

corpus_text = "the cat sat on the mat, the CAT ran."
unique_words = {w.strip(",.").lower() for w in corpus_text.split()}
assert unique_words == {"the", "cat", "sat", "on", "mat", "ran"}
print(vocab, sorted(unique_words))
```

**常见坑:**

**1. key 重复时,后面的值会静默覆盖前面的,不会报错也不会警告:**
```python
pairs = [("a", 1), ("b", 2), ("a", 99)]
d = {k: v for k, v in pairs}
assert d == {"a": 99, "b": 2}     # "a" 原来的 1 被 99 悄悄覆盖了
print(d)
```

**2. `{}` 是空字典,不是空集合——构造空集合必须显式写 `set()`。** 这是新手极易踩的坑,因为字典推导式和集合推导式长得太像了:
```python
empty_dict = {}
empty_set_wrong = {}      # 直觉上像空集合,实际上还是空字典!
empty_set_right = set()
assert type(empty_dict) is dict
assert type(empty_set_wrong) is dict
assert type(empty_set_right) is set
```

**3. 集合是无序的,不能假设遍历/打印顺序和插入顺序一致。** 如果需要"去重但保留原始顺序",集合推导式做不到,标准写法是 `dict.fromkeys()`(字典从 3.7 起保证插入顺序):
```python
items = ["c", "a", "b", "a", "c"]
as_set = {x for x in items}                    # 去重了,但不保证顺序
ordered_unique = list(dict.fromkeys(items))     # 去重 + 保留首次出现的顺序
assert ordered_unique == ["c", "a", "b"]
assert set(as_set) == set(ordered_unique)       # 内容相同,但顺序这个维度上两者不能划等号
print(ordered_unique)
```

---

## 3. `map`/`filter` vs 推导式

**是什么:** `map(func, iterable)` 把 `func` 应用到每个元素上,`filter(func, iterable)` 只保留 `func(x)` 为真的元素——两者都返回一个惰性迭代器(不是 list),是 Python 从函数式语言里继承来的两个内置工具,和推导式做的是同一件事的两种写法。

**为什么课堂不教但很重要:** 课堂可能提过 `map`/`filter` 是"函数式编程"的例子,但很少讲清楚"什么时候该用它们,什么时候该用推导式"——这不是纯粹的风格问题,是有判断依据的:**已经有一个现成的具名函数、不需要专门包一层 `lambda` 时,`map`/`filter` 往往更简洁;需要临时写判断逻辑时,推导式的可读性通常更好**,因为 `for x in iterable` 把变量的来源直接摆在读者眼前,不用先跳进 `lambda` 参数列表里找。

**从最笨的写法讲起:**
```python
nums = [3, 1, 4, 1, 5]

# 笨办法:for 循环 + append
strs_dumb = []
for x in nums:
    strs_dumb.append(str(x))

# map 版本:str 是现成的内置函数,直接传给 map,不用写 lambda x: str(x)
strs_map = list(map(str, nums))

# 推导式版本
strs_comp = [str(x) for x in nums]

assert strs_dumb == strs_map == strs_comp == ["3", "1", "4", "1", "5"]
print(strs_map)
```

**AI 研究代码里的真实例子:** `learning/reasoning-r1/src/r1_zero_track_a.py` 第 41 行,R1-Zero 强化学习任务(GRPO 训练模型解 Countdown 数字游戏)拼装 prompt 时,要把一组 `int` 拼成"7, 3, 12"这样的字符串——`nums` 是 `int` 列表,`str.join` 只接受字符串,`map(str, nums)` 直接用现成的 `str` 函数做转换,不用专门写一个只用一次的 `lambda`:
```python
# 真实写法(r1_zero_track_a.py:41): n_str = ", ".join(map(str, nums))
nums_real = [7, 3, 12]
n_str = ", ".join(map(str, nums_real))
assert n_str == "7, 3, 12"

prompt = f"Use the numbers {n_str} with +, -, *, / and parentheses to make 24.\n"
assert "7, 3, 12" in prompt
print(prompt.strip())
```

**可运行例子:** `filter` 内置函数在本仓库博士学长自己写的代码里(不算 vendor 进来的 `official/repos/`)暂未挖到直接用例——下面这个是示例性质的例子,不是仓库引用:
```python
nums2 = [-3, -1, 0, 2, 5]

# filter 版本:需要一个"判断真假"的函数,这里现场写一个 lambda
positives_filter = list(filter(lambda x: x > 0, nums2))

# 推导式版本
positives_comp = [x for x in nums2 if x > 0]

assert positives_filter == positives_comp == [2, 5]
print(positives_filter)
```

判断依据的具体对比——已有现成函数时 `map` 更省事:
```python
words = ["Model", "GPU", "loss"]
upper_map = list(map(str.upper, words))    # str.upper 是现成的,直接传
upper_comp = [w.upper() for w in words]     # 推导式版本,半斤八两
assert upper_map == upper_comp == ["MODEL", "GPU", "LOSS"]
print(upper_map)
```

条件稍微复杂时,推导式通常更清楚——因为 `for r in records` 把 `r` 的来源摆在明面上,`filter(lambda r: ..., records)` 得先跳进 `lambda` 才知道 `r` 是什么:
```python
records = [{"name": "a", "score": 0.9}, {"name": "b", "score": 0.4}]

passed_filter = list(filter(lambda r: r["score"] >= 0.6, records))
passed_comp = [r for r in records if r["score"] >= 0.6]

assert passed_filter == passed_comp == [{"name": "a", "score": 0.9}]
print(passed_comp)
```

**常见坑:**

**1. `map`/`filter` 返回的是惰性迭代器,不是 list,只能遍历一次,`print` 出来也看不到内容:**
```python
nums3 = [1, 2, 3]
m = map(str, nums3)
print(m)                # <map object at 0x...>,不是 ['1', '2', '3'](地址每次运行都不同)

first_pass = list(m)
second_pass = list(m)   # 第二次是空的——和生成器一样,一次性、用完就空
assert first_pass == ["1", "2", "3"]
assert second_pass == []
```

**2. `filter(None, iterable)` 是个常见速记写法:用 `None` 当"判断函数"传给 `filter`,效果是保留所有 `bool(x)` 为真的元素,丢弃所有假值(`0`/`""`/`None`/`[]` 等):**
```python
raw = ["a", "", "b", None, "c", 0]
cleaned = list(filter(None, raw))
assert cleaned == ["a", "b", "c"]

# 等价的推导式写法,可读性通常更好,因为不用知道"传 None 给 filter"这个隐藏规则
cleaned_comp = [x for x in raw if x]
assert cleaned_comp == cleaned
print(cleaned)
```

---

## 4. `functools.reduce`

**是什么:** `functools.reduce(func, iterable, initial)` 把一个"接收两个参数"的函数连续应用到序列上,从左到右一步步把整个序列"折叠"成一个值——`reduce(f, [a, b, c], init)` 等价于 `f(f(f(init, a), b), c)`。

**为什么课堂不教但很重要:** `reduce` 是函数式编程里"fold"操作的 Python 版本,但 Python 3 特意把它从内置函数移到了 `functools` 模块里(Python 2 时代它还是内置的)。这个"降级"本身就是一个信号:`reduce` 这种"看起来很函数式"的写法,可读性经常配不上它带来的简洁收益,所以官方选择不让它继续占用内置命名空间。**这一节的重点不是教你多用 reduce,而是搞清楚它到底解决什么问题、以及为什么大部分场景里都有更好的替代品。**

**从最笨的写法讲起:** 这一节反过来——`reduce` 才是"看起来更高级但通常更笨"的那个选项:
```python
nums = [2, 3, 4, 5]

# "笨办法"(其实是更清楚的办法):for 循环累乘
product_loop = 1
for x in nums:
    product_loop *= x

# "看起来更函数式"的写法:reduce
from functools import reduce
product_reduce = reduce(lambda acc, x: acc * x, nums, 1)

assert product_loop == product_reduce == 120
print(product_loop, product_reduce)
```
两种写法算出来的结果一样,但 `reduce` 版本需要读者在脑子里模拟"`acc` 从 1 开始,依次乘以 2、3、4、5"这个折叠过程,`for` 循环反而是所见即所得。

**AI 研究代码里的真实例子:** 这是示例性质的例子,仓库内(`learning/` 目录下博士学长自己写的代码,不算 vendor 进来的 `official/repos/`)暂未挖到 `functools.reduce` 的直接用例——这本身也印证了下面要讲的结论:大部分场景确实用不上它,教学仓库里几十个 `src/` 文件都没人主动写过一次。

`reduce` 真正有优势的场景,是需要一个**没有对应内置聚合函数**的自定义二元操作时,比如把一串"单参数函数"依次串联成一个函数(函数组合):
```python
from functools import reduce

def compose(*funcs):
    """把多个单参数函数依次串联:compose(f, g, h)(x) == f(g(h(x)))"""
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)

add_one = lambda x: x + 1
double = lambda x: x * 2
square = lambda x: x ** 2

pipeline = compose(square, double, add_one)     # square(double(add_one(x)))
assert pipeline(3) == square(double(add_one(3))) == 64    # (3+1)*2=8, 8**2=64
print(pipeline(3))
```
这种"把一组东西两两合并,直到只剩一个"的模式,`sum`/`any`/`all` 这些专用聚合函数表达不了,`reduce` 是少数几个能直接写的选项之一。

**可运行例子:** 更常见的情况是"以为需要 reduce,其实有专门的内置函数"——比如求积应该用 `math.prod`,不用现造一个 `reduce`:
```python
import math
from functools import reduce

nums2 = [2, 3, 4, 5]
via_reduce = reduce(lambda acc, x: acc * x, nums2)
via_math_prod = math.prod(nums2)      # Python 3.8+ 专门为"求积"加的内置函数
assert via_reduce == via_math_prod == 120
print(via_math_prod)
```
合并多个字典也是同理,`reduce` 能做,但显式循环 + `.update()` 更直接:
```python
from functools import reduce

dicts = [{"a": 1}, {"b": 2}, {"a": 3, "c": 4}]

merged_reduce = reduce(lambda acc, d: {**acc, **d}, dicts, {})

merged_loop = {}
for d in dicts:
    merged_loop.update(d)

assert merged_reduce == merged_loop == {"a": 3, "b": 2, "c": 4}
print(merged_loop)
```

**常见坑:**

**1. `reduce` 不是内置函数,必须显式 `from functools import reduce`——这是 Python 2 遗留下来的历史坑,Python 2 里它曾经是内置的:**
```python
# 这个代码块里没有 import reduce
try:
    result = reduce(lambda a, b: a + b, [1, 2, 3])
except NameError as e:
    print(e)   # name 'reduce' is not defined
```

**2. 不给 `initial` 参数时,如果 `iterable` 是空的,会直接抛 `TypeError`,而不是返回某个"合理的默认值":**
```python
from functools import reduce
try:
    reduce(lambda a, b: a + b, [])
except TypeError as e:
    print(e)   # reduce() of empty iterable with no initial value

assert reduce(lambda a, b: a + b, [], 0) == 0    # 给了 initial 就不会报错
```

**3. 可读性问题——这是最根本的原因。** `reduce` 的 `lambda` 经常需要读者在脑内完整模拟一遍折叠过程才能明白在算什么;而 `sum()`/`any()`/`all()`/`math.prod()` 这些专用聚合函数,或者干脆写一个显式 `for` 循环,一眼就能看出意图。过度使用 `reduce` 是常见的"伪函数式"反模式——用了函数式的语法,却没有换来函数式该有的可读性收益。

---

## 5. 三元表达式 `x if cond else y`

**是什么:** `x if cond else y` 是 `if/else` 的**表达式版本**——`cond` 为真就取值 `x`,否则取值 `y`,整体是一个表达式,可以直接嵌进赋值语句、函数参数、推导式里,不需要单独占几行语句块。

**为什么课堂不教但很重要:** 课堂教的 `if/else` 是**语句**,只能单独占一块,不能出现在赋值号右边或者参数列表里。AI 研究代码里大量"根据一个条件,在两个值之间选一个"的场景——按 profiling 结果分类"是算力瓶颈还是内存带宽瓶颈"、按序列长度选精度、按分数判定 PASS/FAIL——写成 4 行 `if/else` 块显得特别啰嗦,三元表达式一行写完,还能直接塞进 f-string 或者字典构造里。

**从最笨的写法讲起:**
```python
score = 0.82

# 笨办法:完整的 if/else 语句块,还得先起一个变量名占位
if score >= 0.6:
    verdict_dumb = "PASS"
else:
    verdict_dumb = "FAIL"

# 正式写法:三元表达式,一行完成"判断 + 赋值"
verdict_expr = "PASS" if score >= 0.6 else "FAIL"

assert verdict_dumb == verdict_expr == "PASS"
print(verdict_expr)
```

**AI 研究代码里的真实例子:** `learning/gpu-architecture/src/roofline.py` 第 39 行,roofline 模型分析一个算子是"算力瓶颈"还是"内存带宽瓶颈"——比较算术强度(arithmetic intensity)和 GPU 的 ridge point,一个三元表达式直接给出分类结论:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from roofline import analyze, gemm_profile, layernorm_profile
from common import GPUS

# 真实写法(roofline.py:39): bound = "compute" if ai >= gpu.ridge_point_bf16() else "memory"
h100 = GPUS["H100"]
op_gemm = gemm_profile(m=8192, n=8192, k=8192)          # 大矩阵乘法:算术强度高
op_ln = layernorm_profile(n_tokens=8192, hidden=4096)    # LayerNorm:逐元素操作,算术强度低

result_gemm = analyze(op_gemm, h100)
result_ln = analyze(op_ln, h100)
assert result_gemm["bound_by"] == "compute"    # 本机实测: ai=2730.67,达到 100% 峰值算力
assert result_ln["bound_by"] == "memory"       # 本机实测: ai=2.0,只用到 0.7% 峰值算力,卡在带宽上
print(result_gemm["bound_by"], result_ln["bound_by"])
```
同一个三元表达式,在两种真实工作负载下真的分别走了 `"compute"` 和 `"memory"` 两条分支——不是凑出来的假例子。

**可运行例子:**
```python
def choose_dtype_bytes(seq_len: int) -> int:
    # 序列很长时用 fp8(1 byte)省显存,短序列直接用 bf16(2 bytes)图省事
    return 1 if seq_len > 4096 else 2

assert choose_dtype_bytes(8192) == 1
assert choose_dtype_bytes(512) == 2
print(choose_dtype_bytes(8192), choose_dtype_bytes(512))
```

**常见坑:**

**1. `else` 不能省略——这一点和很多语言的 `cond ? a : b` 一样,但初学者容易先入为主觉得可以像 `if` 语句那样只写一半:**
```python
try:
    compile("x = 1 if True", "<test>", "eval")
except SyntaxError as e:
    print(e)   # invalid syntax
```

**2. 链式/嵌套三元表达式可读性差:**
```python
def grade(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"

assert grade(95) == "A"
assert grade(85) == "B"
assert grade(72) == "C"
assert grade(50) == "F"
print([grade(x) for x in [95, 85, 72, 50]])
```
这一行"能跑",但要读懂它,得从左到右一层层数第几个 `else` 对应哪个分支——超过一次 `else` 分支嵌套基本就该拆成 `if/elif/else`,第 7 节会回来展开讲这个例子。

**3. `and`/`or` 模拟三元表达式的老写法有一个隐藏 bug。** Python 2.5 之前没有三元表达式,当时流行用 `cond and a or b` 模拟,但这个写法在 `a` 本身是假值(`0`/`""`/`None`/`[]`)时会出错:
```python
def old_style_ternary(cond, a, b):
    return cond and a or b

assert old_style_ternary(True, "yes", "no") == "yes"
assert old_style_ternary(False, "yes", "no") == "no"

# 陷阱:cond 是 True,但 a=0 本身是假值,导致 "cond and a" 整体为假,又去取了 b
assert old_style_ternary(True, 0, "fallback") == "fallback"    # 期望 0,实际拿到了 "fallback"!

# 真正的三元表达式没有这个问题,因为它是靠 cond 本身判断,不看 a 的真假
assert (0 if True else "fallback") == 0
print(old_style_ternary(True, 0, "fallback"))
```

---

## 6. 海象运算符 `:=`

**是什么:** `name := expr` 是一个**表达式**——在计算 `expr` 的同时把结果赋值给 `name`,并且整个赋值表达式本身的值就是这个结果,因此可以直接嵌在 `if`/`while` 的条件里,或者推导式的 `if` 子句里。Python 3.8 引入(PEP 572),正式名字是"赋值表达式"(assignment expression),因为运算符长得像一双眼睛加獠牙,通常被叫做"海象运算符"。

**为什么课堂不教但很重要:** 3.8 是 2019 年才发布的,不少教材/课堂内容还停留在"没有 `:=`"的年代。它专门解决一类很具体的问题:**同一个值,要先判断一次、再使用一次,不用 `:=` 就得算(或者查)两遍**——重新计算不只是"多写一行"这么简单,如果这个值来自一次很贵的调用(模型推理、网络请求、缓存查询),两遍就是两倍的真实开销。

**从最笨的写法讲起:** 用一个"调用计数器"来实测——不是靠感觉,是真的数函数被调用了几次:
```python
call_count = 0
def expensive_check(x):
    global call_count
    call_count += 1
    return x * x

data = [3]

# 笨办法:结果被用了两次(一次判断、一次赋值),expensive_check 就被真实调用了两次
call_count = 0
if expensive_check(data[0]) > 5:
    result_dumb = expensive_check(data[0])
assert call_count == 2

# 正式写法:walrus 把"判断"和"取值"合并成一次调用
call_count = 0
if (result_walrus := expensive_check(data[0])) > 5:
    pass
assert call_count == 1

assert result_dumb == result_walrus == 9
print("笨办法调用次数: 2, walrus 调用次数: 1")
```

**AI 研究代码里的真实例子:** `learning/multi-agent-orchestration/lectures/12-cost-multiagent.md` 第 67 行,多智能体成本优化讲义里"用缓存避免重复调用 LLM"的标准写法:
```python
# 真实写法(cost-multiagent.md:67):
#     hash_key = hash(task_description)
#     if cached := cache.get(hash_key):
#         return cached
# 下面用一个带调用计数的 dict 包装类,验证 walrus 版本确实只查了一次缓存
lookups = 0
cache_store = {"task-a": "cached-result"}

class CountingCache:
    def get(self, key):
        global lookups
        lookups += 1
        return cache_store.get(key)

cache = CountingCache()

def handle_request_dumb(task_description):
    if cache.get(task_description):
        return cache.get(task_description)   # 又查了一次缓存
    return "computed-result"

def handle_request_walrus(task_description):
    if cached := cache.get(task_description):
        return cached
    return "computed-result"

lookups = 0
handle_request_dumb("task-a")
assert lookups == 2

lookups = 0
handle_request_walrus("task-a")
assert lookups == 1
print("dumb 查询次数: 2, walrus 查询次数: 1")
```
同一个仓库里 `learning/tool-use-mcp/lectures/04-mcp-server-impl.md` 第 124 行的 `while line := sys.stdin.readline():` 是另一处真实用例(MCP server 从 stdin 逐行读取请求),第 7 节会用这个例子做正面对照。

**可运行例子:** 推导式里避免重复计算,是 `:=` 最常见的用法——`[y for x in data if (y := f(x)) > 0]` 不用 walrus 就得把 `f(x)` 算两遍(一次在 `if` 里判断,一次在输出表达式里取值):
```python
compute_calls = 0
def f(x):
    global compute_calls
    compute_calls += 1
    return x - 3

nums = [1, 2, 3, 4, 5, 6]

compute_calls = 0
result_dumb = [f(x) for x in nums if f(x) > 0]
calls_dumb = compute_calls          # 本机实测: 9 次(6 个元素各判断一次,3 个通过的又各算一次)

compute_calls = 0
result_walrus = [y for x in nums if (y := f(x)) > 0]
calls_walrus = compute_calls        # 本机实测: 6 次(每个元素只在 if 里算一次,y 直接被输出表达式复用)

assert result_dumb == result_walrus == [1, 2, 3]
assert calls_walrus < calls_dumb
print(f"不用 walrus: {calls_dumb} 次调用; 用 walrus: {calls_walrus} 次调用")
```

**常见坑:**

**1. `:=` 不能在语句顶层单独使用,必须配合括号或者出现在 `if`/`while`/推导式这类"条件位置"——这是刻意的语法限制,为了让 `x := 5` 不会和普通赋值语句 `x = 5` 长得太像导致手滑:**
```python
try:
    compile("x := 5", "<test>", "exec")
except SyntaxError as e:
    print(e)   # invalid syntax
```

**2. 反直觉的作用域规则:推导式里的普通 `for` 变量不会泄漏到外层作用域,但 `:=` 赋值的变量会特意泄漏出去——这是 PEP 572 明确设计的行为,不是 bug:**
```python
_ = [i for i in range(5)]
try:
    print(i)     # 推导式有自己的作用域,i 访问不到
except NameError as e:
    print("for 变量不泄漏:", e)

_ = [j for x in range(5) if (j := x * 2) > 4]
print("walrus 变量泄漏出来了:", j)    # 能直接访问到 j——是 x=4 时最后一次赋值的结果
assert j == 8
```
这个差异是故意的:`for` 循环变量的作用域被推导式"隔离"起来,是为了不污染外层命名空间;但 `:=` 赋值的变量如果也被隔离,walrus 想解决的"结果留着外面用"这个需求就落空了,所以语言设计上特意让它能穿透出去。不知道这条规则的话,很容易被这个"意外还活着的变量"绕晕。

---

## 7. one-liner 的取舍——什么时候一行更清晰,什么时候该拆开

**是什么:** 前面 6 节的推导式、`map`/`filter`、`reduce`、三元表达式、`:=` 都有一个共同点——它们都能把原本要写好几行的逻辑压成一行。但"能压成一行"和"应该压成一行"是两件事:这一节综合前面 6 个知识点,做几组"一行确实更清晰"和"硬塞成一行反而该拆开"的正反对照,并给出一个可操作的判断标准。

**为什么课堂不教但很重要:** 课堂不会教这个,因为这不是语法问题,是工程判断力问题——语法书只会告诉你"这样写合法",不会告诉你"这样写要不要被 code review 打回去"。AI 研究代码经常在两个极端反复横跳:要么因为"能跑就行"完全不用这些写法(啰嗦但安全),要么为了炫技把 `map`+`filter`+`reduce`+嵌套三元全部揉进一行(简短但没人敢改)。这一节要建立的是中间的判断力。

**对照组 1:推导式的嵌套深度**

正例——一层过滤 + 一次转换,一行推导式替代 4 行 `for` 循环,信息密度刚好,一眼看完:
```python
logs = ["INFO: start", "ERROR: oom", "INFO: step 1", "ERROR: nan loss"]

errors_loop = []
for line in logs:
    if line.startswith("ERROR"):
        errors_loop.append(line.removeprefix("ERROR: "))

errors_oneline = [line.removeprefix("ERROR: ") for line in logs if line.startswith("ERROR")]

assert errors_loop == errors_oneline == ["oom", "nan loss"]
print(errors_oneline)
```

反例——嵌套推导式 + 三元表达式 + `:=` 全部塞进一行,语法上完全合法,但没人愿意读第二遍:
```python
matrix_data = [[1, -2, 3], [-4, -5, -6], [7, -8, 9]]

# 反例:三层嵌套 + 三元 + walrus 全挤在一行
result_bad = [[(y if (y := v * 2) > 0 else 0) for v in row] for row in matrix_data if any(v > 0 for v in row)]

# 拆开写:每一步给自己的名字,读起来是三个连续动作——
# "过滤掉全负数的行" -> "每行内翻倍" -> "负数清零"
def process_matrix(data):
    rows_with_positive = [row for row in data if any(v > 0 for v in row)]
    doubled = [[v * 2 for v in row] for row in rows_with_positive]
    clipped = [[v if v > 0 else 0 for v in row] for row in doubled]
    return clipped

result_good = process_matrix(matrix_data)
assert result_bad == result_good == [[2, 0, 6], [14, 0, 18]]
print(result_good)
```

**对照组 2:三元表达式的分支数**

正例——一次二选一,三元表达式比 4 行 `if/else` 更直接地表达"这就是个二选一":
```python
def format_count(n):
    return f"{n} item" if n == 1 else f"{n} items"

assert format_count(1) == "1 item"
assert format_count(5) == "5 items"
print(format_count(1), format_count(5))
```

反例——第 5 节留的伏笔:链式三元表达式塞了 3 个 `else`,拆成 `if/elif/else` 之后,新增一档判断只需要照着格式加一行,可读性和可维护性都更好:
```python
def grade_bad(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"

def grade_good(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"

for s in [95, 85, 72, 50]:
    assert grade_bad(s) == grade_good(s)
print([grade_good(s) for s in [95, 85, 72, 50]])
```

**对照组 3:`:=`/`map`/`filter`/`reduce` 的组合数量**

正例——第 6 节引用过的真实 idiom:`while` 条件里放一个 `:=`,只做"读一行、判断是否为空"这一件事,足够简洁也足够清楚(这里用 `io.StringIO` 离线模拟标准输入,不联网也能跑通):
```python
import io

fake_stdin = io.StringIO("first\nsecond\nthird\n")
lines_walrus = []
while line := fake_stdin.readline():
    lines_walrus.append(line.strip())

fake_stdin2 = io.StringIO("first\nsecond\nthird\n")
lines_dumb = []
while True:
    line2 = fake_stdin2.readline()
    if not line2:
        break
    lines_dumb.append(line2.strip())

assert lines_dumb == lines_walrus == ["first", "second", "third"]
print(lines_walrus)
```

反例——把 `map` + `filter` + `reduce` 叠进同一行,不拆开根本猜不出这行在算"偶数的平方和":
```python
from functools import reduce

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 反例:三个函数式工具叠在一行
total_bad = reduce(lambda acc, x: acc + x,
                    filter(lambda x: x > 0,
                           map(lambda x: x ** 2 if x % 2 == 0 else -1, nums)), 0)

# 拆开写:三个步骤各自起名字,配合内置的 sum(),"取偶数 -> 平方 -> 求和"一遍读完
def sum_of_even_squares(data):
    evens = [x for x in data if x % 2 == 0]
    squares = [x ** 2 for x in evens]
    return sum(squares)

total_good = sum_of_even_squares(nums)
assert total_bad == total_good == 220
print(total_good)
```

**判断标准(可操作,不是感觉):**

1. **嵌套推导式超过一层,或者一行里同时出现"过滤条件 + 三元表达式"、"三元表达式嵌套三元表达式"这类组合,就该拆开**——对照组 1、2 的反例都踩了这一条。
2. **需要在脑子里从左到右默读两遍才能确定运算顺序,就该拆开**——`map`/`filter`/`reduce` 叠着用是最容易触发这条的组合,因为读者得从最内层的 `map` 开始,一层层往外剥。
3. **一行如果长到编辑器要自动换行、或者要横向滚动才能看全,基本可以确定要拆**——这不是审美要求,是"读者的短期记忆一次装不下这么多符号"的硬约束。
4. **反过来:如果一行推导式/三元表达式/`:=` 只做一件事(一次过滤、一次二选一、一次"判断+复用"),变量名也说得清楚,那就是一行更清晰的信号**——按标准 1-3 硬拆成 3、4 行,除了变长,不会换来任何可读性收益,对照组 1、2、3 的正例都是这一类。

**常见坑:** 判断标准不是用来教条式套用的公式——同一个团队里,不同 code review 习惯下"能不能接受"的尺度会不一样;而且标准 1-3 说的是"该拆开的信号",不代表一行推导式天生可疑。真正容易出问题的场景,是有人已经隐约觉得"这行不太好懂",却因为"能跑""看起来高级"而不去拆——这一节的判断标准是给这种迟疑一个可以对照检查的清单,不是要你看到 `if`/`for` 同框就一律拆开。

---

## 小结

| 知识点 | 解决的问题 |
|---|---|
| 列表推导式语法与嵌套 | 用一行写完"构造二维结构/拍平嵌套结构",不用手写双重 `for` + `append` |
| 字典推导式 / 集合推导式 | 一行建立 id→对象的索引表,或者从重复数据里提取唯一值集合 |
| `map`/`filter` vs 推导式 | 已有现成具名函数时更简洁;需要临时判断逻辑时推导式通常更可读 |
| `functools.reduce` | 处理"没有对应内置聚合函数"的自定义二元折叠;大部分场景 `sum`/`any`/`all`/`math.prod` 更合适 |
| 三元表达式 | 把"二选一"从语句块压成一个能嵌入赋值/参数/推导式的表达式 |
| 海象运算符 `:=` | 在 `if`/`while`/推导式条件里"判断的同时保留结果",避免同一个值算两遍 |
| one-liner 的取舍 | 嵌套/分支/组合数量超过阈值就拆开;能一遍读完的才配写成一行 |

下一批:[02-unpacking-and-iteration.md](02-unpacking-and-iteration.md)

---

*更新:2026-07-08*
