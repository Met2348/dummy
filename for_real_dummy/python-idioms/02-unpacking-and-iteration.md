# 02 · 解包与迭代惯用法(Unpacking & Iteration Idioms)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 7 个知识点:序列解包与多重赋值、星号解包、`enumerate`、`zip`、`itertools`(`combinations`/`chain`/`pairwise`)、`any`/`all` 短路求值、链式比较。这些写法单独看都很简单,但缺一个就会让你在读博士学长的训练/评测脚本时,每次碰到 `for a, b in zip(...)`、`lo, hi = f(stage)`、`if any(... for ... in ...)` 这类写法都要停下来手动在脑子里展开——本文按"先笨办法,再引出 pythonic 写法解决了什么问题"的顺序讲。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9,torch 2.11.0+cu128,numpy 2.4.6)下实际跑通验证,不是凭空写的。"AI 研究代码里的真实例子"部分,能直接 import 仓库里 `learning/` 目录下真实模块跑的都直接 import 调用(比如 `bpe_trainer.py`/`long_data_packing.py`/`integrity/data.py`/`langchain_style.py` 里的函数都是原样跑的);其中 `int8_basics.py` 的 `*prefix, K = x.shape` 引用的是原文真实存在的代码(已核对行号),但因为它内部是真正的 `torch.Tensor` 运算,这里改用 `numpy.ndarray` 把同一套解包机制单独跑一遍验证——`torch.Tensor` 和 `numpy.ndarray` 的 `.shape`/`.reshape()` 接口是特意保持一致的,行为完全等价,文中会说明。挖不到真实例子的知识点(`itertools.chain`)会如实标注"示例性质",不编造引用。

---

## 1. 序列解包与多重赋值

**是什么:** 一行代码把一个序列(tuple/list/...)"拆开",按位置分别赋给多个变量——最有名的例子是不用临时变量交换两个变量:`a, b = b, a`。

**为什么课堂不教但很重要:** 本科课堂教的赋值几乎都是"一条语句一个左值"(`x = 1`、`y = 2`),交换两个变量要老老实实写 `tmp = a; a = b; b = tmp`。Python 允许等号左边写一串用逗号分开的名字、右边写一串同样个数的值,一次性对应赋值——很多人第一次见会当成"魔法:两边同时生效",但背后的顺序其实很朴素:**右边先被整体求值、打包成一个 tuple,然后再按从左到右的顺序,把这个 tuple 拆开依次赋给左边的每个名字**。研究代码里,一个函数一次返回好几个值(比如 `(min_len, max_len)`)时,调用方几乎总是用这种一行解包接住,而不是先接一个 tuple 再手动按下标取。

**从最笨的写法讲起:**
```python
# 笨办法: C 风格, 借助临时变量交换
a = 1
b = 2
tmp = a
a = b
b = tmp
assert (a, b) == (2, 1)

# 正式写法: 序列解包, 不需要显式临时变量
a, b = 1, 2
a, b = b, a
assert (a, b) == (2, 1)
```
两种写法结果一样,但解包版本少了一行,也不用占用一个只用一次就该扔掉的 `tmp` 名字。

**这是"两边同时生效"的魔法吗?** 不是。Python 官方语言参考手册在讲赋值语句时,专门举了一个例子戳穿这个误解:
```python
x = [0, 1]
i = 0
i, x[i] = 1, 2
# 如果真的是"两边同时生效", x[i] 用的应该是"旧" i=0, 结果该是 x == [2, 1]
# 但 Python 的真实顺序是: 先把右边整体算成一个 tuple (1, 2); 再按从左到右赋值:
#   第一步 i = 1                 (i 已经变成新值 1 了)
#   第二步 x[i], 这里下标 i 现在已经是 1, 所以这一步实际执行的是 x[1] = 2
assert i == 1
assert x == [0, 2]
```
用 `dis` 模块看字节码也能验证"右边先求值完、再依次赋值"这件事——两次 `LOAD` 都发生在 `STORE` 之前:
```python
import dis
dis.dis(compile("a, b = b, a", "<test>", "exec"))
```
输出(3.13,关键几行):
```text
LOAD_NAME                0 (b)
LOAD_NAME                1 (a)
SWAP                     2
STORE_NAME               1 (a)
STORE_NAME               0 (b)
```
先把 `b`、`a` 的值都读到栈上(两次 `LOAD` 都在任何 `STORE` 之前发生),`SWAP` 交换栈顶两个值的位置,最后才依次 `STORE`——赋值真正发生前,右边已经完整算好了,不存在"一边算一边赋"的说法。

**AI 研究代码里的真实例子:** `learning/long-context/src/long_data_packing.py` 第75行,用一行解包接住函数返回的 `(min_len, max_len)` 元组:
```python
def curriculum_lengths(stage: int) -> tuple:
    """阶段 → (min_len, max_len)."""
    stages = {1: (256, 2048), 2: (2048, 8192), 3: (8192, 32768), 4: (32768, 131072)}
    return stages.get(stage, (256, 2048))

def filter_by_curriculum(docs: list, stage: int) -> list:
    """按当前 stage 过滤 doc 长度."""
    lo, hi = curriculum_lengths(stage)          # <- 序列解包, 一行接住两个返回值
    return [d for d in docs if lo <= len(d) <= hi]
```
这是课程学习(curriculum learning)按训练阶段过滤文档长度的真实写法——不解包的话就得写 `result = curriculum_lengths(stage); lo = result[0]; hi = result[1]`,多两行还容易在维护时把下标搞反。跑一遍确认(函数体逐字摘自原文件,行为与原文件完全一致):
```python
def curriculum_lengths(stage: int) -> tuple:
    stages = {1: (256, 2048), 2: (2048, 8192), 3: (8192, 32768), 4: (32768, 131072)}
    return stages.get(stage, (256, 2048))

def filter_by_curriculum(docs: list, stage: int) -> list:
    lo, hi = curriculum_lengths(stage)
    return [d for d in docs if lo <= len(d) <= hi]

lo, hi = curriculum_lengths(2)
assert (lo, hi) == (2048, 8192)

docs = ["x" * 100, "x" * 3000, "x" * 9000]
kept = filter_by_curriculum(docs, 2)
assert kept == [docs[1]]   # 只有长度落在 [2048, 8192] 区间的第二篇被留下
```

**可运行例子:** 用解包一次性拿到 `min`/`max`/`mean` 三个统计量,替代手写三次单独赋值:
```python
def stats(nums):
    return min(nums), max(nums), sum(nums) / len(nums)

losses = [0.9, 0.5, 0.3, 0.7, 0.2]
lo, hi, avg = stats(losses)
assert lo == 0.2
assert hi == 0.9
assert abs(avg - 0.52) < 1e-9
print(f"loss range: [{lo}, {hi}], avg={avg:.3f}")
```

**常见坑:**

**1. 解包目标数量必须完全匹配,否则直接 `ValueError`。**
```python
try:
    a, b = 1, 2, 3
except ValueError as e:
    print(e)   # too many values to unpack (expected 2)

try:
    a, b, c = 1, 2
except ValueError as e:
    print(e)   # not enough values to unpack (expected 3, got 2)
```

**2. `i, x[i] = 1, 2` 不是要背的冷知识,而是提醒你:如果多重赋值左边的某个目标本身是下标/属性表达式(比如 `x[i]`),并且这个表达式用到的变量(`i`)自己也是本次赋值的另一个目标,那么这个下标是在赋值真正执行到它那一步、按从左到右的顺序求值的——这时候前面的目标可能已经被改过了。** 像 `a, b = b, a` 或者两个下标都是字面量常量的 `a[0], a[1] = a[1], a[0]` 永远安全,因为右边的 tuple 在任何赋值发生前就已经完整算好、且左边目标互不依赖:
```python
arr = [1, 2, 3]
arr[0], arr[1] = arr[1], arr[0]
assert arr == [2, 1, 3]   # 安全: 下标 0、1 是常量, 不受赋值影响
```
真正会踩坑的只有"用其中一个目标的新值去算另一个目标的位置"这种比较刁钻的写法,平时很少见,但线上 bug 报告里偶尔会出现,知道这个规则能帮你更快定位。

---

## 2. 星号解包

**是什么:** 赋值时的 `a, *rest = lst` 是把序列"解包"成"一个变量 + 剩下打包成 list";调用时的 `func(*my_list, **my_dict)` 是把一个已有的 list/dict"摊开",变成一个个独立的位置/关键字参数传进去。这和 [python-advanced/01](../python-advanced/01-functions-and-closures.md) 讲过的**定义时**的 `def func(*args, **kwargs)`(把调用方传来的"多余"参数打包收集)方向正好相反——三处用的是同一个符号 `*`/`**`,含义却分别是"部分解包"、"摊开传参"、"收集打包",非常容易混。

**为什么课堂不教但很重要:** 本科课堂顶多讲到"定义函数时的 `*args`",很少提"赋值语句左边也能用 `*`"和"调用时用 `*` 把一个已有列表整个摊开传进去"。研究代码里,`nn.Sequential(*layers)` 这种"我手上有一个 list,想把它变成传给某个函数的一堆独立参数"的场景随处可见——不用 `*` 解包的话,只能笨拙地写 `nn.Sequential(layers[0], layers[1], layers[2], ...)`,而且这要求层数在写代码的时候就已知,根本没法应对"层数是运行时变量"的情况。

**从最笨的写法讲起:**
```python
lst = [1, 2, 3, 4, 5]

# 笨办法: 手动切片
first_dumb = lst[0]
rest_dumb = lst[1:]
assert first_dumb == 1
assert rest_dumb == [2, 3, 4, 5]

# 正式写法: 星号解包赋值
first, *rest = lst
assert first == 1
assert rest == [2, 3, 4, 5]
assert isinstance(rest, list)   # 注意: * 收集到的永远是 list, 不管原序列是 list 还是 tuple

# * 可以出现在开头、中间、末尾, 但一条赋值语句里最多只能有一个 *
*init, last = lst
assert init == [1, 2, 3, 4] and last == 5

first2, *mid, last2 = lst
assert first2 == 1 and mid == [2, 3, 4] and last2 == 5
```

**"打包"和"解包"的方向正好相反,别搞混:**
```python
def pack_demo(*args, **kwargs):
    # 定义时: *args/**kwargs 把"调用方传来的、签名里没显式列出的"参数打包收集成 tuple/dict
    return args, kwargs

packed_args, packed_kwargs = pack_demo(1, 2, x=3)
assert packed_args == (1, 2)
assert packed_kwargs == {"x": 3}

def plain(a, b, x=0):
    return a, b, x

my_list = [1, 2]
my_dict = {"x": 3}
# 调用时: * / ** 把一个已经存在的 list/dict "摊开"成一个个独立的位置/关键字参数
result = plain(*my_list, **my_dict)
assert result == (1, 2, 3)
```
判断口诀:**函数签名里的 `*`/`**` 是打包(把散装的参数收进一个容器);等号右边或者函数调用括号里的 `*`/`**` 是解包(把一个已有容器倒成散装的东西)。**

**AI 研究代码里的真实例子:** `learning/quantization-deploy/src/int8_basics.py` 第38、41、45 行,int8 分组量化(per-group quantization)函数里同时用到了赋值时的星号解包和调用时的星号解包,这是原文(只摘录相关三行,`...` 处省略了和本节主题无关的量化数值计算):
```text
def quantize_per_group(x: torch.Tensor, group_size: int = 128, n_bits: int = 4):
    ...
    *prefix, K = x.shape                              # 赋值时解包: 形状拆成"前面几维" + 最后一维 K
    ...
    x_grouped = x.reshape(*prefix, g, group_size)      # 调用时解包: prefix 这个 list 摊开成 reshape 的位置参数
    ...
    return q.reshape(*prefix, K).to(torch.int8), scale.squeeze(-1)
```
这是给大模型权重做分组量化的真实代码:输入形状可能是 `(batch, seq, hidden)`,也可能只有 `(hidden,)`——`*prefix, K = x.shape` 不管前面有几维,统一把"最后一维"抠出来分组量化,前面所有维度原样保留。不用星号解包的话,不同维度数的输入就得写不同的代码分支去处理。

`torch.Tensor` 的 `.shape`/`.reshape()` 在这一点上和 `numpy.ndarray` 是特意保持一致的接口,下面用 numpy 数组把同一套解包机制单独跑一遍,验证行为完全等价:
```python
import numpy as np

x = np.random.default_rng(0).standard_normal((2, 3, 256)).astype(np.float32)
*prefix, K = x.shape                            # 和 int8_basics.py 第38行完全相同的写法
assert prefix == [2, 3]
assert K == 256

group_size = 128
g = K // group_size
x_grouped = x.reshape(*prefix, g, group_size)    # 和第41行完全相同的写法
assert x_grouped.shape == (2, 3, 2, 128)

back = x_grouped.reshape(*prefix, K)             # 和第45行完全相同的写法
assert back.shape == x.shape
assert np.allclose(back, x)
```

调用时星号解包的另外两个真实例子——`learning/auto-research-frontier/m9.8-redteam-and-integrity/src/integrity/data.py` 第38行,把一个函数的返回值直接解包传给下一个函数,这是原文(逐字摘抄,可以独立运行):
```python
import hashlib
import numpy as np

def make_dataset(name: str):
    """确定性生成 (X, y)。easy-v1 无噪声、hard-v2 有噪声。"""
    if name == "easy-v1":
        rng, noise = np.random.default_rng(1), 0.0
    elif name == "hard-v2":
        rng, noise = np.random.default_rng(2), 0.8
    else:
        raise KeyError(name)
    X = rng.normal(size=(120, 2))
    logits = X[:, 0] + X[:, 1] + rng.normal(scale=noise, size=120)
    y = (logits > 0).astype(int)
    return X, y

def fingerprint(X, y) -> str:
    """数据集指纹: 内容的 sha256."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]

KNOWN_FINGERPRINTS = {
    name: fingerprint(*make_dataset(name)) for name in ("easy-v1", "hard-v2")
}
assert KNOWN_FINGERPRINTS["easy-v1"] == "7fa36a8671d471f4"
```
`make_dataset(name)` 返回一个 `(X, y)` 元组,`fingerprint(*make_dataset(name))` 直接把这个元组解包成 `fingerprint` 的两个位置参数——这是数据集指纹校验(防止"偷换数据集"这类评测造假)的真实写法,不用先接一个中间变量。跑一遍:
```python
import sys
sys.path.insert(0, r"e:\Workspace\dummy\learning\auto-research-frontier\m9.8-redteam-and-integrity\src")
from integrity.data import make_dataset, fingerprint, KNOWN_FINGERPRINTS

fp = fingerprint(*make_dataset("easy-v1"))
assert fp == KNOWN_FINGERPRINTS["easy-v1"]
print("fingerprint:", fp)
```
`**` 版本见 `learning/agent-framework-stack/src/langchain_style.py` 第52行,一个 LangChain 风格的最小实现,把一个 dict 直接解包成 `str.format` 的关键字参数,这是原文(逐字摘抄,`Runnable` 基类只保留本节用得到的部分):
```python
class Runnable:
    def invoke(self, input):
        raise NotImplementedError

class PromptTemplate(Runnable):
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input):
        if isinstance(input, dict):
            return self.template.format(**input)   # 调用时解包: dict 摊开成 format 的关键字参数
        return self.template.format(input=input)

tpl_direct = PromptTemplate("Hello {name}")
assert tpl_direct.invoke({"name": "Bob"}) == "Hello Bob"
```
再跑一遍原始文件里的真实版本(直接 import,不是照抄):
```python
import sys
sys.path.insert(0, r"e:\Workspace\dummy\learning\agent-framework-stack\src")
from langchain_style import PromptTemplate

tpl = PromptTemplate("Hello {name}, you asked: {question}")
out = tpl.invoke({"name": "Alice", "question": "what is RAG?"})
assert out == "Hello Alice, you asked: what is RAG?"
```

**可运行例子:** 用调用时的 `**` 解包,把一批配置 dict 直接传给模型构造函数(类似 `cls(**kwargs)` 按不同配置批量构造模型的写法):
```python
class TinyModelConfig:
    def __init__(self, hidden_size, num_layers, dropout=0.1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

configs = [
    {"hidden_size": 128, "num_layers": 2},
    {"hidden_size": 256, "num_layers": 4, "dropout": 0.2},
]
models = [TinyModelConfig(**cfg) for cfg in configs]
assert models[0].hidden_size == 128 and models[0].dropout == 0.1   # 用的是默认 dropout
assert models[1].num_layers == 4 and models[1].dropout == 0.2
```

**常见坑:**

**1. `*rest` 收集到的永远是 list**,即使解包的是 tuple 也一样(上面已经用 `assert isinstance(rest, list)` 验证过)。这一点在后续想对 `rest` 做 `.append()` 这类 list-only 操作时是好事,但如果你期待"结果和原容器类型一致"就会踩坑。

**2. 一条赋值语句里只能有一个带星号的目标,这是语法层面直接报错,不是运行时错误:**
```python
try:
    compile("a, *b, *c = [1, 2, 3]", "<test>", "exec")
except SyntaxError as e:
    print(e)   # multiple starred expressions in assignment
```

**3. 调用时 `*`/`**` 解包一个长度/键不匹配的对象,报错发生在调用那一行,而不是在定义函数的地方——排查时容易先跑去看错的地方:**
```python
def plain(a, b, x=0):
    return a, b, x

my_list_bad = [1, 2, 3, 4]   # 故意比 plain 能接收的位置参数多
try:
    plain(*my_list_bad)
except TypeError as e:
    print(e)   # plain() takes from 2 to 3 positional arguments but 4 were given
```

---

## 3. `enumerate`

**是什么:** 给一个可迭代对象顺便生成从 0(或指定起始值)开始的下标,一次性把 `(index, value)` 打包成 tuple 产出,不用自己维护一个计数器变量。

**为什么课堂不教但很重要:** 手写计数器(`i = 0; for x in lst: ...; i += 1`)不仅多写两行,还是一个真实的 bug 温床:只要循环体里某个分支有 `continue` 却忘了在那之前 `i += 1`,计数器就会和"真实想表达的量"悄悄错位——而且不会报错,只会安静地产生错误结果,是那种"跑起来能过、但数字不对"的隐蔽 bug。研究代码里几乎所有"打印进度"、"给日志加序号"、"要同时用下标和值"的循环都用 `enumerate`,不手写计数器。

**从最笨的写法讲起:**
```python
turns = ["hi", "my name is bob", "i like pizza"]

# 笨办法: 手写计数器
i = 0
dumb_result = []
for text in turns:
    dumb_result.append((i, text))
    i += 1
assert dumb_result == [(0, "hi"), (1, "my name is bob"), (2, "i like pizza")]

# 正式写法: enumerate
result = list(enumerate(turns))
assert result == dumb_result
```

手写计数器的隐蔽 bug 现场重现(对应上面"为什么课堂不教但很重要"里提到的问题):
```python
lines = ["# comment", "x = 1", "# comment", "y = 2"]

# 笨办法: 跳过注释行时忘了同步计数器
i = 0
buggy_numbered = []
for line in lines:
    if line.startswith("#"):
        continue          # 跳过注释, 但忘了 i += 1!
    buggy_numbered.append((i, line))
    i += 1
assert buggy_numbered == [(0, "x = 1"), (1, "y = 2")]   # 编号从0连续递增, 但已经不是"真实行号"了

# enumerate 天然不会有这个问题: 下标是 enumerate 按"迭代到第几次"自动生成的, 不需要手动维护
correct_numbered = [(idx, line) for idx, line in enumerate(lines) if not line.startswith("#")]
assert correct_numbered == [(1, "x = 1"), (3, "y = 2")]   # 保留了真实的原始行号 1 和 3
```

**AI 研究代码里的真实例子:** `learning/agent-memory-context/src/capstone_memory_chat.py` 第55行,处理多轮对话时用 `enumerate(TURNS, start=1)` 生成"第几轮"编号,这是原文(`TURNS`/`episodic`/`extract_facts` 是同文件里别处定义的对话数据和记忆模块,这里只摘录用到 `enumerate` 的那部分):
```text
for i, user_text in enumerate(TURNS, start=1):
    episodic.add("alice", "user", user_text)
    facts = extract_facts(user_text)
    ...
```
`start=1` 是 `enumerate` 的第二个参数,让编号从 1 开始而不是 0——正好对应"第 1 轮对话"这种人类习惯的计数方式,不用再手写 `i = 1` 然后在循环体末尾 `i += 1`。跑一遍等价复现:
```python
turns = ["hi, my name is alice", "what's rag?", "how does colbert work?"]

log = []
for i, user_text in enumerate(turns, start=1):
    log.append(f"[turn {i}] {user_text}")
assert log[0] == "[turn 1] hi, my name is alice"
assert log[-1] == "[turn 3] how does colbert work?"
```

**可运行例子:** 找出 loss 最低的是第几个 epoch(同时需要下标和值的典型场景):
```python
losses = [0.9, 0.5, 0.3, 0.6, 0.2, 0.4]
best_epoch, best_loss = min(enumerate(losses), key=lambda pair: pair[1])
assert best_epoch == 4
assert best_loss == 0.2
print(f"best epoch: {best_epoch}, loss={best_loss}")
```

**常见坑:**

**1. `enumerate` 返回的是惰性迭代器(和 `zip`/`map` 一样),只能消费一次:**
```python
it = enumerate(["a", "b"])
first_pass = list(it)
second_pass = list(it)     # it 已经耗尽
assert first_pass == [(0, "a"), (1, "b")]
assert second_pass == []   # 空的! 不会报错, 但结果是空
```

**2. `start=` 只影响下标的起始值,不影响"跳过前几个元素"——`enumerate(lst, start=5)` 依然是从 `lst[0]` 开始遍历,只是第一个下标标成 5,不是从 `lst[5]` 开始遍历:**
```python
lst = ["a", "b", "c"]
result = list(enumerate(lst, start=5))
assert result == [(5, "a"), (6, "b"), (7, "c")]   # 遍历的还是全部三个元素, 不是从下标5的元素开始
```

---

## 4. `zip`

**是什么:** 把多个可迭代对象"并排"打包,每次各拿一个元素组成一个 tuple,产出一个新的迭代器;长度以最短的那个为准(默认静默截断,3.10+ 可以用 `strict=True` 关掉这个隐藏行为)。

**为什么课堂不教但很重要:** 手写下标并行遍历多个列表(`for i in range(len(a)): ... a[i] ... b[i] ...`)在处理"错位配对"时特别容易犯下标越界或者"多算一个/少算一个"这类差一错一(off-by-one)的错误。`zip` 把这类下标算术完全藏起来,而且有一个课堂几乎不会提的巧妙用法:把同一个序列和它"自己错开一位的切片"配对(`zip(seq, seq[1:])`),直接拿到所有相邻元素对——这是统计 bigram/相邻 token pair 的标准写法。

**从最笨的写法讲起:**
```python
names = ["alice", "bob", "carol"]
ages = [20, 21, 22]

# 笨办法: 手写下标并行遍历
dumb_pairs = []
for i in range(len(names)):
    dumb_pairs.append((names[i], ages[i]))
assert dumb_pairs == [("alice", 20), ("bob", 21), ("carol", 22)]

# 正式写法: zip
pairs = list(zip(names, ages))
assert pairs == dumb_pairs
```

**AI 研究代码里的真实例子:** `learning/data-curation/src/bpe_trainer.py` 第20行,手写 BPE(Byte Pair Encoding)训练器统计"相邻 token pair 出现次数"时,用 `zip(ids, ids[1:])` 把序列和"错开一位的自己"配对,这是原文:
```python
import collections

def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts
```
`ids` 和 `ids[1:]`(去掉第一个元素的 `ids`)一起 `zip`:第一轮拿到 `(ids[0], ids[1])`,第二轮拿到 `(ids[1], ids[2])`……天然产出所有"相邻两个 token"的 pair;长度不匹配的那一位(最后一个元素没有下一位可配对)被 `zip` 按最短长度自动截断丢弃,不用手写 `range(len(ids) - 1)` 这种容易差一错一的边界条件。跑一遍真实函数(逐字摘自原文件):
```python
import collections

def _get_pair_counts(ids):
    counts = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

ids = [104, 101, 108, 108, 111]   # "hello" 的 UTF-8 字节值
counts = _get_pair_counts(ids)
assert counts[(104, 101)] == 1    # (h, e)
assert counts[(108, 108)] == 1    # (l, l)
assert sum(counts.values()) == len(ids) - 1   # 5个token只有4对相邻pair
```

**`zip(*matrix)` 转置技巧:**
```python
matrix = [[1, 2, 3], [4, 5, 6]]   # 2 行 3 列
transposed = list(zip(*matrix))
assert transposed == [(1, 4), (2, 5), (3, 6)]   # 变成 3 行 2 列 (每个元素是 tuple)
```
`*matrix` 先把 `matrix` 这个"行的列表"解包成两个独立的行参数传给 `zip`,`zip` 再按"列"重新打包——这是不用 numpy 就能一行转置嵌套列表的经典写法。

**`zip` 默认静默截断 vs `strict=True`(3.10+):**
```python
a = [1, 2, 3]
b = [10, 20]        # 故意比 a 短一个

silently_truncated = list(zip(a, b))
assert silently_truncated == [(1, 10), (2, 20)]   # a 的第三个元素被悄悄丢弃, 不报错也不警告!

try:
    list(zip(a, b, strict=True))
except ValueError as e:
    print(e)   # zip() argument 2 is shorter than argument 1
```
研究代码里,如果 `zip` 的两个输入分别来自"模型输出"和"标签",默认的静默截断是个危险的坑——万一两边长度因为 bug 对不上,`zip` 不会告诉你,只会悄悄丢数据;`strict=True` 能把这种问题从"结果算错但没人发现"变成"当场报错"。

**可运行例子:** `zip` 三个及以上可迭代对象:
```python
epochs = [1, 2, 3]
train_losses = [0.9, 0.5, 0.3]
val_losses = [1.0, 0.6, 0.4]

rows = list(zip(epochs, train_losses, val_losses))
assert rows == [(1, 0.9, 1.0), (2, 0.5, 0.6), (3, 0.3, 0.4)]
for epoch, tl, vl in zip(epochs, train_losses, val_losses):
    print(f"epoch {epoch}: train={tl}, val={vl}")
```

**常见坑:**

**1. `zip` 默认静默截断**(上面已经演示),这是最容易被忽视的坑——3.10 以下版本没有 `strict=True`,只能自己先手动 `assert len(a) == len(b)`。

**2. `zip` 返回的也是惰性迭代器,只能消费一次**,和 `enumerate` 同理。

**3. `zip()` 不传参数时返回空迭代器;`zip(单个可迭代对象)` 会把每个元素包成单元素 tuple**,容易和"直接遍历"搞混:
```python
assert list(zip()) == []
assert list(zip([1, 2, 3])) == [(1,), (2,), (3,)]
```

---

## 5. `itertools` 核心工具:`combinations` / `chain` / `pairwise`

### 5.1 `itertools.combinations`

**是什么:** 从一个可迭代对象里"任选 `r` 个,不考虑顺序、不重复",生成所有组合。

**为什么课堂不教但很重要:** 这类"两两组合"的需求手写通常是双重循环加一个 `i < j` 去重条件,读的时候要在脑子里确认"这个条件到底防住了什么"(不和自己配对?不重复数同一对?)。研究代码里做模型两两对战评测(round-robin battle)、特征两两交叉,天天遇到这个场景,`itertools.combinations` 直接把这层意图写在名字里。

**从最笨的写法讲起:**
```python
models = ["vanilla", "lora", "dpo"]

# 笨办法: 双重循环 + i<j 避免重复/避免和自己配对
dumb_pairs = []
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        dumb_pairs.append((models[i], models[j]))
assert dumb_pairs == [("vanilla", "lora"), ("vanilla", "dpo"), ("lora", "dpo")]

# 正式写法: itertools.combinations
import itertools
pairs = list(itertools.combinations(models, 2))
assert pairs == dumb_pairs
```

**AI 研究代码里的真实例子:** `learning/llm-judge-arena/src/mini_arena.py` 第29行 和 `learning/eval-graduation/src/mini_arena.py` 第47行——两个独立实现的 mini chatbot arena 评测脚本,都用 `itertools.combinations` 生成模型两两对战(round-robin)的所有组合,这是两份原文(各自摘录关键部分,`run_pairwise`/`keys` 是各自文件里别处定义的对战函数和模型列表):
```text
# learning/llm-judge-arena/src/mini_arena.py 第29行
def run_round_robin(models, judge):
    out = []
    for a, b in itertools.combinations(models.keys(), 2):
        out.extend(run_pairwise(models[a], a, models[b], b, judge))
        out.extend(run_pairwise(models[b], b, models[a], a, judge))  # both orderings
    return out

# learning/eval-graduation/src/mini_arena.py 第47行, 另一套独立实现, 同样的组合方式
def run_round_robin():
    ...
    for a, b in itertools.combinations(keys, 2):
        ...
```
两份代码是不同批次、独立写的两个 capstone 项目,却不约而同选了 `itertools.combinations(models.keys(), 2)` 来生成"N 个模型两两对战"的对阵表——说明这是评测/竞技场类代码里的标准写法,不是巧合。等价复现:
```python
import itertools
model_names = ["vanilla", "lora", "dpo", "r1_tiny", "phi_tiny"]
battle_pairs = list(itertools.combinations(model_names, 2))
n = len(model_names)
assert len(battle_pairs) == n * (n - 1) // 2 == 10   # 5 个模型两两对战, 一共 10 场
```

**注意——一个真假辨析:** 仓库里搜"pairwise"这个词,会在 `run_pairwise`、"Bradley-Terry pairwise loss"、"pairwise battles"等地方看到大量出现,但那些全部是"两两比较"这个日常英语含义,和 5.3 节要讲的 `itertools.pairwise()`(取**相邻**元素对,比如 `pairwise([1,2,3,4])` → `(1,2),(2,3),(3,4)`)是完全不同的两个概念,只是刚好撞了同一个单词,不要混为一谈——本节 `combinations` 对应的正是"两两比较"这个含义,`run_pairwise` 就是"跑一场两两对战"的意思。

**可运行例子:** `combinations` 的 `r` 不一定是 2,选 3 个做"三方对比"同理:
```python
import itertools
triples = list(itertools.combinations(["a", "b", "c", "d"], 3))
assert triples == [("a", "b", "c"), ("a", "b", "d"), ("a", "c", "d"), ("b", "c", "d")]
assert len(triples) == 4   # C(4,3) = 4
```

---

### 5.2 `itertools.chain`

**是什么:** 把多个可迭代对象首尾相连,当成一个整体来遍历,不需要先拼出一个新的大列表。

**为什么课堂不教但很重要:** `a + b` 拼接列表虽然直观,但会真的分配一份新内存把两边的元素都拷贝一遍;如果只是想"依次遍历完 a 再遍历 b",这份拷贝完全是浪费。`itertools.chain` 不拼出中间结果,遍历到哪个源就临时"路过"哪个源。

**从最笨的写法讲起:**
```python
list_a = [1, 2]
list_b = [3, 4]
list_c = [5]

# 笨办法: 用 + 拼接出一个新的完整列表, 再遍历 (需要真的分配一份新内存装下所有元素)
dumb_concat = list_a + list_b + list_c
assert dumb_concat == [1, 2, 3, 4, 5]

# 正式写法: itertools.chain, 不需要先拼出中间列表
import itertools
chained = list(itertools.chain(list_a, list_b, list_c))
assert chained == dumb_concat
```

**AI 研究代码里的真实例子:** 这是**示例性质**的写法——本仓库 `learning/` 目录下(排除 vendor 进来的第三方仓库 `official/repos/`)目前没有直接调用 `itertools.chain` 的真实代码,只在 vendor 进来的 `tensor2tensor`/HuggingFace `transformers`(通过 `LoRA` 仓库引入)源码里出现过,如实标注,不拿第三方代码充当"博士学长写的真实例子"。

**可运行例子:** 体现"不需要先拼出中间列表"这个惰性优势——`chain` 会先完整耗尽第一个生成器,再开始耗尽下一个,不会提前把两个生成器都物化成列表:
```python
import itertools

def gen_a():
    for x in [1, 2]:
        yield x

def gen_b():
    for x in [3, 4]:
        yield x

result = list(itertools.chain(gen_a(), gen_b()))
assert result == [1, 2, 3, 4]
```

---

### 5.3 `itertools.pairwise`(3.10+)

**是什么:** 直接生成一个序列里"相邻两个元素"的 pair,比如 `pairwise([1,2,3,4])` → `(1,2), (2,3), (3,4)`。

**为什么课堂不教但很重要:** 在 3.10 加入 `pairwise` 之前,取相邻元素对唯一的写法就是 `zip(seq, seq[1:])`——这行代码本身没什么难的,但第一次见到的人得停下来想一下"为什么是 `[1:]` 不是 `[:-1]`"才能确认它是对的。`pairwise` 把这个手写模式收进标准库,变成一个见名知意的内置函数。

**从最笨的写法讲起 / AI 研究代码里的真实例子(这里合并讲,因为这俩就是同一件事):** 第4节 `zip` 已经讲过 `learning/data-curation/src/bpe_trainer.py` 第20行的 `zip(ids, ids[1:])`——这行代码就是"手写 pairwise"的真实写法,`itertools.pairwise` 就是它的内置替代:
```python
import itertools
ids = [104, 101, 108, 108, 111]

manual_pairs = list(zip(ids, ids[1:]))          # 手写版 (bpe_trainer.py 第20行的真实写法)
builtin_pairs = list(itertools.pairwise(ids))   # 3.10+ 内置替代

assert manual_pairs == builtin_pairs == [(104, 101), (101, 108), (108, 108), (108, 111)]
```
用 `itertools.pairwise` 重写一遍 `_get_pair_counts` 的核心循环,证明可以等价替换真实代码:
```python
import collections
import itertools

def _get_pair_counts(ids):   # 原文件里的真实实现, 逐字摘抄
    counts = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

ids = [104, 101, 108, 108, 111]

counts_builtin = collections.Counter()
for a, b in itertools.pairwise(ids):
    counts_builtin[(a, b)] += 1

assert counts_builtin == _get_pair_counts(ids)
```
`itertools.pairwise(ids)` 比 `zip(ids, ids[1:])` 多一个好处:`ids[1:]` 要先切片出一份新的列表(哪怕只用一次也要付出这份拷贝的代价),`pairwise` 不需要这份额外拷贝,而且不用你自己想清楚切片边界——可读性更直接,一看名字就知道在干什么。

**相关工具一览:** `itertools` 里还有 `groupby`(按连续相同的 key 分组)、`islice`(对迭代器做切片)、`accumulate`(前缀和/前缀聚合)、`starmap`(参数已经打包成 tuple 时用的 `map`)——仓库内暂时没有直接调用这几个的真实例子,这里只是提一下它们的存在,不展开讲。

**常见坑:**

**1. `combinations` 的结果顺序由输入的位置决定,不是按值排序**——如果输入本身没排序,输出的 pair 内部顺序也不会自动排序。

**2. `chain`/`pairwise`/`combinations` 返回的都是惰性迭代器,只能消费一次**,和 `zip`/`enumerate` 同理。

**3. `itertools.pairwise` 是 Python 3.10 才加入的**,3.9 及更早版本只能用 `zip(seq, seq[1:])` 手写——反过来说,老代码里如果看到 `zip(seq, seq[1:])`,大概率是在 `pairwise` 还不存在的年代写的。

---

## 6. `any`/`all` 与短路求值

**是什么:** `any(iterable)` 只要有一个元素为真就立刻返回 `True`,不会遍历完剩下的;`all(iterable)` 只要有一个元素为假就立刻返回 `False`——配合生成器表达式使用时,这个"提前停止"不只是省时间,还能避免生成器表达式里那些"本不该被执行"的副作用被执行到。

**为什么课堂不教但很重要:** 手写 `for + flag + break` 检查"是否存在"/"是否全部满足"啰嗦,而且容易漏写 `break`——漏写不会报错,只是白白多跑几轮,最终结果依然正确但性能变差,是那种很难被发现的"隐性 bug"。研究代码里检查"某个模块名是否匹配任意一个 target 前缀"(LoRA 挑选要注入的层)、"一个 batch 里是否有样本触发了某个越界条件",几乎都是 `any(... for ... in ...)`。

**从最笨的写法讲起:**
```python
data = [-1, -2, 3, -4, -5]

# 笨办法: 手写循环 + flag + break
dumb_found = False
for x in data:
    if x > 0:
        dumb_found = True
        break
assert dumb_found is True

# 正式写法: any + 生成器表达式
found = any(x > 0 for x in data)
assert found is True
```

**证明"短路":生成器表达式不会被跑完,遇到第一个满足条件的就停:**
```python
checked = []

def check(x):
    checked.append(x)
    return x > 0

data = [-1, -2, 3, -4, -5]
result = any(check(x) for x in data)
assert result is True
assert checked == [-1, -2, 3]   # 第三个元素 3 就满足条件了, 后面的 -4, -5 根本没被 check 过
```
如果这里用的是列表推导式 `any([check(x) for x in data])`(注意多了一层 `[]`),`check` 会被无条件对全部 5 个元素调用一次:
```python
data = [-1, -2, 3, -4, -5]

checked_list = []
def check_list(x):
    checked_list.append(x)
    return x > 0

any([check_list(x) for x in data])       # 列表推导式先把 [] 里的内容全部算完, 才交给 any 判断
assert checked_list == [-1, -2, 3, -4, -5]   # 5 个元素全被访问了, 短路完全没生效
```
生成器表达式(不加 `[]`)才能让 `any`/`all` 真正提前停止,这也是为什么 `any`/`all` 几乎总是搭配生成器表达式,而不是列表推导式。

**AI 研究代码里的真实例子:** `learning/lora-family/src/common.py` 第68行,LoRA 挑选要注入的目标层时,判断模块名是否以任意一个 target 名字结尾:
```python
def target_linear_modules(model, target_names=("c_attn",)):
    """找到名字结尾匹配 target_names 的 nn.Linear / GPT-2 Conv1D 模块."""
    matches = []
    for name, module in model.named_modules():
        if any(name.endswith(t) for t in target_names):
            matches.append((name, module))
    return matches
```
等价复现:
```python
target_names = ("c_attn",)
module_names = ["h.0.attn.c_attn", "h.0.mlp.c_fc", "h.1.attn.c_attn"]
matched = [name for name in module_names if any(name.endswith(t) for t in target_names)]
assert matched == ["h.0.attn.c_attn", "h.1.attn.c_attn"]
```
这里 `target_names` 通常只有一两个元素,短路省不了多少时间,但**逻辑上**的意义更重要:只要匹配到一个就不用继续比较剩下的,读起来就是"是否存在一个满足条件的",而不是"数一数有几个满足条件的"。

**可运行例子:** `all` 检查一个 batch 里所有样本的形状是否一致(训练前的合法性检查):
```python
batch_shapes = [(4, 128), (4, 128), (4, 128)]
ok = all(shape == batch_shapes[0] for shape in batch_shapes)
assert ok is True

bad_batch_shapes = [(4, 128), (4, 128), (4, 127)]
ok2 = all(shape == bad_batch_shapes[0] for shape in bad_batch_shapes)
assert ok2 is False
```

**常见坑:**

**1. 空序列:`any([])` 恒为 `False`,`all([])` 恒为 `True`**——这不是 bug,是数学上"存在量词"和"全称量词"对空集的标准定义,但第一次见容易觉得反直觉(尤其是 `all([])` 为 `True`)。
```python
assert any([]) is False
assert all([]) is True
```

**2. 别在 `any`/`all` 外面套 `[]` 把生成器表达式变成列表推导式**(上面已经演示过 `any([check_list(x) for x in data])` 会丧失短路)。数据量大、或者生成器里有昂贵操作(比如每个元素要跑一次模型推理)时,这个多余的 `[]` 可能是性能差几个数量级的根源,而且不会报错,只会"莫名其妙变慢"。

---

## 7. 链式比较 `a < b < c`

**是什么:** `a < b < c` 是一个完整的表达式,等价于 `a < b and b < c`,但中间的 `b` 只会被求值一次——不是"先算出 `a < b` 这个布尔值,再拿它和 `c` 比较"。

**为什么课堂不教但很重要:** C/Java 背景的人第一次看到 `a < b < c` 容易按 C 的语义去理解——在 C 里这会先算出 `a < b` 得到一个 `0`/`1`,再拿这个 `0`/`1` 去和 `c` 比较,是一个隐藏的逻辑 bug 温床(C 编译器不会报错,因为 `int < int` 完全合法,只是结果肯定不是你想要的)。Python 的链式比较是专门设计的语法糖,语义上就是"每一对相邻的都要满足",不是"从左到右依次折叠成一个新的比较"。研究代码里做区间检查(数据长度落在 `[lo, hi]` 之间、概率值落在 `[0, 1]` 之间)几乎都用这个写法。

**从最笨的写法讲起:**
```python
lo, x, hi = 10, 15, 20

# 笨办法: and 拼接两次独立比较
dumb = (lo <= x) and (x <= hi)
assert dumb is True

# 正式写法: 链式比较
chained = lo <= x <= hi
assert chained is True
assert chained == dumb
```

**证明"中间的表达式只求值一次",不是拆成两次独立比较:**
```python
calls = []

def b_with_side_effect():
    calls.append("called")
    return 15

# 如果拆成 10 < b_with_side_effect() and b_with_side_effect() < 20, 会被调用 2 次
result = 10 < b_with_side_effect() < 20
assert result is True
assert calls == ["called"]   # 只调用了一次

# 对比: 手写拆成两个独立的 and 表达式, 会真的调用两次
calls2 = []
def b_side_effect2():
    calls2.append("called")
    return 15

manual_result = (10 < b_side_effect2()) and (b_side_effect2() < 20)
assert manual_result is True
assert calls2 == ["called", "called"]   # 调用了两次!
```
如果 `b_with_side_effect()` 是一次昂贵的操作(比如读一次显存占用、跑一次前向传播),链式比较不只是写起来短,还能真的省掉一次调用。

**AI 研究代码里的真实例子:** `learning/long-context/src/long_data_packing.py` 第76行,课程学习(curriculum learning)按文档长度过滤时用链式比较做区间检查:
```python
def filter_by_curriculum(docs: list, stage: int) -> list:
    """按当前 stage 过滤 doc 长度."""
    lo, hi = curriculum_lengths(stage)
    return [d for d in docs if lo <= len(d) <= hi]
```
跑一遍(函数体逐字摘自原文件):
```python
def curriculum_lengths(stage):
    stages = {1: (256, 2048), 2: (2048, 8192), 3: (8192, 32768), 4: (32768, 131072)}
    return stages.get(stage, (256, 2048))

def filter_by_curriculum(docs, stage):
    lo, hi = curriculum_lengths(stage)
    return [d for d in docs if lo <= len(d) <= hi]

docs = ["x" * 100, "x" * 3000, "x" * 9000]
kept = filter_by_curriculum(docs, stage=2)   # stage=2 对应长度区间 (2048, 8192)
assert kept == [docs[1]]
```

**可运行例子:** 检查一批采样概率是否都落在合法区间:
```python
probs = [0.1, 0.5, 0.9, 0.99]
all_valid = all(0.0 <= p <= 1.0 for p in probs)
assert all_valid is True

probs_bad = [0.1, 1.5, 0.9]
all_valid2 = all(0.0 <= p <= 1.0 for p in probs_bad)
assert all_valid2 is False
```

**常见坑:**

**1. 链式比较不能表达"两两不相等"——`a < b < c` 只是 `a<b and b<c` 的语法糖,不是"a、b、c 互不相同":**
```python
a_val, b_val, c_val = 1, 2, 1
assert not (a_val < b_val < c_val)   # 1 < 2 < 1 为 False, 因为 2 < 1 不成立(不是在检查"三者互不相同")
```

**2. 链式相等比较 `x == y == z` 同理是 `x==y and y==z` 的语法糖**,不是"先算 `x==y` 的布尔值,再拿去和 `z` 比较":
```python
x_val = y_val = 1
z_val = 1
assert (x_val == y_val == z_val) is True
```

**3. 混用不同方向的运算符是合法的,但容易读错。** `a < b > c` 合法,等价于 `a<b and b>c`,不是数学上常见的"单调区间"写法:
```python
assert (1 < 5 > 2) is True   # 等价于 1<5 and 5>2, 都成立
```
读起来容易和 `a < b < c` 弄混,不建议这样写——链式比较最好只在"同一个方向"上用(全是 `<`/`<=`,或全是 `>`/`>=`)。

---

## 小结:这一批 7 个知识点解决的问题

| 知识点 | 解决的问题 |
|---|---|
| 序列解包与多重赋值 | 一行接住多个值,不用临时变量、不用手动按下标取 |
| 星号解包 | 从序列里"掐头去尾"取一部分;把一个现成的容器摊开传给函数 |
| `enumerate` | 需要下标时不用手写、不用维护容易和逻辑分支错位的计数器 |
| `zip` | 并行遍历多个序列,配合切片错位还能拿到相邻元素对 |
| `itertools`(`combinations`/`chain`/`pairwise`) | 组合、拼接、相邻配对——三种嵌套循环/手动切片的标准替代 |
| `any`/`all` 与短路求值 | "是否存在"/"是否全部满足"一行写完,还能在满足条件后提前停止 |
| 链式比较 `a < b < c` | 区间检查一行写完,中间表达式只求值一次 |

下一批:[03-containers-and-stdlib.md](03-containers-and-stdlib.md)

---
