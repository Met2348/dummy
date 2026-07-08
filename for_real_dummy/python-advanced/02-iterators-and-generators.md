# 02 · 迭代器与生成器(Iterators & Generators)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇解决一个问题:**`for` 循环背后到底在做什么,以及为什么处理"大到读不完"的数据时不能像列表一样把所有结果一次性塞进内存**——迭代器协议是根,`yield`、生成器表达式、`yield from` 都是长在这根上的语法糖,一层比一层省代码。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9)下用
```
"e:\Workspace\dummy\.venv\Scripts\python.exe" 你的脚本.py
```
实际跑通验证,包括内存占用的具体字节数、生成器"只能用一次"的报错和空结果,都是本机真实跑出来的,不是凭空写的。

---

## 1. 迭代器协议:`__iter__` / `__next__`,`for` 循环的真面目

**是什么:** 一个对象只要同时实现了 `__iter__`(返回一个"迭代器")和 `__next__`(每次吐出下一个值,没有更多值时 `raise StopIteration`),它就能被 `for` 循环遍历。`for` 循环并不认识你的类,它只认这两个方法名。

**为什么课堂不教但很重要:** Python 课教了 `for x in range(10)`、`for x in some_list`,但从没人拆开讲过这背后有一套协议。AI 研究代码里,PyTorch 的 `Dataset`/`DataLoader`、HuggingFace `datasets` 的流式(streaming)模式、几乎所有"能被 `for` 遍历的自定义容器",全都是这套协议的具体实现。不懂协议,看到 `for batch in dataloader:` 会觉得是魔法;懂了协议,就知道 `dataloader` 背后无非是有个 `__iter__` 返回了一个"知道怎么产出下一个 batch"的对象,和你自己写的类没有本质区别。

**从最笨的写法讲起:** C 里没有这套协议——想遍历一个"不知道有多少内容,只能一个一个读"的东西,最像的场景是从文件流里读字符,读到 `EOF` 就停:

```c
// C: 从文件流读字符,不知道还有多少个,读到 EOF(一个"结束哨兵值")就停
int c;
while ((c = fgetc(fp)) != EOF) {
    putchar(c);
}
```

Python 的 `for` 循环对任何自定义对象做的事,结构上和这个一模一样,只是把"哨兵值 EOF"换成了"异常 `StopIteration`"。用手写的类实现同样的效果(不用 `yield`,完全自己管状态,对应 C 里"自己维护一个游标变量"):

```python
class CountUpTo:
    """手动实现的迭代器:不用 yield,自己管状态"""
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self          # 协议要求: __iter__ 返回一个"有 __next__ 的对象"

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration      # 相当于 C 里的 EOF,唯一的"结束"信号
        value = self.current
        self.current += 1
        return value
```

如果没有 `for` 语法糖,你想手动做一样的遍历,得这样写——这其实就是解释器在 `for` 循环内部替你做的事:

```python
it = iter(CountUpTo(3))       # 调用 __iter__
result = []
while True:
    try:
        value = next(it)       # 调用 __next__
    except StopIteration:      # 相当于 C 的 EOF 判断
        break
    result.append(value)
```

**AI 研究代码里的真实例子:** PyTorch 的 `Dataset` 只需要实现 `__len__`/`__getitem__`(这是比迭代器协议更简单的"序列协议"),`DataLoader` 包装之后就是一个可迭代对象,`for batch in dataloader:` 内部调用的正是 `iter(dataloader)` 拿到一个迭代器,再反复 `next()`,耗尽时同样是 `StopIteration`。下面是一个真实、跑通的最小例子(在 [02-pytorch-basics.md](../02-pytorch-basics.md) 学过 Tensor 之后回头看这段会很自然):

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    """最小 PyTorch Dataset:只需要实现 __len__ / __getitem__"""
    def __init__(self, n):
        self.data = list(range(n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(ToyDataset(6), batch_size=2, shuffle=False)
loader_iter = iter(loader)          # 和 iter(CountUpTo(3)) 是同一套协议
batch1 = next(loader_iter)          # tensor([0, 1])
```

**可运行例子:**
```python
class CountUpTo:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# 1. for 循环 / list() 能直接用
assert list(CountUpTo(5)) == [0, 1, 2, 3, 4]

# 2. 手动验证 __iter__/__next__ 就是 for 循环背后调用的东西
c = CountUpTo(3)
it = iter(c)
assert it is c                     # __iter__ 返回了 self
assert next(it) == 0
assert next(it) == 1
assert next(it) == 2
try:
    next(it)
    assert False, "应该抛出 StopIteration"
except StopIteration:
    print("手动 next() 三次后,第四次触发 StopIteration —— 符合预期")

# 3. 证明 for 循环 == while True + next() + 捕获 StopIteration
def manual_for(iterable):
    iterator = iter(iterable)
    result = []
    while True:
        try:
            result.append(next(iterator))
        except StopIteration:
            break
    return result

assert manual_for(CountUpTo(4)) == list(CountUpTo(4)) == [0, 1, 2, 3]
```

再用真实的 PyTorch `DataLoader`验证一遍同一套协议(本机实测,`torch` 2.11.0):
```python
import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, n):
        self.data = list(range(n))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(ToyDataset(6), batch_size=2, shuffle=False)
loader_iter = iter(loader)
assert torch.equal(next(loader_iter), torch.tensor([0, 1]))
assert torch.equal(next(loader_iter), torch.tensor([2, 3]))
assert torch.equal(next(loader_iter), torch.tensor([4, 5]))
try:
    next(loader_iter)
    assert False, "应该 StopIteration"
except StopIteration:
    print("DataLoader 的 batch 耗尽后触发 StopIteration —— 和手写的 CountUpTo 是同一个机制")
```

**常见坑:**
1. 忘记在 `__next__` 里 `raise StopIteration`:`for` 循环会死循环(一直 `next()` 下去,永远等不到"结束"信号)。
2. `__iter__` 忘记 `return self`(默认返回 `None`):`for` 循环立刻报错,本机实测报错信息是 `TypeError: iter() returned non-iterator of type 'NoneType'`。
3. **iterable 不等于 iterator**:`list` 是 iterable(有 `__iter__`)但不是 iterator(没有 `__next__`)——直接对 list 调用 `next(a_list)` 会报 `TypeError: 'list' object is not an iterator`(本机实测)。每次对同一个 list 调 `iter()` 都会拿到一个**全新、独立**的迭代器对象,这就是为什么同一个 list 能被 `for` 循环 100 次而不会"用完"。这一点和第 2、3 节生成器的行为**正相反**——生成器的 `__iter__` 返回的是自己,用一次就耗尽了,提前在这里埋个伏笔。

---

## 2. `yield`:生成器基础——暂停与恢复执行

**是什么:** 函数体里只要出现 `yield` 关键字,这个函数就变成了"生成器函数"——调用它**不会**立即执行函数体,而是立刻返回一个生成器对象;每调用一次 `next()`,函数体就执行到下一个 `yield` 暂停,把值交出来,并且**完整记住当前所有局部变量和执行到了哪一行**,下次 `next()` 会从暂停的地方接着跑,不是从头开始。

**为什么课堂不教但很重要:** 第 1 节手写的 `CountUpTo` 类,本质上就是 Python 想让你少写的东西。AI 研究代码里,数据几十 GB 大是常态,不可能 `data = open(file).readlines()` 一次性读进内存再处理;标准做法是写一个生成器函数,一条一条产出,配合 PyTorch 的 `IterableDataset` 或 HuggingFace `datasets` 的 streaming 模式——这正是本节要讲的东西。

**从最笨的写法讲起:** C 程序员如果想写一个"每次调用都从上次停下的地方继续"的函数,语言不会帮你记"执行到哪一行"这件事,必须自己用一个 `struct` 把状态(当前进度、还剩多少)存起来,每次调用先读 struct、算一步、再把新状态存回 struct——这本质上是手写一个"手动挂起的状态机":

```c
// C:想要"下次调用从上次停下的地方继续",必须自己用 struct 存状态,
// 因为语言不会帮你记住"上次执行到了哪一行"
typedef struct {
    int current;
    int limit;
} Counter;

int counter_next(Counter *c) {
    if (c->current >= c->limit) {
        return -1;            // 用 -1 当"结束"信号,类似 StopIteration
    }
    return c->current++;
}
```

第 1 节的 `CountUpTo` 类就是这个 C 状态机思路翻译成 Python 的样子——`self.current` 就是那个 `struct` 字段。`yield` 版本则是让解释器自动帮你干这件事:

```python
# Python: yield 让解释器自动帮你存"执行到哪一行 + 所有局部变量"
# 你完全不用手写那个 struct,也不用手写 __iter__/__next__
def count_up_to(limit):
    current = 0
    while current < limit:
        yield current
        current += 1
```

对比第 1 节的 `CountUpTo`:一个 `__init__` 存状态、一个 `__iter__`、一个 `__next__`、手动判断结束条件,加起来将近 10 行样板代码;`yield` 版本只需要 4 行,**解释器自动帮你把 `CountUpTo` 那个类生成出来了**——这就是"语法糖"的准确含义:不是变魔术,而是省掉你手写的重复劳动。

**AI 研究代码里的真实例子:** 逐行读大文件,不一次性 `readlines()`,原理和上面的 `count_up_to` 完全一样,只是产出的是文本行而不是数字:

```python
def read_large_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()     # 读一行、处理一行、交出一行,从不持有"全部内容"
```

这跟 PyTorch `IterableDataset.__iter__` 里的典型写法一模一样;HuggingFace `datasets` 的 `streaming=True` 模式内部也是这种"一条一条产出,不一次性物化整个数据集"的思路。

**可运行例子:**
```python
def count_up_to(limit):
    current = 0
    while current < limit:
        yield current
        current += 1

# 1. 调用生成器函数不会立即执行函数体 —— 用打印语句证明"暂停/恢复"
def noisy_gen():
    print("  [生成器内部] 函数体开始执行")
    yield 1
    print("  [生成器内部] 恢复执行,准备产出第二个值")
    yield 2

g = noisy_gen()
print("已拿到生成器对象,类型是", type(g))   # <class 'generator'>,上面不应该有任何 [生成器内部] 输出
v1 = next(g)      # 这里才会看到 "函数体开始执行"
v2 = next(g)      # 这里才会看到 "恢复执行,准备产出第二个值"
assert v1 == 1 and v2 == 2

# 2. next()/for/list 等价验证
gen = count_up_to(3)
assert next(gen) == 0
assert next(gen) == 1
assert next(gen) == 2
try:
    next(gen)
    assert False, "耗尽后应该 StopIteration"
except StopIteration:
    print("count_up_to 耗尽后正确触发 StopIteration")

assert list(count_up_to(5)) == [0, 1, 2, 3, 4]

# 3. 生成器只能遍历一次!
gen2 = count_up_to(3)
first_pass = list(gen2)
second_pass = list(gen2)   # 同一个生成器对象,再遍历一次
print("生成器第一次遍历:", first_pass)    # [0, 1, 2]
print("生成器第二次遍历:", second_pass)   # []  —— 空的!
assert first_pass == [0, 1, 2]
assert second_pass == []
```

本机实测的真实输出(用 `PYTHONIOENCODING=utf-8` 跑,顺序完全符合"暂停—恢复"的预期):
```
已拿到生成器对象,类型是 <class 'generator'>
  [生成器内部] 函数体开始执行
  [生成器内部] 恢复执行,准备产出第二个值
count_up_to 耗尽后正确触发 StopIteration
生成器第一次遍历: [0, 1, 2]
生成器第二次遍历: []
```

注意"函数体开始执行"这行打印是在 `next(g)` 被调用之后才出现的,不是在 `noisy_gen()` 调用的那一刻——这就是"调用生成器函数不会立即执行"的直接证据。

再验证一次逐行读文件的例子(用临时文件模拟"大文件",本机实测跑通):
```python
import tempfile, os

fd, path = tempfile.mkstemp(suffix=".txt")
with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write("line-1\nline-2\nline-3\n")

try:
    lines = list(read_large_file(path))
    assert lines == ["line-1", "line-2", "line-3"]
finally:
    os.remove(path)
```

**常见坑:**
1. 调用生成器函数本身**不会**执行任何代码,只有 `next()`/`for`/`list()` 等真正"要东西"的操作才会触发执行——很多新手以为调用函数就跑完了,结果发现函数体里的 `print`/副作用没有按预期出现。
2. **生成器只能遍历一次**:上面已经实际跑出来了——遍历完(耗尽)之后再遍历,得到的是空结果 `[]`,不会报错也不会自动重新开始,这和 list 可以反复 `for` 完全不同(回顾第 1 节的伏笔:list 每次 `iter()` 都给你一个新迭代器,生成器的 `__iter__` 返回的是自己,一条道走到黑)。想要重新遍历,必须重新调用生成器函数,拿一个全新的生成器对象。
3. 生成器里的 `return value` 不是"把值返回给调用者"的普通含义,而是直接结束生成器并触发 `StopIteration`——这个 `value` 会被悄悄吞掉,`for`/`list()` 根本看不到它,只有下一节的 `yield from` 才能接住它。

---

## 3. 生成器表达式 vs 列表推导:惰性求值与内存

**是什么:** 把列表推导 `[expr for x in iterable]` 的方括号换成圆括号 `(expr for x in iterable)`,就从"立刻算出所有结果、放进一个真实存在的 list"变成了"生成器表达式"——不提前算,谁问它要值它才算一个,而且和上一节的 `yield` 生成器一样,只能用一次(惰性求值,lazy evaluation)。

**为什么课堂不教但很重要:** 课堂教了列表推导(写起来爽、结果直观),但很少有人对比过它和生成器表达式的本质区别:一个是"马上把所有结果攒成一个占内存的 list",一个是"一个随时能吐下一个值的对象,值从来没有被一次性放进过内存"。AI 训练代码里,数据集常常几十 GB,`[process(x) for x in huge_dataset]` 这种写法会先把全部处理结果放进内存再说,内存直接爆掉;生成器表达式/生成器函数不会,这正是为什么 PyTorch `IterableDataset`、HuggingFace `datasets` 的 `streaming=True` 模式内部都是"生成器"风格而不是"列表"风格。

**从最笨的写法讲起:** 这一节"最笨的写法"反而是列表推导——它急着把所有结果都算出来、放进内存,不管你到底用不用得到那么多。C 语言里最接近的类比是"提前建一张查找表" vs "每次要用就现算":

```c
// C:提前建一张完整的查找表,占用 N 个 int 的内存,不管你最后用不用得到
int table[1000000];
for (int i = 0; i < 1000000; i++) table[i] = i * i;

// C:按需现算,从不存下整张表,内存是常数级的
int compute(int i) { return i * i; }
```

Python 里,列表推导对应"提前建表",生成器表达式对应"按需现算":

```python
list_comp = [x ** 2 for x in range(1_000_000)]   # 提前建表:立刻算完 100 万个平方数,全部放进内存
gen_expr  = (x ** 2 for x in range(1_000_000))    # 按需现算:不算,只是记住"怎么算"
```

**AI 研究代码里的真实例子:** 数据预处理管道里,归一化/分词这类"边训练边算"的步骤,通常写成生成器风格,而不是先把整个数据集处理完再喂给模型:

```python
def preprocess_stream(raw_scores):
    for s in raw_scores:
        yield s / 100.0    # 归一化,边训练边算,永远不用等"全部预处理完"再开始
```

概念上等价于生成器表达式 `(s / 100.0 for s in raw_scores)`;HuggingFace `datasets` 的 streaming 模式、PyTorch `IterableDataset` 的 `__iter__`,内部思路都是这种"永远只惦记当前这一条"的风格。

**可运行例子:**

内存对比(`sys.getsizeof` 实测,数字是本机跑出来的真实结果,你自己跑可能有几个字节的出入,但量级不会变):
```python
import sys

N = 1_000_000
list_comp = [x ** 2 for x in range(N)]
gen_expr  = (x ** 2 for x in range(N))

list_size = sys.getsizeof(list_comp)
gen_size  = sys.getsizeof(gen_expr)
print(f"列表推导内存占用: {list_size:,} bytes")          # 本机实测: 8,448,728 bytes (约 8 MB)
print(f"生成器表达式内存占用: {gen_size:,} bytes")         # 本机实测: 200 bytes
assert list_size > gen_size * 1000     # 列表比生成器大几个数量级(本机实测大约 42,243 倍)

# range 扩大 10 倍,生成器表达式的体积应该几乎不变 —— 核心证据:它压根没把值存起来
gen_expr_10x = (x ** 2 for x in range(N * 10))
gen_size_10x = sys.getsizeof(gen_expr_10x)
print(f"range 扩大 10 倍后,生成器表达式内存占用: {gen_size_10x:,} bytes")   # 本机实测: 依然是 200 bytes
assert gen_size_10x == gen_size
```

惰性求值的时间证据(拿"第一个值"的耗时,跟数据总量无关):
```python
import time

def time_first_element(n):
    gen = (x ** 2 for x in range(n))
    t0 = time.perf_counter()
    first = next(gen)
    return time.perf_counter() - t0, first

t_small, _ = time_first_element(1_000)
t_large, _ = time_first_element(50_000_000)
print(f"从 1 千个元素的生成器表达式拿第一个值用时: {t_small:.6f}s")     # 本机实测: 约 0.000002s
print(f"从 5 千万个元素的生成器表达式拿第一个值用时: {t_large:.6f}s")   # 本机实测: 约 0.000001s,量级不变!
assert t_large < 0.01     # 不管 n 多大,取第一个值都是常数时间

# 对比:列表推导拿"第一个值"之前,必须先把全部算完
def time_first_element_listcomp(n):
    t0 = time.perf_counter()
    lst = [x ** 2 for x in range(n)]
    first = lst[0]
    return time.perf_counter() - t0, first

t_list_large, _ = time_first_element_listcomp(50_000_000)
print(f"列表推导构造 5 千万个元素后取第一个值用时: {t_list_large:.6f}s")   # 本机实测: 约 3.49s
assert t_list_large > t_large        # 列表推导慢了几百万倍,因为它被迫先把全部算完
```

"只能用一次"对生成器表达式同样成立(和上一节的 `yield` 生成器行为一致):
```python
gen = (x for x in range(3))
first_pass = list(gen)
second_pass = list(gen)
print("生成器表达式第一次遍历:", first_pass)    # [0, 1, 2]
print("生成器表达式第二次遍历:", second_pass)   # []
assert first_pass == [0, 1, 2]
assert second_pass == []

# 列表推导反过来:可以反复用,因为它就是一个普通的、内容已经算好的 list
lst = [x for x in range(3)]
assert list(lst) == [0, 1, 2]
assert list(lst) == [0, 1, 2]    # 反复取,list 不会被"耗尽"
```

**常见坑:**
1. 生成器表达式和 `yield` 生成器一样,**只能遍历一次**;列表推导得到的是真正的 list,可以反复遍历——上面的代码已经实际跑出了"第二次遍历是空的"这个结果。
2. `sys.getsizeof` 只测"外壳"对象本身的大小:对 list 来说这基本等于全部内容占用的内存(因为内容确实都已经在内存里算好了);对生成器来说这只是生成器对象本身(一个很小的状态机),**不代表未来会产出的所有值占了多少内存**——因为那些值根本还没被算出来,谈不上占内存。
3. 想同时享受"惰性、不占内存"和"能反复遍历"两个好处,list 和生成器表达式都做不到,只能二选一(或者自己写一个类,每次 `__iter__` 被调用时都返回一个全新的生成器——这也是 PyTorch `IterableDataset` 常见的写法,这里先点一下,不展开)。

---

## 4. `yield from`:委托生成器

**是什么:** `yield from another_generator` 是"把内层生成器产出的每一个值,原样一个个转交给外层调用者"的语法糖,免去手写 `for item in another_generator: yield item` 这层嵌套循环。

**为什么课堂不教但很重要:** 数据集常常按文件/来源切成多个 shard(比如预训练语料按天、按来源切成几百个文件),真实训练代码里经常要写一个"依次把每个文件的生成器串起来"的函数——这正是 `yield from` 的主场。更深一层看,`yield from` 不只是转发值,连内层生成器的 `return` 返回值也会一并转发出来(上一节埋的伏笔在这里揭晓);这也是 `async`/`await` 协程(第 4 批会讲)能够成立的底层机制之一。对研究代码来说,最常打交道的场景还是"合并多个数据来源"。

**从最笨的写法讲起:** 不用 `yield from`,手动写嵌套循环转发:

```python
def read_large_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()

# 最笨的写法:手动写一层嵌套 for,自己把内层生成器的每个值转交出去
def read_all_shards_manual(paths):
    for path in paths:
        for line in read_large_file(path):   # 手动内层 for 循环
            yield line                        # 手动一个个转交给外层
```

用 `yield from` 简化:

```python
def read_all_shards(paths):
    for path in paths:
        yield from read_large_file(path)      # 一行顶替"内层 for + yield"两行
```

两者行为完全一致,后者只是少了一层手动转发的样板代码。

**AI 研究代码里的真实例子:** 多 shard 训练语料的合并读取——有多个各自能产出样本的生成器(每个对应一个文件/一个数据源),`yield from` 把它们逐个委托、拼接成一条统一的样本流,上层训练循环完全不需要关心底层到底有几个文件。HuggingFace `datasets` 里 `interleave_datasets`/多分片拼接,概念上就是这类"委托转发"思路的封装版。

**可运行例子:**

用临时文件模拟多个 shard,验证手动版和 `yield from` 版结果完全一致:
```python
import tempfile, os

paths = []
contents = [["a1", "a2"], ["b1", "b2", "b3"]]
for lines in contents:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    paths.append(path)

try:
    manual = list(read_all_shards_manual(paths))
    delegated = list(read_all_shards(paths))
    assert manual == delegated == ["a1", "a2", "b1", "b2", "b3"]
finally:
    for p in paths:
        os.remove(p)
```

验证 `yield from` 能拿到内层生成器的 `return` 值(第 2 节埋的伏笔,在这里揭晓):
```python
def inner():
    yield 1
    yield 2
    return "inner-done"     # 生成器 return 的值,普通 for/list 根本看不到

def outer():
    result = yield from inner()   # yield from 表达式本身的值 = inner() 的 return 值
    yield f"inner returned: {result}"

out = list(outer())
assert out == [1, 2, "inner returned: inner-done"]

# 对比:手动 for 循环转发,拿不到 inner() 的 return 值
def outer_manual():
    for v in inner():
        yield v

assert list(outer_manual()) == [1, 2]    # 少了 "inner returned: ..." 这一条,return 值被吞掉了
```

验证 `yield from` 后面接的是"可迭代对象",不是"要产出的单个值":
```python
def yields_from_list():
    yield from [1, 2, 3]     # 等价于 yield 1; yield 2; yield 3

def yields_a_list_directly():
    yield [1, 2, 3]          # 把整个 list 当一个值产出

assert list(yields_from_list()) == [1, 2, 3]
assert list(yields_a_list_directly()) == [[1, 2, 3]]
```

**常见坑:**
1. `yield from` 后面必须接一个可迭代对象(另一个生成器、list、range 都行),不是"把值列出来"——`yield from [1, 2, 3]` 等价于依次 `yield 1; yield 2; yield 3`,而不是把整个 `[1, 2, 3]` 当成一个值产出(那是普通 `yield [1, 2, 3]` 干的事)。上面的例子已经实测验证了这两者结果完全不同:`[1, 2, 3]` vs `[[1, 2, 3]]`。
2. 容易忘记 `yield from` 表达式本身是有值的(等于内层生成器 `return` 的值),把它当"纯转发、没有返回值"的语句用——遇到需要拿"子生成器总结果"的场景时会漏掉这个信息。
3. 嵌套委托层数一多,出错时的 traceback 会经过好几层 `yield from`,比手写嵌套 `for` 循环更难一眼看出问题出在哪一层——委托方便,但排错成本要认识到。

---

## 小结

| 节 | 核心概念 | 解决的问题 |
|---|---|---|
| 1. 迭代器协议 | `__iter__` + `__next__` + `StopIteration` | 揭示 `for` 循环的本质:反复调用 `__next__` 直到收到"结束"信号 |
| 2. `yield` 生成器 | 暂停与恢复执行,解释器自动存状态 | 用语法糖消掉第 1 节手写迭代器类的全部样板代码 |
| 3. 生成器表达式 vs 列表推导 | 惰性求值(lazy evaluation) | 大数据集/流式处理不会把所有结果一次性塞进内存 |
| 4. `yield from` | 委托生成器 | 简化"把多个生成器的产出拼成一条流"的写法,还能转发 `return` 值 |

下一批:[03-oop-advanced.md](03-oop-advanced.md)

---

*更新:2026-07-07*
