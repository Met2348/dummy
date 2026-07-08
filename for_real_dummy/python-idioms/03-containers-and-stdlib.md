# 03 · 容器与标准库惯用法(Containers & Stdlib Idioms)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 7 个知识点:`collections.Counter`、`collections.defaultdict`、`collections.namedtuple`、`collections.deque`、`dict.get`/`dict.setdefault`、EAFP vs LBYL 哲学、`sorted`/`.sort()` 的 `key=` 参数。它们都是"标准库自带、不用装第三方包"的容器工具和惯用法——课堂教 `dict`/`list`/`tuple` 只教到"能存能取"为止,但 AI 研究代码里统计词频、按 topic 分组、调度请求队列、按多个字段排序,几乎每个模块都在用这批写法,不认识就只能用更笨的写法反复造轮子。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9)下实际跑通验证,不是凭空写的——其中几个"AI 研究代码里的真实例子"直接引用了仓库 `learning/` 目录下的真实模块(mini-vLLM 推理引擎的请求调度、BPE tokenizer 的 pair 频次统计、A2C 强化学习训练循环的滑动窗口、多 agent 系统的消息总线、agent 成本-准确率评估等),标注的文件路径和行号在下笔前都重新读过一遍确认没有过期。

---

## 1. `collections.Counter`

**是什么:** `dict` 的子类,专门用来计数——`Counter(iterable)` 自动统计每个元素出现了多少次,访问不存在的 key 返回 `0` 而不报错。

**为什么课堂不教但很重要:** 课堂教的计数方式是手写一个 `dict`,配合 `d.get(key, 0) + 1` 或者 `if key in d: d[key] += 1 else: d[key] = 1`。AI 研究代码里统计 token 频次(训练 tokenizer 时统计相邻字节对出现次数)、统计类别分布、多 agent 投票时统计"哪个答案出现最多次",几乎都是"给一堆东西计数"这个模式的变体——`Counter` 就是标准库专门为这个模式做的工具,比手写 `dict` 更短,也更不容易漏掉"第一次出现"这个分支。

**从最笨的写法讲起:**
```python
import collections

words = ["cat", "dog", "cat", "bird", "dog", "cat"]

# 笨办法:手写计数,每次都要判断 key 是否已经出现过
counts_dumb = {}
for w in words:
    counts_dumb[w] = counts_dumb.get(w, 0) + 1

# 正式写法:Counter 直接接收一个可迭代对象,自动完成计数
counts_counter = collections.Counter(words)

assert counts_dumb == dict(counts_counter)
assert counts_counter["cat"] == 3
assert counts_counter["dog"] == 2
assert counts_counter["bird"] == 1
print(counts_counter)   # Counter({'cat': 3, 'dog': 2, 'bird': 1})
```
两种写法结果完全一样,但 `Counter` 省掉了手写的 `if`/`get(w, 0)` 判断,而且访问不存在的 key 天然返回 `0`(不会像普通 `dict` 那样抛 `KeyError`)——这一点"常见坑"里会讲一个容易被忽略的细节。

**AI 研究代码里的真实例子:** 手写 BPE tokenizer 训练时,要反复统计"相邻 token pair 出现了多少次"来决定下一步合并哪一对,这是 BPE 算法最核心的一步:
```python
# learning/data-curation/src/bpe_trainer.py 第 18-22 行(原样抄录;补上了文件顶部第 14 行的 import collections 以便独立运行)
import collections

def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = collections.Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

ids = [1, 2, 3, 1, 2, 1, 2]
pair_counts = _get_pair_counts(ids)
assert pair_counts[(1, 2)] == 3
assert pair_counts[(2, 3)] == 1
assert pair_counts[(3, 1)] == 1
print(dict(pair_counts))
```
关键是 `counts[(a, b)] += 1` 这一行:`Counter` 对不存在的 pair 返回 `0`,`0 + 1 = 1` 直接写回去,不需要像手写 `dict` 那样先判断"这个 pair 是不是第一次出现"。配合第 20 行的 `zip(ids, ids[1:])`(自己和"错开一位的自己" `zip`,取相邻元素对的经典写法),两行就完成了"统计所有相邻 pair 频次"。

**可运行例子:** `most_common()` 取出现次数最多的元素,以及 `Counter` 之间的算术运算:
```python
import collections

letters = collections.Counter("abracadabra")
print(letters)   # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

top1 = letters.most_common(1)
assert top1 == [("a", 5)]   # 'a' 出现 5 次,唯一最高,断言没有歧义

c1 = collections.Counter(a=3, b=1)
c2 = collections.Counter(a=1, b=2)
assert (c1 + c2) == collections.Counter(a=4, b=3)   # 对应 key 相加
assert (c1 - c2) == collections.Counter(a=2)        # 相减,结果 <= 0 的 key 被丢弃(b: 1-2=-1 被丢弃)
assert (c1 & c2) == collections.Counter(a=1, b=1)   # 取较小值,类似"交集"
assert (c1 | c2) == collections.Counter(a=3, b=2)   # 取较大值,类似"并集"
print(c1 + c2, c1 - c2, c1 & c2, c1 | c2)
```

**常见坑:**

1. **`most_common()` 对并列计数的元素,顺序不是按元素值排序的,而是取决于插入序(先出现的排前面)。** 这一点在写 assert 验证时特别容易踩坑——两个元素计数相同时,没法保证 `most_common()` 一定按某种"直觉"顺序返回:
```python
import collections

c = collections.Counter(["x", "y", "x", "y", "z"])   # x:2, y:2, z:1 —— x 和 y 并列第一
mc = c.most_common()
print(mc)   # [('x', 2), ('y', 2), ('z', 1)]:x 先出现,排在并列的前面

# 反过来,如果构造顺序变了(y 先出现),并列的顺序也跟着变
c2 = collections.Counter()
c2.update(["y", "x", "y", "x", "z"])
print(c2.most_common())   # [('y', 2), ('x', 2), ('z', 1)]:这次 y 排在前面

# 安全的断言方式:只断言没有歧义的部分——排名最后的 z 没有并列,可以放心断言;
# 前两名只断言"计数值都是 2"和"是 x/y 这个集合",不断言具体谁排在前面
assert mc[-1] == ("z", 1)
assert mc[0][1] == 2 and mc[1][1] == 2
assert {mc[0][0], mc[1][0]} == {"x", "y"}
```
`most_common()` 内部按计数值降序做稳定排序,计数相同的元素维持它们在 `Counter` 里的原始顺序(第一次出现的顺序)——**这不是"按字母序"或任何和元素值本身有关的规则**,写测试时不要凭直觉假设并列部分的顺序。

2. **访问不存在的 key 返回 `0`,但不会像 `defaultdict` 那样把这个 key 写进去。** `counter["missing"]` 只是查询,没有副作用;下一节的 `defaultdict` 行为刚好相反,容易混淆,下面会专门对比。

---

## 2. `collections.defaultdict`

**是什么:** `dict` 的子类,对不存在的 key 提供一个"工厂函数"自动生成默认值,不用手写 `if key not in d: d[key] = ...`。

**为什么课堂不教但很重要:** "按某个字段分组"是 AI 研究代码里极常见的模式——按 topic 收集订阅者、按类别收集样本、按 SM(streaming multiprocessor)编号收集 trace——每次分组都要先判断"这个组是不是第一次出现",手写很啰嗦。`defaultdict` 把这个判断变成语言层面的默认行为。

**从最笨的写法讲起:**
```python
import collections

pairs = [("search_done", "writer"), ("search_done", "critic"), ("error", "logger")]

# 笨办法:每次 append 前都要先判断这个 key 是不是第一次出现
topic_subs_dumb: dict[str, list[str]] = {}
for topic, agent in pairs:
    if topic not in topic_subs_dumb:
        topic_subs_dumb[topic] = []
    topic_subs_dumb[topic].append(agent)

# 正式写法:defaultdict(list) 自动帮你把"第一次出现"的判断做掉
topic_subs = collections.defaultdict(list)
for topic, agent in pairs:
    topic_subs[topic].append(agent)   # 不管 topic 是不是第一次出现,直接 append

assert dict(topic_subs) == topic_subs_dumb == {"search_done": ["writer", "critic"], "error": ["logger"]}
print(dict(topic_subs))
```
`defaultdict(list)` 的参数 `list` 不是"一个默认值",而是"一个不带参数就能调用、产出默认值的工厂"——每次访问不存在的 key,`defaultdict` 就调用一次 `list()` 拿到一个全新的空列表塞进去。这和 `dataclass` 的 `field(default_factory=list)` 是同一个心智模型(可变默认值必须用工厂函数现造,不能写死一个共享的字面量)。

**AI 研究代码里的真实例子:** 仓库讲义里 pub-sub 消息总线的教学实现,按 topic 把订阅者分组:
```python
# learning/multi-agent-orchestration/lectures/10-agent-communication.md 第 21-30 行(原样抄录;补上了 from collections import defaultdict 以便独立运行,讲义正文里这个 import 在别处)
from collections import defaultdict

class MessageBus:
    def __init__(self):
        self.subs = defaultdict(list)

    def subscribe(self, topic, callback):
        self.subs[topic].append(callback)

    def publish(self, topic, payload):
        for cb in self.subs[topic]:
            cb(payload)
```
用 `defaultdict(list)` 换来的好处是:`publish` 一个从来没人订阅过的 topic 时,`self.subs[topic]` 直接返回空列表,`for` 循环零次迭代,不需要专门判断"这个 topic 有没有人订阅"。**但值得注意的是**,同一个仓库 `learning/multi-agent-orchestration/src/message_bus.py` 里生产代码版本的 `MessageBus` 反而选择了普通 `dict` + `setdefault`,不是 `defaultdict`——这个对比留到第 5 节详细讲,两种写法解决的是同一个问题,取舍点在于"要不要让每次查询都可能带有插入副作用"(见下面常见坑 1)。

**可运行例子:**
```python
from collections import defaultdict

bus_subs = defaultdict(list)
received = []
bus_subs["search_done"].append(lambda p: received.append(p))
bus_subs["search_done"].append(lambda p: received.append(p))

for cb in bus_subs["search_done"]:
    cb({"r": 1})
assert received == [{"r": 1}, {"r": 1}]

for cb in bus_subs["nobody_home"]:   # 从没订阅过的 topic,不报错,循环体直接不执行
    cb({"x": 1})
assert bus_subs["nobody_home"] == []
print(dict(bus_subs))
```

**常见坑:**

1. **只是"读一下"也会触发插入,这个副作用容易被忽视。** 上面例子里 `bus_subs["nobody_home"]` 那次访问,虽然只是为了拿到空列表去 `for` 循环,但已经把 `"nobody_home"` 这个 key 真的写进了字典里:
```python
from collections import defaultdict

d = defaultdict(list)
assert len(d) == 0
_ = d["never_written"]        # 只是读,没有 append 任何东西
assert len(d) == 1            # key 已经被插入了!
assert d["never_written"] == []

# 如果只是想"查一下有没有",不想意外插入,应该用 in,而不是直接下标访问
d2 = defaultdict(list)
if "probe" in d2:
    pass
assert len(d2) == 0            # in 判断不会触发插入
```
如果代码里频繁"顺手" `d[key]` 只为判断真假,`defaultdict` 会被悄悄塞进大量空值 key,后面遍历 `d.keys()` 或者 `len(d)` 时结果会比预期多。

2. **工厂参数必须是"可调用对象",不能直接传一个现成的值。** `defaultdict([])` 是常见的手滑写法:
```python
from collections import defaultdict

try:
    defaultdict([])
except TypeError as e:
    print(e)   # first argument must be callable or None

d3 = defaultdict(lambda: "N/A")   # 想要"非空容器之外的默认值",用 lambda 包一层
assert d3["missing"] == "N/A"
```

---

## 3. `collections.namedtuple`

**是什么:** 一个"可以按字段名访问、但依然是不可变 tuple"的轻量级类型——比裸 tuple 可读,比手写 class 省样板代码。

**为什么课堂不教但很重要:** 课堂教的 tuple 只能按下标 `t[0]`/`t[1]` 访问,函数返回多个值时调用方经常要靠注释才能搞清楚"第 0 个是什么、第 1 个是什么"。研究代码里模型的中间输出、一条评估记录、一个坐标/区间,经常想要"返回值有名字,但又不想为了这点小事写一整个 class"——`namedtuple` 就是为这个场景做的。

**从最笨的写法讲起:**
```python
import collections

# 笨办法 1:裸 tuple,可读性差——不看函数定义根本不知道 [0] 是 x 还是 y
point_tuple = (3.0, 4.0)
dist_dumb = (point_tuple[0] ** 2 + point_tuple[1] ** 2) ** 0.5
assert dist_dumb == 5.0

# 笨办法 2:普通 class,可读性够了,但 __init__/__repr__/__eq__ 全部要自己写
class PointClassDumb:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"PointClassDumb(x={self.x}, y={self.y})"
    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

assert PointClassDumb(3.0, 4.0) == PointClassDumb(3.0, 4.0)   # 只有手写了 __eq__ 才能这样比

# 正式写法:namedtuple 一行定义,__repr__/__eq__/解包/下标访问全部免费获得
Point = collections.namedtuple("Point", ["x", "y"])
p = Point(3.0, 4.0)

assert p.x == 3.0 and p.y == 4.0            # 按名字访问,可读
assert p[0] == 3.0 and p[1] == 4.0          # 仍然支持下标(它本质是 tuple)
assert p == Point(x=3.0, y=4.0)             # __eq__ 免费获得
assert repr(p) == "Point(x=3.0, y=4.0)"     # __repr__ 免费获得
x, y = p                                     # tuple 解包免费获得
assert (x, y) == (3.0, 4.0)
print(p)
```

**AI 研究代码里的真实例子:** 老实交代——在 `learning/` 目录(排除 `official/repos/` 下 vendor 进来的第三方仓库)里搜了一遍 `namedtuple`,没有找到博士学长自己写的用例;这一类场景他更常用 `dataclass`(比如 `learning/inference-engine-core/src/mini_vllm.py` 里的 `@dataclass class MiniEngine`,下一节会再引用到)。下面这个例子是**示例性质**,不是仓库原文,用来展示 `namedtuple` 在"轻量级、创建后不需要再改"的记录场景下的典型写法:
```python
import collections

EvalRecord = collections.namedtuple("EvalRecord", ["task", "accuracy", "cost_usd"])

records = [
    EvalRecord("mmlu", 0.82, 0.015),
    EvalRecord("gsm8k", 0.91, 0.020),
]
best = max(records, key=lambda r: r.accuracy)
assert best.task == "gsm8k"
assert best.accuracy == 0.91
print(best)
```

**可运行例子:** 不可变性,以及 `_replace()`/`_asdict()` 这两个配套方法("改一份拷贝""转成 dict"):
```python
import collections

Point = collections.namedtuple("Point", ["x", "y"])
p = Point(3.0, 4.0)

try:
    p.x = 10.0             # namedtuple 继承自 tuple,不允许修改字段
except AttributeError as e:
    print(e)                # can't set attribute

p2 = p._replace(x=10.0)     # 想"改"必须用 _replace() 生成一个新实例
assert p2 == Point(x=10.0, y=4.0)
assert p == Point(x=3.0, y=4.0)     # 原实例不受影响

assert p._asdict() == {"x": 3.0, "y": 4.0}   # 转成普通 dict,方便序列化成 JSON

# 3.6+ 还可以用带类型注解的 typing.NamedTuple 写法,效果等价,但能写类型和默认值
from typing import NamedTuple

class PointTyped(NamedTuple):
    x: float
    y: float
    label: str = "origin"    # 简单的默认值语法糖,不需要 default_factory

pt = PointTyped(1.0, 2.0)
assert pt.label == "origin"
assert pt == (1.0, 2.0, "origin")    # 依然是 tuple,可以直接和裸 tuple 比较
print(pt)
```

**常见坑:**

1. **不可变是"卖点"也是"限制"——`namedtuple` 不适合需要频繁原地修改字段的场景。** 每次"修改"都要 `_replace()` 生成一份新实例,如果字段很多、改动很频繁,不如直接用 `dataclass`。`dataclass` 的可变默认值陷阱和正确写法(`field(default_factory=...)`)已经在 [python-advanced/03-oop-advanced.md 第 620-660 行](../python-advanced/03-oop-advanced.md)详细讲过,这里不重复——**取舍原则**:字段固定、创建后不会再变、想要更省内存,选 `namedtuple`;字段需要默认值语法糖、需要可变、需要写方法,选 `dataclass`。

2. **为什么 `namedtuple` 比普通 class 更省内存:`__slots__` 的原理。** 普通 class 的每个实例默认带一个 `__dict__` 存自己的字段,这个 `__dict__` 本身也占内存;而 `namedtuple` 内部声明了 `__slots__ = ()`,实例不再有 `__dict__`,字段直接存在 tuple 的定长槽位里:
```python
import collections
import sys

class PlainNoSlots:
    def __init__(self, x, y):
        self.x = x
        self.y = y

plain = PlainNoSlots(1, 2)
Point = collections.namedtuple("Point", ["x", "y"])
p = Point(1, 2)

assert hasattr(plain, "__dict__")     # 普通实例有 __dict__
assert not hasattr(p, "__dict__")     # namedtuple 实例没有 __dict__

size_plain_total = sys.getsizeof(plain) + sys.getsizeof(plain.__dict__)
size_namedtuple = sys.getsizeof(p)
assert size_namedtuple < size_plain_total    # 本机实测:namedtuple 明显更省
print("plain(含__dict__):", size_plain_total, " namedtuple:", size_namedtuple)
```
自己写 class 时用 `__slots__ = ("x", "y")` 能达到同样的省内存效果,但要手写;`namedtuple`/`typing.NamedTuple` 相当于"自动帮你做了 `__slots__` 优化"的现成方案。具体字节数因 Python 版本/平台而异,不用记数字,只需要知道"更省"这个结论。

---

## 4. `collections.deque`

**是什么:** 双端队列(double-ended queue)——两端的插入/弹出都是 O(1),不像 `list` 那样只有"尾部"操作快、"头部"操作慢。

**为什么课堂不教但很重要:** 课堂教的队列大多直接拿 `list` 实现,用 `pop(0)` 弹出头部元素——数据量小的时候看不出问题,但 `list.pop(0)` 每弹出一个元素,后面所有元素都要整体往前挪一位,是 O(n) 操作,连续弹出 n 次就是 O(n²)。AI infra 代码里的请求调度队列(每一轮迭代都要从队头取请求)、以及"只保留最近 N 个"的滑动窗口(训练循环里记录最近 100 个 episode 的回报),都要求头部操作和 `append` 一样快——这正是 `deque` 存在的原因。

**从最笨的写法讲起:**
```python
from collections import deque

# 笨办法:用 list 模拟队列
queue_dumb = []
queue_dumb.append("req1")
queue_dumb.append("req2")
queue_dumb.append("req3")
first_out = queue_dumb.pop(0)     # 弹出队头:其余元素要整体往前搬一位,O(n)
assert first_out == "req1"
assert queue_dumb == ["req2", "req3"]

# 正式写法:deque,两端都是 O(1)
queue = deque()
queue.append("req1")
queue.append("req2")
queue.append("req3")
first_out2 = queue.popleft()      # 弹出队头:O(1),不需要搬移剩余元素
assert first_out2 == "req1"
assert list(queue) == ["req2", "req3"]
```
功能上两者等价,区别在复杂度。实测一下差距有多大(队列长度 2 万,逐个弹到空):
```python
import time
from collections import deque

n = 20_000

def bench_list_pop0():
    q = list(range(n))
    while q:
        q.pop(0)

def bench_deque_popleft():
    q = deque(range(n))
    while q:
        q.popleft()

def timeit_best(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

t_list = timeit_best(bench_list_pop0)
t_deque = timeit_best(bench_deque_popleft)
print(f"list.pop(0): {t_list:.4f}s   deque.popleft(): {t_deque:.4f}s")
assert t_deque < t_list * 0.5   # 本机实测:deque 数量级上明显更快,阈值留足余量避免抖动
```

**AI 研究代码里的真实例子:** 有两种用法都很常见,心智模型不一样,分开看。

**a. 当"无界 FIFO 调度队列"用。** mini-vLLM 推理引擎用 `deque` 存放"还没被 admit 进 GPU batch 的请求"。下面这两段是从源文件里摘出的非连续行,中间用 `...` 省略号标记省略的部分,只是为了对照真实代码长什么样,不是能独立运行的完整片段(它们各自的独立可运行版本在紧接着的代码块里):
```text
# learning/inference-engine-core/src/mini_vllm.py 第 15 行 + 第 37 行(原样抄录,... 表示中间省略)
from collections import deque
...
    pending: Deque[Request] = field(default_factory=deque)
```
同样的模式在同目录 `continuous_batching.py` 第 15/28 行声明、第 54 行 `self.running.append(self.pending.popleft())` 实际调度,以及 `scheduling_policies.py` 第 38-39 行(FCFS 策略核心就是一行 `pending.popleft()`)都出现过。`paged_kv.py` 那处更典型——用 `deque` 当"空闲显存块 ID 的池子":
```text
# learning/inference-engine-core/src/paged_kv.py 第 16 行 + 第 36 行、45 行、55 行(原样抄录,... 表示中间省略)
from collections import deque
...
        self.free_ids: deque[int] = deque(range(self.n_blocks))
...
    def alloc_block(self) -> int:
        ...
        blk = self.free_ids.popleft()   # 分配:从左边取
        ...
    def free_block(self, blk: int) -> None:
        ...
        self.free_ids.append(blk)       # 归还:放回右边
```
复现这两处的调度场景(独立可运行,逻辑和上面引用的真实代码一致):
```python
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

@dataclass
class Request:
    rid: int

@dataclass
class MiniEngineLike:
    pending: Deque[Request] = field(default_factory=deque)
    running: List[Request] = field(default_factory=list)
    max_running: int = 2

    def add(self, req: Request) -> None:
        self.pending.append(req)

    def step_admit(self) -> None:
        while self.pending and len(self.running) < self.max_running:
            self.running.append(self.pending.popleft())   # FCFS:从队头取,先来的先服务

engine = MiniEngineLike()
for i in range(4):
    engine.add(Request(rid=i))
engine.step_admit()
assert [r.rid for r in engine.running] == [0, 1]     # 先加入的先被 admit
assert [r.rid for r in engine.pending] == [2, 3]

# 空闲块 ID 池:分配从左边取,归还放回右边
free_ids: deque[int] = deque(range(4))
blk = free_ids.popleft()
assert blk == 0 and list(free_ids) == [1, 2, 3]
free_ids.append(blk)
assert list(free_ids) == [1, 2, 3, 0]
```
这里的 `deque` 是"无界"的——请求可以一直 `append` 进去,只要显存/调度策略允许。

**b. 当"有界滑动窗口"用(`maxlen` 参数)。** A2C 强化学习训练循环里,用 `deque(maxlen=100)` 记录最近 100 个 episode 的回报,算滑动平均——同样只摘录两行做对照,不是独立可运行片段:
```text
# learning/rl-foundations/src/a2c_minimal.py 第 18 行 + 第 71 行(原样抄录,... 表示中间省略)
from collections import deque
...
    ep_returns_window: deque = deque(maxlen=100)
```
复现(独立可运行,逻辑和上面引用的真实代码一致):
```python
from collections import deque

ep_returns_window: deque = deque(maxlen=5)
for ep_return in [10, 20, 30, 40, 50, 60, 70]:
    ep_returns_window.append(ep_return)

assert len(ep_returns_window) == 5                       # 永远不超过 5
assert list(ep_returns_window) == [30, 40, 50, 60, 70]   # 自动丢弃了最老的 10, 20
mean_recent = sum(ep_returns_window) / len(ep_returns_window)
assert mean_recent == 50.0
print(list(ep_returns_window), mean_recent)
```
**这两种用法是完全不同的心智模型**,虽然底层都是 `deque`:普通 `list` 也能实现 FIFO 队列(用法 a),但 `deque` 头部 O(1) 的插入/弹出才是它被选中的原因;`maxlen`(用法 b)则是"固定容量的环形缓冲区",超出容量时自动丢弃另一端的旧数据——这个自动丢弃行为,普通 `list` 需要手动写 `if len(lst) > maxlen: lst.pop(0)` 才能模拟,`deque(maxlen=...)` 直接把这段逻辑内置了。

**可运行例子:** `deque` 还有一个上面两种用法都没展示的能力——`rotate(n)` 整体轮转,常用来实现 round-robin(轮询)调度:
```python
from collections import deque

# rotate(n): n<0 往左转(头部元素转到尾部),常用来实现 round-robin 轮询调度
workers = deque(["gpu0", "gpu1", "gpu2"])

assignments = []
tasks = ["taskA", "taskB", "taskC", "taskD", "taskE"]
for task in tasks:
    worker = workers[0]                 # 永远把当前任务交给"排在最前面"的 worker
    assignments.append((task, worker))
    workers.rotate(-1)                  # 轮转一位:该 worker 转到队尾,下一个 worker 顶上

assert assignments == [
    ("taskA", "gpu0"),
    ("taskB", "gpu1"),
    ("taskC", "gpu2"),
    ("taskD", "gpu0"),   # 转了一圈,回到 gpu0
    ("taskE", "gpu1"),
]
print(assignments)
```

**常见坑:**

1. **随机下标访问是 O(n),不是 O(1)。** `deque` 内部是分块的双向链表结构,`d[i]` 要从最近的一端开始数 `i` 步——这一点和 `list` 的连续内存数组正好相反:
```python
import time
from collections import deque

big_deque = deque(range(100_000))
big_list = list(range(100_000))

def access_middle_deque():
    for _ in range(2000):
        _ = big_deque[50_000]

def access_middle_list():
    for _ in range(2000):
        _ = big_list[50_000]

def timeit_best(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

t_deque_idx = timeit_best(access_middle_deque)
t_list_idx = timeit_best(access_middle_list)
print(f"deque[i]: {t_deque_idx:.4f}s   list[i]: {t_list_idx:.4f}s")
assert t_deque_idx > t_list_idx * 3   # deque 中间下标访问明显慢于 list
```
如果需要频繁随机下标访问,`deque` 不是合适的数据结构——它只在"两端操作"这个场景下才比 `list` 有优势。

2. **`maxlen` 满了之后,`append`/`appendleft` 会静默丢弃另一端的元素,不报错也不警告。**
```python
from collections import deque

d = deque([1, 2, 3], maxlen=3)
d.append(4)          # 从右边加入 4,自动挤掉最左边的 1
assert list(d) == [2, 3, 4]
d.appendleft(0)       # 从左边加入 0,自动挤掉最右边的 4
assert list(d) == [0, 2, 3]
```
用 `maxlen` 存"必须不能丢"的数据(比如还没处理完的请求)会导致数据静默丢失且没有任何异常提示——`maxlen` 只应该用在"本来就该丢旧数据"的场景(滑动窗口、最近 N 条日志),不能和"无界调度队列"(用法 a)混用。

---

## 5. `dict.get` / `dict.setdefault` vs 手写 if-else

**是什么:** `d.get(key, default)` 是"查,查不到就给默认值,不报错"的一行写法;`d.setdefault(key, default)` 是"查,查不到就把默认值写进去,再返回"的一行写法。

**为什么课堂不教但很重要:** 课堂教的字典访问方式是 `d[key]`(查不到直接 `KeyError` 崩溃)或者 `if key in d: ... else: ...`。后者要做两次查找(`in` 一次,取值/赋值再一次),AI 研究代码里到处是"读配置给默认值""按字段分组收集"这两个模式,`get`/`setdefault` 把两次查找合并成一次,顺带把代码从 4 行压成 1 行。

**从最笨的写法讲起:**
```python
config = {"lr": 0.01, "batch_size": 32}

# 笨办法 1:读取"可能不存在的 key",手写 if-else(两次查找:先 in,再取值)
def get_dumb(d, key, default):
    if key in d:
        return d[key]
    else:
        return default

assert get_dumb(config, "momentum", 0.9) == 0.9    # 不存在,给默认值
assert get_dumb(config, "lr", 0.9) == 0.01          # 存在,返回原值

# 正式写法:dict.get(key, default),一行等价
assert config.get("momentum", 0.9) == 0.9
assert config.get("lr", 0.9) == 0.01
assert config.get("missing_key") is None            # 不给 default 时,缺失返回 None,不报错

# 笨办法 2:"分组收集"场景,手写 if key not in d: d[key] = []
items = [("cat", "a"), ("dog", "b"), ("cat", "c")]
tags_dumb = {}
for tag, name in items:
    if tag not in tags_dumb:
        tags_dumb[tag] = []
    tags_dumb[tag].append(name)

# 正式写法:dict.setdefault(key, default)
tags = {}
for tag, name in items:
    tags.setdefault(tag, []).append(name)

assert tags == tags_dumb == {"cat": ["a", "c"], "dog": ["b"]}
print(tags)
```

**AI 研究代码里的真实例子:** 第 2 节提到,`learning/multi-agent-orchestration/src/message_bus.py` 里生产代码版本的 `MessageBus`,和讲义里 `defaultdict(list)` 的教学版本不同,选择了普通 `dict` + `setdefault`/`get` 组合(下面摘录三行做对照,`...` 表示中间省略,不是独立可运行片段——独立可运行版本在紧接着的"可运行例子"里):
```text
# learning/multi-agent-orchestration/src/message_bus.py 第 16、20、26 行(原样抄录,... 表示中间省略)
self.subs: dict[str, list[tuple[str, Callable]]] = {}
...
    def subscribe(self, topic: str, agent_name: str, callback: Callable[[BusMessage], None]) -> None:
        self.subs.setdefault(topic, []).append((agent_name, callback))
...
        for name, cb in self.subs.get(topic, []):
```
`subscribe` 用 `setdefault(topic, [])` 建立分组(等价于第 2 节 `defaultdict(list)` 做的事,但显式一些);`publish` 用 `self.subs.get(topic, [])` 查询,topic 没人订阅时给一个空列表,不需要先判断"这个 topic 存不存在"。**两种写法(`defaultdict` vs `setdefault`+`get`)解决的是同一个问题**,选 `setdefault`+`get` 的好处是:类型注解可以老老实实写 `dict[str, list[...]]`,不用引入 `defaultdict` "访问即插入"的隐藏行为(见第 2 节常见坑 1)——`self.subs` 只有真的调用了 `subscribe` 才会出现新 key,`publish` 单纯查询不会有副作用。

**可运行例子:** 完整复现这个 `MessageBus`,以及 3.9+ 的字典合并操作符 `|`(延伸,不是 `MessageBus` 的一部分,但同样是"操作字典"的常用惯用法):
```python
from typing import Any, Callable

class MessageBus:
    def __init__(self):
        self.subs: dict[str, list[tuple[str, Callable]]] = {}

    def subscribe(self, topic: str, agent_name: str, callback: Callable) -> None:
        self.subs.setdefault(topic, []).append((agent_name, callback))

    def publish(self, topic: str, payload: Any, from_agent: str = "?") -> int:
        delivered = 0
        for name, cb in self.subs.get(topic, []):
            if name == from_agent:
                continue
            cb(payload)
            delivered += 1
        return delivered

bus = MessageBus()
received = []
bus.subscribe("search_done", "writer", lambda p: received.append(p))
bus.subscribe("search_done", "critic", lambda p: received.append(p))
n = bus.publish("search_done", {"r": 1}, from_agent="researcher")
assert n == 2 and len(received) == 2
assert bus.publish("nobody_subscribed", {"x": 1}) == 0   # get(topic, []) 兜底,不用先判断 in

# 延伸:3.9+ 字典合并操作符 |,后面的字典覆盖前面同名 key,不修改原字典
base = {"lr": 0.01, "batch_size": 32}
override = {"batch_size": 64, "epochs": 3}
merged = base | override
assert merged == {"lr": 0.01, "batch_size": 64, "epochs": 3}
assert base == {"lr": 0.01, "batch_size": 32}   # base 没被修改
base |= override                                 # |= 原地更新,等价于 dict.update
assert base == {"lr": 0.01, "batch_size": 64, "epochs": 3}
print(merged, base)
```
`|` 合并操作符在仓库里也有真实用例:`learning/dpo-family/src/capstone_dpo_comparison.py` 第 39 行用 `{k: v.detach() for k, v in s.items()} | {"loss": ..., "margin": ...}` 把一个 tensor 状态字典和一个标量字典拼成一个返回值,不用先 `.update()` 再返回。

**常见坑:** `setdefault` 的默认值参数是"提前算好的",不是"需要时才算的"——这一点和 `defaultdict` 的惰性工厂刚好相反,容易被忽视。
```python
import collections

call_count = 0
def expensive_default():
    global call_count
    call_count += 1
    return []

cache = {"a": [1]}
cache.setdefault("a", expensive_default())   # key "a" 已经存在,但 expensive_default() 依然被调用了一次!
assert call_count == 1                        # 白白算了一次,即使结果被扔掉

dd = collections.defaultdict(expensive_default)
_ = dd["a"]        # key 不存在 -> 调用一次工厂
assert call_count == 2
_ = dd["a"]         # 这次 key 已经存在(上一步写入了),工厂不会再被调用
assert call_count == 2   # 没有变化
```
原因是 Python 的求值规则:`d.setdefault(key, expensive_default())` 里,`expensive_default()` 是一个**函数调用表达式**,在传给 `setdefault` 之前就必须先求值——不管 `setdefault` 内部用不用得上这个值。如果默认值的构造成本很高(比如要读文件、发请求),`setdefault` 会在 key 已存在时也白白付出这个成本;`defaultdict` 传进去的是"工厂函数本身"(没有加括号调用),只有真正需要默认值时才会被调用,这才是"惰性"的。

---

## 6. EAFP vs LBYL 哲学

**是什么:** 两种处理"这件事可能失败"的风格。这两个词是 Python 官方 glossary 的原文用词:

> **EAFP**(Easier to ask for forgiveness than permission,"先斩后奏,错了再说"):assumes the existence of valid keys or attributes and catches exceptions if the assumption proves false ...characterized by the presence of many `try` and `except` statements。
>
> **LBYL**(Look before you leap,"三思而后行"):explicitly tests for pre-conditions before making calls or lookups ...characterized by the presence of many `if` statements。

**为什么课堂不教但很重要:** 课堂教的错误处理大多是 LBYL 直觉(先 `if` 判断,再操作)——glossary 原文也点明 LBYL 是 "common to many other languages such as C"。Python 社区更偏爱 EAFP,原因有两个:**性能上**,乐观路径下避免了"判断"和"操作"两次重复查找;**并发场景下**,glossary 原文举了一个经典例子——`if key in mapping: return mapping[key]` 在多线程环境下 "can fail if another thread removes key from mapping after the test, but before the lookup",这就是 TOCTOU(Time-Of-Check-Time-Of-Use)竞态,EAFP 把"检查"和"使用"合并成同一个原子操作,天然不存在这个窗口。

**从最笨的写法讲起:**
```python
d = {"a": 1, "b": 2}

# LBYL:先查是否存在,再取值——对存在的 key 查了两次(in 一次,取值再一次)
def lookup_lbyl(d, key):
    if key in d:
        return d[key]
    else:
        return -1

assert lookup_lbyl(d, "a") == 1
assert lookup_lbyl(d, "z") == -1

# EAFP:直接取,不存在就走 except——只查一次
def lookup_eafp(d, key):
    try:
        return d[key]
    except KeyError:
        return -1

assert lookup_eafp(d, "a") == 1
assert lookup_eafp(d, "z") == -1
```
两种写法功能完全一样。区别在"乐观路径"(key 通常存在)下 EAFP 只做一次查找,LBYL 做两次;而"key 经常不存在"的场景(异常抛出/捕获本身有开销)LBYL 反而可能更快——EAFP 不是无条件更优,而是在"大多数情况下会成功,只是偶尔失败"这个前提下更划算,这也是研究代码里绝大多数场景的真实分布。

**AI 研究代码里的真实例子:** 两个场景,一个是把 `KeyError` 转成更友好的错误信息,一个是仓库讲义里明确讨论了 TOCTOU 竞态。

**a. 用 EAFP 把宽泛的 `KeyError` 转成语义清晰的 `ValueError`:**
```python
# learning/agent-graduation/src/eval/agent_eval_matter.py 第 97-110 行(原样抄录)
HOLDOUT_BY_GENERALITY = {
    "distribution-specific": "hold out in-distribution samples",
    "task-specific": "hold out out-of-distribution samples",
    "domain-general": "hold out tasks",
    "fully-general": "hold out domains",
}

def required_holdout(generality: str) -> str:
    try:
        return HOLDOUT_BY_GENERALITY[generality]
    except KeyError as exc:
        allowed = ", ".join(sorted(HOLDOUT_BY_GENERALITY))
        raise ValueError(f"unknown generality; expected one of {allowed}") from exc

assert required_holdout("task-specific") == "hold out out-of-distribution samples"
try:
    required_holdout("bogus")
except ValueError as e:
    print(e)   # unknown generality; expected one of distribution-specific, domain-general, fully-general, task-specific
```
如果用 LBYL 写(`if generality not in HOLDOUT_BY_GENERALITY: raise ValueError(...)`),逻辑上是等价的,但 EAFP 版本把"正常路径"(`return HOLDOUT_BY_GENERALITY[generality]`)和"错误路径"(`except` 块)在视觉上分开了,正常路径不需要多缩进一层 `if`。

**b. TOCTOU 竞态——仓库讲义里专门讨论工具调用安全时举的例子。** 下面原样抄录该讲义的示意伪代码(`check_ok`/`file` 是讲义里的占位名,没有给出具体定义,所以不是独立可运行片段——独立可运行、真正做了文件读写的对照版本在紧接着的"可运行例子"里):
```text
# learning/tool-use-mcp/lectures/11-tool-security.md 第 92-101 行(原样抄录)
# 不安全:检查后状态可能变
if check_ok(file):
    open(file).read()  # ← 此处 file 已变

# 安全:原子操作
try:
    open(file, "r").read()
except FileNotFoundError:
    ...
```
`check_ok(file)` 和 `open(file)` 是两条独立的语句,中间有一个时间窗口——如果文件在窗口期被另一个进程删除或替换(多个 agent/多进程同时操作同一份文件时完全可能发生),`check_ok` 判断出"文件存在"之后,`open` 依然可能失败,LBYL 版本对这个失败毫无防备。EAFP 版本把"检查"和"使用"合并成同一个原子操作:`open` 要么成功读到内容,要么直接抛 `FileNotFoundError` 被 `except` 接住,不存在中间的竞态窗口。

**可运行例子:** 复现 TOCTOU 对比,以及 `contextlib.suppress`——EAFP 的 one-liner 化写法:
```python
import contextlib
import os

def read_lbyl(path):
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return "<missing>"

def read_eafp(path):
    try:
        return open(path, "r", encoding="utf-8").read()
    except FileNotFoundError:
        return "<missing>"

tmp_path = "eafp_demo_tmp.txt"
with open(tmp_path, "w", encoding="utf-8") as f:
    f.write("hello")
assert read_lbyl(tmp_path) == read_eafp(tmp_path) == "hello"
os.remove(tmp_path)
assert read_lbyl(tmp_path) == read_eafp(tmp_path) == "<missing>"

# contextlib.suppress:EAFP 的 one-liner 化,"这个异常我预期会发生,忽略它"
with open(tmp_path, "w", encoding="utf-8") as f:
    f.write("x")

# 笨办法:try/except/pass 三行
try:
    os.remove(tmp_path)
except FileNotFoundError:
    pass

# 正式写法:一行表达同样的意思
with contextlib.suppress(FileNotFoundError):
    os.remove(tmp_path)   # 这次文件已经不存在,正常会报错,这里被安静吞掉
assert not os.path.exists(tmp_path)
```

**常见坑:** `except` 的范围写太宽,会把不该捕获的 bug 也一起吞掉——这是 EAFP 被滥用后最常见的后果。
```python
def risky_lookup(d, key):
    try:
        value = d[key]
        return 100 // value          # value 是 0 时会引发 ZeroDivisionError
    except KeyError:                  # 精确捕获:只处理"key 不存在"
        return -1

d2 = {"a": 0}
try:
    risky_lookup(d2, "a")             # value 存在(是 0),但除法本身出错——不是 KeyError,正常向外传播
except ZeroDivisionError as e:
    print(e)   # integer division or modulo by zero

# 反例:如果偷懒写成 except Exception(或裸 except),ZeroDivisionError 也会被"当成 key 不存在"处理掉
def risky_lookup_bad(d, key):
    try:
        value = d[key]
        return 100 // value
    except Exception:
        return -1

assert risky_lookup_bad(d2, "a") == -1   # 除零错误被错误地"吃"成了 -1,和"key 不存在"的返回值完全一样,
                                          # 排查这类 bug 时,现象和"key 真的不存在"毫无区别,非常难查
```
EAFP 的前提是 `except` 只捕获**你明确预期会发生、且知道怎么处理**的那一种异常;`except Exception` 或裸 `except:` 会把 `try` 块里任何代码路径抛出的任何异常都当成同一件事处理——这不是"拥抱失败",而是在制造更难排查的 bug。

---

## 7. `sorted`/`.sort()` 的 `key=` 参数

**是什么:** `sorted(iterable, key=f)` 让排序按 `f(元素)` 的返回值比较大小,而不是元素本身;`key=` 传入一个"提取排序依据"的函数,常见的是 `lambda`,也可以是 `operator.itemgetter`/`operator.attrgetter`。

**为什么课堂不教但很重要:** 课堂教的排序停留在"一串数字直接 `sorted()`"。研究代码里要排序的几乎总是"一堆带结构的记录"——按 accuracy 给几个 agent 配置排名、按 elapsed_s 给日志事件排序、按(成本, 准确率)多个字段综合排名——`key=` 就是把"排序依据"和"排序动作"解耦的机制。

**从最笨的写法讲起:**
```python
records = [("bob", 85), ("amy", 92), ("cy", 92)]

# 笨办法:手动把"排序依据"搬到前面,排完序再搬回去(Schwartzian transform 手写版)
decorated = [(score, name) for name, score in records]
decorated.sort()
result_dumb = [(name, score) for score, name in decorated]
assert result_dumb == [("bob", 85), ("amy", 92), ("cy", 92)]

# 正式写法:key= 直接告诉 sorted 用什么排序,不用手动搬来搬去
result = sorted(records, key=lambda r: r[1])
assert result == result_dumb
print(result)
```
`operator.itemgetter`/`operator.attrgetter` 是 `lambda r: r[1]` / `lambda r: r.attr` 的专用替代——语义更明确(一看就知道是"取某个字段排序",不是任意逻辑),官方文档也指出它通常比等价的 `lambda` 略快(省掉了一次 Python 函数调用的解释开销)。老实交代:在 `learning/` 目录(排除 vendor 的 `official/repos/`)里没有找到博士学长自己用 `itemgetter`/`attrgetter` 的例子,他的排序场景全部直接用 `lambda`——下面这段是**示例性质**,用来展示这个替代写法本身:
```python
import operator
from dataclasses import dataclass

records = [("bob", 85), ("amy", 92), ("cy", 92)]
result_ig = sorted(records, key=operator.itemgetter(1))
assert result_ig == sorted(records, key=lambda r: r[1])

@dataclass
class Student:
    name: str
    score: int

students = [Student("bob", 85), Student("amy", 92), Student("cy", 78)]
result_ag = sorted(students, key=operator.attrgetter("score"))
assert [s.name for s in result_ag] == ["cy", "bob", "amy"]

# itemgetter 还能一次取多个字段,直接组合成多级排序的 key
result_multi = sorted(records, key=operator.itemgetter(1, 0))
assert result_multi == [("bob", 85), ("amy", 92), ("cy", 92)]
print(result_ig, [s.name for s in result_ag], result_multi)
```

**AI 研究代码里的真实例子:** 用元组当 `key=` 做多级排序——先按成本升序,成本相同再按准确率降序,再相同按名字兜底。下面这行是从函数体里摘出的单行(`return` 脱离了所在函数,不是独立可运行片段,独立可运行版本在紧接着的"复现"里):
```text
# learning/agent-graduation/src/eval/agent_eval_matter.py 第 60 行(原样抄录)
return sorted(frontier, key=lambda r: (r.amortized_cost(n_runs), -r.accuracy, r.name))
```
`pareto_frontier` 函数从一批 agent 配置里挑出"性价比帕累托最优"的那些,再排出一个确定性的展示顺序。复现(独立可运行,逻辑和上面引用的真实代码一致):
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentRun:
    name: str
    accuracy: float
    variable_cost_usd: float
    fixed_cost_usd: float = 0.0

    def amortized_cost(self, n_runs: int = 1) -> float:
        return self.variable_cost_usd + self.fixed_cost_usd / n_runs

runs = [
    AgentRun("agent-zulu", accuracy=0.70, variable_cost_usd=0.01),
    AgentRun("agent-costly", accuracy=0.85, variable_cost_usd=0.05),   # 准确率全场最高,但成本也最高
    AgentRun("agent-low", accuracy=0.50, variable_cost_usd=0.01),      # 和 zulu 同价,但准确率更低
    AgentRun("agent-alpha", accuracy=0.70, variable_cost_usd=0.01),    # 和 zulu 同价同准确率,靠 name 分高下
]

ranked = sorted(runs, key=lambda r: (r.amortized_cost(n_runs=1), -r.accuracy, r.name))
ranked_names = [r.name for r in ranked]
assert ranked_names == ["agent-alpha", "agent-zulu", "agent-low", "agent-costly"]
print(ranked_names)
```
`agent-costly` 准确率(0.85)全场最高,但因为成本是第一排序键,反而排在最后——这正是这段代码要表达的业务逻辑:先看性价比(成本),不是先看准确率。`-r.accuracy` 这个写法值得注意:**`lambda` 能直接对数字字段取负号,把"降序"混进一个整体升序的元组里;`itemgetter`/`attrgetter` 做不到这一点**(它们只能整体 `reverse=True`,没法让元组里某一个字段单独反向)——这也是研究代码里多级排序场景几乎都用 `lambda` 元组、不用 `itemgetter` 的原因。

**常见坑:**

1. **`sorted` 是稳定排序(stable sort),可以利用这一点省事,但也可能被无意间依赖。** "稳定"指 key 相同的元素,排序后相对顺序和输入时一致:
```python
pairs = [("a", 2), ("b", 1), ("c", 2), ("d", 1)]

# 两趟排序(先按次要键排,再按主要键排)在稳定排序下等价于一次多级排序
step1 = sorted(pairs, key=lambda p: p[0])       # 先按字母排
step2 = sorted(step1, key=lambda p: p[1])       # 再按数字排,相同数字内部维持字母序
assert step2 == [("b", 1), ("d", 1), ("a", 2), ("c", 2)]
assert step2 == sorted(pairs, key=lambda p: (p[1], p[0]))   # 等价于一次多级排序
```
这不是巧合,是 Python 官方保证的行为(Timsort 是稳定排序算法);但反过来,如果排序代码"恰好"依赖了这个稳定性却没写注释说明,后来者很容易在"优化"排序逻辑时不小心破坏这个隐含前提。

2. **`itemgetter`/`attrgetter` 对缺失的字段直接抛错,不像 `dict.get` 那样能给默认值兜底。**
```python
import operator

bad_records = [{"name": "a", "score": 1}, {"name": "b"}]   # 第二条缺 "score" 字段
try:
    sorted(bad_records, key=operator.itemgetter("score"))
except KeyError as e:
    print(e)   # 'score'

# 想要"缺失字段给默认值"必须退回 lambda + .get()
safe_sorted = sorted(bad_records, key=lambda r: r.get("score", 0))
assert [r["name"] for r in safe_sorted] == ["b", "a"]
```
数据不保证每条记录字段齐全时(比如混合了不同 schema 版本的日志/评估记录),`itemgetter` 反而是更安全的选择——它会在数据有问题时立刻报错,而不是安静地用一个可能错误的默认值继续跑下去;但如果确实需要"缺了就给默认值"的容错行为,只能用 `lambda` + `.get()`。

---

## 小结:这一批 7 个知识点解决的问题

| 知识点 | 解决的问题 |
|---|---|
| `collections.Counter` | 给一批元素计数,不用手写 `get(k, 0) + 1` |
| `collections.defaultdict` | 按 key 分组收集时,不用手写"key 是否第一次出现"的判断 |
| `collections.namedtuple` | 比裸 tuple 可读、比手写 class 省样板代码的不可变记录类型 |
| `collections.deque` | 双端都是 O(1) 的队列/滑动窗口,替代 `list.pop(0)` 的 O(n) 陷阱 |
| `dict.get`/`dict.setdefault` | 把"判断 key 是否存在"和"取值/写入"合并成一次查找 |
| EAFP vs LBYL | 用 `try`/`except` 代替"先检查再做",避免重复查找和 TOCTOU 竞态 |
| `sorted`/`.sort()` 的 `key=` | 把"排序依据"从"排序动作"里解耦出来,支持多级/自定义排序 |

下一批:[04-strings-and-modern-syntax.md](04-strings-and-modern-syntax.md)

---

*更新:2026-07-08*
