# 04 · 类型注解、上下文管理与并发(Typing, Context Managers & Concurrency)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 5 个知识点:类型注解与 `typing` 模块、`with` 上下文管理器、`async`/`await` 协程、`asyncio` 事件循环、`threading` vs `multiprocessing`——是 python-advanced 系列的最后一篇,也是唯一一篇离不开"实际测耗时"的:协程并发快不快、多线程多进程有没有加速,不能靠嘴说,必须真的跑起来算秒数。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9,torch 2.11.0+cu128)下实际跑通验证,不是凭空写的。涉及耗时对比的例子(协程并发、多线程/多进程)文中打印的秒数是本机某次实测的真实输出,具体数字会因机器负载、核数波动,但**结论的方向**——谁比谁快、大约快多少倍——是稳定、可复现的,重新跑一遍数量级不会变。

---

## 1. 类型注解(Type Hints)与 `typing` 模块

**是什么:** 在变量、函数参数、返回值后面用 `名字: 类型` 的语法写清楚"这里期望是什么类型",比如 `x: int`、`def f(x: int) -> str:`。

**为什么课堂不教但很重要:** 翻开 HuggingFace transformers 或 PyTorch 的源码,几乎每个函数签名都长这样:`def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:`。不认识这套写法,你连"这个参数该传什么"都读不出来,只能靠试错或者翻半天文档。

**这里有一个和 C 本质不同的地方,必须先讲清楚:**

```c
int x = 5;        // C:编译器强制检查类型,写 int x = "hello"; 直接编译失败,程序都跑不起来
```

```python
x: int = 5         # Python:x 后面的 ": int" 只是一个注解
x: int = "hello"   # 这一行照样能跑!Python 解释器完全不检查,不会报任何错
```

C 的类型声明是编译器**强制检查**的契约,类型不对代码根本编译不过,连运行的机会都没有。Python 的类型注解只是**写给人看的文档 + 给 mypy/IDE 这类工具做静态检查用的元数据**——Python 解释器在运行时从头到尾不会看这个注解一眼,更不会因为它而报错。这是本节最重要的一句话,"常见坑"部分会用真实跑出来的代码把这一点(以及它带来的坑)完整演示一遍。

**从最笨的写法讲起:**

完全不写类型信息,读代码全靠猜:

```python
# 笨办法 0:什么类型信息都不写
def process_v0(data, threshold):
    return [x for x in data if x > threshold]
```

老代码里常见的折中方案:写进文档字符串,人能看懂,但 IDE/工具没法理解、更不会帮你检查:

```python
# 笨办法 1:写在注释/文档字符串里,人靠自觉遵守,工具无法理解
def process_v1(data, threshold):
    """
    data: 一个数字列表
    threshold: 一个数字,过滤阈值
    """
    return [x for x in data if x > threshold]
```

正式写法:类型注解直接写进函数签名,IDE 能自动补全参数提示,配合 mypy 之类的工具还能在**运行代码之前**就发现"传错类型"的问题:

```python
# 正式写法:类型注解写进签名
def process_v2(data: list[float], threshold: float) -> list[float]:
    return [x for x in data if x > threshold]


# 接着上面 process_v0、process_v1 的定义一起跑(三个函数写在同一个文件/会话里):
assert process_v0([1, 2, 3], 1) == [2, 3]
assert process_v1([1, 2, 3], 1) == [2, 3]
assert process_v2([1, 2, 3], 1) == [2, 3]
print("三种写法运行结果完全一样,区别只在于:人和工具能不能一眼看出参数类型")
```

三种写法功能完全一样,`process_v2` 赢在"自解释"——半年后你自己回来看这个函数,不用再去猜。

**AI 研究代码里的真实例子:** HuggingFace/PyTorch 大量使用 `Optional[Tensor]` 表示"这个参数可以不传,函数内部会给默认值"。下面用真实的 `torch.Tensor` 复现这种写法:

```python
from typing import Optional
import torch


def forward(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids * attention_mask


ids = torch.tensor([1, 2, 3])

out1 = forward(ids)  # 不传 attention_mask,函数内部用默认的全 1 mask
assert torch.equal(out1, ids)

mask = torch.tensor([1, 0, 1])
out2 = forward(ids, attention_mask=mask)  # 显式传入
assert torch.equal(out2, torch.tensor([1, 0, 3]))

print("forward 的类型注解:", forward.__annotations__)
# {'input_ids': <class 'torch.Tensor'>, 'attention_mask': typing.Optional[torch.Tensor], 'return': <class 'torch.Tensor'>}
```

光看签名 `forward(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None)`,不用看函数体、不用看文档,就知道 `attention_mask` 可以不传——这就是类型注解作为"文档"的价值。

**可运行例子:**

```python
from typing import List, Optional, Dict, get_type_hints


def greet(name: str, times: int = 1) -> str:
    return (name + " ") * times


def compute_stats(nums: List[int]) -> Dict[str, float]:
    return {"sum": float(sum(nums)), "avg": sum(nums) / len(nums)}


def find_user(user_id: int) -> Optional[str]:
    db = {1: "Alice", 2: "Bob"}
    return db.get(user_id)  # 找不到时返回 None,所以返回类型标成 Optional[str]


assert greet("Bob", 2) == "Bob Bob "
assert compute_stats([1, 2, 3]) == {"sum": 6.0, "avg": 2.0}
assert find_user(1) == "Alice"
assert find_user(999) is None

# 想在运行时读取一个函数的注解,用 __annotations__ 或 typing.get_type_hints
print(compute_stats.__annotations__)
# {'nums': typing.List[int], 'return': typing.Dict[str, float]}
print(get_type_hints(compute_stats))
# {'nums': typing.List[int], 'return': typing.Dict[str, float]}
```

`typing.List[int]`/`typing.Optional[str]`/`typing.Dict[str, int]` 是老代码里最常见的写法(不少 AI 研究仓库年头较早,至今还在用)。如果你的 Python ≥ 3.9,新代码更推荐用内置的泛型写法,效果完全等价、更简洁:

```python
# 现代写法(Python 3.9+ 内置泛型 + 3.10+ 的 "X | None"),和 typing.List/Optional 等价
def g(a: list[int], b: str | None = None) -> dict[str, int]:
    return {"len": len(a)}


print(g.__annotations__)
# {'a': list[int], 'b': str | None, 'return': dict[str, int]}
assert g([1, 2, 3]) == {"len": 3}
```

**常见坑:**

**1. 注解写错了,代码照样能跑,不会有任何报错。** 这既是类型注解的特性,也是最容易让新手困惑的坑:

```python
def greet2(name: int) -> str:
    # 注解说 name 应该是 int,但传字符串完全不会报错
    return f"hello {name}"


result = greet2("world")
print(result)              # hello world
assert result == "hello world"   # 传字符串进去,程序完全正常运行


def get_count(items: list) -> int:
    # 注解说返回 int,实际返回的是 str —— Python 同样不检查
    return str(len(items))


r = get_count([1, 2, 3])
print(type(r), r)          # <class 'str'> 3
assert isinstance(r, str)  # 证明"返回类型注解"也没有被强制执行
```

变量注解同理,`x: int = "hello"` 这种"注解和实际赋值的类型对不上"的写法,Python 解释器不会有任何异议。

**2. 想要真正抓出这类"注解和实际类型不一致"的错误,需要额外的静态类型检查工具(最常见的是 `mypy`)。** 这类工具的原理是:在**运行代码之前**,单独把你的代码读一遍、按照注解做类型推导和比对,发现不一致就报错——它不是运行时机制,是一个独立的"预检查"步骤,类似 C 编译器帮你在编译期挡掉类型错误。没装这类工具的话,类型注解就纯粹是文档,不会拦下任何错误。

---

## 2. `with` 语句与上下文管理器

**是什么:** `with` 是一个能保证"进入代码块时执行某个准备动作,离开代码块时——不管是正常结束还是中途抛异常——一定执行某个清理动作"的语法结构。背后依赖的是两个魔法方法:`__enter__`(进入时调用)和 `__exit__`(离开时调用,无论是否发生异常都会被调用)。

**为什么课堂不教但很重要:** 你在之前的教程里已经写过 `with torch.no_grad():`,但可能没深想过它到底是什么、为什么要这么写。`torch.no_grad()`/`torch.enable_grad()`/`torch.autocast()` 全都是上下文管理器;`open()` 打开文件、`threading.Lock()` 加锁,也都是。这一节把 `torch.no_grad()` 背后的原理讲透——它大致是"进入时把'要不要追踪梯度'这个全局开关关掉,退出时恢复成进入前的状态",和 C 里 `fopen`/`fclose` 或 `malloc`/`free` 要成对使用是同一类问题:资源/状态"配对"管理,只是 C 需要你自己保证每条代码路径都执行了清理,`with` 把这件事交给语言机制来保证。

**从最笨的写法讲起:**

用一个全局变量模拟"梯度追踪开关"(真实 `torch.no_grad()` 内部机制的简化版),先看不用任何保护机制会出什么问题:

```python
grad_enabled = True


def set_grad_enabled(flag):
    global grad_enabled
    grad_enabled = flag


def risky_computation():
    raise ValueError("模拟计算过程中出错")


print("=== 笨办法:手动开关,没有保护 ===")
set_grad_enabled(True)
set_grad_enabled(False)
try:
    risky_computation()
except ValueError:
    print("捕获到异常(但没有恢复 grad_enabled)")
print("恢复前 grad_enabled =", grad_enabled)
assert grad_enabled is False  # bug:状态被永久破坏了!后续所有代码都会"意外地"不追踪梯度
```

问题很明显:本来想"关闭梯度追踪,算完再打开",但 `risky_computation()` 一抛异常,"打开"那行代码根本没机会执行——`grad_enabled` 就永久卡在 `False` 了,而且这个 bug 不会立刻报错,是那种过一阵子你发现"模型怎么不更新了"才会追查到的隐蔽 bug。

用 `try/finally` 手动保证"不管出不出错,恢复代码一定执行"(接着上面的 `grad_enabled`/`set_grad_enabled`/`risky_computation` 定义继续跑):

```python
print("\n=== 用 try/finally 手动保证恢复 ===")
set_grad_enabled(True)
set_grad_enabled(False)
try:
    risky_computation()
except ValueError:
    print("捕获到异常")
finally:
    set_grad_enabled(True)  # 无论是否异常,一定会执行
print("恢复后 grad_enabled =", grad_enabled)
assert grad_enabled is True  # 这次正确恢复了
```

这样是对的,但每次要用这个"关闭/恢复"逻辑,都要重新写一遍 `try/finally`,容易漏写。`with` 语句就是把这个 `try/finally` 模式**封装成可以复用的语法糖**。

**AI 研究代码里的真实例子:** 真实的 `torch.no_grad()` 就是这个模式的标准实现——推理/评估阶段不需要反向传播,关掉梯度追踪能省显存、省计算:

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
print("进入前:", torch.is_grad_enabled())        # True
assert torch.is_grad_enabled() is True

with torch.no_grad():
    print("with 块内部:", torch.is_grad_enabled())  # False
    assert torch.is_grad_enabled() is False
    y = x * 2
    assert y.requires_grad is False   # no_grad 期间创建的 tensor 不追踪梯度

print("退出后:", torch.is_grad_enabled())         # True,自动恢复
assert torch.is_grad_enabled() is True
```

除了 `no_grad`,`with torch.autocast("cuda"):`(混合精度训练)、`with open(config_path) as f:`(读配置文件)、`with tempfile.TemporaryDirectory() as d:`(用完自动删除的临时目录)都是研究代码里的高频写法。

**可运行例子:** 自己动手实现一个简化版 `torch.no_grad()`,验证它和真实版本行为一致——这样你就知道 `no_grad()` 内部大概是怎么运作的了:

```python
from contextlib import contextmanager

grad_enabled = True


def set_grad_enabled(flag):
    global grad_enabled
    grad_enabled = flag


def risky_computation():
    raise ValueError("模拟计算过程中出错")


# 写法一:上下文管理器类,靠 __enter__/__exit__ 两个魔法方法
class NoGradLike:
    """模拟 torch.no_grad() 的核心逻辑:进入时关闭梯度追踪,退出时恢复原状态。"""

    def __enter__(self):
        global grad_enabled
        self._previous = grad_enabled   # 记住进入前的状态,而不是无脑设成 True
        grad_enabled = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global grad_enabled
        grad_enabled = self._previous   # 恢复成进入前的状态
        return False   # 不吞掉异常,让异常继续往外传播(见下面"常见坑")


set_grad_enabled(True)
try:
    with NoGradLike():
        assert grad_enabled is False
        risky_computation()
except ValueError:
    print("异常正常传播出来了,没有被吞掉")
assert grad_enabled is True   # 即使 with 块内部抛异常,__exit__ 依然被调用,状态被正确恢复


# 写法二:contextlib.contextmanager 装饰器,用 yield 分隔"进入"和"退出"逻辑
@contextmanager
def no_grad_like():
    global grad_enabled
    previous = grad_enabled
    grad_enabled = False
    try:
        yield              # yield 之前的代码 = __enter__,yield 之后的代码 = __exit__
    finally:
        grad_enabled = previous   # 必须用 try/finally 包住 yield,否则和最初的笨办法一样有 bug


set_grad_enabled(True)
with no_grad_like():
    assert grad_enabled is False
assert grad_enabled is True

print("两种写法都验证通过,行为和真实的 torch.no_grad() 完全一致")
```

两种写法本质相同:类写法更清楚地暴露 `__enter__`/`__exit__` 两个阶段;`@contextmanager` 写法更短,一个普通生成器函数,`yield` 之前是准备工作,`yield` 之后(放进 `finally`)是清理工作。真实的 `torch.no_grad()` 用的是第一种(类)写法,原理上和这里的 `NoGradLike` 是一回事。

**常见坑:**

**1. `__exit__` 返回 `True` 会意外吞掉 `with` 块里的异常。** 很多人不知道 `__exit__` 的返回值有特殊含义,随手写了个 `return True` 收尾,结果真正的报错被悄悄吃掉,程序看起来"正常运行",实际上出了 bug 都不知道:

```python
class BadContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("清理完成")
        return True  # 危险!任何返回 True 都会让 with 块里的异常"消失"


with BadContextManager():
    raise RuntimeError("这个异常本应该让程序崩溃")
print("程序还在正常往下走 —— 异常被 __exit__ 悄悄吞掉了")
# 实测:这个 print 真的会执行,整个脚本 exit code 是 0,RuntimeError 完全消失
```

正确做法:`__exit__` 一般应该 `return False`(或者不写 `return`,默认就是 `None`,效果等价于 `False`)——只有明确要"消化掉某种特定异常"时才返回 `True`,而且要在 `exc_type` 上做判断,不能无差别吞掉所有异常。

**2. 用 `@contextmanager` 写法时,忘记把 `yield` 包在 `try/finally` 里。** 如果直接写成:

```python
from contextlib import contextmanager

grad_enabled = True


@contextmanager
def bad_no_grad_like():
    global grad_enabled
    previous = grad_enabled
    grad_enabled = False
    yield                     # 没有 try/finally 保护
    grad_enabled = previous   # with 块内部一旦抛异常,这行永远不会执行到


try:
    with bad_no_grad_like():
        raise ValueError("模拟计算过程中出错")
except ValueError:
    print("捕获到异常")

print("异常发生后 grad_enabled =", grad_enabled)   # False —— 没有被恢复!
assert grad_enabled is False  # 复现 bug:和本节开头"最笨的写法"是同一个错误
```

这就和本节开头"最笨的写法"犯了一模一样的错误——只是把 bug 从"忘记写 `finally`"搬到了"忘记把 `yield` 包进 `try/finally`"里,`with` 语法本身不会自动帮你补上这个保护,该有的 `try/finally` 还是得自己写。

---

## 3. `async`/`await` 协程基础

**是什么:** `async def` 定义一个"协程函数",调用它不会立刻执行,而是先拿到一个"协程对象"。`await` 用来等待一个耗时操作的结果,并且在等待期间**主动让出控制权**,让同一个线程可以先去处理别的任务,而不是傻乎乎地卡在原地空等。这是"单线程内部的并发调度",不是开了多个线程——全程只有一个线程在跑,只是这个线程很会"见缝插针"。

**为什么课堂不教但很重要:** 调用 LLM API(尤其是一次要并发调用多个模型/多个 prompt)是 `async`/`await` 最经典的应用场景;OpenAI、Anthropic 的官方 Python SDK 都提供异步客户端;FastAPI 这类现代 Web 框架里,路由函数大量写成 `async def`。

**从最笨的写法讲起:** 先看纯同步、什么并发机制都不用的版本——用 `time.sleep` 模拟"发一个耗时的网络请求":

```python
import time


def blocking_request(name, delay):
    print(f"[{name}] 开始请求...")
    time.sleep(delay)  # 傻等:这段时间 CPU 完全闲着,程序也做不了任何别的事
    print(f"[{name}] 请求完成")
    return f"{name}-result"


t0 = time.perf_counter()
r1 = blocking_request("任务A", 1)
r2 = blocking_request("任务B", 1)
elapsed = time.perf_counter() - t0

print(f"总耗时: {elapsed:.2f}s")   # 实测: 2.00s
assert elapsed >= 1.9  # 两个任务老老实实排队等,耗时约等于两者之和
```

`time.sleep` 期间,这个线程被操作系统挂起,**什么都干不了**——哪怕你还有另一个完全不相关的任务在排队,它也得等这个 `sleep` 醒过来才能开始。这就是"傻等"。`async`/`await` 要解决的正是这个问题:遇到耗时等待时,让出线程去做点别的,而不是死等。

**AI 研究代码里的真实例子:** 下面是异步调用 LLM API 的典型写法(依赖真实网络和 API key,这里只展示代码结构,不在本机运行):

```python
# 示意代码 —— 真实场景需要网络和 API key,这里只展示写法,不在本机执行
async def call_llm(client, prompt: str) -> str:
    response = await client.chat.completions.create(
        model="some-model",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

`await` 出现在"发出请求、等服务器返回"这一步——网络往返的这几百毫秒里,程序可以先去处理别的协程(比如同时发出去的另一个请求),下一节 `asyncio.gather` 会把这一点变成可以实际测量的耗时对比。

**可运行例子:** 先看正确的驱动方式,再看"单线程内部真的能交替处理多个任务"这件事:

```python
import asyncio
import time


async def fake_request(name, delay):
    print(f"[{name}] 开始请求...")
    await asyncio.sleep(delay)  # 模拟耗时的网络请求;await 期间可以让出控制权
    print(f"[{name}] 请求完成")
    return f"{name}-result"


# 正确用法:用 asyncio.run() 驱动协程真正执行
real_result = asyncio.run(fake_request("A", 1))
print(real_result)
assert real_result == "A-result"


# 两个协程交替推进,证明"单线程内并发调度"是真的,不是嘴上说说
async def ticker(name, delay, times):
    for i in range(times):
        await asyncio.sleep(delay)
        print(f"  <{name}> 第 {i + 1} 次 tick,耗时 {delay}s")


async def demo_concurrency():
    await asyncio.gather(
        ticker("快", 0.3, 3),   # 每 0.3s tick 一次,共 3 次
        ticker("慢", 0.5, 2),   # 每 0.5s tick 一次,共 2 次
    )


t0 = time.perf_counter()
asyncio.run(demo_concurrency())
elapsed = time.perf_counter() - t0
print(f"两个 ticker 总耗时: {elapsed:.2f}s")
# 顺序执行的话要 0.3*3 + 0.5*2 = 1.9s;并发执行接近 max(0.3*3, 0.5*2) = 1.0s
assert elapsed < 1.5
```

本机某次实测输出(打印顺序按实际发生时间交替,不是先跑完一个 ticker 再跑另一个):

```
  <快> 第 1 次 tick,耗时 0.3s
  <慢> 第 1 次 tick,耗时 0.5s
  <快> 第 2 次 tick,耗时 0.3s
  <快> 第 3 次 tick,耗时 0.3s
  <慢> 第 2 次 tick,耗时 0.5s
两个 ticker 总耗时: 1.02s
```

"快" ticker 和"慢" ticker 的打印是交替出现的,这就是"单线程内并发调度"的直接证据——如果是顺序执行,一定是"快"的 3 条打印完,才会开始"慢"的打印。

**常见坑:**

**`async def` 函数不加 `await`(或者不用 `asyncio.run()`)直接调用,不会真正执行函数体,只会得到一个"协程对象"。** 这是新手写 async 代码时几乎必踩的坑:

```python
import asyncio


async def fake_request(name, delay):
    print(f"[{name}] 开始请求...")
    await asyncio.sleep(delay)
    print(f"[{name}] 请求完成")
    return f"{name}-result"


maybe_result = fake_request("A", 1)
print(maybe_result)
# 实测输出: <coroutine object fake_request at 0x00000000042D5FC0>
# 注意:上面这一行 print 之前,"[A] 开始请求..." 根本没有被打印出来 —— 函数体完全没有执行!
print(type(maybe_result))
# 实测输出: <class 'coroutine'>
assert asyncio.iscoroutine(maybe_result)
maybe_result.close()  # 用不上的话记得关掉,否则解释器会警告
```

这个"协程对象"只是一张"待办事项",描述了"将来要执行 `fake_request('A', 1)` 这件事",但**还没有真正跑**。必须用 `await` 某个协程(在另一个协程内部),或者用 `asyncio.run()` 在最外层启动,函数体才会真正执行。

如果这个从未被 `await` 的协程对象后面被垃圾回收,Python 会在退出时打印一条很有用的警告——实测:

```
RuntimeWarning: coroutine 'fake_request' was never awaited
```

**记住这条警告的含义:平时在自己代码里看到 `coroutine '...' was never awaited`,基本可以断定是漏写了 `await` 或者忘了用 `asyncio.run()`。**

---

## 4. `asyncio` 事件循环基础用法

**是什么:** `asyncio.run(coro)` 是几乎所有 asyncio 程序的入口——创建一个事件循环、把协程扔进去执行、执行完再关闭循环,一步到位。`asyncio.gather(*coros)` 把多个协程一起提交给事件循环,让它们"同时"推进,而不是一个等完了再等下一个。

**为什么课堂不教但很重要:** 批量调用 LLM API(打标签、批量翻译、批量生成)是 `gather` 的经典场景——顺序一个个 `await` 等,和用 `gather` 一次性并发跑,耗时能差好几倍,而且请求越多差距越大。

**从最笨的写法讲起:** 顺序 `await` 多个耗时任务,一个接一个地等:

```python
import asyncio
import time


async def fake_llm_call(name, delay):
    """模拟调用一次 LLM API:网络往返 + 排队,耗时但不占 CPU。"""
    await asyncio.sleep(delay)
    return f"{name}-response"


REQUESTS = [("req1", 1.0), ("req2", 1.0), ("req3", 1.0)]


async def sequential_version():
    results = []
    for name, delay in REQUESTS:
        r = await fake_llm_call(name, delay)  # 一个个等,等完一个再发下一个
        results.append(r)
    return results


t0 = time.perf_counter()
seq_results = asyncio.run(sequential_version())
seq_elapsed = time.perf_counter() - t0
print(f"顺序 await 耗时: {seq_elapsed:.2f}s")   # 实测: 3.02s
assert seq_elapsed >= 2.9   # 3 个 1 秒任务顺序执行,总耗时接近 3 秒
```

三个各耗时 1 秒的请求,顺序执行接近 3 秒——完全符合直觉,但白白浪费了"其实可以同时发出去"的机会。

**AI 研究代码里的真实例子:** 批量处理场景下用 `gather` 并发发出多个请求,是研究代码里最常见的加速手段之一(依赖真实 API,这里只展示代码结构):

```python
# 示意代码 —— 真实场景需要网络和 API key,这里只展示写法,不在本机执行
async def label_batch(client, prompts: list[str]) -> list[str]:
    tasks = [call_llm(client, p) for p in prompts]
    return await asyncio.gather(*tasks)   # 所有 prompt 同时发出去,不用排队等
```

**可运行例子:** 同样的三个请求,换成 `asyncio.gather` 并发跑,实测耗时对比(接着上面"从最笨的写法讲起"里的 `REQUESTS`、`fake_llm_call`、`seq_results`、`seq_elapsed` 继续跑,不用重新定义):

```python
async def concurrent_version():
    tasks = [fake_llm_call(name, delay) for name, delay in REQUESTS]
    return await asyncio.gather(*tasks)   # 三个请求同时发出去,谁先好都不耽误别人


t0 = time.perf_counter()
gather_results = asyncio.run(concurrent_version())
gather_elapsed = time.perf_counter() - t0
print(f"gather 并发耗时: {gather_elapsed:.2f}s")   # 实测: 1.01s

assert gather_elapsed < 1.5, f"并发版本应该接近 1 秒,实际 {gather_elapsed:.2f}s"
assert gather_results == seq_results   # 结果内容完全一样,只是耗时天差地别
print(f"并发版本比顺序版本快了约 {seq_elapsed / gather_elapsed:.1f} 倍")   # 实测: 约 3.0 倍
```

3 个耗时 1 秒的任务,顺序执行 3.02s,`gather` 并发执行只要 1.01s——本机实测约快 **3 倍**,和"3 个任务几乎同时开始、几乎同时结束"的预期完全吻合。任务数量越多,`gather` 的优势越明显(10 个 1 秒任务,顺序要 10 秒,`gather` 依然接近 1 秒)。

**常见坑:**

**1. `gather` 返回结果的顺序 = 传入协程的顺序,不是"谁先完成谁排前面"。**

```python
import asyncio


async def slow_first():
    await asyncio.sleep(0.5)
    return "慢任务(0.5s)"


async def fast_second():
    await asyncio.sleep(0.1)
    return "快任务(0.1s)"


async def order_demo():
    return await asyncio.gather(slow_first(), fast_second())


order_result = asyncio.run(order_demo())
print(order_result)   # ['慢任务(0.5s)', '快任务(0.1s)']
# 虽然 fast_second 只用 0.1s 就先完成了,但结果列表里第一个仍然是 slow_first 的结果
assert order_result == ["慢任务(0.5s)", "快任务(0.1s)"]
```

**2. 默认情况下,只要有一个任务抛异常,`gather` 会立刻把异常抛出来,其它任务的结果拿不到。** 想让"某几个任务失败不影响其它任务",要传 `return_exceptions=True`,异常会被当成普通结果放进列表里:

```python
import asyncio


async def ok_task(name):
    await asyncio.sleep(0.1)
    return f"{name}-ok"


async def bad_task(name):
    await asyncio.sleep(0.05)
    raise ValueError(f"{name} 出错了")


# 注意:asyncio.gather(...) 必须在一个"正在运行的事件循环"里调用,
# 不能直接当参数传给 asyncio.run() —— 所以要包一层 async def 函数,
# 内部用 await asyncio.gather(...),再把这个函数整体交给 asyncio.run() 驱动。
async def demo_default():
    return await asyncio.gather(ok_task("A"), bad_task("B"), ok_task("C"))


async def demo_return_exceptions():
    return await asyncio.gather(
        ok_task("A"), bad_task("B"), ok_task("C"), return_exceptions=True
    )


# 默认行为:立刻抛异常
try:
    asyncio.run(demo_default())
except ValueError as e:
    print("捕获到异常:", e)   # B 出错了

# return_exceptions=True:异常被收集进结果列表,不中断其它任务
results = asyncio.run(demo_return_exceptions())
assert results[0] == "A-ok"
assert isinstance(results[1], ValueError)
assert results[2] == "C-ok"
print(results)   # ['A-ok', ValueError('B 出错了'), 'C-ok']
```

批量调用 API 时,如果不加 `return_exceptions=True`,某一个 prompt 触发了异常(比如内容审核拦截),整批任务会直接崩掉,新手很容易一头雾水找不到是哪个任务的问题。

---

## 5. `threading` vs `multiprocessing`(GIL 的影响)

**是什么:** `threading` 在同一个进程里开多条线程,它们共享同一份内存;`multiprocessing` 开多个完全独立的进程,各自有各自独立的内存空间(数据要在进程间传递,得靠序列化/管道,不能直接共享变量)。CPython(标准 Python 解释器)有一个 **GIL(全局解释器锁)**:同一时刻,一个进程内只允许一个线程真正执行 Python 字节码——这意味着单纯靠开多线程,并不能让 CPU 密集型任务跑得更快。

**为什么课堂不教但很重要:** 数据预处理阶段(tokenize、数据增强)常用 `multiprocessing` 加速,比如 `torch.utils.data.DataLoader(num_workers=4)`、HuggingFace `datasets.map(..., num_proc=8)`,背后都是开多个进程并行处理;而 I/O 密集型任务(同时发多个 HTTP 请求、同时读多个文件)用 `threading` 或者上两节的 `asyncio` 反而更划算,因为等待网络/磁盘 I/O 的那一刻,GIL 会被释放,别的线程可以趁机执行。

**从最笨的写法讲起:** "CPU 密集型任务,开线程应该能加速吧?"——这是最容易踩的直觉陷阱,实测证明并不是:

```python
import threading
import time


def cpu_bound_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


if __name__ == "__main__":
    N = 60_000_000

    t0 = time.perf_counter()
    cpu_bound_task(N)
    cpu_bound_task(N)
    single_thread_time = time.perf_counter() - t0
    print(f"单线程顺序跑2次: {single_thread_time:.2f}s")   # 实测: 5.19s

    # "想当然"的做法:CPU 密集型任务也开两个线程试图加速
    t0 = time.perf_counter()
    threads = [threading.Thread(target=cpu_bound_task, args=(N,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threading_time = time.perf_counter() - t0
    print(f"多线程并发跑2次: {threading_time:.2f}s")   # 实测: 4.93s

    print(f"多线程 / 单线程 耗时比: {threading_time / single_thread_time:.2f}")   # 实测: 0.95
    # 关键结论:多线程没有带来随线程数线性增长的加速——2 个线程该是接近 2 倍速度,
    # 实际观察到的是"基本没变化"(有时因为线程切换开销甚至更慢),这就是 GIL 的直接体现
    assert threading_time >= single_thread_time * 0.85
    print("验证通过:CPU 密集型任务,多线程没有带来预期的加速")
```

两个线程"看起来"在同时跑,但 GIL 保证任何时刻只有一个线程真正在执行 Python 字节码——另一个线程只能干等着轮到自己,本质上还是在"排队",只是排队的方式从"代码里显式排"变成了"解释器内部悄悄排",总耗时并没有因为开了线程而缩短到接近一半。

**注意:Windows 上运行含 `multiprocessing`/`threading.Thread` 的脚本,必须放进 `if __name__ == "__main__":` 里**,原因见下面"常见坑"第 2 条。

**AI 研究代码里的真实例子:**

真实的 `DataLoader(num_workers=...)`,`num_workers > 0` 时背后就是在开子进程并行取数据/做预处理:

```python
import torch
from torch.utils.data import DataLoader, Dataset


class SquareDataset(Dataset):
    """一个假数据集:第 i 个样本就是 i 的平方,模拟"预处理"用 CPU 算一下。"""

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return idx * idx


if __name__ == "__main__":
    ds = SquareDataset()

    loader0 = DataLoader(ds, batch_size=4, num_workers=0)  # 主进程自己算
    batches0 = [b.tolist() for b in loader0]

    loader2 = DataLoader(ds, batch_size=4, num_workers=2)  # 后台开 2 个子进程并行取数据
    batches2 = [b.tolist() for b in loader2]

    assert batches0 == batches2 == [[0, 1, 4, 9], [16, 25, 36, 49]]
    print("num_workers=0 和 num_workers=2 结果一致,后者只是换成多进程并行,不改变结果")
```

I/O 密集型场景下用 `concurrent.futures.ThreadPoolExecutor` 并发调用多个 API,不用碰 `async`/`await` 也能拿到加速(这是很多同步代码库最简单的"加速改造"方式):

```python
import time
from concurrent.futures import ThreadPoolExecutor


def call_fake_api(prompt):
    """模拟调用一次 LLM API(同步客户端),耗时但不占 CPU。"""
    time.sleep(0.3)
    return f"response-to-{prompt}"


prompts = [f"prompt{i}" for i in range(5)]

t0 = time.perf_counter()
sequential_results = [call_fake_api(p) for p in prompts]
sequential_time = time.perf_counter() - t0

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=5) as pool:
    concurrent_results = list(pool.map(call_fake_api, prompts))
concurrent_time = time.perf_counter() - t0

print(f"顺序调用: {sequential_time:.2f}s, 线程池并发: {concurrent_time:.2f}s")
# 实测: 顺序调用 1.50s, 线程池并发 0.30s
assert sequential_results == concurrent_results
assert concurrent_time < sequential_time / 2
```

5 个各耗时 0.3 秒的"API 调用",顺序执行 1.50 秒,线程池并发执行只要 0.30 秒——I/O 密集型任务上,`threading` 的加速效果非常真实,和上面 CPU 密集型任务的结果(几乎不加速)形成鲜明对比。

**可运行例子:** 把 CPU 密集型和 I/O 密集型任务放在一起对比 `threading` 与 `multiprocessing`:

```python
import multiprocessing
import threading
import time


def cpu_bound_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def io_bound_task(seconds):
    time.sleep(seconds)  # 模拟"发一个网络请求/读一个慢文件"


if __name__ == "__main__":
    N = 60_000_000

    # ---- CPU 密集型:单线程 / 多线程 / 多进程 三方对比 ----
    t0 = time.perf_counter()
    cpu_bound_task(N)
    cpu_bound_task(N)
    single_thread_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    threads = [threading.Thread(target=cpu_bound_task, args=(N,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threading_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    processes = [multiprocessing.Process(target=cpu_bound_task, args=(N,)) for _ in range(2)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    multiprocessing_time = time.perf_counter() - t0

    print(f"[CPU密集] 单线程: {single_thread_time:.2f}s  多线程: {threading_time:.2f}s  多进程: {multiprocessing_time:.2f}s")
    # 实测: 单线程 4.96s  多线程 5.32s  多进程 3.14s

    assert threading_time >= single_thread_time * 0.85       # 多线程几乎没有加速
    assert multiprocessing_time < single_thread_time * 0.85  # 多进程明显更快(真正用了多核)

    # ---- I/O 密集型:单线程 / 多线程 对比 ----
    t0 = time.perf_counter()
    io_bound_task(0.5)
    io_bound_task(0.5)
    io_bound_task(0.5)
    single_io_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    io_threads = [threading.Thread(target=io_bound_task, args=(0.5,)) for _ in range(3)]
    for t in io_threads:
        t.start()
    for t in io_threads:
        t.join()
    io_threading_time = time.perf_counter() - t0

    print(f"[IO密集] 单线程: {single_io_time:.2f}s  多线程: {io_threading_time:.2f}s")
    # 实测: 单线程 1.50s  多线程 0.50s

    assert io_threading_time < single_io_time / 2  # I/O 密集型任务,多线程有明显加速
    print("结论验证通过:CPU 密集型任务多进程明显更快,I/O 密集型任务多线程明显更快")
```

四组数字放在一起看结论最清楚:CPU 密集型任务上,多线程(5.32s)不比单线程(4.96s)快,多进程(3.14s)明显更快;I/O 密集型任务上,多线程(0.50s)比单线程(1.50s)快了 3 倍。**同一个"threading"模块,对两种任务的效果完全相反**,原因就是 GIL 只在"纯 CPU 计算"时才会成为瓶颈——等待 I/O 的那一刻,GIL 会被释放,其它线程可以插进来干活。

**常见坑:**

**1. CPU 密集型任务用 `threading` 几乎得不到加速,不要想当然地"开线程数=加速倍数"。** 上面的实测已经验证:2 个线程处理 2 份同样的 CPU 密集型任务,耗时和单线程基本在同一量级,不会像多进程那样明显缩短。

**2. Windows(以及较新版本的 macOS 默认)用 `spawn` 方式创建子进程,`multiprocessing` 相关代码必须放进 `if __name__ == "__main__":` 保护里,写在模块顶层会直接报错。** 实测复现:

```python
# 故意不写 if __name__ == "__main__": 保护
import multiprocessing


def worker():
    print("child running")


p = multiprocessing.Process(target=worker)
p.start()
p.join()
```

真实报错(节选):

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module
```

原因:`spawn` 方式会重新启动一个 Python 解释器、重新 `import` 一遍你的脚本来获取要执行的目标函数——如果创建子进程的代码没有 `if __name__ == "__main__":` 保护,子进程重新 `import` 这个脚本时会再次执行"创建子进程"这行代码,导致无限递归创建进程,Python 检测到这种情况会直接抛错拦下来。**结论:凡是用到 `multiprocessing`(包括 `DataLoader(num_workers>0)`)的脚本,主逻辑一律套上 `if __name__ == "__main__":`,这是 Windows 环境下的硬性要求,不是可选的代码风格。**

**3. 判断口诀:I/O 密集型任务(网络请求、读写文件、数据库查询)用 `threading` 或 `asyncio`;CPU 密集型任务(数值计算、数据增强、tokenize 大批文本)用 `multiprocessing`。** 用错方向,轻则白忙活(CPU 密集用了 threading,没有加速),重则效果适得其反(I/O 密集用了 multiprocessing,进程创建开销比省下的等待时间还大)。

---

## 小结

| # | 知识点 | 一句话结论 |
|---|---|---|
| 1 | 类型注解与 `typing` | 给人看的文档 + 给 mypy/IDE 用的元数据,Python 运行时完全不检查 |
| 2 | `with` / 上下文管理器 | `__enter__`/`__exit__` 保证"进入必有配对的退出",哪怕中间抛异常;`torch.no_grad()` 就是这个模式 |
| 3 | `async`/`await` | 单线程内的协作式并发,遇到 `await` 让出控制权;忘记 `await` 只会拿到一个没执行的协程对象 |
| 4 | `asyncio` 事件循环 | `asyncio.run()` 启动,`asyncio.gather()` 让多个协程并发跑,批量任务能快好几倍 |
| 5 | `threading` vs `multiprocessing` | GIL 限制下,CPU 密集型任务用 `multiprocessing`,I/O 密集型任务用 `threading`/`asyncio` |

到这里,20 个"课堂不教但 AI 研究代码里到处都是"的 Python 进阶知识点已经全部完成:函数与闭包进阶(01)、迭代器与生成器(02)、面向对象进阶(03)、类型注解与并发(04)。建议接下来交替看 [numpy-deep-dive/](../numpy-deep-dive/) 系列和这个系列里还没看完的部分——两条线一条补"Python 语言本身",一条补"AI 数值计算的库",搭配着看,读仓库代码时才不会两头卡壳。

---

*更新:2026-07-07*
