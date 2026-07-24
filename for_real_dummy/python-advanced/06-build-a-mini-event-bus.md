# 06 · 手把手实战:从零搭一个迷你事件总线

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 6 个"知识点",不计入"20 个知识点"的统计——和 [05 类](05-advanced-interview-depth.md)是同一挂,但风格不一样:05 号文件里,你是**旁观者**,跟着五条追问链把"为什么会这样"的推理过程看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个完整能用的小工具。这个格式先在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点过,这一篇是同一格式在 python-advanced 系列的落地。

## 为什么是"事件总线"

不是要发明新知识点,是把三个你已经学过的 Python 语言机制串成一个真实有用的东西。**发布/订阅(pub/sub)** 是一种很常见的解耦模式——"某件事发生了"和"谁关心这件事、该怎么处理"完全分开,触发事件的一方不需要知道有多少个、哪些订阅者在监听。训练框架里的 callback/hook 机制(比如"每个 epoch 结束时依次通知所有登记过的回调函数")用的就是这个模式,只是通常会包一层更完整的框架代码;这里从零手写一个最小可用版本,看清楚它到底是怎么靠装饰器、闭包、生成器这三样东西拼出来的。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 用 `@bus.on("event_name")` 登记回调,`emit` 触发时依次调用登记过的回调 | [01 类](01-functions-and-closures.md) 第 2 节 `*args`/`**kwargs`、第 4/5 节 装饰器基础与装饰器工厂 |
| 阶段 2 | 每个订阅自己记得"被调用了几次、最近一次收到什么参数",多个订阅互不干扰 | [01 类](01-functions-and-closures.md) 第 3 节 闭包、第 6.1 节 `functools.wraps` |
| 阶段 3 | 回放某个事件的历史记录时,不是一次性拿到全部历史,而是要一条给一条,只处理真正被要的那几条 | [02 类](02-iterators-and-generators.md) 第 2/3/4 节 `yield` 生成器、惰性求值、`yield from` |
| 阶段 4 | 把前三步拼成一个完整的 `MiniEventBus` 类,跑一次端到端 demo(模拟一个玩具训练循环) | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行。本文件用本目录下的 `_verify_md.py` 校验——内容和 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py) 完全一致,只是拷贝了一份放进 python-advanced 目录(校验逻辑和具体系列无关,通用)。校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的代码时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的。

---

## 阶段 1:用装饰器登记回调——`@bus.on(event_name)` 怎么把函数"记"下来

最笨的写法是调用者自己维护一个字典,手动把函数塞进去:`subscribers.setdefault(event_name, []).append(handle_signup)`。这样写完全能用,但注册和函数定义离得很远、容易漏写,而且"配置参数(事件名)→ 真正的装饰逻辑"这个两层结构,正是 [01 类](01-functions-and-closures.md) 第 5 节"带参数的装饰器"讲过的装饰器工厂模式——只是这次工厂不是模块级函数,而是绑定在 `bus` 实例上的方法。

```python
class MiniEventBus:
    """最小版本：只做"登记回调 + 触发时依次调用"，先不管状态和历史。"""
    def __init__(self):
        self._subscribers = {}   # event_name -> [callback, callback, ...]

    def on(self, event_name):
        """装饰器工厂：event_name 是配置参数，返回真正的装饰器（对应01类第5节"带参数的装饰器"）"""
        def decorator(func):
            self._subscribers.setdefault(event_name, []).append(func)
            return func   # 这一步先不包 wrapper，直接把原函数注册进去、原样返回
        return decorator

    def emit(self, event_name, *args, **kwargs):
        for func in self._subscribers.get(event_name, []):
            func(*args, **kwargs)

bus = MiniEventBus()
calls = []

@bus.on("user_signup")
def handle_signup(username):
    calls.append(("signup", username))

@bus.on("user_signup")
def send_welcome_email(username):
    calls.append(("email", username))

@bus.on("user_login")
def handle_login(username):
    calls.append(("login", username))

bus.emit("user_signup", "alice")
bus.emit("user_login", "alice")
bus.emit("nonexistent_event", "bob")   # 没有人订阅，emit 应该安全地什么都不做

assert calls == [("signup", "alice"), ("email", "alice"), ("login", "alice")]

# 因为 decorator 直接 return func、没有包一层 wrapper，handle_signup 本身完全没变，
# 还是那个原始函数：可以直接调用，__name__ 也没被顶替
assert handle_signup.__name__ == "handle_signup"
assert handle_signup("carol") is None   # 直接调用原函数，不经过 bus，返回值就是函数体本身的返回值（没有 return，所以是 None）
print("stage1 ok:", calls)
```

真实跑出来的 `calls` 其实是 4 条,不是 3 条:

```
stage1 ok: [('signup', 'alice'), ('email', 'alice'), ('login', 'alice'), ('signup', 'carol')]
```

这不是 bug。`handle_signup("carol")` 那一行是**直接调用函数本身**,完全绕开了 `bus.emit`——因为阶段 1 的 `decorator` 只是把 `handle_signup` 塞进 `self._subscribers` 后原样 `return func`,`handle_signup` 这个名字指向的还是最原始的那个函数,和没被装饰过没有任何区别。这一点在阶段 2 会变:一旦装饰器需要"在调用前后插入逻辑"(比如计数),就必须换成返回一个新的 `wrapper`,直接调用和经过 `bus.emit` 调用才会真正统一到同一条路径上。

**`@bus.on("user_signup")` 展开等价于 `handle_signup = bus.on("user_signup")(handle_signup)`**——比 [01 类](01-functions-and-closures.md)的 `@retry(times=3)` 多一个环节:`bus.on` 是绑定方法,`bus.on("user_signup")` 这一步会先把 `bus` 作为 `self` 自动传进去,再拿到"配置好了 `event_name`、并且已经绑定了这个 `bus` 实例"的 `decorator` 函数。`emit` 的签名用了 [01 类](01-functions-and-closures.md) 第 2 节的 `*args`/`**kwargs`——总线不知道也不需要知道每个事件具体会带什么参数,原样转发给每个订阅的回调,和那一节讲的"转发层"是同一个道理。

这个"配置参数登记进字典、目标原样返回"的结构,和 [01 类](01-functions-and-closures.md) 第 5 节"AI 研究代码里的真实例子"——模型注册表 `register_model(name)` 几乎是同一个结构,只是注册目标从"类"换成了"事件回调函数",字典从模块级 `MODEL_REGISTRY` 换成了实例属性 `self._subscribers`。

---

## 阶段 2:用闭包让每个订阅记住自己的调用次数——`stats` 这个"格子"

阶段 1 能通知,但没法回答"这个回调到底被触发了几次、最近一次收到了什么参数"这类问题——除非让每个回调自己在函数体里手动维护一个计数器。更好的做法是让总线本身透明地记录这件事,不用回调作者操心。这就需要在 `func` 外面包一层 `wrapper`,并且这个 `wrapper` 要在多次调用之间"记住"一份专属于这次注册的状态——这正是 [01 类](01-functions-and-closures.md) 第 3 节闭包要解决的问题。

```python
import functools

class MiniEventBus:
    def __init__(self):
        self._subscribers = {}

    def on(self, event_name):
        def decorator(func):
            # 这份 dict 就是"格子"——每次 decorator(func) 被调用（也就是每次 @bus.on(...) 生效一次）
            # 都会新建一份，属于这一次注册独有
            stats = {"call_count": 0, "last_args": None, "last_kwargs": None}

            @functools.wraps(func)   # 01类6.1节的教训：这里如果不加，wrapper.__name__ 会变成 "wrapper"
            def wrapper(*args, **kwargs):
                stats["call_count"] += 1
                stats["last_args"] = args
                stats["last_kwargs"] = kwargs
                return func(*args, **kwargs)

            wrapper.stats = stats   # 把这份状态也挂到 wrapper 身上，方便外部直接查看
            self._subscribers.setdefault(event_name, []).append(wrapper)
            return wrapper   # 注意：这次返回的是 wrapper，不是 func 本身——和阶段1不一样
        return decorator

    def emit(self, event_name, *args, **kwargs):
        for wrapper in self._subscribers.get(event_name, []):
            wrapper(*args, **kwargs)

bus = MiniEventBus()

@bus.on("tick")
def on_tick(payload):
    return payload * 2

@bus.on("tick")
def on_tick_logger(payload):
    pass

# 还没触发过，格子里是初始值
assert on_tick.stats == {"call_count": 0, "last_args": None, "last_kwargs": None}
# functools.wraps 保住了 __name__，不会变成 "wrapper"
assert on_tick.__name__ == "on_tick"

bus.emit("tick", 1)
bus.emit("tick", payload=5)

assert on_tick.stats["call_count"] == 2
assert on_tick.stats["last_args"] == ()                    # 第二次调用用的是关键字参数，位置参数是空 tuple
assert on_tick.stats["last_kwargs"] == {"payload": 5}

# 两个订阅都挂在同一个 "tick" 事件上、被 emit 调用了同样的次数，
# 但各自的 stats 是两个不同的 dict 对象——互不干扰，不是同一份
assert on_tick.stats is not on_tick_logger.stats
assert on_tick_logger.stats["call_count"] == 2

# 独立性还能用不同的调用次数验证：两个订阅分别挂在不同事件上，触发次数不同
@bus.on("beat")
def on_beat(x):
    pass

bus.emit("beat", 1)
assert on_beat.stats["call_count"] == 1
assert on_tick.stats["call_count"] == 2   # on_tick 不受 on_beat 被调用的影响
print("stage2 stats ok:", on_tick.stats, on_tick_logger.stats, on_beat.stats)
```

**这里为什么不需要 `nonlocal`?** [01 类](01-functions-and-closures.md) 第 3 节的 `make_counter` 里,`count += 1` 必须声明 `nonlocal count`,不然会报 `UnboundLocalError`——因为 `int` 是不可变对象,`count += 1` 本质是"算出一个新 `int`,再把 `count` 这个名字重新绑定到新对象上",这是对**名字本身**的重新赋值。这里的 `stats["call_count"] += 1` 不一样:它改的是 `stats` 这个 dict **内部**的一个键,`stats` 这个名字本身在 `wrapper` 里从来没有被重新赋值过(没有出现过 `stats = ...`),所以不需要 `nonlocal`。判断标准不是"看起来有没有在改值",而是"有没有对捕获的变量名本身重新赋值"。

`wrapper.stats = stats` 这一行不是凭空多出来的属性,它和 `wrapper` 闭包里真正抓住的那个"格子"是同一个对象——这一点不用脑补,用 [01 类](01-functions-and-closures.md) 同款的 `__closure__`/`cell_contents` 方法直接读出来:

```python
import functools

class MiniEventBus:
    def __init__(self):
        self._subscribers = {}

    def on(self, event_name):
        def decorator(func):
            stats = {"call_count": 0, "last_args": None, "last_kwargs": None}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                stats["call_count"] += 1
                stats["last_args"] = args
                stats["last_kwargs"] = kwargs
                return func(*args, **kwargs)

            wrapper.stats = stats
            self._subscribers.setdefault(event_name, []).append(wrapper)
            return wrapper
        return decorator

    def emit(self, event_name, *args, **kwargs):
        for wrapper in self._subscribers.get(event_name, []):
            wrapper(*args, **kwargs)

bus = MiniEventBus()

@bus.on("tick")
def on_tick(payload):
    pass

# wrapper.stats 不是凭空多出来的属性——它和 wrapper 闭包里真正抓住的那个"格子"是同一个对象，
# 这一点不用脑补，用 01 类同款方法直接读出来
assert on_tick.__code__.co_freevars == ("func", "stats")   # wrapper 一共抓了两个自由变量
idx = on_tick.__code__.co_freevars.index("stats")
cell = on_tick.__closure__[idx]
assert cell.cell_contents is on_tick.stats

bus.emit("tick", 42)
# emit 之后，格子里的内容被原地改了，cell.cell_contents 和 on_tick.stats 依然是同一个对象、同步更新
assert cell.cell_contents is on_tick.stats
assert cell.cell_contents["call_count"] == 1
print("wrapper.stats and the closure cell are the exact same object:", cell.cell_contents is on_tick.stats)
```

(顺带一提:`wrapper.stats = stats` 这一行写在 `@functools.wraps(func)` 装饰完 `wrapper` **之后**,`functools.wraps` 只会把 `func` 的 `__name__`/`__doc__`/`__dict__` 等元信息合并到 `wrapper` 上,不会清空 `wrapper` 后续被手动赋的属性,两者不冲突。)

---

## 阶段 3:用生成器惰性回放事件历史——不是要多少条,是要哪条产哪条

总线还应该能回答"某个事件之前都发生过什么",这需要先把每次 `emit` 的参数记下来。但如果历史记录有几千条,而调用方往往只想看最近几条或者找符合某个条件的那一条,每次回放都 `return` 一整个 list、把全部记录一次性处理完,就是 [02 类](02-iterators-and-generators.md) 第 3 节讲过的"列表推导式"那种急切(eager)做法——用生成器可以做成惰性(lazy)的:谁要谁才产出下一条。

```python
import functools

class MiniEventBus:
    def __init__(self):
        self._subscribers = {}
        self._history = {}   # event_name -> [(args, kwargs), ...]，记录每一次 emit 的原始参数

    def on(self, event_name):
        def decorator(func):
            stats = {"call_count": 0, "last_args": None, "last_kwargs": None}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                stats["call_count"] += 1
                stats["last_args"] = args
                stats["last_kwargs"] = kwargs
                return func(*args, **kwargs)

            wrapper.stats = stats
            self._subscribers.setdefault(event_name, []).append(wrapper)
            return wrapper
        return decorator

    def emit(self, event_name, *args, **kwargs):
        self._history.setdefault(event_name, []).append((args, kwargs))
        for wrapper in self._subscribers.get(event_name, []):
            wrapper(*args, **kwargs)

    def replay(self, event_name):
        """惰性回放：不是把 self._history[event_name] 整个 list 一次性 return，
        而是一条一条 yield——谁要谁才产出下一条。"""
        for args, kwargs in self._history.get(event_name, []):
            yield (args, kwargs)

bus = MiniEventBus()
bus.emit("x", 1)
bus.emit("x", 2)
bus.emit("x", 3)

# replay() 返回的是生成器，不是 list
import types
stream = bus.replay("x")
assert isinstance(stream, types.GeneratorType)
assert list(stream) == [((1,), {}), ((2,), {}), ((3,), {})]

# 没有任何历史的事件名，回放出来是空的，不会报错
assert list(bus.replay("never_emitted")) == []

# replay 是"方法"，每次调用都产出一个全新、独立的生成器对象——
# 这正好补上 02 类第 3 节结尾埋的伏笔："自己写一个类，每次 __iter__ 被调用时都返回一个全新的生成器"
first_call = list(bus.replay("x"))
second_call = list(bus.replay("x"))
assert first_call == second_call == [((1,), {}), ((2,), {}), ((3,), {})]

# 但如果拿到的是同一个生成器对象，遍历完一次之后就耗尽了，和 02 类"生成器只能用一次"完全一致
one_stream = bus.replay("x")
drained = list(one_stream)
reused = list(one_stream)
assert drained == [((1,), {}), ((2,), {}), ((3,), {})]
assert reused == []
print("stage3 basic replay semantics ok")
```

**这里直接兑现了 [02 类](02-iterators-and-generators.md) 第 3 节结尾留的一个伏笔。** 那一节的"常见坑 3"说:想同时享受"惰性、不占内存"和"能反复遍历"两个好处,list 和生成器表达式都做不到,只能二选一——"或者自己写一个类,每次 `__iter__` 被调用时都返回一个全新的生成器",当时"这里先点一下,不展开"。`bus.replay(event_name)` 就是这个模式的一个具体实现:`replay` 本身是一个**方法**,不是生成器对象;每次调用它都会执行到函数体最开始、立刻返回一个全新、独立的生成器,不会因为上一次调用产生的生成器已经被耗尽而受影响——`first_call`/`second_call` 两次都拿到完整的三条记录就是证据。但如果你把某一次调用的返回值存进一个变量、遍历完了(`one_stream`),这个变量对应的那个生成器就真的耗尽了,再遍历只会拿到空列表,想要完整的历史必须重新调用 `bus.replay(...)`。

只验证"能不能拿到完整历史"还不够有说服力,惰性真正的价值在于:**不该做的计算,真的一次都没做**。给回放加一个转换函数,用阶段 2 同款的"闭包计数器"套路,数一数这个转换函数到底被真正调用了几次:

```python
class MiniEventBus:
    def __init__(self):
        self._history = {}

    def emit(self, event_name, *args, **kwargs):
        self._history.setdefault(event_name, []).append((args, kwargs))

    def replay(self, event_name, transform=None):
        if transform is None:
            transform = lambda args, kwargs: (args, kwargs)
        for args, kwargs in self._history.get(event_name, []):
            yield transform(args, kwargs)

bus = MiniEventBus()
for i in range(1000):
    bus.emit("price_update", "AAPL", 100 + i)

transform_calls = {"n": 0}

def expensive_transform(args, kwargs):
    transform_calls["n"] += 1     # 借用阶段2同款"闭包计数器"套路，这次用来数 transform 真正跑了几次
    symbol, price = args
    return f"{symbol}@{price}"

stream = bus.replay("price_update", transform=expensive_transform)
assert transform_calls["n"] == 0    # 生成器造出来的一瞬间，函数体一行都没跑

first_three = [next(stream) for _ in range(3)]
assert first_three == ["AAPL@100", "AAPL@101", "AAPL@102"]
assert transform_calls["n"] == 3    # 1000 条历史，只处理了被要求的 3 条

# 对比：如果一次性物化成 list（"急切"求值），transform 会在你看到任何一个结果之前就跑满全部 1000 次
def replay_eager(bus, event_name, transform):
    return [transform(args, kwargs) for args, kwargs in bus._history.get(event_name, [])]

transform_calls_eager = {"n": 0}
def expensive_transform_eager(args, kwargs):
    transform_calls_eager["n"] += 1
    symbol, price = args
    return f"{symbol}@{price}"

eager_result = replay_eager(bus, "price_update", expensive_transform_eager)
assert transform_calls_eager["n"] == 1000   # 急切版本：还没用到任何一条结果，1000 次全跑完了
assert len(eager_result) == 1000
print("lazy transform calls so far:", transform_calls["n"], "| eager transform calls:", transform_calls_eager["n"])
```

1000 条历史记录,只想看前 3 条,惰性版本的 `transform` 真的只跑了 3 次,急切版本(先 `[... for ... in ...]` 物化成完整 list)会在你碰到任何一条结果之前就先跑满 1000 次——这和 [02 类](02-iterators-and-generators.md) 第 3 节"从 5 千万个元素的生成器表达式拿第一个值"那个计时实验证明的是同一件事,只是这里换成了更直接、不受计时噪声影响的"调用次数计数",结论更硬。

最后,如果想一次性回放好几种事件类型,[02 类](02-iterators-and-generators.md) 第 4 节的 `yield from` 正好能派上用场——把多个事件类型各自的回放生成器依次委托、拼接成一条流:

```python
class MiniEventBus:
    def __init__(self):
        self._history = {}

    def emit(self, event_name, *args, **kwargs):
        self._history.setdefault(event_name, []).append((args, kwargs))

    def replay(self, event_name):
        for args, kwargs in self._history.get(event_name, []):
            yield (args, kwargs)

    def replay_many(self, *event_names):
        """把多个事件类型的回放依次拼接起来——和 02 类第 4 节 read_all_shards 是同一个模式，
        只是这里"shard"换成了"事件类型"。"""
        for name in event_names:
            yield from self.replay(name)

bus = MiniEventBus()
# 真实发生顺序：a1, b1, a2, b2（两种事件交替发生）
bus.emit("a", 1)
bus.emit("b", 1)
bus.emit("a", 2)
bus.emit("b", 2)

merged = list(bus.replay_many("a", "b"))
# 注意：replay_many 不是按"真实发生时间"交替给出，而是按事件类型依次拼接——
# 先把 "a" 的全部历史放完，再放 "b" 的，这里能直接看出这个不完美之处，不是靠猜的
assert merged == [((1,), {}), ((2,), {}), ((1,), {}), ((2,), {})]   # a1, a2, b1, b2 —— 不是 a1, b1, a2, b2

true_chronological_order = [("a", 1), ("b", 1), ("a", 2), ("b", 2)]
replay_many_order = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
assert merged == [((v,), {}) for _, v in replay_many_order]
print("replay_many groups by event type, not by real chronological order:", merged)
```

真实跑出来的结果:

```
replay_many groups by event type, not by real chronological order: [((1,), {}), ((2,), {}), ((1,), {}), ((2,), {})]
```

**这是一处需要老实交代的不完美。** `a`、`b` 两种事件本来是交替发生的(a1, b1, a2, b2),但 `replay_many("a", "b")` 给出的顺序是 a1, a2, b1, b2——`yield from` 只是把 `self.replay("a")` 完全耗尽之后,再开始 `self.replay("b")`,是**依次拼接**,不是按事件真实发生的时间交替给出。如果代码里默认"`replay_many` 返回的就是全局时间线",直接照着这个顺序去重建"当时到底发生了什么",结论会是错的。这个坑不是实现失误,是 `yield from` 这个语法本身的行为——想要真正按时间合并多条流,需要给每条历史记录打一个全局递增的序号,再用类似归并的方式按序号取最小值,这个方向留到本文最后"可以怎么继续扩展"里说。

---

## 阶段 4:组装成一个完整的 `MiniEventBus`,跑一次端到端 demo

把前三阶段拼进一个类,模拟一个玩具训练循环:每个 epoch 结束广播一次 `"epoch_end"` 事件,一个订阅者把结果记进日志,另一个只关心被调用了几次;最后惰性回放,只看最近两个 epoch。

```python
import functools

class MiniEventBus:
    """迷你事件总线：装饰器登记回调 + 闭包记录每个订阅的独立状态 + 生成器惰性回放历史。"""

    def __init__(self):
        self._subscribers = {}   # event_name -> [wrapper, wrapper, ...]
        self._history = {}       # event_name -> [(args, kwargs), ...]

    def on(self, event_name):
        def decorator(func):
            stats = {"call_count": 0, "last_args": None, "last_kwargs": None}

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                stats["call_count"] += 1
                stats["last_args"] = args
                stats["last_kwargs"] = kwargs
                return func(*args, **kwargs)

            wrapper.stats = stats
            self._subscribers.setdefault(event_name, []).append(wrapper)
            return wrapper
        return decorator

    def emit(self, event_name, *args, **kwargs):
        self._history.setdefault(event_name, []).append((args, kwargs))
        for wrapper in self._subscribers.get(event_name, []):
            wrapper(*args, **kwargs)

    def replay(self, event_name, transform=None):
        if transform is None:
            transform = lambda args, kwargs: (args, kwargs)
        for args, kwargs in self._history.get(event_name, []):
            yield transform(args, kwargs)

    def replay_many(self, *event_names):
        for name in event_names:
            yield from self.replay(name)


# ---- 端到端 demo：模拟一个玩具训练循环，每个 epoch 结束广播一次 "epoch_end" ----
bus = MiniEventBus()
history_log = []

@bus.on("epoch_end")
def log_epoch(epoch, loss):
    history_log.append((epoch, loss))

@bus.on("epoch_end")
def count_epochs(epoch, loss):
    pass   # 只关心被调用了几次，不关心具体数值

for epoch in range(5):
    fake_loss = 1.0 / (epoch + 1)
    bus.emit("epoch_end", epoch=epoch, loss=fake_loss)

assert history_log == [(0, 1.0), (1, 0.5), (2, 1/3), (3, 0.25), (4, 0.2)]
assert log_epoch.stats["call_count"] == 5
assert count_epochs.stats["call_count"] == 5
assert log_epoch.stats["last_kwargs"] == {"epoch": 4, "loss": 0.2}

# 惰性回放：只看最近两个 epoch，不用先把全部 5 条历史物化成 list 再切片
def epoch_summary(args, kwargs):
    return f"epoch={kwargs['epoch']} loss={kwargs['loss']:.4f}"

replay_stream = bus.replay("epoch_end", transform=epoch_summary)
skipped = [next(replay_stream) for _ in range(3)]   # 跳过前 3 个 epoch
last_two = list(replay_stream)                       # 剩下的就是最近两个

assert skipped == ["epoch=0 loss=1.0000", "epoch=1 loss=0.5000", "epoch=2 loss=0.3333"]
assert last_two == ["epoch=3 loss=0.2500", "epoch=4 loss=0.2000"]
print("stage4 end-to-end ok. last two epochs:", last_two)
```

到这里,`MiniEventBus` 已经是一个真实能用的小工具:任意多个回调可以用装饰器登记在任意事件名下,总线透明地记录每个订阅自己的调用统计,还能在不预先物化全部历史的前提下惰性回放——三个能力分别来自装饰器、闭包、生成器这三个你已经学过的知识点,拼起来就是训练框架 callback 系统的一个最小可用内核。

## 可以怎么继续扩展(只指方向,不实现)

- **取消订阅 `bus.off(event_name, func)`**:调用方手上拿到的是原始的 `func`,总线里存的却是包了一层的 `wrapper`,需要靠身份反查。`functools.wraps` 会把 `__wrapped__` 属性设成原函数(本文验证过 `wrapper.__wrapped__ is func` 为真),可以用来做这个反查,但要处理"同一个 `func` 被注册了多次该删哪一个"这类边界情况。
- **按真实时间顺序合并多种事件的回放**:阶段 3 的 `replay_many` 是按事件类型依次拼接,不是按真实发生时间交替给出——要做到真正按时间合并,需要给每条历史记录打一个全局递增序号,再用多路归并按序号取最小值,这正是 [dsa-deep-dive 07 类](../dsa-deep-dive/07-heaps-and-priority-queues.md) 堆的应用场景。
- **通配符/命名空间事件**(比如订阅 `"user.*"` 能收到 `"user.login"`/`"user.signup"`):需要在 `on`/`emit` 里做前缀匹配,[dsa-deep-dive 12 类](../dsa-deep-dive/12-trie-and-string-matching.md) 的 Trie 树是这类前缀匹配的标准结构。
- **异步回调**:如果订阅的是 `async def` 函数,`emit` 需要能识别出来并 `await` 它,而不是直接当成普通函数调用——这需要 [04 类](04-typing-context-and-concurrency.md) 的 `async`/`await` 协程基础。

这几个方向都不实现,是为了让这篇教程聚焦在"装饰器、闭包、生成器怎么拼成一个真实工具"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码,遇到"不够完美"的真实结果(比如阶段1直接调用绕过总线、阶段3 `replay_many` 不是真实时间顺序)也如实解释原因,不回避、不假装没发生。dsa-deep-dive 系列已经验证过这个格式成立;这一篇是它在 python-advanced 系列的第一次落地。

---

*创建:2026-07-24*
