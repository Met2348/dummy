# 01 · 函数与闭包进阶(Functions & Closures)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 6 个知识点:`lambda`、`*args`/`**kwargs`、闭包、装饰器基础、带参数的装饰器、`functools` 三件套。它们是读 PyTorch/HuggingFace 源码时最先卡住新手的一批写法——本文按"从最笨的写法讲起,再引出正式语法"的顺序讲,不是直接甩函数签名让你背。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9,torch 2.11.0+cu128,numpy 2.4.6)下实际跑通验证,不是凭空写的——其中几个"AI 研究代码里的真实例子"直接用了仓库里已经装好的 `torch.nn`/`torch.optim`,不是用假对象模拟的。

---

## 1. `lambda` 表达式

**是什么:** 用一行写出来的匿名函数——没有名字,通常传给别的函数用完就扔。

**为什么课堂不教但很重要:** Python 课只教过 `def`,而 AI 研究代码里大量出现"把一个小函数当参数传进去"的写法,比如 `sorted(data, key=lambda x: ...)`、`filter(lambda p: p.requires_grad, model.parameters())`、PyTorch 的 `torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)`。这些场合专门 `def` 一个函数、取个名字、再传过去,显得很啰嗦——`lambda` 就是为了省掉这一步。

**C 类比:** C 的 `qsort(base, n, size, cmp)` 要求你**必须先在别处定义一个具名的比较函数** `int cmp(const void *a, const void *b)`,再把函数指针传进去——C(标准 C,不算 GCC 扩展)没有"函数字面量"这种东西。Python 的 `lambda` 就是那个"不用先在别处定义、直接内联写出来"的函数字面量。

**从最笨的写法讲起:**
```python
# 笨办法:为了排序,专门 def 一个只用一次的函数
def square(x):
    return x * x

# 等价的 lambda 写法:不用起名字,内联在需要的地方
square_lambda = lambda x: x * x

assert square(5) == square_lambda(5) == 25
```
两种写法功能完全一样。区别只在于:`lambda` 不用单独占一行 `def`、不用起名字,适合"只在一个地方用一次"的场景。

**AI 研究代码里的真实例子:** 冻结部分参数后,用 `filter + lambda` 挑出可训练参数交给优化器,再用 `LambdaLR` 按 `lambda` 定义的规则衰减学习率——这是训练脚本里最常见的组合:
```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
for name, p in model.named_parameters():
    if "bias" in name:
        p.requires_grad = False   # 冻结 bias,只训练 weight

# 真实写法:用 lambda 当筛选条件,不用单独写一个具名函数
trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(trainable_params) == 1                        # 只有 weight 可训练
assert trainable_params[0].shape == model.weight.shape

optimizer = torch.optim.SGD(trainable_params, lr=0.1)
# lr_lambda 是一个 lambda:根据 epoch 数算出学习率的缩放系数
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)

lrs = []
for epoch in range(3):
    lrs.append(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()

assert lrs[0] == 0.1
assert abs(lrs[1] - 0.1 * 0.9) < 1e-9
assert abs(lrs[2] - 0.1 * 0.9 ** 2) < 1e-9
print("学习率变化:", lrs)   # [0.1, 0.09000000000000001, 0.08100000000000002]
```

**可运行例子:** 按 checkpoint 文件名里的 epoch 号排序(训练脚本里找"最新/最好的 checkpoint"的常见写法):
```python
checkpoints = ["model_epoch3.pt", "model_epoch10.pt", "model_epoch1.pt"]
sorted_ckpts = sorted(checkpoints, key=lambda name: int(name.split("epoch")[1].split(".")[0]))
assert sorted_ckpts == ["model_epoch1.pt", "model_epoch3.pt", "model_epoch10.pt"]
print(sorted_ckpts)
```

**常见坑:**

**1. `lambda` 内部只能写一个表达式,不能写语句**(赋值、`return`、`if`/`for` 完整语句都不行)。这不是风格问题,是语法层面直接报错:
```python
# 下面两行如果真的写进文件会导致整个文件 SyntaxError,这里用 compile() 单独验证报错信息
try:
    compile("f = lambda x: (x = x + 1)", "<test>", "exec")
except SyntaxError as e:
    print(e)   # invalid syntax. Maybe you meant '==' or ':=' instead of '='?

try:
    compile("f = lambda x: return x", "<test>", "exec")
except SyntaxError as e:
    print(e)   # invalid syntax
```
条件表达式(`lambda x: a if x > 0 else b`)是可以的,因为它是表达式,不是语句。

**2. 过度使用会牺牲可读性。** 逻辑稍微复杂一点(超过一行能看懂的程度),就应该老老实实 `def` 一个具名函数——研究代码里为了"炫技"把多层逻辑硬塞进一个 `lambda` 反而是减分项。

---

## 2. `*args` / `**kwargs`

**是什么:** `*args` 把函数调用时多出来的**位置参数**打包成一个 tuple;`**kwargs` 把多出来的**关键字参数**打包成一个 dict。

**为什么课堂不教但很重要:** 课堂教的函数都是"参数个数写死"。但 HuggingFace/PyTorch 的代码里,大量函数是"中间层,不关心具体传了什么参数,原样转发给下一层"——`nn.Module.__call__` 内部转发给 `forward(*args, **kwargs)`、`AutoModel.from_pretrained(name, **kwargs)` 把 `kwargs` 转给底层 config、写一个通用的计时/日志装饰器要能包住"任意签名的函数"。不用 `*args`/`**kwargs`,这类"转发层"根本没法写,因为你没法预先知道会传进来什么。

**C 类比:** C 的变参函数,比如 `printf(const char *format, ...)`,内部用 `<stdarg.h>` 的 `va_list`/`va_start`/`va_arg`/`va_end` 手动取出每一个参数——这就是 `*args` 的原型:一堆"不知道具体有几个"的位置参数。**但 `**kwargs` 在 C 里没有对应物**:C 的参数在编译期就按位置固定了,没有"运行时按名字传任意多个参数"这种机制。`**kwargs` 更接近你自己传一个 hashmap 进去,只是 Python 帮你自动打包/解包。

**从最笨的写法讲起:**
```python
def add(a, b, scale=1):
    return (a + b) * scale

# 笨办法:想写一个"调用前后打日志"的通用函数,只能显式列出 add 的每个参数
def call_and_log_dumb(func, a, b, scale):
    print(f"calling {func.__name__} with a={a} b={b} scale={scale}")
    return func(a, b, scale=scale)

result_dumb = call_and_log_dumb(add, 2, 3, 1)
assert result_dumb == 5
# 问题:如果 add 从 3 个参数变成 4 个,call_and_log_dumb 的签名也必须跟着改——
# 这个"日志包装函数"和 add 的具体签名死死耦合在一起了。

# 正式写法:用 *args/**kwargs,不管被包装的函数签名怎么变,这里都不用改
def call_and_log(func, *args, **kwargs):
    print(f"calling {func.__name__} with args={args} kwargs={kwargs}")
    return func(*args, **kwargs)

result = call_and_log(add, 2, 3, scale=10)
assert result == 50
```

**AI 研究代码里的真实例子:** `nn.Module` 的 `forward` 到处都是 `*args, **kwargs`,因为包装层(比如给模型加一层日志/计时/精度转换)不需要知道内部模型的 `forward` 具体接收什么参数,原样转发即可:
```python
import torch
import torch.nn as nn

class LoggingWrapper(nn.Module):
    """任意内部模块都能包一层,不用关心它的 forward 签名长什么样"""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        print("转发 args 形状:", [tuple(a.shape) for a in args], "kwargs 键:", list(kwargs.keys()))
        return self.inner(*args, **kwargs)

inner = nn.Linear(4, 2)
wrapped = LoggingWrapper(inner)

x = torch.randn(3, 4)
out = wrapped(x)   # 调用 nn.Module.__call__,内部再转发给 forward(*args, **kwargs)
assert out.shape == (3, 2)
```

**可运行例子:** 用 `**kwargs` 转发配置参数(类似 `from_pretrained(name, **kwargs)` 把参数转给底层 config 的写法):
```python
def configure_model(hidden_size, num_layers, dropout=0.1):
    return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}

def build_model(**kwargs):
    return configure_model(**kwargs)

cfg = build_model(hidden_size=768, num_layers=12)
assert cfg == {"hidden_size": 768, "num_layers": 12, "dropout": 0.1}
print(cfg)
```

**常见坑:**

**1. 拼写错误的报错,指向的是被调用的内层函数,不是你打错字的那一行。** 比如把 `dropout` 拼成 `droput`:
```python
def configure_model(hidden_size, num_layers, dropout=0.1):
    return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}

def build_model(**kwargs):
    return configure_model(**kwargs)

try:
    build_model(hidden_size=768, num_layers=12, droput=0.1)
except TypeError as e:
    print(e)
    # configure_model() got an unexpected keyword argument 'droput'. Did you mean 'dropout'?
```
较新版本的 Python(本机 3.13.9)会贴心地猜"你是不是想打 `dropout`",但报错说的是 `configure_model()`(被转发到的那个函数),不是 `build_model()`(你实际调用的那一层)——调用链每多包一层 `**kwargs`,排查起来就多一层间接。

**2. 更隐蔽的版本:中间层如果自己也有 `**kwargs` 兜底,拼写错误可能被"悄悄吃掉",完全不报错。**
```python
def train(learning_rate=1e-3, batch_size=32):
    return learning_rate

def launch_experiment(batch_size=32, **kwargs):
    # 疏忽:忘记把 **kwargs 转发给 train
    return train(batch_size=batch_size)

result = launch_experiment(learing_rate=1e-4, batch_size=64)  # 少拼了个 n
assert result == 1e-3   # 静默地退回默认学习率,没有任何报错或警告!
print(result)
```
这是 `**kwargs` 最危险的坑:因为 `launch_experiment` 有 `**kwargs` 接住了所有"未预期"的参数却没有转发下去,拼错的 `learing_rate` 被无声无息地丢弃,训练脚本会用错误的默认学习率跑完全程,而你毫无察觉。**这也是为什么写"转发层"函数时,`**kwargs` 一旦收了就要记得转发出去,不能半路截断。**

---

## 3. 闭包(closure)——函数记住外部变量

**是什么:** 一个函数"记住"了它被定义时所在作用域里的变量——哪怕外层函数已经执行结束、局部变量按理说该被销毁了,内层函数依然能访问到它。

**为什么课堂不教但很重要:** C 里没有闭包这个概念。C 的回调函数如果要"带状态",必须手动多传一个参数,比如 `pthread_create(&t, NULL, thread_func, (void *)arg)` 里的 `arg`、`qsort_r` 的额外上下文参数、GTK 的 `g_signal_connect(obj, "clicked", callback, user_data)` 里的 `user_data`——这些 `void *` 参数就是程序员手动实现的"闭包捕获"。Python 的闭包是语言自动帮你做了这件事。研究代码里,`model.register_forward_hook(hook_fn)` 捕获中间激活值、构造一个"记住了某个超参数"的 loss 函数、给多个独立的计数器/累加器,都是闭包的应用。

**从最笨的写法讲起:**
```python
# 笨办法:用全局变量模拟"函数记住状态"
_count = 0
def increment_global():
    global _count
    _count += 1
    return _count

assert increment_global() == 1
assert increment_global() == 2
# 问题:如果我想要两个互相独立的计数器,全局变量没法区分——
# 只能再开一个不同名字的全局变量,没法"批量生产"独立的计数器。

# 正式写法:闭包——外层函数每调用一次,就产出一个"记住"了自己那份 count 的新函数
def make_counter():
    count = 0
    def increment():
        nonlocal count      # 声明:这个 count 用外层的,不是新建局部变量
        count += 1
        return count
    return increment

counter = make_counter()
assert counter() == 1
assert counter() == 2
assert counter() == 3        # count 被"记住"了,一直在累加
```

**"记住"这两个字具体在说什么,一步步拆开看:** `count` 不是被"复制"进 `increment` 里的。`make_counter()` 每执行一次,就在内存里开一个只属于**这一次调用**的"格子"存 `count`;`increment` 记住的是**这个格子的引用**,不是格子里当时的值。正常情况下,`make_counter()` 函数体跑完之后,它的局部变量应该被销毁——但因为 `increment` 还"抓着"这个格子不放,Python 就不会回收它。这正是"闭包"这个名字的含义:**内层函数把外层的变量"包"在自己身上,带着一起走。**

| 步骤 | 发生了什么 | `count` 这个格子里的值 | `counter` 指向谁 |
|---|---|---|---|
| `counter = make_counter()` | `make_counter()` 执行一次:新建一个"格子"存 `count = 0`;定义 `increment`,它的闭包抓住了这个格子的引用;函数返回 `increment`,格子因为被抓住而没有被销毁 | `0` | `increment`(绑定着这一次调用产生的格子) |
| 第 1 次 `counter()` | 执行到 `nonlocal count; count += 1`——`nonlocal` 声明"这里改的就是外层那个格子",不是新建一个局部变量 | `1` | 同一个 `increment`,格子被原地改了 |
| 第 2 次 `counter()` | 同上,再改一次同一个格子 | `2` | 同一个 `increment` |
| 第 3 次 `counter()` | 同上 | `3` | 同一个 `increment` |

这不是靠脑补的比喻——Python 真的把这个"格子"实现成了一个具体对象,叫 `cell`,可以从函数对象的 `__closure__` 属性上直接读出来,眼见为实:

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter = make_counter()
cell = counter.__closure__[0]     # increment 只抓了一个自由变量,所以这个 tuple 里只有一个 cell
assert cell.cell_contents == 0    # 还没调用过,格子里是初始值 0

counter()
assert cell.cell_contents == 1    # 调用之后,格子里的值被原地改了——从函数对象外部也能读到这次修改

# 关键点:格子是"这一次 make_counter() 调用"独有的,不是所有 increment 共享同一个格子
counter_a = make_counter()
counter_b = make_counter()
counter_a()
cell_a = counter_a.__closure__[0]
cell_b = counter_b.__closure__[0]
assert cell_a is not cell_b       # 两次 make_counter() 调用,产生的是两个不同的格子
assert cell_a.cell_contents == 1 and cell_b.cell_contents == 0   # 互不干扰
print("两个 counter 各自抓着自己的格子,互不影响:", cell_a.cell_contents, cell_b.cell_contents)
```
下面"可运行例子"里 `counter_a`/`counter_b` 互不干扰,根本原因就是这张表 + 这段 `cell` 验证:每次调用 `make_counter()` 都会新开一个格子,两个 `increment` 分别抓住自己的那一个,不会串。

**AI 研究代码里的真实例子:** 用闭包工厂生成 `forward hook`,捕获指定层的中间激活值——这是 PyTorch 里用 `register_forward_hook` 调试/可视化中间层输出的标准写法:
```python
import torch
import torch.nn as nn

def make_hook(storage, name):
    """闭包工厂:每调用一次,产出一个"记得"自己 name 的 hook 函数"""
    def hook(module, inputs, output):
        storage[name] = output.detach().clone()
    return hook

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
activations = {}
model[1].register_forward_hook(make_hook(activations, "after_relu"))

x = torch.randn(2, 4)
out = model(x)

assert "after_relu" in activations
assert activations["after_relu"].shape == (2, 8)
assert (activations["after_relu"] >= 0).all()   # ReLU 之后应该全部非负
print("捕获到的激活值 shape:", tuple(activations["after_relu"].shape))
```

**可运行例子:** 闭包能"批量生产"互相独立的实例,这正是笨办法里全局变量做不到的事:
```python
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter_a = make_counter()
counter_b = make_counter()

assert counter_a() == 1
assert counter_a() == 2
assert counter_b() == 1     # 完全独立,不受 counter_a 的调用次数影响
print("counter_a 走到 2,counter_b 还在 1,两者互不干扰")
```

**常见坑:**

**1. 经典坑:闭包捕获的是变量本身(引用),不是创建那一刻的值。** 用上面的"格子"类比:`for` 循环里的循环变量 `i` 全程只有**一个格子**,不是每轮循环各开一个新格子——所以循环里创建多个 `lambda`,它们抓住的是同一个格子,循环结束后这个格子停在最后一个值,所有闭包读到的都是这个最终值:
```python
funcs = []
for i in range(3):
    funcs.append(lambda: i)

results = [f() for f in funcs]
print(results)          # 直觉可能觉得是 [0, 1, 2],实际是:
assert results == [2, 2, 2]   # 三个 lambda 全部返回循环结束时 i 的最终值!
```
**修复方法:用默认参数,在定义 `lambda` 那一刻就把当前值"冻结"进函数**(默认参数在函数定义时求值一次,之后不再变化):
```python
funcs_fixed = []
for i in range(3):
    funcs_fixed.append(lambda i=i: i)   # i=i:右边的 i 在定义时求值,左边的 i 是新的局部参数

results_fixed = [f() for f in funcs_fixed]
assert results_fixed == [0, 1, 2]
print(results_fixed)
```

**2. 忘记 `nonlocal` 会直接报错,而不是"悄悄"用了外层变量。** 在闭包内给捕获的变量赋值(不只是读取),必须声明 `nonlocal`,否则 Python 会认为你在函数内新建了一个同名局部变量,而这个局部变量在赋值语句执行前还没有值:
```python
def make_counter_broken():
    count = 0
    def increment():
        count += 1     # 没有 nonlocal!Python 把这一行的 count 当成新的局部变量
        return count
    return increment

c = make_counter_broken()
try:
    c()
except UnboundLocalError as e:
    print(e)   # cannot access local variable 'count' where it is not associated with a value
```
**只读不改**(比如上面 `make_hook` 里只是把 `output` 存进 `storage[name]`,没有对 `storage`/`name` 本身重新赋值)是不需要 `nonlocal` 的——只有**重新赋值**捕获的变量本身时才需要。

---

## 4. 装饰器基础——`@decorator` 语法糖

**是什么:** `@decorator` 是 `func = decorator(func)` 的语法糖——本质是"用一个新函数替换掉旧函数",新函数在调用原函数前后可以插入别的逻辑。

**为什么课堂不教但很重要:** PyTorch/HuggingFace 代码里 `@torch.no_grad()`、`@dataclass`(第 03 批会讲)之类的 `@xxx` 写法到处都是。不理解"`@` 到底把函数变成了什么",就只能把它当成看不懂的魔法照抄。

**从最笨的写法讲起:**
```python
# 第一步:不用 @ 语法糖,手动包裹
def train_step():
    print("跑一个训练 step")
    return "done"

def logging_wrapper(func):
    def wrapper():
        print(f"[LOG] 开始调用 {func.__name__}")
        result = func()
        print(f"[LOG] {func.__name__} 调用结束")
        return result
    return wrapper

train_step = logging_wrapper(train_step)   # 手动包裹:这一行就是装饰器的本质
result = train_step()
assert result == "done"

# 第二步:@ 语法糖,和上面完全等价
def logging_wrapper2(func):
    def wrapper(*args, **kwargs):     # 用上一节的 *args/**kwargs,让装饰器能包裹任意签名的函数
        print(f"[LOG] 开始调用 {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} 调用结束")
        return result
    return wrapper

@logging_wrapper2        # 等价于:train_step2 = logging_wrapper2(train_step2)
def train_step2(batch_id):
    return batch_id * 2

r2 = train_step2(5)
assert r2 == 10
```
`@logging_wrapper2` 和手写的 `train_step = logging_wrapper(train_step)` 做的是同一件事,只是 `@` 把"重新赋值"这一步挪到了函数定义的正上方,读起来更顺。

**AI 研究代码里的真实例子:** `@torch.no_grad()` 是研究代码里最常见的装饰器之一——它会真的改变函数内部张量的 autograd 行为,不只是加个日志:
```python
import torch

w = torch.randn(3, 3, requires_grad=True)

def compute_with_grad(w):
    return (w * 2).sum()

@torch.no_grad()
def compute_without_grad(w):
    return (w * 2).sum()

y1 = compute_with_grad(w)
y2 = compute_without_grad(w)

assert y1.requires_grad is True     # 正常计算,记录了计算图,可以反向传播
assert y2.requires_grad is False    # no_grad 装饰后,不记录计算图,省显存、跑得快
print("y1.requires_grad =", y1.requires_grad, " y2.requires_grad =", y2.requires_grad)
```
推理/验证阶段用 `@torch.no_grad()` 包住整个函数,是训练脚本里几乎必写的优化——原理会在 [04-typing-context-and-concurrency.md](04-typing-context-and-concurrency.md) 讲 `with` 语句和上下文管理器时深入展开(`torch.no_grad` 同时是上下文管理器和装饰器)。

**可运行例子:** 一个计时装饰器,训练脚本里常用来测量某个 step 的耗时:
```python
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时 {elapsed:.6f}s")
        return result
    return wrapper

@timeit
def fake_train_one_step(batch_size):
    return sum(range(batch_size))

result = fake_train_one_step(1000)
assert result == sum(range(1000))
```

**常见坑:**

**1. 忘记在装饰器内部 `return wrapper`,装饰后的函数会变成 `None`。**
```python
def broken_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    # 忘记 return wrapper!

@broken_decorator
def foo():
    return 1

print(foo)      # None —— foo 已经不是函数了
try:
    foo()
except TypeError as e:
    print(e)    # 'NoneType' object is not callable
```

**2. 装饰后函数的 `__name__`/`__doc__` 会丢失。** 手写的 `logging_wrapper` 包裹之后,函数的身份信息会被 `wrapper` 顶替——这个问题下一节 `functools.wraps` 专门解决,这里先验证一下问题确实存在:
```python
def logging_wrapper(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@logging_wrapper
def train_step():
    """跑一个训练 step"""
    return "done"

assert train_step.__name__ == "wrapper"    # 不是 "train_step"!
assert train_step.__doc__ is None          # docstring 也丢了
print(train_step.__name__, train_step.__doc__)
```

---

## 5. 带参数的装饰器——装饰器工厂

**是什么:** 当装饰器本身也需要配置参数时(比如"重试几次"“缓存多大”),要在普通装饰器的外面再包一层——写一个"调用后返回装饰器"的函数,俗称装饰器工厂。

**为什么课堂不教但很重要:** `@retry(times=3)`、`functools.lru_cache(maxsize=128)`,以及 fairseq/detectron2/torchvision 这类研究代码库里 `@register_model("resnet50")` 式的注册写法,都是"给装饰器传参数"。不理解这里的三层函数嵌套,只会觉得这些写法"看起来像装饰器又不太一样",没法真正看懂。

**从最笨的写法讲起:**
```python
# 笨办法:普通装饰器没法接收配置参数,每种配置只能复制一份、改个数字
def retry_3_times(func):
    def wrapper(*args, **kwargs):
        for attempt in range(1, 4):
            try:
                return func(*args, **kwargs)
            except ValueError:
                if attempt == 3:
                    raise
        return None
    return wrapper

def retry_5_times(func):   # 复制粘贴,只改了数字 —— 典型的笨办法信号
    def wrapper(*args, **kwargs):
        for attempt in range(1, 6):
            try:
                return func(*args, **kwargs)
            except ValueError:
                if attempt == 5:
                    raise
        return None
    return wrapper
# 问题:想要 4 次、10 次重试,就要再复制一份——次数应该是"参数",不是"写死在函数名里"

# 正式写法:装饰器工厂——外层函数接收配置,中层函数才是真正的装饰器
def retry(times=3, delay=0.01):
    """最外层:接收装饰器的配置参数"""
    def decorator(func):
        """中间层:接收被装饰的函数,和普通装饰器一样"""
        def wrapper(*args, **kwargs):
            """最内层:真正替换掉原函数的部分"""
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except ValueError as e:
                    last_exc = e
                    print(f"第 {attempt} 次尝试失败: {e}")
            raise last_exc
        return wrapper
    return decorator
```
三层的分工:`retry(times=3)` 先执行,返回 `decorator`;`@decorator` 再去装饰 `func`,返回 `wrapper`。所以 `@retry(times=3)` 展开等价于 `func = retry(times=3)(func)`——比普通装饰器 `func = decorator(func)` 正好多了一层调用。

**AI 研究代码里的真实例子:** 模型注册表(model registry)是 fairseq、detectron2、torchvision 等仓库的标配写法——用带参数的装饰器把类注册进一个全局表,之后就能"用字符串名字构造模型",不用写一长串 `if name == "resnet50": ... elif name == "vit_base": ...`:
```python
MODEL_REGISTRY = {}

def register_model(name):
    """装饰器工厂:把类注册进全局表,同时不改变类本身"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register_model("resnet50")
class ResNet50:
    pass

@register_model("vit_base")
class ViTBase:
    pass

assert MODEL_REGISTRY["resnet50"] is ResNet50
assert MODEL_REGISTRY["vit_base"] is ViTBase

# 之后可以按名字构造模型,不用硬编码 import + if/elif 链
model = MODEL_REGISTRY["resnet50"]()
assert isinstance(model, ResNet50)
print("注册表:", list(MODEL_REGISTRY.keys()))
```

**可运行例子:** 用 `retry(times=3)` 装饰一个"前两次失败、第三次成功"的函数(模拟调用会限流/偶尔失败的 API):
```python
import time

def retry(times=3, delay=0.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except ValueError as e:
                    last_exc = e
                    print(f"第 {attempt} 次尝试失败: {e}")
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator

call_count = {"n": 0}

@retry(times=3)
def flaky_api_call():
    call_count["n"] += 1
    if call_count["n"] < 3:
        raise ValueError("模拟网络错误/限流")
    return "成功"

result = flaky_api_call()
assert result == "成功"
assert call_count["n"] == 3
print("总共尝试次数:", call_count["n"])
```

**常见坑:** 忘记加括号——`@retry` 写成不带调用的形式,而不是 `@retry()`。这不会报语法错误,但含义完全变了——`retry` 本身直接被当成了装饰器,于是被装饰的函数 `func` 变成了 `times` 参数的值:
```python
def retry(times=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry            # 少了一层 ()!
def bar():
    return 42

print(bar)        # <function retry.<locals>.decorator at ...> —— bar 变成了 decorator 函数,不是能正常工作的 bar
try:
    bar()
except TypeError as e:
    print(e)      # retry.<locals>.decorator() missing 1 required positional argument: 'func'
```
判断口诀:**装饰器工厂永远要"调用一次"才能得到真正的装饰器**——没有配置参数需要传时,也要写成 `@retry()` 而不是 `@retry`(除非从一开始就设计成普通装饰器,不走工厂模式)。

---

## 6. `functools` 三件套:`wraps` / `partial` / `lru_cache`

### 6.1 `functools.wraps`

**是什么:** 一个专门给装饰器用的装饰器,把被装饰函数的 `__name__`/`__doc__`/`__module__` 等元信息,复制到包装函数上。

**为什么课堂不教但很重要:** 上一节埋的伏笔在这里补上——手写装饰器时,`wrapper` 会"顶替"原函数的身份,导致 `func.__name__` 变成 `"wrapper"`、`func.__doc__` 变成 `None`。这在调试、打日志、自动生成文档时都会造成困惑:日志里打印的函数名千篇一律都是 `wrapper`,完全看不出是哪个函数出的问题。

**从最笨的写法讲起:**
```python
# 笨办法:手动一行行拷贝元信息,还容易漏掉字段
def my_decorator_manual(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# 正式写法:functools.wraps 一行搞定,而且复制的字段更全(包括 __module__、__wrapped__ 等)
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

**AI 研究代码里的真实例子:** 给上一节的计时装饰器补上 `functools.wraps`,这是研究代码里"写装饰器"的标准模板——几乎所有正式的装饰器都会加这一行:
```python
import functools

def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def train():
    """训练一个 epoch,返回本 epoch 的平均 loss"""
    return 0.5

print("不用 wraps:", train.__name__, "|", train.__doc__)
assert train.__name__ == "wrapper"      # 元信息丢失
assert train.__doc__ is None

def my_decorator_fixed(func):
    @functools.wraps(func)              # 关键一行
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator_fixed
def train2():
    """训练一个 epoch,返回本 epoch 的平均 loss"""
    return 0.5

assert train2.__name__ == "train2"
assert train2.__doc__ == "训练一个 epoch,返回本 epoch 的平均 loss"
print("用 wraps 后:", train2.__name__, "|", train2.__doc__)
```

**常见坑:** 不是所有代码都会因为丢了 `__name__` 直接报错——大部分情况下程序照样能跑,只是**调试、日志、自动化工具(比如根据函数名做分发/查找的注册表机制)会悄悄出错或者提示信息变得毫无意义**,这种"能跑但不对"的坑比直接报错更难排查。所以规矩很简单:**只要写装饰器,就默认加 `functools.wraps`,不要等出问题了才想起来补。**

---

### 6.2 `functools.partial`

**是什么:** 提前固定一个函数的部分参数,生成一个"参数更少"的新函数。

**为什么课堂不教但很重要:** 多进程场景下(比如 `multiprocessing.Pool.map` 或 `ProcessPoolExecutor.map`),worker 函数往往只能接收"一个"参数,但实际逻辑需要好几个超参数——训练脚本里用 `functools.partial` 提前把超参数固定进去,再把"看起来只有一个参数"的函数传给多进程 API,是标准套路。PyTorch 的算子调用、`DataLoader` 的 `collate_fn` 绑定 `tokenizer`,也经常这么写。

**从最笨的写法讲起:**
```python
def scale_and_shift(x, scale, shift):
    return x * scale + shift

# 笨办法:想要一个"固定了 scale=2, shift=1"的新函数,只能手写一个新函数
def double_plus_one_dumb(x):
    return scale_and_shift(x, scale=2, shift=1)

assert double_plus_one_dumb(5) == 11

# 正式写法:functools.partial 直接从已有函数"派生"出新函数,不用手写
import functools

double_plus_one = functools.partial(scale_and_shift, scale=2, shift=1)
assert double_plus_one(5) == 11
```
`double_plus_one_dumb` 和 `double_plus_one` 效果完全一样,但 `partial` 不用为每一组固定参数手写一个新函数。

**AI 研究代码里的真实例子:** 用 `partial` 固定卷积的 `stride`/`padding`,生成一个"更简单"的卷积函数——这是需要多次用同一组超参数调用底层算子时的常见写法:
```python
import functools
import torch
import torch.nn.functional as F

conv_stride2 = functools.partial(F.conv2d, stride=2, padding=1)

x = torch.randn(1, 3, 8, 8)
weight = torch.randn(6, 3, 3, 3)
out = conv_stride2(x, weight)
assert out.shape == (1, 6, 4, 4)
print("out.shape =", tuple(out.shape))
```

**可运行例子:** 多进程 worker 场景(这里用 `map` 模拟 `Pool.map` 的调用方式,避免教程环境里真开进程的额外复杂度):
```python
import functools

def scale_and_shift(x, scale, shift):
    return x * scale + shift

worker_fn = functools.partial(scale_and_shift, scale=0.5, shift=0)
results = list(map(worker_fn, [10, 20, 30]))   # 相当于 pool.map(worker_fn, data_chunks)
assert results == [5.0, 10.0, 15.0]
print(results)
```

**常见坑:** `partial` 固定的关键字参数,和后续调用时的**位置参数**可能冲突——直觉容易出错:
```python
import functools

def f(a, b, c):
    return (a, b, c)

g = functools.partial(f, b=2)   # 把 b 固定成关键字参数

try:
    g(1, 3)     # 直觉:a=1, c=3, b 用固定的 2 —— 但实际会报错!
except TypeError as e:
    print(e)    # f() got multiple values for argument 'b'

# 原因:g(1, 3) 等价于 f(1, 3, b=2)。位置参数 1, 3 依次填给 a, b,
# 于是 b 同时被"位置参数 3"和"partial 固定的关键字 b=2"赋值,冲突。
ok = g(1, c=3)   # 正确用法:固定过关键字参数后,调用时避开对应位置,或者全用关键字传参
assert ok == (1, 2, 3)
print(ok)
```

---

### 6.3 `functools.lru_cache`

**是什么:** 给函数自动加缓存——相同参数第二次调用时直接返回上次的结果,不重新计算。`lru` = least recently used,缓存满了会淘汰最久没用的结果。

**为什么课堂不教但很重要:** 研究代码里经常有"传入参数一样、结果就一定一样"的纯函数,而且这个函数计算量不小、还会被反复调用同样的参数(比如按固定的 `seq_len`/`d_model` 计算 position encoding 表)——`lru_cache` 一行装饰器就能避免重复计算,不用自己维护缓存字典。

**从最笨的写法讲起:**
```python
# 笨办法:手写字典做缓存,自己管理"查缓存 -> 没有就算 -> 存进去"三步
manual_cache = {}
calls = {"n": 0}

def fib_manual_cache(n):
    if n in manual_cache:
        return manual_cache[n]
    calls["n"] += 1
    result = n if n < 2 else fib_manual_cache(n - 1) + fib_manual_cache(n - 2)
    manual_cache[n] = result
    return result

assert fib_manual_cache(10) == 55

# 正式写法:functools.lru_cache 一行装饰器,效果一样,不用自己管字典
import functools

@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

assert fib(10) == 55
```

**AI 研究代码里的真实例子:** 缓存按 `(seq_len, d_model)` 计算出来的 Transformer 位置编码表——同一组超参数下,这张表在训练全程是固定的,不需要每次 forward 都重新算一遍三角函数:
```python
import functools
import numpy as np

calls = {"n": 0}

@functools.lru_cache(maxsize=None)
def get_positional_encoding(seq_len, d_model):
    calls["n"] += 1
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

pe1 = get_positional_encoding(128, 64)
pe2 = get_positional_encoding(128, 64)   # 同样的 (seq_len, d_model),应该命中缓存

assert calls["n"] == 1        # 底层计算只跑了一次
assert pe1 is pe2             # 直接返回同一个对象,不是重新算出来的副本
print("底层计算次数:", calls["n"])
```

**可运行例子:** 用 `cache_info()` 直接看命中情况(调试缓存是否生效时很有用):
```python
import functools
import numpy as np

@functools.lru_cache(maxsize=None)
def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

get_positional_encoding(128, 64)   # 第 1 次调用:miss,真的算了一遍
get_positional_encoding(128, 64)   # 第 2 次调用:hit,直接用缓存

info = get_positional_encoding.cache_info()
print(info)   # CacheInfo(hits=1, misses=1, maxsize=None, currsize=1)
assert info.hits >= 1
assert info.misses == 1
```

**常见坑:** `lru_cache` 靠参数当字典的 key,**参数必须可哈希**——传 `list`/`dict`/`np.ndarray` 这类可变对象会直接报错,而不是"缓存不生效"这么温和:
```python
import functools

@functools.lru_cache(maxsize=None)
def process(data):
    return sum(data)

try:
    process([1, 2, 3])
except TypeError as e:
    print(e)   # unhashable type: 'list'

ok = process((1, 2, 3))   # 换成 tuple(可哈希)就正常了
print(ok)
```
研究代码里想缓存"输入是一个数组"的函数时,常见的规避方式是缓存的参数只用 `shape`/`seq_len` 这类可哈希的标量(就像上面的位置编码例子),而不是直接把数组传给带 `lru_cache` 的函数。

---

## 小结:这一批 6 个知识点解决的问题

| 知识点 | 解决的问题 |
|---|---|
| `lambda` 表达式 | 一次性用的小函数,不用专门起名字、占一整块 `def` |
| `*args`/`**kwargs` | 让函数能接收"数量、名字都不确定"的参数,并原样转发给下一层 |
| 闭包(closure) | 函数记住定义时所在环境里的变量,不用全局变量也能"带状态" |
| 装饰器基础 `@decorator` | 用统一语法在函数外面"包一层逻辑",不用每次手动重新赋值 |
| 带参数的装饰器(装饰器工厂) | 让"包一层逻辑"本身也能配置(次数、名字、开关等) |
| `functools.wraps` | 装饰后保留原函数的 `__name__`/`__doc__`,不破坏调试和内省 |
| `functools.partial` | 提前固定部分参数,派生出一个"参数更少"的新函数 |
| `functools.lru_cache` | 自动缓存重复调用的结果,避免对相同参数重复计算 |

下一批:[02-iterators-and-generators.md](02-iterators-and-generators.md)

---

*更新:2026-07-07*
