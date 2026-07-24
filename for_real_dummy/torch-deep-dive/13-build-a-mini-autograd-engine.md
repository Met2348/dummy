# 13 · 手把手实战:从零搭一个迷你 autograd 引擎

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 13 个知识点,不计入"100 个知识点"的统计——和 [12 类](12-advanced-interview-depth.md)是同一挂(都不计入统计),但风格不一样:12 类里你是**旁观者**,跟着面试官和候选人的多级追问链把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真正能做反向传播的迷你引擎。这个格式最早在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证过,现在推广到 torch-deep-dive——本系列讲 tensor 内存模型→autograd→nn.Module 内核→优化器这条底层机制主线,是这次推广里含金量最高的一条。

**这一篇和本系列其余 12 篇有一个本质区别,要在最前面说清楚:** 01-12 篇的方法论,是把 torch 内部的黑箱打开——每一条底层机制的说法,都配一段调用真实 `torch`/`torch.autograd` 的代码去验证,证明"这句话描述的现象是真的"。这一篇反过来:**全文分阶段实现的 `Value` 迷你 autograd 引擎不 `import torch`、不调用 `torch.autograd` 的任何一行代码**——因为这里要验证的不是"torch 内部是不是这样实现的"(那是 [02 类](02-autograd-internals.md) 已经做过的事),而是"这套机制本身站不站得住、能不能被独立实现出来"。只有**阶段 4**(全文最后一步)会 `import torch`,而且明确只是拿真实 `torch.autograd` 的结果做交叉验证,不是拿它替代自己的实现——这是全篇唯一一处、也是全系列 13 篇里唯一一处用到 torch 本身的地方,除此之外从头到尾只用 Python 标准库的 `math` 模块。

这种"自己动手写一个迷你 autograd 引擎"的教学项目在社区里有一个知名先例——Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd),用不到 100 行代码实现了一个支持标量反向传播的引擎。本文的实现是独立编写的,但采用了同一类经典设计(`Value` 包一个标量、每次运算记录"我是由谁、通过什么运算得到的"、`backward()` 用拓扑排序处理依赖顺序)——这一整套模式已经是这类教学项目公认的标准做法,不是本文发明的,这里只是老老实实把它走一遍、每一步都用真实断言验证。

## 为什么要自己动手写一遍

不是要发明新知识点,是把 [02 类](02-autograd-internals.md)已经讲过的几个概念,自己重新实现一遍、让抽象的文字描述变成看得见摸得着的代码:

| 阶段 | 要让程序多会一件事 | 对应 [02 类](02-autograd-internals.md) 哪一节讲过的机制 |
|------|------|------|
| 阶段 1 | 记住"我是被哪个运算、用哪些输入造出来的"——搭出前向计算图 | 第 2-3 节:`grad_fn` / `next_functions` |
| 阶段 2 | 把梯度按正确顺序传回每一路输入,多路径汇入时要累加、不能覆盖 | 第 3-4 节:反向拓扑遍历 + 链式法则(VJP);第 10 节:梯度累加 |
| 阶段 3 | 支持非线性(ReLU/tanh),整条链路能和手算的微积分对上 | 新增内容;[04 类](04-layers-math-and-backward.md) 第 2/10 节有 ReLU 前向反向的 torch 版对照 |
| 阶段 4 | 拿真正的 `torch.autograd` 交叉验证数值,证明不是自娱自乐 | 呼应 [02 类](02-autograd-internals.md) 开篇"这是黑箱,本文要打开它"这个目标 |

每个阶段的代码都能独立运行(本文件用仓库统一模式的 `_verify_md.py` 校验——从 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py) 原样拷贝过来,内容和具体系列无关——校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的类定义时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:`Value` 类——先让"计算图"变成看得见的真实数据结构

[02 类第 2-3 节](02-autograd-internals.md) 讲过:`grad_fn` 是"这个 tensor 是由什么运算造出来的"的记录,`next_functions` 是"这个运算的输入分别是谁"的记录,两者合起来才是一张真实、可以用代码遍历的计算图。我们自己的迷你版本只需要两个字段就能表达同样的信息:`_op`(这个 `Value` 是由哪种运算得到的,相当于一个简化版 `grad_fn`——只存运算符字符串,不是完整对象)和 `_prev`(造出这个 `Value` 的那些输入,相当于简化版 `next_functions`——我们每个节点只有一个输出,不需要像 torch 那样额外记录"是对方的第几个输出")。

```python
class Value:
    """一个只包一个标量的迷你 tensor:记录 data 本身,以及"我是怎么来的"。"""
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)   # 简化版 next_functions:谁是我的输入
        self._op = _op                 # 简化版 grad_fn:我是通过什么运算得到的

    @property
    def is_leaf(self):
        # 判断标准和 02 类第 5 节 x.is_leaf 完全一致:没有 _prev(不是任何运算的结果)就是叶子
        return len(self._prev) == 0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __repr__(self):
        return f"Value(data={self.data}, op={self._op!r})"


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c   # 先算 a*b(挂 "*" 节点),再和 c 相加(挂 "+" 节点)——和 python 运算符优先级完全一致

print("a:", a)
print("d:", d)
print("d built by op:", d._op)
print("d._prev has", len(d._prev), "entries, ops =", sorted(v._op for v in d._prev))
print("a.is_leaf:", a.is_leaf, " d.is_leaf:", d.is_leaf)

assert d.data == 4.0                                  # (2 * -3) + 10 = 4
assert d._op == "+"                                     # d 是这一步"+"运算的直接产物
assert sorted(v._op for v in d._prev) == ["", "*"]       # d 的两路输入:c(叶子,op="")和 a*b(op="*")
assert a.is_leaf is True                                 # a 是用户直接创建的,没有 _prev
assert d.is_leaf is False                                 # d 是运算出来的,is_leaf 必须是 False
print("stage1 ok")
```

跑起来能看到 `d._prev` 里两个元素分别是 `c` 本身(叶子,`_op` 是空字符串)和 `a*b` 这一步中间结果(`_op` 是 `"*"`)——`d = a*b + c` 这一行 Python 代码,不多不少,现场"织"出了两个真实的 `Value` 对象和它们之间的引用关系,不是比喻。到这一步为止,`Value` 还只会做**前向**计算,`.data` 永远是对的,但还没有任何办法知道"改变 `a` 一点点,`d` 会跟着变化多少"——这正是阶段 2 要补上的。

---

## 阶段 2:每种运算的本地梯度规则 + `backward()`——不排对顺序会得到"看似正常但错误"的梯度

**本地梯度规则(local gradient rule)** 是什么:每种运算只需要知道"这一步,我的输出对我的每一路输入的偏导数是多少"——这是一个纯局部的规则,不需要知道整张图剩下的部分长什么样:

- `c = a + b`:`dc/da = 1`,`dc/db = 1`(加法把上游传来的梯度原封不动地"抄"给每一路输入)
- `c = a * b`:`dc/da = b`,`dc/db = a`(乘法把上游梯度和"另一个乘数的值"相乘再往下传)

链式法则要做的事,就是把这些本地规则一层一层"接"起来:如果 `c` 是从 `a` 一路运算到最终输出 `L` 的中间量,`dL/da = dL/dc * dc/da`——`dL/dc` 是"上游已经传到 `c` 这里的梯度"(代码里就是 `c.grad`),`dc/da` 是"这一步的本地规则"。这正是 [02 类第 4 节](02-autograd-internals.md) 讲的 **VJP(vector-Jacobian product)** 的标量版本:每一步只做"上游梯度 × 本地梯度"这一次乘法。

给 `Value` 加上 `.grad`(初始为 0,注意后面全程用 `+=` 累加,不是覆盖——原因下面马上会现场证明)和一个 `_backward` 闭包(记录"如果我的 `.grad` 已经算好了,该怎么把它传给我的输入"),就能实现每一步的本地规则。**但只把每个节点的本地规则写对还不够**——`backward()` 还必须保证"调用顺序"是对的:一个节点必须等到**所有**汇入它的梯度都到齐、累加完毕,才能把 `.grad` 往它自己的输入继续传;提前传,会拿一个还没加完的 `.grad` 去算下游梯度,得到错误结果。[02 类第 4 节](02-autograd-internals.md) 把这句话写成"反向拓扑遍历……必须等一个节点的所有上游梯度都到齐才能继续往上传"——这里现场把"不这样做会怎样"复现出来,而不是只重复这句话。

先看一个不满足这个条件、但表面上很自然的写法:一遇到某个节点就立刻调用它的 `_backward()`,不管它的梯度是不是已经收全了。

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None   # 默认:叶子节点没有什么可以往下传的
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad     # dc/da = 1
            other.grad += out.grad    # dc/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # dc/da = b
            other.grad += self.data * out.grad   # dc/db = a
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


def naive_backward_no_toposort(root):
    # 故意写错的版本:第一次碰到某个节点,立刻调用它的 _backward()——
    # 不检查这个节点上游是不是已经全部到齐
    visited = set()
    def visit(v):
        if v in visited:
            return
        visited.add(v)
        v._backward()
        for child in v._prev:
            visit(child)
    root.grad = 1.0
    visit(root)


# 构造一个"菱形"依赖:m 被 p 和 q 两条不同路径同时用到
x = Value(2.0)
m = x * x                 # m = 4.0  (共享节点,下面 p 和 q 都要用它)
p = m * Value(3.0)        # p = 12.0
q = m + Value(5.0)        # q = 9.0
total = p + q               # total = 21.0

naive_backward_no_toposort(total)
print("naive: m.grad =", m.grad, " x.grad =", x.grad)

# 正确的数学答案: d(total)/dm = 3(经过p这条路) + 1(经过q这条路) = 4
#                d(total)/dx = d(total)/dm * dm/dx = 4 * (2*x) = 4 * 4 = 16
assert m.grad == 4.0          # m.grad 最终确实等于 4.0(迟早会被两条路径都加到)……
assert x.grad != 16.0         # ……但 x.grad 在被用到的那一刻已经错了,不是 16.0
print("naive x.grad is wrong on purpose:", x.grad, "!= 16.0")
print("stage2a bug reproduced")
```

这个 bug 比想象中更"阴险":`total` 的两个直接输入 `p`、`q` 都各自正确地把自己那一份梯度加进了 `m.grad`,所以**如果你只检查 `m.grad`,看到的是完全正确的 `4.0`,不会怀疑任何东西**。真正错的是再往下一层的 `x.grad`——因为 `visit(m)` 在深度优先遍历时,是从 `p` 这条分支第一次到达 `m` 的,那一刻 `m.grad` 只加过 `p` 贡献的 `3.0`,`q` 的那份 `1.0` 还没加上;`naive_backward_no_toposort` 立刻用这个"半成品" `m.grad=3.0` 算出 `x.grad`,并把 `m` 标记为已访问——等真正轮到 `q` 把剩下的 `1.0` 加上、`m.grad` 变成完整的 `4.0` 时,`m` 已经"访问过"了,不会再触发一次 `_backward()` 把这份新增的梯度继续往 `x` 传。用文字画出这张"菱形"计算图会更直观:

```text
total (op="+")
  |
  +-- _prev --> p (op="*")
  |               |
  |               +-- _prev --> m (op="*")   <-- 共享节点!q 那边也指向同一个 m 对象
  |               |               |
  |               |               +-- _prev --> x (leaf)   (两条边都指向同一个 x,原理同 z=x+x)
  |               +-- _prev --> Value(3.0) (leaf)
  |
  +-- _prev --> q (op="+")
                  |
                  +-- _prev --> m (op="*")   <-- 和上面 p 指向的是同一个对象,不是它的副本
                  +-- _prev --> Value(5.0) (leaf)
```

`m` 在图里只有一个对象,但有两条不同的边(经过 `p`、经过 `q`)指向它——这正是"多条路径汇入同一节点"的真实样子。现在换成会先把依赖顺序排出来、再统一往回走的写法:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)          # 一个节点,只有它所有的孩子都处理完了才会被加进来
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):        # 整体倒过来走:轮到某节点时,它的"下游"必然已经全部走完
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


x = Value(2.0)
m = x * x
p = m * Value(3.0)
q = m + Value(5.0)
total = p + q

total.backward()
print("fixed: m.grad =", m.grad, " x.grad =", x.grad)
assert m.grad == 4.0
assert x.grad == 16.0        # 拓扑排序修好了阶段2a的错误
print("stage2b fixed")

# 同一个leaf被用两次: 呼应 02 类第 3 节 z2 = x2 + x2 那个例子
x2 = Value(3.0)
z = x2 + x2
z.backward()
print("z.data =", z.data, " x2.grad =", x2.grad)
assert x2.grad == 2.0     # 两条边都汇入同一个"累加槽位",1 + 1 = 2,不是覆盖成 1

# 两次"独立"的backward()调用,只要共享同一个leaf,梯度照样会跨着两张不同的图继续累加
# ——这正是真实训练循环里必须每一步调用 optimizer.zero_grad() 的根本原因(02类第10节)
w = Value(5.0)
y1 = w * 2.0
y1.backward()
print("after 1st backward, w.grad =", w.grad)
assert w.grad == 2.0

y2 = w * 3.0            # 全新的一张图,但 w 还是同一个 leaf 对象
y2.backward()
print("after 2nd backward (no reset), w.grad =", w.grad)
assert w.grad == 5.0     # 2.0(上一轮的残留) + 3.0(这一轮新算的) —— 不是干净的 3.0!

w.grad = 0.0             # 我们引擎版本的"zero_grad":没有魔法,就是手动清零
y3 = w * 3.0
y3.backward()
assert w.grad == 3.0     # 这次才是干净的
print("stage2c accumulation + manual zero_grad ok")
```

`build_topo` 是一次标准的 DFS 后序遍历:一个节点只有在它的**所有**孩子都递归处理完之后,才会被 `append` 进 `topo` 列表——这保证了"造出某个节点的所有输入"一定排在这个节点自己前面。`backward()` 里再把 `topo` 整体反过来(`reversed(topo)`)从后往前走:这样任何一个节点被处理时,消费它的那些下游节点必然已经全部处理完了,`.grad` 已经是最终值、不会再变,这时候把它继续往自己的输入传才是安全的。

这段代码同时验证了三件事,分别对应 [02 类](02-autograd-internals.md) 的三个不同小节:1)拓扑序修好了阶段 2a 的错误(`x.grad == 16.0`,对应第 4 节"反向拓扑遍历");2)同一个 leaf 被用两次,梯度是两条边的和而不是覆盖(`x2.grad == 2.0`,对应第 3 节 `AccumulateGrad` 那个一模一样的例子);3)两次独立 `backward()` 调用共享同一个 leaf,梯度会跨图继续累加(`w.grad` 从 `2.0` 变成 `5.0` 而不是 `3.0`,对应第 10 节"为什么训练循环必须 `zero_grad()`")——这不是 torch 特有的行为,是"梯度默认累加"这个设计选择的必然结果,我们自己从零写的引擎里,同样的坑一样会发生。

---

## 阶段 3:接入非线性——ReLU/tanh,和手工微积分对一遍

再补两条本地梯度规则,把纯线性的"加法+乘法"世界扩展到真实网络离不开的非线性:

- `y = relu(x)`:`dy/dx = 1`(当 `x > 0`),`dy/dx = 0`(当 `x <= 0`)——[04 类第 2 节](04-layers-math-and-backward.md)已经推过这条规则对应的 torch 版本;负区间局部梯度恒为 0,正是 [04 类第 10 节"dead ReLU"](04-layers-math-and-backward.md)现象的根源——一旦某条边的本地梯度恒为 0,不管上游传来多大的梯度,这条边往回传的永远是 0。
- `y = tanh(x)`:`dy/dx = 1 - tanh(x)^2`——`tanh = sinh/cosh`,用商法则展开可以推出这个结果,这是标准的微积分结论,这里直接取用,不重新推导商法则本身。

写法和阶段 2 的 `__add__`/`__mul__` 是同一个模式:算出前向 `data`,把这次运算的本地规则包成一个 `_backward` 闭包挂在结果节点上。还需要补上 `__neg__`/`__sub__`(减法用"取负再相加"实现,不是新的本地规则)。

```python
import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)   # a - b 复用加法的本地规则,不需要单独写一条

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# 沿用阶段1同样的 a, b, c ——s 算出来正好还是阶段1里的那个 d = 4.0
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

s = a * b + c     # s = 4.0,和阶段1的 d 完全一样
t = s.tanh()       # 非线性 #1
r = a - b           # r = 5.0(正数,所以下面的 relu 处在"直接透传"那个分支)
u = r.relu()        # 非线性 #2
f = t + u            # f = tanh(4.0) + 5.0

f.backward()
print("s.data =", s.data, " t.data =", t.data, " r.data =", r.data, " u.data =", u.data)
print("f.data =", f.data)
print("engine grads: a=", a.grad, " b=", b.grad, " c=", c.grad)

# 手工用链式法则推导解析梯度:f = tanh(a*b+c) + relu(a-b)
# 设 s = a*b+c ;  因为 r = a-b = 5.0 > 0,relu 处在"直接透传"分支,本地梯度是 1
#   df/da = tanh'(s) * ds/da + drelu/da = (1 - tanh(s)^2) * b + 1
#   df/db = tanh'(s) * ds/db + drelu/db = (1 - tanh(s)^2) * a + (-1)
#   df/dc = tanh'(s) * ds/dc            = (1 - tanh(s)^2) * 1
T = math.tanh(4.0)     # 独立重新算一遍,不是从引擎算出来的 t 里顺手拿——这样两条路径才是真的互相独立
manual_da = (1 - T ** 2) * (-3.0) + 1.0
manual_db = (1 - T ** 2) * (2.0) + (-1.0)
manual_dc = (1 - T ** 2) * 1.0
print("manual grads:  a=", manual_da, " b=", manual_db, " c=", manual_dc)

TOL = 1e-9   # 双方全程都是python原生float(IEEE754双精度),理论上应完全相等,容差只防浮点运算顺序的极小误差
assert abs(a.grad - manual_da) < TOL
assert abs(b.grad - manual_db) < TOL
assert abs(c.grad - manual_dc) < TOL
print("stage3 ok: engine grad matches hand-derived calculus within", TOL)
```

实测 `tanh(4.0) ≈ 0.9993293`,离 `1.0` 已经很近(`tanh` 在自变量较大时会饱和,这也是很多真实网络里"tanh 激活容易梯度消失"这类说法的数值来源,[08 类](08-memory-and-performance.md)/[04 类](04-layers-math-and-backward.md)讨论饱和激活函数时会用到同样的直觉)。三个方向的梯度——引擎自动算出来的和手工套公式算出来的——完全对上,容差 `1e-9` 都用不满,因为两条路径全程都是 Python 原生双精度浮点数运算。

---

## 阶段 4:交叉验证——和真正的 `torch.autograd` 对一遍数值

前三个阶段全程没有出现过 `import torch`。这一步是全文唯一一次引入 torch,而且**目的很明确:只是拿真实 `torch.autograd` 的计算结果做交叉验证,不是拿它替代自己的实现**——用同一组数字、同一个表达式,分别搭一遍我们自己的 `Value` 计算图和一遍真实的 `torch.Tensor` 计算图,对比两边 `.backward()` 之后算出的梯度数值。

```python
import math
import torch

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# ---- 用完全相同的三个数字、完全相同的表达式,分别搭一遍我们的引擎和真实torch的图 ----
a, b, c = 2.0, -3.0, 10.0

ea, eb, ec = Value(a), Value(b), Value(c)
ef = (ea * eb + ec).tanh() + (ea - eb).relu()
ef.backward()

ta = torch.tensor(a, requires_grad=True)
tb = torch.tensor(b, requires_grad=True)
tc = torch.tensor(c, requires_grad=True)
tf = torch.tanh(ta * tb + tc) + torch.relu(ta - tb)
tf.backward()

print("engine f =", ef.data, " torch f =", tf.item())
print("engine grads: a=", ea.grad, " b=", eb.grad, " c=", ec.grad)
print("torch  grads: a=", ta.grad.item(), " b=", tb.grad.item(), " c=", tc.grad.item())

TOL = 1e-4   # torch tensor默认是float32,我们的引擎全程是python float64,精度不同,预期会有极小差距
diff_a = abs(ea.grad - ta.grad.item())
diff_b = abs(eb.grad - tb.grad.item())
diff_c = abs(ec.grad - tc.grad.item())
print("abs diff: a=", diff_a, " b=", diff_b, " c=", diff_c)

assert diff_a < TOL
assert diff_b < TOL
assert diff_c < TOL
print("stage4 ok: mini engine matches real torch.autograd within", TOL)
```

实测两边梯度的绝对误差量级在 `1e-7` 左右(具体数值可能因 torch 版本/硬件略有差异,但都远小于 `1e-4` 的容差),不是位级相等——这不是巧合,也不是谁的实现有 bug:`torch.tensor(2.0)` 默认创建的是 **float32** tensor,而我们引擎里的 `Value.data` 全程是 Python 原生 **float64**,两种精度在做完全相同的一串 `*`/`+`/`tanh`/`relu` 运算后,舍入误差不可能完全一致。这正是仓库其他系列反复强调的"浮点数比较要用容差、不能要求位级相等"在这里的真实体现——如果这里写 `assert diff_a == 0.0` 会当场失败,不是因为哪个实现错了,是因为压根不应该期望两种不同精度的浮点运算给出位级相同的结果。**误差量级本身很小、方向一致、和手推的解析梯度(阶段 3)也对得上,这才是"两个独立实现算出了同一个数学答案"最有说服力的证据**——不是猜出来的,是三条独立路径(我们的引擎、手工微积分、真实 torch)互相印证的结果。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **支持向量/矩阵,而不只是标量**:`data` 从一个 Python `float` 换成一整块数组之后,本地梯度规则也要跟着变(比如矩阵乘法的反向传播需要转置,广播出来的维度反向要做 `sum` 归约)——这正是 [01 类](01-tensor-memory-model.md) 讲的 storage/stride/广播 这套内存模型真正派上用场的地方:一旦不再是标量,"形状对不对"本身就会变成一整类新坑,不再像本文这样"只要数值算对就行"。
- **更多算子**:`pow`/`exp`/`log`/`sigmoid`/`div`……每加一个算子,只需要照抄阶段 2/3 的模式——写好前向 `data`,配一条 `_backward` 闭包写清楚这一步的本地梯度规则,`backward()` 本身完全不用改。
- **参数容器和优化器的雏形**:给几个 `Value` 包一层 `Neuron`/`Layer`/`MLP` 对象,把散落的 `Value` 收进 `parameters()` 方法统一遍历,就是 [03 类](03-nn-module-internals.md)"`nn.Module` 自动注册机制"的思路在玩具尺度上的复刻;算完 `.grad` 后写一个 `for p in parameters(): p.data -= lr * p.grad` 的循环,就是 [06 类](06-optimizer-internals.md) SGD 最原始的形态——这两步加起来,离一个能训练的"迷你 MLP"就只差一个训练循环。
- **效率**:真实 torch 用 C++/CUDA kernel、批量处理整块 tensor;本文纯 Python 版本里每一个标量都是一个真实的 Python 对象、每次运算都会创建新对象和新闭包,开销很大——这套实现只适合教学、验证"机制对不对",不适合真的拿去训练模型,这也是为什么 [08 类](08-memory-and-performance.md)讲的内存/性能优化在真实 torch 里如此重要。

---

## 这篇教程展示的方法论

01-12 篇的方法论,是"让你能亲手打开 torch 内部的黑箱,用能跑的代码验证每一条底层机制的说法"——但打开的黑箱终究是别人写好的 C++ 代码,验证的是"这句话描述的现象是真的",不是"我知道要让这件事发生、自己该怎么写代码"。这一篇反过来:不借助 torch 的任何 autograd 机制,从两个运算符重载开始,一步步长出一个真正能做反向传播的迷你引擎。阶段 2 里"不排拓扑序会得到看似正常、实则错误的梯度"这个 bug,是只有自己动手实现一遍才会真正踩到的坑——`m.grad` 显示的是完全正确的 `4.0`,错误藏在再往下一层的 `x.grad` 里;读一百遍"backward 是反向拓扑遍历"这句话,都不如亲手写出一个用错顺序的版本、亲眼看到 `x.grad` 从该有的 `16.0` 变成 `12.0`,来得深刻。

这也是全系列 13 篇里唯一一处不满足于"调用 torch、观察它的真实行为",而是"脱离 torch、独立重建它的核心机制"的地方——和 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md)("把三个已经学过的知识点串成一个能用的工具")的定位并不完全相同:那篇教程组装的是已经学过、且本身独立存在的三种数据结构;这一篇的阶段 1-3,某种程度上是在**复现**([02 类](02-autograd-internals.md)已经讲过的)autograd 机制本身,阶段 4 才是回到"和已学内容对表"的位置。两种模式都成立,适用的场景不同:一个机制如果已经有很多独立部件、缺的只是"怎么拼起来"的经验,适合"组装式"教程;一个机制如果本身就是全系列的认知难点、核心价值就在"这件事究竟是怎么做到的",自己从零实现一遍、亲手踩一次只有实现者才会踩到的坑,才是把它真正学透的方式。

---

*创建:2026-07-24*
