# 03 · nn.Module 系统内核(nn.Module Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲 `nn.Module`——整个 PyTorch 建模系统的地基。表面上它就是"继承一下、写个 `forward`"的模板代码,但面试真正想挖的是:`self.linear = nn.Linear(...)` 这句看起来像 C 语言 struct 字段赋值的代码,背后到底发生了什么?一个模型对象怎么"知道"自己有哪些参数、哪些子模块?`.to(device)`/`.parameters()`/`.state_dict()`/`.train()` 这些"一键操作整个模型"的方法,是怎么把操作传递到每一层、每一个参数上的?想清楚这一层,才能真正理解为什么"把子层存进普通 list 而不是 `nn.ModuleList`"是一个不报错、却会让训练完全失效的经典陷阱。

**本文定位和 [02-pytorch-basics.md](../02-pytorch-basics.md) 第 4 节的关系:** 那一节已经用 C 语言 struct+函数指针的类比,教会你"怎么用"——继承 `nn.Module`、必须调用 `super().__init__()`、`self.weight = nn.Parameter(...)`、实现 `forward()`。这里完全不重复"怎么写一个 Module",只回答一个问题:上面这几行你已经会背的代码,**内部机制上到底是怎么运作的**——`__setattr__` 怎么拦截赋值、参数是怎么被"收编"进模型的、递归遍历/模式切换/hook 这些"高级操作"底层依赖的是同一套什么样的数据结构。

本文所有代码例子已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用)下实际跑通验证。凡是"底层机制"部分给出的结论,能用 `inspect.getsource` 现场读源码的地方就直接读 PyTorch 安装包里的真实源码(不转述文档),能用 `assert` 验证的结论一律现场跑一遍,数出实际的参数个数、打印实际的内部字典状态,不是凭经验断言。

**本篇统一结构(与 01 一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究/工程场景
5. 可运行例子(assert 验证 + 现场内省打印)
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

---

## 1. `nn.Module.__setattr__` —— Parameter / Module / Buffer 是怎么被自动"收编"的

**是什么:**
```python
def __setattr__(self, name: str, value) -> None: ...
```
`nn.Module` 重写了 Python 对象属性赋值的协议方法 `__setattr__`。这意味着你在任何 `nn.Module` 子类的 `__init__` 里写的每一句 `self.xxx = yyy`,都不是"直接把值塞进 `self.__dict__`"这么简单——它会先经过这个方法的拦截和判断。

**一句话:** `self.linear = nn.Linear(4, 2)` 这种赋值语句会被 `__setattr__` 拦截,根据等号右边这个对象的**类型**(`Parameter`?`Module`?`Buffer`?都不是?),分别登记进 `self._parameters` / `self._modules` / `self._buffers` 三个内部字典之一,而不是存进 Python 对象常规的 `self.__dict__`。

**底层机制/为什么这样设计:**

不转述,直接读 `torch/nn/modules/module.py` 里的真实源码(可以自己跑 `inspect.getsource(nn.Module.__setattr__)` 打出来对照——下面贴的是保留判断主干、去掉次要校验分支的摘录,完整版一共 104 行):

```python
def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
    def remove_from(*dicts_or_sets) -> None:
        for d in dicts_or_sets:
            if name in d:
                if isinstance(d, dict):
                    del d[name]
                else:
                    d.discard(name)

    params = self.__dict__.get("_parameters")
    if isinstance(value, Parameter):
        if params is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call"
            )
        remove_from(
            self.__dict__, self._buffers, self._modules,
            self._non_persistent_buffers_set,
        )
        self.register_parameter(name, value)
    elif params is not None and name in params:
        if value is not None:
            raise TypeError(
                f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                "(torch.nn.Parameter or None expected)"
            )
        self.register_parameter(name, value)
    else:
        modules = self.__dict__.get("_modules")
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )
            remove_from(
                self.__dict__, self._parameters, self._buffers,
                self._non_persistent_buffers_set,
            )
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(...)
            modules[name] = value
        else:
            buffers = self.__dict__.get("_buffers")
            if isinstance(value, Buffer) or buffers is not None and name in buffers:
                # ...(校验类型后)
                self.register_buffer(name, value, persistent)
            else:
                super().__setattr__(name, value)   # 走到这里,才是"普通属性"
```
(为了聚焦逻辑主干,略去了部分类型校验和 hook 回调,完整版建议自己在仓库里跑一遍 `inspect.getsource` 对照原文,一共 104 行。)

判断顺序是一条**四选一的 if/elif 链**:是 `Parameter` 就登记进 `_parameters`;是 `Module`(包括你自己写的子类)就登记进 `_modules`;是 `Buffer`(或者这个名字之前已经被登记成 buffer 过)就登记进 `_buffers`;都不是,才落到最后的 `super().__setattr__(name, value)`——也就是普通 Python 对象该有的行为,存进 `self.__dict__`。

**为什么要这样设计?** 核心原因是把"模型的状态"拆成三个**结构化、按类型分类**的容器,而不是让它们和普通属性混在同一个 `self.__dict__` 里。这样 `parameters()`/`buffers()`/`children()` 这些方法才能做到**不用挨个检查每个属性的类型**,直接遍历对应字典就行——本质是一种"写的时候多做一点分类工作,换取读的时候极快、极简单"的设计,后面第 3、5、9 节的递归遍历方法全部建立在这三个字典之上。

**这里有一个几乎所有人都会漏掉的细节:** `nn.Linear` 被登记进 `_modules` 字典之后,它就**不在** `self.__dict__` 里了(见上面源码里的 `remove_from(self.__dict__, ...)`——赋值前会先把这个名字从 `__dict__` 里清掉)。那问题来了:你写 `self.linear(x)` 的时候,Python 是怎么找到 `linear` 这个属性的?

答案是 `nn.Module` **同时也重写了 `__getattr__`**(现场读源码,同样一字不差):
```python
def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
    if "_parameters" in self.__dict__:
        _parameters = self.__dict__["_parameters"]
        if name in _parameters:
            return _parameters[name]
    if "_buffers" in self.__dict__:
        _buffers = self.__dict__["_buffers"]
        if name in _buffers:
            return _buffers[name]
    if "_modules" in self.__dict__:
        modules = self.__dict__["_modules"]
        if name in modules:
            return modules[name]
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )
```
`__getattr__`(注意不是 `__getattribute__`)是 Python 属性查找协议里的"兜底方法"——只有在常规查找(`self.__dict__`、类属性、MRO 上的描述符……)全部失败之后,Python 才会调用它。因为 `__setattr__` 把 `linear` 从 `__dict__` 里搬走了,常规查找必然失败,于是 `__getattr__` 被触发,依次去 `_parameters`→`_buffers`→`_modules` 里手动查——这两个方法是完全对称设计的一对:一个负责"写的时候分类存",一个负责"读的时候按同样的分类顺序找回来"。

**AI 研究场景:** 调试模型时如果发现某个属性"明明赋值了",却在 `model.parameters()`、`model.to(device)` 之后表现得像"根本不存在"——第一反应应该是检查这个属性有没有真的被三个内部字典之一收编(`"xxx" in model._parameters` 之类),而不是怀疑赋值语句本身写错了。这正是第 8 节 `nn.ModuleList` vs 普通 `list` 那个经典陷阱的根源:普通 `list` 不是 `Module` 也不是 `Parameter`,`__setattr__` 只会把它当成普通属性存进 `__dict__`,列表里那些 `nn.Linear` 层因此从来没有被登记过。

**可运行例子:**
```python
import torch
import torch.nn as nn
import inspect

class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)           # nn.Module 子类实例
        self.w = nn.Parameter(torch.randn(3))     # nn.Parameter
        self.plain = torch.randn(3)                # 普通 tensor(哪怕后面手动设 requires_grad)

d = Demo()

# 三个内部字典各自登记了什么,现场打印
print("_modules keys:", list(d._modules.keys()))
print("_parameters keys:", list(d._parameters.keys()))
print("_buffers keys:", list(d._buffers.keys()))
print("__dict__ 里的非内部属性:", [k for k in d.__dict__ if not k.startswith("_") and k != "training"])

assert "linear" in d._modules
assert "w" in d._parameters
assert "plain" not in d._modules and "plain" not in d._parameters and "plain" not in d._buffers
assert "plain" in d.__dict__          # 普通 tensor 走正常的 __dict__ 存储路径
assert "linear" not in d.__dict__      # 被 __setattr__ 拦截,根本没有存进 __dict__!
assert "w" not in d.__dict__

# 但 d.linear / d.w 依然能正常访问 —— 靠的是 __getattr__ 兜底查找
assert d.linear is d._modules["linear"]
assert d.w is d._parameters["w"]

# inspect.getsource 现场验证:上面贴的源码不是转述,是真的从安装的 torch 包里读出来的
src = inspect.getsource(nn.Module.__setattr__)
assert "params = self.__dict__.get(\"_parameters\")" in src
assert len(src.splitlines()) == 104

print("Section 1 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`nn.Module` 是怎么知道 `self.layer = nn.Linear(...)` 里的 `layer` 是一个需要参与参数管理的子模块的?"—— 期望答出"`nn.Module` 重写了 `__setattr__`,按赋值对象的类型分别登记进 `_parameters`/`_modules`/`_buffers` 三个字典"。
- **追问 1:** "这些信息具体存在哪?为什么不直接存进 `self.__dict__`?"—— 期望答出"三个独立字典,为了让 `parameters()`/`children()` 这些方法能快速、精确地按类型遍历,不用扫描整个 `__dict__` 做类型判断"。
- **追问 2(能把只会调用、不懂原理的候选人问住):** "你刚说 `nn.Linear` 被存进了 `_modules` 字典、根本没进 `self.__dict__`——那我写 `self.linear` 访问它的时候,Python 是怎么找到的?"—— 期望能答出还有一个对称的 `__getattr__` 兜底实现,而不是"大概框架内部处理了"这种模糊回答。能进一步说出"`__getattr__` 只有在常规查找失败时才会被调用"说明是真的理解 Python 属性协议,不是死记 PyTorch 的行为。
- **追问 3:** "如果一个 `nn.Module` 子类忘记调用 `super().__init__()`,直接在 `__init__` 里写 `self.linear = nn.Linear(...)` 会发生什么?"—— 期望说出会报错,并能推出原因:`_modules` 这个字典本身是在 `super().__init__()` 里才被创建的,`__setattr__` 里 `self.__dict__.get("_modules")` 拿到 `None`,直接抛 `AttributeError`。

**常见坑:** 忘记调用 `super().__init__()`——这是新手最容易踩、报错信息却经常被囫囵看过去的坑。实测报错信息:
```python
import torch.nn as nn

class Forgetful(nn.Module):
    def __init__(self):
        # 没有 super().__init__()!
        self.linear = nn.Linear(2, 2)

try:
    Forgetful()
except AttributeError as e:
    assert str(e) == "cannot assign module before Module.__init__() call"
```
这条报错信息其实已经把原因写得很清楚了("在 `Module.__init__()` 调用之前尝试赋值模块"),理解了上面的机制就不会再觉得这是一条"看不懂的报错"。

---

## 2. `nn.Parameter` —— 被特殊对待的 Tensor 子类

**是什么:**
```python
nn.Parameter(data=None, requires_grad=True)
```
`torch.Tensor` 的一个子类,`requires_grad` 默认就是 `True`;更关键的是,它会被上一节讲的 `nn.Module.__setattr__` **专门识别**——赋值给某个 `Module` 的属性时,自动登记进 `_parameters` 字典。

**一句话:** 让一个 tensor 变成"模型参数"需要两件事同时成立——`requires_grad=True`(能求梯度)**加上**"被 `nn.Module` 自动收编进 `_parameters`"(能被 `parameters()`/优化器看到);普通 tensor 就算手动打开 `requires_grad`,也只满足第一条,永远不满足第二条。

**底层机制/为什么这样设计:**

现场读 `nn.Parameter.__new__` 的源码:
```python
def __new__(cls, data=None, requires_grad=True):
    if data is None:
        data = torch.empty(0)
    if type(data) is torch.Tensor or type(data) is Parameter:
        return torch.Tensor._make_subclass(cls, data, requires_grad)
    ...
```
标准路径就一行:`torch.Tensor._make_subclass(cls, data, requires_grad)`——它不拷贝底层数据,只是把同一块 storage"包"上一层新的 Python 类型外壳(`Parameter` 而不是普通 `Tensor`),顺便把 `requires_grad` 设好。这就是为什么 `nn.Parameter(t)` 和原来的 `t` 完全共享同一块内存(`data_ptr()` 相同)——呼应 [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 1 节"tensor = 元数据 + 指向 storage"的心智模型:`Parameter` 改变的只是"外壳类型"这个元数据,不是数据本身。

`Parameter` 类文档字符串里有一句话点破了设计动机:如果没有 `Parameter` 这个专门的类型,`nn.Module` 就没法区分"这是一个需要被训练、保存、遍历的模型参数"和"这只是我想暂存一下的临时 tensor(比如 RNN 里上一步的 hidden state)"——**类型**是 `__setattr__` 唯一能拿到手的、不需要你手动额外声明的判断依据。这也是为什么本节最容易被问倒的地方:`requires_grad=True` 只是"参与梯度计算"的必要条件,不是"变成可训练参数"的充分条件,自动注册这一步靠的完全是类型识别,和 `requires_grad` 的值没有任何关系。

**AI 研究场景:** 手写自定义层时,任何"应该被 `optimizer.step()` 更新"的量都必须包一层 `nn.Parameter`,哪怕它一开始是通过某种计算得到的初始值(比如从预训练权重矩阵里切出一部分)。反过来,任何"只是想跟着 forward 走一遍、但不该被训练"的中间量(比如某个手写 RNN 层缓存的上一步 hidden state)则**不能**包 `nn.Parameter`,否则会被优化器误当成"参数"去更新,产生完全不该发生的梯度下降。

**可运行例子:**
```python
import torch
import torch.nn as nn

assert issubclass(nn.Parameter, torch.Tensor)

t = torch.randn(3)
p = nn.Parameter(t)                     # 默认 requires_grad=True
assert isinstance(p, nn.Parameter) and isinstance(p, torch.Tensor)
assert p.requires_grad is True
assert p.data_ptr() == t.data_ptr()       # 只是包了层壳,不拷贝底层内存

# 核心对比:同样 requires_grad=True,一个用 nn.Parameter 包过,一个是普通 tensor
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.official = nn.Parameter(torch.randn(4))
        self.imposter = torch.randn(4)
        self.imposter.requires_grad_(True)     # 手动打开 requires_grad,但没包 nn.Parameter

m = M()
assert "official" in m._parameters
assert "imposter" not in m._parameters          # 关键:requires_grad=True 也不会被自动注册!
assert [n for n, _ in m.named_parameters()] == ["official"]   # imposter 完全不出现

# imposter 的 requires_grad 确实是 True(autograd 不管注册不注册,梯度照样能算)
assert m.imposter.requires_grad is True

# 用 m.parameters() 喂优化器,验证"看不见就永远学不到"
opt = torch.optim.SGD(m.parameters(), lr=0.1)
official_before, imposter_before = m.official.clone(), m.imposter.clone()

loss = m.official.sum() + m.imposter.sum()
loss.backward()
assert m.imposter.grad is not None               # 梯度是真的算出来了
opt.step()

assert not torch.equal(m.official, official_before)  # 官方参数:优化器真的更新了它
assert torch.equal(m.imposter, imposter_before)         # 冒牌参数:有梯度,但优化器根本看不到它,纹丝不动
print("official 变了:", not torch.equal(m.official, official_before))
print("imposter 没变:", torch.equal(m.imposter, imposter_before))
print("Section 2 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`nn.Parameter` 和一个 `requires_grad=True` 的普通 tensor 有什么区别?"—— 很多人只会说"没区别,都能求梯度",这是不完整的答案。
- **追问(区分度很高):** "如果我在 `__init__` 里写 `self.w = torch.randn(3, requires_grad=True)`(不包 `nn.Parameter`),然后用 `optimizer = torch.optim.SGD(model.parameters(), ...)` 训练,`w` 会被更新吗?"—— 期望准确答出"不会,因为它压根不会出现在 `model.parameters()` 里,尽管 `loss.backward()` 依然能给它算出 `.grad`"。这个反例能立刻分辨候选人是"背过 `requires_grad` 三个字"还是真的理解注册机制。
- **追问(深挖):** "`nn.Parameter(existing_tensor)` 这一步,是把 `existing_tensor` 的数据拷贝了一份,还是共享内存?"—— 期望答出共享(`_make_subclass` 只换类型外壳),并能联系到 01 篇的 storage 心智模型。

**常见坑:** 反过来,把不该训练的中间状态误包成 `nn.Parameter`(比如某个应该由公式手算、不参与梯度下降的缓存值),会导致它意外出现在 `parameters()` 里被优化器"悄悄改动"——这类 bug 的症状是"模型效果时好时坏、缓存状态莫名其妙漂移",往往要过很久才会被发现是这个原因。判断标准很直接:**这个值该不该被 `loss.backward()` + `optimizer.step()` 这条链路改动**——该,就包 `nn.Parameter`;不该,后面第 5 节的 `register_buffer` 才是正确选择。

---

## 3. `parameters()` / `named_parameters()` —— 递归遍历的真实实现

**是什么:**
```python
model.parameters(recurse=True)                 # 生成器,依次产出所有 Parameter 对象
model.named_parameters(prefix="", recurse=True)  # 生成器,产出 (点分路径名, Parameter) 二元组
```

**一句话:** 一个模型可能是"模块套模块"好几层嵌套出来的树状结构,`named_parameters()` 靠**递归遍历这棵模块树**,把每一层自己直接持有的参数收集起来,名字用属性路径拼成 `middle.inner.fc.weight` 这样的点分字符串;`parameters()` 只是丢掉了名字的简化版本。

**底层机制/为什么这样设计:**

现场读源码,递归的核心其实不在 `named_parameters` 本身,而在它调用的 `named_modules()`(自己先递归展开整棵模块树)和 `_named_members`(拿着展开结果去每个模块自己的 `_parameters` 字典里取值):
```python
def named_modules(self, memo=None, prefix="", remove_duplicate=True):
    if memo is None:
        memo = set()
    if self not in memo:
        if remove_duplicate:
            memo.add(self)
        yield prefix, self                        # 先产出自己
        for name, module in self._modules.items():   # 再逐个递归子模块
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

def _named_members(self, get_members_fn, prefix="", recurse=True, remove_duplicate=True):
    modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        for k, v in get_members_fn(module):          # get_members_fn = lambda m: m._parameters.items()
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v
```
`named_modules()` 是一个**先产出自己、再递归产出每个子模块**的先序遍历(pre-order):自己先出场,然后按 `_modules` 字典的插入顺序,一个个子模块地展开它们各自的整棵子树。`_named_members` 只是在这个"模块访问顺序"上,对每个访问到的模块取一次它自己 `_parameters` 字典里的内容。

**这里有一个直觉容易出错、必须现场验证的细节:** 先序遍历意味着"父模块自己直接持有的参数,一定排在它子模块的参数前面"——**不是**"整棵子树的参数收集完,才轮到下一个兄弟模块"。构造一个三层嵌套、且父模块和子模块都各自持有参数的模型就能验证这一点:

**AI 研究场景:** `named_parameters()` 带名字这个特性,是"按名字模式匹配、选择性冻结/解冻部分参数"这类微调代码的基础设施——[02-pytorch-basics.md](../02-pytorch-basics.md) 6.3 节展示过的 `for param in model.adapter.parameters(): param.requires_grad = True` 只是最简单的版本,真实的 LoRA/adapter-tuning 微调代码往往需要更精细的名字匹配,比如 `if "lora_" in name or "adapter" in name`,这种按名字子串判断"这个参数属不属于我要训练的那部分"的写法,离不开 `named_parameters()` 而不是不带名字的 `parameters()`。另一个高频场景是**按名字分组设置不同的优化器超参数**——比如 `AdamW` 里"bias 和 LayerNorm 的权重不做 weight decay,其余权重正常做"这个几乎所有 Transformer 训练脚本都有的写法,本质就是遍历 `named_parameters()`,按名字里是否包含 `"bias"`/`"LayerNorm"` 把参数分进两个 `param_groups`。

**可运行例子:**
```python
import torch
import torch.nn as nn

class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

class Middle(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.bias_extra = nn.Parameter(torch.zeros(2))    # Middle 自己也直接持有一个参数

class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.middle = Middle()                # 注意:__init__ 里 middle 写在 top_w 前面
        self.top_w = nn.Parameter(torch.randn(2))

model = Outer()

mod_order = [n for n, _ in model.named_modules()]
param_order = [n for n, _ in model.named_parameters()]
print("named_modules 访问顺序:", mod_order)
print("named_parameters 产出顺序:", param_order)

# 实测顺序:父模块自己的参数,排在它子模块的参数之前 —— 即便 __init__ 里 middle 写在 top_w 前面
assert param_order == [
    "top_w",                     # Outer 自己的参数最先出场(和 __init__ 里赋值顺序无关!)
    "middle.bias_extra",          # Middle 自己的参数,先于它的孙子 Inner.fc
    "middle.inner.fc.weight",
    "middle.inner.fc.bias",
]

# 手动复现这套逻辑,验证和 named_parameters() 输出完全一致
manual = []
for mod_prefix, mod in model.named_modules():
    for pname in mod._parameters:
        manual.append(mod_prefix + ("." if mod_prefix else "") + pname)
assert manual == param_order

# parameters() 只是丢掉了名字
plain = list(model.parameters())
assert len(plain) == len(param_order)
assert all(torch.equal(a, b) for a, (_, b) in zip(plain, model.named_parameters()))

# recurse=False:只看直接挂在 Outer 上的参数,middle 是子模块,不算
top_only = [n for n, _ in model.named_parameters(recurse=False)]
assert top_only == ["top_w"]

print("Section 3 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "一个模型子模块嵌套了好几层,`model.parameters()` 是怎么把所有层级的参数都找出来的?"—— 期望答出"递归遍历 `_modules` 字典,收集每一层自己的 `_parameters`"。
- **追问 1:** "`named_parameters()` 返回的名字里那些点号(`.`)是怎么来的?"—— 期望答出"是递归过程中,每往下一层就拼接一次属性名,最终是从根模块到参数的完整属性路径"。
- **追问 2(区分度很高,容易被直觉带偏):** "如果父模块自己也直接持有一个 `nn.Parameter`,子模块也有自己的参数,`named_parameters()` 遍历出来的顺序是子模块的参数在前,还是父模块自己的参数在前?"—— 大多数人凭直觉会猜"越深层的越先出来"或者"和 `__init__` 里赋值顺序一致",实测结论是**父模块自己的参数永远先于它子模块的参数**,因为遍历顺序由 `named_modules()` 的先序遍历决定,和 `__init__` 里属性赋值的先后顺序无关。
- **追问 3:** "`recurse=False` 传进去之后,`middle` 这个子模块自己的参数还会不会被看到?"—— 期望答出不会,`recurse=False` 只取当前模块自己 `_parameters` 字典里的内容,不会展开 `_modules`。

**常见坑:** 依赖 `named_parameters()`/`state_dict()` 的**顺序**做位置相关的事情(比如假设"第一个参数一定是某个特定层的权重"),这个顺序虽然是确定性的(不是随机的),但它由模块树结构和 `_modules`/`_parameters` 字典的插入顺序共同决定,**不等于**你代码里直觉上认为的"声明顺序"或"深度优先到底"顺序——真正需要按名字定位参数时,应该用 `named_parameters()` 返回的名字做字典查找,而不是假设某个固定的位置下标。

---

## 4. `state_dict()` vs `parameters()` —— 存档用的快照,和训练用的句柄

**是什么:**
```python
model.state_dict(destination=None, prefix="", keep_vars=False)   # 返回一个 OrderedDict
```

**一句话:** `state_dict()` 返回一个**带名字的 `OrderedDict`**,不仅包含所有 parameter,还包含下一节要讲的 buffer(比如 BatchNorm 的 running_mean);`parameters()` 只是一个不带名字、也不含 buffer 的生成器——两者服务的场景完全不同:`state_dict()` 是给"保存/加载模型"用的完整快照,`parameters()` 是给"优化器"用的可训练句柄集合。

**底层机制/为什么这样设计:**

`state_dict()` 内部对每个模块调用 `_save_to_state_dict`,逻辑很直白:
```python
def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in self._buffers.items():
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
```
两个关键点,都值得现场验证而不是凭印象:第一,**parameter 和 persistent buffer 都会被塞进同一个 `destination` 字典**,这就是为什么 `state_dict()` 的 key 集合比 `named_parameters()` 多;第二,默认 `keep_vars=False` 时,存进去的是 `param.detach()`——也就是说 `state_dict()` 里的 tensor 默认**不带 `grad_fn`**,是纯粹"看数值"的快照视图,但 `detach()` 不拷贝数据(呼应 01 篇第 8 节),所以修改 `state_dict()` 里的 tensor 依然会影响原模型的参数,只是这份修改不会被 autograd 追踪。

**AI 研究场景:** 存 checkpoint、跨设备/跨精度加载、断点续训,全部围绕 `state_dict()` 而不是 `parameters()`——因为存档需要"这个值该恢复到哪个名字对应的位置"这层名字信息,还需要 buffer(BatchNorm 的统计量必须原样恢复,否则推理结果会不一致,这是很多人复现论文时精度对不上的隐藏原因之一)。`parameters()` 则专门给 `torch.optim.XXX(model.parameters(), lr=...)` 这种"我只关心哪些 tensor 需要被更新梯度"的场景用,完全不需要名字。

**可运行例子:**
```python
import torch
import torch.nn as nn
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.register_buffer("running_stat", torch.zeros(2))

net = Net()
sd = net.state_dict()
assert isinstance(sd, OrderedDict)

param_names = set(n for n, _ in net.named_parameters())
buffer_names = set(n for n, _ in net.named_buffers())
print("param_names:", param_names)
print("buffer_names:", buffer_names)
print("state_dict keys:", list(sd.keys()))

assert set(sd.keys()) == param_names | buffer_names        # state_dict = parameters + buffers 的并集
assert "running_stat" in sd and "running_stat" not in param_names   # buffer 在这里,但不在 parameters() 里
assert list(sd.keys()) == ["running_stat", "fc.weight", "fc.bias"]   # 顺序由模块树遍历决定(呼应第3节)

# parameters() 是纯生成器,没有名字、没有字典接口
gen = net.parameters()
assert not hasattr(gen, "keys")
assert not isinstance(gen, dict)

# state_dict() 默认 detach 过:不带 grad_fn,但仍与原参数共享底层内存(浅拷贝,不是深拷贝)
w = net.fc.weight
sd_w = sd["fc.weight"]
assert sd_w.requires_grad is False              # detach 过
assert w.requires_grad is True                    # 原参数还是可训练的
assert sd_w.data_ptr() == w.data_ptr()             # 但共享同一块底层内存

# keep_vars=True:不 detach,直接就是同一个 Parameter 对象
sd_keep = net.state_dict(keep_vars=True)
assert sd_keep["fc.weight"] is w

print("Section 4 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`model.state_dict()` 和 `model.parameters()` 有什么区别?"—— 期望至少答出"前者带名字、包含 buffer,后者是纯参数生成器"。
- **追问 1:** "为什么保存/加载模型要用 `state_dict()` 而不是直接存 `list(model.parameters())`?"—— 期望答出"缺名字就没法在加载时对应回正确的层;而且会丢失 BatchNorm 之类的 buffer,加载回来的模型推理结果会不一致"。
- **追问 2(容易漏答):** "`state_dict()` 里的 tensor,还能被用来算梯度、反向传播吗?"—— 期望答出"默认不能,因为默认 `keep_vars=False` 会 `detach()` 掉,除非显式传 `keep_vars=True`"。

**常见坑:** 以为 `state_dict()` 返回的是完全独立的一份拷贝——它不是,默认是"浅拷贝"(对 parameter/buffer 做了 `detach()`,但底层存储仍然共享),直接在 `state_dict()` 返回的字典里用 in-place 操作改动某个 tensor 的值,会连带影响原模型参数,这种"改了副本却影响了本体"的 bug 很隐蔽,涉及需要真正独立副本的场景应该显式 `.clone()`。

---

## 5. `register_buffer` —— 跟着模型走、但不受优化器管的状态

**是什么:**
```python
model.register_buffer(name, tensor, persistent=True)
```

**一句话:** buffer 是"属于模型状态、但不是可训练参数"的 tensor——它会被 `state_dict()` 保存/加载,会跟着 `.to(device)` 走,但 `requires_grad` 恒为 `False`,永远不出现在 `parameters()` 里,因此永远不会被 `optimizer.step()` 更新。

**底层机制/为什么这样设计:**

一个模型的状态并不只有"该被梯度下降更新的参数"这一种。典型例子是 `BatchNorm` 的 `running_mean`/`running_var`——这两个值是训练过程中**用滑动平均公式从每个 batch 的统计量里累积出来的**(下一批 04 篇会推导具体公式),不是反向传播算出梯度、优化器一步步"走"出来的。如果没有 buffer 这个类别,要么把它们错误地包成 `Parameter`(会被优化器错误更新),要么用普通属性存(不会跟着 `.to(device)` 走,也不会被存进 checkpoint,换个设备或者重新加载模型后统计量直接丢失)。

第 1 节读 `__setattr__` 源码时其实已经埋了伏笔:除了识别 `isinstance(value, Parameter)` 和 `isinstance(value, Module)`,还有一支判断 `isinstance(value, Buffer)`——`torch.nn.Buffer` 是较新版本加入的、和 `nn.Parameter` 对称的写法,允许你像 `self.w = nn.Parameter(...)` 一样直接 `self.counter = nn.Buffer(...)`,不用手动调用 `register_buffer`。两条路径最终都会落到同一处:把 tensor 存进 `self._buffers[name]`。

**AI 研究场景:** 除了 `BatchNorm` 的 running 统计量,buffer 在研究代码里另一个高频场景是**自监督/扩散模型里的 EMA(指数滑动平均)影子权重**——比如 BYOL/MoCo 的 teacher 网络、部分扩散模型训练脚本维护的 EMA 版本模型权重,这些值是按 `shadow = decay * shadow + (1-decay) * param` 这种自定义公式手动更新的,不走 `loss.backward()`,所以本质上也该是 buffer 而不是 parameter——虽然某些实现出于工程简便会用独立模型对象存 EMA 权重,但"状态需要保存/跟设备走,但不受优化器管"这个判断标准是完全一致的。

**可运行例子:**
```python
import torch
import torch.nn as nn

bn = nn.BatchNorm1d(4)
param_names = set(n for n, _ in bn.named_parameters())
buffer_names = set(n for n, _ in bn.named_buffers())
print("BatchNorm 的 parameters:", param_names)
print("BatchNorm 的 buffers:", buffer_names)

assert param_names == {"weight", "bias"}                                # 可学习的 gamma/beta
assert buffer_names == {"running_mean", "running_var", "num_batches_tracked"}
assert bn.running_mean.requires_grad is False                             # buffer 恒为 False
assert "running_mean" in bn.state_dict()                                   # 会被保存

# running_mean 是"统计"出来的,不是"梯度下降"出来的:只做前向、完全不调用 backward()
before = bn.running_mean.clone()
x = torch.randn(8, 4) * 5 + 3
bn.train()
_ = bn(x)                                     # 没有 backward()!
after = bn.running_mean.clone()
assert not torch.equal(before, after)           # running_mean 依然变了
print("running_mean 只因前向传播就改变:", before.tolist(), "->", after.tolist())

# 对比:同样只前向不 backward,真正的 parameter(weight/bias)纹丝不动
w_before = bn.weight.clone()
_ = bn(x)
assert torch.equal(w_before, bn.weight)          # parameter 只能被 optimizer.step() 改

# persistent=False:依然是 buffer(跟 .to() 走),但不进 state_dict
class WithNonPersistent(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(3), persistent=False)

m2 = WithNonPersistent()
assert "cache" in dict(m2.named_buffers())
assert "cache" not in m2.state_dict()

# .to(device) 会带着 buffer 一起走
assert torch.cuda.is_available()
bn_gpu = bn.to("cuda")
assert bn_gpu.running_mean.device.type == "cuda"
assert bn_gpu.weight.device.type == "cuda"

# 2.x 新增的 nn.Buffer:和 nn.Parameter 对称的写法,不用手动调用 register_buffer
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = nn.Buffer(torch.zeros(1))    # __setattr__ 识别出 Buffer 类型,自动登记
assert "counter" in M()._buffers

print("Section 5 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "BatchNorm 的 `running_mean`/`running_var` 是 parameter 还是 buffer?为什么?"—— 期望答出"buffer,因为它们是统计量滑动平均出来的,不是梯度下降学出来的"。
- **追问 1:** "怎么用代码证明 `running_mean` 不是被反向传播更新的?"—— 期望想到"只做前向不调用 `backward()`,观察 `running_mean` 是否依然变化"这个实验设计,而不是只会背结论。
- **追问 2(容易漏答):** "如果我在 `__init__` 里直接写 `self.running_mean = torch.zeros(4)`,不调用 `register_buffer`,会怎样?"—— 期望答出"它会变成一个普通属性,不进 `_buffers`,不会跟 `.to(device)` 走,也不会被 `state_dict()` 保存"——这是本节最该被追问出来的坑,直接对应第 2 节"普通 tensor 不会被自动注册"的姊妹版本。

**常见坑:** 混淆"没有 `requires_grad`"和"不是 buffer"——一个 tensor 完全可以 `requires_grad=False` 却根本没有通过 `register_buffer`/`nn.Buffer` 登记,这种情况下它既不是 parameter 也不是 buffer,只是普通属性,`.to(device)`/`state_dict()` 都不会管它,已用代码验证:
```python
import torch
import torch.nn as nn

class Sloppy(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean = torch.zeros(4)   # 忘了用 register_buffer,当成普通属性赋值

m = Sloppy()
assert "running_mean" not in m._buffers
assert "running_mean" not in m.state_dict()     # 存档时悄悄丢失,不会有任何报错提示
m_gpu = m.to("cuda")
assert m_gpu.running_mean.device.type == "cpu"    # 模型"移到了GPU",这个属性却留在了CPU
```

---

## 6. `train()` / `eval()` —— 切换的是一个会被递归广播的标志位

**是什么:**
```python
model.train(mode: bool = True)     # 设置 self.training = mode,并递归设置所有子模块
model.eval()                        # 等价于 model.train(False)
```

**一句话:** `train()`/`eval()` 本身不包含任何"具体行为"的逻辑,它只是把 `self.training` 这个布尔值**递归**同步到模型树上的每一个子模块——真正"看到这个标志位之后做什么不同的事",是 `Dropout`、`BatchNorm` 这些具体层各自 `forward()` 里的判断逻辑。

**底层机制/为什么这样设计:**
```python
def train(self, mode: bool = True) -> Self:
    if not isinstance(mode, bool):
        raise ValueError("training mode is expected to be boolean")
    self.training = mode
    for module in self.children():
        module.train(mode)     # 递归:每个子模块自己再往下广播一层
    return self

def eval(self) -> Self:
    return self.train(False)
```
递归方式和第 3 节的 `named_parameters()`、第 9 节的 `modules()` 是同一种模式——通过 `children()` 拿到直接子模块,让每个子模块自己再调用一次 `train(mode)` 继续往下传,最终整棵模块树上每一层的 `self.training` 都被同步成同一个值。值得留意的是:`self.training = mode` 这一行赋值,同样会经过第 1 节讲的 `__setattr__`——只是因为 `bool` 既不是 `Parameter` 也不是 `Module`/`Buffer`,会直接落到最后的 `else` 分支,存进普通的 `self.__dict__`,和 `_parameters`/`_modules`/`_buffers` 三个专用字典完全无关。

这个标志位本身"空",意义全部来自下游层怎么读它。这里先建立"`train()`/`eval()` 到底改变了什么状态"这个框架,`Dropout` 训练时按概率把神经元输出置零、推理时原样通过;`BatchNorm` 训练时用**当前 batch** 自己的均值方差归一化(同时顺带更新 `running_mean`/`running_var`,呼应第 5 节),推理时改用**训练阶段累积下来**的 `running_mean`/`running_var`,不再看当前这批数据自己的统计量——具体的反向传播数学公式、为什么这么设计能保证训练/推理一致性,留到下一批 [04-layers-math-and-backward.md](04-layers-math-and-backward.md) 详细展开。

**AI 研究场景:** 忘记在验证/推理前调用 `model.eval()`,是训练代码里最常见、后果又最隐蔽的 bug 之一——`Dropout` 会继续随机丢神经元(导致同一个输入每次推理结果都不一样),`BatchNorm` 会用验证 batch 自己的统计量而不是训练阶段学到的统计量(如果验证 batch 很小,统计量噪声很大,推理效果会明显变差)。同理,验证完之后忘记切回 `model.train()` 就继续训练,也会让 `BatchNorm` 在训练阶段错误地使用"eval 模式的行为",相当于停止了统计量的更新。

**可运行例子:**
```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.bn = nn.BatchNorm1d(4)

net = Net()
assert net.training is True and net.block.training is True   # 默认就是训练模式

net.eval()
# 递归验证:不只是顶层,所有子模块的 training 标志都被同步改了
assert net.training is False and net.block.training is False and net.block.drop.training is False
assert net.bn.training is False

net.train()
assert net.block.drop.training is True    # train() 同理递归传回 True

# --- Dropout:训练模式随机丢弃,推理模式原样通过(pass-through) ---
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.ones(1000)

drop.train()
y_train = drop(x)
zero_ratio = (y_train == 0).float().mean().item()
print("训练模式置零比例(约0.5):", zero_ratio)
assert 0.4 < zero_ratio < 0.6
assert not torch.equal(y_train, x)

drop.eval()
y_eval = drop(x)
assert torch.equal(y_eval, x)             # 推理模式:恒等函数,不丢弃任何值

# --- BatchNorm:训练模式用当前 batch 的统计量,推理模式用累积的 running 统计量 ---
bn = nn.BatchNorm1d(3)
bn.train()
x1 = torch.randn(16, 3) * 10 + 5
out_train = bn(x1)
assert out_train.mean(dim=0).abs().max().item() < 1e-5          # 用x1自己的统计量归一化:均值~0
assert (out_train.var(dim=0, unbiased=False) - 1).abs().max().item() < 1e-4   # 方差~1

bn.eval()
x2 = torch.randn(16, 3) * 10 + 5           # 换一批新数据
out_eval = bn(x2)
manual_with_running = (x2 - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps)
manual_with_running = manual_with_running * bn.weight + bn.bias
assert torch.allclose(out_eval, manual_with_running, atol=1e-5)    # 用的是训练阶段积累的running统计量

manual_with_x2_own_stat = (x2 - x2.mean(dim=0)) / torch.sqrt(x2.var(dim=0, unbiased=False) + bn.eps)
manual_with_x2_own_stat = manual_with_x2_own_stat * bn.weight + bn.bias
assert not torch.allclose(out_eval, manual_with_x2_own_stat, atol=1e-3)   # 不是用x2自己的统计量

print("Section 6 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`model.eval()` 具体做了什么?"—— 期望答出"把 `self.training` 递归设置为 `False`",而不是笼统地说"关闭训练模式"。
- **追问 1(能把只会调用不懂原理的候选人问住):** "`model.eval()` 这一行代码本身,有没有改变任何一个具体的计算逻辑?"—— 期望答出"没有,它只是改了一个标志位;真正的行为分叉在 `Dropout`/`BatchNorm` 各自的 `forward()` 里,`eval()` 本身对这些逻辑一无所知"。
- **追问 2:** "为什么 `Dropout`/`BatchNorm` 在 `eval()` 模式下要用不同的计算方式,而不是永远用同一套逻辑?"—— 期望结合场景说出"训练要引入随机性/用当前batch统计量来提升泛化能力和拟合训练分布,推理要确定性输出/用稳定统计量保证结果可复现",具体数学留给 04 篇。
- **追问 3(工程场景):** "验证集上跑完 `model.eval()`,忘记切回 `model.train()` 就继续下一轮训练,会有什么后果?"—— 期望答出"BatchNorm 会停止用当前 batch 更新 running 统计量、Dropout 停止随机丢弃,相当于训练过程被静默改变了行为,且不会报错"。

**常见坑:** 只对最外层模型调用了 `.eval()`,却误以为"用了 `nn.Sequential`/自定义容器包起来的子模块可能没生效"——实际上只要模型是标准的 `nn.Module` 组合(用 `nn.ModuleList`/正常子模块属性,而不是第 8 节要讲的普通 `list`),`.eval()` 的递归机制能保证覆盖所有层级,不存在"漏掉某个子模块"的问题;真正会漏掉的,是第 8 节里那种没被正确注册的"隐身"子模块——它们既不出现在 `parameters()` 里,也不会被 `train()`/`eval()` 的递归遍历到,永远停留在初始的 `training=True`。

---

## 7. `register_forward_hook` / `register_forward_pre_hook` / `register_full_backward_hook` —— 给前向/反向"打点"

**是什么:**
```python
module.register_forward_pre_hook(hook)     # hook(module, args) -> None 或修改后的输入;在 forward() 之前触发
module.register_forward_hook(hook)          # hook(module, args, output) -> None 或修改后的输出;在 forward() 之后触发
module.register_full_backward_hook(hook)     # hook(module, grad_input, grad_output) -> None 或新的 grad_input;反向传播算到这一层时触发
```

**一句话:** 三个 hook 分别在"这一层 `forward()` 执行之前"、"这一层 `forward()` 执行之后"、"反向传播算到这一层的梯度时"这三个时机被自动调用,不需要修改模型的 `forward()` 代码本身——这是在不侵入模型定义的前提下,"窃听"模型内部任意一层输入输出/梯度的标准手段。

**底层机制/为什么这样设计:**

每个 `Module` 对象内部维护了几个 `OrderedDict`(`_forward_pre_hooks`/`_forward_hooks`/`_backward_hooks`,第 1 节看过的 `__init__` 源码里已经初始化好了),`register_xxx_hook` 只是往对应字典里塞一个 `(handle_id, hook函数)` 的键值对,返回一个可以用来 `.remove()` 撤销注册的 `RemovableHandle`。真正触发调用的地方在 `Module.__call__`(每次你写 `model(x)`,实际执行的是 `__call__`,它内部才会调 `forward`)——`__call__` 会依次:①按顺序跑一遍 `_forward_pre_hooks` 里的每个 hook(有机会篡改输入)→②调用真正的 `forward()`→③按顺序跑一遍 `_forward_hooks`(有机会篡改输出)。反向传播的 hook 则挂在 autograd 图对应这一层输出的节点上,在 `.backward()` 执行到这里、算出对这一层输入的梯度时触发,时间上必然晚于整个前向过程,发生在你调用 `.backward()` 之后。

**AI 研究场景:** 最常见的用法是用 `forward_hook` 捕获某一中间层的激活值,用于可视化或调试——比如分析某一层输出的分布是否出现异常(全 0、方差爆炸)、做特征可视化、或者在不修改模型 `forward()` 代码的前提下把某个中间表示抽出来喂给下游的探针(probing)模型。这是研究代码里"不改一行模型定义,却要拿到内部某层输出"的标准手段,比手动改 `forward()` 返回值再改调用方代码侵入性小得多。

**可运行例子:**
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h)

model = MLP()
log = []
captured = {}

def pre_hook(module, args):
    log.append(("pre", args[0].shape))

def fwd_hook(module, args, output):
    log.append(("fwd", output.shape))
    # 典型场景:捕获中间激活值用于可视化/调试,detach 一份纯数值快照,不拖入计算图
    captured["fc1_output"] = output.detach().clone()

def bwd_hook(module, grad_input, grad_output):
    log.append(("bwd", tuple(g.shape if g is not None else None for g in grad_input)))

h1 = model.fc1.register_forward_pre_hook(pre_hook)
h2 = model.fc1.register_forward_hook(fwd_hook)
h3 = model.fc1.register_full_backward_hook(bwd_hook)

x = torch.randn(2, 4, requires_grad=True)
out = model(x)
# forward 阶段结束,只触发了 pre / fwd 两个,backward 还没被调用过
assert log == [("pre", torch.Size([2, 4])), ("fwd", torch.Size([2, 3]))]
assert captured["fc1_output"].shape == (2, 3)
assert not captured["fc1_output"].requires_grad

out.sum().backward()
assert log[-1] == ("bwd", (torch.Size([2, 4]),))    # backward 真正执行到 fc1 时才触发,晚于所有前向日志
print("hook 执行顺序日志:", log)

# forward_hook 还能直接修改输出(返回非 None 就替换原输出)
def clamp_hook(module, args, output):
    return output.clamp(min=0)

h4 = model.fc2.register_forward_hook(clamp_hook)
out2 = model(torch.full((1, 4), -100.0))
assert (out2 >= 0).all()          # 被 hook 强行夹到非负

for h in (h1, h2, h3, h4):
    h.remove()                     # RemovableHandle:用完记得撤销,避免hook累积内存泄漏

print("Section 7 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "怎么在不修改模型 `forward()` 代码的前提下,拿到某个中间层的输出?"—— 期望答出 `register_forward_hook`,并能说出 hook 的签名 `(module, input, output)`。
- **追问 1:** "`register_forward_pre_hook` 和 `register_forward_hook` 的触发时机具体差在哪?能不能说出一个必须用 pre-hook 而不是普通 hook 的场景?"—— 期望答出"pre-hook 在 forward 执行前触发,能修改即将传入的输入;比如要在数据真正进入某一层之前统一做一次归一化/裁剪,又不想改这层内部实现"。
- **追问 2(容易被问倒):** "`register_full_backward_hook` 里的 `grad_input`/`grad_output`,分别对应的是这一层的什么?这个 hook 是在 `model(x)` 执行时触发,还是在 `loss.backward()` 执行时触发?"—— 期望准确说出"`grad_output` 是损失对这一层**输出**的梯度,`grad_input` 是损失对这一层**输入**的梯度;hook 在 `backward()` 反向传播执行到这一层时才触发,和 `forward()` 的调用时机没有关系"——很多人会误以为三个 hook 都在 `model(x)` 一行代码内全部触发完。
- **追问 3(开放题):** "如果要监控训练过程中每一层梯度的范数(比如排查梯度消失/爆炸),你会怎么用 hook 实现?"—— 期望想到给每一层挂 `register_full_backward_hook`,在 hook 里记录 `grad_output`/`grad_input` 的 norm,而不是手动改每一层的 `forward()`/`backward()`。

**常见坑:** 忘记调用 `handle.remove()`——hook 一旦注册,除非手动移除,会一直留在模块的 `_forward_hooks` 等字典里,如果在一个循环里反复注册同一个 hook(比如放在训练循环内部而不是外面注册一次),会导致同一个 hook 被调用越来越多次,而且 hook 闭包里如果引用了大 tensor(比如上面例子的 `captured` 字典),会造成不必要的显存/内存占用,长期运行容易表现成"看起来没做什么但内存一直涨"的排查难题。

---

## 8. `nn.ModuleList` / `nn.ModuleDict` vs 普通 `list` / `dict` —— 一个不报错、却让参数"隐身"的坑

**是什么:**
```python
nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])         # 专门用来存一组子模块的容器
nn.ModuleDict({"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)})  # 专门用来存一组子模块的字典容器
```

**一句话:** `nn.ModuleList`/`nn.ModuleDict` 本身也是 `nn.Module` 的子类,赋值给某个 `Module` 的属性时会被第 1 节的 `__setattr__` 当成"子模块"正常登记进 `_modules`;普通 Python `list`/`dict` 则完全不是 `Module`,不是 `Parameter`,`__setattr__` 只会把它们当成普通属性存进 `__dict__`——里面装的那些 `nn.Linear` 层因此从来没有被模型"看见"过。

**底层机制/为什么这样设计:**

回到第 1 节的 `__setattr__` 判断链:它能识别的类型只有 `Parameter`、`Module`、`Buffer` 三种。一个普通的 `[nn.Linear(4,4), nn.Linear(4,4)]` 列表对象,它自己的类型是内建的 `list`,不满足任何一个 `isinstance` 判断,于是必然落到最后 `super().__setattr__(name, value)` 这个"什么都不做,普通存属性"的分支——**尽管列表内部装的每一个元素明明都是货真价实的 `nn.Module`**。`nn.ModuleList` 存在的唯一意义,就是把"一组子模块"包装成一个本身也是 `Module` 的容器,让外层的 `__setattr__` 能够识别、登记、进而让 `parameters()`/`.to(device)`/`state_dict()`/`train()` 这些递归机制都能正确地展开到列表里的每一层。

这个坑最阴险的地方在于:`forward()` 里对着普通 `list` 做 `for layer in self.layers: x = layer(x)` 完全能跑,输出的 shape、数值都正常——Python 语言层面从不检查"你从一个 list 里取出来的对象是不是被框架登记过",能调用就能算前向传播。真正出问题的是"这一层压根不会被训练"这件事,而这件事**不会有任何报错或警告**,只会在训练很久之后发现效果异常时才可能被怀疑到。

**AI 研究场景:** 手写模型时,任何"数量取决于超参数"的重复子层(比如 Transformer 的多层 block、多头注意力里的每个 head、多分支网络的每个分支)几乎必然要用循环生成一组子模块——这正是 `nn.ModuleList` 的典型使用场景,替代"写死 `self.layer1`、`self.layer2`……"这种不方便参数化的写法。`nn.ModuleDict` 同理,常见于多任务模型里"每个任务一个输出头"这种按名字索引子模块的场景。

**可运行例子:**
```python
import torch
import torch.nn as nn

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# 特意让模型除了"藏起来的 list"之外还有一个正常注册的层 —— 这样优化器不会在构造时就
# 因为"参数列表整个是空的"直接报错,才是真正"能跑、不报错、但训练是错的"的隐蔽版本
class BadStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(3)]   # 普通 python list!
        self.head = nn.Linear(4, 1)                            # 正常注册的输出层

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

class GoodStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

bad, good = BadStack(), GoodStack()
x = torch.randn(2, 4)

out_bad = bad(x)      # forward 完全正常,不报错 —— 这正是这个坑最阴险的地方
assert out_bad.shape == (2, 1)

n_bad, n_good = count_params(bad), count_params(good)
print("BadStack 参数量(只看到 head 的5个):", n_bad)
print("GoodStack 参数量(head + 3层隐藏层):", n_good)
assert n_bad == (4 * 1 + 1)                            # 只看到 head 的 5 个参数,3层隐藏层全部"隐身"
assert n_good == (4 * 1 + 1) + 3 * (4 * 4 + 4)          # head + 3层隐藏层,60 + 5 = 65
assert n_good - n_bad == 60                              # 整整 60 个参数凭空消失

# 参数客观存在,只是 model.parameters() 找不到它们
assert sum(p.numel() for l in bad.layers for p in l.parameters()) == 60

# --- 用 model.parameters() 喂优化器:能正常构造、能正常训练、loss会下降,但隐藏层永远学不到东西 ---
opt = torch.optim.SGD(bad.parameters(), lr=0.1)   # 正常构造成功,不报错!(因为head的参数不是空的)
hidden_w_before = bad.layers[0].weight.clone()
head_w_before = bad.head.weight.clone()

for _ in range(5):
    opt.zero_grad()
    loss = bad(x).sum()
    loss.backward()
    opt.step()

assert bad.layers[0].weight.grad is not None                  # 梯度是能算出来的(autograd不看有没有注册)
assert torch.equal(bad.layers[0].weight, hidden_w_before)       # 隐藏层权重纹丝不动,训练5步也白搭
assert not torch.equal(bad.head.weight, head_w_before)            # head 正常在更新 —— 训练"看起来"完全正常!

# --- .to(device) 同理不会移动 list 里的层 ---
assert torch.cuda.is_available()
bad_gpu, good_gpu = bad.to("cuda"), good.to("cuda")
assert bad_gpu.layers[0].weight.device.type == "cpu"        # 没被移动!还在CPU
assert bad_gpu.head.weight.device.type == "cuda"               # 正常注册的层被移动了
assert good_gpu.layers[0].weight.device.type == "cuda"

# --- ModuleDict 同理 ---
class BadDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = {"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)}     # 普通 dict

class GoodDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)})

assert count_params(BadDict()) == 0
assert count_params(GoodDict()) == 2 * (4 * 1 + 1)

print("Section 8 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "为什么要用 `nn.ModuleList` 而不是普通 Python `list` 存一组子层?"—— 很多人只会背"因为要用 ModuleList",期望能说出根本原因:`__setattr__` 只认 `Parameter`/`Module`/`Buffer` 三种类型,普通 `list` 不是这三者之一。
- **追问 1(杀伤力很强):** "如果我不小心用了普通 `list`,`forward()` 跑起来会报错吗?"—— 期望准确答出"完全不会报错,前向传播照常执行、输出 shape 正常,唯一的问题是这些层的参数不会出现在 `parameters()` 里,不会被优化器更新、也不会跟 `.to(device)` 走"。这是本节的核心陷阱,能不能想到"不报错"这个后果比想到"要用 ModuleList"这个结论本身更能体现理解深度。
- **追问 2(区分度很高):** "怎么快速排查一个模型是不是踩了这个坑?"—— 期望能提出可操作的方法,比如对比 `sum(p.numel() for p in model.parameters())` 和"预期参数量"是否吻合,或者检查 `model.__dict__` 里有没有不该出现在那里的 `nn.Module` 实例。
- **追问 3:** "既然 `nn.ModuleList` 的元素本质还是普通的 `nn.Module`,它和 `nn.Sequential` 有什么区别?"—— 期望答出"`nn.Sequential` 会自动按顺序依次调用每一层当作 `forward`;`nn.ModuleList` 只负责登记、不定义任何执行顺序,你必须自己在 `forward()` 里写循环调用每一层,这在层与层之间不是简单串联(比如有残差连接、跳跃连接)时是必须的"。

**常见坑:** 上面例子已经完整演示。这里额外补充一个容易被忽略的边界情况:如果一个模型**除了**藏在普通 `list` 里的层之外,再没有任何正常注册的参数,`torch.optim.SGD(model.parameters(), ...)` 在构造阶段就会直接抛出 `ValueError: optimizer got an empty parameter list`——这种情况反而不算隐蔽,因为程序会在训练开始前就报错。真正难排查的,是像上面例子这样"模型里还有别的正常层"的情况:优化器构造成功、训练能跑、loss 也在下降(因为其他层在正常学习),藏起来的那部分却在从头到尾用随机初始化的权重推理,不会有任何报错或警告提示你这一点。

---

## 9. `children()` vs `modules()` —— 只看下一层,还是拍平整棵树

**是什么:**
```python
model.children()        # 只产出直接子模块,不递归
model.modules()          # 递归产出模型树上所有层级的模块,包括自己
```

**一句话:** `children()` 只往下看一层;`modules()` 是把整棵模块树从根(自己)到叶子全部拍平遍历一遍——第 3 节 `named_parameters()`、第 6 节 `train()`/`eval()` 的递归能力,底层依赖的正是这两个方法之一。

**底层机制/为什么这样设计:**
```python
def children(self):
    for _name, module in self.named_children():
        yield module

def named_children(self):
    memo = set()
    for name, module in self._modules.items():     # 只看自己的 _modules 字典,不下探
        if module is not None and module not in memo:
            memo.add(module)
            yield name, module

def modules(self, remove_duplicate=True):
    for _, module in self.named_modules(remove_duplicate=remove_duplicate):
        yield module

def named_modules(self, memo=None, prefix="", remove_duplicate=True):
    if memo is None:
        memo = set()
    if self not in memo:
        if remove_duplicate:
            memo.add(self)
        yield prefix, self                             # 先产出自己
        for name, module in self._modules.items():
            submodule_prefix = prefix + ("." if prefix else "") + name
            yield from module.named_modules(memo, submodule_prefix, remove_duplicate)   # 再递归子模块
```
`children()` 只读一层 `_modules` 字典,不往下走;`modules()`(通过 `named_modules()`)在此基础上,每访问到一个模块就先 `yield` 它自己,再对它的每个子模块递归调用同一个方法——这是标准的**先序遍历**(自己先出场,然后依次展开每个子树),也正是第 3 节"父模块参数排在子模块参数前面"这个现象的根源:两者用的是同一套遍历顺序。

**AI 研究场景:** `model.apply(fn)` 是基于 `children()`(不是 `modules()`)实现权重初始化的标准手段——`apply` 会先递归到最深层的子模块,对每个子模块调用一次 `fn`,最后才对自己调用 `fn`(自底向上,和 `modules()` 的自顶向下顺序刚好相反),常见写法是 `model.apply(init_weights)`,在 `init_weights(m)` 里用 `isinstance(m, nn.Linear)` 判断类型后初始化对应的权重。`children()` 本身则常用于"只处理直接子层"的场景,比如冻结/解冻模型的某几个顶层模块。

**可运行例子:**
```python
import torch
import torch.nn as nn

class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.act = nn.ReLU()

model = Outer()
ch = list(model.children())
mo = list(model.modules())

assert len(ch) == 2                    # 只看直接子模块:inner、act
assert ch[0] is model.inner and ch[1] is model.act
assert model.inner.fc not in ch          # children() 完全看不到"孙子"模块

assert len(mo) == 4                     # 自己 + inner + inner.fc + act,递归展开所有层级
assert mo[0] is model                    # modules() 第一个永远是自己
assert mo[1] is model.inner
assert mo[2] is model.inner.fc            # inner 的子模块也被展开了,children() 看不到这一层
assert mo[3] is model.act

# apply() 基于 children() 递归,自底向上(先子后父),和 modules() 的自顶向下顺序刚好相反
seen = []
model.apply(lambda m: seen.append(type(m).__name__))
print("apply 遍历到的类型(自底向上):", seen)
print("modules() 遍历到的类型(自顶向下):", [type(m).__name__ for m in mo])
assert seen == ["Linear", "Inner", "ReLU", "Outer"]
assert [type(m).__name__ for m in mo] == ["Outer", "Inner", "Linear", "ReLU"]

print("Section 9 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`model.children()` 和 `model.modules()` 有什么区别?"—— 期望答出"`children()` 只返回直接子模块,不递归;`modules()` 递归返回所有层级,包含自己"。
- **追问 1:** "`model.modules()` 返回的列表里,第一个元素是什么?"—— 期望答出"是 `model` 自己",很多人会误以为第一个是"第一层子模块"。
- **追问 2(区分度很高):** "`model.apply(fn)` 内部是基于 `children()` 还是 `modules()` 实现的?调用顺序是自顶向下还是自底向上?"—— 期望能答出"基于 `children()` 递归,顺序是自底向上(所有子模块处理完,才轮到自己)",这和 `modules()` 本身自顶向下的顺序是相反的,能答出这一点说明真的读过源码而不是靠记忆规则。
- **追问 3(工程场景):** "如果我只想冻结模型最顶层的两个子模块,不想动更深层的东西,应该用 `children()` 还是 `modules()`?"—— 开放题,期望能结合两者的遍历范围差异给出合理判断(取决于"顶层"具体指代码里哪一层结构)。

**常见坑:** 把 `len(list(model.children()))` 当成"模型总共有多少层"——这个数字只统计直接子模块,一个只有 2 个直接子模块的模型完全可能内部嵌套了几十层实际计算(比如 `self.backbone`、`self.head` 两个属性,`backbone` 内部可能是几十层的 ResNet)。要统计"模型总共有多少个 `nn.Module` 对象"、或者要写通用的"遍历所有层做某件事"的代码,应该用 `modules()`,而不是对 `children()` 的结果再手动写递归。

---

## 10. 参数共享(Weight Tying)—— 两个属性名指向同一个对象

**是什么:** 让模型里两个不同的属性,直接指向**同一个** `nn.Parameter` 对象——不是数值相等的两份拷贝,是 Python 意义上的同一个对象。最经典的场景是语言模型里 embedding 层和输出投影层共享同一份权重矩阵。

**一句话:** 权重共享不需要任何专门的 API,只需要普通的 Python 赋值语句 `self.output_proj.weight = self.embedding.weight`——把已经存在的 `nn.Parameter` 对象赋给另一个属性,第 1 节讲的 `__setattr__` 机制会照常把它登记进 `_parameters`,只是这次登记的是"一个早已存在的对象",而不是新建的对象。

**底层机制/为什么这样设计:**

`nn.Embedding(vocab_size, hidden).weight` 的形状是 `(vocab_size, hidden)`;`nn.Linear(hidden, vocab_size, bias=False).weight` 的形状是 `(out_features, in_features) = (vocab_size, hidden)`——两者形状恰好一致,这不是巧合,而是"输出投影"和"embedding 查表"在数学上本来就是同一个操作的两种视角:embedding 是"用 one-hot 向量去乘权重矩阵查一行",输出投影是"用隐藏状态去乘权重矩阵的转置,给每个词算一个分数",共享同一个权重矩阵是让模型"理解一个词的方式"和"生成一个词的方式"用同一套表示,这是 GPT-2 等模型的标准做法(也是常见论文里 "tied embeddings" 的实现方式)。

实现上完全不需要框架提供任何"共享"相关的特殊接口——`self.output_proj.weight = self.embedding.weight` 这一行代码,右边取出的是已经存在的 `nn.Parameter` 对象,左边触发 `output_proj` 这个 `nn.Linear` 实例的 `__setattr__`,走的和任何一次 `self.xxx = nn.Parameter(...)` 完全相同的代码路径:识别出类型是 `Parameter`→调用 `register_parameter`,把 `output_proj._parameters["weight"]` 指向这同一个对象。两个属性名(`embedding.weight` 和 `output_proj.weight`)只是两条**不同的路径**,最终指向内存里的**同一块数据**。

**AI 研究场景:** Weight tying 除了减少参数量(对大词表模型,embedding 矩阵可能占掉总参数量的相当一部分),更重要的效果是正则化——强迫"输入侧的词表示"和"输出侧的词预测"共用一套向量空间,经验上能提升小模型的语言建模效果。这是阅读 GPT-2/nanoGPT 一类的开源实现代码时一定会遇到的模式,理解它"不是特殊机制,只是对象引用"能帮助你更快看懂类似 `self.lm_head.weight = self.transformer.wte.weight` 这样的代码在做什么、以及为什么这么写是安全的。

**可运行例子:**
```python
import torch
import torch.nn as nn

vocab_size, hidden = 10, 4

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.output_proj = nn.Linear(hidden, vocab_size, bias=False)
        # 权重共享:不是拷贝,是让两个属性名指向同一个 nn.Parameter 对象
        self.output_proj.weight = self.embedding.weight

    def forward(self, token_ids):
        h = self.embedding(token_ids)      # (batch, hidden)
        return self.output_proj(h)           # (batch, vocab_size)

model = TinyLM()

# 1) 不是"数值相等",是"同一个对象"
assert model.embedding.weight is model.output_proj.weight
assert model.embedding.weight.data_ptr() == model.output_proj.weight.data_ptr()

# 2) named_parameters() 自动去重:两个属性名都指向它,但只算一个参数,只出现一次
names = [n for n, _ in model.named_parameters()]
print("named_parameters:", names)
assert names == ["embedding.weight"]        # output_proj.weight 没有重复出现(呼应第3节的memo去重机制)
assert sum(p.numel() for p in model.parameters()) == vocab_size * hidden   # 不是2倍,是1倍

# 3) 优化一个,两边都变 —— 因为本来就是同一块内存
token_ids = torch.tensor([1, 2, 3])
loss = model(token_ids).sum()
loss.backward()
assert model.embedding.weight.grad is not None
assert model.embedding.weight.grad is model.output_proj.weight.grad   # 同一个叶子tensor,梯度也是同一份

opt = torch.optim.SGD(model.parameters(), lr=0.1)
before = model.embedding.weight.clone()
opt.step()
after_embed, after_proj = model.embedding.weight, model.output_proj.weight
assert not torch.equal(before, after_embed)
assert after_embed is after_proj             # step() 之后依然是同一个对象,不是"碰巧数值相等"

print("Section 10 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "语言模型里 embedding 层和输出层权重共享(weight tying)是怎么实现的?"—— 期望答出"直接把一个层的 `.weight` 属性赋值给另一个层的 `.weight` 属性,两者指向同一个 `nn.Parameter` 对象"。
- **追问 1(区分度很高):** "这种共享需要 PyTorch 提供什么特殊 API 吗?"—— 期望答出"完全不需要,就是普通的 Python 对象引用赋值,`nn.Module` 的 `__setattr__` 机制本身就能正确处理这种情况,没有为'共享'写任何专门代码"。
- **追问 2:** "两个共享权重的层,反向传播的时候梯度是怎么算的?会不会互相覆盖?"—— 期望答出"两条路径各自算出的梯度会累加到同一个 `.grad` 上(这是 autograd 梯度累加的通用规则,不是共享权重的特殊逻辑),因为它们本来就是同一个叶子 tensor"。
- **追问 3(容易被问倒):** "`named_parameters()` 遍历到这个共享的权重时,会把它算成一个参数还是两个?为什么?"—— 期望答出"只算一个,因为 `named_parameters()`/`named_modules()` 内部用一个 `memo` 集合按对象身份去重,同一个 `Parameter` 对象不管被几个属性名指向,只会被 `yield` 一次"——能连回第 3 节的 `_named_members`/`memo` 机制,说明知识体系是打通的。

**常见坑:** 在权重共享**之后**又对其中一个属性重新赋值成一个新的 `nn.Parameter`(比如某处代码不小心写了 `self.output_proj.weight = nn.Parameter(torch.randn(...))` 覆盖掉共享关系),这一步不会有任何报错——`__setattr__` 允许覆盖已存在的 parameter,`output_proj.weight` 会安静地指向一个新对象,`embedding.weight` 保持原样,共享关系就此断开,而代码表面上完全看不出发生了什么,只有对比 `data_ptr()` 或者 `is` 才能发现;另外要注意共享权重必须在**形状**上匹配(`nn.Embedding.weight` 是 `(vocab, hidden)`,要和 `nn.Linear.weight` 的 `(out_features, in_features)` 对上),形状不匹配的两个层不能简单共享,需要转置或额外处理。

---

## 小结:这一批 10 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `__setattr__` 自动注册 | 按赋值对象类型(`Parameter`/`Module`/`Buffer`/其他)分别登记进 `_parameters`/`_modules`/`_buffers`/`__dict__`;配套的 `__getattr__` 负责读回来 |
| 2 | `nn.Parameter` | `Tensor` 子类,`requires_grad` 默认 `True`,且被 `__setattr__` 特殊识别自动注册;普通 tensor 即使 `requires_grad=True` 也不会被注册 |
| 3 | `parameters()`/`named_parameters()` | 基于 `named_modules()` 的先序遍历实现递归;父模块自己的参数排在子模块参数之前,和 `__init__` 里赋值顺序无关 |
| 4 | `state_dict()` vs `parameters()` | 前者是带名字、含 buffer、默认 detach 的 `OrderedDict` 存档快照;后者是纯参数生成器,给优化器用 |
| 5 | `register_buffer` | 跟着 `.to(device)`/`state_dict()` 走,但 `requires_grad` 恒 `False`,不在 `parameters()` 里,不受优化器管;BatchNorm 的 running 统计量是典型例子 |
| 6 | `train()`/`eval()` | 只是把 `self.training` 递归广播到所有子模块;具体行为分叉在 Dropout/BatchNorm 各自的 `forward()` 里 |
| 7 | 三种 hook | pre-hook 在 forward 前、forward-hook 在 forward 后、backward-hook 在反向传播算到这一层时触发,常用于捕获中间激活值 |
| 8 | `ModuleList`/`ModuleDict` vs list/dict | 普通容器不被 `__setattr__` 识别,里面的层参数完全不出现在 `parameters()`/`.to(device)`/`state_dict()` 里,且不报错 |
| 9 | `children()` vs `modules()` | 前者只看直接子模块,后者递归拍平整棵树(含自己);`apply()` 基于 `children()`,顺序与 `modules()` 相反 |
| 10 | 参数共享(weight tying) | 两个属性名指向同一个 `nn.Parameter` 对象,不需要特殊 API;`named_parameters()` 按对象身份去重,只算一次 |

下一批:[04-layers-math-and-backward.md](04-layers-math-and-backward.md)

---

*更新:2026-07-07*
