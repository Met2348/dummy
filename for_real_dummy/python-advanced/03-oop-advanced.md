# 03 · 面向对象进阶(OOP Advanced)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这篇要解决的问题:大二 Python 课教了"怎么写一个能跑的 class"(基本语法层面),但没教"class 这套系统里那些进阶机关分别是干什么用的、什么时候该用"——而博士学长仓库里几乎每一个模型类、数据集类、配置类、优化器类都在用这些机关。看不懂这些,读仓库代码时会频繁卡在"这行 `@xxx` 是什么意思"上。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9)下实际跑通验证,不是凭空写的;涉及报错的地方(`TypeError`/`ValueError`/`AttributeError`),报错文案都是真实抛出后原样抄录的,不是我猜的。

---

## 0. 前置:class 极简回顾

如果你已经能熟练写 `class`、`__init__`、`self`,这一节可以跳过,直接看第 1 节。

**一句话:** Python 的 `class` 大致相当于 C 里手动维护的"一个 struct + 一堆操作它的函数,每个函数第一个参数都传这个 struct 的指针"这种模式。

用 C 类比一下,你可能写过这样的代码:

```c
struct Point {
    double x;
    double y;
};

void point_init(struct Point* p, double x, double y) {
    p->x = x;
    p->y = y;
}

double point_dist_from_origin(struct Point* p) {
    return sqrt(p->x * p->x + p->y * p->y);
}

// 使用:
struct Point p;
point_init(&p, 3.0, 4.0);
double d = point_dist_from_origin(&p);
```

注意规律:每个"操作 `Point` 的函数"第一个参数都是 `struct Point* p`。Python 的 `class` 做的事情本质上是把"struct 定义"和"这些操作函数"打包在一起,并且把那个"总是要传的指针参数"变成隐式的——这个隐式参数就叫 `self`。

对应的 Python 写法:

```python
import math

class Point:
    def __init__(self, x, y):    # __init__ 相当于上面的 point_init 函数,创建对象时自动调用
        self.x = x                # self 就是那个"隐式传入的 struct 指针"
        self.y = y

    def dist_from_origin(self):   # 相当于 point_dist_from_origin(struct Point* p)
        return math.sqrt(self.x ** 2 + self.y ** 2)

p = Point(3.0, 4.0)      # 等价于 C 里的: struct Point p; point_init(&p, 3.0, 4.0);
assert p.x == 3.0
assert p.y == 4.0
assert p.dist_from_origin() == 5.0    # 等价于 point_dist_from_origin(&p)

# p.dist_from_origin() 只是语法糖,背后其实就是 Point.dist_from_origin(p) —— self 就是 p 自己
assert p.dist_from_origin() == Point.dist_from_origin(p)
```

**对应关系表:**

| C | Python | 作用 |
|---|---|---|
| `struct Point` | `class Point:` | 定义"数据长什么样" |
| `point_init(struct Point* p, ...)` | `def __init__(self, ...):` | 创建时的初始化逻辑 |
| `struct Point* p`(每个函数的第一个参数) | `self`(每个方法的第一个参数) | 指向"当前这个实例"的隐式引用 |
| `p->x` | `self.x` | 访问实例自己的字段 |
| `point_dist_from_origin(&p)` | `p.dist_from_origin()` | 调用"操作这个实例"的函数,`self` 自动传入,不用手写 |

这就是 [02-pytorch-basics.md](../02-pytorch-basics.md) 第 4 节讲 `nn.Module` 时用的同一套类比("`nn.Module` 就像 C 里的 struct + 函数指针")。接下来整篇要讲的,就是这套"struct + 函数"系统里那些进阶机关。

---

## 1. `@property` / `@staticmethod` / `@classmethod`

**是什么:** 三个都是"装饰一个方法,改变它的调用方式或身份"的装饰器,但改变的方式完全不同——`@property` 让方法调用起来像访问属性;`@staticmethod` 声明"这个方法根本不需要 `self`,只是顺便放在类里管理";`@classmethod` 把第一个参数从"实例"换成了"类本身",常用于写"备用构造函数"。

**为什么课堂不教但很重要:** 这三个装饰器在 AI 研究代码里几乎无处不在:模型的 `.device`、`.dtype` 这类"看起来像字段、其实是实时计算出来的属性"用的是 `@property`;`AutoModel.from_pretrained(...)` 这类"备用构造函数"用的是 `@classmethod`;一些和具体实例无关、只是逻辑上归属某个类的校验/工具函数用 `@staticmethod`。你在 [02-pytorch-basics.md](../02-pytorch-basics.md) 里已经用过 `AutoModel.from_pretrained("bert-base-uncased")`,当时没解释这个 `from_pretrained` 到底是什么写法——这一节揭晓:它就是一个 `@classmethod`。

**从最笨的写法讲起:** 假设要写一个"温度"类,要求摄氏温度不能低于绝对零度(-273.15℃)。

笨办法 1:直接暴露属性,谁都能改,没有任何校验:

```python
class TemperatureNaive:
    def __init__(self, celsius):
        self.celsius = celsius

t = TemperatureNaive(25)
t.celsius = -300   # 物理上不可能,但没人阻止你这么赋值
assert t.celsius == -300   # 完全没有校验,这是个 bug,但程序不会报错
```

笨办法 2:改用 Java/C++ 那一套 `get_x()`/`set_x()` 方法,校验逻辑倒是加上了:

```python
class TemperatureGetterSetter:
    def __init__(self, celsius):
        self._celsius = celsius   # 前缀下划线表示"内部使用,不要直接碰"(约定,非强制)

    def get_celsius(self):
        return self._celsius

    def set_celsius(self, value):
        if value < -273.15:
            raise ValueError(f"温度不能低于绝对零度,收到 {value}")
        self._celsius = value

t2 = TemperatureGetterSetter(25)
t2.set_celsius(30)
assert t2.get_celsius() == 30
```

问题:调用方式从假想中的 `t.celsius` 变成了 `t2.get_celsius()` / `t2.set_celsius(30)`。如果这个类之前在几十个地方都写着 `t.celsius`,现在全部要改成方法调用,是一次破坏性变更。

正式写法:`@property` 让校验逻辑加上了,调用方式还留在 `obj.celsius`——鱼与熊掌可以兼得:

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius    # 这里其实调用的是下面的 setter

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError(f"温度不能低于绝对零度,收到 {value}")
        self._celsius = value

    @property
    def fahrenheit(self):     # 只读属性,没有对应的 setter,基于 celsius 实时算出来
        return self._celsius * 9 / 5 + 32
```

**AI 研究代码里的真实例子:**
- `@property`:HuggingFace `transformers` 的 `PreTrainedModel`(仓库路径 `learning/lora-family/official/repos/LoRA/examples/NLU/src/transformers/modeling_utils.py` 第 172-174 行)定义了 `device` 属性。`model.device` 调用起来完全像访问一个字段,但背后其实是扫描模型所有参数、实时算出它们在哪个设备上,不是一个提前存好的值。
- `@classmethod`:同一个文件第 839-840 行,`from_pretrained` 就是这么定义的:`@classmethod` `def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):`,函数体内部(第 1058 行)执行 `model = cls(config, *model_args, **model_kwargs)`。这正是 [02-pytorch-basics.md](../02-pytorch-basics.md) 里 `AutoModel.from_pretrained("bert-base-uncased")` 背后的写法——`cls` 在这里是调用时的具体模型类,内部用 `cls(...)` 构造出实例再返回,所以它是一个"备用构造函数",和普通的 `__init__` 走不同的参数入口(一个是"给配置直接建",一个是"给一个路径/名字,内部去读文件再建")。

**可运行例子:**

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError(f"温度不能低于绝对零度,收到 {value}")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9 / 5 + 32

t = Temperature(25)
assert t.celsius == 25          # 调用方式和最笨的写法一模一样
assert t.fahrenheit == 77.0     # 像访问属性一样,但背后是个方法在算

t.celsius = 30                  # 赋值时自动跑校验逻辑
assert t.celsius == 30

try:
    t.celsius = -300
    assert False, "应该抛异常"
except ValueError as e:
    print(f"捕获到预期异常: {e}")     # 温度不能低于绝对零度,收到 -300


# staticmethod / classmethod
class Dataset:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_valid_split_ratio(ratio):
        # 不需要访问任何实例数据(没有 self),只是逻辑上属于 Dataset 这个概念
        # 写成模块级普通函数在功能上完全等价,放进类里纯粹是为了组织代码
        return 0 < ratio < 1

    @classmethod
    def from_csv_path(cls, path):
        # cls 是类本身(这里是 Dataset),不是某个具体实例
        print(f"(假装)从 {path} 读取数据...")
        fake_data = [1, 2, 3]
        return cls(fake_data)     # 等价于 Dataset(fake_data)

assert Dataset.is_valid_split_ratio(0.8) is True
assert Dataset.is_valid_split_ratio(1.5) is False

ds = Dataset.from_csv_path("fake/path.csv")
assert ds.data == [1, 2, 3]
```

**常见坑:**

1. **`classmethod` 里偷懒硬编码类名,而不是用 `cls`。** 子类继承之后行为会跟预期不一致:

```python
class DatasetBad:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_list_hardcoded(cls, items):
        return DatasetBad(list(items))   # 写死类名,没用 cls

class SubDataset(DatasetBad):
    def extra_method(self):
        return "我是子类特有方法"

sub_bad = SubDataset.from_list_hardcoded([1, 2])
assert type(sub_bad) is DatasetBad           # 期望应该是 SubDataset,但因为硬编码,实际类型是父类!
assert not hasattr(sub_bad, "extra_method")  # 子类方法完全用不了
```

把 `DatasetBad(...)` 换成 `cls(...)`,子类继承后才会得到正确的子类实例。这也是为什么上面 HuggingFace 的 `from_pretrained` 要用 `cls(...)` 而不是写死某个具体模型类——不然所有具体的模型子类(`BertModel`、`GPT2Model`……)调用 `from_pretrained` 时都会得到错误的类型。

2. **`staticmethod` 和 `classmethod` 搞混。** 记忆窍门:需要"类本身"(常见于备用构造函数,内部要 `cls(...)`)用 `classmethod`;完全不需要访问类或实例、只是逻辑上归类摆放的普通函数用 `staticmethod`。如果一个 `staticmethod` 里发现自己需要构造类的实例,那多半应该改成 `classmethod`。

---

## 2. 魔法方法(dunder methods):`__init__` / `__repr__` / `__eq__` / `__len__` / `__getitem__`

**是什么:** 名字前后各带两个下划线的方法(dunder = "double underscore"),Python 不会让你显式调用它们,而是在特定语法背后自动调用。`Point(3, 4)` 背后自动调 `__init__`(第 0 节已经讲过);`print(obj)` 背后调的是 `__repr__`;`obj1 == obj2` 背后调的是 `__eq__`;`len(obj)` 背后调的是 `__len__`;`obj[i]` 背后调的是 `__getitem__`。这些方法就是"重载运算符/内置函数行为"的机制。

**为什么课堂不教但很重要:** PyTorch 的 `Dataset` 类要求你必须实现 `__len__` 和 `__getitem__`——这一点必须点清楚:**这不是 PyTorch 发明的语法,是 Python 本身的协议**(`len(obj)` 和 `obj[i]` 对任何 Python 对象都适用,PyTorch 只是要求自定义数据集类支持这两个操作,`DataLoader` 才能拿来批量取数据、shuffle)。仓库里所有自定义数据集类都是这个套路。

**从最笨的写法讲起:** 不定义任何 dunder 方法时,Python 会用最原始的默认行为:

```python
class PointNaive:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p1 = PointNaive(1, 2)
print(p1)     # <__main__.PointNaive object at 0x00000000040D6E40> —— 完全没有可读信息

p2 = PointNaive(1, 2)
assert (p1 == p2) is False    # 默认 == 比较的是"是不是同一个对象"(id),不是内容!
```

`print()` 打出一串内存地址,`==` 比较的是身份不是内容——这两个默认行为对调试和写业务逻辑基本没用,这就是为什么几乎每个正经的类都要自己定义 `__repr__` 和(在需要"内容相等"语义时)`__eq__`。

**AI 研究代码里的真实例子:** `learning/lora-family/official/repos/LoRA/examples/NLU/src/transformers/data/datasets/glue.py` 第 70 行 `class GlueDataset(Dataset):`(这里的 `Dataset` 就是 `torch.utils.data.dataset.Dataset`),第 159-163 行:

```python
def __len__(self):
    return len(self.features)

def __getitem__(self, i) -> InputFeatures:
    return self.features[i]
```

`DataLoader` 在训练循环里能做到"告诉我数据集有多大""给我第 i 条数据""帮我随机打乱、分 batch",靠的全部是这两个方法——`DataLoader` 本身完全不知道你的数据是图片还是文本,它只知道"这个对象支持 `len()` 和 `[]`"这两个 Python 协议。

**可运行例子:**

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        # print(obj)、repr(obj)、在 REPL 里直接敲变量名,都会调用这个方法
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other):
        # obj1 == obj2 背后调用的就是这个方法
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

pt1 = Point(1, 2)
pt2 = Point(1, 2)
pt3 = Point(9, 9)

assert repr(pt1) == "Point(x=1, y=2)"
print(pt1)             # 输出 Point(x=1, y=2),而不是内存地址

assert pt1 == pt2       # 现在是"内容相等"而不是"同一个对象"
assert pt1 is not pt2   # 但仍然是两个不同的对象(is 检查的是身份,不受 __eq__ 影响)
assert pt1 != pt3


class MiniDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        # len(obj) 背后调用这个方法
        return len(self._data)

    def __getitem__(self, idx):
        # obj[idx] 背后调用这个方法
        return self._data[idx] * 2   # 假装做了点预处理

ds = MiniDataset([1, 2, 3, 4])
assert len(ds) == 4     # len(ds) 触发 __len__
assert ds[0] == 2       # ds[0] 触发 __getitem__(0)
assert ds[3] == 8

# 只要定义了 __getitem__,for 循环也能直接用
# (Python 会自动从下标 0 开始递增去取,直到抛出 IndexError 才判定"迭代结束")
assert [item for item in ds] == [2, 4, 6, 8]

# 没有 __len__ 时,len() 会直接报错,不是返回 0 或别的什么
class NoLen:
    pass

try:
    len(NoLen())
    assert False, "应该报错"
except TypeError as e:
    print(f"捕获到预期异常: {e}")    # object of type 'NoLen' has no len()
```

**常见坑:** 定义了 `__eq__` 但没有定义 `__hash__`,对象会变得**不能放进 `set`,也不能当 `dict` 的 key**——因为 Python 的规则是"你自定义了 `__eq__`,就默认认为你可能改变了对象的相等语义,原来那个基于 `id()` 的默认 `__hash__` 不再安全,于是直接把 `__hash__` 设成 `None`":

```python
class PointNoHash:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, PointNoHash):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    # 注意: 没有定义 __hash__

assert PointNoHash.__hash__ is None   # Python 自动把 __hash__ 设成了 None

p = PointNoHash(1, 2)
try:
    s = {p}   # 放进 set 需要能被 hash
    assert False, "应该报错"
except TypeError as e:
    print(f"捕获到预期异常: {e}")    # unhashable type: 'PointNoHash'
```

修复方式是同时定义 `__hash__`,并且要用**和 `__eq__` 里同一组字段**来算 hash(否则会出现"两个对象 `==` 相等,但 hash 值不同"这种违反契约的情况,导致 `set`/`dict` 出现诡异的重复):

```python
class PointFixed:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, PointFixed):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))   # 用同样参与 __eq__ 比较的字段来算 hash

s = {PointFixed(1, 2), PointFixed(1, 2)}
assert len(s) == 1   # 内容相等 + hash 相等 -> set 认为是同一个元素,正确去重
```

---

## 3. 继承与 `super()`

**是什么:** 继承让一个类(子类)自动拥有另一个类(父类)的字段和方法,只需要写"和父类不一样的部分"。`super()` 是子类里用来"调用父类版本的方法"的机制,最常见的场景是 `__init__` 里先调用 `super().__init__(...)` 把父类的初始化逻辑跑一遍,再补充子类自己的初始化。

**为什么课堂不教但很重要:** [02-pytorch-basics.md](../02-pytorch-basics.md) 第 4 节你已经见过这个写法——每个自定义模型类都继承 `nn.Module`,`__init__` 第一行永远是 `super().__init__()`。当时只是要求"固定写法,必须调用",这一节讲清楚背后到底发生了什么、为什么"忘记调用"是个真实存在的坑。

**从最笨的写法讲起:** 不用继承,每个类都完整复制一遍相同的代码:

```python
class DogNoInherit:
    def __init__(self, name):
        self.name = name          # 和 CatNoInherit 里的这行代码一模一样,纯复制

    def describe(self):
        return f"我是 {self.name},{self.speak()}"  # 这行也是复制来的

    def speak(self):
        return f"{self.name} 汪汪叫"


class CatNoInherit:
    def __init__(self, name):
        self.name = name          # 又复制了一遍

    def describe(self):
        return f"我是 {self.name},{self.speak()}"  # 又复制了一遍

    def speak(self):
        return f"{self.name} 喵喵叫"

dn = DogNoInherit("旺财")
cn = CatNoInherit("咪咪")
assert dn.describe() == "我是 旺财,旺财 汪汪叫"
assert cn.describe() == "我是 咪咪,咪咪 喵喵叫"
```

`__init__` 和 `describe` 这两段代码在两个类里一字不差地重复了一遍。以后如果 `describe()` 的格式要改(比如加个时间戳),要同时改好几个类,漏改一个就会不一致——继承要解决的正是这个问题:公共逻辑写一次放在父类里,子类只写"不同的部分"。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} 发出了一些声音"

    def describe(self):
        return f"我是 {self.name},{self.speak()}"


class Dog(Animal):
    def speak(self):              # 覆盖(override)父类的方法,describe 直接继承,不用重写
        return f"{self.name} 汪汪叫"


class Cat(Animal):
    def __init__(self, name, is_indoor):
        super().__init__(name)     # 调用父类的 __init__,把"设置 name"这部分逻辑复用
        self.is_indoor = is_indoor  # 子类自己新增的字段

    def speak(self):
        return f"{self.name} 喵喵叫"
```

**AI 研究代码里的真实例子:** [02-pytorch-basics.md](../02-pytorch-basics.md) 里的 `LinearLayer`/`Adapter` 都是这个模式——继承 `nn.Module`,`__init__` 第一行 `super().__init__()`,再定义自己的层。`nn.Module.__init__()` 内部实际做的事情是初始化几个内部字典(用来登记这个模型有哪些子层、哪些可训练参数),后面 `self.linear = nn.Linear(...)` 这种赋值之所以能被 `nn.Module` "自动发现并登记为子模块",全靠 `super().__init__()` 提前把这套登记机制建好。

**可运行例子:**

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} 发出了一些声音"

    def describe(self):
        return f"我是 {self.name},{self.speak()}"


class Dog(Animal):
    def speak(self):              # 覆盖(override)父类的方法,describe 直接继承,不用重写
        return f"{self.name} 汪汪叫"


class Cat(Animal):
    def __init__(self, name, is_indoor):
        super().__init__(name)     # 调用父类的 __init__,把"设置 name"这部分逻辑复用
        self.is_indoor = is_indoor  # 子类自己新增的字段

    def speak(self):
        return f"{self.name} 喵喵叫"


a = Animal("神秘生物")
d = Dog("旺财")
c = Cat("咪咪", is_indoor=True)

assert a.speak() == "神秘生物 发出了一些声音"
assert d.speak() == "旺财 汪汪叫"
assert c.speak() == "咪咪 喵喵叫"

# describe() 没有在子类里重新定义,直接继承自 Animal,
# 但里面调用的 self.speak() 会根据实际类型决定用哪个版本(多态/动态分派)
assert d.describe() == "我是 旺财,旺财 汪汪叫"
assert c.describe() == "我是 咪咪,咪咪 喵喵叫"
assert c.is_indoor is True

assert isinstance(d, Animal)   # Dog 也是一种 Animal
assert isinstance(c, Animal)
```

**常见坑:** 忘记调用 `super().__init__()`。下面用一个简化版模拟 `nn.Module` 的真实行为——父类在 `__init__` 里做了"内部登记",子类如果不调用 `super().__init__()`,这个登记就不会发生:

```python
class FakeModuleBase:
    def __init__(self):
        self._registered_params = {}   # 模拟 nn.Module 内部维护的参数登记表

    def register(self, name, value):
        self._registered_params[name] = value


class GoodLayer(FakeModuleBase):
    def __init__(self):
        super().__init__()          # 正确: 先让父类完成它的初始化
        self.register("weight", 1.0)


class BadLayer(FakeModuleBase):
    def __init__(self):
        # 忘记调用 super().__init__() —— _registered_params 根本不存在
        self.register("weight", 1.0)


good = GoodLayer()
assert good._registered_params == {"weight": 1.0}

try:
    bad = BadLayer()
    assert False, "应该在 register 内部访问 self._registered_params 时报错"
except AttributeError as e:
    print(f"捕获到预期异常: {e}")   # 'BadLayer' object has no attribute '_registered_params'
```

这不是凭空模拟的场景——真的用 PyTorch 的 `nn.Module` 复现一遍会得到同样形状的报错(以下代码同样已在仓库 `.venv` 里用真实 torch 2.11.0 实际跑过):

```python
import torch.nn as nn

class BadPyTorchLayer(nn.Module):
    def __init__(self):
        # 忘记 super().__init__()
        self.linear = nn.Linear(4, 2)

try:
    layer = BadPyTorchLayer()
    assert False
except AttributeError as e:
    print(f"真实 PyTorch 报错: {e}")   # cannot assign module before Module.__init__() call
```

---

## 4. `@dataclass`

**是什么:** 一个类装饰器,根据你写的字段类型注解,自动生成 `__init__`、`__repr__`、`__eq__` 这些样板方法——你只需要声明"有哪些字段",不用手写一行行的 `self.x = x`。

**为什么课堂不教但很重要:** 训练配置/超参数类几乎总是用 `@dataclass` 定义——这类类往往有几十上百个字段(学习率、batch size、epoch 数、要不要开混合精度……),如果手写 `__init__`,光是 `self.xxx = xxx` 就要重复几十行。HuggingFace `transformers` 的 `TrainingArguments` 就是这个模式(仓库路径 `learning/lora-family/official/repos/LoRA/examples/NLU/src/transformers/training_args.py` 第 58-59 行:`@dataclass` `class TrainingArguments:`)。

**从最笨的写法讲起:**

```python
class TrainingConfigManual:
    def __init__(self, lr, batch_size, epochs, model_name="bert-base"):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name

    def __repr__(self):
        return (f"TrainingConfigManual(lr={self.lr}, batch_size={self.batch_size}, "
                f"epochs={self.epochs}, model_name={self.model_name!r})")

    def __eq__(self, other):
        if not isinstance(other, TrainingConfigManual):
            return NotImplemented
        return (self.lr, self.batch_size, self.epochs, self.model_name) == \
               (other.lr, other.batch_size, other.epochs, other.model_name)
```

只有 4 个字段,就已经写了 `__init__` + `__repr__` + `__eq__` 三段纯样板代码(每个字段在三处地方各出现一次)。真实的 `TrainingArguments` 有上百个字段,手写方式的重复量会直接爆炸,而且每加一个字段要同步改三个地方,极容易漏改。

正式写法:

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    model_name: str = "bert-base"     # 可以给默认值,和普通函数参数一样
```

字段声明一次,`__init__`/`__repr__`/`__eq__` 全部自动生成。

**AI 研究代码里的真实例子:**
- `training_args.py` 第 58-59 行的 `TrainingArguments` 本尊,以及第 372 行:`logging_dir: Optional[str] = field(default_factory=default_logdir, ...)`——用到了下面"常见坑"里会讲的 `default_factory`。
- `learning/auto-research-frontier/m9.1-autonomy-ladder-and-map/src/autonomy_map/systems.py` 第 24-32 行的 `@dataclass(frozen=True)` `class System:` 是你自己在 auto-research 系列里已经跑过的模块——`frozen=True` 让实例创建后字段不能再被改,适合"配置一旦定下来就不该被改"的场景。
- 仓库自己的 `learning/agent-code-eval/src/webarena_mock.py` 第 17-21 行,是"可变默认值陷阱"(下面常见坑会讲到)的正确写法,而不是 `cart_items: List[str] = []`:

```python
from typing import List
from dataclasses import dataclass, field

@dataclass
class ShopState:
    page: str = "home"
    cart_items: List[str] = field(default_factory=list)
    confirmed: bool = False
```

**可运行例子:**

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    model_name: str = "bert-base"

cfg1 = TrainingConfig(lr=1e-4, batch_size=32, epochs=10)
cfg2 = TrainingConfig(lr=1e-4, batch_size=32, epochs=10)

assert cfg1.lr == 1e-4                     # __init__ 自动生成了
assert repr(cfg1) == "TrainingConfig(lr=0.0001, batch_size=32, epochs=10, model_name='bert-base')"  # __repr__ 自动生成了
assert cfg1 == cfg2                        # __eq__ 自动生成了(按字段值比较,不是 id)
assert cfg1 is not cfg2
```

**常见坑:** 可变默认值(mutable default)陷阱。直接给字段写 `= []` 当默认值:

```python
from dataclasses import dataclass

@dataclass
class BadConfig:
    layers: list = []   # 直接写 [] 当默认值
```

这段代码**在类定义那一行就直接报错**,不用等到实例化(本机实测,原样抄录):

```
ValueError: mutable default <class 'list'> for field layers is not allowed: use default_factory
```

`dataclass` 专门做了这层检查,因为如果真的允许 `= []`,所有实例会共享同一个 list 对象(和下面这个普通函数的经典陷阱是同一个根因)——`dataclass` 选择直接报错而不是放任这个坑存在。正确写法是用 `field(default_factory=list)`,让每个实例创建时单独调用一次 `list()` 拿到一个全新的空列表:

```python
from dataclasses import dataclass, field

@dataclass
class GoodConfig:
    layers: list = field(default_factory=list)

g1 = GoodConfig()
g2 = GoodConfig()
g1.layers.append("attention")

assert g1.layers == ["attention"]
assert g2.layers == []      # g2 不受 g1 影响: default_factory 保证每个实例拿到独立的新 list
```

对比一下:如果是**普通函数**(不是 dataclass)用可变对象当默认参数,Python **不会报错**,但行为一样诡异——这是 Python 最经典的陷阱之一,`dataclass` 的强制检查本质上就是在帮你提前拦截这类 bug:

```python
def append_to(item, target=[]):   # 默认值只在函数定义时创建一次,不是每次调用都创建新的
    target.append(item)
    return target

r1 = append_to(1)
r2 = append_to(2)
assert r1 is r2          # 是同一个 list 对象!
assert r2 == [1, 2]      # 不是期望的 [2] —— r1 的结果"泄漏"到了 r2 里
```

---

## 5. 抽象基类 `abc.ABC`

**是什么:** `abc.ABC`(Abstract Base Class)让你在父类里用 `@abstractmethod` 标记"子类必须自己实现的方法"。如果子类没有实现全部标记的方法,**实例化那一刻就直接报错**,不需要等到真正调用那个方法才发现问题。

**为什么课堂不教但很重要:** 研究代码库里经常要定义"一整个家族的类都必须遵守同一套接口"——比如"所有优化器都必须有 `step()` 方法""所有 tokenizer 都必须有 `encode()`/`decode()`"。`abc.ABC` 就是拿来写这种"强制契约"的标准工具。仓库里 `learning/rl-foundations/official/repos/baselines/baselines/common/vec_env/vec_env.py` 第 29 行的 `class VecEnv(ABC):` 就是这么写的,第 48-49、60-61、72-73 行分别用 `@abstractmethod` 标记了 `reset`、`step_async`、`step_wait` 三个方法——所有具体的并行环境实现都必须提供这三个方法,强化学习训练代码才能不管具体是哪种环境,统一调用这三个接口。

**从最笨的写法讲起:** 不用 `abc.ABC`,只在父类方法里手动 `raise NotImplementedError`,靠"自觉"和文档约定:

```python
class OptimizerNaive:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        raise NotImplementedError("子类必须实现 step 方法")


class ForgetfulOptimizer(OptimizerNaive):
    pass   # 忘记实现 step 了,但这行代码本身完全不会报错


opt = ForgetfulOptimizer(lr=0.1)   # 实例化成功,看起来一切正常!
try:
    opt.step(params=[1.0], grads=[0.1])   # 只有真正调用 step 的时候才炸
    assert False
except NotImplementedError as e:
    print(f"直到调用时才报错: {e}")   # 子类必须实现 step 方法
```

问题:`ForgetfulOptimizer(lr=0.1)` 这行本身完全没有报错,程序看起来一切正常。如果 `step()` 是训练循环跑到某个不常见分支才第一次被调用到,这个"忘记实现"的 bug 可能要等程序跑了很久才暴露。

**AI 研究代码里的真实例子:** 上面提到的 `VecEnv(ABC)`。用 `abc.ABC` 之后,任何一个忘记实现 `reset`/`step_async`/`step_wait` 的子类,在被创建(`__init__`)那一刻就会直接报错——问题在"写错代码"和"第一次跑到那个分支"之间的间隔被压缩没了,这就是"强制契约"相对"自觉遵守"的核心价值:**越早报错,排查成本越低**。

**可运行例子:**

```python
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def step(self, params, grads):
        """子类必须实现: 根据梯度更新参数"""
        ...

    def zero_grad_reminder(self):
        # 普通方法,子类可以直接继承使用,不强制覆盖
        return "别忘了每个 step 之前调用 zero_grad()"


# 不写 step 会怎样? —— 实例化那一刻就直接报错,不用等到调用
class BrokenOptimizer(Optimizer):
    pass   # 没有实现 step

try:
    broken = BrokenOptimizer(lr=0.01)
    assert False, "应该在实例化时就报错,而不是等到调用 step 才发现"
except TypeError as e:
    print(f"捕获到预期异常: {e}")
    # Can't instantiate abstract class BrokenOptimizer without an implementation for abstract method 'step'

# 写了 step 之后才能正常实例化
class SGD(Optimizer):
    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]

sgd = SGD(lr=0.1)
assert sgd.step(params=[1.0, 2.0], grads=[0.1, 0.2]) == [0.99, 1.98]
assert sgd.zero_grad_reminder() == "别忘了每个 step 之前调用 zero_grad()"
```

**常见坑:** 抽象基类本身(哪怕所有抽象方法后来都被写了)也不能被直接实例化——只要 `Optimizer` 还有没实现的 `@abstractmethod`,`Optimizer(lr=0.1)` 这行本身就会报错,报错文案和上面 `BrokenOptimizer` 那次几乎一样(本机实测):

```python
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def step(self, params, grads):
        ...

try:
    o = Optimizer(lr=0.1)   # Optimizer 自己也没有实现 step,一样不能实例化
    assert False
except TypeError as e:
    print(f"捕获到预期异常: {e}")
    # Can't instantiate abstract class Optimizer without an implementation for abstract method 'step'
```

容易搞混的一点:`abc.ABC` 只检查"有没有实现该方法(方法名是否存在)",不检查"实现得对不对"——签名不对、逻辑写错了,`abc.ABC` 都不会报错,它只负责最低限度的"有没有这个方法"把关,不是自动测试工具。

---

## 小结

| # | 知识点 | 解决的问题 |
|---|---|---|
| 0 | class 极简回顾 | class = struct + 操作它的函数打包,self = 隐式传入的"指针" |
| 1 | `@property`/`@staticmethod`/`@classmethod` | 属性访问加校验且不改调用方式 / 逻辑挂靠类但不需要实例 / 备用构造函数 |
| 2 | 魔法方法(dunder methods) | `print`/`==`/`len`/`obj[i]` 等内置语法背后自动调用的协议方法 |
| 3 | 继承与 `super()` | 公共逻辑写一次,子类复用父类初始化,不重复代码 |
| 4 | `@dataclass` | 自动生成 `__init__`/`__repr__`/`__eq__`,配置类/超参数类标配 |
| 5 | 抽象基类 `abc.ABC` | 强制子类实现约定接口,实例化时(而不是调用时)就报错 |

下一批:[04-typing-context-and-concurrency.md](04-typing-context-and-concurrency.md)

---

*更新:2026-07-07*
