# 04 · Keras 模型构建三套 API 内核(Keras Model-Building API Internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲 Keras 到底是什么。表面上"三套 API"(Sequential/Functional/Subclassing)只是"写模型的三种风格",面试真正想挖的是:这三套 API 底层共享的是同一套什么样的地基?为什么 Functional API 拼出来的模型不需要喂任何一次真实数据,就能打印出完整的层级拓扑图,而写法上和 PyTorch `nn.Module` 几乎一模一样的 Subclassing API 做不到?`model(x)` 和你在 `call()` 里写的那行 `return ...` 之间,`training` 参数是怎么"凭空"传进来的?一个层为什么不在 `__init__` 里就把权重建好,非要拖到第一次调用才建?想清楚这几个问题,才能理解 Keras 3 年前那场"`tf.keras` 到底指向谁"的架构变动,以及为什么今天大多数面试和教程默认你活在 Keras 2 的心智模型里。

**本文和 torch-deep-dive 的关系:** [torch-deep-dive/03-nn-module-internals.md](../torch-deep-dive/03-nn-module-internals.md) 讲透了 `nn.Module`——PyTorch 建模系统的地基:`__setattr__` 怎么把参数"收编"进 `_parameters`/`_modules` 字典,`parameters()`/`state_dict()`/`train()` 这些"一键操作整个模型"的方法怎么靠递归遍历这几个字典实现。本篇要在 TensorFlow 这一侧回答同一个问题:Keras 的地基是什么?答案是 `tf.Module`——但它解决"怎么知道模型有哪些变量"这个问题的方式,和 PyTorch 的 `__setattr__` 拦截式登记完全不是同一套思路(第 1 节详细展开这个对比)。理解了第 1 节之后再看后面 6 节会发现:Sequential/Functional/Subclassing 三套 API,归根结底都只是"以不同方式生产一个 `tf.Module` 实例"的三条路径,区别只在于"框架能替你自动做多少事"和"你愿意用多少自由度换这些自动化"。

**本篇统一结构(与 00-roadmap.md 定义一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究/工程场景(本系列这一段落是根据真实训练/部署场景重构的场景化例子,不是仓库代码引用,完整声明见 [00-roadmap.md](00-roadmap.md)"0. 环境声明"最后一段)
5. 可运行例子(assert 验证 + 现场内省打印;WSL2 `~/tf-venv` 下 TensorFlow 2.21.0 实测跑通,`source ~/tf-venv/bin/activate` 已自动设置 `TF_USE_LEGACY_KERAS=1`,`tf.keras` 锁定解析到 Keras 2 实现)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.Module` —— Keras `Layer`/`Model` 的真正基类

**是什么:**
```python
# tf.Module(name=None)
# —— TensorFlow 提供的、比 Keras 更底层的"带变量管理能力的容器基类"
```

**一句话:** 呼应 torch-deep-dive 对 `nn.Module` 的定位——那里 `nn.Module` 是 PyTorch 整个建模系统的地基,这里 `tf.Module` 是 TensorFlow(不只是 Keras)整个建模系统的地基;区别在于 Keras 不是"重新发明一遍地基",而是让 `tf_keras.layers.Layer` 直接继承 `tf.Module`——**Keras 是搭在 `tf.Module` 之上的一层糖衣**,`build()`/`call()`/`fit()` 这些"Keras 味道"很重的机制,全部是在 `tf.Module` 已经提供的变量管理能力上叠加出来的。

**底层机制/为什么这样设计:**

现场验证继承关系,不是转述文档:
```python
import tensorflow as tf
import tf_keras

mro_names = [c.__name__ for c in tf_keras.layers.Layer.__mro__]
print("Layer MRO:", mro_names)
assert mro_names == [
    "Layer", "Module", "AutoTrackable", "Trackable", "LayerVersionSelector", "object",
]
assert issubclass(tf_keras.layers.Layer, tf.Module)          # Layer 是 tf.Module 的子类,不是平行实现
assert issubclass(tf_keras.Model, tf_keras.layers.Layer)      # Model 又是 Layer 的子类,三层继承
print("PASS: tf_keras.layers.Layer 直接继承 tf.Module,不是重新发明一套追踪机制")
```

`tf.Module` 要解决的核心问题和 `nn.Module` 完全一样:一个模型是"属性套属性"任意嵌套出来的树状结构,框架需要某种机制,能从根对象出发自动找到所有分散在这棵树里的 `tf.Variable`。但**实现思路和 PyTorch 完全不同**,这是本节最值得记住的对比:

- PyTorch 的 `nn.Module.__setattr__`(torch03 第 1 节)是**写时登记**:每次 `self.xxx = yyy`,当场判断类型、分类存进 `_parameters`/`_modules`/`_buffers` 三个专用字典,`parameters()` 之后只需要遍历这几个提前分好类的字典,一个是"贵在写、便宜在读"的设计。
- `tf.Module` 恰好相反,是**读时反射**:它没有为 `Variable`/`Module` 类型准备专用分类字典,`.variables`/`.submodules` 每次被访问,都现场调用内部的 `_flatten()` 方法,用反射重新扫描一遍整棵对象树的所有属性,按类型过滤出符合条件的叶子。`tf.Module.variables` 的官方文档字符串里明确写着"for performance reasons you may wish to cache the result of calling this method if you don't expect the return value to change"——这句提示直接点破了这是一个**每次调用都要重新扫描、有真实开销**的操作,不是 O(1) 查字典。

这个差异带来一个可以现场验证的推论:**PyTorch 里"把子层存进普通 `list` 而不是 `nn.ModuleList`,会让参数完全隐身"的经典陷阱(torch03 第 8 节),在 `tf.Module`/Keras 这边根本不成立**。原因是 `tf.Module` 的父类 `AutoTrackable` 也重写了 `__setattr__`,但职责不是"只认几个特定类型分类存字典",而是把赋值的 `list`/`dict` **透明包装**成同样可追踪的 `ListWrapper`/`_DictWrapper`(服务于 `tf.train.Checkpoint`),原生容器本身不再是追踪的障碍。

**AI 研究/工程场景:** 从 PyTorch 转过来的人经常带着"一定要用官方容器类包住子层列表"的肌肉记忆,在 TF/Keras 这边这个顾虑大部分时候是不必要的——用普通 `list`/`dict` 存一组子层,`.variables`/`.trainable_variables`/`.to()`(Keras 层面是 `.weights`/`.get_weights()`)照样能找到。但也不代表 TF 这边完全没有等价的"隐身"风险:如果你把变量存进一个自定义的、不是 `list`/`dict`/`tf.Module` 的容器对象里(比如一个手写的小 wrapper 类),`AutoTrackable` 的透明包装机制不认识它,反射一样会漏掉——判断标准和 PyTorch 一样朴素:这个属性最终有没有被框架的追踪机制"看见",不能想当然。

**可运行例子:**
```python
import tensorflow as tf

# --- 1) 读时反射 vs 写时登记:绕过 __setattr__ 追踪,.variables 依然能找到 ---
m = tf.Module()
object.__setattr__(m, "sneaky", tf.Variable(7.0, name="sneaky"))  # 绕过 AutoTrackable.__setattr__
assert "sneaky" in m.__dict__
assert len(m.variables) == 1 and m.variables[0].name == "sneaky:0"
print("绕过追踪逻辑直接写 __dict__,.variables 依然找到了 —— 证明是读时反射,不是写时登记")

# --- 2) 普通 python list/dict 里的 tf.Variable 会被自动发现(对比 PyTorch 的经典陷阱) ---
class Container(tf.Module):
    def __init__(self):
        super().__init__()
        self.plain_list = [tf.Variable(1.0, name="v1"), tf.Variable(2.0, name="v2")]
        self.plain_dict = {"a": tf.Variable(3.0, name="v3")}
        self.w = tf.Variable(4.0, name="w")

c = Container()
found_names = sorted(v.name for v in c.variables)
assert found_names == ["v1:0", "v2:0", "v3:0", "w:0"]
print("普通 list/dict 里的变量全部被发现:", found_names)

# --- 3) name_scope 不是自动生效的,必须显式 with self.name_scope 包住变量创建 ---
class Scoped(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w_plain = tf.Variable(1.0, name="w")           # 不进 name_scope
        with self.name_scope:
            self.w_scoped = tf.Variable(1.0, name="w")        # 进 name_scope

s = Scoped(name="my_mod")
assert s.w_plain.name == "w:0"                    # 没有模块名前缀
assert s.w_scoped.name.startswith("my_mod/")        # 有模块名前缀
print("不进 name_scope:", s.w_plain.name, " | 进 name_scope:", s.w_scoped.name)

# --- 4) 嵌套模块:submodules / variables / trainable_variables ---
class Inner(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(1.0, trainable=True, name="inner_w")
        self.frozen = tf.Variable(9.0, trainable=False, name="inner_frozen")

class Outer(tf.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.top = tf.Variable(2.0, trainable=True, name="top")

o = Outer()
assert o.submodules == (o.inner,)
assert [v.name for v in o.variables] == ["top:0", "inner_frozen:0", "inner_w:0"]   # 自己的变量在前,子模块的在后
assert [v.name for v in o.trainable_variables] == ["top:0", "inner_w:0"]            # frozen 被过滤掉
print("submodules:", o.submodules)
print("variables 顺序(自己的在前,子模块的在后):", [v.name for v in o.variables])

print("Section 1 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.keras.layers.Layer` 和 `tf.Module` 是什么关系?"—— 期望答出"`Layer` 直接继承自 `tf.Module`,`tf.Module` 才是 TensorFlow 变量管理机制真正的地基"。
- **追问 1(区分度很高):** "`tf.Module` 是怎么知道 `self.w = tf.Variable(...)` 这个 `w` 应该被 `.variables` 收集到的?和 PyTorch `nn.Module` 的 `__setattr__` 拦截机制是同一个思路吗?"—— 期望准确答出"不是,`tf.Module` 靠的是 `.variables` 被访问时现场反射扫描对象树,不是赋值时分类登记进专用字典"。能进一步说出"这意味着 `.variables` 是有实际开销的、每次都重新算的操作,不是查字典",说明是真正理解而不是背概念。
- **追问 2(能验证是否只是背答案):** "如果我故意绕过 `__setattr__`,直接操作 `self.__dict__` 塞一个变量进去,`tf.Module` 的追踪机制还能不能发现它?PyTorch 那边呢?"—— 期望说出"TF 这边因为是读时反射,只要变量真实存在于对象属性里就能被扫到,绕不开;PyTorch 那边因为依赖 `__setattr__` 主动分类登记,绕过之后 `_parameters` 里就是空的,`parameters()` 找不到"。
- **追问 3:** "PyTorch 里有个经典陷阱:子层存进普通 `list` 而不是 `nn.ModuleList`,参数会完全隐身。这个陷阱在 TF/Keras 里存在吗?"—— 期望答出"不存在(或者说不是同一种形式),因为 `AutoTrackable.__setattr__` 会把普通 `list`/`dict` 透明包装成可追踪的 wrapper,原生容器不是追踪的障碍"——这个问题专门用来判断候选人是不是"死记 PyTorch 的坑,不问青红皂白套到 TF 上"。

**常见坑:** 反过来,从 TF/Keras 转向 PyTorch 的人容易低估"容器类型"这件事在 PyTorch 里的重要性——TF 这边"随手用 list 存子层"几乎从来不会出问题,容易带着"框架应该都会自动处理好"的预期去写 PyTorch 代码,结果在 `nn.ModuleList` 这个坑上摔一跤。另外一个真实的坑是**滥用 `.variables`/`.trainable_variables` 做高频操作**——比如写在训练循环内部每个 step 都调用一次,因为是反射扫描不是查字典,模型层数一多会有实际可测量的性能损耗,官方文档已经明确提示"建议缓存结果";正确做法是在训练循环外调用一次、缓存成一个 list,循环内部复用这个缓存。

---

## 2. Sequential API —— 最简单,但表达力有限的一种

**是什么:**
```python
# tf.keras.Sequential(layers=None, name=None)
# —— 把层按顺序首尾相连,上一层的输出自动作为下一层的输入
```

**一句话:** `Sequential` 是三套 API 里限制最多、代码最少的一种——只能表达"一条直线到底"的拓扑(单输入单输出、每层只能被用一次、不能有分支或跳连),换来的是`keras.Sequential([layer1, layer2, ...])` 这一行就能定义一个模型。

**底层机制/为什么这样设计:**

`Sequential` 在类继承链上直接继承自 `Functional`(下一节详细讲的类),不是一个平行的独立实现:
```python
import tf_keras as keras

mro_names = [c.__name__ for c in keras.Sequential.__mro__]
print("Sequential MRO:", mro_names)
assert "Functional" in mro_names
assert mro_names.index("Sequential") < mro_names.index("Functional") < mro_names.index("Model")
print("PASS: Sequential 是 Functional 的子类,不是独立实现")
```

这个继承关系不是巧合:`Sequential` 内部维护一个层列表,只要输入 shape 是已知的(比如第一层写了 `input_shape=`,或者已经喂过一次真实数据),它就会在背后用第 3 节要讲的 Functional/KerasTensor 机制,把这条线性链条组装成一个真正的计算图。这解释了一个反直觉的现象:`Sequential` 模型不需要你手写 `call()`,却依然拥有 `.summary()` 打印完整拓扑的能力——因为它本质上是"自动帮你调用了一遍 Functional API"的语法糖,不是靠单独实现了一套拓扑记录逻辑。

反过来,如果构造 `Sequential` 时**不知道输入 shape**(没写 `input_shape`,也没有 `keras.Input` 打头),模型会保持"未构建"状态,直到第一次真实调用发生,内部才会补上这一步 Functional 组装——这一点和第 6 节要讲的 `build()` 延迟构建是同一个"能拖就拖,拖到信息齐全为止"的设计哲学。

**AI 研究/工程场景:** Sequential 最适合的场景是"确定没有分支需求"的部分——比如一个 CNN 分类器的 backbone 是标准的卷积堆叠、或者一个 MLP 探针(probing head)只是几层全连接加激活函数。一旦需求出现"多输入"(比如图文双塔模型)、"跳连"(ResNet 的 shortcut)、"共享层被调用多次"(Siamese 网络两次调用同一个 encoder),Sequential 立刻表达不了,必须换 Functional 或 Subclassing——这也是为什么真实的研究代码里 Sequential 往往只出现在模型的某个子部件上,而不是整个模型。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf

# --- 1) 明确知道输入shape:构造完立刻built,summary()立刻可用 ---
model_known = keras.Sequential([
    keras.layers.Dense(8, activation="relu", input_shape=(4,)),
    keras.layers.Dense(3),
])
assert model_known.built is True
print("已知input_shape,构造完立刻built:", model_known.built)

# --- 2) 不知道输入shape:未built,summary()报错;调用一次之后才built ---
model_lazy = keras.Sequential([
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3),
])
assert model_lazy.built is False
try:
    model_lazy.summary()
    assert False, "不应该走到这里"
except ValueError as e:
    assert "has not yet been built" in str(e)
    print("未built时summary()报错:", str(e))

x = tf.random.normal((2, 4))
_ = model_lazy(x)                      # 真实调用一次,触发延迟构建
assert model_lazy.built is True

# --- 3) built之后,Sequential 内部其实已经组装出了 Functional 风格的 KerasTensor 拓扑 ---
from tf_keras.src.engine.keras_tensor import KerasTensor
assert isinstance(model_lazy.inputs[0], KerasTensor)
assert isinstance(model_lazy.outputs[0], KerasTensor)
print("model_lazy.inputs:", model_lazy.inputs)
print("model_lazy.outputs:", model_lazy.outputs)

print("Section 2 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`Sequential` API 有什么限制?为什么不能表达 ResNet 的跳连结构?"—— 期望答出"只能表达单输入单输出的线性链条,每层只能被使用一次,没法表达多输入/多输出/层复用/跳连"。
- **追问 1(区分度很高):** "`Sequential` 需要你手写 `call()` 方法吗?它是怎么知道整个模型的计算顺序、还能打印出 `.summary()` 拓扑的?"—— 期望答出"不需要,`Sequential` 本身是 `Functional` 的子类,添加层的时候如果输入 shape 已知,内部会自动用 Functional/KerasTensor 机制把这条线性链条组装成真正的计算图,`.summary()` 打印的是这个自动组装出来的图,不是靠单独实现"。
- **追问 2:** "如果 `Sequential` 第一层没写 `input_shape`,构造完之后立刻调用 `.summary()` 会发生什么?为什么?"—— 期望说出"报错说模型还没 built;因为没有输入 shape 信息,没法组装出具体的计算图,必须等第一次真实调用(或显式 `build()`)才能补齐"。
- **追问 3(开放题):** "什么情况下你会选择 Sequential 而不是 Functional,即便 Functional 能表达的东西更多?"—— 期望能说出"当子模块结构确实是纯线性的时候,Sequential 语法更短、更直接,没必要为了'更强大'而牺牲可读性,两者不是互斥关系,经常在同一个模型里搭配用(比如整体是 Functional,某个子部件用 Sequential 搭 backbone)"。

**常见坑:** 在 `Sequential` 已经 `built` 之后,试图 `.add()` 一个输入维度不匹配的新层,或者中途 `.pop()` 之后忘记模型的 `built`/输出 shape 状态可能没有完全按预期回退——`Sequential` 的"渐进式构建"机制在标准的"一次性 `.add()` 完再使用"场景下很可靠,但如果代码逻辑里有"构建一部分、跑一次、再继续 `.add()`"这种交错模式,容易遇到状态不一致的问题,不如老老实实一次性定义完再使用。另外一个坑是**误以为 `input_shape` 参数属于 `Dense` 层本身**——它实际上是 Keras 在 `Sequential` 语境下识别的一个特殊 kwarg,用来隐式插入一个 `Input` 层,单独拿 `Dense(8, input_shape=(4,))` 在 Functional/Subclassing 语境里使用行为并不直观,不建议在 Sequential 之外的场景沿用这个写法。

---

## 3. Functional API 与 KerasTensor 符号追踪

**是什么:**
```python
# inputs = tf.keras.Input(shape=(4,))     # 创建一个 KerasTensor(符号占位符,不含真实数值)
# outputs = SomeLayer()(inputs)            # 层作用在 KerasTensor 上,返回新的 KerasTensor
# model = tf.keras.Model(inputs, outputs)  # 把输入输出对包装成一个模型
```

**一句话:** Functional API 的核心机制是 `keras.Input()` 产生的 `KerasTensor`——一种**不含真实数值、只携带 shape/dtype 和"我是怎么来的"这段历史信息的符号张量**;层作用在 `KerasTensor` 上时不会做"看起来像是在算数值"的普通调用,而是走一条专门的符号构图路径,把整个模型的连接关系(DAG)记录成一份真实的数据结构——这份记录正是 `.summary()` 能打印出完整拓扑的原因。

**底层机制/为什么这样设计:**

`Layer.__call__` 内部会先判断"这次调用的输入是不是符号化的 KerasTensor",如果是,直接切换到完全不同的代码路径:
```python
import tf_keras as keras
import inspect

src = inspect.getsource(keras.layers.Layer.__call__)
assert "_in_functional_construction_mode" in src
assert "_functional_construction_call" in src
idx = src.find("_in_functional_construction_mode(")
print(src[max(0, idx-260):idx+160])
print("PASS: __call__ 内部真实存在这条符号构图分支判断")
```

这条分支存在的意义,是把"构建模型拓扑"和"真实跑数据"这两件事在代码路径上彻底分开。当你写 `h1 = Dense(8)(inp)`(`inp` 是 `KerasTensor`):
1. `Dense` 层的 `build()` **真的会被调用**,一个真实的 `tf.Variable` 权重矩阵被创建出来(不是占位符,是货真价实的变量);
2. `Dense` 层的 `call()` 方法**也真的会执行一次**——但输入不是你的真实数据,而是一个在临时 `FuncGraph`(一次性刮痧用的计算图,和 `tf.function` trace 用的机制同源)里创建的占位 `SymbolicTensor`,`call()` 里的 `tf.matmul` 等操作会在这个临时图里真实运行一遍,用来推导出正确的输出 shape/dtype;
3. 输出被包装成一个新的 `KerasTensor`,并被打上 `_keras_history = KerasHistory(layer, node_index, tensor_index)` 标记,同时 `Dense` 层内部的 `_inbound_nodes` 列表新增一个 `Node` 对象,记录"这次调用的输入 KerasTensor 是谁、输出 KerasTensor 是谁"。

`keras.Model(inputs, outputs)` 构造时,就是**从 `outputs` 的 `_keras_history` 反向遍历回 `inputs`**,把沿途经过的每一个 `Node` 串起来,重建出完整的层连接 DAG。`.summary()` 的"Connected to"列,打印的就是这份真实存在的 DAG 数据,不是猜出来的。

对比之下,Subclassing API 的 `call()` 是纯粹的 Python 命令式代码:框架只知道"这个 `Model` 对象通过 `tf.Module` 反射机制(第 1 节)找到了哪些子层属性",完全不知道 `call()` 内部**以什么顺序、调用了几次、把谁的输出传给了谁**——因为这些信息只有在真实执行 `call()` 的那一刻才存在,而且下一次调用完全可能因为 `if`/`for` 走了不同分支而变成不同的连接方式,根本没有一个"固定不变"的 DAG 可言。这就是"为什么 Functional 能自动打印完整拓扑,Subclassing 不行"的根本原因——不是 Keras 不想支持,而是 Subclassing 的设计初衷(用任意 Python 控制流写 `call()`)和"存在一份静态、确定的 DAG"这件事在逻辑上互斥。

**AI 研究/工程场景:** 需要用 `plot_model()`/`.summary()` 做架构文档、或者需要拿到模型中间某个具名张量做多任务学习的辅助输出头(`keras.Model(inputs, [main_out, aux_out])`)时,Functional API 几乎是唯一自然的选择——这些能力全部建立在"存在一份真实的 KerasTensor DAG"这个事实上。另一个高频场景是**层复用/权重共享**(比如 Siamese 网络两个分支调用同一个 `encoder` 实例):Functional API 下,同一个层实例被调用两次会产生两个不同的 `Node`(`_inbound_nodes` 长度变成 2),`.summary()` 依然能正确显示两条独立的连接路径,而 Subclassing 下这种复用完全隐藏在 `call()` 内部,外部没有任何办法在不读源码的情况下知道一个层被用了几次。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf

# --- 1) KerasTensor 是符号占位符,不含真实数值 ---
inp = keras.Input(shape=(4,), name="inp")
assert not hasattr(inp, "numpy")            # 没有 .numpy(),因为压根没有真实数值
print("KerasTensor:", inp)

# --- 2) call() 在符号构图阶段真的执行了一次(用占位tensor,不是真实数据) ---
call_log = []
class Noisy(keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], 3))
    def call(self, inputs):
        call_log.append(type(inputs).__name__)
        return tf.matmul(inputs, self.w)

layer = Noisy()
out = layer(inp)                              # 符号调用
assert layer.built is True                    # build() 真的被调用了,真实变量已创建
assert call_log == ["SymbolicTensor"]           # call()确实执行了一次,输入是placeholder,不是KerasTensor本身
print("符号构图阶段 call() 的输入类型:", call_log)

# --- 3) _keras_history / _inbound_nodes:DAG是真实存在的数据结构,不是比喻 ---
assert out._keras_history.layer is layer
assert out._keras_history.node_index == 0
node = layer._inbound_nodes[0]
assert node.input_tensors is inp
assert node.outputs is out
print("PASS: KerasTensor的_keras_history和层的_inbound_nodes互相印证,构成真实的DAG记录")

# --- 4) 层被复用两次 -> _inbound_nodes 长度变成2,各自独立记录 ---
shared = keras.layers.Dense(5, name="shared")
a_in = keras.Input(shape=(4,), name="a")
b_in = keras.Input(shape=(4,), name="b")
a_out = shared(a_in)
b_out = shared(b_in)
assert len(shared._inbound_nodes) == 2
assert shared._inbound_nodes[0].input_tensors is a_in
assert shared._inbound_nodes[1].input_tensors is b_in
print("同一层被复用两次,_inbound_nodes 各自独立记录:", len(shared._inbound_nodes))

print("Section 3 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "Functional API 的 `.summary()` 为什么能打印出完整的层级连接拓扑,而 Subclassing API 不行?"—— 期望答出"Functional 下 `keras.Input()` 产生的是 KerasTensor,层作用在它上面时会走符号构图路径,记录下真实的层连接 DAG;Subclassing 的 `call()` 是任意 Python 代码,框架不知道内部的调用顺序和连接关系"。
- **追问 1(区分度很高):** "`Dense(8)(some_keras_tensor)` 这一行代码执行的时候,`Dense` 层的 `call()` 方法真的被调用了吗?权重真的被创建了吗?"—— 很多人会以为符号构图"只是记录了个形状,没有真正跑代码",期望答出"两者都是真的——`build()` 真的创建了 `tf.Variable`,`call()` 也真的执行了一次,只是输入换成了一个在临时图里的占位 tensor,不是你的真实数据"。
- **追问 2(深挖):** "如果一个自定义层的 `call()` 里有 Python 副作用(比如往一个列表 `append`、打印日志),用 Functional API 搭图的时候,这个副作用会触发几次?"—— 期望能推理出"符号构图阶段会触发一次(因为 `call()` 真的执行了一次),之后每次真实调用模型推理/训练又会各自触发——这一点和 `tf.function` trace 阶段副作用只触发一次是同一类问题,容易被忽视"。
- **追问 3:** "同一个 `Dense` 层实例,在 Functional API 里被两个不同的输入调用了两次,这算一个层还是两个层?权重共享吗?"—— 期望答出"是同一个层实例、权重完全共享,但会产生两个独立的 `Node`(`_inbound_nodes` 长度为2),`.summary()`/`plot_model()` 能正确显示两条独立的连接路径,而不是报错或者覆盖"。

**常见坑:** 在自定义层的 `call()` 里写了依赖 Python `if`/`for` 对**具体数值**做判断的逻辑(比如 `if tf.reduce_sum(inputs) > 0`),放进 Functional API 搭图时不会报错,但符号构图阶段传入的是占位 tensor,不是真实数据,这类条件判断在图构建期会用占位阶段"恰好算出来"的值走某一个固定分支,之后不管真实输入是什么,模型的计算图都固定死在那一条分支上——这是 Functional API 和真正的动态控制流(第 4 节 Subclassing 里的用法)之间容易被忽视的语义差异,凡是需要"依据运行时真实数值动态分支"的逻辑,必须用 Subclassing API,不能塞进 Functional 图里。

---

## 4. Model 子类化(Subclassing)API —— 最灵活,写法上最像 PyTorch `nn.Module`

**是什么:**
```python
# class MyModel(tf.keras.Model):
#     def __init__(self, ...):
#         super().__init__()
#         self.layer1 = ...
#     def call(self, inputs, training=None):
#         return ...
```

**一句话:** 继承 `keras.Model`,在 `__init__` 里创建子层、在 `call()` 里用任意 Python 代码把它们串起来——这一套写法和 PyTorch `nn.Module.__init__` + `forward()` 几乎逐行对应,换来的代价是第 3 节讲过的"Keras 不再拥有一份静态可读的连接拓扑"。

**底层机制/为什么这样设计:**

`keras.Model` 是 `keras.layers.Layer` 的子类,`Layer` 又是 `tf.Module` 的子类(第 1 节的继承链)——所以子层赋值给 `self.xxx` 之后能被自动发现,靠的还是第 1 节讲的 `tf.Module` 反射机制,这一点 Subclassing 和 Sequential/Functional 完全共享,没有任何特殊处理。真正区别三套 API 的,只有"`call()` 内部这段代码是被谁、在什么时候执行"——Sequential/Functional 会在符号构图阶段额外执行一次(第 3 节),Subclassing 的 `call()` **只在真实调用(真实数据或后面章节会讲的 `tf.function` 追踪)时才会被执行,从来不会有"提前跑一遍占位数据"这一步**。

`keras.Model.__init__` 内部有一个显式检查,强制要求这一步必须先发生:
```python
import tf_keras as keras
import tensorflow as tf

class Forgetful(keras.Model):
    def __init__(self):
        # 没有 super().__init__()!
        self.dense = keras.layers.Dense(3)
    def call(self, x):
        return self.dense(x)

try:
    m = Forgetful()
    m(tf.random.normal((2, 4)))
    assert False
except RuntimeError as e:
    assert "forgot to call `super().__init__()`" in str(e)
    print("忘记 super().__init__() 的真实报错:", str(e))
```
这个检查本质上和 PyTorch 的"忘记 `super().__init__()` 导致 `_modules` 字典不存在"是同一类问题(torch03 第 1 节)——`tf.Module.__init__` 需要先跑,`name`/`name_scope` 这些内部状态才存在;区别只是 TF/Keras 这边选择了显式抛出一个措辞友好的 `RuntimeError`,而不是让底层的 `AttributeError` 直接冒出来。

**AI 研究/工程场景:** 任何需要"依据运行时真实数值做条件分支"的模型都只能用 Subclassing——比如 MoE 路由(根据 gating 结果决定这个 batch 走哪几个专家子网络)、RL 里 agent 根据环境反馈决定要不要多算一步、树形结构网络的递归调用。这些场景和 torch-deep-dive 02 篇讲 PyTorch define-by-run 动态图时举的例子是同一类问题——Keras Subclassing API 在"能不能用原生 Python 控制流写模型"这一点上,和 PyTorch 的自由度是对等的,代价也对等(拿不到静态拓扑、`.summary()` 打印不出连接图)。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf

# --- 1) call()里可以用真正依赖运行时数值的Python控制流,Functional API做不到(呼应第3节常见坑) ---
class Adaptive(keras.Model):
    def __init__(self):
        super().__init__()
        self.small_path = keras.layers.Dense(4, name="small_path")
        self.big_path = keras.layers.Dense(4, name="big_path")

    def call(self, x):
        if tf.reduce_sum(x) > 0:              # 真实依赖输入数值,不是占位符
            return self.small_path(x) + 1.0
        else:
            return self.big_path(x) - 1.0

m = Adaptive()
pos_out = m(tf.ones((1, 3)))
neg_out = m(-tf.ones((1, 3)))
assert pos_out.shape == (1, 4) and neg_out.shape == (1, 4)
print("正输入走small_path, 负输入走big_path,两条分支都真实执行了")

# --- 2) plain Layer 没有 fit/compile,Model 才有;Model 是 Layer 的子类 ---
plain_layer = keras.layers.Layer()
assert not hasattr(plain_layer, "fit")
assert not hasattr(plain_layer, "compile")
model_instance = keras.Model()
assert hasattr(model_instance, "fit") and hasattr(model_instance, "compile")
assert issubclass(keras.Model, keras.layers.Layer)
print("PASS: fit()/compile() 是 Model 独有的,不是 Layer 就有的")

# --- 3) 子层依然靠 tf.Module 反射机制被自动发现,和 Sequential/Functional 共享同一套底层追踪 ---
class TwoLayerMLP(keras.Model):
    def __init__(self):
        super().__init__()
        self.h1 = keras.layers.Dense(8, name="h1")
        self.h2 = keras.layers.Dense(3, name="h2")
    def call(self, x):
        return self.h2(self.h1(x))

mlp = TwoLayerMLP()
_ = mlp(tf.random.normal((2, 4)))
assert isinstance(mlp, tf.Module)
assert len(mlp.submodules) == 2                 # h1, h2 都被 tf.Module 反射机制发现
print("submodules 数量:", len(mlp.submodules))

print("Section 4 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "Model 子类化 API 和 PyTorch 的 `nn.Module` 写法有什么相似和不同?"—— 期望答出"`__init__` 建子层、`call()`(对应 `forward()`)里用任意 Python 代码组织前向逻辑,这一层高度相似;不同点在于底层子层发现机制(TF 是 `tf.Module` 反射,PyTorch 是 `__setattr__` 登记)和 Subclassing 牺牲的能力(拿不到静态拓扑)"。
- **追问 1:** "为什么 Subclassing API 没法像 Functional API 那样自动画出完整的连接拓扑图?"—— 期望能连回第 3 节:"因为 `call()` 只在真实调用时执行,框架没有一个'提前跑一遍拿到 DAG'的机会,而且 `call()` 内部完全可能有数据依赖的分支,压根不存在一份'唯一正确'的静态连接图"。
- **追问 2(容易漏答):** "一个只继承 `keras.layers.Layer`(不是 `keras.Model`)的类,可以直接调用它的 `.fit()` 方法训练吗?"—— 期望答出"不行,`fit()`/`compile()`/`evaluate()` 是 `Model` 类才有的,`Layer` 只负责`build`/`call`/权重追踪这些更基础的能力;`Model` 是 `Layer` 的子类,在此基础上叠加了训练循环相关的一整套机制"。
- **追问 3(开放题):** "什么场景下你会选 Subclassing 而不是 Functional,即使 Functional 也能表达出你要的结构?"—— 期望说出"结构本身依赖运行时真实数值做条件分支的场景(MoE 路由、RL 动态展开、递归网络),这些 Functional API 原则上无法正确表达(第3节常见坑提到:图构建期占位数据算出来的分支会被写死),只能用 Subclassing"。

**常见坑:** 混淆"Subclassing 灵活"和"Subclassing 就应该完全抛弃 Functional 的规范性"——很多实际项目里最佳实践是"整体用 Functional 或 Sequential 搭出主干,只在真正需要动态控制流的局部子模块用 Subclassing 包一层",而不是整个模型无差别上 Subclassing,否则会丢失 `.summary()` 拓扑可视化、`plot_model()` 架构文档、以及后面章节要讲的部分序列化便利性,这些收益本可以在不需要动态特性的那部分模型上白拿。另一个坑是把 `call()` 写成有副作用、不是纯函数的形式(比如在 `call()` 里修改 Python 层面的外部可变状态)——这在 eager 模式下也许能正常工作,但一旦被后续章节要讲的 `tf.function` 包裹加速,副作用的触发次数会变得不直观(只在 trace 阶段触发,这一点和第 3 节 Functional 符号构图阶段的坑本质相同)。

---

## 5. `call()` 与 `__call__` 的关系、`training` 参数的自动传播陷阱

**是什么:**
```python
# model(x)                          # 真正被调用的是 Layer.__call__(继承自Model)
# model.call(x)                     # 你重写的这个方法,是__call__内部才会调用的"业务逻辑"
# def call(self, inputs, training=None): ...   # 声明training参数,Keras会自动尝试注入
```

**一句话:** 你写 `model(x)` 的时候,真正执行的入口永远是框架实现的 `Layer.__call__`(负责 `build()` 调度、`training`/`mask` 参数注入、混合精度 cast 等一整套前后处理),它才会在内部转手调用你写的 `call()`;`training` 参数如果你的 `call()` 签名里声明了,Keras 会在你没有显式传值时尝试自动推断注入,但**这个自动推断的默认值是"推理模式",不是"训练模式"**,这一点和 PyTorch `model.train()`默认之后维持训练模式的直觉正好相反。

**底层机制/为什么这样设计:**

Keras 在层构造时,就用 `tf_inspect.getfullargspec(self.call)` 检查你的 `call()` 方法签名里有没有声明 `training` 这个参数名,记录成一个内部标志:
```python
import tf_keras as keras
import inspect

src = inspect.getsource(keras.layers.Layer._init_call_fn_args)
assert "CallFunctionSpec" in src
assert "getfullargspec" in src
print(src)
print("PASS: Keras在层构造时就用getfullargspec检查call()签名里有没有training参数")
```
之后每次 `__call__` 执行,如果检测到 `call()` 需要 `training` 参数、调用者又没有显式传值,框架会尝试从"当前调用上下文"里取一个值注入进去——这个上下文由外层调用链传递(比如你在另一个层的 `call()` 内部调用了这个子层,子层会继承外层收到的 `training` 值),但**如果压根没有任何外层上下文**(也就是你直接在最外面写 `model(x)`),没有任何地方显式声明过"这是训练"这件事,框架只能取一个安全的默认值——`False`(推理行为)。

这个设计和 PyTorch 的 `model.train()`/`model.eval()` 有本质区别:PyTorch 的训练/推理模式是一个**持久化的、挂在对象上的状态**(`self.training`,调用一次 `.train()` 之后一直生效,直到你再调用 `.eval()`);Keras 的 `training` 是一个**随每次调用传递的瞬时参数**,`Layer` 对象本身默认不记住"我现在是训练模式还是推理模式"这件事。两种设计的风险方向完全相反:PyTorch 的坑是"忘记调用 `.eval()`,推理时其实还在用训练模式跑"(torch03 第 6 节);Keras 的坑是"忘记显式传 `training=True`,训练时其实在悄悄用推理模式跑"。

**AI 研究/工程场景:** 手写 `GradientTape` 自定义训练循环(不用 `model.fit()`)是这个陷阱出现频率最高的场景——`model.fit()` 内部会自动在训练步骤里传 `training=True`、验证步骤里传 `training=False`,这一层代劳很容易让人误以为"Keras 会自己搞定 training 状态",直到某天脱离 `fit()` 手写训练循环,`with tf.GradientTape() as tape: y = model(x)` 忘了加 `training=True`,`Dropout` 层实际上完全没有丢弃任何神经元、`BatchNormalization` 的滑动统计量完全没有被更新,训练看起来能跑、loss 也在下降,只是模型质量会比正常训练差一截,而且没有任何报错提示这件事发生了。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf

# --- 1) Dropout:直接调用不传training,和显式training=False完全一致(都是推理行为) ---
tf.random.set_seed(0)
x = tf.ones((4, 10))
drop = keras.layers.Dropout(0.9)   # 90%丢弃率,容易观察

out_default = drop(x)                    # 不传training
out_infer   = drop(x, training=False)
out_train   = drop(x, training=True)

zero_ratio = lambda t: float(tf.reduce_mean(tf.cast(t == 0, tf.float32)))
assert zero_ratio(out_default) == 0.0
assert zero_ratio(out_infer) == 0.0
assert zero_ratio(out_train) > 0.8         # 接近90%
assert bool(tf.reduce_all(out_default == out_infer))     # 数值完全一致,不只是"看起来像"
print("不传training的zero_ratio:", zero_ratio(out_default), " | training=True的zero_ratio:", zero_ratio(out_train))

# --- 2) BatchNormalization:忘记training=True,滑动统计量根本不会更新 ---
bn = keras.layers.BatchNormalization()
x_bn = tf.random.normal((8, 4)) * 10 + 5
bn.build(x_bn.shape)
before = bn.moving_mean.numpy().copy()
with tf.GradientTape():
    _ = bn(x_bn)                          # 忘记传 training=True !
after = bn.moving_mean.numpy().copy()
assert (before == after).all()              # moving_mean纹丝不动
print("忘记training=True: moving_mean 是否变化:", not (before == after).all())

bn2 = keras.layers.BatchNormalization()
bn2.build(x_bn.shape)
before2 = bn2.moving_mean.numpy().copy()
with tf.GradientTape():
    _ = bn2(x_bn, training=True)          # 正确传了
after2 = bn2.moving_mean.numpy().copy()
assert not (before2 == after2).all()         # moving_mean 真实更新了
print("正确传training=True: moving_mean 是否变化:", not (before2 == after2).all())

# --- 3) 嵌套调用:外层显式传了training,内层子层调用即使没手动转发,也会自动继承 ---
class Outer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.bn = keras.layers.BatchNormalization()
    def call(self, x, training=None):
        return self.bn(x)                  # 注意:没有显式把training往下传!

outer = Outer()
_ = outer(x_bn, training=True)             # 只在最外层传了
assert not (outer.bn.moving_mean.numpy() == 0).all()   # 内层bn的moving_mean依然真实更新了
print("外层training=True,内层bn即使没手动转发也继承到了训练模式")

print("Section 5 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`model(x)` 和 `model.call(x)` 有什么区别?为什么不直接调用 `call()`?"—— 期望答出"`__call__` 是真正的入口,内部做了 `build()` 调度、`training`/`mask` 参数处理等一系列前后处理,再转手调用你写的 `call()`;直接调 `call()` 会跳过这些机制,是不推荐的用法"。
- **追问 1(区分度很高):** "如果我在自定义训练循环里,直接写 `y = model(x)`(不传 `training`),`Dropout`/`BatchNorm` 会按训练模式还是推理模式跑?"—— 期望准确答出"推理模式(`training=False`),这是很多人会踩的坑,因为直觉上'正在训练'应该默认是训练行为"。
- **追问 2(能看出是否真的做过对比):** "这一点和 PyTorch 的 `model.train()`/`model.eval()` 相比,坑出现的方向是不是一样的?"—— 期望说出"方向相反:PyTorch 默认构造完就是 train 模式,坑是'忘了切到 eval 推理时还在训练模式';Keras 默认没有任何上下文时是推理行为,坑是'忘了显式声明训练时其实在推理模式'"。
- **追问 3:** "一个自定义层内部调用了另一个子层,子层的 `call()` 需不需要你手动把 `training` 参数转发进去?"—— 期望答出"通常不需要手动转发,Keras 有一套调用上下文机制,只要外层收到了显式的 `training` 值,内层子层调用会自动继承,除非外层调用本身就没有任何上下文(比如你直接从最外面调用整个模型却没传)"。

**常见坑:** 反过来也有过度纠正的坑——有人为了"保险"给每一次调用都强行加 `training=True`,结果在 `model.evaluate()`/`model.predict()` 内部真实调用时被这个强行传入的值覆盖预期行为,导致验证阶段 `Dropout` 依然在丢弃神经元、`BatchNorm` 依然在用当前 batch 的统计量而不是训练阶段积累的滑动统计量,验证集指标出现不该有的抖动。正确做法是不在自己的业务代码里手写死 `training=True/False`,而是让 `model.fit()`/`.evaluate()`/`.predict()` 这些高层 API 自动决定,只有在手写 `GradientTape` 训练循环、`predict` 逻辑时才需要你自己显式声明。

---

## 6. `build()` 延迟构建机制

**是什么:**
```python
# def build(self, input_shape):
#     self.w = self.add_weight("w", shape=(input_shape[-1], self.units))
# —— 第一次真实调用时才会被触发,而不是 __init__ 阶段
```

**一句话:** `Dense(units=64)` 在 `__init__` 阶段只知道"输出多少维",不知道"输入是多少维"——权重矩阵的完整 shape 要等第一次真实数据(或已知的符号输入)流过来才能确定,所以 Keras 把"创建权重"这一步从 `__init__` 里拆出来,单独放进 `build(input_shape)`,由框架在第一次调用时自动触发。

**底层机制/为什么这样设计:**

对比一下没有 `build()` 机制的纯 `tf.Module` 写法就能看出这个设计解决了什么问题——第 1 节引用过的 `tf.Module` 官方 Dense 示例,构造时必须显式传 `input_dim`:
```python
import tensorflow as tf
import tf_keras as keras

class RawDense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([input_dim, output_size]))
        self.b = tf.Variable(tf.zeros([output_size]))
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

raw = RawDense(input_dim=4, output_size=3)     # 必须自己算出并传入input_dim
assert raw.w.shape == (4, 3)

# Keras Dense: 只需要units,input_dim靠第一次调用自动推导
d = keras.layers.Dense(3)
assert d.built is False
try:
    _ = d.kernel
    assert False
except AttributeError as e:
    assert "no attribute 'kernel'" in str(e)
    print("build()之前访问kernel:", str(e))

_ = d(tf.random.normal((2, 4)))               # 第一次真实调用,触发build()
assert d.built is True
assert d.kernel.shape == (4, 3)                  # 4是从没提供过的输入shape自动推导出来的
print("build()之后kernel.shape:", d.kernel.shape)
```

触发这一步的调度逻辑在 `Layer._maybe_build` 里,现场读源码能看到两个关键设计点:第一,只有当子类**真的重写了** `build()`(基类默认实现被标记了 `_is_default`)才会调用它,避免对没有权重的层做无意义的空调用;第二,不管子类的 `build()` 有没有规规矩矩调用 `super().build(input_shape)`,框架都会在调用完之后**无条件**再调用一次 `Layer.build(self, input_shapes)`,把 `self.built` 设成 `True`。源码里的注释原话是:"We must also ensure that the layer is marked as built... since user defined build functions may not be calling `super.build()`"——这是一处专门为"用户可能会漏写"设的安全网,直接决定了一个常被口口相传的说法其实不成立:

**AI 研究/工程场景:** 这套机制最大的实际价值,是让你写模型的时候不需要提前手算每一层的输入维度——`Dense(128)` 后面接 `Dense(64)`,你完全不需要知道上一层输出到底是多少维,Keras 会在第一次真实数据流过时自动把这些形状拼接对上。在处理变长序列/动态 batch 场景中,`build()` 还有一个不那么显眼但很关键的作用:它只依赖 `input_shape` 里"形状"这个信息(通常允许 batch 维是 `None`),不依赖具体数值,所以同一个层可以在 `Dense(128).build((None, 4))` 之后接受任意 batch size 的真实输入,不需要为每个 batch size 重新建一次权重。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf

# --- 验证一个常被以讹传讹的说法:"忘记调用 super().build() 会导致每次调用都重新build" ---
call_count = {"n": 0}

class ForgetsSuperBuild(keras.layers.Layer):
    def build(self, input_shape):
        call_count["n"] += 1
        self.w = self.add_weight("w", shape=(input_shape[-1], 2))
        # 故意不调用 super().build(input_shape) !

    def call(self, x):
        return tf.matmul(x, self.w)

layer = ForgetsSuperBuild()
assert layer.built is False

x = tf.random.normal((2, 4))
_ = layer(x)
assert layer.built is True and call_count["n"] == 1

_ = layer(x)
_ = layer(x)
assert layer.built is True and call_count["n"] == 1     # 没有重复build!框架有安全网兜底built标志位
print("忘记super().build()之后,build()总共被调用次数:", call_count["n"], "(不会无限重建)")

# --- build() 只在"第一次"触发,即便第二次调用的输入shape变了,也不会自动重新build ---
d = keras.layers.Dense(3)
_ = d(tf.random.normal((2, 4)))          # 第一次:输入最后一维是4
kernel_shape_1 = d.kernel.shape
try:
    _ = d(tf.random.normal((2, 6)))        # 第二次:输入最后一维变成6,和已建好的kernel不匹配
    assert False
except Exception as e:
    print("shape变化但没有重新build,真实调用报错:", type(e).__name__, str(e)[:150])
assert d.kernel.shape == kernel_shape_1   # kernel形状没有被自动改变

print("Section 6 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "为什么 Keras 的层要把权重创建拆成 `__init__` 和 `build()` 两步,而不是直接在 `__init__` 里建好?"—— 期望答出"`__init__` 阶段通常不知道输入维度(比如 `Dense(units)` 只指定输出维度),权重的完整 shape 需要看到输入之后才能确定,所以拖到第一次调用时,由框架自动调用 `build(input_shape)`"。
- **追问 1(区分度很高):** "如果我自己重写 `build()`,忘记调用 `super().build(input_shape)`,会有什么后果?是不是每次调用都会重新触发 `build()`,反复创建变量?"—— 期望答出(如果候选人只会背模板答案,这里容易翻车)"不会反复重建,框架的 `_maybe_build` 内部无论如何都会在你的 `build()` 执行完之后再调用一次基类 `build()` 把 `built` 标志位设为 `True`,这是一处专门兜底的安全网";能进一步说出"不写 `super().build()` 依然不是好习惯,因为基类 `build()` 还会记录 `_build_input_shape` 等状态,只是不会导致的是'反复重建'这个具体后果",说明理解得足够细。
- **追问 2:** "`build(input_shape)` 拿到的 `input_shape` 参数,是具体的数值 shape,还是可能包含 `None`?"—— 期望答出"可能包含 `None`(通常是 batch 维),`build()` 依赖的是静态形状信息,不需要具体数值,这也是为什么同一层能兼容不同 batch size 调用而不用重新建权重"。
- **追问 3:** "已经 built 过的层,如果之后传入一个和已建权重不兼容的新 shape(比如输入特征维度变了),会自动重新 build 吗?"—— 期望答出"不会,`built` 一旦是 `True` 就不会再触发 `build()`,shape 不兼容会在真实的矩阵运算里报错,而不是被自动'修正'"。

**常见坑:** 把应该依赖输入 shape 才能确定形状的权重创建写在了 `__init__` 里,通过强行要求用户在构造时手动传入输入维度来"绕开" `build()`——这在能跑的意义上没问题,但破坏了 Keras 层"不需要提前知道输入维度"这个核心易用性优势,而且如果这样的层被用在 Functional/Sequential 里,会丧失自动 shape 推导的便利。另一个真实的坑是在 `build()` 里创建权重时使用了错误的 `input_shape` 索引(比如以为 `input_shape[0]` 是特征维,实际上 `input_shape[0]` 通常是 batch 维、`input_shape[-1]` 才是最后一个特征维)——这个错误往往不会立刻报错,而是建出一个形状错误的权重矩阵,在后续的 `call()` 矩阵运算里才报出一个和"权重形状错误"表面上不直接相关的维度不匹配错误,排查时容易舍近求远。

---

## 7. `get_config()` / `from_config()` 序列化契约,以及 Keras 2/3 现状面试怎么答

**是什么:**
```python
# def get_config(self):
#     config = super().get_config()
#     config.update({"units": self.units})
#     return config
# —— 返回一个JSON可序列化的dict,记录"重建这个层/模型需要哪些构造参数"
#
# @classmethod
# def from_config(cls, config):
#     return cls(**config)          # 基类默认实现就是这样,通常不需要重写
```

**一句话:** `get_config()`/`from_config()` 是 Keras 对象"如何描述自己、如何被重新构造出来"的契约——`get_config()` 把构造这个层需要的参数打包成一个 dict(不包含权重数值本身),`from_config()` 拿着这个 dict 重新 `__init__` 一个新实例;这套契约是 `.keras`/SavedModel 格式、`clone_model()`、跨进程分发模型结构等一切"需要重建模型结构"场景的基础设施。

**底层机制/为什么这样设计:**

一个常被过时教程强调的说法是"自定义层不重写 `get_config()`,构造参数就会丢失,序列化必坏"——这在当前版本已经不完全准确,基类 `get_config()` 其实有一套自动兜底机制:
```python
import tf_keras as keras
import inspect

src = inspect.getsource(keras.layers.Layer.get_config)
assert "_auto_get_config" in src
assert "getfullargspec(self.__init__)" in src
print("PASS: 基类get_config()内部确实有自动尝试推导__init__参数的兜底逻辑")
```
基类实现会先检查子类有没有真的重写 `get_config()`;如果没有,不会直接放弃,而是尝试**自动内省 `__init__` 的参数列表**,把能在实例上找到同名属性、且值是简单可序列化类型(`int`/`str`/`float`/`list`/`dict` 等)的参数自动收进 config——这一步对"构造参数就是简单值、且赋值成了同名属性"的常见写法通常是有效的。但这个自动兜底是有边界的:一旦 `__init__` 接收了一个**不可序列化**的参数(比如一个裸的 Python 函数、自定义对象),自动推导会主动放弃并抛出一个**清楚指出原因、给出示范代码**的 `NotImplementedError`,而不是悄悄丢失这个参数——"要么自动补全,要么明确报错让你自己写",不存在"静默丢参数导致 load 出来的模型行为不对但没有任何提示"这种最坏情况。

理解了这套自动兜底机制的边界,再看 **Keras 2/3 现状**才有意义:自 TF 2.16 起,`pip install tensorflow` 默认让 `tf.keras` 指向 **Keras 3**——一次多后端(TensorFlow/JAX/PyTorch 都能作为 Keras 3 的执行后端)重写,内部模块结构、部分序列化细节和历史上"内嵌在 TF 里的 Keras 2"并不完全相同(比如下面例子会看到,模块路径从 `keras.src.engine.*` 变成了 `keras.src.models.*`/`keras.src.layers.*`)。但面试题库和大多数现存教程默认的心智模型仍然是经典 Keras 2——这正是本系列在 [00-roadmap.md](00-roadmap.md) 里做出的主动选择:显式安装 `tf_keras` 包、设置 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典实现。

**AI 研究/工程场景:** 判断"自己环境里 `tf.keras` 到底是哪个版本",最直接的方法不是看 `tensorflow.__version__`(这个不管走哪条路径都不会变),而是看**具体某个 Keras 对象实例的类所在的模块路径**——`type(model).__module__`,如果落在 `tf_keras.*` 说明是经典 Keras 2 实现,落在 `keras.src.*` 说明是 Keras 3。跨团队协作或者复现别人的开源代码时,如果对方仓库依赖的是 Keras 2 时代的某些内部行为(比如某些 `get_config()` 细节、`tf.keras.backend` 私有 API),而你本地环境是"什么都没设置的默认 TF 2.16+"(即 Keras 3),复现时出现的诡异报错往往第一时间应该怀疑这层版本不一致,而不是代码本身有 bug。

**可运行例子:**
```python
import tf_keras as keras
import tensorflow as tf
import json

# --- 1) 显式实现get_config():标准写法,JSON可序列化,能正确round-trip ---
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], self.units))
    def call(self, x):
        return tf.matmul(x, self.w)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

layer = Linear(units=16, name="my_linear")
config = layer.get_config()
assert config["units"] == 16
assert json.dumps(config)                        # 确认真的是JSON可序列化的
layer2 = Linear.from_config(config)
assert layer2.units == 16 and layer2.name == "my_linear"
print("显式get_config()正常round-trip:", config)

# --- 2) 不写get_config():基类自动兜底,简单类型参数依然能被正确推导 ---
class NoConfig(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], self.units))
    def call(self, x):
        return tf.matmul(x, self.w)
    # 没有重写get_config()!

nc = NoConfig(units=99, name="nc")
auto_config = nc.get_config()
assert auto_config["units"] == 99                # 自动推导成功,不是"必然丢失"
nc2 = NoConfig.from_config(auto_config)
assert nc2.units == 99
print("没写get_config(),基类自动兜底推导出的config:", auto_config)

# --- 3) 参数不可序列化时(裸函数),自动兜底会主动放弃并明确报错,不是静默丢参数 ---
class BadLayer(keras.layers.Layer):
    def __init__(self, activation_fn, **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = activation_fn        # 一个裸python函数,不是简单可序列化类型
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], 4))
    def call(self, x):
        return self.activation_fn(tf.matmul(x, self.w))

bad = BadLayer(activation_fn=tf.nn.relu, name="bad")
try:
    bad.get_config()
    assert False
except NotImplementedError as e:
    assert "must override `get_config()`" in str(e)
    print("不可序列化参数导致的明确报错(不是静默丢失):", str(e).splitlines()[1])

# --- 4) Keras 2/3 现状:怎么判断当前 tf.keras 到底是哪个版本 ---
model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
assert type(model).__module__.startswith("tf_keras.")     # 本系列锁定的legacy Keras 2
print("当前 tf.keras 版本判定 —— type(model).__module__:", type(model).__module__)

import keras as bare_keras                                   # 绕过tf.keras重定向,直接拿Keras 3包
assert bare_keras is not tf.keras
m3 = bare_keras.Sequential([bare_keras.layers.Dense(2)])
assert type(m3).__module__.startswith("keras.src.")         # Keras 3 的模块路径
print("裸 import keras 拿到的是 Keras 3 —— type(m3).__module__:", type(m3).__module__)
print("bare_keras is tf.keras:", bare_keras is tf.keras)

print("Section 7 OK")
```

**面试怎么问 + 追问链:**
- **Q:** "`get_config()`/`from_config()` 是干什么用的?为什么自定义层通常建议重写它?"—— 期望答出"描述'重建这个对象需要哪些构造参数'的序列化契约,是模型保存/加载、`clone_model()` 等场景重建模型结构的基础;自定义层的 `__init__` 参数框架不一定能完全自动推导,显式实现更可靠"。
- **追问 1(能分辨是否只会背模板):** "如果自定义层完全不写 `get_config()`,是不是构造参数就必然丢失、模型没法正确保存加载了?"—— 期望答出"不一定,基类有自动兜底机制,会尝试内省 `__init__` 参数并从实例属性里读值,对简单可序列化类型的参数通常能自动成功;只有遇到不可序列化的参数时才会明确报错要求你手写"——能答出这一点说明是真的读过当前版本行为,不是背了旧版本的说法。
- **追问 2(呼应00环境声明,面试高频):** "你知道 Keras 3 吗?为什么你的项目/环境里用的还是 Keras 2?"—— **不能答"不知道"**,期望答出"知道,TF 2.16 起 `pip install tensorflow` 默认 `tf.keras` 就是 Keras 3(多后端架构);本环境是有意识地装了 `tf_keras` 包并设置 `TF_USE_LEGACY_KERAS=1`,主动锁定回 Keras 2 行为,原因是当前面试题库和大部分教程材料的心智模型还停留在 Keras 2,选择兼容性更成熟、社区素材更多的版本是工程判断,不是不了解新版本"。
- **追问 3:** "怎么用代码判断当前 `tf.keras` 实际解析到的是 Keras 2 还是 Keras 3?"—— 期望答出"`type(某个model或layer实例).__module__`,落在 `tf_keras.*` 是 Keras 2,落在 `keras.src.*` 是 Keras 3;`tensorflow.__version__` 判断不了这件事"。

**常见坑:** 把 `get_config()` 和"保存权重数值"搞混——`get_config()` 只负责结构/超参数这一层("这是一个 `units=16` 的 `Linear` 层"),完全不包含 `w`/`b` 这些真实权重的数值,权重的保存加载是另一套独立机制(留给 12 类"序列化与部署基础"展开)。另一个坑是自定义层的 `get_config()` 里忘记调用 `super().get_config()` 而是自己从空 dict 开始拼——这样会丢失 `name`/`trainable`/`dtype` 这些基类本该自动带上的通用字段,`from_config()` 重建出来的实例可能连名字都对不上,正确写法永远是"先拿基类的 config,再 `update()` 自己独有的参数"。

---

## 小结:这一批 7 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `tf.Module` | Keras `Layer`/`Model` 的真正基类;靠**读时反射**(`_flatten`)发现变量,不是 PyTorch 式的**写时登记**;普通 list/dict 里的变量也能被自动发现 |
| 2 | Sequential API | `Sequential` 是 `Functional` 的子类,只能表达线性拓扑;内部渐进式组装 KerasTensor 图,这也是它能自动 `.summary()` 的原因 |
| 3 | Functional API / KerasTensor | `keras.Input()` 产生的 KerasTensor 是符号占位符;层作用其上会触发真实的 `build()`+一次占位 `call()`,记录 `_keras_history`/`_inbound_nodes` 构成真实 DAG,这是 `.summary()` 拓扑的来源 |
| 4 | Model 子类化 API | 写法上和 PyTorch `nn.Module`+`forward()` 高度对应;子层发现依然靠 `tf.Module` 反射(和其他两套API共享);代价是没有静态 DAG |
| 5 | `call()`/`__call__`/`training` | `__call__` 是真正入口,`training` 是随调用传递的瞬时参数,不是持久状态;直接调用不传值默认等价于 `training=False`,方向和 PyTorch 的 `train()`默认坑相反 |
| 6 | `build()` 延迟构建 | 权重 shape 依赖输入,拖到第一次调用才建;`_maybe_build` 有安全网,忘记 `super().build()` 不会导致重复建权重 |
| 7 | `get_config()`/`from_config()` | 序列化契约,描述"怎么重建",不含权重数值;基类有自动兜底但对不可序列化参数会明确报错;Keras 2/3 现状用 `type(obj).__module__` 判断 |

下一批:[05-layers-math-and-backward.md](05-layers-math-and-backward.md)

---
