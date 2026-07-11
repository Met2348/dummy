# 01 · Tensor 基础与 tf.Variable 深挖(Tensor and Variable)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这是全系列的地基,和 [torch-deep-dive/01-tensor-memory-model.md](../torch-deep-dive/01-tensor-memory-model.md) 是"同一个位置"的一批,但两边地基要打的坑完全不同——torch 的地基问题是"一个 tensor 在内存里到底长什么样"(storage/stride),TF 的地基问题是**"值"和"状态"这两种东西在 TF2 里是靠两个完全不同的类型分开管的**(`tf.constant` 的不可变值 vs `tf.Variable` 的可变状态),外加 eager/graph 两套执行模式并存、以及为了应付"数据形状本身不规整"这类场景专门长出来的 `RaggedTensor`/`SparseTensor` 两个特化类型。基础的数组创建、reshape、索引这些和 numpy 高度重合的部分不重复展开,建议先看 [numpy-deep-dive/01-creation-and-init.md](../numpy-deep-dive/01-creation-and-init.md),这里只讲 TF 独有、且是面试高频的部分。

本文所有代码例子已在 00 篇声明的环境(WSL2、TF 2.21.0、`~/tf-venv`、GPU 可见、`TF_USE_LEGACY_KERAS=1`)下用 `source ~/tf-venv/bin/activate && python ...` 实际跑通验证。所有报错信息都是现场触发后原样抄录(`assert 子串 in str(e)` 核验,不是转述文档);"AI 研究/工程场景"段落按 00 篇声明,是根据真实训练/部署场景重构的例子,不是仓库代码引用。

**本篇统一结构(与 00 篇模板一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 本系列的核心加深点,不停在"怎么用"
4. AI 研究/工程场景
5. 可运行例子(带 `assert`,能内省的地方现场打印内部状态)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.constant` vs `tf.Variable` —— 不可变值与可变状态的分野,及 ResourceVariable 的底层实现

**是什么:**
```
tf.constant(value, dtype=None, shape=None, name='Const')
# 创建一个不可变的 EagerTensor:数据一旦写入,没有任何方法能修改它自身

tf.Variable(initial_value, trainable=True, dtype=None, name=None)
# 创建一个可变的 ResourceVariable:可以被 assign()/assign_add() 反复原地更新
```

**一句话:** `tf.constant` 产生的是一份"写死"的数值,之后只能基于它算出新 tensor,不能修改它自身;`tf.Variable` 是一个可以被反复原地更新的"有状态容器",背后由 `ResourceVariable` 实现,是模型权重、优化器动量、BatchNorm 滑动统计量等一切"需要跨 step 持久化并被修改"的数据的唯一合法载体。

**底层机制/为什么这样设计:**

`tf.constant` 返回的对象类型是 `EagerTensor`,没有 `assign` 之类的方法——不可变性不是"约定俗成"的软规则,而是这个类型压根没给你留修改的接口。`tf.Variable` 返回的类型是 `ResourceVariable`,它内部持有一个 `handle`:一个 `dtype=resource` 的特殊 tensor,指向 TF 运行时管理的一块**可变**内存,不是普通的数值 tensor。这个 `handle` 本身才是"Variable 是什么"的答案——Variable 不直接"是"数据,而是一个指向可变数据的资源句柄,`.assign()`/`.assign_add()` 这类方法本质上是"通过这个 handle 向运行时发一条'改写这块内存'的指令"。

这个设计不是从一开始就有的。TF1 时代的 `Variable`(后来被称为 `RefVariable`)语义更原始:变量的"读"操作没有在图里留下显式的节点,并发读写(尤其是分布式/多设备场景)下"这次读到的到底是修改前还是修改后的值"没有硬保证,必须靠手动 `tf.control_dependencies` 强行插入执行顺序约束。TF2 把 `ResourceVariable` 变成了唯一实现——**每一次读取变量,都会在图里插入一个显式的 `ReadVariableOp` 节点**,这个节点和其它 op 一样参与正常的数据依赖排序,"这次读到的是哪个版本的值"不再是隐式约定,而是图结构本身能回答的问题。实测:即便显式调用已被标记废弃的 `tf.compat.v1.disable_resource_variables()`,在 eager 模式下创建出来的仍然是 `ResourceVariable`,调用时会打印告警文案 `non-resource variables are not supported in the long term`——原始的 `RefVariable` 路径在 TF2 eager 下已经名存实亡。

`.assign()` 修改的是同一个 Python 对象的内部状态(`id(v)` 前后不变),这一点和"重新绑定"完全不同——一旦不小心写成 `v = new_tensor` 这种直接赋值,`v` 就从"持久化的可变容器"退化成了普通 tensor,彻底丢失 Variable 身份。

**AI 研究/工程场景:** 模型的可训练权重(Dense 层的 kernel/bias)、BatchNorm 的 running mean/var、优化器的动量 slot,全部必须是 `tf.Variable`;而输入数据、超参数常量、前向过程中的中间激活值都只是普通 tensor。自定义训练循环里如果不小心把权重写成了 `w = tf.zeros(...)` 而不是 `w = tf.Variable(tf.zeros(...))`,`GradientTape.gradient()` 会对这个"权重"返回 `None`——因为 tape 默认只自动 watch可训练的 Variable,这是 02 篇的内容,这里先埋一个伏笔。

**可运行例子:**
```python
import tensorflow as tf

# --- 不可变 vs 可变:类型和API能力的差异 ---
c = tf.constant([1, 2, 3])
v = tf.Variable([1, 2, 3])

assert type(c).__name__ == "EagerTensor"
assert type(v).__name__ == "ResourceVariable"
assert not hasattr(c, "assign")          # constant 没有 assign 方法
assert hasattr(v, "assign")

try:
    c.assign([4, 5, 6])
    assert False
except AttributeError as e:
    assert "no attribute 'assign'" in str(e)

# --- Variable 的"可变"是真正的原地修改,不是重新绑定 ---
vid_before = id(v)
v.assign([9, 9, 9])
assert id(v) == vid_before                # 同一个python对象,只是内部状态变了
assert v.numpy().tolist() == [9, 9, 9]

v.assign_add([1, 1, 1])
assert v.numpy().tolist() == [10, 10, 10]

# --- 底层是一个 resource handle,不是普通数值tensor ---
assert v.handle.dtype == tf.resource      # handle本身是"resource"这种特殊dtype
assert not hasattr(c, "handle")           # 普通tensor没有这个概念

# --- 关键证据:变量的每一次"读"在图里是一个显式的 ReadVariableOp 节点 ---
v2 = tf.Variable(1.0)

@tf.function
def read_twice(var):
    a = var + 0.0
    b = var + 0.0
    return a, b

concrete = read_twice.get_concrete_function(v2)
op_types = [op.type for op in concrete.graph.get_operations()]
assert op_types.count("ReadVariableOp") == 2     # 读了两次,图里就有两个显式节点

# --- assign 的形状/dtype 校验(不是"能塞就塞") ---
v3 = tf.Variable([1.0, 2.0, 3.0])
try:
    v3.assign([1.0, 2.0])       # 形状不匹配
    assert False
except ValueError as e:
    assert "Shape mismatch" in str(e)

try:
    v3.assign(tf.constant([1, 2, 3], dtype=tf.int32))   # dtype不匹配(显式int32 tensor)
    assert False
except ValueError as e:
    assert "requested dtype float32" in str(e)

print("point 1 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.constant` 和 `tf.Variable` 分别用在什么场景?区别是什么?"—— 期望答出"不可变值 vs 可变状态",并能说出"该用哪个"的判断标准是"这份数据要不要跨 step 持久化并被修改"。
- **追问 1:** "`tf.Variable` 底层是怎么实现'可变'这件事的?一个理论上应该值不可变的 tensor 世界里,它是怎么做到原地修改的?"—— 期望答出 `ResourceVariable` + resource handle,背后是运行时管理的一块可变 buffer,不是普通 tensor 的"值"。
- **深挖追问(区分度很高):** "为什么叫 `ResourceVariable`?TF1 时代不是这样吗?这个名字想解决什么问题?"—— 期望提到 TF1 `RefVariable` 在并发读写下语义不清晰(没有显式依赖边),TF2 用 resource handle + 显式 `ReadVariableOp` 把每次读也变成图里一个有明确位置的节点,才能保证正确的执行顺序。
- **追问(工程向):** "如果我在自定义训练循环里不小心用 `tf.zeros` 而不是 `tf.Variable` 存权重,会发生什么?"—— 期望答"`GradientTape` 默认不 watch 非 Variable 的 tensor,梯度会是 `None`"(02 篇展开)。

**常见坑:** 把 `v = v.assign(...)` 和直接 `v.assign(...)` 混为一谈——`.assign()` 本身就是原地操作,重新赋值虽不算错但多此一举;真正危险的是**误用 `v = new_tensor` 直接重新绑定**,这会让 `v` 变成一个普通 tensor,彻底脱离 Variable 身份(不再可训练、不再出现在 `model.trainable_variables` 里),而且 Python 语法上完全不会报错,只会在后面某个不相关的地方(比如权重死活不更新)才暴露问题。

---

## 2. eager 执行模型与 `.numpy()` 互操作

**是什么:**
```
tf.executing_eagerly()   # 返回当前是否处于eager模式(顶层代码默认True)
tensor.numpy()            # 把一个EagerTensor转换成numpy.ndarray,触发实际拷贝(见第8点)
```

**一句话:** TF2 默认所有代码都是"写一行马上执行一行"的 eager 模式,不需要 `Session`;但只有真正的 `EagerTensor` 才有 `.numpy()` 方法——一旦这段代码被 `tf.function` 追踪成图,内部的 tensor 类型会变成不支持 `.numpy()` 的符号 tensor,这是判断"我现在是不是真的在 eager 模式下"最快的手感测试。

**底层机制/为什么这样设计:**

TF1 是 define-and-run:所有代码先构建图,`.numpy()` 这种"立即拿到具体数值"的操作根本没有意义(图还没跑,没有值)。TF2 eager 模式下,每行代码在 Python 解释器执行的瞬间,同时在 C++ 层实际跑掉对应的 kernel,`EagerTensor` 背后已经有了真实计算好的数值缓冲区,`.numpy()` 只是把这块缓冲区(可能在 GPU 上)读出来转换成一个 numpy 数组的一层瘦封装。

关键的分野在于 `tf.function` 追踪期间:`tf.executing_eagerly()` 在顶层代码返回 `True`,但在被 `@tf.function` 装饰的函数体内部执行 tracing 时返回 `False`——因为 tracing 阶段 Python 代码确实在"跑"(用来发现控制流、决定图结构),但产生的中间结果不是"真实数值",而是描述"这一步该做什么运算"的符号占位。实测在 TF 2.21 里,这类 tensor 的类型是 `SymbolicTensor`(区别于 `EagerTensor`),`SymbolicTensor` 没有 `.numpy()` 方法——这是实测验证的结果,直接印证"eager"和"graph tracing"背后是两套不同的 tensor 实现,不是同一个类型的两种状态标记。

GPU 场景下,`.numpy()` 会隐式触发一次设备到主机(device-to-host)的拷贝,这个拷贝是**同步**的、会等待 GPU 计算完成——这是"训练循环里过度频繁调用 `.numpy()` 会拖慢速度"这条常见性能坑的根源(10 篇会展开)。

**AI 研究/工程场景:** 调试一个新模型结构时,可以在 eager 模式下随时 `print(tensor.numpy())` 查看中间激活值分布(比如检查是否出现 NaN),这是 TF2 相对 TF1 最大的开发体验提升;但训练循环如果被 `@tf.function` 包裹以提速,循环内部就不能再直接 `.numpy()` 或者用它做 `print` 调试——只能用 `tf.print`(03 篇的内容,这里先提一句)。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- 顶层默认就是eager,随时能拿到真实数值 ---
assert tf.executing_eagerly() is True

x = tf.constant([1.0, 2.0, 3.0])
y = x * 2
assert isinstance(y.numpy(), np.ndarray)
assert y.numpy().tolist() == [2.0, 4.0, 6.0]

# --- 被 tf.function 追踪时,eager状态和tensor类型都变了 ---
@tf.function
def f(t):
    assert tf.executing_eagerly() is False           # tracing阶段不是eager
    assert type(t).__name__ == "SymbolicTensor"        # 不是EagerTensor
    try:
        t.numpy()
        return False
    except AttributeError as e:
        return "no attribute 'numpy'" in str(e)

result = f(x)
assert result.numpy() == True     # f返回的是一个bool,被自动包装成tensor

# --- GPU tensor 的 .numpy() 会透明地做一次device-to-host拷贝 ---
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        g = tf.constant([1.0, 2.0])
    assert '/device:GPU:0' in g.device
    arr = g.numpy()                     # 拷回host,拿到普通numpy数组
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1.0, 2.0]

print("point 2 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "TF2 默认是 eager 模式,这意味着什么?和 TF1 比,对你写代码有什么直接影响?"
- **追问 1:** "`tf.function` 装饰的函数内部,`tf.executing_eagerly()` 会返回什么?为什么?"—— 期望答 `False`,并说出 tracing 阶段的本质是"跑一遍 Python 来发现图结构",不是真的在算数值。
- **深挖追问:** "你能不能证明,`tf.function` 内部的 tensor 和外面的 `EagerTensor` 不是同一种东西?"—— 期望能想到打印 `type()` 对比,而不是空口背概念。
- **追问(性能向):** "训练循环里频繁调用 `.numpy()` 会有什么代价?"—— 期望提到 GPU 场景下的同步 device-to-host 拷贝会阻塞流水线。

**常见坑:** 在 `@tf.function` 函数体内部直接对参数调用 `.numpy()` 或者用 `print()` 打印张量值,会在**第一次**调用(tracing 阶段)就抛出 `AttributeError`(`SymbolicTensor` 没有 `.numpy()`)——很多人以为这类报错是"有时候能跑有时候不能跑"的随机现象,其实是没意识到 tracing 只在第一次调用时发生,之后复用的都是编译好的图,这不是运行时偶发问题,而是结构性的、必然发生的。

---

## 3. `tf.TensorShape` —— 静态 shape vs 动态 shape(为 03 篇 retracing 埋伏笔)

**是什么:**
```
tensor.shape        # 返回一个 tf.TensorShape 对象:追踪期能确定到什么程度的形状,可能含 None
tf.shape(tensor)     # 返回一个 int32 Tensor:运行时才真正算出来的、永远完全确定的形状
```

**一句话:** `.shape` 是"我现在能不能提前知道"的静态信息(有些维度确实不知道,用 `None` 占位);`tf.shape()` 是不管你事先知不知道、运行时刨根问底算出来的动态信息,类型是一个真正的 Tensor,而不是 Python 元组。

**底层机制/为什么这样设计:**

在 eager 模式下两者几乎没有区别(数值都是已知的,静态 shape 天然完全确定),差异只在**图追踪**场景下才会暴露。用 `input_signature=[tf.TensorSpec(shape=[None, 3], ...)]` 固定住一个 `tf.function` 的追踪签名后,函数体内部 `t.shape` 打印出来是 `(None, 3)`——批次维度在追踪期确实不知道,因为同一份编译好的图要被不同 batch size 的输入复用;而 `tf.shape(t)` 打印出来是一个图节点(`Tensor("Shape:0", ...)`),不是一个具体数字,它的值要等到某次具体调用喂进具体数据后才能算出来,但同一张图用不同 batch size 调用两次,能正确算出两个不同的动态形状。

有些算子的输出长度**依赖运行时数值**而不是输入形状——比如 `tf.boolean_mask`,输出长度取决于 mask 里有多少个 `True`,这个数字在编译期原理上不可能知道。这类算子在 `tf.function` 追踪期间,输出的静态 shape 里必然会出现 `None`,即便 eager 模式下某一次具体调用能给出一个看似确定的形状。

一句话埋伏笔:静态 shape 里出现 `None`(尤其是同一个 `tf.function` 被不同形状调用时)正是 03 篇 `tf.function` retracing 的触发条件之一——这里不展开,03 篇专门讲。

**AI 研究/工程场景:** 写模型 `call()` 方法时,经常需要在 shape 未知的维度上做 reshape(比如 Transformer 里 `seq_len` 可变),这时候必须用 `tf.shape(x)` 拿运行时的真实数字去构造新形状,不能依赖 `x.shape`(可能是 `None`,直接拿去做算术会报错)。

**可运行例子:**
```python
import tensorflow as tf

# --- eager下,静态shape和动态shape的值总是一致、都完全已知 ---
x = tf.constant([[1, 2, 3], [4, 5, 6]])
assert x.shape.as_list() == [2, 3]                 # 静态:TensorShape对象
assert tf.shape(x).numpy().tolist() == [2, 3]       # 动态:真正的Tensor

# --- 图追踪场景下,两者才真正分道扬镳 ---
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32)])
def f(t):
    static_shape = t.shape                # 追踪期已知的部分:(None, 3)
    dynamic_shape = tf.shape(t)             # 永远是一个图节点,不是具体数字
    assert static_shape.as_list() == [None, 3]
    assert isinstance(dynamic_shape, tf.Tensor)
    return dynamic_shape

r1 = f(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
r2 = f(tf.constant([[1.0, 2.0, 3.0]]))
assert r1.numpy().tolist() == [2, 3]      # 同一张图,两次不同batch size都能正确算出
assert r2.numpy().tolist() == [1, 3]

# --- 依赖运行时数值的op(如boolean_mask),追踪期输出形状必然是None ---
@tf.function
def g(data, mask):
    out = tf.boolean_mask(data, mask)
    assert out.shape.as_list() == [None]     # 编译期不可能知道mask里有几个True
    return out

r3 = g(tf.constant([10, 20, 30]), tf.constant([True, False, True]))
assert r3.numpy().tolist() == [10, 30]

# --- 常见坑:对静态shape里的None做Python算术,eager下不会暴露,tf.function追踪期会报错 ---
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32)])
def h(t):
    return t.shape[0] // 2

try:
    h(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert False
except TypeError as e:
    assert "unsupported operand type(s) for //: 'NoneType' and 'int'" in str(e)

print("point 3 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "`tensor.shape` 和 `tf.shape(tensor)` 有什么区别,什么时候必须用后者?"
- **追问:** "为什么 `tf.function` 追踪时,某些 tensor 的 shape 会变成 `None`,即使这次调用传进去的输入形状是完全确定的?"—— 期望说出"图要兼容多次不同调用"、"某些 op 的输出长度依赖运行时数值,编译期原理上不可能知道"。
- **深挖追问(可留给 03 篇):** "这个 `None` 维度会不会导致每次调用都重新追踪图?"—— 简单提及"是 retracing 的触发条件之一,03 篇详细展开",不需要在这里给出完整答案。

**常见坑:** 直接对 `x.shape[0]`(可能是 `None`)做 Python 算术(比如 `x.shape[0] // 2`),在 eager 模式下不会报错(具体形状总是已知的),一旦这段代码被 `tf.function` 用不定 batch size 追踪,`None // 2` 会直接抛 `TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'`——这类坑的典型特征是"在 notebook 里单独跑这行完全正常,包进 `tf.function` 训练脚本却报错",根源是没有分清"这次凑巧知道"和"结构上保证知道"的区别。

---

## 4. dtype 转换与自动提升规则

**是什么:**
```
tf.cast(x, dtype)   # 显式转换dtype的标准方式
x.dtype              # tensor的dtype属性,只读,无法原地修改
```

**一句话:** TF 在 dtype 这件事上比 numpy/Python 苛刻得多——两个 dtype 不同的 tensor 直接做运算(比如 `float32 + int32`)**默认直接报错**,不会像 numpy 那样自动帮你提升到公共类型;唯一的"自动"只发生在"tensor 运算 Python 原生标量"这一种情况,此时标量会被悄悄转换成匹配 tensor 的 dtype。

**底层机制/为什么这样设计:**

用真实报错验证:`tf.constant(1.0) + tf.constant(1, dtype=tf.int32)` 直接抛 `InvalidArgumentError`,报错信息精确点出"expected...float tensor but is a int32 tensor"——TF 的每个 kernel(这里是 `AddV2`)在注册时就要求两个输入是完全相同的 dtype,没有为你做隐式提升的中间层,`int32 + int64` 同理报错。

这不是"TF 没做好",而是一个有意的设计取舍:自动类型提升本身就是隐藏 bug 的常见来源(numpy 里 `int + float -> float` 看似方便,但涉及隐式的、容易被忽略的精度损失和额外内存开销),在训练框架这种对数值精度、性能都极度敏感的场景下,TF 选择"强制显式 `tf.cast`,出问题第一时间报错",而不是"帮你偷偷转换,出问题也不告诉你"。

例外是 tensor 和 Python 原生 `int`/`float` 字面量运算——`tf.constant(1.0) + 2` 能正常工作,因为 `2` 会按照"参考另一个操作数的 dtype"这个 hint 被 `tf.convert_to_tensor` 转换成 `float32`。但如果隐式转换会造成精度丢失或语义不明确(比如给一个 `int32` tensor 加一个带小数的 Python `float` 字面量 `2.5`),同样会直接报错,而不是偷偷截断——这个反例已经实测验证。

补充(面试加分项,不是默认行为):TF 提供了一套可选的"NumPy 行为"开关 `tf.experimental.numpy.experimental_enable_numpy_behavior()`,打开后类型提升规则会向 numpy 的自动提升规则靠拢,但这是**显式选择的实验特性**,不是 Tensor 的默认行为——默认行为永远是"不同 dtype 直接报错"。

**AI 研究/工程场景:** 混合精度训练(`mixed_float16`)场景下,模型权重是 float32、前向计算用 float16,`loss` 计算前必须有清晰的 `tf.cast` 边界,不能依赖"运算会自动帮我处理精度"这种侥幸心理——这也是为什么 `LossScaleOptimizer`(08 篇会展开)相关代码里到处能看到显式的 `tf.cast` 调用。从 `tf.data` 管道读出来的整数 label(常见 `int64`)如果直接和 `float32` 的 logits 做运算会立刻报错,这是新手第一次搭 `tf.data` 管道时几乎必踩的坑。

**可运行例子:**
```python
import tensorflow as tf

# --- 不同dtype的tensor直接运算:默认直接报错,不会像numpy那样自动提升 ---
a = tf.constant(1.0)                      # float32
b = tf.constant(1, dtype=tf.int32)

try:
    _ = a + b
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "float tensor but is a int32 tensor" in str(e)

i32 = tf.constant(1, dtype=tf.int32)
i64 = tf.constant(1, dtype=tf.int64)
try:
    _ = i32 + i64
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "int32 tensor but is a int64 tensor" in str(e)

# --- 例外:tensor 和 python 原生标量运算,标量会被隐式转换成匹配的dtype ---
d = a + 2          # python int字面量
assert d.dtype == tf.float32 and d.numpy() == 3.0

e_ = a + 2.5
assert e_.dtype == tf.float32 and e_.numpy() == 3.5

f_ = b + 2          # int + int,同dtype家族
assert f_.dtype == tf.int32 and f_.numpy() == 3

# --- 但如果隐式转换会丢精度/破坏语义,同样直接报错,不会偷偷截断 ---
try:
    _ = b + 2.5      # int32 tensor + 带小数的python float
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "int32 tensor but is a float tensor" in str(e)

# --- 唯一正确路径:显式 tf.cast ---
h_ = tf.cast(b, tf.float32) + a
assert h_.dtype == tf.float32 and h_.numpy() == 2.0

# --- python混合list字面量:创建时就会被统一提升成公共dtype(这一步在tf.constant内部完成,不是运算时) ---
mixed = tf.constant([1, 2.5, 3])
assert mixed.dtype == tf.float32
assert mixed.numpy().tolist() == [1.0, 2.5, 3.0]

print("point 4 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.constant(1.0) + tf.constant(1, dtype=tf.int32)` 会发生什么?"—— 期望直接说出会报 `InvalidArgumentError`,而不是"自动转成 float"这种 numpy 思维的错误答案(这是一道专门筛选"有没有真的用过 TF"的问题)。
- **追问 1:** "那为什么 `tf.constant(1.0) + 2` 又能正常跑?这不矛盾吗?"—— 期望说清楚"tensor+tensor"和"tensor+python 标量"是两条不同规则,后者对标量做了 dtype hint 转换。
- **深挖追问(开放题):** "你觉得 TF 为什么要设计成不自动提升,而 numpy 要自动提升?这是不是 TF 的缺陷?"—— 期望能提出"训练框架对精度/性能极度敏感,宁可显式报错也不要隐式行为"这个工程取舍,而不是简单说"TF 更严格"。

**常见坑:** 从 `tf.data.Dataset` 或 pandas/numpy 管道里读出来的整数特征,dtype 经常是 `int64`(numpy 默认整数 dtype),直接和模型内部默认 `float32` 的权重/logits 做运算必然报错;标准做法是在数据管道入口处统一 `tf.cast(x, tf.float32)`,而不是走一步报一次错、见招拆招地到处补 cast。

---

## 5. `tf.device` 设备放置与 GPU/CPU 数据搬运

**是什么:**
```
with tf.device('/CPU:0'):   # 上下文管理器,块内创建的op/tensor优先放在指定设备
    ...
with tf.device('/GPU:0'):
    ...
tensor.device                # 只读属性,查看这个tensor实际所在的设备(完整字符串)
```

**一句话:** `tf.device()` 是一个"放置意愿"的上下文管理器——TF2 默认开启的软放置(soft device placement)策略会在你的意愿和"这个 op 到底有没有对应设备的 kernel 实现"之间做协调:能满足就满足,不能满足就安静地降级到 CPU,而不是直接报错崩溃。

**底层机制/为什么这样设计:**

实测 `tf.config.get_soft_device_placement()` 默认返回 `True`,这意味着"放置请求"不是硬性指令。用一个真实的反例验证软放置到底在做什么:在 `with tf.device('/GPU:0')` 里创建一个字符串 tensor,`.device` 属性显示确实是 GPU;但接下来对它调用 `tf.strings.upper()`(字符串操作没有 GPU kernel 实现),结果 tensor 的 `.device` 却变成了 CPU——TF 在发现"GPU 上没有这个 op 的 kernel"时,悄悄把这一步计算挪到 CPU 执行,不报错也不警告。关掉软放置(`tf.config.set_soft_device_placement(False)`)后重跑同样的代码,才会看到"这个请求本来应该失败"的真面目:报 `InvalidArgumentError`,明确写着 `Could not satisfy device specification...enable_soft_placement=0`。

跨设备直接做运算(CPU 上的 tensor 和 GPU 上的 tensor 直接相加)在当前实测环境下**能够直接成功**,TF 会自动在需要的地方插入拷贝——这是 TF2 eager 模式为了让心智负担最小化做的自动化,不需要像更底层的框架那样必须先手动搬到同一设备才能运算(虽然为了性能可控,大规模训练代码里仍然推荐显式管理设备放置,而不是依赖这种"能跑就行"的自动拷贝)。显式跨设备拷贝的标准写法是:在目标 `tf.device()` 作用域内对源 tensor 调用 `tf.identity()`。

**AI 研究/工程场景:** 推荐系统里的超大 embedding 表(可能几十 GB,超过单卡显存),经典做法是显式 `with tf.device('/CPU:0')` 把 embedding 变量放在 CPU/host 内存里,只把查表算出来的那一小片结果送上 GPU 参与后续计算——这是"模型太大放不下 GPU"这类工程约束下,`tf.device` 从"调试小技巧"变成"生产必需品"的真实场景;`tf.data` 的预处理管道(CPU 密集型的字符串解析、图像解码)也常显式钉在 CPU 上,把 GPU 完全留给矩阵运算。

**可运行例子:**
```python
import tensorflow as tf

assert tf.config.list_physical_devices('GPU'), "本篇例子要求GPU可见,请确认已激活 tf-venv"
assert tf.config.get_soft_device_placement() is True     # TF2默认开启软放置

# --- tf.device 只是"放置意愿",不是绝对保证 ---
with tf.device('/CPU:0'):
    c = tf.constant([1.0, 2.0, 3.0])
assert '/device:CPU:0' in c.device

with tf.device('/GPU:0'):
    g = tf.constant([1.0, 2.0, 3.0])
assert '/device:GPU:0' in g.device

# --- 显式跨设备拷贝:目标device作用域内 tf.identity ---
with tf.device('/CPU:0'):
    g_on_cpu = tf.identity(g)
assert '/device:CPU:0' in g_on_cpu.device
assert g_on_cpu.numpy().tolist() == g.numpy().tolist()

# --- 软放置的真实证据:请求GPU,但op没有GPU kernel时会静默降级到CPU ---
with tf.device('/GPU:0'):
    s = tf.constant(["hello", "world"])          # Const op可以"挂"在GPU上
    assert '/device:GPU:0' in s.device
    upper = tf.strings.upper(s)                    # 但字符串op没有GPU kernel
    assert '/device:CPU:0' in upper.device          # 被软放置悄悄挪到了CPU
    assert upper.numpy().tolist() == [b'HELLO', b'WORLD']

# --- 关掉软放置,才会看到"本该失败"的请求真正报错 ---
tf.config.set_soft_device_placement(False)
try:
    with tf.device('/GPU:0'):
        s2 = tf.constant(["a", "b"])
        _ = tf.strings.upper(s2)
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "Could not satisfy device specification" in str(e)
    assert "enable_soft_placement=0" in str(e)
finally:
    tf.config.set_soft_device_placement(True)      # 恢复默认,不影响后续代码

print("point 5 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "`with tf.device('/GPU:0')` 一定能保证这段代码在 GPU 上跑吗?"—— 期望答"不一定,取决于软放置策略 + 这个 op 有没有 GPU kernel"。
- **追问 1:** "如果一个 op 没有对应设备的 kernel 实现,TF 默认会怎么处理?"—— 期望答"软放置默认开启,会自动降级到有 kernel 的设备,不报错"。
- **深挖追问:** "怎么证明'降级'真的发生了,而不是猜的?"—— 期望能想到"打印执行前后 tensor 的 `.device` 属性对比,或者关掉软放置看它是否真的报错"。
- **追问(工程向):** "多 GPU 训练时,为什么有些代码会显式把某些变量放在 CPU 上?"—— 期望提到显存容量约束、host-device 拷贝的性能权衡。

**常见坑:** 误以为跨设备直接运算(CPU tensor + GPU tensor)在任何场景下都能无脑工作而不需要关心——虽然本例验证了简单四则运算能自动处理,但这种"隐式拷贝"一旦发生在训练循环内部的高频路径上,会成为难以察觉的性能黑洞(每一步都在偷偷做 host-device 拷贝而不自知),排查这类问题要靠 `tf.profiler`(10 篇展开)而不是靠"能跑就不管"的直觉。

---

## 6. ragged tensor 简介(变长序列场景)

**是什么:**
```
tf.ragged.constant(nested_list)     # 从"参差不齐"的嵌套python list直接创建
rt.values        # 拍平后的所有元素,一个规整的Tensor
rt.row_splits    # 每一"行"在values里的起止下标,长度 = 行数+1
rt.to_tensor(default_value=...)   # 转成补齐(pad)后的规整Tensor
```

**一句话:** `RaggedTensor` 是 TF 专门用来表示"每一行长度不一样"的数据(最典型是 NLP 里不等长的句子/token 序列)的原生类型——它不是偷偷补 0 再压缩,而是老老实实用 `values`(拍平的真实数据)+ `row_splits`(每行边界索引)两个规整 tensor 组合表达出"参差"的形状,不用为了凑成矩形而提前补 0 浪费内存,也不用为了避免补 0 被迫写变长 for 循环。

**底层机制/为什么这样设计:**

先看动机:普通 `tf.constant` 遇到长度不一致的嵌套 list 会直接拒绝——`tf.constant([[1,2,3],[4,5]])` 实测报 `ValueError: Can't convert non-rectangular Python sequence to Tensor.`,因为常规 Tensor 的底层内存模型要求严格矩形(dense),不存在"每行长度不同"这种形状。

`RaggedTensor` 的解法:把这份参差不齐的数据拆成两部分都是规整 tensor——`values` 是所有元素首尾相接拍平后的一维 tensor,`row_splits` 记录"第 i 行"对应 `values` 里 `[row_splits[i], row_splits[i+1])` 这一段。这样"形状不规整"这个麻烦被转移到了一个规整的"索引"tensor 上,而 `values` 本身可以像普通稠密 tensor 一样高效地做向量化运算——`ragged_rank` 表示这种"参差"嵌套了几层(普通二维参差是 1,list 的 list 的 list 是 2)。

关键是很多 elementwise 运算(`+`、`*`、`tf.reduce_mean(..., axis=1)` 等)在 `RaggedTensor` 上**直接支持运算符重载和 ragged-aware 实现**,不需要先手动 `to_tensor()` 补齐——比如 `tf.reduce_mean(rt, axis=1)` 会正确地"只在每行自己实际拥有的元素上"求平均,不会被 pad 出来的 0 污染均值,这正是"变长序列不想为了对齐而补 0 破坏统计量"这个诉求的直接解法。但两个 `row_splits` 结构不同的 `RaggedTensor` 直接相加会失败——实测报 `InvalidArgumentError: Condition x == y did not hold`,并把两边 `row_splits` 的差异点打印出来,这是"运算符重载不代表可以随便乱加,底层还是要求两边的参差结构完全对齐"这一事实的证据。

**AI 研究/工程场景:** NLP 里一个 batch 内的句子 token 数量天然不同,`tf.ragged.constant` 能让你在真正喂进 `Embedding` 层之前,先用 ragged 原生的 `row_lengths()`/`reduce_mean(axis=1)` 之类的操作做统计或简单池化,不必提前决定一个"padding 到多长"的超参数;到了真正要进 RNN/Transformer 这类要求规整输入的层时,再显式 `.to_tensor(default_value=0)` 补齐,并且通常还要配合一个"这一位是不是 padding 出来的"mask(常见做法是 `tf.sequence_mask(rt.row_lengths())`)。

**可运行例子:**
```python
import tensorflow as tf

# --- 动机:普通 Tensor 的内存模型要求严格矩形,参差数据直接被拒绝 ---
try:
    _ = tf.constant([[1, 2, 3], [4, 5]])
    assert False
except ValueError as e:
    assert "Can't convert non-rectangular Python sequence to Tensor" in str(e)

# --- RaggedTensor: values(拍平数据) + row_splits(每行边界) ---
rt = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])
assert rt.shape.as_list() == [3, None]
assert rt.values.numpy().tolist() == [1, 2, 3, 4, 5, 6]
assert rt.row_splits.numpy().tolist() == [0, 3, 5, 6]     # 第i行是 values[row_splits[i]:row_splits[i+1]]
assert rt.row_lengths().numpy().tolist() == [3, 2, 1]

# --- 补齐成规整Tensor(需要用到时才做,不是存储时就做) ---
dense = rt.to_tensor(default_value=0)
assert dense.numpy().tolist() == [[1, 2, 3], [4, 5, 0], [6, 0, 0]]

# --- 支持运算符重载和ragged-aware的规约,不用先补0 ---
rt_f = tf.ragged.constant([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]])
row_mean = tf.reduce_mean(rt_f, axis=1)
assert row_mean.numpy().tolist() == [2.0, 4.5, 6.0]        # 只在每行真实元素上求均值,不被pad的0拉低

doubled = rt_f * 2
assert doubled.to_tensor(default_value=0.0).numpy().tolist() == [[2.0, 4.0, 6.0], [8.0, 10.0, 0.0], [12.0, 0.0, 0.0]]

# --- 常见坑:两个 row_splits 结构不一致的 RaggedTensor,直接elementwise运算会报错,不会静默错位 ---
rt_a = tf.ragged.constant([[1, 2, 3], [4, 5]])
rt_b = tf.ragged.constant([[1, 2], [3, 4, 5]])
try:
    _ = rt_a + rt_b
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "Condition x == y did not hold" in str(e)

print("point 6 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "如果一个 batch 里句子长度不一样,你有几种处理方式?`RaggedTensor` 解决的是什么问题?"
- **追问 1:** "`RaggedTensor` 底层是怎么存的?是不是偷偷帮你补 0 了?"—— 期望说出 `values`+`row_splits` 这套"拍平+索引"的表达方式,强调不补 0、不浪费内存。
- **深挖追问:** "`tf.reduce_mean(ragged_tensor, axis=1)` 和'先 pad 补 0 再求 mean'比,结果为什么不一样?"—— 期望说出"补 0 会拉低均值(分母算上了不存在的 0),ragged-aware 的 reduce 只在真实元素上算",这是一个很容易被"我以为都一样"绊倒的问题。

**常见坑:** 两个 `RaggedTensor` 的结构(每行长度分布)不一致时直接做 elementwise 运算,不会有"自动广播/自动对齐"这种智能行为,会直接报错(而不是静默产生错误结果)——报错信息里的 `row_splits` 差异是排查的关键线索;另外 `.to_tensor()` 补齐之后如果忘记同步生成 mask,后续层(尤其是 RNN 的最后一个隐状态、attention 里的 padding 位置)可能会被 pad 出来的 0 污染计算结果,这是"表面上代码能跑、数值却悄悄错了"的典型陷阱。

---

## 7. sparse tensor 简介

**是什么:**
```
tf.sparse.SparseTensor(indices, values, dense_shape)   # 三元组:非零元素坐标、对应值、整体形状
tf.sparse.to_dense(st)             # 转成普通稠密Tensor(没记录的位置补0)
tf.sparse.reorder(st)               # 把indices重新排序成规范的行主序
```

**一句话:** `SparseTensor` 是"只记录非零元素在哪、值是多少"的三元组表示(`indices` + `values` + `dense_shape`),专门应付那种"维度巨大但绝大多数位置都是 0"的数据——用稠密 Tensor 存会直接因为内存爆炸变得不可行,用稀疏表示则只花费和"非零元素个数"成正比的内存。

**底层机制/为什么这样设计:**

一个 `[1000,1000]` 的矩阵如果只有 2 个非零元素,用稠密表示要存 100 万个数字,用 `SparseTensor` 只需要存 2 组坐标 + 2 个值——这个数量级差异在推荐系统的超高维 one-hot 特征交叉、超大词表 one-hot 编码这类场景下是决定"能不能训练得动"的关键,不是单纯的"省一点内存"。

大部分 `tf.sparse.*` 专用算子(比如 `tf.sparse.sparse_dense_matmul`)都要求 `indices` 按行主序排好,不满足会直接报错——实测 `tf.sparse.to_dense` 遇到乱序 `indices` 报 `InvalidArgumentError: indices[1] is out of order. Many sparse ops require sorted indices. Use tf.sparse.reorder to create a correctly ordered copy.`,报错信息本身就直接给出了修复方式。

和 `RaggedTensor` 一个值得对比、容易被面试问到的细节:`RaggedTensor` 支持 `*` 这类 Python 运算符重载直接做 elementwise 运算,但 `SparseTensor` **不支持**——实测 `sparse_a + sparse_b` 直接抛 `TypeError: unsupported operand type(s) for +: 'SparseTensor' and 'SparseTensor'`,必须老老实实调用 `tf.sparse.add(a, b)` 这样的专用函数;同理普通的 `tf.matmul` 也不认识 `SparseTensor`,必须用 `tf.sparse.sparse_dense_matmul`。这个设计差异背后的原因是:稀疏矩阵的"加法""乘法"在算法层面比稠密/ragged 复杂得多(要处理索引对齐、结果的稀疏模式可能变化等),TF 选择不通过运算符重载"假装"它和普通四则运算一样简单,而是逼你显式调用语义明确的专用 API。

**AI 研究/工程场景:** 推荐系统里"用户 ID/物品 ID 做 one-hot 再和 embedding 表相乘"这类操作,本质上就是一次 `sparse_dense_matmul`——直接算稠密矩阵乘法在维度是"词表大小"这个量级时完全不现实,`SparseTensor` + 专用稀疏算子是这类场景唯一可行的路径;`tf.sparse.SparseTensor` 也是 `tf.data` 管道处理变长/高维稀疏特征列(feature column)的标准载体之一。

**可运行例子:**
```python
import tensorflow as tf

# --- SparseTensor: indices(坐标) + values(值) + dense_shape(整体形状) ---
st = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dense = tf.sparse.to_dense(st)
assert dense.numpy().tolist() == [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]

# --- 内存量级对比:只存非零元素,不是存整个矩形 ---
big_sparse = tf.sparse.SparseTensor(indices=[[0, 500], [999, 3]], values=[1.0, 1.0], dense_shape=[1000, 1000])
assert len(big_sparse.values.numpy()) == 2                  # 稠密表示则需要 1_000_000 个元素

# --- 和 RaggedTensor 的关键差异:SparseTensor 不支持运算符重载 ---
st2 = tf.sparse.SparseTensor(indices=[[0, 0], [2, 1]], values=[10, 20], dense_shape=[3, 4])
try:
    _ = st + st2
    assert False
except TypeError as e:
    assert "unsupported operand type(s) for +: 'SparseTensor' and 'SparseTensor'" in str(e)

added = tf.sparse.add(st, st2)                                # 必须用专用函数
assert tf.sparse.to_dense(added).numpy().tolist() == [[11, 0, 0, 0], [0, 0, 2, 0], [0, 20, 0, 0]]

# --- 多数稀疏算子要求 indices 按行主序排好 ---
unordered = tf.sparse.SparseTensor(indices=[[1, 2], [0, 0]], values=[2, 1], dense_shape=[3, 4])
try:
    _ = tf.sparse.to_dense(unordered)
    assert False
except tf.errors.InvalidArgumentError as e:
    assert "indices[1] is out of order" in str(e)
    assert "tf.sparse.reorder" in str(e)

reordered = tf.sparse.reorder(unordered)
assert reordered.indices.numpy().tolist() == [[0, 0], [1, 2]]
assert tf.sparse.to_dense(reordered).numpy().tolist() == [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]

# --- 稀疏x稠密矩乘要用专用API,普通 tf.matmul 不认识 SparseTensor ---
sp = tf.sparse.SparseTensor(indices=[[0, 0], [0, 2], [1, 1]], values=[1.0, 2.0, 3.0], dense_shape=[2, 3])
w = tf.constant([[1.0], [1.0], [1.0]])
result = tf.sparse.sparse_dense_matmul(sp, w)
assert result.numpy().tolist() == [[3.0], [3.0]]

try:
    _ = tf.matmul(sp, w)
    assert False
except (ValueError, TypeError) as e:
    assert "SparseTensor" in str(e)

print("point 7 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "什么场景下你会用 `SparseTensor` 而不是普通 Tensor?"
- **追问 1:** "`SparseTensor` 支持 `+` 运算符吗?和 `RaggedTensor` 比呢?"—— 期望准确说出"不支持,`RaggedTensor` 支持",这个对比本身就是考察"是不是真的用过两者,而不是只知道概念"的好问题。
- **深挖追问:** "`to_dense` 报 'indices is out of order' 是什么意思,为什么会有这个要求?"—— 期望提到"多数稀疏算子的实现依赖有序索引才能高效工作,乱序通常是用户自己手动构造 `SparseTensor` 时没注意坐标顺序导致的"。

**常见坑:** 手动构造 `SparseTensor` 时如果 `indices` 没有按行主序排好(比如按数据原始出现顺序而不是坐标顺序塞进去),很多下游稀疏算子会直接报错要求先 `tf.sparse.reorder`——这个坑经常在"自己拼接/过滤 `SparseTensor` 的 indices"这种手写数据处理逻辑里出现,容易被误认为是"数据本身有问题"而不是"顺序不对"。

---

## 8. tensor 与 numpy 的内存共享/拷贝语义

**是什么:**
```
tensor.numpy()                    # EagerTensor -> numpy.ndarray
np.array(tensor)                   # 等价的另一种写法
tf.constant(np_array)               # numpy.ndarray -> Tensor
tf.convert_to_tensor(np_array)       # 同上,更通用的入口
```

**一句话:** 和 PyTorch 的 `torch.from_numpy()` 会刻意共享内存不同,TF 在 numpy 和 Tensor 之间的转换**无论哪个方向都是真拷贝**——修改其中一份数据,另一份完全不受影响,这是实测验证过的结论,不是"大概率"或"看情况"。

**底层机制/为什么这样设计:**

双向验证:① `numpy_array -> tf.constant` 之后再修改原始 numpy 数组,tensor 的值不变;② `tensor.numpy()` 拿到 numpy 数组后修改这份数组(`arr.flags.writeable` 确实是 `True`,可以写),原 tensor 的值同样不变;③ `np.array(tensor)` 这种等价写法结果一致;④ 哪怕 tensor 本来就在 GPU 上,`.numpy()` 拿到的数组照样是完全独立、可写的一份主机内存拷贝。

为什么 TF 不像 PyTorch 那样提供零拷贝路径:核心原因是 TF 的 Tensor 被设计成**不可变值**(第 1 点讲过,不可变性是 `EagerTensor` 最基本的契约),如果 `.numpy()` 返回的数组和 tensor 共享内存,那么"修改这个 numpy 数组"就等于间接破坏了 tensor 的不可变性保证,会让依赖"tensor 一旦创建值就不会变"这个前提的其它代码(执行引擎的内部假设、缓存逻辑等)出问题。反过来 `tf.constant(np_array)` 如果零拷贝,用户在 tensor 创建后修改原 numpy 数组也会造成同样的问题。用一次明确的拷贝换取"不可变性在任何情况下都成立"这个更强的保证,是 TF 在这里做的取舍。

`tf.Variable.numpy()` 同理是"某一时刻的快照拷贝",不是"持续同步的实时视图"——`v.assign(...)` 之后,之前调用 `.numpy()` 拿到的旧数组不会跟着变化,这从另一个角度印证了"每次 `.numpy()` 都是一次独立的、全新的内存拷贝",而不是返回同一块底层 buffer 的某种引用。这和第 1 点讲的 `ResourceVariable` 必须靠显式 `ReadVariableOp` 去读值是同一套自洽的设计:每次读到的都必须是"当前时刻的一份独立快照",不是随手给你一个内存地址。

**AI 研究/工程场景:** 用 `.numpy()` 在训练循环里做频繁的调试打印或者写入日志文件(比如每隔 N 步导出中间激活值分布做可视化)在语义上是完全安全的——不用担心"不小心通过日志代码修改了训练用的真实 tensor"这类跨语言常见的内存安全问题;代价是这种转换有真实的拷贝开销,GPU tensor 还要多付出一次设备到主机的搬运成本,不适合在训练主循环的高频路径上滥用(10 篇"内存与性能"会展开)。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# --- numpy -> tensor:创建时就是真拷贝,后续改原数组不影响tensor ---
np_arr = np.array([1.0, 2.0, 3.0])
t = tf.constant(np_arr)
np_arr[0] = 999.0
assert t.numpy().tolist() == [1.0, 2.0, 3.0]         # 没有跟着变

np_arr2 = np.array([1.0, 2.0, 3.0])
t2 = tf.convert_to_tensor(np_arr2)
np_arr2[0] = 777.0
assert t2.numpy().tolist() == [1.0, 2.0, 3.0]

# --- tensor -> numpy:同样是真拷贝,修改返回的数组不影响原tensor ---
t3 = tf.constant([1.0, 2.0, 3.0])
arr3 = t3.numpy()
assert arr3.flags.writeable is True                  # 可写,不是只读视图
arr3[0] = 999.0
assert t3.numpy().tolist() == [1.0, 2.0, 3.0]          # 原tensor不受影响

# --- 两次 .numpy() 调用返回的是两个完全独立的数组对象 ---
t4 = tf.constant([1, 2, 3])
a1 = t4.numpy()
a2 = t4.numpy()
assert a1 is not a2
a1[0] = 555
assert a2.tolist() == [1, 2, 3]                        # a1的修改不影响a2

# --- np.array(tensor) 是等价写法,结论一致 ---
t5 = tf.constant([1.0, 2.0])
arr5 = np.array(t5)
arr5[0] = 42.0
assert t5.numpy().tolist() == [1.0, 2.0]

# --- Variable.numpy() 是"当时那一刻"的快照,不是持续同步的实时视图 ---
v = tf.Variable([1.0, 2.0, 3.0])
snapshot = v.numpy()
v.assign([9.0, 9.0, 9.0])
assert snapshot.tolist() == [1.0, 2.0, 3.0]            # 快照不会跟着assign变
assert v.numpy().tolist() == [9.0, 9.0, 9.0]

# --- 即便tensor本来就在GPU上,.numpy() 拿到的也是独立、可写的host内存拷贝 ---
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gt = tf.constant([1.0, 2.0, 3.0])
    garr = gt.numpy()
    assert garr.flags.writeable is True
    garr[0] = 111.0
    assert gt.numpy().tolist() == [1.0, 2.0, 3.0]        # GPU上的原tensor不受影响

print("point 8 all assertions passed")
```

**面试怎么问 + 追问链:**
- **Q:** "`tensor.numpy()` 拿到的 numpy 数组,和原 tensor 是共享内存还是独立拷贝?你怎么证明?"—— 期望能现场说出"修改其中一个,看另一个变不变"这种验证方法,而不是凭印象回答。
- **追问 1:** "PyTorch 的 `torch.from_numpy()` 是共享内存的,TF 为什么不这样设计?"—— 期望连回"Tensor 不可变性"这个核心契约,而不是简单说"TF 比较保守"。
- **深挖追问:** "如果 TF 的 `.numpy()` 真的做了零拷贝,会带来什么潜在问题?"—— 期望能推演出"用户修改 numpy 数组会静默改变一个理论上不可变的 tensor,可能让依赖'值不会变'这个假设的其它代码产生错误结果"。

**常见坑:** 把 TF 当成 PyTorch 来用,想当然地认为 `tensor.numpy()` 能拿到一个"高效的零拷贝视图"用来做原地写入优化——实际上每次调用都有真实的拷贝成本,循环里反复对同一个不变的 tensor 调用 `.numpy()` 是纯浪费;反过来也不能依赖"改了 `.numpy()` 返回值就能影响原 tensor"这种在 TF 里根本不成立的 PyTorch 心智模型。

---

## 小结:这一批 8 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `tf.constant` vs `tf.Variable` | 不可变值 vs 可变状态;Variable 靠 resource handle + 显式 `ReadVariableOp` 实现"可变" |
| 2 | eager 执行与 `.numpy()` | 顶层 eager,`tf.function` 追踪期是 `SymbolicTensor`,没有 `.numpy()` |
| 3 | `TensorShape` 静态 vs 动态 | `.shape` 追踪期可能含 `None`;`tf.shape()` 永远是运行时才确定的 Tensor |
| 4 | dtype 转换与提升 | 不同 dtype 的 tensor 运算默认直接报错,只有 python 标量会被隐式转换 |
| 5 | `tf.device` 设备放置 | 软放置默认开启,没 kernel 就静默降级到 CPU,关掉才会真报错 |
| 6 | ragged tensor | `values`+`row_splits` 表达变长序列,ragged-aware 规约不被 padding 污染 |
| 7 | sparse tensor | `indices`+`values`+`dense_shape`,专用 API 而非运算符重载,要求有序索引 |
| 8 | tensor/numpy 内存语义 | 双向都是真拷贝,不像 `torch.from_numpy()` 共享内存,换来不可变性保证 |

下一批:[02-gradienttape-internals.md](02-gradienttape-internals.md) —— GradientTape 自动微分机制(全系列重中之重之一)。

---

*更新:2026-07-09*
