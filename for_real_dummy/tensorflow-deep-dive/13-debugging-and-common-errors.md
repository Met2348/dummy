# 13 · 调试与常见报错精解(Debugging and Common Errors)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这是本系列目前唯一一批主题从"机制怎么工作"切换成"机制坏掉时报错长什么样、怎么读、怎么定位"的内容——前面的 [02-gradienttape-internals.md](02-gradienttape-internals.md)、[03-tf-function-and-autograph.md](03-tf-function-and-autograph.md)、[04-keras-api-internals.md](04-keras-api-internals.md)、[10-memory-and-performance.md](10-memory-and-performance.md) 等章节讲的是机制本体,本篇要做的是反过来:拿着这些机制去解读一条条真实报错,而不是每次都靠 Google 报错文本、复制粘贴 Stack Overflow 上"试了好像有用"的答案却说不清原因。

**本篇和其它章节的关系:** 9 条报错分别对应 03(AutoGraph 转换边界)、01/04(动态 shape)、02(GradientTape 记录规则)、10(显存机制)、00 环境声明(GPU 库发现)、04/12(Keras 2/3 分裂)——每一条都会明确点出对应哪一章的机制,不重复推导机制本身,只讲"这个机制被违反/被误解时,报错文本说了什么、你该怎么读"。

**验证方法论(全系列统一,这一篇格外重要):** 本文**每一条报错/警告文本都是在 WSL2 `~/tf-venv`(TensorFlow 2.21.0,RTX 3080 Ti Laptop GPU,驱动 595.97)下现场触发后原样抄录**,不是凭经验转述或者从旧版本文档里搬运——TF 的报错文本经常随小版本演进变化字眼,如果你本机版本不同,文本可能有出入,但报错指向的机制原因是稳定的。凡是报错文本里包含内存地址、内部计数器 id 这类每次运行都会变的部分,代码里的校验用的是"关键子串"而不是整段全等比较,这和本文强调的验证方法本身保持诚实一致。**本篇很多"可运行例子"本身就是故意触发一个报错、用 `try/except` 接住它——这类代码块能正常跑完(exit 0)才是预期行为,"跑出报错"恰恰是这段代码正确工作的证明,不代表代码坏了。**

**关于"AI 研究/工程场景"段落:** 沿用 00 篇声明——仓库里没有真实 TensorFlow/Keras 代码可引用,这些场景是根据真实训练/部署中会遇到的具体问题重构的,不是仓库引用。

**本篇统一结构(与前面章节一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `tf.function` 追踪失败排查 —— AutoGraph 不是无所不能的 Python→图翻译器

**是什么:** `tf.function` 第一次被调用时,会先跑一遍 Python 函数体做"追踪"(tracing)——期间 AutoGraph 把能识别的 `if`/`for`/`while` 控制流改写成 `tf.cond`/`tf.while_loop` 等图原生算子。**追踪失败**指的是这个改写过程本身在 trace 阶段就没能完成,报错发生在"图还没构建完"这一步,不是图构建好之后真正执行时数值算错。

**一句话:** AutoGraph 能吃下的 Python 写法是一个经过设计、但**有明确边界**的子集;写法一旦越界,轻则在 trace 阶段直接报错(本节的主角),重则不报错但语义早就不是你以为的那样(那是第 6 节 retracing 陷阱的地盘)。

**底层机制/为什么这样设计:**

**场景 A——`if` 条件不是标量,AutoGraph 转不成 `tf.cond`:**

`tf.cond(pred, true_fn, false_fn)` 的设计前提是 `pred` 必须是单个 `bool`——它的语义是"二选一,整体执行其中一个分支",不是"逐元素挑"(逐元素的需求应该用 `tf.where`)。AutoGraph 把 Python 里的 `if <tensor条件>:` 改写成 `tf.cond` 时,天然继承了这个"必须标量"的约束——如果条件 tensor 的 `shape` 不是 `()`,追踪阶段直接拒绝。

**这里先插一句和"怎么跑例子"有关的真实坑,本节和第 6 节的代码都会用到:** AutoGraph 转换 `if`/`for` 这类控制流,依赖 `inspect.getsource()` 拿到函数的真实源码文本去做 AST 改写;如果被装饰的函数不是定义在一个真实存在的 `.py` 文件里(比如直接在 REPL 里敲,或者整段代码是当作字符串传给 `python -c "..."` 执行),`inspect.getsource()` 拿不到源码,AutoGraph 没法做完整转换,报错会变成完全不同的另一种(TF 自己的报错文本也印证了这点,末尾会提示"your source code may not be visible to AutoGraph")。为了让本节的例子不管用什么方式跑都能复现同一条真实报错,下面统一先把函数体写进一个真实的临时 `.py` 文件、再用 `importlib` 正常导入执行——这不是多余的工程,是确保"读到的报错文本"和"真实项目里(写在真实 `.py` 文件里)会看到的报错文本"一致:

```python
import importlib.util
import tempfile
import os
import tensorflow as tf

code = '''
import tensorflow as tf

@tf.function
def f_if(x):
    if x > 0:
        return x * 2
    else:
        return x * 3
'''
fd, path = tempfile.mkstemp(suffix="_agmod.py")
os.write(fd, code.encode())
os.close(fd)
spec = importlib.util.spec_from_file_location("_agmod_if", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # 真实文件路径,AutoGraph能正确inspect.getsource()

try:
    mod.f_if(tf.constant([1.0, -1.0, 2.0]))   # 条件是非标量 tensor
except ValueError as e:
    print(e)
    assert "condition of if statement expected to be" in str(e)
    assert "scalar" in str(e)

# 标量条件走同一份代码完全没问题,AutoGraph正确转成了tf.cond
r1 = mod.f_if(tf.constant(5.0))
r2 = mod.f_if(tf.constant(-5.0))
assert r1.numpy() == 10.0 and r2.numpy() == -15.0
os.unlink(path)
```

实测报错原文:
```
ValueError: in user code:

    File "...", line N, in f_if  *
        if x > 0:

    ValueError: condition of if statement expected to be `tf.bool` scalar, got Tensor("Greater:0", shape=(3,), dtype=bool); to check for None, use `is not None`
```

**场景 B——Python `list.append` 在 `tf.while_loop` 里"跨作用域"取值,不是 `for i in range(tensor)` 本身的问题:**

这里有一个容易被搞错的细节值得先澄清:`for i in range(n):` 这个具体写法,即使 `n` 是一个 Tensor,AutoGraph **也能正确处理**——这是 AutoGraph 对"直接写在 for 语句里的 `range(...)`"做的专门 AST 识别,不是通用规则(同样用上面提到的"写进真实临时文件再 `importlib` 导入"手法,确保 AutoGraph 能拿到源码):

```python
import importlib.util
import tempfile
import os
import tensorflow as tf

code = '''
import tensorflow as tf

@tf.function
def f_range(n):
    total = 0.0
    for i in range(n):          # n是Tensor,但这个写法被AutoGraph特殊识别
        total += tf.cast(i, tf.float32)
    return total
'''
fd, path = tempfile.mkstemp(suffix="_agmod.py")
os.write(fd, code.encode())
os.close(fd)
spec = importlib.util.spec_from_file_location("_agmod_range", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

r = mod.f_range(tf.constant(5))
assert r.numpy() == 10.0        # 0+1+2+3+4=10,追踪和执行都正确,不报错
os.unlink(path)
```

真正会炸的,是循环体内用一个**Python list** 去 `append` 每次迭代产出的 tensor,然后试图在循环结束后使用这个 list:

```python
import importlib.util
import tempfile
import os
import tensorflow as tf

code = '''
import tensorflow as tf

@tf.function
def f_list(n):
    out = []
    for i in tf.range(n):
        out.append(i)            # 每次迭代的i是while_loop body这个嵌套FuncGraph的局部张量
    return out                    # 试图把它带出while_loop的作用域
'''
fd, path = tempfile.mkstemp(suffix="_agmod.py")
os.write(fd, code.encode())
os.close(fd)
spec = importlib.util.spec_from_file_location("_agmod_list", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

try:
    mod.f_list(tf.constant(3))
except Exception as e:
    print(type(e).__name__, ":", e)
    assert "is out of scope and cannot be used here" in str(e)
os.unlink(path)
```

实测报错原文(节选,`id=...` 是进程相关的内存地址,每次运行都会变):
```
InaccessibleTensorError: <tf.Tensor 'while/Placeholder:0' shape=() dtype=int32> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
Please see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.

<tf.Tensor 'while/Placeholder:0' shape=() dtype=int32> was defined here:
    ...
The tensor <tf.Tensor 'while/Placeholder:0' shape=() dtype=int32> cannot be accessed from FuncGraph(name=f_list, id=...), because it was defined in FuncGraph(name=while_body_111, id=...), which is out of scope.
```

`tf.range(n)`(`n` 是 Tensor)驱动的 `for` 循环,被 AutoGraph 转成一个真正的 `tf.while_loop`——循环体是一个**独立的、嵌套的 FuncGraph**。这个子图每次迭代产出的 `i`,只是这个子图内部的一个占位符;Python 的 `out.append(i)` 只是在外层把这个占位符的引用塞进了一个普通 list,但循环真正在图里"跑"完之后,子图的生命周期已经结束,list 里存的引用自然也就失效了——这不是"list 不支持"这么简单,而是**图执行模型下,状态想要跨越循环边界流动,必须走"循环携带变量"(loop-carried variable,对应 return 值)这条明确的通道,Python 层面的旁路引用不算数**。这正是 [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) 会提到的 `tf.TensorArray` 存在的意义——它是 TF 专门设计的、能在图内合法地"每次迭代写一点、循环结束后统一读出"的容器:

```python
import importlib.util
import tempfile
import os
import tensorflow as tf

code = '''
import tensorflow as tf

@tf.function
def f_list_fixed(n):
    out = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    for i in tf.range(n):
        out = out.write(out.size(), i)   # TensorArray本身也是loop-carried变量(注意out=重新赋值)
    return out.stack()
'''
fd, path = tempfile.mkstemp(suffix="_agmod.py")
os.write(fd, code.encode())
os.close(fd)
spec = importlib.util.spec_from_file_location("_agmod_fixed", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

r = mod.f_list_fixed(tf.constant(3))
assert r.numpy().tolist() == [0, 1, 2]
os.unlink(path)
```

**AI 研究/工程场景:** 变长 batch 的推理逻辑里,根据 batch 内某个统计量(比如"这批里最长序列超过阈值就切一种更省显存的计算路径")做分支,如果这个统计量是非标量 tensor 规约后忘了转成标量,追踪阶段就会被场景 A 拦下来;自定义训练循环里想手写"每一步都收集一点中间结果供后续可视化/调试"的逻辑,eager 模式下用 Python list 测试一切正常,一旦为了性能包上 `@tf.function` 立刻在追踪阶段炸掉(场景 B),这是从 eager 原型转向图执行时很典型的一类"看起来能跑、一优化就炸"。

**面试怎么问 + 追问链:**
- **Q:** "`tf.function` 装饰的函数,第一次调用就直接报错,还没算出任何结果,你怎么判断这是不是 AutoGraph 转换失败,而不是数值/逻辑错误?"—— 期望能说出"追踪失败发生在图构建阶段,报错信息里通常会有 `in user code` 或者明确提到某个 AutoGraph 相关的转换问题,和运行时报错(比如 shape 不匹配)在语义上是两类不同阶段的错误"。
- **追问 1:** "为什么 `if` 的条件必须是标量?"—— 期望说出 `tf.cond` 的语义是"整体二选一执行一个分支",不是逐元素选择,后者是 `tf.where` 的职责,条件非标量意味着"该走哪个分支"这件事本身在语义上就没有单一答案。
- **深挖追问(区分度高):** "同样是 `tf.range` 驱动的循环,为什么示例里 `total += tf.cast(i, ...)` 完全没事,换成往 Python list 里 `append` 就报错?"—— 期望能说出"简单变量重新赋值天然符合 `while_loop` 的 loop-carried variable 语义,而 Python list 的 `append` 是一种旁路的、图执行模型无法追踪的可变状态改动,循环体本身是嵌套的子 FuncGraph,子图内的张量出了作用域就失效",能提到 `tf.TensorArray` 是加分项。

**常见坑:** 把"AutoGraph 追踪失败"和"图执行时的运行时错误"混为一谈,浪费时间去查数值/shape,实际上问题出在 trace 阶段代码本身没能被合法转换;另一个坑是看到 `range()` 就一律假设"Python 内置的 `range` 不能传 Tensor 会报错"——`for i in range(tensor):` 这个具体写法其实是被 AutoGraph 特殊照顾的,真正的雷区是循环体内部怎么"保存"每次迭代的结果,而不是 `range` 本身。

---

## 2. shape 不匹配 debug —— `None` 维度不是"不检查",只是把检查推迟到了运行时

**是什么:** Keras/`tf.function` 允许你在声明输入形状时,把某一维写成 `None`(比如 batch 维、变长序列的时间步维),表示"这一维的具体大小要等真正跑数据的时候才知道"。

**一句话:** `None` 维度不是"跳过检查",而是把"这一维必须匹配"这条约束从**定义模型/追踪函数的那一刻**,推迟到**真正传入具体数据的那一次调用**——这中间可能隔着好几行代码、好几次成功的调用,报错炸开的位置和真正埋雷的位置经常对不上。

**底层机制/为什么这样设计 + 可运行例子:**

Functional API 构建模型时,Keras 只会做**符号层面的形状推导**——`None` 和 `None` 在符号层面永远视为"兼容"(反正现在还不知道具体值,没法说它们不兼容),所以模型能顺利搭起来、`summary()` 也能正常打印,不会有任何报错:

```python
import tensorflow as tf

input_a = tf.keras.Input(shape=(None, 4))   # 序列长度未知,特征维固定为4
input_b = tf.keras.Input(shape=(None, 4))
added = tf.keras.layers.Add()([input_a, input_b])   # 符号shape (None,None,4)+(None,None,4) -> 兼容,不报错
model = tf.keras.Model([input_a, input_b], added)
assert added.shape.as_list() == [None, None, 4]      # 模型定义阶段,没有任何报错

# 第一次真实调用:两边序列长度恰好都是5,顺利跑通
a_ok = tf.random.normal((2, 5, 4))
b_ok = tf.random.normal((2, 5, 4))
out_ok = model([a_ok, b_ok])
assert out_ok.shape == (2, 5, 4)

# 第二次调用:序列长度不一致(5 vs 7),只有到这一步,不兼容才第一次被发现
a_bad = tf.random.normal((2, 5, 4))
b_bad = tf.random.normal((2, 7, 4))
try:
    model([a_bad, b_bad])
except tf.errors.InvalidArgumentError as e:
    print(e)
    assert "required broadcastable shapes" in str(e)
```

实测报错原文(节选):
```
InvalidArgumentError: Exception encountered when calling layer 'add' (type Add).

{{function_node __wrapped__AddV2_device_/job:localhost/replica:0/task:0/device:GPU:0}} required broadcastable shapes [Op:AddV2] name:

Call arguments received by layer 'add' (type Add):
  • inputs=['tf.Tensor(shape=(2, 5, 4), dtype=float32)', 'tf.Tensor(shape=(2, 7, 4), dtype=float32)']
```

**这个"延迟"不是 Keras Functional API 独有的,`tf.function` 的 `input_signature` 是同一套逻辑:**声明 `None` 只是告诉 tracing"这一维不用固定成某个具体数字也能追踪成功",完全不代表"这一维在运行时可以随便不匹配":

```python
import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec([None, 4]), tf.TensorSpec([None, 4])])
def add_fn(a, b):
    return a + b

# 装饰这一步本身不会触发任何追踪,自然也不可能在这一步报错
r_ok = add_fn(tf.random.normal((3, 4)), tf.random.normal((3, 4)))
assert r_ok.shape == (3, 4)

try:
    add_fn(tf.random.normal((3, 4)), tf.random.normal((5, 4)))   # 两个None维的实际值不一致
except tf.errors.InvalidArgumentError as e:
    print(e)
    assert "required broadcastable shapes" in str(e)
```

**AI 研究/工程场景:** 变长序列的 NLP/时序模型经常用 `(None, feature_dim)` 声明输入,如果数据管道里两路特征(比如文本 token 序列和对应的 mask 序列)理论上应该始终等长,但某个预处理分支有 bug 导致偶尔算出不一样的长度——单元测试如果恰好只覆盖了"长度一致"的样本,模型定义、`model.summary()`、甚至训练脚本跑的前几个 batch 都完全正常,直到线上遇到某条真实数据触发长度不一致,才第一次报错,这时候排查起来往往要往前追溯好几层数据处理逻辑,而不是模型定义那一行。

**面试怎么问 + 追问链:**
- **Q:** "Keras 模型里两路输入都声明成 `(None, ...)`,为什么模型能正常搭建、`summary()` 也不报错,但跑到某个 batch 突然报形状错误?"—— 期望说出"`None` 维度只在符号构建阶段被视为'兼容,因为还不知道具体值',真正的相等性约束要等实际数据流过才检查"。
- **追问 1:** "`tf.function` 的 `input_signature` 里写 `None`,是不是意味着这一维永远不会因为形状问题报错?"—— 期望能纠正这个误解:`input_signature` 的 `None` 只影响"要不要为每个不同的具体形状重新追踪"(呼应第 6 节),不影响运行时算子本身对形状合法性的检查。
- **追问 2(工程向):** "遇到这种'延迟报错',你怎么最快定位是数据管道的哪一步开始产生了不一致的形状?"—— 期望能提出"从报错位置往回追:先确认这条 batch 输入数据的真实 shape,再顺着数据管道逐步往前打印/断言每一步的 shape,而不是只盯着模型定义代码看"。

**常见坑:** 把"模型定义阶段没报错"当成"这段形状逻辑就是对的"的证据——`None` 维度会让很多本该在开发阶段发现的形状设计问题一路"闷声"活到线上某个具体输入才暴露;测试用例如果只覆盖了"凑巧形状一致"的输入,这类 bug 完全测不出来,针对 `None` 维度的模型,值得专门构造"故意让两路输入不一致"的测试用例。

---

## 3. eager 模式关闭排查 —— "为什么 `print` 出来的不是数值" 有两种完全不同的原因

**是什么:** `tf.executing_eagerly()` —— 查询当前代码是否处于 eager 执行模式(算子立即求值,能拿到真实数值)。TF2 默认全程 eager,但**至少两种场景**会让它返回 `False`,而且这两种场景的调试思路完全不是一回事。

**一句话:** 一种是"局部的、预期内的"——`tf.function` 追踪期间,这只是 03 篇讲过的图构建机制的正常表现;另一种是"全局的、多半是历史遗留代码导致的"——`tf.compat.v1.disable_eager_execution()` 把整个进程剩余生命周期都拖回 TF1 静态图心智模型,而且**没有回头路**。

**底层机制/为什么这样设计 + 可运行例子:**

**场景 A——`tf.function` 追踪期间,`executing_eagerly()` 为 `False` 是设计使然:**

```python
import tensorflow as tf

assert tf.executing_eagerly() is True   # 顶层默认eager

@tf.function
def f(t):
    print("追踪期间:", tf.executing_eagerly())
    return t * 2

x = tf.constant([1.0, 2.0, 3.0])
r = f(x)
assert r.numpy().tolist() == [2.0, 4.0, 6.0]   # 结果完全正确,eager=False不代表算错了
```

追踪期间,函数参数已经不是 eager 模式下那种"背后有真实数值的 `EagerTensor`",而是纯粹的图节点占位符——这也是为什么在 `tf.function` 内部对参数调用 `.numpy()` 会直接报错:

```python
import tensorflow as tf

@tf.function
def g(t):
    try:
        t.numpy()
        return "no error"
    except AttributeError as e:
        return str(e)

result = g(tf.constant([1.0, 2.0]))
assert result.numpy().decode() == "'SymbolicTensor' object has no attribute 'numpy'"
```

这不是 bug,是**在追踪阶段,参数根本不存在"当下的具体数值"这个概念**——`.numpy()` 需要一个真实存过的值,而追踪只是在构建"以后要怎么算"的图,这个语义在 [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) 会展开;这里要记住的是:**只要是在 `@tf.function` 装饰的函数追踪期间看到 `executing_eagerly()` 是 `False`,这是正常现象,不是排查目标**。

**场景 B——`tf.compat.v1.disable_eager_execution()`:全局关闭,而且关掉之后当场就回不去了:**

这是 TF1 遗留代码库里常见的写法(很多 TF1 教程/迁移脚本会在文件顶部调用它),一旦执行,**整个进程剩余的生命周期**都会切回 TF1 的 define-and-run 心智模型:

```python
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
assert tf.executing_eagerly() is False

a = tf.constant(3)
b = tf.constant(4)
c = a + b
print("直接print(c):", c)   # 不是7,是一个尚未求值的符号张量描述

try:
    c.numpy()
except AttributeError as e:
    assert "object has no attribute 'numpy'" in str(e)

# 必须用session才能拿到真实数值,这是TF1的标准用法
with tf.compat.v1.Session() as sess:
    val = sess.run(c)
    assert val == 7

# 关键坑:试图重新打开eager,不会成功
try:
    tf.compat.v1.enable_eager_execution()
except ValueError as e:
    assert str(e) == "tf.enable_eager_execution must be called at program startup."
```

实测:`print(c)` 的输出是 `Tensor("add:0", shape=(), dtype=int32)`——和 TF1 时代教程里的截图一模一样;而**试图重新 enable 会直接报错**,`ValueError: tf.enable_eager_execution must be called at program startup.`,也就是说这个开关本质上只在解释器刚启动、还没执行任何图相关操作时才允许扳动,一旦调用过 `disable_eager_execution()`,这个进程就回不到 eager 模式了。

**AI 研究/工程场景:** 排查一个"明明按 TF2 教程写的代码,却各种行为像 TF1"的老项目时,最先要做的排查不是逐行读业务逻辑,而是搜索整个代码库(包括被 import 的第三方内部工具库)有没有任何地方调用了 `disable_eager_execution()`——因为它的作用域是全局的、且不可逆,只要进程里任何一处执行过这行代码,后面写的所有"看起来很 TF2"的代码全都会以 TF1 语义运行,新人几乎不可能靠只读自己那部分代码发现问题。

**面试怎么问 + 追问链:**
- **Q:** "`tf.executing_eagerly()` 返回 `False`,你怎么判断是正常现象还是代码有问题?"—— 期望能区分"在 `tf.function` 追踪期间"(正常)和"顶层/eager上下文里意外返回False"(说明进程某处调用过 `disable_eager_execution`)这两种情况。
- **追问 1:** "如果确认是 `disable_eager_execution()` 导致的,怎么恢复?"—— 期望直接答"不能在运行时恢复,这个函数设计上只能在程序启动时调用一次;唯一可靠的办法是找到并删掉那行调用,而不是在后面加一行 `enable_eager_execution()` 去'修复'"。
- **追问 2(工程向):** "为什么 TF 要把这个开关设计成'一旦关闭就不能在运行时重新打开'?"—— 期望能推理出"eager/graph 是两套底层执行机制,已经创建的资源/已经追踪过的图状态很可能和另一套机制不兼容,允许运行时来回切换的工程复杂度和潜在 bug 面远大于收益,不如直接约束成'只能在启动时决定一次'"。

**常见坑:** 看到 `tf.function` 内部 `executing_eagerly()` 是 `False` 就当成 bug 去排查,浪费时间;反过来,遇到全局 eager 模式失效的老代码,想当然地在报错的地方加一行 `enable_eager_execution()` 想"修复",却不知道这个调用本身就会因为"不是在程序启动时调用"而报错,治本的办法是删掉 `disable_eager_execution()` 那一行,而不是想办法在后面"重新打开"。

---

## 4. GPU 未被识别 + OOM 排查 —— 先确认卡真的在用,再谈显存不够

**是什么:** 两类不同层次的"GPU 资源"排查——第一层是"TensorFlow 压根没发现这块卡"(`tf.config.list_physical_devices('GPU')` 返回空列表,但 `import tensorflow` 本身不报任何错),第二层是"卡确实在用,但这次要分配的显存超过了物理上限"(`tf.errors.ResourceExhaustedError`)。这两层放在一条里讲,是因为第一层排查不清楚,第二层的讨论根本无从谈起——一个 `list_physical_devices('GPU')` 都是空的进程,不可能产生 GPU OOM,它所有计算都静默跑在 CPU 上。

**一句话:** GPU 没被识别是"没有任何异常,只有一个不易察觉的空列表和一条不会明说缺哪个库的 WARNING";GPU OOM 是"有明确的异常类型和报错文本,但 TF 给的字段远没有 PyTorch 丰富(呼应 [10-memory-and-performance.md](10-memory-and-performance.md))"。

**底层机制/为什么这样设计:**

**第一层——GPU 库分散在 pip 装的 `nvidia/*/lib` 子目录里,`LD_LIBRARY_PATH` 不对,TF 就"安静地"退回 CPU:**

这是 00 篇环境声明里记录的、本系列搭建环境时踩到的第一个真实坑:`pip install tensorflow[and-cuda]` 把 CUDA 运行时拆成十几个独立的 `nvidia-*-cu12` 包,每个包各自的 `.so` 文件躺在自己的 `site-packages/nvidia/<组件名>/lib/` 目录下(本机实测,`~/tf-venv` 里这样的目录有 11 个:`cusparse`、`cuda_nvrtc`、`cuda_runtime`、`cublas`、`curand` 等各占一个),没有任何一个是 `ld.so` 默认会去找的路径。下面用一个自包含的例子复现这个坑——用子进程故意剥离 `LD_LIBRARY_PATH`,不需要依赖你本机是不是恰好环境配置错误也能稳定复现:

```python
import os
import subprocess
import sys

env_broken = dict(os.environ)
env_broken.pop("LD_LIBRARY_PATH", None)   # 故意模拟"没有source激活脚本"的状态
env_broken["TF_CPP_MIN_LOG_LEVEL"] = "0"   # 不压制WARNING,让线索完整出现

code = "import tensorflow as tf; print('GPU_LIST:', tf.config.list_physical_devices('GPU'))"
result = subprocess.run(
    [sys.executable, "-c", code],
    env=env_broken, capture_output=True, text=True, timeout=60,
)
assert result.returncode == 0                       # 注意:import本身不报错,进程正常退出
assert "GPU_LIST: []" in result.stdout                # GPU列表是空的
assert "Cannot dlopen some GPU libraries" in result.stderr
# 报错信息说"missing libraries mentioned above",但翻遍整段stderr都不会出现任何具体的.so文件名——
# 这正是00篇环境声明里记录的真实坑:症状明确,但TF自己给的诊断信息不够精确,不会明说缺的是哪个库
assert "libcudart" not in result.stderr
assert "libcublas" not in result.stderr
assert "libcudnn" not in result.stderr
```

实测 `result.stderr` 的关键片段:
```
I0000 ... cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions...
W0000 ... gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries
mentioned above are installed properly if you would like to use GPU. Follow the guide at
https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
```

**排查方法论(不是"随便设一下 `LD_LIBRARY_PATH` 试试",而是有顺序的):**

1. **先确认硬件/驱动层没问题**,这一步和 Python/venv 完全无关:`nvidia-smi` 应该能看到卡(本机实测输出 `NVIDIA GeForce RTX 3080 Ti Laptop GPU, 595.97, 16384 MiB`)。如果这一步都不行,问题在驱动/WSL2 GPU 直通配置,不在 TF。
2. **确认症状**:`tf.config.list_physical_devices('GPU')` 是空列表,但 `import tensorflow` 没有抛异常——这个组合基本就是库加载失败,而不是驱动问题(驱动真的坏了,`nvidia-smi` 这层会先报错)。
3. **检查 `LD_LIBRARY_PATH` 有没有设置、指向哪里**,和 nvidia 系列 pip 包实际安装到了哪些目录做对比。本机正确配置下 `LD_LIBRARY_PATH` 展开后是 11 条路径,每条对应一个 `nvidia/<组件>/lib` 子目录。
4. **应用修复**(00 篇环境声明里记录的命令,本质是把所有 `nvidia/*/lib` 子目录动态拼成一条 `LD_LIBRARY_PATH`):
   ```bash
   export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)
   ```
5. **确认修复生效**——重开一个正确 `source` 过 `activate` 的进程,`list_physical_devices('GPU')` 应该非空(本系列所有其它章节的例子都是在这个状态下跑的)。

**一个走过的弯路,诚实记录:** 最初尝试过用 `ctypes.CDLL("libcudart.so.12")` 之类的手段去"精确定位到底缺哪个 `.so`",但发现这个思路在这个场景下并不可靠——只要 `import tensorflow` 这一步已经执行过(哪怕 GPU 注册失败),某些 CUDA 库可能已经通过 TF 自己的内部加载机制被装进了进程地址空间,这时候再用 `ctypes.CDLL` 去探测同名库,拿到的是"已加载"的假象,不代表它真的能通过 `LD_LIBRARY_PATH` 搜索路径找到。**真正该核对的是 `LD_LIBRARY_PATH` 环境变量本身的内容,和 pip 包实际安装路径是否对得上,而不是逐个猜库文件名。**

**第二层——GPU 确实被识别了,但这次分配超过了物理显存,`ResourceExhaustedError`:**

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0   # 前提:第一层的坑已经排除,卡确实被识别了

mem_before = tf.config.experimental.get_memory_info('GPU:0')
print("分配前:", mem_before)

try:
    with tf.device('/GPU:0'):
        x = tf.random.normal((100000, 100000, 100))   # 故意申请一个天文数字大小
except tf.errors.ResourceExhaustedError as e:
    print(e)
    assert "OOM when allocating tensor" in str(e)
    assert "GPU_0_bfc" in str(e)   # bfc = best-fit-with-coalescing,TF默认GPU分配器的名字
```

实测报错原文:
```
{{function_node __wrapped__RandomStandardNormal_device_/job:localhost/replica:0/task:0/device:GPU:0}}
OOM when allocating tensor with shape[100000,100000,100] and type float on
/job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:RandomStandardNormal] name:
```

和 PyTorch 那条动辄四五个数字(`Tried to allocate`/`total capacity`/`free`/`allocated by PyTorch`/`reserved but unallocated`)的 OOM 报错相比,TF 这条**明显更"惜字如金"**——只给了申请的 shape/dtype 和分配器名字,不会像 PyTorch 那样在报错文本里直接摊开 allocated/reserved/free 的分解。这不是 TF 偷懒,而是内存管理的实现细节本来就不同:本系列环境变量里设了 `TF_FORCE_GPU_ALLOW_GROWTH=true`(呼应 [10-memory-and-performance.md](10-memory-and-performance.md)),显存按需增长而不是启动时独占整卡,但报错文本本身不会因为这个设置变得更详细——**想要更细的数字,要主动查,不会自动给你**:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
assert len(gpus) > 0
mem_info = tf.config.experimental.get_memory_info('GPU:0')
print("current:", mem_info["current"], "peak:", mem_info["peak"])
assert "current" in mem_info and "peak" in mem_info
# 想看更细的分池信息(不同大小请求各自的分配情况),TF没有和torch.cuda.memory_summary()
# 完全对等的公开API,这是TF内存自省能力上比PyTorch弱的一个实际差距,排查时需要用
# get_memory_info在关键节点前后手动打点对比,而不是指望一次调用就拿到全貌
```

**AI 研究/工程场景:** 云端多卡机器上,新同事第一次配置自己的训练环境,`pip install tensorflow` 之后发现训练"能跑但巨慢"——这是第一层坑最容易被误诊的表现:没人告诉你 GPU 没被用到,代码不报错、loss 也在下降,只是每个 step 慢得离谱,直到有人想起来查一下 `list_physical_devices('GPU')` 才发现列表是空的;OOM 排查则更多出现在"别人调好的模型/batch size,换了台显存更小的机器直接跑"这种场景,需要养成先看 `get_memory_info` 的 `current`/`peak`,再决定是缩小 batch 还是别的省显存手段。

**面试怎么问 + 追问链:**
- **Q:** "`tf.config.list_physical_devices('GPU')` 返回空列表,但代码没有任何报错,你怎么排查?"—— 期望不是立刻怀疑驱动坏了,而是分层排查:先 `nvidia-smi` 确认硬件/驱动层没问题,再检查 Python 层面的库加载(`LD_LIBRARY_PATH` 之类)。
- **追问 1:** "TF 的 WARNING 说 'Cannot dlopen some GPU libraries',但没说清楚是哪个库,你会怎么进一步定位?"—— 期望能提出"核对 `LD_LIBRARY_PATH` 实际内容和 pip 装的 nvidia 系列包的真实安装路径",也可以聊到"用 `ctypes.CDLL` 逐个探测库文件名"这个思路本身有局限(可能因为库已被间接加载而给出误导性的"成功"),体现出这不是拍脑袋定位,而是有过验证、知道哪些方法可靠哪些不可靠。
- **追问 2(联系10篇):** "GPU OOM 报错里,TF 给的信息和 PyTorch 比起来有什么不同?"—— 期望能说出"TF 的 OOM 报错只给 shape/dtype/分配器名字,没有 PyTorch 那种 allocated/reserved/free 的明细分解,需要主动调用 `get_memory_info` 才能拿到更细的数字"。

**常见坑:** GPU 没被识别时,把"训练变慢"误诊断成模型太大/数据管道太慢,花大量时间优化不该优化的地方,而没有意识到根本没在用 GPU;OOM 排查时看到报错就本能地调小 batch size,不看 `get_memory_info` 里 `current`/`peak` 的具体数字,也不检查是不是有未释放的中间变量持续占着显存导致"看起来"不该 OOM 的规模也 OOM 了。

---

## 5. NaN/Inf 定位 —— `enable_check_numerics()` 把"事后诸葛亮"变成"当场抓包"

**是什么:**
```
tf.debugging.enable_check_numerics()   # 全局开关:此后每个op的每个输出都会被检查
```
开启后,TF 会在**每一个算子执行完、产出结果的那一刻**检查输出里有没有 NaN/Inf,一旦发现,立刻在这个算子这里抛异常,而不是让 NaN 静默地继续往后传播、最后只在 loss 这种终点位置才被发现。

**一句话:** 不开这个开关,NaN 是"事后"才被你发现的(往往是几层计算之后,loss 变成 NaN 才第一次注意到);开了之后,NaN/Inf 第一次出现的那个算子会**当场**把执行拦下来,报错信息直接告诉你是哪个 op、什么形状、输入是什么。

**底层机制/为什么这样设计:**

**先不开这个开关,看 NaN 是怎么"静默传播"的:**

```python
import tensorflow as tf

x = tf.constant([1000.0, 1.0, 2.0])
h1 = tf.exp(x)                    # exp(1000)数值溢出成+inf
assert bool(tf.reduce_any(tf.math.is_inf(h1)))
print("h1:", h1.numpy())           # [inf, 2.7182817, 7.389056]

h2 = h1 - h1[0]                     # inf - inf = nan(经典nan生成模式)
assert bool(tf.reduce_any(tf.math.is_nan(h2)))
print("h2:", h2.numpy())           # [nan, -inf, -inf]

h3 = h2 * 2                          # nan会一直静默地传播下去,不会自己消失也不会报错
assert bool(tf.reduce_any(tf.math.is_nan(h3)))
print("h3:", h3.numpy())           # [nan, -inf, -inf]
```

和 [11-debugging-and-common-errors.md](../torch-deep-dive/11-debugging-and-common-errors.md)(torch-deep-dive 同一位置的知识点)讲的规律完全一致:**NaN 几乎不是凭空出现的,通常是先有 Inf(数值溢出),这个 Inf 后续参与了 `inf-inf`/`inf*0`/`0/0` 这类运算才变成 NaN**——排查时只搜"NaN 第一次出现在哪"经常已经晚了一步,真正的病根(第一次出现 Inf 的地方)在更早的算子。

**开启 `enable_check_numerics()`,同样的溢出,这次当场被拦下来:**

```python
import tensorflow as tf

tf.debugging.enable_check_numerics()

try:
    x2 = tf.constant([1000.0])
    h1b = tf.exp(x2)               # 这一步产出+inf,立刻被拦下,不会等到后面的减法才报错
    h2b = h1b - h1b[0]
    print("不应该跑到这里:", h2b.numpy())
except tf.errors.InvalidArgumentError as e:
    print(e)
    assert "Detected Infinity or NaN in output" in str(e)
    assert 'op "Exp"' in str(e)     # 明确点名是Exp这个op产出的
```

实测报错原文:
```
InvalidArgumentError: {{function_node __wrapped__CheckNumericsV2_device_/job:localhost/replica:0/task:0/device:GPU:0}}

!!! Detected Infinity or NaN in output 0 of eagerly-executing op "Exp" (# of outputs: 1) !!!
  dtype: <dtype: 'float32'>
  shape: (1,)
  # of +Inf elements: 1

  Input tensor: tf.Tensor([1000.], shape=(1,), dtype=float32)

 : Tensor had +Inf values [Op:CheckNumericsV2] name:
```

对比不开开关时的版本:不开时,你只能看到**最终结果里有 NaN**,得自己一层层回溯是哪一步先出问题;开了之后,报错**直接点名是 `Exp` 这个 op**,连它的输入张量具体数值(`[1000.]`)和"+Inf 元素个数"都一并打印——不用你自己写二分排查代码。

**这个机制在 `tf.function` 内部同样有效(不只是 eager 模式的专利):**

```python
import tensorflow as tf

tf.debugging.enable_check_numerics()

@tf.function
def bad_fn(x):
    h = tf.exp(x)
    return h - h[0]

try:
    bad_fn(tf.constant([1000.0]))
except tf.errors.InvalidArgumentError as e:
    print(str(e)[:300])
    assert "Detected Infinity or NaN in output" in str(e)
    assert 'graph op "Exp"' in str(e)
    assert "bad_fn" in str(e)      # 会指出具体是哪个被追踪的函数
```

图模式下报错文本换了个说法(`graph op` 而不是 `eagerly-executing op`),但同样精确点名了产出异常值的 op,还带上了这个 op 是在哪个 `tf.function`(`bad_fn`)里创建的调用栈——这一点和 torch-deep-dive 里 `detect_anomaly()` 的设计目的高度一致(都是用运行时开销换"精确定位"),但 TF 这里做的检查更直接:不需要额外的上下文管理器包住 forward+backward,只要全局开一次就对之后所有 op 生效。

**AI 研究/工程场景:** 大模型训练几千步之后 loss 突然变成 NaN,是训练里排查成本很高的经典问题——`enable_check_numerics()` 因为要对**每个 op** 做检查,开销不小(不适合在正式大规模训练任务里全程开启),标准做法是先想办法用更小的模型/更少的数据/固定的种子复现问题,复现之后再开这个开关,当场把第一次出现异常值的具体算子和输入揪出来,而不是在几十层网络里凭经验瞎猜是哪一层的问题。

**面试怎么问 + 追问链:**
- **Q:** "训练中途 loss 变成 NaN,你怎么定位是哪一步先出的问题?"—— 期望能提出系统性方法,而不是"调小学习率试试"这种没有诊断过程的猜测。
- **追问 1(核心机制理解):** "NaN 一般是怎么产生的?"—— 期望说出"通常先有 Inf(数值溢出),Inf 参与 `inf-inf`/`inf*0`/`0/0` 这类运算后才变成 NaN",这决定了排查要往"更早"的地方找,不是只看 NaN 第一次出现的位置。
- **追问 2(工具向):** "`tf.debugging.enable_check_numerics()` 具体是怎么做到精确定位的?为什么不干脆一直开着?"—— 期望说出"对每个op的每次输出都做检查,开销和检查的op数量成正比,所以适合'复现问题后针对性开启',不适合全程跑在正式训练任务上"。

**常见坑:** 只检查"loss 是不是 NaN",不往前追查是模型内部哪一步先产生的异常值,直接上调小学习率/加梯度裁剪这类"通用止血"手段——训练有时确实不崩了,但真正的病根(某层权重初始化不合理、某处存在真实的数值溢出风险)没被解决,换个数据分布或超参又可能复发;另一个坑是 `enable_check_numerics()` 开着调试完之后忘了关(或者忘了这是进程级全局状态,想在同一个脚本后面跑一段正式训练),导致后续代码莫名变慢却想不起来原因。

---

## 6. retracing 性能陷阱的识别与修复 —— 图跑得对,但你可能根本没在用图

**是什么:** [03-tf-function-and-autograph.md](03-tf-function-and-autograph.md) 讲过 retracing 的触发条件(输入 shape/dtype 变化、传入 Python 原生对象而非 tensor 等)。本节要解决的是一个更实际的问题:**你怎么知道自己的代码正在反复触发 retrace**——很多时候代码"跑起来完全正确",唯一的问题是性能远低于预期,而 retracing 本身不会报错,只会默默地一遍遍重新追踪。

**一句话:** 识别 retracing 有两条现成的路——**看 TF 自己主动打的 WARNING 日志**,以及**自己在函数体里埋一个计数器,数它被"追踪"了几次而不是"调用"了几次**;两者结合能又快又准地确认问题,而不是凭"感觉变慢了"去猜。

**底层机制/为什么这样设计:**

**方法 1——TF 会在 retrace 次数明显偏多时,主动打印一条 WARNING(不需要你自己配置任何东西):**

```python
import tensorflow as tf

@tf.function
def f(x):
    return x * 2

for i in range(6):
    f(i)   # 每次传入的是不同的Python int(不是tensor!)——每个不同的值都会触发一次新的追踪
```

实测:跑到第 5、第 6 次调用时,TF 自动打印(通过标准 `logging`,不是异常,不会中断执行):
```
WARNING:tensorflow:5 out of the last 5 calls to <function f at 0x...> triggered tf.function retracing.
Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function
repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead
of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has
reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to
https://www.tensorflow.org/guide/function#controlling_retracing and
https://www.tensorflow.org/api_docs/python/tf/function for more details.
```

这条 WARNING 本身就是一份现成的排查清单——直接列出了三类最常见的根因,不用你自己去猜。

**方法 2——自己动手验证:在函数体里放一个只在"追踪"时才会执行的计数器,对比"调用次数"和"追踪次数":**

```python
import tensorflow as tf

trace_count = {"n": 0}

@tf.function
def f(x):
    trace_count["n"] += 1   # 这行代码只在追踪期间被Python解释器真正执行一次,
    return x * 2              # 图跑起来之后这行不会再被执行(它不是图的一部分,是trace时的副作用)

for i in range(6):
    f(i)                       # 传Python int,每次都是不同的值
assert trace_count["n"] == 6   # 6次调用触发了6次追踪——retrace了个彻底

# 用pretty_printed_concrete_signatures()能直接看到这6个各自独立的concrete function
sig_text = f.pretty_printed_concrete_signatures()
assert sig_text.count("Input Parameters:") == 6
```

**对照组:换成传真正的 Tensor(而不是 Python 原生值),同样是 6 次调用,只追踪 1 次:**

```python
import tensorflow as tf

trace_count2 = {"n": 0}

@tf.function
def g(x):
    trace_count2["n"] += 1
    return x * 2

for i in range(6):
    g(tf.constant(i, dtype=tf.float32))   # 传Tensor,shape/dtype都一致

assert trace_count2["n"] == 1   # 只追踪了1次,后面5次调用直接复用同一份concrete function
```

**为什么 Python int 和 Tensor 待遇差这么多:** `tf.function` 判断"要不要复用已有的 concrete function"的依据,对 Tensor 参数是看 `(shape, dtype)`(具体数值可以变,追踪出的图靠占位符处理运行时的真实值);但对 Python 原生对象(`int`/`float`/`str`/`bool`/`None`……)是把**具体值本身**当成追踪缓存 key 的一部分——因为这类值在追踪出的图里会被当成编译期常量直接烧进图结构(比如不同的 `int` 可能走不同的 Python `if` 分支),TF 没法假设"数值不同但图结构一样",只能对每个不同的值都重新追踪一次。这正是为什么"循环里传不同的 Python 数字给 `tf.function`"是最容易被无意中写出来的性能陷阱之一。

**AI 研究/工程场景:** 超参搜索脚本里,循环内反复调用同一个被 `@tf.function` 装饰的训练 step 函数,把当前的 `learning_rate`/`epoch` 这类 Python 数字也当成参数直接传进去(而不是包成 `tf.Variable` 或 `tf.constant`)——外层循环跑几十上百次,每次数值都不同,等于每次都触发一次完整重新追踪,训练"能跑但一直很慢"且不会有任何报错提示哪里错了,只有 TF 自己打的那条 retracing WARNING,如果没人留意日志,很容易被当成"这个模型天生就这么慢"。

**面试怎么问 + 追问链:**
- **Q:** "怎么判断一个 `tf.function` 装饰的函数是不是在反复不必要地 retrace?"—— 期望提到"看 TF 自动打的 retracing WARNING",以及"自己在函数体里埋计数器,对比调用次数和函数体真正被 Python 解释器执行的次数"这两条路。
- **追问 1(核心):** "为什么传 Python int 每次都会触发重新追踪,传 Tensor 就不会?"—— 期望能说出"Tensor 的追踪缓存 key 是 (shape, dtype),具体数值靠占位符在图里处理;Python 原生值本身会被当成追踪缓存 key 的一部分,因为它可能被编译进图结构本身(比如决定走哪个if分支),TF 没法假设不同的值能安全复用同一张图"。
- **追问 2(修复向):** "发现了不必要的 retracing,有哪些修复手段?"—— 期望能提到"把 Python 数字包成 `tf.constant`/`tf.Variable` 再传入"、"用 `input_signature` 固定签名"、"`reduce_retracing=True` 让 TF 自动尝试放宽形状匹配"这几条(WARNING 文本本身就提示了后两条)。

**常见坑:** 只凭"感觉这个模型跑得比预期慢"去优化计算本身(换更快的算子、精简网络结构),却没有先排除 retracing 的可能性——如果根因是 retracing,前者的优化收效甚微,因为大部分时间根本没花在"执行图"上,而是花在一遍遍"重新构建图"上;另一个坑是把 WARNING 日志级别调没了(比如全局设置只看 ERROR)之后,连这条现成的诊断线索都看不到,排查时缺了一个本该唾手可得的信号。

---

## 7. 设备不一致报错 —— TF 比 PyTorch "宽容"得多,这既是特性也是陷阱

**是什么:** 参与同一次运算的多个 tensor 分布在不同设备(比如一个在 `/GPU:0`、一个在 `/CPU:0`)。

**一句话:** **这里最值得记住的一点,恰恰是"TF 大多数时候根本不会报这个错"**——不像 PyTorch 那样对跨设备运算直接硬报错,TF 默认开启"软设备放置"(soft device placement),运算会被自动安排到合适的设备、必要时插入隐式拷贝,真正会报错的场景,反而需要一些"非默认"的额外条件同时满足。

**底层机制/为什么这样设计:**

**先看默认行为——CPU tensor 和 GPU tensor 直接相加,完全不报错:**

```python
import tensorflow as tf

assert tf.config.get_soft_device_placement() is True   # TF2默认开启软设备放置

with tf.device('/CPU:0'):
    c = tf.constant([1.0, 2.0, 3.0])
with tf.device('/GPU:0'):
    g = tf.constant([1.0, 2.0, 3.0])

assert c.device.endswith("CPU:0")
assert g.device.endswith("GPU:0")

mixed = c + g              # 一个CPU tensor、一个GPU tensor,直接相加
assert mixed.device.endswith("GPU:0")   # 不报错!TF自动决定了在哪个设备上执行、需要时插入拷贝
assert mixed.numpy().tolist() == [2.0, 4.0, 6.0]
```

这和 [11-debugging-and-common-errors.md](../torch-deep-dive/11-debugging-and-common-errors.md)(torch-deep-dive 同一主题)里 PyTorch 的行为形成直接反差——PyTorch 对这种情况是硬报错(`Expected all tensors to be on the same device...`),逼你显式处理;TF 的哲学不同,`tf.device()` 更多是"倾向性提示",不是"强制契约",算子调度器会在需要时自动安排数据搬运。

**真正会报错的场景,需要"用了某个设备上根本没有对应 kernel 实现的算子"+"关掉软放置兜底"同时满足:**

```python
import tensorflow as tf

# 软放置默认开启时:字符串op被强制放到GPU,底层没有GPU kernel,直接"悄悄"退回CPU执行,不报错
with tf.device('/GPU:0'):
    s = tf.constant(["hello", "world"])
    upper = tf.strings.upper(s)          # tf.strings系列大多只有CPU kernel实现
assert upper.device.endswith("CPU:0")     # 实际执行设备被静默改写成了CPU,而不是GPU
assert [x.decode() for x in upper.numpy()] == ["HELLO", "WORLD"]

# 关掉软放置兜底,同样的代码,这次才真正报错
tf.config.set_soft_device_placement(False)
try:
    with tf.device('/GPU:0'):
        s2 = tf.constant(["a", "b"])
        _ = tf.strings.upper(s2)
except tf.errors.InvalidArgumentError as e:
    print(str(e)[:300])
    assert "Could not satisfy device specification" in str(e)
    assert "enable_soft_placement=0" in str(e)
tf.config.set_soft_device_placement(True)   # 还原,避免影响同进程内后续代码
```

实测报错原文:
```
InvalidArgumentError: Could not satisfy device specification '/job:localhost/replica:0/task:0/device:GPU:0'.
enable_soft_placement=0. Supported device types [CPU]. All available devices
[/job:localhost/replica:0/task:0/device:GPU:0, /job:localhost/replica:0/task:0/device:CPU:0]. [Op:StringUpper] name:
```

**AI 研究/工程场景:** 从 PyTorch 转 TF 的工程师,常年养成"设备不一致必报错"的直觉,遇到 TF 里 CPU/GPU 混用却"莫名其妙没报错、结果却是对的"反而会疑惑;真正需要警惕的是**反过来**的情况——软放置的"自动兜底"意味着一段代码在开发机上(某些算子恰好都有 GPU kernel)全程跑在 GPU 上,换到另一台环境或者换了 TF 版本(某个算子的 GPU kernel 支持发生变化)之后,同一段代码可能悄悄退回 CPU 执行,性能大幅下降但不会有任何报错提示——这种"静默性能劣化"比"直接报错"更难排查,因为没有任何异常信息指向问题所在,只能通过实际测量执行时间或者显式检查 `.device` 才能发现。

**面试怎么问 + 追问链:**
- **Q:** "TensorFlow 里 CPU tensor 和 GPU tensor 直接做运算,会报错吗?"—— 期望不是想当然地照搬 PyTorch 的经验,而是能准确说出"默认不会,TF2 默认开启软设备放置,会自动决定执行设备并按需插入拷贝"。
- **追问 1(反常识,区分度高):** "那什么情况下才会真正报设备相关的错?"—— 期望说出"需要同时满足:用了目标设备上没有对应 kernel 实现的算子,并且显式关闭了软设备放置(`set_soft_device_placement(False)`)",能现场对比"软放置开/关"两种行为差异是加分项。
- **追问 2(工程向):** "如果 TF 大多数时候都不会因为设备不一致报错,这在实际工程里会带来什么隐患?"—— 期望能提出"某些算子缺少GPU kernel 时会被静默安排到CPU执行,这种'性能劣化但不报错'的情况比硬报错更难发现,需要主动检查 `.device` 或测量耗时才能确认,不能假设'没报错就等于全程都在GPU上跑'"。

**常见坑:** 把 PyTorch 的"设备不一致=硬报错"经验直接套用到 TF,遇到"没报错"就默认全程在 GPU 上执行,不去实际核实——真正的坑往往不是报错,而是性能异常却查不出原因;另外,`tf.config.set_soft_device_placement(False)` 这种全局状态修改,如果只是想临时验证某处的 kernel 支持情况,记得改完之后要还原回 `True`,否则会影响同进程内后续所有代码的容错行为。

---

## 8. Keras 2/3 版本冲突报错排查 —— 同一行 `tf.keras.optimizers.legacy.Adam`,两种环境下完全不同的命运

**是什么:** 00 篇环境声明里记录过本系列的主动选择——装 `tf_keras` 包并设置 `TF_USE_LEGACY_KERAS=1`,让 `tf.keras` 解析回经典 Keras 2 实现,而不是 TF 2.16 起默认的 Keras 3。本节要用真实触发的报错回答一个问题:**如果不这么做,具体会坏成什么样?**

**一句话:** 有两种"没配置对"的方式,坏的方式完全不同——**装了 `tf_keras` 但忘了设环境变量**,是"安静地换了一套实现"(不报错,但某些历史 API 悄悄失效);**设了环境变量但没装 `tf_keras` 包**,是"用到具体功能时才报错",而且报错文本本身其实把原因和解法都写清楚了,前提是你得先找到那条 WARNING。

**底层机制/为什么这样设计:**

**先看正确配置下的基准(control group,本系列其它章节的例子都是在这个状态下跑的):**

```python
import tensorflow as tf

m = tf.keras.Sequential([tf.keras.layers.Dense(4)])
assert type(m).__module__ == "tf_keras.src.engine.sequential"   # 落在tf_keras.*,不是keras.src.*

legacy_adam = tf.keras.optimizers.legacy.Adam
assert legacy_adam.__module__ == "tf_keras.src.optimizers.legacy.adam"   # 真正能用的legacy实现
```

**坏法 1——`TF_USE_LEGACY_KERAS` 没设置(或者是 `0`),`tf_keras` 包其实装了,但没被启用:**

```python
import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)   # 模拟"忘记设置这个环境变量"

import tensorflow as tf

m = tf.keras.Sequential([tf.keras.layers.Dense(4)])
assert type(m).__module__ == "keras.src.models.sequential"   # 落在keras.src.*,是Keras 3,不是tf_keras

# 访问tf.keras.optimizers.legacy.Adam这个属性本身不报错,
# 但拿到的不是真正的legacy实现,而是一个"占位符"类
placeholder = tf.keras.optimizers.legacy.Adam
assert placeholder.__name__ == "LegacyOptimizerWarning"   # 名字已经暗示了这是个警告占位符,不是真优化器

# 只有真正尝试实例化,才会暴露问题
try:
    placeholder(learning_rate=0.001)
except ImportError as e:
    print(e)
    assert "not supported in Keras 3" in str(e)
```

实测报错原文(实例化时才触发,访问属性本身不会报错):
```
ImportError: `keras.optimizers.legacy` is not supported in Keras 3. When using `tf.keras`, to continue
using a `tf.keras.optimizers.legacy` optimizer, you can install the `tf_keras` package (Keras 2) and
set the environment variable `TF_USE_LEGACY_KERAS=True` to configure TensorFlow to use `tf_keras`
when accessing `tf.keras`.
```

这条报错文本的质量很高——不但说清楚了原因(Keras 3 不支持这个命名空间),还直接给出了两步修复方案,和 00 篇环境声明里的配置要求一字不差。**这里最容易踩的坑是"只访问属性、不实例化"时完全不会报错**——如果代码里只是把 `tf.keras.optimizers.legacy.Adam` 这个类传来传去(比如作为配置项存在字典里),不到真正 `.compile()`/实例化那一刻,不会有任何异常提示配置错了。

**坏法 2——反过来,`TF_USE_LEGACY_KERAS=1` 设了,但 `tf_keras` 包压根没装:**

```python
import builtins
import os

# 用import hook模拟"忘记pip install tf_keras",不需要真的卸载包
_real_import = builtins.__import__
def _fake_import(name, *args, **kwargs):
    if name == "tf_keras" or name.startswith("tf_keras."):
        raise ImportError("No module named 'tf_keras'")
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _fake_import

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
print("tf.keras:", tf.keras)   # import tensorflow本身不报错

try:
    tf.keras.Sequential([tf.keras.layers.Dense(4)])   # 真正用到tf.keras的那一刻才报错
except ImportError as e:
    print(e)
    assert str(e) == "Keras cannot be imported. Check that it is installed."
```

实测:`import tensorflow` 这一步本身完全不报错,`tf.keras` 这个属性此时是一个 `KerasLazyLoader` 占位对象;真正报错发生在第一次**使用** `tf.keras.Sequential(...)` 这类功能时,异常文本很笼统(`Keras cannot be imported. Check that it is installed.`),单看这一行不容易反应过来问题出在"`TF_USE_LEGACY_KERAS` 和 `tf_keras` 包没配套"——但 TF 同时还会打一条更明确的 WARNING:
```
WARNING:tensorflow:Your environment has TF_USE_LEGACY_KERAS set to True, but you do not have the
tf_keras package installed. You must install it in order to use the legacy tf.keras.
Install it via: `pip install tf_keras`
```

**这是本节要强调的调试习惯:异常本身(`ImportError`)措辞很笼统,真正有用的诊断信息在一条容易被忽略的 WARNING 日志里,不是异常文本本身——遇到含糊的 Keras 相关 `ImportError`,养成往上翻日志(而不是只看最后一行 Traceback)的习惯。**

**AI 研究/工程场景:** 团队里 A 同学在自己电脑上装了 `tf_keras` 并配好环境变量,写的代码用了 `tf.keras.optimizers.legacy.Adam` 来兼容一份历史 checkpoint 的优化器状态;B 同学 clone 代码后只 `pip install tensorflow`,没注意 README 里那句环境变量配置,`import` 阶段一切正常,代码能跑,直到真正训练那一步命中 `LegacyOptimizerWarning` 被实例化才第一次报错——如果 B 同学只看最后一行 `ImportError` 去搜索,不一定能第一时间联想到"原来是 Keras 3/2 的兼容层问题",这也是为什么本系列在 00 篇就把这条环境依赖写得很显眼。

**面试怎么问 + 追问链:**
- **Q:** "`TF_USE_LEGACY_KERAS` 这个环境变量是干什么的,为什么需要它?"—— 期望能说出"TF 2.16 起 `tf.keras` 默认指向 Keras 3(多后端架构),和历史上内嵌在 TF 里的 Keras 2 语义/内部结构不同,这个环境变量配合装好的 `tf_keras` 包,可以让 `tf.keras` 继续解析回经典实现"。
- **追问 1(区分度高):** "如果只设了环境变量、没装 `tf_keras` 包,或者反过来只装了包没设环境变量,分别会发生什么?"—— 期望能分别说出两种不同的失败模式:前者是使用时报 `ImportError`(但配套 WARNING 说得很清楚);后者是完全不报错,但静默换成了 Keras 3 的实现,某些历史 API(比如 `optimizers.legacy` 命名空间)会变成访问不报错、实例化才报错的"占位符"。
- **追问 2(工程向):** "这种'不报错但行为不对'的配置问题,怎么在 CI 或者代码审查阶段提前发现,而不是等到线上才暴露?"—— 期望能提出"在环境搭建脚本/CI 里加一个显式的断言步骤,检查 `type(model).__module__` 之类的指纹是否落在预期的实现上",把隐式的环境假设变成显式的可验证检查。

**常见坑:** 只排查"报错文本说了什么",忽略了报错文本旁边、格式不那么醒目的 WARNING 日志——本节两个"坏法"的最终 `ImportError` 文本都不算特别友好,但配套的 WARNING 信息量大得多;另一个坑是想当然地认为"`import tensorflow` 没报错就说明 Keras 配置是对的",实际上 Keras 2/3 的分裂问题几乎都是"导入阶段沉默、使用阶段才暴露",不能只靠能不能 `import` 来判断环境是否配置正确。

---

## 9. 梯度为 None 排查 —— 没有异常,只有沉默(和几条容易被忽略的 WARNING)

**是什么:** `tape.gradient(target, sources)` 对 `sources` 里某个变量返回 `None`——表示 TF 认为 `target` 的计算过程根本没有用到这个变量,梯度自然无从谈起。

**一句话:** 这是本篇和 [02-gradienttape-internals.md](02-gradienttape-internals.md)、以及 torch-deep-dive 对应章节反差最大的一条——**PyTorch 遇到类似情况(比如根本没有 `grad_fn`)会直接抛 `RuntimeError`,TF 的 `tape.gradient()` 默认只是安静地在返回列表里放一个 `None`,不抛任何异常**;真正的报错(如果有的话)往往要等到你把这个 `None` 传给 optimizer 的 `apply_gradients()` 才可能出现,而且不一定是报错,更多时候只是一条 WARNING。

**底层机制/为什么这样设计:**

`GradientTape` 默认(`watch_accessed_variables=True`)只会自动记录**在 `with` 代码块内部、被实际读取过的、`trainable=True` 的 `tf.Variable`**——这条规则里的每一个限定词,反过来看都是一种"为什么会拿到 None"的原因:

```python
import tensorflow as tf

# 原因1:变量在tape作用域内压根没被用到(target的计算路径没有经过它)
v = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = tf.constant(5.0) * 2   # 全程没有读取v
grad = tape.gradient(z, v)
assert grad is None             # 不报错,安静地给你一个None

# 原因2:watch_accessed_variables=False时,trainable变量也不会被自动watch,
# 必须显式tape.watch()
v2 = tf.Variable(3.0)
with tf.GradientTape(watch_accessed_variables=False) as tape2:
    loss2 = v2 * v2               # 虽然用到了v2,但因为没有watch,tape没有记录这次读取
grad2 = tape2.gradient(loss2, v2)
assert grad2 is None

v2b = tf.Variable(3.0)
with tf.GradientTape(watch_accessed_variables=False) as tape2b:
    tape2b.watch(v2b)             # 显式watch之后,同样的计算就能正常求导
    loss2b = v2b * v2b
grad2b = tape2b.gradient(loss2b, v2b)
assert grad2b.numpy() == 6.0     # d(v^2)/dv at v=3 == 6,补上watch之后恢复正常
```

**原因3(最隐蔽,容易和"忘记 with"搞混):运算发生在 `with` 代码块退出之后,是常见的缩进/代码结构错误:**

```python
import tensorflow as tf

v6 = tf.Variable(2.0)
with tf.GradientTape() as tape6:
    pass                          # tape作用域内什么都没做
y6 = v6 * 3                        # 这一行在tape的with块外面执行(缩进错误的典型后果)
grad6 = tape6.gradient(y6, v6)
assert grad6 is None                # 同样安静地返回None,没有任何异常提示"你的计算在作用域外"
```

**可运行例子(下游会怎样:`apply_gradients` 对 `None` 梯度的两种不同反应):**

如果**全部**梯度都是 `None`,`apply_gradients` 才会真正报错(因为这时候它确实没有任何可更新的东西):

```python
import tensorflow as tf

v = tf.Variable(3.0)
opt = tf.keras.optimizers.SGD(0.1)
try:
    opt.apply_gradients([(None, v)])
except ValueError as e:
    print(e)
    assert "No gradients provided for any variable" in str(e)
```

实测报错原文:
```
ValueError: No gradients provided for any variable: (['Variable:0'],). Provided `grads_and_vars` is
((None, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>),).
```

但如果梯度列表里**只有部分**是 `None`(更常见的真实场景——多个变量,其中某几个和 loss 无关),`apply_gradients` **不会报错**,只是跳过那些 `None` 对应的变量,同时打一条 WARNING:

```python
import tensorflow as tf

v4 = tf.Variable(1.0, name="v4")
v5 = tf.Variable(2.0, name="v5")
opt = tf.keras.optimizers.SGD(0.1)

with tf.GradientTape() as tape4:
    loss4 = v4 * v4   # v5全程没被用到

grads = tape4.gradient(loss4, [v4, v5])
assert grads[0] is not None and grads[1] is None   # v4有梯度,v5是None

opt.apply_gradients(zip(grads, [v4, v5]))            # 不报错,只是跳过v5
assert v5.numpy() == 2.0                              # v5的值确实没被更新,静默"符合预期"
```

实测:`apply_gradients` 执行期间打印(不中断执行):
```
WARNING:tensorflow:Gradients do not exist for variables ['v5:0'] when minimizing the loss. If you're
using `model.compile()`, did you forget to provide a `loss` argument?
```

**这条 WARNING 用变量名指出了具体是哪个变量没有梯度**(这里因为显式传了 `name="v5"` 才能一眼看出是谁;如果变量没有显式命名,TF 会用自动生成的名字,排查时不一定能立刻对应到代码里的哪个变量,这也是"给关键变量显式命名"这个习惯在调试时的实际价值)。

**AI 研究/工程场景:** 多任务学习模型里,某几个 batch 因为某个任务的数据缺失,对应那部分 loss 项没有被计算,导致那部分专属参数在这几个 batch 里天然拿不到梯度——如果这是预期行为(该任务这一步确实不该更新),`apply_gradients` 的静默跳过刚好是想要的效果;但如果这不是预期行为(比如是数据管道的 bug 导致某个任务的分支被意外跳过了),同样的"静默跳过"就会让这个 bug 长期不被发现,只有认真盯 WARNING 日志或者定期检查各部分参数是否真的在更新,才能及时发现。

**面试怎么问 + 追问链:**
- **Q:** "`tape.gradient()` 对某个变量返回了 `None`,你能想到几种可能的原因?"—— 期望不止说出一种,比如"变量根本没被用到"、"`watch_accessed_variables=False` 但没显式 `watch()`"、"计算发生在 `with` 块外面"。
- **追问 1(和 PyTorch 对比,考察是否真的理解设计差异):** "PyTorch 里类似情况经常直接抛异常,为什么 TF 的 `tape.gradient()` 更倾向于安静地返回 `None`?"—— 期望能提出一种合理解释,比如"TF 的 `unconnected_gradients` 参数设计本身就允许把'无关联'当成一种合法结果处理(默认 `NONE`,也可以选择 `ZERO`),这暗示 TF 认为'某个变量在某次计算里用不上'是正常场景之一(比如多任务/条件分支模型),而不是一定意味着代码有 bug"。
- **追问 2(工程向,容易漏答):** "如果梯度列表里只有一部分是 `None`,`apply_gradients` 会报错吗?"—— 期望能准确说出"不会报错,只是跳过 `None` 对应的变量并打一条 WARNING,只有全部梯度都是 `None` 时才会真正抛 `ValueError`",这个区分点很容易被只测过"全 None"或者只测过"部分 None"其中一种场景的人漏答。

**常见坑:** 看到 `None` 就本能地在最外层给相关变量加 `trainable=True` 或者到处补 `tape.watch()`,却没有先分清楚是上面三种原因里的哪一种——比如原因 3(缩进导致计算跑到 `with` 块外面)靠加 `watch()` 根本没用,治标不治本;另一个坑是只关注 `ValueError`("No gradients provided for any variable"),却忽略了"部分 None、只有 WARNING、训练看起来完全正常继续跑"这种更隐蔽的情况——某个本该更新的参数,可能已经因为这个原因静默地再也没被更新过,不主动检查就发现不了。

---

## 小结:这一批 9 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `tf.function` 追踪失败 | AutoGraph 能转换的 Python 子集有明确边界(`if` 条件必须标量、循环体状态必须走 loop-carried 变量/`TensorArray`,不能用 Python list 旁路),`for i in range(tensor)` 本身反而是被特殊照顾的写法 |
| 2 | shape 不匹配(None维度) | `None` 维度只在符号构建阶段被视为"兼容",真正的相等性约束推迟到实际数据流过才检查,模型定义/前几次调用没报错不代表形状逻辑是对的 |
| 3 | eager 模式关闭 | `tf.function` 追踪期间 `executing_eagerly()` 为 `False` 是正常现象;`disable_eager_execution()` 是全局且不可逆的,治本方法是删掉那行调用而不是想办法重新 enable |
| 4 | GPU未识别 + OOM | GPU未识别是"不报错但列表为空",需要分层排查(驱动→库加载路径);OOM报错信息比PyTorch简略,需要主动查 `get_memory_info` |
| 5 | NaN/Inf定位 | 不开 `enable_check_numerics()` 时 NaN 静默传播,开了之后当场精确定位到产出异常值的具体 op 和输入,eager/图模式都覆盖 |
| 6 | retracing陷阱 | TF会主动打WARNING提示"N次调用N次retrace";自己验证用函数体内计数器;根因常是把Python原生值(而非Tensor)直接传给tf.function |
| 7 | 设备不一致 | 和PyTorch的核心差异:TF默认软设备放置,大多数跨设备运算不报错而是自动插入拷贝;真正报错需要"无对应kernel的算子"+"关闭软放置"同时满足 |
| 8 | Keras 2/3冲突 | 两种"没配对"的坏法:少环境变量→静默换成Keras 3(占位符类实例化时才报错);少tf_keras包→使用时ImportError(但配套WARNING说得很清楚) |
| 9 | 梯度为None | 三种成因(变量未参与计算、`watch_accessed_variables=False` 未显式watch、计算在tape作用域外);下游 `apply_gradients` 对"全None"报错、对"部分None"只警告不报错 |

---

*更新:2026-07-09*
