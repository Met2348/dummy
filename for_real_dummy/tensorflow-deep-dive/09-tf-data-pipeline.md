# 09 · tf.data 输入管道机制(TF 独有子系统)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批和前面几批的性质不太一样:tf.data 在 PyTorch 里没有直接对应物。PyTorch 的 `DataLoader` 骨子里是一个**运行时迭代器**——`for batch in loader` 每一步都是 Python 主进程在实际调度多进程 worker、从 `multiprocessing.Queue` 里取数据,迭代器本身不是一个能被整体分析、整体优化的"对象"。tf.data 的 `Dataset` 完全不同:你写下 `.map().batch().shuffle()` 这一串链式调用的那一刻,构造出来的是**一整张可以被内省、被整体优化的管道图**——这和 03 类 `tf.function` 把 Python 代码 trace 成图,是同一种设计哲学在"数据加载"这一层的体现。这个差异是本篇的核心看点,第 7 点会正面展开。

**本篇统一结构(和前几批一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(真实在 WSL2 `~/tf-venv` 里跑过,能内省的地方现场打印内部状态)
6. **面试怎么问 + 追问链**
7. 常见坑

**关于验证环境和"AI 研究/工程场景"段落的诚实声明(简短重复,完整版见 [00-roadmap.md](00-roadmap.md) 第 0 节):** 本文所有代码已在 WSL2 + `~/tf-venv`(TensorFlow 2.21.0,GPU 直通已验证)下用 `_verify_md.py` 逐块独立跑通验证,不是纸面推导。"AI 研究/工程场景"各条是根据真实训练/部署场景重构的例子,不是仓库引用——**唯独第 8 点是例外**:那一点在仓库 `learning/` 目录下真的挖到了两处非第三方仓库的真实 PyTorch 数据管道代码,会明确标注文件路径和行号,不属于这条豁免声明的范围。

---

## 1. `Dataset.from_tensor_slices` —— "切片"不是"复制"

**是什么:**
```text
tf.data.Dataset.from_tensor_slices(tensors)
```
把一个(或一组结构化的)tensor-like 输入,沿着**第 0 维**切开,构造一个 `Dataset`——原输入第 0 维有多长,`Dataset` 就有多少个元素,每个元素是原输入去掉第 0 维之后剩下的那一"片"。

**一句话:** "slices" 这个词要按字面理解——沿 axis 0 切片,不是"把数据复制 N 份"的意思;但构造这一步本身,TF 确实会把输入数据完整拷贝进自己管理的存储里(不是像 `torch.from_numpy()` 那样和原 numpy 数组共享同一块内存),"切片"和"构造时拷贝一次"是两件不冲突的事,分开理解才不会混。

**底层机制/为什么这样设计:**

先看官方 docstring 原文(已验证,本机 TF 2.21.0 安装包里直接读出来的):"Creates a `Dataset` whose elements are slices of the given tensors. The given tensors are sliced along their first dimension. This operation preserves the structure of the input tensors, removing the first dimension of each tensor and using it as the dataset dimension."

这段话精确定义了"slices"的含义:输入一个形状 `(4, 3)` 的 tensor,`from_tensor_slices` 产出的 `Dataset` 有 4 个元素,每个元素形状 `(3,)`;而 `Dataset.from_tensors`(不带 s)完全不切片,直接把整个 `(4, 3)` tensor 当成**唯一一个**元素。这是新手最容易搞反的一对 API——一个看 cardinality(元素个数)和 element_spec(单个元素的 shape)就能立刻分辨。

"复制"是另一层事实:`from_tensor_slices` 拿到 numpy 数组之后,会在构造这一刻把数值拷贝进 TF 自己管理的 tensor 存储(`EagerTensor` 有自己独立的 C++ 侧内存),不是持有一个指向原 numpy buffer 的引用——这一点和 PyTorch 的 `torch.from_numpy()`(默认共享内存,两边互相看得见修改)是相反的设计。验证方式很直接:构造完 `Dataset` 之后修改原始 numpy 数组,`Dataset` 迭代出来的值完全不受影响。

这个"构造时整体拷贝一次"的事实,带来一条官方文档明确写出来的告诫(已验证原文):"if `tensors` contains a NumPy array, and eager execution is not enabled, the values will be embedded in the graph as one or more `tf.constant` operations. For large datasets (> 1 GB), this can waste memory and run into byte limits of graph serialization." 换句话说,把一个超大 numpy 数组整个塞给 `from_tensor_slices`,eager 模式下问题不大(顶多多占一份内存),但一旦这个构造过程发生在**图模式**下(比如被包在 `tf.function` 里,或者导出 SavedModel 触发 tracing),这份数据会被当成常量整个嵌进 `GraphDef`,大数据集容易撞上 protobuf 的序列化字节上限。真正的大数据集应该用 `TFRecord` 或流式的 `from_generator`(第 8 点会用到)方案,`from_tensor_slices` 的定位始终是"能整个装进内存的中小数据集"。

**AI 研究/工程场景:** 能完整放进内存的中小规模数据(比如小型图像数据集读成 numpy 数组、tokenize 之后的小型语料、强化学习里一小批 rollout 样本)直接 `from_tensor_slices` 是最快的起步方式,不需要先落盘成 TFRecord;调试新模型结构、写单元测试时,几十个样本的 `from_tensor_slices` 比搭一整套文件读取管道省事得多。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

# 复制语义:构造后修改原始数组,Dataset 里的值不受影响
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ds = tf.data.Dataset.from_tensor_slices(arr)
arr[:] = -1.0
out = list(ds.as_numpy_iterator())
assert out == [1.0, 2.0, 3.0, 4.0]      # 没有跟着变,证明构造时已经拷贝了一份
assert list(arr) == [-1.0, -1.0, -1.0, -1.0]

# 切片语义:from_tensor_slices 沿 axis 0 切成 N 个元素;from_tensors 整体是 1 个元素
arr2 = np.arange(12).reshape(4, 3)
ds_slices = tf.data.Dataset.from_tensor_slices(arr2)
ds_whole = tf.data.Dataset.from_tensors(arr2)
assert ds_slices.cardinality().numpy() == 4
assert ds_slices.element_spec.shape == (3,)
assert ds_whole.cardinality().numpy() == 1
assert ds_whole.element_spec.shape == (4, 3)

# 结构保持:(features, labels) 元组按第0维一一配对,不需要手写 zip
feats = np.arange(10).reshape(5, 2)
labels = np.array([0, 1, 0, 1, 0])
ds_tuple = tf.data.Dataset.from_tensor_slices((feats, labels))
first_x, first_y = next(iter(ds_tuple))
assert first_x.numpy().tolist() == [0, 1]
assert first_y.numpy() == 0

# 官方 docstring 里"大数组会被嵌进图"的告诫原文确实存在(已验证,不是转述)
doc = tf.data.Dataset.from_tensor_slices.__doc__
assert "embedded in the graph as one or more" in doc
assert "byte limits of graph serialization" in doc
```

**面试怎么问 + 追问链:**
- **Q:** "`from_tensor_slices` 和 `from_tensors` 有什么区别?" —— 期望能讲出"沿 axis 0 切片,变成 N 个元素" vs "整个输入就是 1 个元素",最好能现场用 cardinality/element_spec 举例说明。
- **追问 1:** "如果我传进去一个 10GB 的 numpy 数组,`from_tensor_slices` 会怎么样?" —— 期望知道数据会被拷贝进 TF 自己的存储,eager 模式下问题不大(多占一份内存),但如果构造过程发生在图模式/SavedModel 导出这类 tracing 场景,会被嵌成 `tf.constant` 撞上 graph 序列化的字节上限,应该换成 `TFRecord` 或 `from_generator`。
- **追问 2(容易问倒):** "`from_tensor_slices` 构造出的 `Dataset` 对象,在你没有开始迭代之前,占多少内存?是 N 份独立拷贝,还是 1 份?" —— 期望答"逻辑上是 1 份完整拷贝,每个元素是这份底层存储的惰性切片视图,不是构造时就物化出 N 份独立对象"——这是"切片"和"复制"这两层语义各自准确的边界。

**常见坑:**
- 元组/字典结构传参时,各分量第 0 维长度必须完全一致,不一致会在构造阶段直接报错(已现场触发,不是转述):
```python
import tensorflow as tf
import numpy as np

try:
    tf.data.Dataset.from_tensor_slices((np.zeros((5, 2)), np.zeros((7,))))
    assert False, "expected an error"
except ValueError as e:
    assert "Dimensions 5 and 7 are not compatible" in str(e)
```
- 反过来误以为大数组用 `from_tensor_slices` 完全没有内存代价——有,构造时就整体拷贝了一次;"之后迭代不会再重复拷贝出 N 份"和"根本没拷贝过"是两回事,不要混为一谈。

---

## 2. `map()` 与 `num_parallel_calls` —— map 函数会被自动 trace,和 `tf.function` 是同一套机制

**是什么:**
```text
dataset.map(map_func, num_parallel_calls=None, deterministic=None, name=None)
```
对 `Dataset` 的每个元素应用 `map_func`,返回一个新 `Dataset`;`num_parallel_calls` 控制这个变换允许并行执行的程度(整数,或 `tf.data.AUTOTUNE`,见第 5 点)。

**一句话:** `map_func` 不是被"每个元素调用一次的 Python 函数",而是像 `@tf.function` 一样只在构造 `.map()` 这一刻被 **trace 一次**——传进去的是符号张量(graph tensor),trace 出来的图之后被复用来处理数据集里的每一个真实元素,`map_func` 里的原生 Python `if`/`for` 也会像 `tf.function` 一样经过 AutoGraph 转换,这正是本系列在 03 类要重点讲的 tracing 机制在 tf.data 里的直接体现。

**底层机制/为什么这样设计:**

`Dataset.map()` 内部会把 `map_func` 包装成一个类似 `tf.function` 的可调用图函数(`StructuredFunctionWrapper`),构造 `.map()` 调用时,TF 用一个只携带 shape/dtype 信息、没有具体数值的**符号张量**去调用一次 `map_func`,把 Python 函数体里发生的所有 TF 算子调用录制成一张图;之后不管这个 `Dataset` 实际有多少个元素,处理每个元素时执行的都是**这张已经录制好的图**,不会再退回 Python 逐元素重新调用一次 `map_func` 本体。这和 `tf.function` "trace 一次、复用多次"完全是同一套底层机制——区别只在于 `tf.function` 是你显式装饰的,`.map()` 是 tf.data 在背后自动帮你做的,开发者往往意识不到这一层。

正因为是同一套 tracing 机制,`map_func` 里的原生 Python 控制流(`if`/`for`/`while`)也会经过 **AutoGraph** 转换,和 03 类要讲的 `@tf.function` 自动转换控制流是同一个组件——不需要手写 `tf.cond`,写"看起来像普通 Python"的判断逻辑,AutoGraph 会把它转换成图内可以按元素真实值走不同分支的算子。

`num_parallel_calls` 决定的是"trace 出来的这张图,允许同时对多少个元素并行执行",而不是"要不要并行调用 Python 函数本体"——因为 Python 函数本体已经在构造时被 trace 完并"消耗"成图了,运行期根本不会再有 Python 函数调用这一步,并行执行的是 C++ 层面的图算子,不受 Python GIL 限制,这也是为什么把 `num_parallel_calls` 调大对一个 I/O 密集的 map 函数收益如此明显。

**AI 研究/工程场景:** 训练视觉模型时 `map_func` 里做 JPEG 解码 + resize + 归一化,这是典型的 I/O/CPU 密集型工作,不设置 `num_parallel_calls`(默认顺序执行)会让 GPU 大部分时间在等数据;NLP 场景 `map_func` 里做 tokenize,同理。这也是为什么几乎所有官方教程里 `.map()` 后面都紧跟着一个 `num_parallel_calls=tf.data.AUTOTUNE`,而不是留空。

**可运行例子:**
```python
import tensorflow as tf

# map_func 只在构造 .map() 时被 Python 层调用一次(trace),
# 不管数据集实际有多少个元素,Python 函数体都不会被重新调用
call_count = {"n": 0}

def transform(x):
    call_count["n"] += 1
    return x * 2

ds = tf.data.Dataset.range(5).map(transform)
assert call_count["n"] == 1              # 构造完成,只 trace 了一次
result = list(ds.as_numpy_iterator())
assert call_count["n"] == 1              # 迭代完全部5个元素,依然只有1次 Python 调用
assert result == [0, 2, 4, 6, 8]

# trace 期间传进 map_func 的是符号张量,不是普通 Python 数值
def inspect_arg(x):
    assert type(x).__name__ == "SymbolicTensor"
    return x

_ = list(tf.data.Dataset.range(3).map(inspect_arg).as_numpy_iterator())

# 原生 Python if 在 map_func 里会被 AutoGraph 转换,和 @tf.function 走的是同一套机制,
# 结果完全一致——不是巧合。
# 一个真实的旁支坑:AutoGraph 的转换依赖 inspect.getsource() 读取函数的真实源码;
# 如果这段代码本身是用 exec()/`python -c` 这种"没有真实源文件"的方式跑起来的(本篇验证
# 脚本恰好就是这样跑每个代码块的),AutoGraph 会读不到源码,静默退化成不转换、直接调用
# 原函数,进而在 if 命中符号 tensor 时报错——这不是这个例子专属的怪癖,是 AutoGraph 一个
# 真实存在、容易被忽略的限制(正常写在 .py 脚本文件里跑没有这个问题)。这里用"写一个真实
# 临时 .py 文件再 import"绕开这个限制,让 AutoGraph 能读到源码:
import tempfile, os, importlib.util

src = '''
def with_python_if(x):
    if x > 2:
        return x * 100
    else:
        return x
'''
tmpdir = tempfile.mkdtemp(prefix="tf_autograph_demo_")
path = os.path.join(tmpdir, "mapfn_mod.py")
with open(path, "w") as f:
    f.write(src)
spec = importlib.util.spec_from_file_location("mapfn_mod", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mapped_result = list(tf.data.Dataset.range(5).map(mod.with_python_if).as_numpy_iterator())
assert mapped_result == [0, 1, 2, 300, 400]

tf_fn_version = tf.function(mod.with_python_if)
tf_fn_result = [int(tf_fn_version(tf.constant(i, dtype=tf.int64))) for i in range(5)]
assert tf_fn_result == [0, 1, 2, 300, 400]
assert mapped_result == tf_fn_result   # map()内部机制和tf.function产出完全一致的结果,不是巧合
```

第二段,`num_parallel_calls` 对一个模拟 I/O 延迟的 map 函数的真实吞吐影响:
```python
import tensorflow as tf
import time

def slow_fn(x):
    def _sleep(v):
        time.sleep(0.05)
        return v
    return tf.py_function(_sleep, [x], tf.int64)

N = 8
t0 = time.perf_counter()
list(tf.data.Dataset.range(N).map(slow_fn).as_numpy_iterator())         # 不设置 num_parallel_calls -> 顺序执行
t_seq = time.perf_counter() - t0

t0 = time.perf_counter()
list(tf.data.Dataset.range(N).map(slow_fn, num_parallel_calls=N).as_numpy_iterator())
t_par = time.perf_counter() - t0

print(f"sequential: {t_seq:.3f}s | num_parallel_calls={N}: {t_par:.3f}s")
# 实测(WSL2 + RTX 3080Ti Laptop):sequential ≈0.47s(≈8*0.05s,几乎是纯串行下界),
# num_parallel_calls=8 ≈0.07s(接近单次sleep的时间,证明8个元素真的在并行处理)
assert t_par < t_seq / 3
```

**面试怎么问 + 追问链:**
- **Q:** "`.map()` 里的函数,是数据集有多少个元素就被 Python 调用多少次吗?" —— 期望答"不是,只在构造 `.map()` 时被 trace 一次,之后执行的是 trace 出来的图,和 `tf.function` 的 tracing 机制是同一回事"。
- **追问 1(区分度高):** "那我在 `map_func` 里写一个 `if tensor_value > 0:` 这种原生 Python 判断,不会报错吗?tracing 阶段这个 tensor 又没有具体值。" —— 期望答"会被 AutoGraph 自动转换成图内条件算子(概念上类似 `tf.cond`),这正是 `.map()` 和 `@tf.function` 共享的机制,不需要手写 `tf.cond`"。
- **追问 2:** "`num_parallel_calls` 并行的是什么?Python 函数本体的多次调用吗?" —— 期望能纠正这个常见误解:Python 函数本体只调用一次用来 trace,`num_parallel_calls` 并行的是**trace 出来的图**在处理不同元素时的执行,是 C++ 层面的并发,不受 GIL 限制。

**常见坑:**
- 以为 `.map()` 是逐元素调用 Python 函数,于是在 `map_func` 里写 `print()` 想每个元素都打印一次调试信息,结果发现无论数据集多大,`print` 只在构造阶段执行了一次(Python 层面的副作用只在 trace 时发生);想看到每个元素的信息应该用 `tf.print`(呼应 03 类会展开的 `tf.function` 内 `print` vs `tf.print` 陷阱)。
- 把 `num_parallel_calls` 当成"开几个 Python 线程/进程"来理解——它控制的是 trace 出来的图算子的并行执行程度,和 Python 多线程/多进程模型不是一回事,这也是它能完全绕开 GIL 的原因。

---

## 3. `batch` / `shuffle` / `prefetch` 的顺序 —— 顺序不同,行为真的不同(实测)

**是什么:**
```text
dataset.shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)
dataset.batch(batch_size, drop_remainder=False)
dataset.prefetch(buffer_size)
```
三个最常用的链式调用;`shuffle` 从一个大小为 `buffer_size` 的缓冲区里随机取元素,`batch` 把连续若干个元素打包成一个 batch,`prefetch` 让"准备下一批数据"和"消费当前数据"在后台线程里重叠执行。

**一句话:** 这三者的**相对顺序**会实实在在改变管道的行为,不是文字上"最佳实践"层面的偏好问题——`shuffle` 放在 `batch` 前后,决定的是"打乱粒度是单个样本还是整个 batch";`prefetch` 放在管道的什么位置,决定的是"它到底在帮谁做后台预取"。

**底层机制/为什么这样设计:**

`shuffle(buffer_size)` 的机制是:维护一个大小为 `buffer_size` 的缓冲区,每次要产出一个元素时,从缓冲区里随机挑一个吐出去,同时从上游再补一个新元素进来——它操作的对象永远是"当前这个 `Dataset` 的元素",不关心这个元素在下游会不会被后续算子打包。所以:
- `shuffle` 在 `batch` **之前**:此时 `Dataset` 的元素是单个样本,`shuffle` 打乱的是单个样本的顺序,之后 `batch` 只是机械地把打乱后的样本流每 N 个打包一次——每个 batch 内部的样本组合是随机的。
- `shuffle` 在 `batch` **之后**:此时 `Dataset` 的元素已经是一个个打包好的 batch,`shuffle` 打乱的是"batch 出场的顺序",batch **内部**的样本从始至终没有被打乱过,还是原始数据流里连续排列的那几个样本。

`prefetch(buffer_size)` 的机制是:在当前节点后面开一个后台线程,不断地从**它的上游**拉取元素填满一个缓冲区,让消费者(下一个算子,或者你的训练循环)永远有现成的元素可取,不用等上游现算现取。关键在于"它的上游"这四个字——`prefetch` 只能让**排在它前面**的那部分工作和"消费"重叠起来,排在它**后面**的算子,该怎么同步执行还是怎么同步执行,不会被这一次 `prefetch` 覆盖到。

**AI 研究/工程场景:** 图像/NLP 训练里,`shuffle` 必须放在 `batch` 之前几乎是标配——如果 `shuffle` 放在 `batch` 之后,相当于每个 batch 永远由原始顺序里挨在一起的样本组成,如果原始数据本身是按类别/来源排过序的(很常见,比如按文件夹分类存放的图像数据集),会导致每个 batch 的样本分布严重不均匀,训练不稳定;`prefetch` 应该放在管道**最后**,确保它预取的是"最终喂给模型的那份数据",而不是某个中间阶段还没做完 `map`/`batch` 的半成品。

**可运行例子(第一部分:`shuffle`/`batch` 顺序对"打乱粒度"的真实影响,可复现的随机种子):**
```python
import tensorflow as tf

# shuffle 在 batch 之前:打乱粒度是单个样本,batch 内部不再是原始连续片段
ds_shuffle_then_batch = tf.data.Dataset.range(20).shuffle(buffer_size=20, seed=0).batch(4)
batches_a = [b.numpy().tolist() for b in ds_shuffle_then_batch]
consecutive_a = [all(b[i + 1] - b[i] == 1 for i in range(len(b) - 1)) for b in batches_a]
assert not any(consecutive_a)          # 没有任何一个 batch 内部是连续整数

# shuffle 在 batch 之后:打乱的只是 batch 出场顺序,batch 内部还是原始连续片段
ds_batch_then_shuffle = tf.data.Dataset.range(20).batch(4).shuffle(buffer_size=5, seed=0)
batches_b = [b.numpy().tolist() for b in ds_batch_then_shuffle]
consecutive_b = [all(b[i + 1] - b[i] == 1 for i in range(len(b) - 1)) for b in batches_b]
assert all(consecutive_b)              # 每一个 batch 内部都还是连续整数,只是 batch 顺序被打乱了

print("shuffle->batch:", batches_a)
print("batch->shuffle:", batches_b)
```
实测输出:`shuffle->batch` 得到的 5 个 batch 全部由不连续的数字随机组成(比如 `[14, 19, 15, 8]`);`batch->shuffle` 得到的 5 个 batch 每一个内部都是 4 个连续整数(比如 `[16, 17, 18, 19]`),只是这些 batch 出现的先后顺序被打乱了——这就是"顺序不一样,打乱粒度真的不一样"的直接证据,不是文字断言。

**可运行例子(第二部分:`prefetch` 位置对"是否真的重叠"的真实影响,关掉 autotune 看"教科书机制"本身):**
```python
import tensorflow as tf
import time

def make_slow_map():
    def _sleep(v):
        time.sleep(0.05)
        return v
    def slow_fn(x):
        return tf.py_function(_sleep, [x], tf.int64)
    return slow_fn

def no_autotune_opts():
    opts = tf.data.Options()
    opts.autotune.enabled = False   # 先关掉自动优化,看清 prefetch 本身的机制
    return opts

N_ITEMS, CONSUME_SLEEP = 6, 0.05

def run_consumer(ds):
    t0 = time.perf_counter()
    for _ in ds.as_numpy_iterator():
        time.sleep(CONSUME_SLEEP)   # 模拟"训练一步"要消费这条数据
    return time.perf_counter() - t0

# 好顺序:慢的 map 在前,prefetch 在后 -> 后台线程能提前把"下一条"的 map 算好
ds_good = tf.data.Dataset.range(N_ITEMS).map(make_slow_map()).prefetch(2).with_options(no_autotune_opts())
t_good = run_consumer(ds_good)

# 坏顺序:prefetch 在前,慢的 map 在后 -> prefetch 缓冲的是"未经 map 的原始整数"(几乎不耗时),
# 真正耗时的 map 计算被留在了 prefetch 后面,还是在消费者线程里同步发生,完全没有被重叠
ds_bad = tf.data.Dataset.range(N_ITEMS).prefetch(2).map(make_slow_map()).with_options(no_autotune_opts())
t_bad = run_consumer(ds_bad)

print(f"map->prefetch(好顺序): {t_good:.3f}s | prefetch->map(坏顺序): {t_bad:.3f}s")
assert t_good < t_bad * 0.85                                  # 好顺序明显更快
assert t_bad > N_ITEMS * CONSUME_SLEEP * 0.9                   # 坏顺序几乎退化成完全串行(接近 N*(map+consume) 下界)
```
实测(关闭 autotune):好顺序约 0.39s,坏顺序约 0.62s,后者已经非常接近"完全不重叠"的朴素串行估计 `6*(0.05+0.05)=0.60s`——证明 `prefetch` 放在慢速 `map` 前面时,那次 `prefetch` 基本没有起到任何重叠效果,是名副其实的"位置不对,等于白写"。

**一个诚实的额外发现(不在教科书里,是这次验证时意外测出来的,和第 5 点 `AUTOTUNE` 直接相关):** 上面两段代码里我特意手动把 `opts.autotune.enabled` 设成 `False`,是因为如果**不关**(也就是所有新建 `Dataset` 默认的状态),好顺序和坏顺序的实测耗时几乎没有差别,都远快于朴素串行估计:
```python
import tensorflow as tf
import time

def make_slow_map():
    def _sleep(v):
        time.sleep(0.05)
        return v
    def slow_fn(x):
        return tf.py_function(_sleep, [x], tf.int64)
    return slow_fn

N_ITEMS, CONSUME_SLEEP = 6, 0.05

def run_consumer(ds):
    t0 = time.perf_counter()
    for _ in ds.as_numpy_iterator():
        time.sleep(CONSUME_SLEEP)
    return time.perf_counter() - t0

# 不做任何 with_options 处理 —— autotune 保持默认开启状态
ds_bad_default = tf.data.Dataset.range(N_ITEMS).prefetch(2).map(make_slow_map())
t_bad_default = run_consumer(ds_bad_default)
ds_none_default = tf.data.Dataset.range(N_ITEMS).map(make_slow_map())
t_none_default = run_consumer(ds_none_default)

naive_serial = N_ITEMS * (0.05 + CONSUME_SLEEP)
print(f"[autotune默认开启] prefetch放在map前面: {t_bad_default:.3f}s | 完全不写prefetch: {t_none_default:.3f}s")
print(f"朴素串行估计: {naive_serial:.3f}s")
assert t_bad_default < naive_serial * 0.85
assert t_none_default < naive_serial * 0.85
```
实测两者都在 0.36s 左右,明显快于 0.60s 的朴素串行估计——原因是 `tf.data` 的 autotune 优化器(默认开启)会在图优化阶段**自动往管道里插入/调整 `prefetch` 缓冲区**,不完全依赖开发者手写的 `.prefetch()` 调用出现在哪个位置。这不代表"顺序不重要"这条机制性结论是错的(上面关闭 autotune 之后的实测已经证明机制本身千真万确),而是提醒一件更实际的事:**现代 TF 默认配置下,这个经典坑已经被运行时自动优化悄悄兜底了一部分**,但理解底层机制依然重要——面试会考机制本身,复杂管道、旧版本 TF、或者显式关闭了 autotune 的场景下,手动摆对顺序依然是必须的。

**面试怎么问 + 追问链:**
- **Q:** "`shuffle` 应该放在 `batch` 前面还是后面?为什么?" —— 期望答"前面,这样打乱粒度是单个样本;放后面只是打乱 batch 出场顺序,batch 内部还是原始顺序",最好能提到"如果原始数据按类别排序存放,顺序放错会导致 batch 内类别分布不均匀"这类具体后果。
- **追问 1:** "`prefetch` 为什么建议放在链条最后?" —— 期望答"`prefetch` 只能重叠它上游的工作,放在中间会导致后面的算子依然同步阻塞,起不到应有的效果"。
- **追问 2(区分度很高):** "如果我现在直接用默认配置跑一遍,`prefetch` 放前面放后面居然测不出明显差异,是不是说明这个说法过时了?" —— 期望候选人不会直接被"实测数字看起来一样"带偏,能说出"TF 默认开启的 autotune 优化器会自动插入/调整 prefetch,掩盖了手动摆放顺序的影响;关掉 autotune 或者面对更复杂、优化器覆盖不到的管道时,底层机制和顺序问题依然成立",体现"知道默认行为,但不会把默认行为等同于机制本身不存在"的判断力。

**常见坑:**
- 把慢速 `map` 放在 `prefetch` 之后,以为写了 `prefetch` 就一定有性能提升,结果因为顺序反了完全没有效果(上面已实测复现)。
- `.batch()` 要求同一批次内所有元素形状完全一致,如果上游元素长度不定(常见于变长序列),直接 `batch()` 会报错,需要改用 `padded_batch`(第 8 点会展开),下面是真实触发的报错文本:
```python
import tensorflow as tf

ds_ragged = tf.data.Dataset.from_generator(
    lambda: iter([[1, 2], [1, 2, 3]]),
    output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
)
try:
    list(ds_ragged.batch(2).as_numpy_iterator())
    assert False, "expected an error"
except tf.errors.InvalidArgumentError as e:
    assert "Cannot batch tensors with different shapes in component 0" in str(e)
```
- 以为 `shuffle(buffer_size)` 是把**整个数据集**打乱——如果 `buffer_size` 小于数据集总大小,只是在一个滑动窗口内做局部打乱,不是全局均匀打乱,数据集很大而 `buffer_size` 设置过小时,随机性会明显不足。

---

## 4. `cache()` —— 内存缓存 vs 磁盘缓存,以及"该缓存在 shuffle/augmentation 前面还是后面"

**是什么:**
```text
dataset.cache(filename="")
```
把这个 `Dataset` **第一次完整迭代**产出的所有元素缓存下来,后续再次迭代(比如下一个 epoch)时直接从缓存读取,不再重新执行 `cache()` 之前的所有变换。不传 `filename` 时缓存在内存里;传一个路径时缓存到磁盘上的文件。

**一句话:** `cache()` 缓存的是"它前面那部分管道,在第一次跑完之后产出的确定结果"——所以任何写在 `cache()` **之前**、你希望"每个 epoch 都不一样"的操作(典型的比如随机数据增强、shuffle),一旦被缓存,就会被永远冻结成第一次跑出来的那个值,之后每个 epoch 读到的都是同一份,不会再变。

**底层机制/为什么这样设计:**

`cache()` 的实现很朴素:第一次迭代时,一边把上游元素正常吐出去,一边把这些元素原样另存一份(内存或磁盘);从第二次迭代开始,直接读这份存档,不再重新执行 `cache()` 之前的任何计算图节点。这个设计的价值在于——很多预处理步骤是**确定性**且**计算量大**的(比如从磁盘解码大图、复杂的特征工程),没必要每个 epoch 都重新算一遍;但正因为"存档"这个动作是"原样保存第一次跑出来的结果",任何**非确定性**的算子(`tf.random.*`、`shuffle`)如果出现在 `cache()` 之前,它们"随机"出来的那次结果也会被一起存档,后续 epoch 读到的不是"重新随机一次",而是"重放第一次的随机结果"——这是本节唯一但很容易踩的坑,而且**不会报错**,是那种更危险的"安静地算错"。

磁盘缓存(`cache(filename)`)会在给定路径生成两个文件:`{filename}.index` 和 `{filename}.data-00000-of-00001`——前者是索引,后者是实际数据,机制上类似 TF 的 checkpoint 文件命名习惯。选内存还是磁盘缓存,取决于处理后的数据集能不能整个装进内存:装得下,内存缓存最快;装不下但重复计算的开销很大(比如大规模图像 decode),磁盘缓存依然能省下"重新计算"这部分,只是要多付一次磁盘 I/O。

**AI 研究/工程场景:** 标准做法是把"确定性、开销大"的预处理(解码、resize、tokenize)放在 `cache()` **之前**,把"随机、开销小"的操作(shuffle、random crop/flip 这类数据增强)放在 `cache()` **之后**——这样每个 epoch 都能省下重复解码的开销,同时依然保留每个 epoch 不同的随机性。这条经验法则背后就是本节讲的机制,不是死记硬背的规则。

**可运行例子(内存 vs 磁盘缓存文件的真实产物):**
```python
import tensorflow as tf
import os, tempfile, shutil

ds_mem = tf.data.Dataset.range(5).cache()      # 不传文件名:纯内存缓存
_ = list(ds_mem.as_numpy_iterator())

tmpdir = tempfile.mkdtemp(prefix="tfcache_")
cache_path = os.path.join(tmpdir, "mycache")
ds_disk = tf.data.Dataset.range(5).cache(cache_path)   # 传文件名:磁盘缓存
_ = list(ds_disk.as_numpy_iterator())          # 必须完整迭代一遍,缓存才会被写完整

files = sorted(os.listdir(tmpdir))
assert files == ["mycache.data-00000-of-00001", "mycache.index"]
shutil.rmtree(tmpdir)
```

**可运行例子(随机增强/shuffle 放在 `cache()` 前后,行为真的不一样):**
```python
import tensorflow as tf

def rand_augment(x):
    return x + tf.random.uniform([], 0, 1000, dtype=tf.int64)

# 正确写法:随机增强放在 cache() 之后 -> 增强本身不会被缓存,每个 epoch 都重新随机
ds_after = tf.data.Dataset.range(3).cache().map(rand_augment)
epoch1_after = list(ds_after.as_numpy_iterator())
epoch2_after = list(ds_after.as_numpy_iterator())
assert epoch1_after != epoch2_after

# 常见坑:随机增强放在 cache() 之前 -> 第一次迭代产生的"随机"结果被整体缓存,
# 之后每个 epoch 读到的都是同一份,增强从此不再随机(不报错,是悄悄错误)
ds_before = tf.data.Dataset.range(3).map(rand_augment).cache()
epoch1_before = list(ds_before.as_numpy_iterator())
epoch2_before = list(ds_before.as_numpy_iterator())
assert epoch1_before == epoch2_before
```
```python
import tensorflow as tf

# 同样的道理对 shuffle 成立:shuffle 放在 cache() 之后 -> 每个 epoch 重新洗牌
ds_shuf_after = tf.data.Dataset.range(10).cache().shuffle(10)
e1 = list(ds_shuf_after.as_numpy_iterator())
e2 = list(ds_shuf_after.as_numpy_iterator())
assert e1 != e2

# shuffle 放在 cache() 之前 -> 第一次洗好的顺序被整体缓存,以后每个 epoch 都是同一个顺序
ds_shuf_before = tf.data.Dataset.range(10).shuffle(10).cache()
e1b = list(ds_shuf_before.as_numpy_iterator())
e2b = list(ds_shuf_before.as_numpy_iterator())
assert e1b == e2b
```

**面试怎么问 + 追问链:**
- **Q:** "`cache()` 应该放在 shuffle/数据增强前面还是后面?" —— 期望答"后面,不然随机性会被第一次的结果冻结住";能进一步区分"确定性、开销大的操作放前面(享受缓存收益),随机操作放后面(保留随机性)"是加分项。
- **追问 1:** "这个坑会报错提醒你吗?" —— 期望答"不会,这是最麻烦的一类坑——程序完全正常运行,只是训练时你以为每个 epoch 都在做新的随机增强,实际上一直在用同一份"。
- **追问 2:** "内存缓存和磁盘缓存怎么选?" —— 期望答"取决于处理后的数据集能不能完整放进内存;装得下选内存最快,装不下但预处理开销大依然值得用磁盘缓存省重复计算,只是要多付一次磁盘 I/O"。

**常见坑:**
- 随机数据增强/`shuffle` 写在 `cache()` 之前,导致"随机"从第二个 epoch 起失效(上面已实测复现,且不会报错)。
- 磁盘缓存文件路径复用:如果修改了 `cache()` **之前**的管道逻辑(比如换了一种预处理方式),但缓存路径没变,可能读到的是上一次运行遗留的旧缓存内容而不是新逻辑的结果——磁盘缓存文件不会因为上游逻辑变了就自动失效,换了预处理逻辑记得手动清理旧的缓存文件。
- 缓存了一个还没做完整个 epoch 的 `Dataset`(比如手动打断了第一次迭代):本机实测(TF 2.21.0)这种情况下重新完整迭代一次并不会报错、也不会用不完整的缓存,而是老老实实重新走一遍上游计算——行为上是安全的,但也意味着"打断第一次迭代"并不能让你白得一份缓存,预期这一点和实际观察到的行为要对齐。

---

## 5. `tf.data.AUTOTUNE` —— 把"并行度该开多大"交给运行时猜,而不是开发者猜

**是什么:**
```text
tf.data.AUTOTUNE   # 一个整数哨兵值,等于 -1
```
可以传给 `map(num_parallel_calls=...)`、`interleave(num_parallel_calls=...)`、`prefetch(buffer_size=...)` 等参数,告诉 tf.data 运行时:这个数字不用你(开发者)猜,由运行时根据当前 CPU 负载、各阶段实测耗时动态调整。

**一句话:** 手动给 `num_parallel_calls`/`prefetch` 缓冲区大小设一个固定数字,本质上是在猜一个"当前机器、当前负载下"的最优值——`AUTOTUNE` 把这个猜测工作交给 tf.data 运行时,让它在训练过程中持续实测各阶段耗时,动态调整并行度和缓冲区大小,而不是让你在代码里写死一个可能只适合当前这台机器、当前这一次运行的数字。

**底层机制/为什么这样设计:**

最优并行度不是一个静态常数——同一段 `map` 代码,在 8 核和 32 核的机器上最优 `num_parallel_calls` 显然不同;哪怕是同一台机器,训练过程中如果其他阶段(比如 GPU 计算)负载发生变化,输入管道能"合理"占用的 CPU 资源也会跟着变。手写死一个数字,本质上是把这个动态优化问题简化成了开发者一次性拍脑袋的静态决策。`AUTOTUNE`(值为 `-1`,和 `tf.data.experimental.AUTOTUNE` 是同一个值)是一个哨兵,告诉 tf.data 的 `autotune` 组件:这个位置的并行度/缓冲区大小由运行时自己的性能模型决定,它会在训练过程中持续采样各阶段的真实吞吐,用一个内部优化算法在"整个输入管道的总内存/CPU 预算"约束下重新分配各阶段的并行度,而不是各阶段各自为政地抢资源。

这个自动调整由 `tf.data.Options().autotune` 这一组配置项控制,常读到的字段有 `enabled`、`cpu_budget`、`ram_budget`、`autotune_algorithm`、`initial_parallelism`、`min_parallelism`——默认新建的 `Options()` 上这些字段读出来都是 `None`,含义是"不覆盖,交给 C++ 运行时的内部默认值",不是"关掉了"(第 3 点已经验证过,默认状态下 autotune 确实是生效的)。需要精细控制时,可以显式给 `ram_budget` 这类字段赋值,覆盖运行时的默认预算。

**AI 研究/工程场景:** 几乎所有官方教程和生产代码里,`num_parallel_calls`/`prefetch` 参数默认都直接写 `tf.data.AUTOTUNE`,不是偷懒,而是这个值本来就应该由运行时决定;唯一需要手动指定固定数字的场景通常是**调试/复现性要求高**的情况(比如要精确控制某个基准测试用了几个并行 worker),或者你比运行时更了解某个特定资源约束(比如显式限制 CPU 占用避免和同机器的其它进程抢核)。

**可运行例子:**
```python
import tensorflow as tf

assert tf.data.AUTOTUNE == -1
assert tf.data.AUTOTUNE == tf.data.experimental.AUTOTUNE

opts = tf.data.Options()
at = opts.autotune
for field in ("enabled", "cpu_budget", "ram_budget", "autotune_algorithm", "initial_parallelism", "min_parallelism"):
    assert hasattr(at, field)
# 新建 Options() 默认读出来是 None —— 意思是"交给C++运行时内部默认值",不是"关掉了"
assert at.enabled is None
assert at.cpu_budget is None
assert at.ram_budget is None

# 可以显式覆盖预算,比如给 tf.data 运行时设一个更保守的内存上限
opts2 = tf.data.Options()
opts2.autotune.ram_budget = 256 * 1024 * 1024
ds2 = tf.data.Dataset.range(5).with_options(opts2)
assert ds2.options().autotune.ram_budget == 256 * 1024 * 1024
```
```python
import tensorflow as tf
import time

def slow_map(x):
    def _sleep(v):
        time.sleep(0.03)
        return v
    return tf.py_function(_sleep, [x], tf.int64)

N = 10
t0 = time.perf_counter()
list(tf.data.Dataset.range(N).map(slow_map, num_parallel_calls=1).as_numpy_iterator())
t_fixed1 = time.perf_counter() - t0

t0 = time.perf_counter()
list(tf.data.Dataset.range(N).map(slow_map, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator())
t_auto = time.perf_counter() - t0

print(f"num_parallel_calls=1(手写死不并行): {t_fixed1:.3f}s | AUTOTUNE: {t_auto:.3f}s")
# 实测(WSL2 + RTX 3080Ti Laptop):固定写死1 ≈0.36s,AUTOTUNE ≈0.05s,
# 差距 7 倍以上 —— 不需要开发者猜"该开几路并行",运行时自己就找到了远好于1的并行度
assert t_auto < t_fixed1 * 0.5
```

**面试怎么问 + 追问链:**
- **Q:** "`tf.data.AUTOTUNE` 具体是怎么工作的?" —— 期望答出"它是一个哨兵值(-1),告诉运行时这个位置的并行度/缓冲区大小不由开发者写死,而是运行时持续采样各阶段实测吞吐、在整体资源预算下动态分配"。
- **追问 1:** "既然有 `AUTOTUNE`,为什么还需要手动指定固定的 `num_parallel_calls`?" —— 期望答出"调试、需要严格复现性、或者开发者对资源约束有运行时不知道的额外信息(比如要给同机器上的其它任务预留 CPU)"这类场景。
- **追问 2(呼应第 3 点):** "第 3 点里 `prefetch` 位置放错、但 autotune 默认开启时几乎测不出差异,这和 `AUTOTUNE` 是同一个机制吗?" —— 期望答"是同一个 `tf.data.Options().autotune` 体系:AUTOTUNE 传给 `num_parallel_calls`/`prefetch` 是让某一个算子的参数自动决定;而默认开启的 autotune 优化器还会在图优化阶段自动插入/调整整条管道里的 buffer,是更全局的一层自动化,两者是同一套底层机制在不同粒度上的应用"。

**常见坑:**
- 把 `AUTOTUNE` 和"不需要理解并行机制"划等号——`AUTOTUNE` 解决的是"这个数字该设多少"的调参问题,不代表可以完全不理解第 2/3 点讲的 tracing、并行、prefetch 重叠这些底层机制,遇到复杂管道/性能异常时,不理解机制就无从排查。
- 只给 `map()` 设置 `AUTOTUNE`,却忘了给 `prefetch()` 的 `buffer_size` 也设成 `AUTOTUNE`(而是写死一个固定数字或者干脆不写)——两处都应该交给运行时统一决定,只优化一半容易出现资源分配不均衡。

---

## 6. `interleave` —— 从多个数据源轮询交替读取

**是什么:**
```text
dataset.interleave(map_func, cycle_length=AUTOTUNE, block_length=1,
                    num_parallel_calls=None, deterministic=None)
```
对 `dataset` 的每个元素应用 `map_func`(通常 `map_func` 返回一个新的 `Dataset`,比如"文件名 -> 这个文件对应的 `TextLineDataset`"),然后把这些子 `Dataset` **交替**读取合并成一个 `Dataset`,而不是把它们首尾拼接。

**一句话:** `interleave` 解决的是"我有很多个分片文件,想让读取管道同时从多个文件里轮流吐数据出来"这个问题——`cycle_length` 决定同一时刻有多少个子数据源被"同时打开"参与轮询,`block_length` 决定每次轮到某个子数据源时连续取几个元素再轮转到下一个。

**底层机制/为什么这样设计:**

如果只是把 N 个分片文件的内容简单拼接(`concatenate`),读完第一个文件之前完全不会碰第二个文件——这在很多场景下是不可接受的:如果数据是按某种规律分片的(比如按类别、按采集时间分文件),先读完一个文件才读下一个,会导致训练早期只看到某一类数据,晚期才看到另一类,这对 shuffle 的有效性和训练稳定性都是负面影响。`interleave` 的做法是:同时"打开" `cycle_length` 个子数据源的迭代器,按顺序轮询——从第 1 个子源取 `block_length` 个元素,切到第 2 个子源取 `block_length` 个,……,轮完 `cycle_length` 个子源之后回到第 1 个继续,某个子源耗尽了,就从尚未加入轮询的下一个候选子源里补一个进来顶替它的位置。这样从训练一开始,管道吐出来的数据就是"来自多个分片混合"的,不需要额外等待。

`num_parallel_calls`(通常配合 `tf.data.AUTOTUNE`)让多个子源的读取本身也并行执行,这是读取分片文件(尤其是从网络文件系统/对象存储读取,I/O 延迟高)时的标准做法;并行之后默认结果顺序可能不再是确定性的(哪个子源先准备好就先吐出来),`deterministic=True` 可以要求即使并行读取,最终吐出的顺序也和顺序执行时完全一致(代价是牺牲一部分并行收益,因为要等"该轮到的那个"准备好,不能抢跑)。

**AI 研究/工程场景:** 大规模训练数据通常按文件切成很多分片(TFRecord shard、按天/按来源切分的日志文件),`files.interleave(lambda f: tf.data.TFRecordDataset(f), cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE)` 是读取这类分片数据的标准写法——既能让多个分片文件的 I/O 并行发生,又能保证训练从一开始就看到"跨分片混合"的数据,不用先读完一个分片才碰下一个。

**可运行例子(用真实写到磁盘的分片文件,验证 `cycle_length`/`block_length` 的精确轮询顺序):**
```python
import tensorflow as tf
import os, tempfile, shutil

tmpdir = tempfile.mkdtemp(prefix="tfinterleave_")
paths = []
for i in range(3):
    p = os.path.join(tmpdir, f"shard_{i}.txt")
    with open(p, "w") as f:
        f.write("\n".join(f"f{i}-L{j}" for j in range(3)))   # 每个分片3行
    paths.append(p)

files_ds = tf.data.Dataset.from_tensor_slices(sorted(paths))

# cycle_length=3(3个分片全部同时"打开"参与轮询), block_length=1(默认): 严格轮询
ds1 = files_ds.interleave(lambda p: tf.data.TextLineDataset(p), cycle_length=3, block_length=1)
out1 = [x.numpy().decode() for x in ds1]
assert out1 == ["f0-L0", "f1-L0", "f2-L0", "f0-L1", "f1-L1", "f2-L1", "f0-L2", "f1-L2", "f2-L2"]

# block_length=2: 每轮到一个分片连续取2行再轮转到下一个
ds2 = files_ds.interleave(lambda p: tf.data.TextLineDataset(p), cycle_length=3, block_length=2)
out2 = [x.numpy().decode() for x in ds2]
assert out2 == ["f0-L0", "f0-L1", "f1-L0", "f1-L1", "f2-L0", "f2-L1", "f0-L2", "f1-L2", "f2-L2"]

# cycle_length=2: 只有2个分片同时"打开",第3个分片要等前面某个耗尽后才会被拉进轮询
ds3 = files_ds.interleave(lambda p: tf.data.TextLineDataset(p), cycle_length=2, block_length=1)
out3 = [x.numpy().decode() for x in ds3]
assert out3 == ["f0-L0", "f1-L0", "f0-L1", "f1-L1", "f0-L2", "f1-L2", "f2-L0", "f2-L1", "f2-L2"]

# num_parallel_calls 并行读取 + deterministic=True: 并行但顺序依然完全可复现,和顺序版一致
ds4 = files_ds.interleave(lambda p: tf.data.TextLineDataset(p), cycle_length=3,
                           num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
out4 = [x.numpy().decode() for x in ds4]
assert out4 == out1

shutil.rmtree(tmpdir)
```
`cycle_length=2` 那组结果值得多看一眼:轮询开始时只有 `shard_0`/`shard_1` 两个文件被"打开",严格交替吐出 `f0-L0, f1-L0, f0-L1, f1-L1, f0-L2, f1-L2`——两个文件都吐完 3 行、同时耗尽后,`shard_2` 才被拉进轮询,后面连续吐出它的 3 行。这证明 `cycle_length` 限制的是"同时参与轮询的数据源数量",不是"总共能读多少个数据源"。

**用一张图看懂"轮询窗口"——上面两组实测结果背后,`cycle_length` 到底在限制什么:**

```text
cycle_length=3, block_length=1 —— 3 个分片全部同时"打开"参与轮询:

  同时打开的窗口:  [shard_0]   [shard_1]   [shard_2]
  取出顺序:   f0-L0 → f1-L0 → f2-L0 → f0-L1 → f1-L1 → f2-L1 → f0-L2 → f1-L2 → f2-L2
              └──────── 轮完一圈,每个源各取1个 ────────┘

cycle_length=2, block_length=1 —— 只有 2 个分片同时"打开",shard_2 排队等候:

  同时打开的窗口:  [shard_0]   [shard_1]          (shard_2 还没入场,在队列里等)
  取出顺序:   f0-L0 → f1-L0 → f0-L1 → f1-L1 → f0-L2 → f1-L2
                                                     │
                                 shard_0、shard_1 同时耗尽 → 窗口补位
                                                     ▼
  补位后的窗口:    [shard_2]
  取出顺序(续): f2-L0 → f2-L1 → f2-L2
```

窗口(同时打开的数据源集合)大小永远等于 `cycle_length`,和分片总数无关——3 个分片、`cycle_length=2` 时,窗口从始至终只装得下 2 个分片,`shard_2` 必须等窗口里有位置空出来才能加入,这正是上面表格里"两个文件同时耗尽后才轮到 shard_2"这个现象的直接原因。

**面试怎么问 + 追问链:**
- **Q:** "`interleave` 和直接把多个 `Dataset` `concatenate` 起来有什么区别?" —— 期望答"`concatenate` 是读完一个再读下一个,`interleave` 是多个数据源同时打开、轮询交替吐出数据,从一开始就能看到跨数据源混合的样本"。
- **追问 1:** "`cycle_length` 和 `block_length` 分别控制什么?" —— 期望能准确说出"cycle_length 控制同时打开几个数据源,block_length 控制每轮到一个数据源连续取几个元素",最好能现场画出轮询示意或者举例验证。
- **追问 2(容易问倒):** "如果有 10 个分片文件,`cycle_length=3`,读取过程中某个分片什么时候会被"换下场"、下一个新分片什么时候会被拉进来?" —— 期望答"某个正在参与轮询的分片被读完耗尽后,立刻从还没参与轮询的剩余分片里按顺序拉一个新的补上它的位置,不是等所有 3 个都耗尽才一起换"——这一点上面 `cycle_length=2` 的实测例子已经间接验证过(两个一起耗尽是因为两个分片长度恰好相等,不是巧合设计)。

**常见坑:**
- 误以为 `cycle_length` 越大越好——`cycle_length` 越大,同时打开的文件句柄/子迭代器越多,内存和文件描述符的开销也越大,不是无脑设大就好,通常交给 `tf.data.AUTOTUNE` 决定。
- `cycle_length` 传 0 或负数会直接报错(已现场触发,不是转述):
```python
import tensorflow as tf

files_ds = tf.data.Dataset.from_tensor_slices(["a", "b"])
try:
    bad = files_ds.interleave(lambda p: tf.data.Dataset.from_tensor_slices([p]), cycle_length=0)
    list(bad.as_numpy_iterator())
    assert False, "expected an error"
except tf.errors.InvalidArgumentError as e:
    assert "cycle_length must be greater than zero" in str(e)
```
- 需要严格可复现的实验(比如对比两次训练的 loss 曲线)却用了 `num_parallel_calls` 而没加 `deterministic=True`——并行 `interleave` 默认可能因为"谁先准备好谁先吐出来"导致顺序在不同运行之间轻微漂移,影响复现性。

---

## 7. tf.data 与 PyTorch DataLoader 的设计哲学对比 —— 声明式管道图 vs 命令式迭代器 + 多进程 worker

**是什么:**
```text
# tf.data:链式调用构造出一个 Dataset 对象,是一整张可以被内省/优化的管道图
ds = tf.data.Dataset.from_tensor_slices(x).map(f).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# PyTorch:DataLoader 包一层 Dataset,是一个运行时迭代器
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

**一句话:** `tf.data.Dataset` 是一个**声明式构建出来的对象**——每次 `.map()`/`.batch()`/`.prefetch()` 调用都在组装一张新的、可以被向下传递和整体分析的图节点,你能在真正开始训练之前就"看到"这张图的完整结构;`DataLoader` 是一个**命令式的运行时迭代器**——`for batch in loader` 触发的是 Python 主进程通过 `multiprocessing` 拉起 `num_workers` 个独立子进程,各自跑一份 `Dataset.__getitem__`,通过队列把结果传回主进程,这套机制在你调用 `for` 之前根本不"存在",没有一个可以被提前审视的图结构。

**底层机制/为什么这样设计:**

先看 tf.data 这一侧,`Dataset` 的每一次变换调用(`.map()`、`.batch()`、`.prefetch()`……)返回的都是一个新的 `XxxDataset` 对象,这个对象持有一个指向"上一步 `Dataset`"的引用——这不是比喻,是可以在 Python 里用内部方法真的走一遍的链表结构:

```python
import tensorflow as tf

ds = tf.data.Dataset.range(10)
ds2 = ds.map(lambda x: x * 2)
ds3 = ds2.batch(4)
ds4 = ds3.prefetch(1)

chain = []
node = ds4
while node is not None:
    chain.append(type(node).__name__)
    inputs = node._inputs() if hasattr(node, "_inputs") else []
    node = inputs[0] if inputs else None

assert chain == ["_PrefetchDataset", "_BatchDataset", "_MapDataset", "_RangeDataset"]
```
这条链和 torch-deep-dive 02 篇里"沿着 `grad_fn.next_functions` 反向遍历计算图"是完全同一种验证思路——`Dataset` 管道确实是一个真实存在、可以被程序遍历的数据结构,不是一个只在概念上成立的比喻。正因为管道在开始迭代之前就已经是一整张完整的图,`tf.data.Options()` 才能在这张图上做**整体优化**:`Options().experimental_optimization`(map 融合、no-op 消除……)、`Options().autotune`(第 5 点)、`Options().experimental_threading` 这些配置项,本质上都是在对这张图做类似 03 类 `tf.function` 里 Grappler 图优化那样的整体重写和调度决策,而不是孤立地优化某一步。

再看 PyTorch 这一侧,`DataLoader(dataset, num_workers=N)` 在你写下这行代码的时候,只是保存了一堆配置参数,什么都还没发生;真正的机制在你写 `for batch in loader` 触发 `__iter__` 的那一刻才启动——PyTorch 用 `multiprocessing` 拉起 `N` 个**完全独立的操作系统进程**,每个进程各自 import 一份用户代码、各自持有 `Dataset` 的一份(拷贝或者只读共享,取决于操作系统的 fork 语义)、各自调用 `__getitem__` 取样本,通过 `multiprocessing.Queue` 把 collate 好的 batch 序列化后传回主进程。这是一套彻头彻尾的**运行时行为**,`DataLoader` 对象本身不是一张能提前审视、能被整体重写的图——它是一堆配置 + 一段"调用 `__iter__` 时才会执行"的调度逻辑。

这个差异直接决定了两边的优化空间:tf.data 能做"全局图优化"(比如把连续的 `map` 融合成一次调用,减少 Python/graph 边界切换次数),因为优化器能看到完整管道;DataLoader 的性能调优空间主要在"调参"这个层面——`num_workers` 开多少、`prefetch_factor` 设多大、要不要 `pin_memory`,本质上是在经验性地调节"起多少个独立 OS 进程、每个进程提前准备多少个 batch",而不是让某个统一的优化器分析整条链路后自动决策。

**AI 研究/工程场景:** 排查"这条数据管道到底做了什么变换"时,tf.data 这边可以顺着上面 `_inputs()` 的思路把整条链路的每一步类型和参数都打印出来做静态审查;PyTorch 这边只能翻 `Dataset.__getitem__`/`collate_fn` 的源码,没有一个"管道对象"能一次性告诉你完整链路——这也是为什么 PyTorch 社区更依赖清晰的代码组织和文档,而不是指望从 `DataLoader` 对象本身内省出管道逻辑。

**可运行例子:**
```python
import tensorflow as tf

# Options() 暴露的是"整条管道图"级别的优化开关,这是声明式图结构才能提供的能力
opts = tf.data.Options()
assert hasattr(opts, "experimental_optimization")   # map融合/no-op消除等图级优化
assert hasattr(opts, "autotune")                     # 第5点讲的自动并行度/缓冲区决策
assert hasattr(opts, "experimental_threading")       # 线程池级别的调度配置

# element_spec 是整条管道的"静态形状契约",batch 之后最后一维长度未知,体现为 None
# (对应01类会展开的静态shape vs动态shape,这里是它在tf.data里的具体案例)
ds = tf.data.Dataset.range(10).map(lambda x: x * 2).batch(4)
assert ds.element_spec.shape.as_list() == [None]

# repeat() 同样只是往图里加一个节点,不是一个会立刻展开的 Python 循环
ds_rep = tf.data.Dataset.range(3).repeat(2)
assert list(ds_rep.as_numpy_iterator()) == [0, 1, 2, 0, 1, 2]
ds_inf = tf.data.Dataset.range(3).repeat()             # 不传参数 = 无限重复
assert ds_inf.cardinality().numpy() == tf.data.INFINITE_CARDINALITY
```

**面试怎么问 + 追问链:**
- **Q:** "tf.data 和 PyTorch DataLoader 最本质的设计差异是什么?" —— 期望答出"声明式构建的管道图对象" vs "命令式的运行时迭代器 + 多进程 worker",而不是停留在"两个都能加载数据"这种表面回答。
- **追问 1:** "这个差异实际上带来了什么不同的优化空间?" —— 期望能答出"tf.data 能做全局图优化(算子融合、统一的 autotune 资源分配),DataLoader 的性能调优主要靠经验性地调 num_workers/prefetch_factor 这类参数,没有一个统一优化器能看到完整链路"。
- **追问 2(深挖):** "你怎么向我证明 tf.data 的管道'真的'是一个可遍历的图结构,不只是一种说法?" —— 期望候选人能想到"用类似遍历链表的方式,通过 `_inputs()` 这类内部方法从最后一个 Dataset 节点一路走回最初的数据源,现场打印出每一层节点的类型",而不是单纯背概念——这条追问和 torch-deep-dive 02 篇"证明计算图是真实数据结构"的追问是同一种考法,只是换了个子系统。
- **追问 3:** "DataLoader 的多进程 worker,为什么 tf.data 不需要类似的东西来绕开 GIL?" —— 期望答出"tf.data 的 `map`/`interleave` 并行执行的是 trace 出来的图算子(C++ 层面),不是重新调度 Python 字节码,天然不受 GIL 限制,不需要靠多进程来绕开它"(呼应第 2 点)。

**常见坑:**
- 把"tf.data 更快"或者"DataLoader 更快"当成一个可以脱离具体场景一概而论的结论——两者的性能特征本质上来自不同的架构权衡(全局图优化 vs 进程级并行),具体场景下谁更快取决于预处理逻辑本身是不是能被表达成高效的 TF 图算子、还是必须依赖只有 Python 生态才有的库(比如某些 CV 增强库只有 numpy/PIL 实现,没有对应的 TF 算子,这种情况硬套 tf.data 反而可能更慢)。
- 想当然地认为两边的 `num_workers`/`num_parallel_calls` 概念可以套用同一套调参直觉——DataLoader 的 `num_workers` 是"开几个操作系统进程",数量通常不超过 CPU 核心数,开太多反而因为进程间调度和 IPC 序列化开销得不偿失;tf.data 的 `num_parallel_calls` 是"图算子的并行执行度",两者的开销模型完全不同,不能直接照搬对方的经验数字。

---

## 8. 与仓库真实 PyTorch DataLoader 代码的跨框架分工对照

这一点不是一个新的 API 知识点,而是第 7 点"设计哲学对比"的延伸——真的在仓库 `learning/` 目录下(排除 `official/repos/` 第三方仓库)挖到了两处非第三方、和数据管道直接相关的真实代码/笔记,值得做一次具体的"分工说明",而不是空谈架构差异。

先验证这两处引用确实存在(核实过程本身不需要装 torch,只是读取仓库文件的文本内容做字符串核对——这台机器上仓库的绝对路径,和 00 篇环境声明里"环境高度绑定这台机器"是同一个做法):
```python
p1 = "/mnt/e/Workspace/dummy/learning/rlhf-classic/src/rm_minimal.py"
p2 = "/mnt/e/Workspace/dummy/learning/storage-dataops/lectures/02-dataloader.md"

content1 = open(p1, encoding="utf-8").read()
assert "DataLoader(list(ds), batch_size=args.batch, shuffle=True," in content1
assert "collate_fn=collate" in content1

content2 = open(p2, encoding="utf-8").read()
assert "num_workers=N" in content2
assert "prefetch_factor=K" in content2
assert "pin_memory=True" in content2
```

**引用 1:`learning/storage-dataops/lectures/02-dataloader.md`。** 这份笔记把 PyTorch 数据加载拆成 4 个阶段:`fetch(存储读取) → decode(解码) → augment(增强) → collate(拼 batch + host-to-device)`,并总结了三个关键参数:`num_workers=N`(N 个 worker 进程并行)、`prefetch_factor=K`(每个 worker 提前准备 K 个 batch)、`pin_memory=True`(锁页内存,加速 host-to-device 拷贝)。这个 4 阶段划分和本篇讲的 tf.data 概念几乎是逐点对应的:`fetch` 对应 `from_tensor_slices`/`TFRecordDataset`/`interleave`(第 1、6 点),`decode`+`augment` 对应 `map()`(第 2 点),`collate` 对应 `batch()`(第 3 点)。

**引用 2:`learning/rlhf-classic/src/rm_minimal.py` 第 93-97 行。** 这是仓库里训练 reward model 脚本的真实代码,不是教学示例(这段引用文本本身的真实性由上面那段 `open()` 读文件的代码验证过,这里只是原样摘录,不需要在 TF venv 里装 torch 才能"运行"这段引用):
```text
from torch.utils.data import DataLoader
train_loader = DataLoader(list(ds), batch_size=args.batch, shuffle=True,
                          collate_fn=collate)
```
其中 `collate`(第 81-91 行)做的事情是对一个 batch 内的 `chosen`/`rejected` 文本样本调用 `tokenizer(..., padding=True, truncation=True)`,把长度不同的文本 pad 成同一个 batch 内的统一长度。

**分工说明:** 同样是"变长样本怎么拼成一个 batch"这个问题,两边给出的是两种不同哲学的答案。PyTorch 这边,"拼 batch"这件事被显式地表达成一个用户自己写的 Python 函数(`collate_fn`),这个函数运行在 `num_workers` 个独立子进程里的某一个上,pad 逻辑(`tokenizer(padding=True)`)完全是 Python/第三方库代码,DataLoader 本身不关心这个函数内部做了什么。tf.data 这边没有一个专门叫"collate_fn"的概念——"变长样本怎么拼 batch"被表达成一个内置的图算子 `padded_batch`,pad 这个动作是图里的一个节点,不是用户手写的自由 Python 函数:

```python
import tensorflow as tf

# 用变长整数序列模拟"tokenize之后长度不一的样本",对应 rm_minimal.py 里 collate_fn
# 内部调用 tokenizer(padding=True) 做的事——tf.data 用内置的 padded_batch 达到同样效果
raw_sequences = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]]

def to_tensor(seq):
    return tf.constant(seq, dtype=tf.int32)

ds = (
    tf.data.Dataset.from_generator(
        lambda: (to_tensor(s) for s in raw_sequences),
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    .padded_batch(2, padded_shapes=[None])   # pad到"当前batch内"的最大长度,不是全局最大长度
    .prefetch(tf.data.AUTOTUNE)
)

batches = [b.numpy().tolist() for b in ds]
assert batches == [[[1, 2, 0], [3, 4, 5]], [[6, 0, 0, 0], [7, 8, 9, 10]]]
```
第一个 batch 里 `[1, 2]` 和 `[3, 4, 5]` 的最大长度是 3,所以 `[1, 2]` 被 pad 成 `[1, 2, 0]`;第二个 batch 里 `[6]` 和 `[7, 8, 9, 10]` 的最大长度是 4,`[6]` 被 pad 成 `[6, 0, 0, 0]`——`padded_shapes=[None]` 意味着"pad 到当前这个 batch 内的最大长度",不是整个数据集的全局最大长度,这一点和 `rm_minimal.py` 里 `tokenizer(padding=True)` 默认的"pad 到 batch 内最长"是同一个语义,只是 PyTorch 这边是调库函数在 Python 里现算,tf.data 这边是图算子在图里现算。

| 维度 | `learning/rlhf-classic/src/rm_minimal.py`(PyTorch) | tf.data 等价写法 |
|---|---|---|
| "怎么拼 batch"这件事怎么表达 | 用户手写的 `collate_fn`,自由 Python 函数 | 内置图算子 `padded_batch`,不需要用户写拼接逻辑本身 |
| 变长 pad 逻辑在哪执行 | `num_workers` 个子进程里,Python/第三方库(`tokenizer`)代码 | trace 出来的图节点,C++ 层执行 |
| 并行手段 | 显式 `num_workers` 参数,OS 级子进程 + 序列化传输 | `num_parallel_calls`/`AUTOTUNE`,同进程内 C++ 线程池 |
| 开发者能控制的粒度 | 整个 `collate_fn` 函数体,想干什么都行,但要自己负责效率 | 组合内置算子(`map`/`padded_batch`/`interleave`……),灵活性换取了可被整体优化的确定性结构 |

这张表想说明的核心事实是:两个框架不是"一个更先进、一个更过时",而是各自把"数据管道"这个问题放在了不同的抽象层级上——PyTorch 把管道的具体实现细节完全交还给开发者用普通 Python 写,换来极高的灵活性(collate_fn 里能写任意逻辑);tf.data 把管道表达成一组受限的、可组合的声明式算子,换来整条链路能被运行时整体分析和优化的确定性结构。选哪一个,本质上是在"灵活性"和"可优化性"之间做权衡,这也是第 7 点"设计哲学对比"最终要落到的结论。

---

## 小结:这一批 8 个知识点解决的问题

| # | 知识点 | 核心结论 | 本机验证情况 |
|---|---|---|---|
| 1 | `Dataset.from_tensor_slices` | 沿 axis 0 切片成 N 个元素(不是复制N份),但构造时会把输入数据整体拷贝进 TF 自己的存储(不是像 torch.from_numpy 那样共享内存) | 已实测(mutate-after-construct 验证拷贝语义,cardinality/element_spec 验证切片语义) |
| 2 | `map()` 与 `num_parallel_calls` | map 函数只在构造时被 trace 一次(和 tf.function 同一套机制),支持 AutoGraph;num_parallel_calls 并行的是 trace 出来的图,不是重新调度 Python | 已实测(call_count 验证只 trace 一次,SymbolicTensor 类型验证,AutoGraph if 转换验证,并行提速 7x 实测) |
| 3 | `batch`/`shuffle`/`prefetch` 顺序 | shuffle 在 batch 前后决定打乱粒度(单样本 vs batch顺序);prefetch 只重叠它上游的工作,放错位置等于白写 | 已实测(shuffle/batch顺序用连续性检测验证;prefetch位置用关闭autotune后的计时验证;额外发现默认autotune会自动兜底部分场景) |
| 4 | `cache()` | 缓存它前面管道第一次跑出来的结果;随机操作(shuffle/增强)必须放在cache之后,否则"随机"会被冻结成第一次的值且不报错 | 已实测(磁盘缓存文件产物验证;augment/shuffle前后cache的跨epoch差异验证) |
| 5 | `tf.data.AUTOTUNE` | 哨兵值-1,把并行度/缓冲区大小的决策交给运行时动态调整,不需要开发者猜固定数字 | 已实测(AUTOTUNE值验证,Options().autotune字段内省,固定值vs AUTOTUNE吞吐对比7x+) |
| 6 | `interleave` | 同时"打开"cycle_length个数据源轮询交替读取,block_length控制每轮连续取几个,常用于分片文件读取 | 已实测(真实写磁盘分片文件,精确验证cycle_length/block_length的轮询顺序,含并行deterministic一致性) |
| 7 | 与 PyTorch DataLoader 设计哲学对比 | tf.data是声明式构建的可整体优化管道图对象;DataLoader是命令式运行时迭代器+多进程worker模型 | 已实测(_inputs()链式遍历验证管道确为可遍历图结构,呼应torch 02篇grad_fn遍历的验证思路) |
| 8 | 与仓库真实 PyTorch 代码的跨框架分工对照 | 同一个"变长样本拼batch"问题:PyTorch用用户手写collate_fn(自由但需自己保证效率),tf.data用内置padded_batch图算子(受限但可被整体优化) | 真实仓库引用(learning/storage-dataops/lectures/02-dataloader.md + learning/rlhf-classic/src/rm_minimal.py:93-97,已验证文件内容,非编造) |

下一批:[10-memory-and-performance.md](10-memory-and-performance.md)

---

*更新:2026-07-09*
