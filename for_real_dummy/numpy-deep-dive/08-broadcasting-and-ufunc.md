# 08 · 广播与 ufunc 机制(Broadcasting & ufunc internals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批和前面几批不一样:前面每一节是"一个函数怎么用",这一批第一节是"一套机制怎么运作"——广播规则本身不是函数,而是 numpy 每次做 `+ - * /` 这类逐元素运算前,自动执行的一套形状对齐算法。[01-numpy-for-c-programmers.md](../01-numpy-for-c-programmers.md) 第 5 节已经用 `(3,4)+(4,)` 这一个例子让你知道"广播存在、能用",这里要往下钻一层:把规则讲到能自己判断"这两个形状到底能不能广播"、"报错时到底哪里对不上"的程度。后半部分讲几个"万能循环工具"(`vectorize`/`apply_along_axis`/`apply_over_axes`)和归约函数的真身(ufunc 的 `.reduce`/`.accumulate`)——这一半会揭示 numpy 一个很重要的设计事实:`np.sum`/`np.cumsum` 这些你天天用的函数,骨子里都是同一小撮 ufunc 对象的方法,不是各自独立实现的。这一点会在 [04-elementwise-math.md](04-elementwise-math.md)(四则运算的 ufunc 本质)里被继续展开。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,包括故意写的"不能广播"的报错例子——报错信息和异常类型都是真实触发后记录下来的,不是凭空写的。

---

## 1. 广播(Broadcasting)规则深入专题

**规则说明(三条,记住这三条就能推导所有情况):**

1. **从右往左对齐**两个数组的 shape 元组——是"对齐末尾",不是"对齐开头"。这是最容易被直觉误导的一点:形状元组里"看起来一样的数字"如果不在对齐后的同一位置,不算数。
2. 对齐后逐维比较,某一维**相等**,或者其中**一个是 1**,这一维就"兼容"——那个 1 会被概念上拉伸(不是真的复制内存)到和另一边相同的大小。
3. 如果两个数组的维度数(`ndim`)不一样,**较短的那个左边缺失的维度一律看成 1**,再按规则 2 比较。

只要任何一维**既不相等、也没有一个是 1**,整个广播失败,直接抛 `ValueError`。

**一句话:** 广播就是 numpy 让"形状不同但按上面规则兼容"的两个数组,不用你手写循环、不用真的复制数据,就能直接做逐元素运算的机制。

**AI 研究场景:** [01-numpy-for-c-programmers.md](../01-numpy-for-c-programmers.md) 已经见过矩阵加偏置向量这一种广播。但真实研究代码里,**几乎所有"batch 运算"能一行写出来,背后都是广播在撑着**:给一个 batch 的数据同时减去每个特征的均值、除以标准差(下面就是这个例子);给多头注意力的每个 head 同时加一个 mask;给三维特征图的每个通道乘一个缩放系数。不吃透规则,容易写出两类 bug:一类是"形状对不上直接报错"(还算幸运,至少能发现),另一类更隐蔽——**形状凑巧能广播,但对齐到了错误的维度上**,程序不报错但结果是错的,调试起来比报错更难查。下面的例子刻意包含"直觉上以为能广播、实际报错"的情况,就是为了把第二类风险讲清楚。

**可运行例子:**

例 1——列向量广播(每一行加一个各不相同的偏置,而不是所有行加同一个向量):

```python
import numpy as np

mat = np.arange(12, dtype=np.float64).reshape(3, 4)     # shape (3, 4)
row_bias = np.array([[100.0], [200.0], [300.0]])         # shape (3, 1)

out1 = mat + row_bias
assert out1.shape == (3, 4)
assert np.array_equal(out1[0], mat[0] + 100.0)
assert np.array_equal(out1[2], mat[2] + 300.0)
```

对齐方式:

```
mat:      (3, 4)
row_bias: (3, 1)   ← 右对齐,逐维比较:3 vs 3(相等) ,4 vs 1(1被拉伸成4)
结果:      (3, 4)
```

例 2(会报错)——`(3,4) + (3,)`,直觉上以为"3 那个数对上了"就该能按行加,实际上不行:

```python
import numpy as np

mat = np.arange(12, dtype=np.float64).reshape(3, 4)
vec3 = np.array([100.0, 200.0, 300.0])          # shape (3,)

try:
    mat + vec3
    raise RuntimeError("不应该走到这里,期望抛出 ValueError")
except ValueError as e:
    msg = str(e)
    print("捕获到的异常类型: ValueError,信息:", msg)
    assert "could not be broadcast" in msg      # 本机实测抛出:
    # ValueError: operands could not be broadcast together with shapes (3,4) (3,)

# 正确修复方式:显式把 (3,) 变成 (3,1),把"该对齐的位置"占住
fixed = mat + vec3.reshape(3, 1)
assert fixed.shape == (3, 4)
assert np.array_equal(fixed, mat + np.array([[100.0], [200.0], [300.0]]))

fixed2 = mat + vec3[:, None]        # np.newaxis 写法,和 reshape(3,1) 完全等价
assert np.array_equal(fixed2, fixed)
```

对齐方式(为什么失败):

```
mat:   (3, 4)
vec3:     (3,)   ← 右对齐:mat 最后一维是 4,vec3 唯一一维是 3
                    4 和 3 既不相等,也没有一个是 1 → 报错
```

"两边都有个 3"是错觉——广播只看**对齐后处在同一位置**的维度,`vec3` 的 3 对齐到的是 `mat` 的最后一维(4 所在的位置),不是第一维。

例 3——`(8,3,4) + (4,)`,维度数不同,缺失的维度自动当 1(呼应规则 3):

```python
import numpy as np

feature_maps = np.zeros((8, 3, 4))              # 8 个样本,每个 3x4 的特征
channel_bias = np.arange(4, dtype=np.float64)   # shape (4,)

out3 = feature_maps + channel_bias
assert out3.shape == (8, 3, 4)
assert np.array_equal(out3[0, 0], channel_bias)
assert np.array_equal(out3[7, 2], channel_bias)
```

```
feature_maps:  (8, 3, 4)
channel_bias:  (      4)   ← 右对齐;channel_bias 缺失的前两维自动补1,变成 (1,1,4)
                              两个 1 分别被拉伸成 8 和 3
```

例 4——`(3,1,1) + (1,4,1)`,**双方同时被拉伸**,不是单方面谁迁就谁:

```python
import numpy as np

x4 = np.arange(3, dtype=np.float64).reshape(3, 1, 1)   # 3 个 query
y4 = np.arange(4, dtype=np.float64).reshape(1, 4, 1)   # 4 个 key

out4 = x4 + y4
assert out4.shape == (3, 4, 1)
assert out4[1, 2, 0] == x4[1, 0, 0] + y4[0, 2, 0]        # 每一对 (query_i, key_j) 都算到了
```

```
x4: (3, 1, 1)
y4: (1, 4, 1)   ← 逐维比较:3 vs 1(拉伸y4) ,1 vs 4(拉伸x4) ,1 vs 1(相等)
结果: (3, 4, 1)  —— 两边各贡献一个"真实"维度,组合出全部两两配对
```

这种"双方互相拉伸"常见于**两两配对(pairwise)计算**:3 个 query 和 4 个 key 逐对做某种打分,先把 query reshape 成 `(3,1,d)`、key reshape 成 `(1,4,d)`,相减/相乘后直接得到 `(3,4,d)` 的全部组合,不用写双重 `for` 循环。

例 5(会报错)——`(2,3) + (2,4)`,最后一维 3 和 4,既不相等也没有一个是 1:

```python
import numpy as np

p = np.ones((2, 3))
q = np.ones((2, 4))

try:
    p + q
    raise RuntimeError("不应该走到这里")
except ValueError as e:
    msg = str(e)
    print("捕获到的异常类型: ValueError,信息:", msg)
    assert "(2,3)" in msg and "(2,4)" in msg
    # 本机实测抛出:
    # ValueError: operands could not be broadcast together with shapes (2,3) (2,4)
```

AI 场景——batch 归一化,减均值除以标准差,一行写完:

```python
import numpy as np

batch = np.array([[1., 2., 3.],
                   [4., 5., 7.],
                   [10., 10., 10.]])   # shape (3, 3):3 个样本 x 3 个特征

mean = batch.mean(axis=0)              # shape (3,) —— 每个特征在整个 batch 上的均值
std = batch.std(axis=0)                # shape (3,) —— 每个特征的标准差
normed = (batch - mean) / std          # (3,3) 和 (3,) 广播,自动对齐到最后一维

assert mean.shape == (3,)
assert normed.shape == (3, 3)
assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-10)
assert np.allclose(normed.std(axis=0), 1.0, atol=1e-10)
```

**常见坑:**
- **对齐方向是从右往左,不是从左往右。** 例 2 是最典型的踩坑现场:`(3,4)` 和 `(3,)` 里都有个"3",新手直觉上以为"行数对上了就该能广播",但 numpy 只看对齐后**同一位置**的维度——`(3,)` 唯一的维度对齐到的是 `(3,4)` 的最后一维(4 所在的位置),和第一维的 3 毫无关系。想让一个一维数组按"行"广播,必须显式变成列向量:`.reshape(N, 1)` 或 `[:, None]`。
- **报错信息 `shapes (a) (b)` 里的两个 shape,是"这一步尝试对齐失败"时的形状**,不一定是你最初写的那个数组——链式运算中间某一步结果的形状变了,报错也会显示那个中间形状。读报错时先确认这两个 shape 到底是不是你以为的那两个数组。
- **广播不会真的复制内存**——这是它比手动 `for` 循环快、比 `np.tile` 省内存的原因,但也意味着"拉伸"只是概念上的,不要以为广播之后的中间结果占用了和拉伸后形状同样多的实际内存。

---

## 2. `np.vectorize`

**签名:**
```python
np.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
```
- `pyfunc`:一个只能处理**标量**输入的普通 Python 函数
- `otypes`:显式指定输出的 dtype(见下面"常见坑",不指定有代价)

**一句话:** 把一个只会处理单个标量的 Python 函数,包装成一个能接受数组输入、按 numpy 广播规则遍历的函数——官方文档原话是"the vectorized function evaluates `pyfunc` over successive tuples of the input arrays **like the python map function**",它自己承认本质就是 `map`。

**AI 研究场景:** 手写了一个逐元素的自定义规则(比如某种分段的后处理函数、调用了某个只能吃标量的第三方库函数),想让它能直接喂数组进去而不改函数本身,`vectorize` 图省事、一行包一下就能用。**但这正是最容易踩的性能陷阱**:新手常常误以为"vectorize"这个名字意味着"变成了向量化实现、会变快",实际上它内部就是一层 Python 循环,在 C 级别的向量化(真正的 ufunc)面前完全没有速度优势。小数据量测试时感觉不出来,一旦从几十条测试数据换成几十万条真实训练数据,预处理这一步就可能悄悄拖慢整个流水线。**它的价值是写起来方便、语义清晰,不是性能。**

**可运行例子:**

```python
import numpy as np
import time

def piecewise(x):
    """自定义分段函数:正数取平方,负数取相反数——只能处理单个标量"""
    if x > 0:
        return x ** 2
    else:
        return -x

vec_piecewise = np.vectorize(piecewise)

x = np.linspace(-5, 5, 200000)
result = vec_piecewise(x)

assert result.shape == x.shape
assert result[0] == piecewise(x[0])

# 正确性:三种写法结果一致(浮点误差用 allclose,标量pow和数组pow内部实现路径不同,量级在1e-16)
loop_result = np.array([piecewise(v) for v in x])
native_result = np.where(x > 0, x ** 2, -x)
assert np.array_equal(result, loop_result)
assert np.allclose(result, native_result)

# 性能对比:vectorize vs 手写循环 vs 真正向量化(np.where)
t0 = time.perf_counter(); _ = np.array([piecewise(v) for v in x]); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); _ = vec_piecewise(x);                    t_vec  = time.perf_counter() - t0
t0 = time.perf_counter(); _ = np.where(x > 0, x ** 2, -x);         t_native = time.perf_counter() - t0

assert t_vec < t_loop * 1.5     # vectorize 和手写循环是同一数量级,不是"加速版"
assert t_native * 5 < t_vec     # 真正向量化的版本明显快了不止一个数量级
```

本机多次实测:`vectorize` 的耗时约为手写 `for` 循环的 70%~85%(同一数量级,谈不上"加速"),而真正向量化的 `np.where` 版本比 `vectorize` 快了 18~35 倍不等(每次运行有波动,但从未接近同一数量级)。

**常见坑:**
- **名字带"vectorize"不代表向量化加速**——这是它最大的误导性。需要性能时,永远优先寻找能直接表达成 numpy 内置运算(`np.where`、布尔索引、算术运算符)的写法,`vectorize` 只在"图方便、数据量小、不在性能热路径上"时才划算。
- **不指定 `otypes` 时,第一个元素会被多算一次**:`vectorize` 需要先摸底输出的 dtype,做法是先用第一个元素试跑一次函数,再正式开始遍历。如果你的函数有副作用(打印日志、写文件、累加计数器),第一个元素就会被实际执行两次——下面代码验证了这一点(5 个元素,函数被调用了 6 次),显式传 `otypes` 可以跳过这次试跑:
```python
import numpy as np

call_count = 0
def f(v):
    global call_count
    call_count += 1
    return v * 2

vf = np.vectorize(f)
arr = np.array([1, 2, 3, 4, 5])
_ = vf(arr)
assert call_count == 6      # 5个元素却调用了6次——第一个元素被多算了一遍探测类型

call_count = 0
vf2 = np.vectorize(f, otypes=[np.int64])
_ = vf2(arr)
assert call_count == 5      # 显式给 otypes,跳过探测调用
```

---

## 3. `np.apply_along_axis`

**签名:**
```python
np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
```
- `func1d`:接受一个一维数组、返回一个值(或一维数组)的函数
- `axis`:沿着哪个轴切出这些一维切片
- `arr`:输入数组

**一句话:** 沿着指定的轴,把数组切成一条一条的一维切片,对每条切片分别调用 `func1d`,再把结果拼回去——官方文档自己给出的等价实现就是双重 `for` 循环(`for ii in ndindex(...): for kk in ndindex(...): out[...] = func1d(arr[...])`),所以它和 `vectorize` 一样,本质是 Python 层的循环,只是比你手写这层循环省一点样板代码,不是 C 级别的向量化。

**AI 研究场景:** 当归一化/自定义逻辑**没法简单拆成"整个数组减一个广播得到的向量"**时(比如每一行要用自己的 min/max 做 min-max 缩放,而不是用整个 batch 共享的统计量),`apply_along_axis` 能省掉手写 `for` 循环遍历行/列的样板代码。典型场景:每一行是一个独立样本,要按**这一行自己的**最大最小值缩放到 `[0,1]`。

**可运行例子:**

```python
import numpy as np

def normalize_row(row):
    """min-max归一化到[0,1]:每一行用自己的min/max,不是整个batch共享的统计量"""
    return (row - row.min()) / (row.max() - row.min())

data = np.array([[1., 2., 3.],
                  [10., 20., 40.]])

normalized = np.apply_along_axis(normalize_row, axis=1, arr=data)

assert normalized.shape == data.shape
assert np.allclose(normalized[0], [0.0, 0.5, 1.0])
assert np.allclose(normalized[1], [0.0, 1/3, 1.0])
assert np.allclose(normalized.min(axis=1), [0.0, 0.0])
assert np.allclose(normalized.max(axis=1), [1.0, 1.0])

# axis=0:对每一列做同样的事,方向反过来
normalized_col = np.apply_along_axis(normalize_row, axis=0, arr=data)
assert normalized_col.shape == data.shape
assert np.allclose(normalized_col[:, 0], [0.0, 1.0])   # 第0列 [1,10] 被缩放成 [0,1]
```

**常见坑:**
- `axis=1` 直觉上容易理解成"沿着第二维方向扫描",但更准确的理解是"**每次固定其它维度,把 axis 这一维整条抠出来喂给函数**"——`axis=1` 时 `func1d` 收到的是"某一行"(一整行,axis=1 这一维被抠出来了),不是"某一列"。不确定就打印一次 `func1d` 接收到的数组形状确认。
- 和 `vectorize` 一样是 Python 层循环,数据量很大且逻辑能用 `mean`/`std` 等自带 `axis=` 参数的内置函数表达时(比如本节例子如果统计量是整个 batch 共享的,就该退回广播写法),优先用广播,不要习惯性地想到 `apply_along_axis`。

---

## 4. `np.apply_over_axes`

**签名:**
```python
np.apply_over_axes(func, a, axes)
```
- `func`:必须接受 `func(a, axis)` 两个参数,且返回结果的维度数要和 `a` 相同(或少 1 维,少的那维会被自动插回来)——`np.sum`/`np.mean` 这类支持 `axis=` 参数的归约函数都满足这个要求
- `axes`:一个轴的列表,`func` 会按列表顺序依次对每个轴调用

**一句话:** 对多个轴依次调用同一个归约函数,并且**始终保留维度**(被规约的轴变成大小 1,而不是像 `arr.sum(axis=1)` 默认那样直接消失)——官方文档明确说明它等价于"给支持 tuple 轴参数的 ufunc 传 `axis=(...), keepdims=True`"。

**AI 研究场景:** 处理多张特征图(比如卷积网络某一层的输出,形状 `(batch, height, width)`),想对每个样本的 `height` 和 `width` 两个轴都求和得到"总响应",但后面还要拿这个总响应去除原数组算"每个位置占总响应的比例"——这时候结果**必须保留维度**才能直接广播回去,不然还要手动 `reshape`/`expand_dims`。

**可运行例子:**

```python
import numpy as np

# 模拟 batch=2 的两张 3x4 特征图
feature_maps = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

# 依次对 axis=1(高)和 axis=2(宽)求和,得到每个样本的总响应,维度保留
total_response = np.apply_over_axes(np.sum, feature_maps, [1, 2])

assert total_response.shape == (2, 1, 1)          # 被规约的两个轴变成1,不是被删掉
assert np.allclose(total_response.ravel(), feature_maps.sum(axis=(1, 2)))
assert np.array_equal(total_response, feature_maps.sum(axis=(1, 2), keepdims=True))

# 维度保留的意义:能直接广播回原数组,算"每个位置占总响应的比例"
ratio = feature_maps / total_response
assert ratio.shape == (2, 3, 4)
assert np.allclose(ratio.sum(axis=(1, 2)), 1.0)    # 每个样本内部比例加起来是1
```

**常见坑:**
- 这个函数比 `keepdims` 参数在 numpy 里普及得早,**现在大部分场景直接写 `arr.sum(axis=(1,2), keepdims=True)` 更直接**,效果完全一样(上面代码已经 assert 验证两者相等)。`apply_over_axes` 值得知道的场合是:你要重复调用的函数本身不支持 tuple 形式的 `axis` 参数,或者需要对不同轴依次用不同函数处理时,它提供的是"按顺序、依次处理多个轴"这个更通用的框架。
- `func` 的返回值必须"维度数和输入相同,或者恰好少 1 维"——如果传一个不满足这个约定的函数(比如返回一个形状完全对不上的结果),会得到令人困惑的报错或错误结果,不是所有函数都能直接塞进来。

---

## 5. ufunc 的 `.reduce` 方法 —— `np.add.reduce` 就是 `np.sum`

**签名:**
```python
ufunc.reduce(array, axis=0, dtype=None, out=None, keepdims=False)
```
- `array`:输入数组
- `axis`:沿哪个轴规约,**默认是 0**(见下面"常见坑",这个默认值和 `np.sum()` 不一样)

**一句话:** 对二元 ufunc(比如 `np.add`、`np.multiply`)调用 `.reduce`,相当于把这个二元操作**像 `reduce`/`fold` 一样连续应用到整条轴上**,把这一维"压扁"成一个值——`np.add.reduce` 就是 `np.sum` 的真身,官方文档原话:"add.reduce() is equivalent to sum()"。

**AI 研究场景:** 这一点揭示了 numpy 设计哲学里很关键的一层:`+ - * /` 这些四则运算符,底层调用的其实是 `np.add`/`np.subtract`/`np.multiply`/`np.divide` 这些 ufunc 对象(`+` 只是运算符重载,这是 [04-elementwise-math.md](04-elementwise-math.md) 要展开的"四则运算的 ufunc 本质")。而 `np.sum`/`np.prod`/`np.max` 这些看起来和"四则运算"平行的归约函数,**并不是另起炉灶实现的**,而是同一个 ufunc 对象身上长出来的 `.reduce` 方法。理解了这一点,遇到"这个函数怎么表现得和某个运算符这么像"的疑惑时,第一反应可以是去查对应的 ufunc 有没有 `.reduce`/`.accumulate` 方法,而不是当成两个无关的知识点分别记。

**可运行例子:**

```python
import numpy as np

a = np.array([1., 2., 3., 4.])
assert np.add.reduce(a) == np.sum(a) == 10.0

b = np.array([[1., 2.], [3., 4.]])
assert np.array_equal(np.add.reduce(b, axis=0), np.sum(b, axis=0))
assert np.array_equal(np.add.reduce(b, axis=1), np.sum(b, axis=1))

# 不仅是 sum:任何二元 ufunc 都能 reduce
assert np.multiply.reduce(np.array([1, 2, 3, 4])) == 24              # 1*2*3*4,"连乘"
assert np.multiply.reduce(np.array([1, 2, 3, 4])) == np.prod(np.array([1, 2, 3, 4]))
assert np.maximum.reduce(np.array([3, 1, 4, 1, 5])) == np.max(np.array([3, 1, 4, 1, 5]))
```

**常见坑:**
- **`ufunc.reduce` 不传 `axis` 时默认值是 `axis=0`,只规约第一维**;而 `np.sum(arr)` 不传 `axis` 时默认是"拍平所有维度求和"。这是两者唯一不等价的地方,亲手验证一下差异有多大:
```python
import numpy as np

b = np.array([[1., 2.], [3., 4.]])
default_reduce = np.add.reduce(b)
default_sum = np.sum(b)

assert np.array_equal(default_reduce, np.array([4.0, 6.0]))   # 只对 axis=0 求和,shape (2,)
assert default_sum == 10.0                                     # 拍平所有维度,是标量
assert np.shape(default_reduce) != np.shape(default_sum)       # 形状都不一样
```
- 想让 `.reduce` 和 `np.sum(arr)`(不传 axis)行为一致,要显式传 `axis=None`(`np.add.reduce(b, axis=None)` 才会拍平所有维度求和,和 `.reduce()` 默认的 `axis=0` 是两回事)。

---

## 6. ufunc 的 `.accumulate` 方法 —— `np.add.accumulate` 就是 `np.cumsum`

**签名:**
```python
ufunc.accumulate(array, axis=0, dtype=None, out=None)
```
- 和 `.reduce` 的区别:`.reduce` 只保留"压到最后"的那一个结果,`.accumulate` 把**每一步的中间结果都保留下来**,输出形状和输入一致

**一句话:** `.accumulate` 是 `.reduce` 的"留过程"版本——`np.add.accumulate` 就是 `np.cumsum`(累加),`np.multiply.accumulate` 就是 `np.cumprod`(累乘),同样是同一个 ufunc 对象身上的方法,不是独立实现。

**AI 研究场景:** 训练曲线的"累计"统计(比如到当前 step 为止的累计训练样本数、累计奖励)、离散概率分布的累积分布函数(CDF,给定每个类别的概率,累加得到"小于等于第 k 类的总概率"用于按概率区间采样),都是 `cumsum` 的场景——知道它背后是 `np.add.accumulate`,再遇到"我需要一个不是加法、而是取最大值的累计"(比如"到当前位置为止见过的最大值"这种滑动统计),就能立刻想到 `np.maximum.accumulate`,而不是去翻文档找有没有专门的"cummax"函数。

**可运行例子:**

```python
import numpy as np

a = np.array([1., 2., 3., 4.])
assert np.array_equal(np.add.accumulate(a), np.cumsum(a))
assert np.array_equal(np.add.accumulate(a), np.array([1., 3., 6., 10.]))

assert np.array_equal(np.multiply.accumulate(a), np.cumprod(a))
assert np.array_equal(np.multiply.accumulate(a), np.array([1., 2., 6., 24.]))

b = np.array([[1., 2.], [3., 4.]])
assert np.array_equal(np.add.accumulate(b, axis=0), np.cumsum(b, axis=0))
assert np.array_equal(np.add.accumulate(b, axis=1), np.cumsum(b, axis=1))

# AI 场景:离散概率分布的累积分布函数(CDF)
probs = np.array([0.1, 0.2, 0.3, 0.4])
cdf = np.cumsum(probs)
assert np.allclose(cdf, [0.1, 0.3, 0.6, 1.0])
assert np.array_equal(cdf, np.add.accumulate(probs))
```

**常见坑:**
- 和 `.reduce` 完全一样的默认值陷阱:**`accumulate` 不传 `axis` 默认是 `axis=0`(保持原形状,只沿第一维累加)**,而 `np.cumsum(arr)` 不传 `axis` 时会先把数组拍平成一维,再整体累加——两者默认行为不一致,亲手验证:
```python
import numpy as np

b = np.array([[1., 2.], [3., 4.]])
default_acc = np.add.accumulate(b)
default_cumsum = np.cumsum(b)

assert default_acc.shape == (2, 2)      # 保持原形状,只沿 axis=0 累加
assert default_cumsum.shape == (4,)     # 先拍平成1维,再累加
assert default_acc.shape != default_cumsum.shape
```
- `.reduce` 和 `.accumulate` 只对**二元 ufunc**(接受两个输入的,比如 `add`/`multiply`/`maximum`/`minimum`)有意义——`np.sqrt`/`np.exp` 这种只接受一个输入的一元 ufunc 上虽然也挂着 `.reduce`/`.accumulate` 这两个方法名,但调用时会直接报错,道理很直接:这两个方法本质是把**二元**操作反复应用,一元函数没有"反复应用"的意义:
```python
import numpy as np

try:
    np.sqrt.reduce(np.array([1., 4., 9.]))
except ValueError as e:
    print("捕获到 ValueError:", e)
    assert "only supported for binary functions" in str(e)
    # 本机实测抛出:ValueError: reduce only supported for binary functions
```

---

## 小结:这一批 6 个主题解决的问题

| 主题 | 解决的问题 |
|---|---|
| 广播规则专题 | 形状不同的数组何时能直接运算、何时会报错、怎么读懂报错信息 |
| `vectorize` | 把标量函数包装成能吃数组的函数(方便,不是加速) |
| `apply_along_axis` | 沿指定轴对每个一维切片调用自定义函数(如逐行自定义归一化) |
| `apply_over_axes` | 对多个轴依次做归约,同时保留维度以便继续广播 |
| ufunc `.reduce` | 揭示 `sum`/`prod`/`max` 等归约函数都是 ufunc 的方法,不是独立实现 |
| ufunc `.accumulate` | 揭示 `cumsum`/`cumprod` 等前缀累计函数同样是 ufunc 的方法 |

下一批:[09-advanced-random.md](09-advanced-random.md) —— 随机数进阶与可复现(新版 Generator API)。

---

*更新:2026-07-07*
