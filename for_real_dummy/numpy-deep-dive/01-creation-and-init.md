# 01 · 创建与初始化(Creation & Initialization)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**怎么生成一个起始的 array**——无论是全零占位、随机初始化权重,还是构造一段等差序列做实验数据。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `np.array`

**签名:**
```python
np.array(object, dtype=None)
```
- `object`:一个 Python list/tuple(可以嵌套,对应多维)
- `dtype`:强制指定数据类型,不写的话 numpy 自动推断(全是整数就是 int64,有小数就是 float64)

**一句话:** 把 Python 的 list/tuple 转换成 numpy 的 ndarray。

**AI 研究场景:** 这是几乎所有 numpy 代码的起点——手写一组测试数据、把 JSON/CSV 读出来的 Python list 转成能做数值运算的数组、单元测试里构造"标准答案"用来 assert 比对,都是 `np.array`。你在 [01-numpy-for-c-programmers.md](../01-numpy-for-c-programmers.md) 里第一次见到的 `np.array([[1,2],[3,4]])` 就是这个。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

assert a.shape == (3,)
assert b.shape == (2, 2)
assert a.dtype == np.int64          # 全是整数,自动推断成 int64(本机验证结果)

c = np.array([1, 2, 3], dtype=np.float32)
assert c.dtype == np.float32        # 显式指定,常用于要求精度/内存可控的场景
```

**常见坑:** 不指定 `dtype` 时,如果 list 里混了 int 和 float(比如 `[1, 2.0, 3]`),numpy 会自动把整个数组"升级"成 float64——这在深度学习里经常导致"为什么我这个数组是 float64 不是 float32"的困惑,内存翻倍还可能跟模型权重 dtype 对不上报错。**训练代码里养成习惯:关键数组显式写 `dtype=np.float32`。**

---

## 2. `np.zeros` / `np.ones`

**签名:**
```python
np.zeros(shape, dtype=float)
np.ones(shape, dtype=float)
```
- `shape`:一个整数(一维)或者元组(多维),比如 `(3, 4)`
- `dtype`:默认是 float64

**一句话:** 生成一个指定形状、元素全是 0(或全是 1)的数组。

**AI 研究场景:**
- **占位/累加器:** 训练循环里经常需要一个"从 0 开始累加"的容器,比如 `total_loss = np.zeros(num_batches)`,之后每个 batch 填一个值。
- **mask 初始化:** attention mask、padding mask 常常先 `np.ones` 全部允许,再对该屏蔽的位置赋 0(或反过来,视实现约定而定)。
- **bias 初始化:** 神经网络的 bias 项传统上初始化成全 0(weight 才需要随机初始化——如果 weight 也全初始化成 0,所有神经元算出来的梯度完全一样,永远学不出差异,这叫"对称性问题",[02-pytorch-basics.md](../02-pytorch-basics.md) 的 autograd 部分背后就是这个原理)。

**可运行例子:**
```python
import numpy as np

total_loss = np.zeros(5)          # 5 个 batch 的 loss 占位
assert total_loss.sum() == 0.0

bias = np.zeros(10)               # 10 个神经元的 bias,标准初始化
assert (bias == 0).all()

mask = np.ones((3, 3))            # 3x3 全部"允许"的 mask
assert mask.sum() == 9
```

**常见坑:** `shape` 传单个整数和传元组的结果不同——`np.zeros(3)` 是一维 `(3,)`。如果想要"3行1列"的二维数组,必须写 `np.zeros((3, 1))`,漏掉括号写成 `np.zeros(3, 1)` 会报错(因为第二个位置参数是 `dtype`,不是形状的一部分)。

---

## 3. `np.full`

**签名:**
```python
np.full(shape, fill_value, dtype=None)
```
- `fill_value`:填充的值,不局限于 0 或 1

**一句话:** `zeros`/`ones` 的推广——生成指定形状、元素全部是同一个**任意值**的数组。

**AI 研究场景:** attention 里给"禁止关注"的位置打分数设成 `-inf`(softmax 之后趋近于 0),或者给某个默认奖励/默认置信度打统一初始分,都不是 0/1,这时候需要 `full`。

**可运行例子:**
```python
import numpy as np

mask_value = np.full((2, 3), -np.inf)
assert mask_value.shape == (2, 3)
assert np.all(mask_value == -np.inf)

default_score = np.full(5, 0.5)
assert np.allclose(default_score, [0.5] * 5)
```

**常见坑:** `fill_value` 的类型要和你期望的 `dtype` 匹配,不匹配时 numpy 会做隐式转换(比如往 `dtype=int` 的数组里塞一个很大的浮点数可能溢出或产生意外结果),不确定就显式传 `dtype`。

---

## 4. `np.empty`

**签名:**
```python
np.empty(shape, dtype=float)
```

**一句话:** 分配一块指定形状的内存,但**不初始化内容**——里面是当时内存里残留的随机垃圾值,不是 0。

**AI 研究场景:** 当你马上要把每个位置都覆盖赋值时(比如逐行填充的 for 循环),用 `empty` 比 `zeros` 快一点点(省去清零的开销)——这是性能敏感场景下的预分配技巧;numpy 内部很多函数也用 `empty` 做临时缓冲区。对研究代码来说这是个"知道即可"的细节,不是高频操作。

**可运行例子:**
```python
import numpy as np

buf = np.empty((3, 3))
assert buf.shape == (3, 3)

for i in range(3):
    buf[i] = i          # 必须全部覆盖后才能放心使用

assert np.array_equal(buf[:, 0], [0, 1, 2])
```

**常见坑:** **千万不要在 `empty` 之后立刻读取里面的值当作有意义的数据**——内容是未初始化的内存垃圾,不是 0。这是新手最容易踩的坑,常常误以为它和 `zeros` 一样"安全",结果读到一堆随机数导致诡异的 bug。不确定就用 `zeros`,只有明确知道会整体覆盖时才用 `empty`。

---

## 5. `np.zeros_like` / `np.ones_like`

**签名:**
```python
np.zeros_like(a, dtype=None)
np.ones_like(a, dtype=None)
```
- `a`:一个已有的数组,**形状和 dtype 会被自动复用**

**一句话:** 生成一个和已有数组 `a` 形状、dtype 都相同的全 0(或全 1)数组,不用手动重复写 shape。

**AI 研究场景:** 训练循环里想给某个参数配一个同形状的梯度累加器或 mask,比如 `grad_accum = np.zeros_like(param)`——不需要手动查 `param` 的 shape 再传给 `zeros`,直接复用,不会因为模型结构改了导致 shape 对不上。

**可运行例子:**
```python
import numpy as np

param = np.random.randn(3, 4).astype(np.float32)
grad_accum = np.zeros_like(param)

assert grad_accum.shape == param.shape
assert grad_accum.dtype == param.dtype
assert (grad_accum == 0).all()
```

**常见坑:** 它是"根据已有数组推形状",不是"根据数字推形状"——手上只有一个 shape 元组而没有现成数组时,该用普通的 `zeros`,不是 `zeros_like`。

---

## 6. `np.arange`

**签名:**
```python
np.arange(start, stop, step=1)
```
- `start`:起始值(包含)
- `stop`:结束值(**不包含**——最容易踩的坑)
- `step`:步长,默认 1,可以是负数(倒序)或小数

**一句话:** 生成一个等差数列,C 语言里 `for (i=start; i<stop; i+=step)` 循环变量的数组版。

**AI 研究场景:**
- **生成索引:** `np.arange(len(dataset))` 拿到数据集下标数组,配合 `np.random.shuffle` 打乱训练顺序。
- **位置编码(positional encoding):** Transformer 的位置编码公式里,`np.arange(seq_len)` 生成"第几个 token"这个序列,再代入 sin/cos 公式。
- **学习率调度:** 画学习率曲线、生成 epoch 序列做可视化,几乎都以 `np.arange` 开头。

**可运行例子:**
```python
import numpy as np

idx = np.arange(5)
assert list(idx) == [0, 1, 2, 3, 4]     # 5 个元素,不包含 5 本身

positions = np.arange(0, 10, 2)
assert list(positions) == [0, 2, 4, 6, 8]
```

**常见坑:** `stop` 不包含在结果里,和 Python 内置 `range()` 行为一致,但如果从"含头含尾"习惯的语言转过来容易多算/少算一个元素。涉及浮点数步长(比如 `step=0.1`)时,由于浮点误差,元素个数有时会和手算的差 1——需要精确控制点数时改用下面的 `np.linspace`。

---

## 7. `np.linspace`

**签名:**
```python
np.linspace(start, stop, num=50)
```
- `start`:起始值(包含)
- `stop`:结束值(**包含**——和 `arange` 相反!)
- `num`:一共生成多少个点(不是步长)

**一句话:** 在 `[start, stop]` 这个闭区间里,生成 `num` 个均匀分布的点。

**AI 研究场景:** 画图/画损失曲线的 x 轴、生成一组要测试的超参数(比如学习率在对数区间上均匀取点做 grid search)、可视化决策边界时生成网格坐标(配合下面第 15 节的 `meshgrid`)。

**可运行例子:**
```python
import numpy as np

x = np.linspace(0, 1, 5)
assert len(x) == 5
assert x[0] == 0.0
assert x[-1] == 1.0                      # 包含终点,这点和 arange 不同
assert np.allclose(x, [0, 0.25, 0.5, 0.75, 1.0])
```

**常见坑:** 和 `np.arange` 最容易搞混。**记忆窍门:`arange` 控制"步长"(第三个参数是 step),`linspace` 控制"要几个点"(第三个参数是 num)。** 想要"精确 N 个点"用 `linspace`,想要"精确的间隔"用 `arange`。

---

## 8. `np.eye` / `np.identity`

**签名:**
```python
np.eye(N, M=None, k=0)
np.identity(n)
```
- `np.eye`:`N` 行、`M` 列(不写 `M` 默认等于 `N`),`k` 可以让 1 出现在偏移的对角线上
- `np.identity(n)`:只能生成 `n×n` 的方阵,没有 `M`/`k` 参数,功能是 `eye` 的子集

**一句话:** 生成单位矩阵(对角线是 1,其余是 0)。

**AI 研究场景:**
- **线性代数验证:** 验证矩阵求逆是否正确的标准做法是 `assert np.allclose(A @ A_inv, np.eye(n))`——"矩阵乘自己的逆等于单位矩阵"是最常用的数值验证手段之一。
- **one-hot 编码:** `np.eye(num_classes)[label]` 是生成 one-hot 向量的经典技巧(取单位矩阵的第 `label` 行,该类别位置是 1,其余是 0)。
- **正则化:** 岭回归(ridge regression)等算法需要在矩阵上加 `λ·I` 防止不可逆,直接用 `np.eye`。

**可运行例子:**
```python
import numpy as np

I = np.eye(3)
assert I.shape == (3, 3)
assert np.trace(I) == 3                  # 对角线之和(迹)等于维度

# one-hot 编码技巧
num_classes = 4
label = 2
one_hot = np.eye(num_classes)[label]
assert list(one_hot) == [0.0, 0.0, 1.0, 0.0]

assert np.array_equal(np.eye(3), np.identity(3))   # 方阵情形下两者等价
```

**常见坑:** 默认生成 float64 的单位矩阵,如果 one-hot 之后要直接喂进要求 float32 的训练代码,记得加 `dtype=np.float32`。日常绝大多数场景 `eye` 和 `identity` 随便选,`eye` 更常用因为更灵活(支持非方阵和偏移对角线)。

---

## 9. `np.random.seed`

**签名:**
```python
np.random.seed(seed)
```

**一句话:** 固定随机数生成器的起始状态,让之后所有 random 调用都可复现。

**AI 研究场景:** 论文/实验代码的黄金法则——开头设 seed,否则每次跑出来的权重初始化、数据打乱顺序都不一样,复现不了自己或别人的结果,也没法调试(结果对不上的 bug,没法复现就没法修)。

**可运行例子:**
```python
import numpy as np

np.random.seed(123)
r1 = np.random.rand(3)

np.random.seed(123)
r2 = np.random.rand(3)

assert np.array_equal(r1, r2)            # 同样的 seed,结果完全一样
```

**常见坑:** seed 只保证"从这一刻起"的随机序列可复现——如果 seed 之后程序里还调用了其他会消耗随机状态的代码(哪怕你没注意到),后面的随机数就对不上了。想要每个实验片段互相独立、更精细可控,更稳妥的方式是第 09 批会讲的 `np.random.default_rng()`(新版 Generator API),但目前绝大多数教程和老代码依然用这个更简单的全局 seed 方式。

---

## 10. `np.random.randn`

**签名:**
```python
np.random.randn(*shape)
```
- 参数是**多个位置参数**(不是元组!),比如 `np.random.randn(3, 4)` 生成 `(3,4)` 的数组

**一句话:** 从标准正态分布 `N(0, 1)`(均值 0、标准差 1)采样,生成随机数组。

**AI 研究场景:** 神经网络权重不能初始化成 0(前面提到的对称性问题),标准做法是从正态分布随机采样。`randn` 是最基础的版本(实际训练框架会用更讲究的 Xavier/He 初始化,但都是在 `randn` 基础上乘了一个和层大小相关的缩放因子,见下一节)。

**可运行例子:**
```python
import numpy as np

np.random.seed(42)
w1 = np.random.randn(3, 3)

np.random.seed(42)
w2 = np.random.randn(3, 3)
assert np.array_equal(w1, w2)          # 同样的 seed,可复现

np.random.seed(0)
w3 = np.random.randn(3, 3)
assert not np.array_equal(w1, w3)      # 不同 seed,结果不同
```

**常见坑:** 参数是 `randn(3, 4)`,不是 `randn((3, 4))`——这点和 `zeros`/`ones` 正好相反(那两个要求形状是元组或单个整数)。传成 `randn((3, 4))` 会得到意外的结果,是 numpy API 里少数几个"形状参数格式不统一"的地方,只能记住这个例外。

---

## 11. `np.random.normal`(对比 `randn`)

**签名:**
```python
np.random.normal(loc=0.0, scale=1.0, size=None)
```
- `loc`:均值
- `scale`:**标准差**(不是方差!)
- `size`:形状,元组或整数

**一句话:** 从**任意均值和标准差**的正态分布采样;`randn` 固定是均值 0、标准差 1 的"标准"正态分布。

**AI 研究场景:** Xavier/He 初始化公式要求"标准差是某个和层大小相关的数",这时不能直接用 `randn`(标准差锁死是 1),要用 `normal(0, std)`,或者数学等价地 `randn(...) * std`——两种写法结果一样,但 `normal` 更直观地表达"我要的标准差是多少"。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
w = np.random.normal(loc=0.0, scale=0.01, size=(100, 100))
assert abs(w.mean()) < 0.01           # 均值应接近 0
assert abs(w.std() - 0.01) < 0.01     # 标准差应接近设定值 0.01

# 等价写法验证:normal(0, std) 数学上等于 randn() * std
np.random.seed(0)
w2 = np.random.randn(100, 100) * 0.01
assert np.allclose(w, w2)
```

**常见坑:** `scale` 参数是**标准差**不是方差——这是统计学里最容易搞混的一对术语。如果心里想的是"方差是 0.01"却直接传给 `scale`,标准差就会偏大(方差要开根号才是标准差)。

---

## 12. `np.random.uniform`

**签名:**
```python
np.random.uniform(low=0.0, high=1.0, size=None)
```
- `low`/`high`:采样区间 `[low, high)`
- `size`:形状,元组或整数(注意和 `randn` 的参数格式不同)

**一句话:** 从均匀分布(区间内每个值概率相等)采样,区别于 `randn`/`normal` 的正态分布(中间概率高、两头概率低)。

**AI 研究场景:** 早期部分初始化方案(如 LeCun 初始化)用均匀分布;超参数搜索里,学习率、dropout 概率这类"知道大致范围但不知道具体最优值"的参数,常用均匀采样(学习率一般先取 log 再均匀采样)。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
samples = np.random.uniform(low=-1.0, high=1.0, size=1000)

assert samples.min() >= -1.0
assert samples.max() < 1.0
assert abs(samples.mean()) < 0.1        # 均匀分布,均值应接近 (low+high)/2 = 0
```

**常见坑:** `size` 参数是元组或整数(`size=(3,4)`),而 `randn` 是拆开的位置参数(`randn(3,4)`)——这两个函数经常前后写,一不小心就把参数格式搞反。

---

## 13. `np.random.randint`

**签名:**
```python
np.random.randint(low, high=None, size=None)
```
- 区间 `[low, high)`(和 `arange` 一样,不含 `high`)

**一句话:** 生成随机**整数**。

**AI 研究场景:** 随机采样一个 batch 的下标(`np.random.randint(0, len(dataset), size=batch_size)`)、生成随机类别标签做测试数据、强化学习里随机选一个动作做探索(ε-greedy 的随机分支)。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
idx = np.random.randint(0, 10, size=5)

assert idx.shape == (5,)
assert idx.min() >= 0
assert idx.max() < 10                # 不包含 10
assert idx.dtype.kind == 'i'         # 整数类型,不是浮点
```

**常见坑:** 容易和 `random.uniform`(浮点、含糊区间)以及下面的 `random.choice`(可以从任意数组采样,不局限于连续整数)搞混——只在"我要的是连续整数区间"时才用 `randint`。

---

## 14. `np.random.choice`

**签名:**
```python
np.random.choice(a, size=None, replace=True, p=None)
```
- `a`:一个数组,或者一个整数(等价于 `range(a)`)
- `replace`:是否放回抽样(`True`=可重复,`False`=不重复)
- `p`:每个元素被抽中的概率(不写就是均匀概率)

**一句话:** 从给定数组里随机挑元素,可控制是否重复抽取、可控制每个元素的抽中概率。

**AI 研究场景:**
- **不重复子集采样:** 从数据集里随机挑一个子集做验证集,`replace=False` 保证不重复。
- **按概率采样:** 强化学习里"根据策略网络输出的概率分布选动作"、类别不均衡数据集里按自定义权重过采样少数类,都要用 `p` 参数。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
data = np.array([10, 20, 30, 40, 50])

subset = np.random.choice(data, size=3, replace=False)
assert len(subset) == len(set(subset))     # 不放回抽样,结果不应该有重复

biased = np.random.choice(data, size=1000, p=[0.9, 0.025, 0.025, 0.025, 0.025])
assert (biased == 10).mean() > 0.8          # 90% 概率的元素,抽1000次应占大多数(实测约89%)
```

**常见坑:** `replace=True`(默认,放回抽样)意味着结果**可能出现重复元素**。如果想要的是"打乱后取前 N 个不重复元素",要么显式传 `replace=False`,要么用下面第 09 批要讲的 `np.random.permutation`。

---

## 15. `np.meshgrid`

**签名:**
```python
np.meshgrid(*xi, indexing='xy')
```

**一句话:** 把几个一维坐标数组组合成对应维度的网格坐标矩阵——"给我 x 轴和 y 轴的刻度,输出整个网格上每个点的 (x,y) 坐标"。

**AI 研究场景:**
- **可视化决策边界:** 画分类模型的决策边界图,先用 `meshgrid` 铺出整个平面的网格点,把每个点喂进模型预测,再画出预测结果的等高线/颜色区域——这是可视化分类器的标准套路。
- **坐标特征:** 某些视觉模型(如 CoordConv)需要给每个像素位置附加 `(x,y)` 坐标作为额外特征,用 `meshgrid` 生成。

**可运行例子:**
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([10, 20])

X, Y = np.meshgrid(x, y)
assert X.shape == (2, 3)     # 形状是 (len(y), len(x)) —— 注意顺序!
assert Y.shape == (2, 3)

assert np.array_equal(X[0], x)        # X 的每一行都是原始的 x
assert np.array_equal(Y[:, 0], y)     # Y 的每一列都是原始的 y
```

**常见坑:** 输出形状是 `(len(y), len(x))` 而不是 `(len(x), len(y))`——"行对应 y、列对应 x"的顺序和直觉容易反,是 `meshgrid` 最容易踩的坑。如果这个顺序和你的假设不一致,画出来的图会横纵颠倒。

---

## 小结:这一批 15 个函数解决的问题

| 函数 | 解决的问题 |
|---|---|
| `array` | Python 数据 → numpy 数组 |
| `zeros`/`ones` | 占位/累加器/标准初始化 |
| `full` | 非 0/1 的统一填充(如 mask 用 -inf) |
| `empty` | 不清零的快速预分配(会被整体覆盖时用) |
| `zeros_like`/`ones_like` | 复用已有数组的 shape/dtype |
| `arange` | 等间隔序列(控制步长) |
| `linspace` | 等间隔序列(控制点数) |
| `eye`/`identity` | 单位矩阵、one-hot 编码 |
| `random.seed` | 固定随机状态,保证可复现 |
| `random.randn` | 标准正态分布随机初始化 |
| `random.normal` | 任意均值/标准差的正态分布(Xavier/He 初始化) |
| `random.uniform` | 均匀分布随机采样 |
| `random.randint` | 随机整数(采样下标) |
| `random.choice` | 从数组按概率/是否重复采样 |
| `meshgrid` | 网格坐标(可视化/坐标特征) |

下一批:[02-shape-and-structure.md](02-shape-and-structure.md) —— 形状与结构操作。

---

*更新:2026-07-07*
