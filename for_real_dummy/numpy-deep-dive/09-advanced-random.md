# 09 · 随机数进阶与可复现(Advanced Random & Reproducibility)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**怎么让"随机"这件事既好用又可控**——原地打乱和返回新数组的打乱有什么区别、为什么 numpy 官方现在推荐用一个独立的 `Generator` 对象而不是一份全局共享的随机状态、以及怎么用离散分布模拟 dropout、建模计数类事件。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

[01-creation-and-init.md](01-creation-and-init.md) 第 9–14 节已经讲过 `np.random.seed`/`randn`/`normal`/`uniform`/`randint`/`choice` 这些最基础的随机函数(生疏的话回去扫一眼),这一批不重复那些内容,直接从"打乱顺序"讲起,进阶到 numpy 官方现在推荐的新版随机数 API。

---

## 1. `np.random.shuffle`

**签名:**
```python
np.random.shuffle(x)
```
- `x`:要打乱的数组,**没有返回值**(函数签名里也看不出返回值,只能靠记住这个例外)
- 多维数组只沿第一个轴(`axis=0`)打乱——比如打乱一个 `(样本数, 特征数)` 的矩阵,打乱的是"样本的顺序",每个样本内部的特征不会被拆散重排

**一句话:** 把数组"原地"(in-place)打乱顺序——直接修改传进去的数组本身,不返回新数组。

**AI 研究场景:** 打乱训练数据顺序,防止模型学到数据的排列规律(比如数据集是按类别顺序排好的,不打乱直接切 batch,可能出现"一个 batch 全是同一类"的情况,训练会不稳定甚至学不动)。更常见的用法其实不是直接打乱数据本身,而是打乱一份下标数组 `np.arange(len(dataset))`,这样可以用同一份打乱后的下标去同步索引 `X` 和 `y`——这个模式在第 6 节会具体写成一个可复用的函数。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
idx = np.arange(10)
ret = np.random.shuffle(idx)

assert ret is None                              # 没有返回值,这是"原地"的直接证据
assert sorted(idx.tolist()) == list(range(10))  # 元素集合没变,只是顺序变了

# 多维数组:只打乱第一个轴(把每一"行"当成一个整体样本来打乱,不拆散行内部的元素)
np.random.seed(0)
M = np.arange(12).reshape(4, 3)   # 假装是 4 个样本、每个样本 3 个特征
np.random.shuffle(M)

assert M.shape == (4, 3)
assert set(M.flatten().tolist()) == set(range(12))   # 元素集合不变
assert sorted(M[:, 0].tolist()) == [0, 3, 6, 9]        # 每个样本内部的 3 个特征仍然绑在一起,没被拆散
```

**常见坑:** 最大的坑是"以为它和大多数 numpy 函数一样会返回新数组"——`b = np.random.shuffle(a)` 里的 `b` 永远是 `None`,真正被打乱的是 `a` 自己。如果 `a` 是从别处传进来的、被调用者和其他代码共享的数组,`shuffle` 会悄悄修改调用者手里的原始数据,这是很隐蔽的"副作用" bug。**不确定安不安全,就用下一节的 `permutation`。**

---

## 2. `np.random.permutation`

**签名:**
```python
np.random.permutation(x)
```
- `x`:一个数组,或者一个整数 `n`(整数时等价于打乱 `np.arange(n)`)
- **返回一个打乱后的新数组**,原数组不受影响

**一句话:** 和 `shuffle` 做同一件事(打乱顺序),但返回一个新数组,不改动传入的原始数据。

**AI 研究场景:** 这是比 `shuffle` **更常用、更安全**的选择,尤其是需要用"同一份打乱顺序"去同步索引多个数组的时候(比如 `X` 和 `y`):生成一份打乱后的下标数组 `order = np.random.permutation(len(X))`,再用 `X[order]`、`y[order]` 分别取值——原始 `X`/`y` 保持不动,方便对照调试,也不会因为某个函数内部意外提前打乱了共享数据而导致下游代码顺序对不上。第 6 节的 train/test split 实战就是靠它实现的。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
a = np.arange(10)
b = np.random.permutation(a)

assert a is not b
assert list(a) == list(range(10))               # 原数组 a 完全没变
assert sorted(b.tolist()) == list(range(10))     # b 是打乱后的新数组

# 直接传整数:等价于 permutation(np.arange(n))
np.random.seed(0)
order = np.random.permutation(10)
assert np.array_equal(order, b)                  # 同样的 seed,结果和上面完全一致

# 典型用法:用同一份打乱下标,同步打乱 X 和 y,保持"样本-标签"对应关系
X = np.array([[1], [2], [3], [4]])
y = np.array(['a', 'b', 'c', 'd'])
order2 = np.random.permutation(4)
X_shuf, y_shuf = X[order2], y[order2]

mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
assert X_shuf[0, 0] == mapping[y_shuf[0]]        # 打乱后 X/y 依然一一对应,没有错位
```

**常见坑:** 数据量很大时,`permutation` 需要额外分配一份新数组的内存,而 `shuffle` 原地操作不需要——如果确定不需要保留原顺序、也确定没有其他代码共享这份数组,`shuffle` 会更省内存。但**默认优先选 `permutation`**,只有在明确追求极致内存/性能且理解了原地修改风险时才换成 `shuffle`,这是"安全优先,性能其次"的稳妥顺序。

---

## 3. `np.random.default_rng()`

**签名:**
```python
np.random.default_rng(seed=None)
```
- `seed`:整数(推荐显式传,保证可复现);不传的话用系统熵源自动生成,每次运行结果都不同
- 返回一个 `Generator` **对象**——之后所有随机操作都调用这个对象的方法(`rng.random()`、`rng.integers()` ……),而不是 `np.random.xxx()`

**一句话:** numpy 官方现在推荐的"新版"随机数 API 入口——创建一个**独立的**随机数生成器对象,取代"整个进程共享同一份状态"的老方式(`np.random.seed(...)` 之后接 `np.random.xxx(...)`)。

**AI 研究场景:** 为什么要迁移?核心原因是**老方式的全局状态会互相干扰**。[01-creation-and-init.md](01-creation-and-init.md) 第 9 节讲过的 `np.random.seed`,设置的是**整个 numpy 进程只有一份、大家共用**的状态——你的数据加载代码、模型初始化代码、甚至某个第三方库内部,只要都调用了 `np.random.xxx()`,读写的就是同一份状态,谁先调用、调用了几次,都会影响后面其他人拿到的随机数。具体会踩坑的场景:

- **多个实验脚本/模块共用一个进程:** 你和同事的代码都写了 `np.random.seed(42)`,如果这两段代码在同一个 Python 进程里先后运行(比如一个脚本导入了两个各自"以为独占随机状态"的模块),后调用的一方的"可复现"就被前面的调用悄悄打乱了。
- **多进程并行跑实验/多个 DataLoader worker:** 每个进程/worker 需要一份独立、不重复的随机流,共用一份全局状态没法干净隔离。
- **同一份代码里需要几路互不干扰的随机数:** 比如"打乱数据用的随机源"和"模型权重初始化用的随机源"想分开管理,老的全局状态没法区分这两者,改一处可能牵连另一处。

`default_rng()` 生成的每个 `Generator` 对象带着**自己独立的一份内部状态**,想要几路独立随机流就创建几个对象,互不影响——这是目前 numpy 官方文档推荐的写法(老的 `np.random.seed` 全局方式仍然保留、大量教程和旧代码还在用,官方称之为 legacy,不是不能用,但新代码建议迁移)。

**可运行例子:**
```python
import numpy as np

rng1 = np.random.default_rng(123)
rng2 = np.random.default_rng(123)

x1 = rng1.random(3)
_ = rng2.random(1000)      # 模拟"另一个实验/模块"疯狂消耗 rng2 的随机状态
x2 = rng1.random(3)         # rng1 完全不受 rng2 影响,继续自己独立的序列

rng1_replay = np.random.default_rng(123)   # 用相同 seed 重新创建一个,验证序列是否可复现
y1 = rng1_replay.random(3)
y2 = rng1_replay.random(3)

assert np.array_equal(x1, y1)
assert np.array_equal(x2, y2)               # rng1 的结果和"从头重放"完全一致,没被 rng2 的调用污染
```

**常见坑:** 别把 `np.random.default_rng(seed)` 和 `np.random.seed(seed)` 弄混——前者返回一个**对象**,要用 `rng.random()` 这种方法调用方式;后者是设置全局状态,之后还是用 `np.random.xxx()` 调用。**一份代码里选定一套 API,不要混用**(又调一次 `np.random.seed()`、又建一个 `default_rng()`),否则代码里同时存在两份互不感知的随机状态,反而更难保证复现性。

---

## 4. `rng.random()` / `rng.integers()`

**签名:**
```python
rng.random(size=None)
rng.integers(low, high=None, size=None, endpoint=False)
```
- `rng.random`:采样 `[0, 1)` 区间的均匀分布浮点数,`size` 是形状(元组或整数)
- `rng.integers`:采样整数,默认区间 `[low, high)`**不含** `high`(和老的 `randint` 一样);传 `endpoint=True` 则区间变成**含** `high`

**一句话:** 新 Generator API 下最常用的两个基础采样方法——`rng.random()` 对应老 API 的 `np.random.rand()`(采样 `[0,1)` 均匀浮点数),`rng.integers()` 对应老 API 的 `np.random.randint()`(采样整数)。

**AI 研究场景:** 现在看新一点的 numpy 代码或官方文档示例,几乎都是 `rng = np.random.default_rng(seed)` 开头,后面全是 `rng.xxx()`——如果心里没有"这个新方法对应老 API 的哪个",看新代码会觉得眼生、甚至怀疑是不是换了一套完全不同的功能。这一节就是建立这张对应表,让你能"秒懂"新代码里的调用,不用每次现查文档:

| 老 API(全局状态) | 新 API(`Generator` 对象方法) | 采样内容 |
|---|---|---|
| `np.random.rand(d0, d1, ...)` | `rng.random(size)` | `[0, 1)` 均匀浮点数 |
| `np.random.randint(low, high, size)` | `rng.integers(low, high, size)` | 整数 |
| `np.random.randn(*shape)` / `np.random.normal(loc, scale, size)` | `rng.standard_normal(size)` / `rng.normal(loc, scale, size)` | 正态分布 |
| `np.random.uniform(low, high, size)` | `rng.uniform(low, high, size)` | 任意区间均匀分布 |
| `np.random.choice(a, size, replace, p)` | `rng.choice(a, size, replace, p)` | 按概率/是否重复采样 |

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(0)

# rng.random 对应老 API 的 rand:[0,1) 均匀浮点数,参数是 size(元组或整数)
r = rng.random((2, 3))
assert r.shape == (2, 3)
assert (r >= 0).all() and (r < 1).all()

# rng.integers 对应老 API 的 randint:默认区间 [low, high) 不含 high
i = rng.integers(0, 10, size=5)
assert i.shape == (5,)
assert i.min() >= 0 and i.max() < 10

# 新 API 多了老 API 没有的选项:endpoint=True 把区间变成含 high
i_inclusive = rng.integers(0, 10, size=1000, endpoint=True)
assert i_inclusive.max() == 10          # 老的 randint 想达到"含10"必须写成 randint(0, 11)
```

**常见坑:** `rng.random()` 的参数是 `size=(2, 3)` 这种**元组风格**,不是 `np.random.rand(2, 3)` 那种拆开写的位置参数风格——虽然两者采样的是同一个 `[0,1)` 均匀分布,但传参写法必须按新 API 的元组风格来,写成 `rng.random(2, 3)` 会直接报错(numpy 会把第二个位置参数 `3` 当成 `dtype` 去解析,报 `TypeError: Cannot interpret '3' as a data type`,亲测踩过这个错误信息)。

---

## 5. `np.random.binomial` / `np.random.poisson`

**签名:**
```python
rng.binomial(n, p, size=None)
rng.poisson(lam=1.0, size=None)
```
- `binomial`:`n` 次独立试验,每次成功概率 `p`,返回"成功次数"(整数);`n=1` 时就是"单次伯努利试验"(结果要么 0 要么 1)
- `poisson`:`lam`(λ,读作 lambda)是"平均发生率"——描述"已知平均发生次数,一段时间/一个区域内实际发生了几次"这类计数随机事件

(老 API 对应 `np.random.binomial(n, p, size)`、`np.random.poisson(lam, size)`,参数含义和新 API 的 `rng.binomial`/`rng.poisson` 完全一样,只是挂在全局状态还是 `rng` 对象上的区别——上一节讲过的对应关系在这里同样适用。)

**一句话:** `binomial` 模拟"做 n 次抛硬币类的独立试验,统计成功几次";`poisson` 模拟"给定平均发生率,统计一段时间/空间里实际发生几次"这类计数事件。两者的结果都是**非负整数(计数)**,不是连续值,这点上和 `randn`/`uniform` 这些连续分布不同。背后的概率论证明不必深究,记住"什么时候用"就够了。

**AI 研究场景:**
- **dropout 的本质就是 binomial 采样:** 训练时随机"丢弃"一部分神经元/连接,每个神经元独立地按保留概率 `p` 决定这一轮是否参与计算——这正是 `n=1` 的 binomial 采样生成一个 0/1 mask,再和激活值逐元素相乘。深度学习框架内部的 dropout 层做的就是这件事,只是包装成了一个 layer,调用时你看不到底层这次采样。
- **poisson 建模计数类数据/事件到达:** 很多"单位时间/空间内发生几次"的现象天然是计数分布,比如模拟一个排队/流量类的强化学习环境里"每个时间步到达几个请求"、统计"每个区域出现几次点击/异常事件"这类数据,用 `poisson` 采样比强行套连续分布更符合"结果只能是非负整数"的性质。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(0)

# ---- binomial:dropout 的本质 ----
activations = np.ones(10)          # 假设是某一层 10 个神经元的输出(简化成全 1,方便看效果)
keep_prob = 0.7
mask = rng.binomial(n=1, p=keep_prob, size=activations.shape)   # n=1:每个神经元独立抛一次"保留/丢弃"

assert set(np.unique(mask)).issubset({0, 1})                     # 只有 0 和 1 两种取值
dropped = activations * mask / keep_prob                          # inverted dropout:按保留概率放大剩下的值,推理时不用再缩放
assert np.array_equal(dropped[mask == 0], np.zeros((mask == 0).sum()))   # 丢弃位置精确为 0

# n>1 的场景:20 次独立试验,每次成功率 0.5,统计成功几次
successes = rng.binomial(n=20, p=0.5, size=5)
assert (successes >= 0).all() and (successes <= 20).all()

# ---- poisson:计数事件建模 ----
lam = 3.0                                    # 平均每个时间步到达 3 个请求
arrivals = rng.poisson(lam=lam, size=100_000)
assert arrivals.dtype.kind == 'i'            # 计数结果一定是整数,不会出现 2.5 个请求
assert abs(arrivals.mean() - lam) < 0.05     # 采样次数够多,均值应接近设定的 lam(泊松分布性质:均值≈方差≈lam)
```

**常见坑:**
- `p`(binomial)是"概率",必须落在 `[0, 1]`;`lam`(poisson)是"平均发生次数",可以是任意非负数(不是概率,完全可以大于 1,比如平均每步到达 3 个请求就是 `lam=3`)——两个参数含义不同,不要混着套用。
- dropout 场景里容易漏掉 `/ keep_prob` 这一步缩放(inverted dropout 的标准做法);训练时丢了、推理时又没做对应缩放,两个阶段的期望输出量级会对不上。
- 两者的返回值都是整数计数,如果后续代码期待的是概率或连续值,不要直接当连续型数据处理(比如直接对计数值套用假设正态分布的统计检验就不合适)。

---

## 6. 训练/验证集划分实践

前面 5 节讲的都是"怎么生成随机性",这一节把它们串起来解决一个几乎每个训练脚本都要写的实际问题:**把数据集随机切成训练集和验证集**,并且验证切分本身是对的(没有重叠、比例正确)。这是这一批内容的实战收尾。

**手写实现:**
```python
import numpy as np

def train_test_split(n_samples, test_ratio=0.2, rng=None):
    """返回 (train_idx, test_idx) 两个不重叠的下标数组。"""
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n_samples)   # 用 permutation 而不是 shuffle:不改动任何已有数组,直接拿到一份新的打乱下标
    n_test = int(n_samples * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx
```

为什么这里用 `permutation` 而不是 `shuffle`(第 1、2 节的区别在这里正好用上):这个函数只需要"一份打乱后的下标",不需要改动调用者传进来的任何数据——如果用 `shuffle`,得先手动造一个 `np.arange(n_samples)` 数组再原地打乱它,多一步而且容易和"是不是在打乱原始数据"搞混;直接 `permutation(n_samples)` 一步到位,更清楚也更安全。

**验证:划分后两部分没有重叠、比例正确、样本与标签依然一一对应**
```python
import numpy as np

def train_test_split(n_samples, test_ratio=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx

rng = np.random.default_rng(42)          # 用第 3 节的新 Generator API,固定 seed 让这份划分可复现
X = rng.random((100, 4))                  # 假数据集:100 个样本,4 个特征
y = rng.integers(0, 2, size=100)          # 假标签:二分类

train_idx, test_idx = train_test_split(len(X), test_ratio=0.2, rng=rng)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 1. 比例正确
assert len(test_idx) == 20
assert len(train_idx) == 80

# 2. 没有重叠:两组下标的交集必须是空集
assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()

# 3. 覆盖完整且各自内部无重复:并集等于全部下标,两边各自都没有重复值
assert set(train_idx.tolist()) | set(test_idx.tolist()) == set(range(100))
assert len(set(train_idx.tolist())) == len(train_idx)
assert len(set(test_idx.tolist())) == len(test_idx)

# 4. X 和 y 用同一组下标切分,样本-标签的对应关系没有错位
assert X_train.shape == (80, 4) and X_test.shape == (20, 4)
assert y_train.shape == (80,) and y_test.shape == (20,)
```

**常见坑:**
- 最容易犯的错是**分别对 `X` 和 `y` 各自调用一次随机打乱/划分**——两次随机结果不一样,`X` 和 `y` 就错位了(样本和标签对不上,模型实际在学习噪声,而且这种 bug 不会报错,只会让效果莫名其妙变差,非常难查)。正确做法永远是:**只生成一次打乱下标,`X` 和 `y` 都用这同一份下标去切。**
- 需要多次实验对比,或者做 K 折交叉验证时,要想清楚这里的 `rng` 是要固定 seed(保证每次都复现同一份划分)还是要显式换 seed(保证多次实验拿到不同的划分)——两者不能兼得,需要根据实验目的显式决定,不要"跑起来发现对了就行"。
- 真实项目里更推荐直接用 `scikit-learn` 的 `train_test_split`(内置按标签分层抽样 `stratify=`,以及更多边界情况的处理);这里手写是为了吃透 `permutation` 的实际用法,不是提倡在生产代码里重复造轮子。

---

## 小结:这一批 6 节解决的问题

| 函数/方法 | 解决的问题 |
|---|---|
| `random.shuffle` | 原地打乱数组(无返回值,直接修改原数组) |
| `random.permutation` | 打乱并返回新数组(不改原数组,更常用更安全) |
| `random.default_rng()` | 创建独立的 `Generator` 对象,替代全局共享的随机状态,避免多实验/多进程互相干扰 |
| `rng.random` / `rng.integers` | 新 API 下的基础采样,对应老 API 的 `rand`/`randint` |
| `random.binomial` / `random.poisson` | 离散计数分布——dropout mask(binomial)、事件计数建模(poisson) |
| 训练/验证集划分 | 用 `permutation` 手写 `train_test_split`,assert 验证无重叠、比例正确,串联本批全部内容 |

下一批:[10-io-and-verification.md](10-io-and-verification.md) —— IO 与验证工具。

---

*更新:2026-07-07*
