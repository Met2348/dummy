# 05 · 归约与统计(Reduction & Statistics)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**怎么把一个数组"压扁"成更少的数字,提炼出有用的统计量**——把一个 batch 的 loss 压成一个标量、把 logits 压成预测类别、算一份数据的均值方差做归一化,都是"归约"。这一批也是新手在 `axis` 参数上最容易"方向感走反"的地方,第1节会用一个具体二维数组把方向彻底讲清楚,后面所有函数都复用同一套方向规则。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `np.sum`

**签名:**
```python
np.sum(a, axis=None, keepdims=False)
```
- `a`:输入数组
- `axis`:沿哪个维度"压扁"求和;不写(默认 `None`)就是把所有元素加成一个标量
- `keepdims`:压扁之后,要不要把那个维度保留成"长度为1"而不是直接消失——下面详细讲

**一句话:** 把数组沿某个方向"压扁"成加总的结果——这是这一整批函数里最基础、也必须先吃透的一个,因为 `axis` 的方向感在 `mean`/`max`/`std`... 几乎所有归约函数里都通用,不会重复讲。

**AI 研究场景:** 训练循环里每个样本算完 loss 之后,要把整个 batch 的 loss "压成一个数"才能 `.backward()`——这就是损失函数 `reduction='sum'` 的底层实现(`reduction='sum'` 就是 `losses.sum()`,下一节的 `mean` 对应 `reduction='mean'`)。此外,统计一个 mask 里有多少个 1、一个矩阵每一行的总能量,都是 `sum` 沿不同 `axis` 的应用。

**可运行例子:**
```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
assert a.shape == (2, 3)          # 2行3列 -> axis=0对应"行"(长度2),axis=1对应"列"(长度3)

assert a.sum() == 21              # 不写axis:全部元素加成一个标量

col_sum = a.sum(axis=0)           # 压扁axis=0(长度2的"行"维度)
assert list(col_sum) == [5, 7, 9] # [1+4, 2+5, 3+6] 跨行相加,按列给结果,结果个数=列数
assert col_sum.shape == (3,)

row_sum = a.sum(axis=1)           # 压扁axis=1(长度3的"列"维度)
assert list(row_sum) == [6, 15]   # [1+2+3, 4+5+6] 跨列相加,按行给结果,结果个数=行数
assert row_sum.shape == (2,)

# keepdims:压扁之后要不要把该维度保留成长度1,方便和原数组广播
row_sum_kd = a.sum(axis=1, keepdims=True)
assert row_sum_kd.shape == (2, 1)

normalized = a / row_sum_kd            # 把每一行归一化成"和为1"(概率/attention权重常见操作)
assert normalized.shape == (2, 3)
assert np.allclose(normalized.sum(axis=1), [1.0, 1.0])

# 漏掉keepdims,广播直接报错(而不是安静地算出错误结果)
raised = False
try:
    a / a.sum(axis=1)          # a.sum(axis=1).shape==(2,),没法跟(2,3)广播
except ValueError:
    raised = True
assert raised

# 但axis=0的情况不用keepdims也能广播成功,因为压扁后的shape (3,) 正好和原数组最后一维对齐
also_ok = a / a.sum(axis=0)
assert also_ok.shape == (2, 3)
```

**方向到底怎么记,核心规则只有一条:`sum(axis=k)` 会把 `a.shape` 里第 `k` 个数字"删掉",剩下的形状就是结果的形状。**

| 操作 | 压扁(消失)的维度 | 结果 shape | 直觉 |
|---|---|---|---|
| `a.sum()` | 全部维度 | `()` 标量 | 所有元素加在一起 |
| `a.sum(axis=0)` | 第0维,长度2("行") | `(3,)` | 跨行相加,一列一个结果 |
| `a.sum(axis=1)` | 第1维,长度3("列") | `(2,)` | 跨列相加,一行一个结果 |

**常见坑 1(重灾区):`axis=0` 不是"沿着行方向处理",而是"把行这个维度消掉"。** 很多新手看到 `axis=0` 第一反应是"逐行处理",于是猜 `a.sum(axis=0)` 会得到"每一行的和"——这是反的!"每一行的和"其实是 `axis=1` 的结果(`[6, 15]`,长度等于行数)。`axis=0` 消掉的是"有几行"这件事,所以得到的是"每一列的和"(`[5, 7, 9]`,长度等于列数)。不确定的时候别猜,现场 `print(结果.shape)` 对一下,或者把上面这张表背下来对照。

**常见坑 2:`keepdims=True` 决定"压扁的维度是彻底消失,还是保留成长度1"——这在后续要做广播运算时是刚需。** 上面例子把每一行归一化成"和为1"就是典型场景:`a.sum(axis=1)` 得到 shape `(2,)`,直接拿去和原始的 `(2,3)` 做除法会触发 `ValueError: operands could not be broadcast together with shapes (2,3) (2,)`(实测报错信息);必须用 `keepdims=True` 让它变成 `(2,1)` 才能正确广播。更精细的一点是:**`keepdims` 是不是必须加,取决于被压扁的维度是不是"最后一维"。** 广播规则是从右往左对齐(见 [01-numpy-for-c-programmers.md](../01-numpy-for-c-programmers.md) 第5节),`axis=1` 压掉的正好是最后一维,结果 `(2,)` 没法和原数组最后一维的"3"对齐,所以报错;而 `axis=0` 压扁后的 `(3,)` 恰好和原数组最后一维的"3"对上了,不加 `keepdims` 也能广播成功。为了不用每次都心算这个,**新手阶段"只要归约完还要跟原数组广播运算,就无脑加 `keepdims=True`",更省心,唯一代价是多保留一个没必要的维度,可读性上反而更清楚。**

---

## 2. `np.mean`

**签名:**
```python
np.mean(a, axis=None, keepdims=False)
```
- `axis`/`keepdims` 的方向和作用跟上一节 `sum` 完全一致——"哪个维度消失、要不要保留成1"是同一套规则,后面所有归约函数都通用,不再重复画表。

**一句话:** 沿某个方向求平均值,等价于 `sum(axis) / 该维度的元素个数`。

**AI 研究场景:** 这是 `reduction='mean'` 的底层实现,深度学习里比 `reduction='sum'` 更常用的默认选项——因为 `mean` 让 loss 的数值大小不随 batch size 变化(batch 从32改成256,`sum` 会让loss数值直接变8倍,`mean` 不会),更适合和固定的学习率配合。归一化数据时"减均值"(`x - x.mean(axis=0)`)也是这个函数。

**可运行例子:**
```python
import numpy as np

per_sample_loss = np.array([0.5, 0.3, 0.9, 0.1])   # 一个batch里4个样本各自的loss

loss_sum  = per_sample_loss.sum()     # reduction='sum'
loss_mean = per_sample_loss.mean()    # reduction='mean'
assert np.allclose(loss_sum, 1.8)
assert np.allclose(loss_mean, 0.45)
assert np.allclose(loss_mean, loss_sum / len(per_sample_loss))

b = np.array([[1., 2., 3.],
              [4., 5., 6.]])
assert np.allclose(b.mean(axis=0), [2.5, 3.5, 4.5])   # 按列平均,方向规则和sum一致
assert np.allclose(b.mean(axis=1), [2.0, 5.0])         # 按行平均
```

**常见坑:** 同一个 batch,`sum` 和 `mean` 数值差一个"元素个数"的倍数——如果代码里混用了两种 reduction(比如自己实现的一部分 loss 用 `sum`,另一部分用框架默认的 `mean`),训练出来的有效学习率会因为这个倍数差异而不一致,是一个隐蔽但很常见的坑。看到别人的 loss 数值和自己的数量级对不上,先检查是不是这里的差异。

---

## 3. `np.std` / `np.var`

**签名:**
```python
np.std(a, axis=None, ddof=0, keepdims=False)
np.var(a, axis=None, ddof=0, keepdims=False)
```
- `ddof`("Delta Degrees of Freedom"):分母从"元素个数 N"改成"N - ddof"。默认 `ddof=0`
- `axis`/`keepdims` 规则同第1节

**一句话:** `var` 是"每个元素离均值的平方差"的平均值(方差);`std` 是 `var` 开根号(标准差),单位和原始数据一致,更好解读。

**AI 研究场景:** Batch Normalization 的核心运算就是 `(x - x.mean(axis)) / x.std(axis)`——把每一层的激活值强制拉回"均值0、标准差1"的分布,是训练深层网络的关键技术之一;数据预处理阶段的标准化(z-score normalization)也是同一个公式,直接影响特征是否在同一数量级、进而影响收敛速度。

**可运行例子:**
```python
import numpy as np

# 经典统计学教材例子
data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
assert data.mean() == 5.0

pop_std = data.std()              # 默认 ddof=0 -> 总体标准差
sample_std = data.std(ddof=1)     # ddof=1 -> 样本标准差
assert np.allclose(pop_std, 2.0)
assert np.allclose(sample_std, np.sqrt(32 / 7))   # 约2.138,和pop_std明显不同

# batch normalization实战:强制拉回均值0、标准差1
rng = np.random.default_rng(0)
batch = rng.normal(loc=3.0, scale=2.0, size=(100, 4))   # 100个样本,4个特征
bn = (batch - batch.mean(axis=0)) / batch.std(axis=0)
assert np.allclose(bn.mean(axis=0), 0.0, atol=1e-10)
assert np.allclose(bn.std(axis=0), 1.0, atol=1e-10)
```

**常见坑(默认值不一致型的坑,容易踩):`np.std`/`np.var` 默认 `ddof=0`,算的是*总体标准差*(除以N),不是很多统计课本/其他工具默认的*样本标准差*(除以N-1,需要显式传 `ddof=1`)。** 上面例子里同一份数据,`ddof=0` 算出来标准差精确是 `2.0`,`ddof=1` 算出来约是 `2.138`——两个数字都"对",区别只是分母用 N 还是 N-1,但如果你是从 Excel/pandas(`pandas.DataFrame.std()` 默认就是 `ddof=1`!)或统计课本迁移过来,不留意这个默认值差异,算出来的标准差会"莫名"对不上,还以为自己代码写错了。**深度学习场景(比如 BatchNorm)几乎总是用 numpy 默认的 `ddof=0`;做统计推断/小样本估计时通常需要 `ddof=1`,视场景选择,别无脑套默认值。**

---

## 4. `np.max` / `np.min`

**签名:**
```python
np.max(a, axis=None, keepdims=False)
np.min(a, axis=None, keepdims=False)
```
- `axis`/`keepdims` 规则同第1节

**一句话:** 沿某个方向取最大值/最小值(只要值,不要位置——位置要用下一节的 `argmax`/`argmin`)。

**AI 研究场景:** 数值稳定性技巧——实现 softmax 时先减去当前这一行的最大值再取 `exp`,防止 `exp(很大的数)` 数值溢出(数学上 `softmax(x) == softmax(x - max(x))`,减完最大值后指数运算最大也就是 `exp(0)=1`,不会溢出);梯度裁剪(gradient clipping)前先看一眼梯度的最大绝对值来决定裁剪阈值;检查数据集里像素值/特征值的取值范围也常用到。

**可运行例子:**
```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
shifted = logits - logits.max()          # softmax数值稳定性技巧
sm = np.exp(shifted) / np.exp(shifted).sum()
assert np.allclose(sm.sum(), 1.0)
assert np.allclose(sm, [0.6590, 0.2424, 0.0986], atol=1e-3)

c = np.array([[1, 9, 3],
              [4, 5, 6]])
assert list(c.max(axis=0)) == [4, 9, 6]     # 按列取最大,方向规则同sum
assert list(c.min(axis=1)) == [1, 4]        # 按行取最小

# np.max是"归约"(数组变少),不要和np.maximum(逐元素二元比较)搞混
d = np.array([1, 5, 3])
e = np.array([4, 2, 6])
assert list(np.maximum(d, e)) == [4, 5, 6]   # 逐点取较大者,不是归约,形状不变
```

**常见坑:** `np.max(a)` 是"归约"(一个数组变少变小),不要和 `np.maximum(a, b)` 搞混——后者是"逐元素二元比较"(两个同形状数组,对应位置比大小,结果形状不变),属于第 04 批[逐元素运算](04-elementwise-math.md)的范畴。两个名字长得像(`max` vs `maximum`),但一个是"压扁数组求极值",一个是"两个数组逐点比较",经常被搞混。另外,文档里有时会看到 `np.amax` 这个名字,和 `np.max` 结果完全等价(历史遗留的两套命名),新代码统一写 `np.max` 即可。

---

## 5. `np.argmax` / `np.argmin`

**签名:**
```python
np.argmax(a, axis=None)
np.argmin(a, axis=None)
```
- 返回的是**下标(位置)**,不是值本身
- 不写 `axis`:先把数组摊平成一维,再返回"摊平之后"的下标——容易踩坑,见下文

**一句话:** 找最大值/最小值"在哪个位置",而不是"值是多少"。

**AI 研究场景:** 分类模型的标准做法——网络最后一层输出的是每个类别的分数(logits/概率),真正的"预测类别"是这些分数里最大值所在的下标:`pred = np.argmax(logits, axis=1)`(每个样本一行,沿着"类别"这个方向找最大值的位置)。`argmin` 常见于最近邻查找,比如 k-means 里把每个点分配给"距离最近的簇中心",就是对距离数组取 `argmin`。

**可运行例子:**
```python
import numpy as np

logits_batch = np.array([[0.1, 0.7, 0.2],    # 样本1:类别1得分最高
                          [0.6, 0.1, 0.3]])   # 样本2:类别0得分最高
preds = np.argmax(logits_batch, axis=1)       # 沿"类别"方向(axis=1)找最大值位置
assert list(preds) == [1, 0]

# 不写axis:先摊平成一维,再找下标
flat = np.array([[1, 5], [5, 2]])             # 摊平后是[1,5,5,2]
assert flat.argmax() == 1                     # 第一个最大值(5)在摊平后的下标1
idx = np.unravel_index(flat.argmax(), flat.shape)   # 把摊平下标换算回(行,列)
assert idx == (0, 1)

# 并列最大值时,永远返回"最先遇到的"下标
tie = np.array([3, 1, 3, 0])
assert tie.argmax() == 0     # 下标0和下标2都是最大值3,返回靠前的0
```

**常见坑:** 两个连着的坑。(1)不传 `axis` 时,`argmax` 会先把多维数组摊平(flatten)成一维再找下标,返回的是"摊平之后的下标",如果还想知道它在原始多维数组里的"行、列"坐标,要配合 `np.unravel_index(下标, 原始shape)` 换算,直接拿这个下标当行号/列号用会用错。(2)出现并列最大值时,`argmax`/`argmin` 保证返回"最靠前"的那个下标(不是随机也不是最后一个),依赖这个确定性行为写测试是安全的,但如果你的直觉以为它会随机选一个,预期就会落空。

---

## 6. `np.median`

**签名:**
```python
np.median(a, axis=None, keepdims=False)
```

**一句话:** 把数据排序后取"正中间"的值(50% 分位数),偶数个元素时取中间两个的平均。

**AI 研究场景:** 报告实验指标时,如果数据里有离群点干扰(比如推理延迟里偶尔有一次卡顿导致的极端值、强化学习里偶尔一局出现的极端回报),`mean` 会被这个离群点带偏,`median` 对离群点不敏感,能更真实反映"典型情况"。技术报告里常同时给出 mean 和 median(或者下一节的 p50/p90/p99)。

**可运行例子:**
```python
import numpy as np

assert np.median(np.array([3, 1, 2])) == 2.0          # 奇数个:正中间那个值
assert np.median(np.array([1, 2, 3, 4])) == 2.5        # 偶数个:中间两个(2和3)的平均

# 离群点场景:5次推理延迟(ms),其中一次卡顿
latencies = np.array([10, 11, 9, 10, 500])
assert np.median(latencies) == 10.0     # 中位数几乎不受500这个离群点影响
assert latencies.mean() == 108.0        # 均值被500严重拉高,不能代表"典型延迟"
```

**常见坑:** 偶数个元素时,中位数是"中间两个值的平均",**这个结果可能根本不是原始数据里真实存在的一个数**(比如上面的 `2.5`,原数组里并没有 2.5 这个元素)——如果需要的是"数据集里真实存在、排在中间的那个样本"(而不是插值出来的数),median 不一定满足要求,得自己排序后用下标去取。

---

## 7. `np.percentile` / `np.quantile`

**签名:**
```python
np.percentile(a, q, axis=None)   # q 取值范围 [0, 100]
np.quantile(a, q, axis=None)     # q 取值范围 [0, 1]
```

**一句话:** 都是"分位数"——"有 q% / q 比例的数据小于等于这个值";两者是同一个概念的两种刻度,`percentile` 用百分制(0~100),`quantile` 用小数制(0~1),换算关系就是 `percentile(a, x) == quantile(a, x/100)`。

**AI 研究场景:** 线上服务的延迟报告标准写法是 p50/p90/p99(50%/90%/99% 分位数)——p99 直接反映"最慢的1%用户体验有多差",比单纯看均值更能暴露长尾问题;数据预处理时用 1%/99% 分位数做"截断式"归一化(把超出这个范围的极端值裁剪掉),比直接用 min/max 更抗离群点干扰。

**可运行例子:**
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

p50 = np.percentile(arr, 50)     # 百分制:50
q50 = np.quantile(arr, 0.5)      # 小数制:0.5
assert np.allclose(p50, q50)
assert np.allclose(p50, np.median(arr))    # 50分位数就是中位数

p90 = np.percentile(arr, 90)
q90 = np.quantile(arr, 0.90)
assert np.allclose(p90, q90)
assert np.allclose(p90, 9.1)

# 抗离群点归一化:用1%/99%分位数做裁剪边界,而不是min/max
lo, hi = np.percentile(arr, [1, 99])
clipped = np.clip(arr, lo, hi)
assert np.allclose((lo, hi), (1.09, 9.91))

# 把百分制的90直接传给quantile是最常见的混用错误,会被直接拒绝执行
raised = False
try:
    np.quantile(arr, 90)     # 90超出了quantile要求的[0,1]范围
except ValueError:
    raised = True
assert raised
```

**常见坑:** 两者刻度不同,`percentile` 是 0~100、`quantile` 是 0~1,**混着用最容易犯的错就是把百分制的数字(比如 90)直接传给 `quantile`**——好在 `quantile` 会直接报错拒绝执行(`Quantiles must be in the range [0, 1]`,实测报错信息),而不是默默算出一个错误结果,算是"坏得明显"而不是"坏得隐蔽"。记忆口诀:**`percentile` 的 `p` 联想"百分比(percent)",`quantile` 就是普通的 0~1 小数——两个函数名字都很像"分位数",真正区分它们的只有 q 的刻度。**

---

## 8. `np.cumsum` / `np.cumprod`

**签名:**
```python
np.cumsum(a, axis=None)
np.cumprod(a, axis=None)
```

**一句话:** "累积"——从头到尾一边走一边把之前所有元素累加(或累乘),**输出和输入的元素个数完全一样,不是"压扁变少"**,这是它和前面 `sum`/`mean`/`max` 最本质的区别——那些叫"归约(reduce)",这个属于"累积(accumulate)"。

**AI 研究场景:** 强化学习里计算"回报"(return)本质上是未来奖励的累积和,`cumsum` 是这个计算最基础的构件(真正的折扣回报还要乘衰减系数,但核心操作就是累加);PCA 降维时判断"保留几个主成分能解释90%的方差",要看 `explained_variance_ratio` 的累积和什么时候越过 0.9 这条线,`np.cumsum` 直接给出这条"越往后累积贡献越大"的曲线。

**可运行例子:**
```python
import numpy as np

rewards = np.array([1.0, 1.0, 1.0, 1.0])
returns = np.cumsum(rewards)
assert list(returns) == [1.0, 2.0, 3.0, 4.0]      # 形状没变,还是4个数,只是变成累积值

g = np.array([[1, 2, 3],
              [4, 5, 6]])
assert np.array_equal(np.cumsum(g, axis=0), [[1, 2, 3], [5, 7, 9]])     # 沿"行"方向累加
assert np.array_equal(np.cumsum(g, axis=1), [[1, 3, 6], [4, 9, 15]])    # 沿"列"方向累加

# 不传axis:先摊平成一维,再逐个累加 —— 不会像sum()那样压成标量
flat_cs = np.cumsum(g)
assert flat_cs.shape == (6,)
assert list(flat_cs) == [1, 3, 6, 10, 15, 21]

# PCA式场景:判断保留几个主成分能解释90%的方差
explained_ratio = np.array([0.6, 0.25, 0.1, 0.05])
cum_explained = np.cumsum(explained_ratio)
assert np.allclose(cum_explained, [0.6, 0.85, 0.95, 1.0])
n_components = np.argmax(cum_explained >= 0.9) + 1     # 第一次越过0.9的位置
assert n_components == 3

probs = np.array([0.2, 0.5, 0.3])
survival = np.cumprod(1 - probs)     # 累乘:逐步"存活率"
assert np.allclose(survival, [0.8, 0.4, 0.28])
```

**常见坑:** 不传 `axis` 时,`cumsum` 和 `sum` 的"默认行为"看起来像但完全不同——**`sum()` 不传 axis 会压成一个标量,但 `cumsum()` 不传 axis 是"先摊平成一维,再逐个累加",输出还是一堆数(元素总数不变),完全不会变成标量**,上面例子里 `g` 是 `(2,3)` 的矩阵,`np.cumsum(g)` 却给出了 `(6,)` 的一维结果,和 `g.sum()` 给出标量的行为很不一样。把"累积类"(`cumsum`/`cumprod`,形状不变)和"归约类"(`sum`/`mean`/`max`...,形状变小)分清楚,是理解这一整批函数最关键的分类框架。

---

## 9. `np.all` / `np.any`

**签名:**
```python
np.all(a, axis=None)
np.any(a, axis=None)
```
- `a`:通常是一个布尔数组(或者能隐式转换成布尔的数组,0是False,非0是True)

**一句话:** `all`——是不是每一个元素都是真;`any`——是不是至少有一个元素为真。

**AI 研究场景:** 训练时最常用的健康检查——`np.any(np.isnan(grad))` 检查梯度里是否出现了 NaN(梯度爆炸/除零错误的信号,几乎是每个训练脚本都该有的断言);单元测试里验证"ReLU 输出必须全部非负"这类不变量,写成 `assert np.all(output >= 0)`,比逐元素检查直观得多。

**可运行例子:**
```python
import numpy as np

grad = np.array([0.1, -0.2, np.nan, 0.4])
assert np.any(np.isnan(grad)) == True          # 梯度里混入了NaN,训练该报警了

relu_out = np.array([0.0, 1.5, 0.0, 3.2])
assert np.all(relu_out >= 0)                   # ReLU输出必须全部非负,这是个不变量检查

h = np.array([[1, 2], [3, 4]])
assert list(np.all(h > 2, axis=1)) == [False, True]    # 第一行不全>2,第二行全部>2

# 空数组上的"空真值"(vacuous truth)
empty = np.array([])
assert np.all(empty > 0) == True     # 空数组:all恒真
assert np.any(empty > 0) == False    # 空数组:any恒假
```

**常见坑:** 空数组上,`np.all` 恒返回 `True`,`np.any` 恒返回 `False`——这叫"空真值"(对空集的所有元素成立这件事永远为真),和 Python 内置的 `all([])==True`、`any([])==False` 是同一套逻辑。**容易踩的坑是:如果上游数据管道出 bug 导致传进来一个空数组,`assert np.all(x >= 0)` 会安静地通过检查,给你一种"数据没问题"的假象,掩盖了真正的 bug(数组是空的这件事本身就不对)。** 写健壮的检查时,最好再加一条 `assert x.size > 0`,确保数组不是意外传空的。

---

## 10. `np.count_nonzero`

**签名:**
```python
np.count_nonzero(a, axis=None)
```

**一句话:** 数一数数组里有多少个"非零"元素——对布尔数组来说就是数"有多少个 True"。

**AI 研究场景:** 统计预测正确的样本数(`np.count_nonzero(preds == labels)`,和 `(preds == labels).sum()` 结果一样,但语义上更直白地表达"我在数满足条件的个数");模型剪枝(pruning)研究里检查"剪枝之后还剩多少非零权重",直接衡量模型的稀疏度(sparsity = 1 - count_nonzero(W) / W.size)。

**可运行例子:**
```python
import numpy as np

preds  = np.array([1, 2, 1, 3])
labels = np.array([1, 2, 2, 3])
correct = np.count_nonzero(preds == labels)
assert correct == 3
assert correct == (preds == labels).sum()        # 布尔数组上两种写法结果等价

pruned = np.array([0.0, 0.001, 0.0, 0.0, 5.0])   # 剪枝后的权重,2个被置0
assert np.count_nonzero(pruned) == 2

# "非零"是严格按字面意义判断的,极小的浮点噪声也算非零
almost_zero = np.array([0.0, 1e-20, 0.0])
assert np.count_nonzero(almost_zero) == 1     # 1e-20被算成"非零"!
```

**常见坑:** "非零"是严格按字面意义判断的——**一个极小但不精确等于0的浮点数(比如 `1e-20`)会被算成"非零"**,这在剪枝研究里是个真实的坑:如果剪枝算法因为浮点误差没有把权重精确置为 `0.0`(而是变成一个极小的残留值),用 `count_nonzero` 算出来的稀疏度会比实际"虚低"。需要"足够接近0就算0"的容差判断时,应该先用 `np.abs(w) < 阈值` 转成布尔数组再数,而不是直接对原始浮点数组调用 `count_nonzero`。

---

## 11. `np.unique`

**签名:**
```python
np.unique(a, return_counts=False)
```
- `return_counts=True`:除了返回去重后的值,还额外返回每个值出现的次数(两个返回值一一对应,按位置配对)

**一句话:** 找出数组里所有不重复的值,并按从小到大排序返回。

**AI 研究场景:** 检查分类数据集的类别分布/是否存在类别不平衡——`values, counts = np.unique(labels, return_counts=True)` 一行代码就能看出"每个类别各有多少样本",是数据集探索(EDA)阶段第一个该跑的检查之一;也常用来给 token id 数组去重、验证一批 ID 里没有意外的重复。

**可运行例子:**
```python
import numpy as np

labels = np.array([0, 1, 1, 2, 2, 2, 0, 1])
values, counts = np.unique(labels, return_counts=True)
assert list(values) == [0, 1, 2]
assert list(counts) == [2, 3, 3]     # values[i]这个类别出现了counts[i]次

# 类别不平衡检查
imbalanced = np.array([0] * 95 + [1] * 5)
v2, c2 = np.unique(imbalanced, return_counts=True)
assert list(v2) == [0, 1]
assert list(c2) == [95, 5]           # 一眼看出类别1只占5%,严重不平衡
```

**常见坑:** 返回的类别永远是**排过序**的,不是"第一次出现的顺序"——如果逻辑依赖"标签第一次出现的先后顺序"(比如想按出现顺序重新编码类别),直接用 `unique` 的结果会出错,需要额外传 `return_index=True` 并自己按下标还原顺序。另外,`counts` 和 `values` 是靠**位置**一一对应的两个独立数组,后续如果只对其中一个做了排序/筛选而忘了对另一个做同样操作,两者就会错位配对,统计结果全错。

---

## 12. `np.bincount`

**签名:**
```python
np.bincount(a, minlength=0)
```
- `a`:**必须是非负整数**数组(和 `unique` 不同,`unique` 什么类型都能处理)
- `minlength`:强制结果至少有多少个"桶",不够就用 0 补齐

**一句话:** 专门给非负整数设计的"计数器"——统计 0、1、2... 一直到最大值,每个整数各出现了多少次,是 `unique(..., return_counts=True)` 在"标签是连续小整数"这个常见场景下的更快替代。

**AI 研究场景:** 计算类别权重(class weights)来缓解类别不平衡——用每个类别出现次数的倒数作为该类别在 loss 里的权重,`weights = 1.0 / np.bincount(labels)`,是处理不平衡数据集最基础的一招。

**可运行例子:**
```python
import numpy as np

labels = np.array([0, 0, 1, 2, 2, 2])
counts = np.bincount(labels)
assert list(counts) == [2, 1, 3]              # 类别0出现2次,类别1出现1次,类别2出现3次

class_weights = 1.0 / counts                   # 出现次数越少,权重越大
assert np.allclose(class_weights, [0.5, 1.0, 1/3])

# 结果长度是max(a)+1,当前batch缺某个类别时长度会"缩水"
sparse_labels = np.array([0, 0, 2])            # 这个batch里没有类别1
bc_bad = np.bincount(sparse_labels)
assert list(bc_bad) == [2, 0, 1]               # 长度3(=max+1),不是想要的"类别总数4"

bc_fixed = np.bincount(sparse_labels, minlength=4)   # 显式声明"总共4个类别"
assert list(bc_fixed) == [2, 0, 1, 0]
```

**常见坑:** 结果的长度是 `max(a) + 1`,**如果当前这个 batch 里恰好没出现某个类别(尤其是编号最大的那个类别),`bincount` 的输出长度会"缩水",跟预期的类别总数对不上**——上面例子里 `sparse_labels` 缺了类别1,`bincount` 默认给出的是长度3的结果(而不是"类别总数4"),必须显式传 `minlength=类别总数` 才能强制补齐。按 batch 统计类别数再拼接、或者小 batch 场景下,一定要传 `minlength`,否则不同 batch 算出来的计数数组长度可能不一致,没法直接相加/对齐。

---

## 13. `np.histogram`

**签名:**
```python
np.histogram(a, bins=10, range=None)
```
- 返回两个数组:`(counts, bin_edges)`——`counts` 是每个桶里有多少个数,`bin_edges` 是桶的分界线
- `range`:强制指定统计的取值范围;不传就用 `[a.min(), a.max()]` 自动决定

**一句话:** 把连续数值切成若干个桶(bin),数一数每个桶里落了多少个数——画直方图背后的计算,本身不画图。

**AI 研究场景:** 调试模型时"看一眼某一层激活值/梯度的分布"是最常用的排查手段之一——比如检查 ReLU 之后是不是有一大半神经元恒为0(死神经元问题)、检查权重分布是不是随着训练逐渐发散;强化学习里检查一批 episode 的回报分布,分类模型里检查预测置信度分布(是不是过度自信,全部堆在接近1的地方)。

**可运行例子:**
```python
import numpy as np

vals = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
counts, edges = np.histogram(vals, bins=4)
assert counts.sum() == len(vals)          # 每个数都被分到了某个桶里,总数不变
assert len(edges) == len(counts) + 1      # 分界线永远比桶数多1
assert np.allclose(counts, [1, 2, 3, 4])

# 固定range,让不同批次的直方图可以直接比较
fixed_counts, fixed_edges = np.histogram(vals, bins=4, range=(0, 8))
assert fixed_edges[0] == 0.0 and fixed_edges[-1] == 8.0
assert np.allclose(fixed_counts, [1, 5, 4, 0])
```

**常见坑:** `bin_edges` 的长度永远是"桶数 + 1"(可以类比:把一条线切成4段,需要5个切点),上面例子 `bins=4` 却拿到了5个 `edges`,把它当成和 `counts` 等长直接一一配对会数组越界或者错位。另外,不传 `range` 时,每次统计用的桶边界是"这批数据自己的 min/max"——**如果拿两个不同 batch 各自默认 range 跑出来的直方图去比较,桶的边界压根不一样,数值上没有可比性**,想比较训练前后/不同 batch 的分布变化,必须显式传同一个 `range`(就像上面例子里固定成 `(0, 8)` 那样)。

---

## 14. `np.average`

**签名:**
```python
np.average(a, axis=None, weights=None)
```
- `weights`:每个元素的权重,形状要和 `a` 匹配;不传就退化成普通 `mean`

**一句话:** 加权平均——`np.mean` 是"每个元素权重相等"的特例,`np.average` 允许给每个元素指定不同的重要性。

**AI 研究场景:** 这是它和 `np.mean` 唯一但关键的区别:多个验证子集大小不同时,要按子集样本数加权才能得到整体上正确的平均指标(不能直接把各子集的准确率再简单平均,样本少的子集会被不合理地赋予和样本多的子集同等的话语权);强化学习里"重要性采样"(importance sampling)修正 off-policy 数据分布偏差时,每条样本要按重要性权重加权平均。

**可运行例子:**
```python
import numpy as np

scores = np.array([80.0, 90.0, 70.0])     # 3个验证子集各自的准确率
sizes  = np.array([1, 1, 2])              # 对应的子集大小(第3个子集样本数是前两个的2倍)

weighted_mean = np.average(scores, weights=sizes)
plain_mean = np.mean(scores)
assert np.allclose(weighted_mean, (80 * 1 + 90 * 1 + 70 * 2) / 4)   # = 77.5
assert not np.allclose(weighted_mean, plain_mean)                    # 明显不同于普通均值80.0

assert np.allclose(np.average(scores), np.mean(scores))    # 不传weights时,两者完全等价

# np.mean根本没有weights参数
raised = False
try:
    np.mean(scores, weights=sizes)
except TypeError:
    raised = True
assert raised
```

**常见坑:** `np.mean` 根本没有 `weights` 这个参数——如果想当然地写 `np.mean(scores, weights=sizes)` 会直接 `TypeError` 报错(报错反而是好事,不会静默算错)。这也是为什么需要专门记住 `average` 这个名字:**要加权,就必须换成 `average`,`mean` 做不到这件事。** 上面例子里 `weighted_mean`(77.5)和 `plain_mean`(80.0)明显不同,提醒自己:子集大小不一样时,不能图省事直接对几个百分比数字取普通平均。

---

## 15. `np.ptp`

**签名:**
```python
np.ptp(a, axis=None, keepdims=False)
```
- 函数名是 "peak to peak"(峰到峰)的缩写

**一句话:** 极差——最大值减最小值(`a.max(axis) - a.min(axis)`),衡量数据"摊开的范围有多宽"。

**AI 研究场景:** 数据预处理前快速摸底一个特征的取值范围(比如确认像素值到底是0~255还是已经归一化到0~1,决定预处理方式);更实用的一个场景是**揪出"死特征"**——如果某一列特征的 `ptp` 是 0,说明这一整列全是同一个常数,对模型训练毫无信息量(方差为0,梯度也不会因为它有任何变化),应该在特征工程阶段直接丢弃。

**可运行例子:**
```python
import numpy as np

X = np.array([[1.0, 2.0, 7.0],
              [4.0, 2.0, 7.0],
              [9.0, 2.0, 7.0]])       # 3个样本,3个特征;第2、3列是常数特征

col_ptp = np.ptp(X, axis=0)
assert np.allclose(col_ptp, [8.0, 0.0, 0.0])

dead_cols = np.where(col_ptp == 0)[0]
assert list(dead_cols) == [1, 2]      # 第1、2列(0-indexed)是死特征,该丢弃
```

**常见坑:** 函数名 `ptp` 缩写不直观(第一次见完全猜不出意思),而且本质上只是 `max - min` 这个简单运算的一个"快捷方式"——不少代码库里的人干脆直接手写 `a.max(axis) - a.min(axis)`,可读性反而更好。看懂别人代码里出现 `ptp` 时不困惑即可,不必强求自己写代码时也用它。

---

## 16. `np.corrcoef` / `np.cov`

**签名:**
```python
np.corrcoef(x, rowvar=True)
np.cov(m, rowvar=True, ddof=None)      # ddof不传时,内部效果等价于ddof=1
```
- `rowvar=True`(默认!):把**每一行**当成一个变量,**每一列**当成一次观测/采样
- `ddof`:和第3节的 `std`/`var` 是同一个参数,但 `cov` 默认效果是 `ddof=1`(样本协方差),和 `std`/`var` 默认的 `ddof=0` 正好相反,是 numpy 自己内部都不统一的一个坑

**一句话:** `cov`(协方差矩阵)衡量每两个变量"一起变化"的程度(正相关/负相关/不相关),但数值大小受量纲影响没法直接比较;`corrcoef`(相关系数矩阵)是协方差"归一化"到 `[-1, 1]` 之后的版本,不同变量之间可以直接比较相关性强弱。

**AI 研究场景:** 特征工程阶段分析特征之间的相关性,发现高度相关(冗余)的特征并剔除,减少特征维度、缓解共线性问题;分析模型学到的 embedding 各个维度之间是否存在冗余(如果两个维度相关系数接近1,说明这两维在编码几乎相同的信息)。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(0)
n_samples, n_features = 200, 3
X = rng.normal(size=(n_samples, n_features))     # 标准"每行一个样本"矩阵,shape (200, 3)
X[:, 1] = X[:, 0] * 2 + rng.normal(scale=0.01, size=n_samples)   # 让特征1和特征0几乎线性相关

# 直接把X丢进去是错的:默认rowvar=True把"每一行(每个样本)"当成一个变量
wrong = np.corrcoef(X)
assert wrong.shape == (n_samples, n_samples)     # (200,200) 样本之间的"相关性",不是我们要的

# 正确做法:告诉它"每一列才是一个变量"
correct = np.corrcoef(X, rowvar=False)
assert correct.shape == (n_features, n_features)  # (3,3) 这才是特征相关性矩阵
assert correct[0, 0] == 1.0                        # 对角线永远是1
assert correct[0, 1] > 0.99                        # 特征0和特征1几乎完全线性相关

# cov的ddof默认效果和std/var不一致
cov_default = np.cov(X, rowvar=False)
cov_ddof1 = np.cov(X, rowvar=False, ddof=1)
cov_ddof0 = np.cov(X, rowvar=False, ddof=0)
assert np.allclose(cov_default, cov_ddof1)         # cov默认效果等价于ddof=1(样本协方差)
assert not np.allclose(cov_default, cov_ddof0)      # 跟ddof=0结果不同
```

**常见坑(两个,都很致命,容易得到"形状对但数值全错"的结果):**

1. **`rowvar=True` 是默认值,意味着"每一行是一个变量、每一列是一次观测"——这和 AI/ML 里"每一行是一个样本、每一列是一个特征"的标准约定(设计矩阵 `X.shape == (n_samples, n_features)`)正好是反的。** 如果直接把形状 `(样本数, 特征数)` 的矩阵传进去不做任何处理,得到的是一个 `(样本数, 样本数)` 的"样本相关性矩阵",而不是想要的 `(特征数, 特征数)` 特征相关性矩阵——形状经常直接对不上导致后续代码报错,但如果样本数恰好等于特征数,形状还一样,结果却是完全错误的数字,更隐蔽。**记住:标准 ML 矩阵传给 `corrcoef`/`cov`,几乎总要加 `rowvar=False`(或者提前转置 `X.T`)。**
2. **`np.cov` 的默认行为等价于 `ddof=1`(样本协方差),而第3节讲过 `np.std`/`np.var` 默认是 `ddof=0`(总体方差)——同一个 numpy 库,统计量家族内部默认值都不统一**,不能想当然地认为"整个 numpy 统计模块的默认口径一致"。需要精确控制时,`cov`/`corrcoef` 都支持显式传 `ddof=0` 或 `ddof=1`,别依赖记忆里的默认值,不确定就查一下或者两个都跑一遍对比。

---

## 小结:这一批 16 个函数解决的问题

| 函数 | 解决的问题 |
|---|---|
| `sum` | 沿某方向加总(理解 axis 方向的基础) |
| `mean` | 平均值,对应 loss 的 `reduction='mean'` |
| `std`/`var` | 标准差/方差,BatchNorm 与数据标准化的核心 |
| `max`/`min` | 极值(softmax 数值稳定、梯度裁剪) |
| `argmax`/`argmin` | 极值的位置(分类预测取标签、最近邻查找) |
| `median` | 抗离群点的"典型值" |
| `percentile`/`quantile` | 分位数(p50/p90/p99 延迟报告、抗离群点裁剪) |
| `cumsum`/`cumprod` | 累积(不压缩形状!强化学习回报、PCA 累积方差) |
| `all`/`any` | 布尔归约(NaN 检查、不变量断言) |
| `count_nonzero` | 数满足条件的个数(准确率、剪枝稀疏度) |
| `unique` | 去重+计数(类别分布/不平衡检查) |
| `bincount` | 非负整数专用计数器(类别权重) |
| `histogram` | 分桶计数(激活值/梯度分布调试) |
| `average` | 加权平均(区别于普通 mean) |
| `ptp` | 极差(死特征检测) |
| `corrcoef`/`cov` | 相关性/协方差(特征冗余分析) |

下一批:[06-linear-algebra.md](06-linear-algebra.md) —— 线性代数。

---

*更新:2026-07-07*
