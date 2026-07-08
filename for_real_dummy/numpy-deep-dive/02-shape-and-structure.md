# 02 · 形状与结构操作(Shape & Structure)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**数据本身不变,但"摆放方式"要改**——把多维数组拆开、拼起来、转个方向、加一个维度、去掉一个维度,或者铺开重复。这是从"读懂 shape 报错"走向"看一眼就知道该用哪个函数改形状"的关键一批。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `.shape` / `.reshape`

**签名:**
```python
a.shape                      # 属性(不是方法),只读,返回一个 tuple
a.reshape(newshape)          # newshape:整数或元组;可以用 -1 表示"这一维自动推断"
```
- `a.shape`:每个维度的长度,比如 `(3, 4)` 表示 3 行 4 列
- `a.reshape(...)`:总元素个数不能变,只是换一种"摆法";`-1` 那一维会自动算出该是多少

**一句话:** `.shape` 是"读身份证"——这个数组现在长什么形状;`.reshape` 是"重新排列成另一个形状",但底层数据本身不变。

**AI 研究场景:**
- **展平接全连接层:** 卷积/图像特征是 `(batch, H, W, C)` 这种多维,喂进全连接层之前要 `reshape(batch, -1)` 展平成二维。
- **multi-head attention 拆头:** 把 `(batch, seq_len, hidden)` reshape 成 `(batch, seq_len, num_heads, head_dim)`,再配合下面的 `transpose` 调整顺序,是 attention 实现里最开头的固定动作。
- **调试:** "为什么维度对不上"的报错,第一反应永远是 `print(x.shape)`——这是 numpy/torch 代码里使用频率最高的调试手段,没有之一。

**可运行例子:**
```python
import numpy as np

a = np.arange(12)
assert a.shape == (12,)

b = a.reshape(3, 4)
assert b.shape == (3, 4)
assert np.array_equal(b[0], [0, 1, 2, 3])

c = a.reshape(2, -1)             # -1 表示"这一维自动推断"
assert c.shape == (2, 6)

# 常见做法:卷积/图像特征展平后接全连接层
batch_images = np.random.randn(8, 28, 28, 3).astype(np.float32)   # (batch,H,W,C)
flat = batch_images.reshape(8, -1)
assert flat.shape == (8, 28 * 28 * 3)

# reshape 默认尽量返回 view(共享内存)
b[0, 0] = 999
assert a[0] == 999                # 改 view 影响了原数组
```

**常见坑:**
- **元素总数必须对得上。** `np.arange(12).reshape(3, 5)` 直接报错(12 ≠ 15)。
- **`reshape` vs `resize`,三个东西容易混:**`a.reshape(...)` 不修改原数组(返回新形状的 view 或 copy);而 `a.resize(...)`(方法)是**原地**修改,没有返回值,而且允许"元素数量对不上"——多退(截断),少补(补 0);顶层函数 `np.resize(a, shape)` 又不一样,元素不够时用**重复原数据**的方式填满,而不是补 0。三者语义差异很大,写训练代码时几乎总该用 `reshape`,`resize` 系列出现的场景很少,不确定就选 `reshape`。
- **reshape 返回的到底是 view 还是 copy,取决于内存是否连续。** 大部分"简单"reshape 是 view(改结果会改原数组);但如果数组之前被转置过导致内存不连续,reshape 会被迫悄悄复制一份新内存(不会报错,只是不再和原数组共享)。不确定就用 `np.shares_memory(a, b)` 查一下。

```python
import numpy as np

# 坑1:元素总数必须对得上
try:
    np.arange(12).reshape(3, 5)
    assert False
except ValueError:
    pass                          # 12 != 15,直接报错

# 坑2:reshape vs resize —— 三种完全不同的行为
a = np.arange(6)
a.resize((3, 4), refcheck=False)  # 原地修改,元素不够用0补
assert a.shape == (3, 4)
assert a.sum() == 15              # 0+1+2+3+4+5=15,其余位置是补的0

b = np.resize(np.arange(6), (3, 4))   # 顶层函数:元素不够时"重复"原数据填满
assert b.shape == (3, 4)
assert b.sum() == 30              # 原地resize补0(和为15),np.resize补重复数据(和为30)

# 坑3:reshape 是 view 还是 copy 取决于内存是否连续
m = np.arange(12).reshape(3, 4)
mt = m.T                          # 转置后内存不再连续
r = mt.reshape(12)
assert not np.shares_memory(mt, r)   # 被迫复制了一份新内存,而不是 view
```

---

## 2. `.T` / `np.transpose`

**签名:**
```python
a.T                              # 属性,把所有维度顺序整体倒过来
np.transpose(a, axes=None)       # axes:一个 tuple,指定新的维度顺序;不写就等价于 .T
```
- `a.T`:二维时就是"矩阵转置"(行列互换);高维时是把**所有**维度顺序整体倒过来
- `np.transpose(a, axes)`:自己指定任意的维度重排顺序,不局限于整体倒序

**一句话:** 交换数组的维度顺序——二维最常见的用法是"行变列、列变行"。

**AI 研究场景:**
- **线性代数:** 全连接层里 `X @ W.T` 是最常见的写法之一(权重矩阵按"输出维度 x 输入维度"存的时候需要转置才能对上矩阵乘法的维度)。
- **attention 算分数:** `scores = Q @ K.transpose(...)` 需要把 K 的最后两个维度换过来做矩阵乘法,才能得到 `(seq_len, seq_len)` 的注意力分数矩阵。
- **图像通道顺序转换:** 数据加载器读进来的图像是 `(H, W, C)`,卷积网络要求 `(C, H, W)`,这时不能偷懒用 `.T`(会把所有维度整体倒序,不是你想要的那种"只调整顺序"),必须显式 `np.transpose(img, (2, 0, 1))`。

**可运行例子:**
```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
assert A.shape == (2, 3)
assert A.T.shape == (3, 2)
assert np.array_equal(A.T, [[1, 4], [2, 5], [3, 6]])

# 高维:.T 是整体倒序,不一定是你想要的顺序
img = np.arange(60).reshape(4, 5, 3)      # 假装是 (H=4, W=5, C=3)
img_T = img.T
assert img_T.shape == (3, 5, 4)            # 整体倒序:(C, W, H)
assert img_T[2, 3, 1] == img[1, 3, 2]      # 坐标对应关系也整体反过来了

# 想要 (C, H, W) 必须显式指定 axes,不能靠 .T
img_chw = np.transpose(img, (2, 0, 1))
assert img_chw.shape == (3, 4, 5)
assert img_chw[2, 1, 3] == img[1, 3, 2]

assert np.array_equal(np.transpose(A), A.T)   # 不写 axes 时和 .T 完全等价
```

**常见坑:** 二维时 `.T` 很直观,但维度 ≥ 3 时 `.T` 是把**所有**维度顺序整体倒过来,不是只交换你想要的那两个——`(4,5,3).T` 变成 `(3,5,4)`,是把 `(0,1,2)` 这个维度顺序整体倒成 `(2,1,0)`。想要精确控制"只调整某几个维度、其他不动"时,要么用 `np.transpose(a, axes)` 显式写清楚每一维去哪,要么用下一节的 `swapaxes`/`moveaxis`。

---

## 3. `np.swapaxes` / `np.moveaxis`

**签名:**
```python
np.swapaxes(a, axis1, axis2)          # 只交换两个指定的轴,其余不动
np.moveaxis(a, source, destination)   # 把某个轴"搬"到目标位置,其余轴顺序自动补齐
```

**一句话:** 比 `.T`/`np.transpose` 更"精细"的维度重排工具——只动你指定的轴,不用把所有维度顺序都写一遍。

**AI 研究场景:**
- **attention 只换最后两维:** 算 QK^T 时只需要交换最后两个维度,`K.swapaxes(-1, -2)` 不影响 batch 和 head 维度,比写全 `transpose(0, 1, 3, 2)` 更不容易写错。
- **调整 batch 维度位置:** 把 `(batch, seq_len, hidden)` 转成 `(seq_len, batch, hidden)`(有些库/老代码要求 seq_len 在前),用 `np.moveaxis(a, 0, 1)`——只关心"把 batch 这个维度挪到第二个位置",不用手写完整的维度排列。

**可运行例子:**
```python
import numpy as np

a = np.random.randn(2, 3, 4, 5)          # 假设是 (batch, head, seq_len, dim)
b = np.swapaxes(a, -1, -2)               # 只交换最后两维,常见于 attention 算 K^T
assert b.shape == (2, 3, 5, 4)

c = np.random.randn(8, 16, 32)           # (batch, seq_len, hidden)
d = np.moveaxis(c, 0, 1)                 # 把 batch(轴0) 挪到位置1,其余顺序自动补齐
assert d.shape == (16, 8, 32)            # (seq_len, batch, hidden)

# moveaxis 和手写 transpose 等价,方便验证理解
e = np.transpose(c, (1, 0, 2))
assert np.array_equal(d, e)
```

**常见坑:** `swapaxes(a, i, j)` 和 `moveaxis(a, src, dst)` 传参逻辑不一样——`swapaxes` 是"交换两个轴的位置"(对称操作,`swapaxes(a,0,1)` 和 `swapaxes(a,1,0)` 结果相同);`moveaxis` 是"把某个轴搬到指定位置"(不对称,`moveaxis(a,0,1)` 和 `moveaxis(a,1,0)` 结果不同,是两种不同的重排)。写高维 attention/RNN 代码时如果搞混这两种语义,shape 可能凑巧对得上但值算错了——这类 bug 不会报错,只会让结果悄悄变得不对,是最隐蔽的一类维度错误。

---

## 4. `.flatten` / `.ravel`

**签名:**
```python
a.flatten(order='C')     # 总是返回一份新内存的 copy
a.ravel(order='C')       # 尽量返回 view,不行时才 copy
```
- `order='C'`:按行优先展开(最后一个维度变化最快,C 语言的内存顺序),`order='F'` 是列优先(Fortran顺序)

**一句话:** 把多维数组展平成一维——两者结果的值完全一样,区别只在于底层是否复制内存。

**AI 研究场景:** 卷积层输出的多维 feature map 展平后接全连接层;把多维梯度展平成一维向量,喂给某些要求"参数看成一个大向量"的优化算法(比如 L-BFGS);统计一个数组所有元素的分布之前先展平成 1D 方便处理。

**可运行例子:**
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

f = a.flatten()
r = a.ravel()
assert np.array_equal(f, [1, 2, 3, 4, 5, 6])
assert np.array_equal(r, [1, 2, 3, 4, 5, 6])

# 关键区别:flatten 总是 copy,ravel 尽量返回 view
f[0] = 999
assert a[0, 0] == 1          # flatten 的修改不影响原数组

r[0] = 888
assert a[0, 0] == 888        # ravel 的修改影响了原数组!说明这是 view

assert np.shares_memory(a, r)
assert not np.shares_memory(a, f)
```

**常见坑:** **`flatten` 总是复制一份新内存,`ravel` 能不复制就不复制(但不保证——如果原数组内存不连续,`ravel` 也会被迫复制)。** 只读场景两者性能几乎没区别;但如果要修改展平后的结果又不想连累原数组,必须用 `flatten`。反过来,处理超大数组、明确不需要保留原数组不变、且性能敏感时,`ravel` 能省一次内存复制。记不清就默认用更安全的 `flatten`,除非已经用 profiler 定位到这里是瓶颈。

---

## 5. `np.squeeze`

**签名:**
```python
np.squeeze(a, axis=None)     # 或 a.squeeze(axis=None)
```
- `axis`:不写就删掉所有大小为 1 的维度;写了具体的轴,就只删那一个(如果那个轴大小不是 1 会报错)

**一句话:** 把数组里所有(或指定的)"大小为 1"的维度去掉。

**AI 研究场景:** 单样本推理时模型输出常常带一个多余的 batch 维度,比如 `(1, num_classes)`,拿出来当普通向量用就 `squeeze()`;某些统计操作(比如 `sum(..., keepdims=True)`)故意保留大小为 1 的维度方便广播,后续不再需要广播时用 squeeze 去掉,恢复成"干净"的形状。

**可运行例子:**
```python
import numpy as np

out = np.random.randn(1, 10)             # 单样本预测,batch维度是多余的1
squeezed = np.squeeze(out)
assert squeezed.shape == (10,)

b = np.random.randn(1, 3, 1)
assert np.squeeze(b, axis=0).shape == (3, 1)      # 只去掉第0维
assert np.squeeze(b).shape == (3,)                 # 不指定轴,所有大小为1的维度都被去掉

# 指定的轴如果大小不是1,直接报错
c = np.random.randn(2, 3)
try:
    np.squeeze(c, axis=0)
    assert False, "应该报错"
except ValueError:
    pass
```

**常见坑:** 不写 `axis` 时会把**所有**大小为 1 的维度一次性去掉——如果数组里不止一个维度大小是 1(比如 `(1, 3, 1)`),可能删得比预期多,导致后续和别的数组运算时形状对不上,而且这种错误往往不会立刻报错,是过一会儿才在别处炸出来的那类坑。建议养成习惯:显式写 `axis=0`(或具体是哪个轴),而不是依赖"全部删除"的默认行为。

---

## 6. `np.expand_dims` / `np.newaxis`

**签名:**
```python
np.expand_dims(a, axis)     # axis:在哪个位置插入一个新的大小为1的维度
a[:, np.newaxis]            # 等价写法,np.newaxis 就是 None 的别名,直接在索引里插入新维度
```

**一句话:** `squeeze` 的反操作——插入一个大小为 1 的新维度。

**AI 研究场景:**
- **凑维度才能广播:** 一个形状 `(N,)` 的向量要和 `(N, D)` 的矩阵做"每行减去这个向量"之类的运算,直接算会因为广播规则对不上而报错,需要先 `expand_dims` 成 `(N, 1)`。
- **单样本推理补 batch 维度:** 训练时输入总是 `(batch, features)`,推理时手里只有一条数据 `(features,)`,要 `expand_dims(x, axis=0)` 补回 batch 维度才能喂给模型。
- **灰度图补 channel 维度:** 读进来是 `(H, W)`,卷积网络要求 `(H, W, 1)`,用 `expand_dims(img, axis=-1)`。

**可运行例子:**
```python
import numpy as np

x = np.array([1, 2, 3])                 # 单条数据,shape (3,)
x_batch = np.expand_dims(x, axis=0)
assert x_batch.shape == (1, 3)           # 补上 batch 维度

x_batch2 = x[np.newaxis, :]              # 等价写法
assert np.array_equal(x_batch, x_batch2)

img = np.random.randn(28, 28)            # 灰度图,没有 channel 维度
img_c = np.expand_dims(img, axis=-1)
assert img_c.shape == (28, 28, 1)

assert np.squeeze(x_batch).shape == x.shape   # expand_dims 和 squeeze 互为逆操作
assert np.newaxis is None                      # newaxis 本质就是 None
```

**常见坑:** `np.newaxis` 本质就是 Python 内置的 `None`,所以别人代码里偶尔会看到 `a[:, None]` 代替 `a[:, np.newaxis]`,两者完全等价,只是 `np.newaxis` 可读性更好、更明确表达意图。另外 `axis` 支持负数(`axis=-1` 表示"插在最后"),写不确定输入有几维的通用函数时特别有用。

---

## 7. `np.concatenate`

**签名:**
```python
np.concatenate(arrays, axis=0)
```
- `arrays`:一个 list/tuple,包含若干个数组
- `axis`:沿哪个**已经存在**的维度拼接;这些数组在其它维度上必须形状完全相同

**一句话:** 把若干个数组沿着已有的某个维度首尾相接,**不增加新维度**。

**AI 研究场景:** 把多个 mini-batch 算出来的中间结果拼回一个大数组做后处理;数据增强时把原始数据和增强后的数据在样本维度(axis=0)上拼起来扩充数据集;把 multi-head attention 每个 head 算完的输出在最后一维拼回 `(seq_len, num_heads * head_dim)`。

**可运行例子:**
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])       # (2, 2)
b = np.array([[5, 6], [7, 8]])       # (2, 2)

cat0 = np.concatenate([a, b], axis=0)
assert cat0.shape == (4, 2)          # 沿 axis0 拼,行数翻倍
assert np.array_equal(cat0[2], [5, 6])

cat1 = np.concatenate([a, b], axis=1)
assert cat1.shape == (2, 4)          # 沿 axis1 拼,列数翻倍

# 多头 attention 输出拼接的简化示意
head_outputs = [np.random.randn(4, 8) for _ in range(3)]   # 3个head,每个(seq_len=4,head_dim=8)
merged = np.concatenate(head_outputs, axis=-1)
assert merged.shape == (4, 24)       # (seq_len, num_heads*head_dim)
```

**常见坑(和 [03-how-to-look-up-not-memorize.md](../03-how-to-look-up-not-memorize.md) 提过的 cat/stack 呼应,这里讲背后原理):** `concatenate` 要求参与拼接的数组**维度数(ndim)相同**,除了拼接的那一维之外其余维度必须完全一致——`(2,3)` 和 `(2,4)` 能沿 axis=1 拼成 `(2,7)`,但不能沿 axis=0 拼(第二维 3≠4)。本质原理是:`concatenate` **不创造新的维度**,拼接后 ndim 和输入一样,只是某一维的长度做了累加。这也是为什么它处理不了"我有 N 个形状相同的数组,想要一个全新的 N 这个维度"的场景——那正是下一节 `stack` 要解决的问题。

---

## 8. `np.stack`

**签名:**
```python
np.stack(arrays, axis=0)
```
- 要求所有输入数组 shape **完全相同**
- 结果比输入**多一个维度**,新维度长度就是 `len(arrays)`

**一句话:** 把若干个形状相同的数组"摞"在一起,专门创造一个新维度来表示"这是第几个数组"。

**AI 研究场景:** 把 for 循环里逐个样本算出来的结果(每个 shape 相同)重新组装成带 batch 维度的数组,比如 `np.stack([model(x) for x in samples])`;把每个 epoch 的 loss 数组堆叠起来做多轮实验对比;合成 RGB 图像时把三个 `(H, W)` 单通道数组 `np.stack([r,g,b], axis=-1)` 堆成 `(H, W, 3)`。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

s0 = np.stack([a, b], axis=0)
assert s0.shape == (2, 3)              # 新增维度在最前,长度=数组个数(2)
assert np.array_equal(s0[0], a)

s1 = np.stack([a, b], axis=1)
assert s1.shape == (3, 2)              # 新增维度插在最后
assert np.array_equal(s1[:, 0], a)

# RGB 合成示意:3个(H,W)单通道 -> 1个(H,W,3)
r = np.zeros((4, 4))
g = np.ones((4, 4))
bch = np.full((4, 4), 2.0)
rgb = np.stack([r, g, bch], axis=-1)
assert rgb.shape == (4, 4, 3)
assert np.allclose(rgb[0, 0], [0, 1, 2])

# 验证背后原理:stack = 先每个都 expand_dims,再 concatenate
manual = np.concatenate([np.expand_dims(a, axis=0), np.expand_dims(b, axis=0)], axis=0)
assert np.array_equal(manual, s0)
```

**常见坑:** `stack` 强制要求所有输入 shape 完全相同(哪怕只差一个维度的长度也直接报错)——这正是它和 `concatenate` 本质区别的来源:`concatenate` 在"已有的维度"上做加法(ndim 不变),`stack` 是先给每个数组统一 `expand_dims` 插入一个新维度、再 `concatenate`。也就是说 `np.stack(arrays, axis=k)` 数学上等价于 `np.concatenate([np.expand_dims(a, axis=k) for a in arrays], axis=k)`(上面例子已经验证过)。理解了这层等价关系,就不用死记硬背"到底该用哪个"了——想清楚"要不要新增一个维度"就知道答案。

---

## 9. `np.hstack` / `np.vstack`

**签名:**
```python
np.hstack(arrays)      # horizontal,水平方向拼接
np.vstack(arrays)      # vertical,垂直方向拼接
```

**一句话:** `concatenate` 的两个"语法糖"版本,分别固定在水平/垂直方向拼接,不用自己想清楚 axis 该填几。

**AI 研究场景:** 快速拼表格型数据(把几个特征列拼成更大的特征矩阵);写实验脚本图快,不想每次都琢磨 axis 是 0 还是 1。生产代码里更推荐显式用 `concatenate(..., axis=)`,因为下面会看到 `hstack`/`vstack` 对一维数组的行为和二维不一致,显式写 axis 更不容易踩坑。

**可运行例子:**
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 二维时:vstack 等价于 concatenate axis=0,hstack 等价于 axis=1
assert np.array_equal(np.vstack([a, b]), np.concatenate([a, b], axis=0))
assert np.array_equal(np.hstack([a, b]), np.concatenate([a, b], axis=1))

# 一维时:hstack 还是 axis=0(唯一的轴),但 vstack 会先各自补一维再拼接
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

assert np.array_equal(np.hstack([x, y]), np.concatenate([x, y], axis=0))
assert np.hstack([x, y]).shape == (6,)          # 一维拼完还是一维

assert np.vstack([x, y]).shape == (2, 3)        # 一维输入,vstack 结果却变成了二维!
```

**常见坑:** **`hstack`/`vstack` 对一维和二维数组的行为并不对称。** 二维时 `hstack`=`concatenate(axis=1)`、`vstack`=`concatenate(axis=0)`,很直观;但一维数组没有"列"的概念,`vstack` 处理一维输入时会先把每个一维数组当成一行、`expand_dims` 成二维再拼接——所以两个 `(3,)` 的一维数组经过 `vstack` 会变成 `(2, 3)` 的二维结果,这和"vstack 只是 concatenate 的语法糖"的直觉不完全一致,很容易把新手绊倒。不确定行为时,显式用 `np.concatenate` 并自己写清楚 `axis`,比死记这些特例更可靠。

---

## 10. `np.split` / `np.array_split`

**签名:**
```python
np.split(a, indices_or_sections, axis=0)
np.array_split(a, indices_or_sections, axis=0)
```
- `indices_or_sections`:整数 N 表示"平均分成 N 份"(`split` 要求必须能整除,`array_split` 不要求);列表比如 `[3, 7]` 表示"按下标切开的分割点"

**一句话:** `concatenate` 的反操作——把一个数组沿某个维度切成若干份。

**AI 研究场景:** 把一个大 batch 切成若干个小 batch 手动分发给多个 GPU/进程;把 multi-head attention 合并计算完的大矩阵按 head 切开分别处理;把数据集按下标切成 train/val/test 三份。

**可运行例子:**
```python
import numpy as np

a = np.arange(9)

# split:必须能整除,否则报错
parts = np.split(a, 3)
assert len(parts) == 3
assert np.array_equal(parts[0], [0, 1, 2])
assert np.array_equal(parts[1], [3, 4, 5])

# 按指定下标切:[3,7] 表示在下标3和7处切开,得到3段
parts2 = np.split(a, [3, 7])
assert np.array_equal(parts2[0], [0, 1, 2])
assert np.array_equal(parts2[1], [3, 4, 5, 6])
assert np.array_equal(parts2[2], [7, 8])

try:
    np.split(np.arange(10), 3)      # 10 不能被3整除
    assert False
except ValueError:
    pass

# array_split:不整除也不报错,前面几份会多分1个凑数
b = np.arange(10)
uneven = np.array_split(b, 3)
assert [len(p) for p in uneven] == [4, 3, 3]     # 10 = 4+3+3
```

**常见坑:** `np.split(a, 3)` 要求 `a` 的长度必须能被 3 整除,不能整除会直接抛 `ValueError`;想要"尽量平均分,不能整除也没关系"就用 `np.array_split`——两个名字长得像,唯一区别是"严格模式"和"宽松模式"。分布式训练里把 batch 切给不同数量的 worker 时,数据量常常不能整除 worker 数,这时必须用 `array_split`,否则程序会在数据量不凑巧的那一次直接崩溃。

---

## 11. `np.tile`

**签名:**
```python
np.tile(a, reps)
```
- `reps`:一个整数或元组,表示"整个数组"在每个维度上重复几次

**一句话:** 把整个数组当作一个"图案",在指定方向上完整地重复铺贴若干次。

**AI 研究场景:** 给每个样本复制同一份模板(比如把一个共享的 mask/位置编码模板复制给 batch 里的每一条数据,虽然很多框架里这种场景会交给广播自动处理,不需要真的复制内存,但手动构造测试数据、或者目标库不支持广播时仍然需要真实复制);数据增强/单元测试里人工构造带重复模式的输入。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])

t1 = np.tile(a, 2)
assert np.array_equal(t1, [1, 2, 3, 1, 2, 3])          # 整个[1,2,3]重复2遍

t2 = np.tile(a, (2, 1))
assert t2.shape == (2, 3)
assert np.array_equal(t2, [[1, 2, 3], [1, 2, 3]])       # 复制成2行,常用来给batch复制模板

# 二维数组的 tile:行、列方向分别重复
b = np.array([[1, 2], [3, 4]])
t3 = np.tile(b, (2, 3))
assert t3.shape == (4, 6)          # 行方向重复2次、列方向重复3次
```

**常见坑(tile vs repeat,下一节详细对比):** `tile` 是"重复整个数组"——结构原封不动,像贴瓷砖一样一整块一整块地铺。`reps` 的维度数如果比 `a` 的维度数多,`a` 会被自动在前面补维度;如果比 `a` 少,则从最后一个维度开始对齐。不确定结果形状时,先用小数组试一遍、`print(shape)` 确认,再套用到真实数据上。

---

## 12. `np.repeat`

**签名:**
```python
np.repeat(a, repeats, axis=None)
```
- `repeats`:每个元素重复几次(单个整数=统一次数;也可以传数组,给每个元素指定不同的重复次数)
- `axis`:不写就先展平成 1D 再重复;写了就沿指定轴对每个"切片"重复

**一句话:** 把数组里**每一个元素**原地重复若干次——不是整体重复,是元素级别的重复。

**AI 研究场景:**
- **最近邻上采样:** 图像的最近邻插值放大,本质就是在 H、W 两个维度上分别 `repeat`,把每个像素"变粗"。
- **标签展开到子词:** 一个词被 tokenizer 切成多个 subword token,词级别的标签要 repeat 到每个 subword 上才能和 token 序列对齐。
- **按权重过采样:** 类别不均衡数据集里对少数类样本过采样,用 `repeats` 数组给每个样本指定不同的重复次数。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])

r1 = np.repeat(a, 2)
assert np.array_equal(r1, [1, 1, 2, 2, 3, 3])          # 每个元素自己重复2次

# 对比 tile:长度一样,顺序完全不同
t1 = np.tile(a, 2)
assert np.array_equal(t1, [1, 2, 3, 1, 2, 3])
assert not np.array_equal(r1, t1)

# 每个元素指定不同的重复次数
r2 = np.repeat(a, [1, 2, 3])
assert np.array_equal(r2, [1, 2, 2, 3, 3, 3])

# 图像最近邻上采样示意:2x2 -> 4x4(每个像素放大2倍)
img = np.array([[1, 2], [3, 4]])
up = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
assert up.shape == (4, 4)
assert np.array_equal(up, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
```

**常见坑(tile vs repeat 的核心区别):** 两者最容易被当成"差不多的东西",但原理完全不同——**`tile` 复制的是"整体结构"(图案式重复:`[1,2,3]` 重复2次是 `[1,2,3,1,2,3]`),`repeat` 复制的是"每个元素"(元素式重复:`[1,2,3]` 每个重复2次是 `[1,1,2,2,3,3]`)。** 判断该用哪个的窍门:想象你要的结果——"同一批数据完整地再来一遍"(比如给每个样本复制一份相同的模板)用 `tile`;"每一条数据自己紧挨着变成好几份副本"(比如上采样、标签展开到 subword)用 `repeat`。

---

## 13. `np.pad`

**签名:**
```python
np.pad(a, pad_width, mode='constant', constant_values=0)
```
- `pad_width`:每个维度前后各填充多少——单个数字表示所有维度、前后都一样;精确控制则写 `((before_0,after_0), (before_1,after_1), ...)`
- `mode`:填充方式,`'constant'`(最常用,填固定值,默认0)之外还有 `'edge'`(重复边缘值)、`'reflect'`(镜像)等

**一句话:** 在数组边缘补充额外的值,不改变原始数据,只是给它"包一圈"或"接长"。

**AI 研究场景:**
- **序列 padding:** NLP 里不同长度的句子要凑成同一个 batch,必须把短句子 pad 到和最长句子一样长(pad token 通常是 0),这是几乎所有 NLP 训练代码开头都会做的事。
- **卷积的 SAME padding:** 想让卷积输出和输入尺寸相同,需要先在图像边缘 pad 一圈 0,再做卷积,否则输出尺寸会比输入小。
- **对齐不定长数据:** 一批长度不同的时间序列/点云数据,pad 成统一长度才能拼成一个规整的 numpy 数组(不整齐的数据没法直接组成矩形数组)。

**可运行例子:**
```python
import numpy as np

# 一维:序列 padding 到固定长度
seq = np.array([1, 2, 3])
padded = np.pad(seq, (0, 2), mode='constant', constant_values=0)   # 后面补2个0
assert np.array_equal(padded, [1, 2, 3, 0, 0])

# 二维:图像四周各 pad 1 圈 0(常见于卷积 SAME padding)
img = np.array([[1, 2], [3, 4]])
padded_img = np.pad(img, ((1, 1), (1, 1)), mode='constant')
assert padded_img.shape == (4, 4)
assert np.array_equal(padded_img[1:3, 1:3], img)     # 中间还是原图
assert padded_img[0, 0] == 0                          # 四周补的是0

# batch 内不同长度序列 pad 到同一长度的典型写法
lengths = [2, 4, 3]
max_len = max(lengths)
batch = np.zeros((len(lengths), max_len), dtype=np.int64)
for i, l in enumerate(lengths):
    batch[i, :l] = np.arange(1, l + 1)
assert np.array_equal(batch[0], [1, 2, 0, 0])       # 长度2的序列,后面补0到长度4
```

**常见坑:** `pad_width` 的嵌套结构容易写错——`((1,1),(1,1))` 表示"两个维度,每个维度前后各填 1 个",不是"填充值是 (1,1)";数组维度数和 `pad_width` 里的元组个数对不上会直接报错。另外 `mode='constant'` 不传 `constant_values` 时默认填 0(通常够用),但如果是 attention mask 之类需要填 `-inf`、或者需要填 1 表示"合法位置"的场景,一定要显式传 `constant_values`,不要以为默认值总是符合需求。

---

## 14. `np.broadcast_to`

**签名:**
```python
np.broadcast_to(a, shape)
```
- `shape`:目标形状,必须和 `a` 的原形状在广播规则下兼容,否则报错

**一句话:** 把一个数组按广播规则"虚拟地"扩展成更大的形状——不实际复制数据,只是创建一个"看起来更大"的只读 view。

**AI 研究场景:** 显式验证/调试广播行为(不确定两个数组能不能广播时,先用 `broadcast_to` 试算一下,能跑通就说明形状兼容);需要把广播的中间结果"物化"成真实数组传给某些不支持广播的第三方函数/C 扩展时,先 `broadcast_to` 再 `.copy()`;用一行代码把"广播规则在脑子里怎么推"变成可以直接验证的东西,适合调试和教学。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])                # shape (3,)
b = np.broadcast_to(a, (4, 3))          # 广播成(4,3),4行,每行都是[1,2,3]
assert b.shape == (4, 3)
assert np.array_equal(b[0], a)
assert np.array_equal(b[3], a)

# 验证:broadcast_to 的结果和手动运算的广播结果一致
mat = np.zeros((4, 3))
manual = mat + a
assert np.array_equal(b, manual)

# 广播不兼容时报错
try:
    np.broadcast_to(np.array([1, 2, 3]), (4, 4))   # (3,) 没法广播成(4,4)
    assert False
except ValueError:
    pass

# 结果是只读的,不能直接赋值
try:
    b[0, 0] = 999
    assert False
except ValueError:
    pass
```

**常见坑:** `broadcast_to` 返回的是**只读**(read-only)view,而且底层"看起来的多份数据"其实共享同一块内存(压根没有真的复制出多份)——尝试赋值会直接报 `ValueError: assignment destination is read-only`。需要一个可写的、真实占用内存的扩展结果时,要在后面加 `.copy()`。这也是理解 numpy 广播"零拷贝"本质的最佳例子——平时写 `a + b` 触发的广播,底层就是类似这样不真正复制内存的操作。

---

## 15. `np.flip` / `np.fliplr` / `np.flipud`

**签名:**
```python
np.flip(a, axis=None)   # axis 不写就翻转所有维度;可以只指定一个或多个轴
np.fliplr(a)             # left-right,水平翻转,固定等价于 flip(a, axis=1)
np.flipud(a)             # up-down,垂直翻转,固定等价于 flip(a, axis=0)
```

**一句话:** 沿指定维度把元素顺序颠倒过来(镜像)。

**AI 研究场景:** 图像数据增强里最常用的"水平翻转"(人脸、物体识别这类任务水平翻转后语义通常不变,是最廉价有效的增强手段之一,但要注意有方向语义的图像,比如文字,不适用);时间序列反转(双向 RNN 这类模型需要同时看正向和反向序列,输入 `flip` 一下就是反向序列);把递增序列变成递减序列构造测试数据。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3, 4])
assert np.array_equal(np.flip(a), [4, 3, 2, 1])

img = np.array([[1, 2, 3], [4, 5, 6]])

assert np.array_equal(np.fliplr(img), np.flip(img, axis=1))     # 水平翻转
assert np.array_equal(np.fliplr(img), [[3, 2, 1], [6, 5, 4]])

assert np.array_equal(np.flipud(img), np.flip(img, axis=0))     # 垂直翻转
assert np.array_equal(np.flipud(img), [[4, 5, 6], [1, 2, 3]])

# 数据增强:水平翻转一批图像(batch, H, W)
batch = np.random.randn(8, 4, 4)
flipped = np.flip(batch, axis=2)          # 只翻转 W 这一维,不影响 batch 和 H
assert flipped.shape == batch.shape
assert np.allclose(flipped[:, :, 0], batch[:, :, -1])   # 翻转后第0列 = 原来的最后一列
```

**常见坑:** `fliplr`/`flipud` 的命名来自"二维图像的左右/上下"这个直觉,但它们对高维数组也能用——`fliplr` 固定翻转 axis=1,`flipud` 固定翻转 axis=0,不管数组实际有几维。如果数组维度顺序不是你以为的 `(H, W, ...)`(比如批处理场景是 `(batch, H, W)` 而不是 `(H, W)`),`fliplr`/`flipud` 翻的可能不是你想要的那个方向。批处理场景更推荐直接用 `np.flip(a, axis=指定编号)` 明确写清楚要翻哪一维,不要想当然地用 `fliplr`/`flipud`。

---

## 小结:这一批 15 个函数解决的问题

| 函数 | 解决的问题 |
|---|---|
| `.shape`/`.reshape` | 查看/改变数组的形状(总元素数不变) |
| `.T`/`transpose` | 交换维度顺序(矩阵转置、通道顺序转换) |
| `swapaxes`/`moveaxis` | 只调整指定的一两个维度,不动其余维度 |
| `flatten`/`ravel` | 展平成一维(copy vs 尽量 view) |
| `squeeze` | 去掉大小为1的多余维度 |
| `expand_dims`/`newaxis` | 插入大小为1的新维度(凑广播、补batch维) |
| `concatenate` | 沿已有维度拼接,不增加新维度 |
| `stack` | 新增一个维度,把多个同形状数组摞起来 |
| `hstack`/`vstack` | concatenate 的水平/垂直语法糖(小心一维特例) |
| `split`/`array_split` | concatenate 的反操作:切分数组(严格/宽松整除) |
| `tile` | 重复"整个数组"(图案式铺贴) |
| `repeat` | 重复"每个元素"(元素级展开) |
| `pad` | 边缘填充(序列/图像对齐到统一尺寸) |
| `broadcast_to` | 显式、零拷贝地按广播规则扩展形状 |
| `flip`/`fliplr`/`flipud` | 沿指定维度镜像翻转 |

下一批:[03-indexing-and-selection.md](03-indexing-and-selection.md)

---

*更新:2026-07-07*
