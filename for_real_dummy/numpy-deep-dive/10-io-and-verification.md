# 10 · IO 与验证工具(IO & Verification Utilities)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批解决两类问题:**怎么把数组存到磁盘上/读回来**(存中间实验结果、存模型权重),以及**怎么确认一段 numpy 代码算出来的结果真的是对的**(浮点数怎么比、内存到底有没有共享)。这是 numpy 系列的最后一批,收尾之后接力棒交给 torch。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `np.save` / `np.load`

**签名:**
```python
np.save(file, arr)
np.load(file)
```
- `file`:文件路径(字符串);如果没写 `.npy` 后缀,`save` 会自动补上
- `arr`:要保存的数组
- `load` 读回的是一个和原来 shape、dtype 完全一致的 ndarray

**一句话:** 把单个 numpy 数组原样存成二进制的 `.npy` 文件,`load` 读回来时 shape、dtype、数值精度都不会有任何损失。

**AI 研究场景:** 跑一次耗时的推理/特征提取后,把结果(embedding 矩阵、中间激活值)存成 `.npy`,下次直接 `load` 复现,不用重新完整跑一遍实验;或者把某次实验用的输入数据固定存下来,方便不同代码版本之间做"结果是否一致"的回归验证。相比 Python 自带的 `pickle`,`.npy` 是 numpy 专门为"数组"设计的二进制格式,读写更快,读取时也不需要 `allow_pickle=True`(pickle 反序列化不受信任的数据存在执行任意代码的安全顾虑,数组本身不需要这个)。

**可运行例子:**
```python
import numpy as np
import tempfile, os

arr = np.arange(12, dtype=np.float32).reshape(3, 4)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "features.npy")
    np.save(path, arr)
    loaded = np.load(path)

    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype
    assert np.array_equal(loaded, arr)      # 二进制格式,读回来是精确相等,不是"接近"

    # 常见坑实测:文件名不写后缀,save 会自动补上 .npy
    path_noext = os.path.join(tmpdir, "no_ext")
    np.save(path_noext, arr)
    assert os.path.exists(path_noext + ".npy")
    assert not os.path.exists(path_noext)
```

**常见坑:** 文件名不写 `.npy` 后缀时,`np.save` 会自动补上(上面例子已实测)——如果之后手动拼路径去读文件,很容易出现"明明存了,读的时候却报文件不存在"的困惑。另外 `.npy` 是二进制格式,不能直接用文本编辑器打开阅读;需要人类可读的格式,看第 3 节的 `savetxt`。

---

## 2. `np.savez` / `np.savez_compressed`

**签名:**
```python
np.savez(file, **arrays)
np.savez_compressed(file, **arrays)
```
- `file`:路径,自动补 `.npz` 后缀
- `**arrays`:任意个数的关键字参数,`名字=数组`,这个名字就是之后取出时用的 key
- `savez_compressed` 参数完全一样,唯一区别是内部用 zip 压缩存储

**一句话:** 把多个数组打包存成一个 `.npz` 文件(本质是一个 zip 包),比逐个调 `save` 方便;`savez_compressed` 是同样的东西再加上压缩,通常体积更小但存取更慢。

**AI 研究场景:** 保存一整套模型权重时(比如手写 NumPy 实现的 MLP,每一层都有一个 `W` 和 `b`),用 `savez` 一次性打包成一个文件、每个数组带名字,而不是散落一堆 `w1.npy`、`b1.npy`、`w2.npy`……`load` 回来后可以用字典风格的 `ckpt["w1"]` 按名字取,管理起来清楚很多——这和 [04 教程](../04-how-to-practice-with-jupyter.md)里"给 notebook 留清楚笔记"是同一种"未来的自己会感谢现在的自己"的思路。

**可运行例子:**
```python
import numpy as np
import tempfile, os

w1 = np.random.randn(4, 3).astype(np.float32)
b1 = np.zeros(3, dtype=np.float32)
w2 = np.random.randn(3, 1).astype(np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.npz")
    np.savez(path, w1=w1, b1=b1, w2=w2)

    with np.load(path) as ckpt:
        assert set(ckpt.files) == {"w1", "b1", "w2"}
        assert np.array_equal(ckpt["w1"], w1)
        assert np.array_equal(ckpt["b1"], b1)

    # savez_compressed:同样的数据,体积通常更小(实测:802 字节 -> 621 字节)
    path_c = os.path.join(tmpdir, "model_compressed.npz")
    np.savez_compressed(path_c, w1=w1, b1=b1, w2=w2)
    with np.load(path_c) as ckpt_c:
        assert np.array_equal(ckpt_c["w1"], w1)

    assert os.path.getsize(path_c) < os.path.getsize(path)
```

**常见坑:** `np.load` 读 `.npz` 时返回的不是普通字典——是一个 `NpzFile` 对象(`isinstance(ckpt, dict)` 是 `False`),取值方式很像字典(`ckpt["w1"]`、`ckpt.files` 看有哪些 key),但它按需惰性读取,用完最好像上面例子一样用 `with` 语句或者手动 `.close()`。另外如果偷懒用位置参数 `np.savez(path, w1, b1)` 不写名字,key 会自动变成 `arr_0`、`arr_1`,读的时候完全看不出对应哪个变量——显式命名基本是必须的。压缩版本省的是磁盘空间,花的是 CPU 时间(压缩/解压都要计算),数据基本是随机数(不好压缩)时收益有限,数据有很多重复值/规律(比如很多 0)时收益明显。

---

## 3. `np.savetxt` / `np.loadtxt`

**签名:**
```python
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', header='')
np.loadtxt(fname, delimiter=None, skiprows=0)
```
- `X`:要存的数组,只支持 1 维或 2 维(没有更高维的文本表示)
- `fmt`:每个数字的格式化字符串,默认是很高精度的科学计数法;写窄了(比如 `%.4f`)会截断精度
- `delimiter`:分隔符,存 CSV 就传 `","`;`loadtxt` 默认按任意空白分隔,要和存的时候用的分隔符对应上

**一句话:** 把数组存成纯文本(人眼可以直接打开读懂的那种),`loadtxt` 再读回来;和二进制的 `.npy` 相反,这是"可读但不一定精确、通常更占空间"的格式。

**AI 研究场景:** 需要一份人可以直接打开看、或者拿去 Excel/别的非 Python 工具处理的小型结果表格时用,比如每个 epoch 的 loss/准确率汇总表。日常训练里权重、特征矩阵这些大数据几乎不用文本格式——同样的数据文本存储通常比二进制大得多,读写也慢;只有"数据量小 + 需要跨工具/人工检查"这个具体场景才该选文本,大规模数值数据老老实实用第 1、2 节的二进制格式。

**可运行例子:**
```python
import numpy as np
import tempfile, os

data = np.array([[0, 1/3, 0.75], [1, 2/3, 0.82], [2, 1/7, 0.91]])

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "metrics.csv")
    np.savetxt(path, data, fmt="%.4f", delimiter=",", header="epoch,loss,acc")

    loaded = np.loadtxt(path, delimiter=",")
    assert np.allclose(loaded, data, atol=1e-4)     # 数值验证要用 allclose,不能用 array_equal

    # 精度确实被 fmt="%.4f" 截断了(实测):
    # data[0, 1] 原始是 0.3333333333333333,存成4位小数再读回来变成 0.3333
    assert not np.array_equal(loaded, data)
```

**常见坑:** 文本格式默认按 `fmt` 指定的精度截断——例子里 `1/3` 存成 4 位小数的文本再读回来就不再和原始值精确相等,所以验证 `loadtxt` 的结果**必须用 `np.allclose`,不能用 `array_equal`**(下一节详细讲这两者的区别)。另外 `loadtxt` 默认按空白分隔,存 CSV 时如果忘了传 `delimiter=","` 保持一致,读的时候会报错或者解析错位。

---

## 4. `np.allclose` / `np.isclose`

**签名:**
```python
np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
```
- 判断公式:`|a - b| <= atol + rtol * |b|`
- `rtol`:相对容差——允许的误差跟着 `b` 的数值大小按比例放大
- `atol`:绝对容差——不管数值多大都固定的误差余量,数值在 0 附近时起决定作用
- `equal_nan`:是否把两边都是 `nan` 判定成"相等"(默认不是——`nan` 默认连自己都不等于自己)

**一句话:** `isclose` 是逐元素比较,返回一个和输入同形状的布尔数组;`allclose` 是"是否全部元素都满足 isclose"的汇总,返回单个 `True`/`False`——`allclose(a, b)` 本质上就是 `isclose(a, b).all()`。

**AI 研究场景:** 这是 [04-how-to-practice-with-jupyter.md](../04-how-to-practice-with-jupyter.md) 里"用 assert 验证结果"这套方法论背后最核心的工具——浮点数几乎不可能精确相等(累积的舍入误差、CPU/GPU 不同计算顺序导致的微小差异),验证"手写的反向传播和 PyTorch autograd 算出来的梯度是否一致""换个实现方式结果有没有变",全靠 `allclose`,不是 `==`。

**可运行例子:**
```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0, 3.0 + 1e-6])

elementwise = np.isclose(a, b)
assert elementwise.dtype == bool
assert list(elementwise) == [True, True, True]

# allclose 就是 isclose 的结果再 all() 一下 —— 用代码验证这个关系本身
assert np.allclose(a, b) == bool(np.isclose(a, b).all())
assert np.allclose(a, b) == True

# 收紧 atol 后,原本"够近"的会被判定为不同(第三个元素差了 1e-6,远大于 1e-9)
assert not np.allclose(a, b, rtol=0, atol=1e-9)

# nan 默认不等于自己,isclose/allclose 也遵守这个规则,除非显式 equal_nan=True
assert np.isclose(np.nan, np.nan) == False
assert np.isclose(np.nan, np.nan, equal_nan=True) == True
assert np.allclose([1.0, np.nan], [1.0, np.nan]) == False
assert np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True) == True
```

**常见坑:** `rtol`/`atol` 是加在一起生效的(`atol + rtol * |b|`),不是二选一。数值在 0 附近时 `rtol * |b|` 几乎是 0,起决定作用的是 `atol`;数值很大时 `atol` 几乎可以忽略,起决定作用的是 `rtol`——只调其中一个不代表控制住了整体的比较精度。另一个容易忽略的地方:比较里含 `nan` 时,默认整个比较判定为"不相等"(哪怕两边都是 `nan`),这是刻意的数学约定(IEEE 754 标准规定 `nan != nan`),需要的话显式传 `equal_nan=True`。

---

## 5. `np.array_equal` / `np.array_equiv`

**签名:**
```python
np.array_equal(a1, a2, equal_nan=False)
np.array_equiv(a1, a2)
```
- 两者都**没有** `rtol`/`atol`——不允许任何容差,本质是"形状相同 + 逐元素 `==` 全部为真"
- `array_equiv` 多一条规则:两个数组形状不同,但其中一个可以广播成另一个的形状时,广播后再比较

**一句话:** 和第 4 节的 `allclose` 是完全不同的另一种比较——`array_equal`/`array_equiv` 要求精确相等,不允许任何浮点误差;`array_equiv` 只是在此基础上多接受"形状可以广播"的情况。

**AI 研究场景:** 验证不该有浮点误差的场景,比如 tokenizer 输出的 token id 序列是否一致、分类模型预测的整数标签是否一致、两个数组的 shape 是否完全对得上——这些是整数或结构性的比较,不相等就是真的错了,不需要(也不应该)用 `allclose` 的容差去"放过"。

**可运行例子:**
```python
import numpy as np

ids1 = np.array([1, 2, 3])
ids2 = np.array([1, 2, 3])
assert np.array_equal(ids1, ids2)

ids3 = np.array([[1, 2, 3]])             # 形状 (1,3),和 ids1 的 (3,) 不同
assert not np.array_equal(ids1, ids3)     # array_equal 要求形状完全一致
assert np.array_equiv(ids1, ids3)         # array_equiv 允许广播后比较,(3,) 能广播成 (1,3)

# 浮点数不要用这两个做验证!
x = np.array([0.1 + 0.2])
y = np.array([0.3])
assert x[0] != y[0]                       # 0.30000000000000004 != 0.3(浮点精度问题,不是bug)
assert not np.array_equal(x, y)           # 精确比较,判定不相等
assert np.allclose(x, y)                  # 这种场景该用 allclose,不是 array_equal
```

**常见坑:** 千万不要拿 `array_equal` 去验证任何经过浮点运算的结果(哪怕数学上"应该"相等)——累积的浮点误差几乎总会导致判定为不相等,报出一个"看起来明明是对的却 assert 失败"的诡异结果,这时候该换成 `allclose`。`array_equiv` 的广播比较用得不多,大多数场景老老实实用 `array_equal` 就够,只有明确需要"形状允许广播"时才用 `array_equiv`。

---

## 6. `.copy()` vs 视图(view)语义

**签名(这一节严格来说是几种"行为"而不是单个函数):**
```python
a.copy()                # 显式拷贝,永远返回独立内存的新数组
a[i:j]                  # 基础切片(冒号切片)—— 通常返回视图(共享内存)
a[[i, j, k]]             # 花式索引(列表/数组做下标)—— 返回拷贝
a.reshape(shape)         # 内存连续时 —— 返回视图
```

**一句话:** "视图"(view)是"看向同一块内存的另一个窗口"——改视图,原数组跟着变;`.copy()` 是显式复制出一份完全独立的内存——改哪个都不影响另一个。numpy 里各种操作到底默默返回的是视图还是拷贝,规则并不统一,是数值 bug 最常见的来源之一。

**AI 研究场景:** 从一个大数组里切一部分做实验、取一个 mini-batch,但不能影响原始数据——如果误以为切片是独立拷贝,直接在切片上原地修改(比如归一化、mask 赋值),原始数据会被静默污染且**不报错**。这是最难排查的一类 bug,因为程序表现"正常运行",只是某个看似不相关的地方结果不对。

**可运行例子(用 `np.may_share_memory` 实测,不凭记忆猜):**
```python
import numpy as np

original = np.arange(10)

# 1. 基础切片(冒号切片)-> 视图,共享内存
sl = original[2:5]
assert np.may_share_memory(original, sl)      # 实测确实共享内存
sl[0] = 999
assert original[2] == 999                      # 改视图,原数组跟着变了!

# 2. 花式索引(传入列表做下标)-> 拷贝,不共享内存
original2 = np.arange(10)
fancy = original2[[2, 3, 4]]
assert not np.may_share_memory(original2, fancy)
fancy[0] = 999
assert original2[2] == 2                        # 改拷贝,原数组不受影响

# 3. 布尔索引 -> 同样是拷贝
mask = original2 > 5
selected = original2[mask]
assert not np.may_share_memory(original2, selected)

# 4. reshape 在内存连续时通常是视图
reshaped = original2.reshape(2, 5)
assert np.may_share_memory(original2, reshaped)

# 5. 需要独立数据时,显式 .copy() 保证安全,不管原本是不是视图
safe = original[2:5].copy()
assert not np.may_share_memory(original, safe)

# 6. 旁注:显式的 .view() 方法 —— 重新解释同一块内存的 dtype(不是数值转换,是重新解释比特!)
f = np.array([1.0], dtype=np.float32)
as_int = f.view(np.int32)
assert as_int[0] != 1     # 不是数值1,是 float32 数值 1.0 的比特模式被当 int32 读出来的怪数字

# 7. .view() 要求字节数能整除新 dtype 的大小,凑不整就报错
f3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # 3 个 float32 = 12 字节
try:
    f3.view(np.int64)                               # 12 字节除不尽 8 字节(int64一个的大小)
    assert False, "应该报错但没报错"
except ValueError:
    pass                                             # 符合预期
```

**常见坑:** 判断"视图还是拷贝"不能靠猜,大致的经验规律是——**基础切片(冒号切片)、reshape(内存连续时)、`.T`/转置都是视图;花式索引(列表/数组做下标)、布尔索引、大部分"新计算出结果"的函数都是拷贝**——但这只是经验规律不是保证,不确定时用下一节的 `np.may_share_memory` 实测,或者干脆需要独立数据就显式 `.copy()`,不要心存侥幸。另外容易和"视图"这个概念混在一起的是上面例子里第 6/7 步用到的 `ndarray.view()` 方法——它是显式把同一块内存**重新解释**成另一个 dtype(不是数值转换,是重新解释比特!),而且要求字节数能整除新 dtype 的大小,凑不整就报 `ValueError`,这和"切片/reshape 隐式产生的视图"是两回事,只是恰好都叫"view"容易混淆。

---

## 7. `.astype`

**签名:**
```python
a.astype(dtype, copy=True)
```
- `dtype`:目标类型
- `copy`:默认 `True`——即使目标 dtype 和原数组完全相同,默认也会返回一个新数组;只有显式传 `copy=False` 且 dtype 恰好相同时,才可能省掉这次拷贝

**一句话:** 把一个**已经存在**的数组转换成另一个 dtype,返回一个新数组,默认不修改原数组。

**AI 研究场景:** 模型权重要求 float32 但读进来的数据是 float64、混合精度训练要把某些张量转成 float16、把 bool 类型的 mask 转成 float 参与数值乘法、把 argmax 算出来的索引转成 int 当下标——这些场合都要显式 `astype`。

**和创建时 `dtype=` 的区别:** `dtype=` 是在**创建数组的时刻**指定类型(比如 `np.array([1,2,3], dtype=np.float32)`,数据是新生成的,压根不存在"转换"这一步);`astype` 是对一个**已经存在**的数组重新申请内存、按新类型转换每个元素的值,是有实际计算开销的操作,不是免费的。

**可运行例子:**
```python
import numpy as np

x = np.array([1.5, 2.7, 3.2], dtype=np.float64)
x32 = x.astype(np.float32)

assert x32.dtype == np.float32
assert x.dtype == np.float64            # 原数组没有被修改
assert np.allclose(x, x32, atol=1e-6)   # 数值上几乎相等(float32精度稍低,不能用 array_equal)

# 浮点转整数是截断(不是四舍五入!)
y = np.array([1.9, -1.9, 2.5])
y_int = y.astype(np.int32)
assert list(y_int) == [1, -1, 2]         # 向 0 截断,不是 round()

# bool 转 float,常用于 mask 参与数值计算
mask = np.array([True, False, True])
mask_f = mask.astype(np.float32)
assert np.array_equal(mask_f, [1.0, 0.0, 1.0])

# 默认 copy=True:即使目标 dtype 相同,也返回独立的新数组
z = np.array([1, 2, 3])
z_same = z.astype(np.int64, copy=True)
assert not np.may_share_memory(z, z_same)
```

**常见坑:** 浮点转整数是**截断取整(向零方向截断)**,不是四舍五入——`1.9` 变成 `1`,`-1.9` 变成 `-1`,不是很多人以为的 `round()` 行为,需要四舍五入要先 `np.round()` 再 `astype`。另外从大精度转小精度(float64→float32,或更极端的 float16/int8)可能悄悄损失精度甚至溢出,不会报错——排查"看起来对但训练发散/精度莫名下降"时,值得检查是不是哪里做了不合适的 astype。

---

## 8. `np.errstate`

**签名:**
```python
with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
    ...
```
- 每个参数控制一类数值事件的处理策略:`'ignore'`(忽略)、`'warn'`(警告,大部分事件的默认值)、`'raise'`(直接抛异常)等
- `divide`:除零;`over`:上溢(结果太大存不下);`under`:下溢(结果太小,精度不够表示);`invalid`:无效运算(比如 `0/0`、负数开方、`log(负数)`)

**一句话:** 一个上下文管理器(context manager),临时改变 numpy 遇到除零/溢出/无效运算这类数值事件时的反应方式,`with` 块结束后自动恢复成原来的设置。

**AI 研究场景:** 有些运算你**明知道**会产生 `inf`/`nan`(比如 `log(0)` 故意用来表示"概率为 0 对应负无穷的对数似然"、除以一个可能为 0 的分母之后再用 `np.where`/`np.nan_to_num` 处理特殊值),但 numpy 默认会为此打印一堆 `RuntimeWarning` 刷屏,还容易掩盖真正需要关注的警告——用 `errstate` 精确圈定"这一小段代码里,这类警告是预期内的,不用报",而不是在脚本开头一次性全局关掉(那样其他地方真正 bug 产生的 nan 也会被悄悄放过)。

**可运行例子:**
```python
import numpy as np

with np.errstate(divide="ignore", invalid="ignore"):
    a = np.array([1.0, 0.0, -1.0])
    b = np.array([0.0, 0.0, 0.0])
    result = a / b            # 1/0=inf, 0/0=nan, -1/0=-inf —— 都是预期内的特殊值

assert np.isinf(result[0])
assert np.isnan(result[1])
assert np.isinf(result[2])

# errstate 只在 with 块内生效,块外自动恢复默认设置
current = np.geterr()
assert current["divide"] == "warn"     # 没有被永久改掉

# log(0) 故意允许产生 -inf 的场景(比如对数似然)
with np.errstate(divide="ignore"):
    log_probs = np.log(np.array([0.5, 0.0, 1.0]))
assert log_probs[1] == -np.inf
```

**常见坑:** `errstate` 只能控制 numpy **自己内部**的浮点异常处理策略,不是 Python 通用的 `warnings.filterwarnings`,更管不了非 numpy 抛出的异常。范围一定要缩小到"确实预期会产生特殊值"的那几行代码,不要图省事在脚本开头用 `np.seterr(all='ignore')` 全局关掉——这样做之后,其他地方因为真正 bug 产生的 nan/inf 会被一起静默放过,之后排查问题反而更难。

---

## 9. `np.may_share_memory`

**签名:**
```python
np.may_share_memory(a, b, max_work=None)
```
- `a`, `b`:两个数组
- `max_work`:排查内存重叠问题时愿意花的计算量,默认(`may_share_memory` 的默认值)只做一次快速的内存边界检查,不是精确判断实际元素是否重叠

**一句话:** 检测两个数组是否**可能**共享同一块底层内存的诊断工具——上一节 view/copy 的判断就是靠它实测出来的,不是靠猜或者靠"经验规律"。

**AI 研究场景:** 排查"改了一个数组,另一个看起来毫不相关的数组也跟着变了"这类诡异 bug 时的第一件事——在怀疑的两个数组之间跑一下 `np.may_share_memory`,比盯着代码看半天更快确认是不是视图共享内存导致的意外修改。写单元测试时,也可以显式断言某个函数返回的是独立拷贝(不该和输入共享内存),防止未来重构不小心改成返回视图,悄悄引入 bug。

**可运行例子:**
```python
import numpy as np

a = np.arange(10)
view = a[::2]           # 跳步切片,仍然是视图
copy = a[::2].copy()

assert np.may_share_memory(a, view) == True
assert np.may_share_memory(a, copy) == False

b = np.arange(10)                          # 完全无关的另一个数组
assert np.may_share_memory(a, b) == False

# "may"(可能)这个名字的含义:它默认只做内存边界检查,可能"保守地"判定为共享,
# 这是 numpy 官方文档自带的经典例子 —— 同一个二维数组的两"列"
grid = np.zeros((3, 4))
col0 = grid[:, 0]
col1 = grid[:, 1]
assert np.may_share_memory(col0, col1) == True    # 边界重叠 -> 判定为"可能共享"
assert np.shares_memory(col0, col1) == False       # 精确检查:两列的具体元素其实完全不相交
```

**常见坑:** `may_share_memory` 为了速度,默认只检查两个数组底层内存的"地址范围"有没有重叠,重叠就返回 `True`——但地址范围重叠不代表两个数组真正共用的是同一批元素(上面 `grid` 的两列就是例子:同一块内存,但列 0 和列 1 的元素其实完全不同)。名字里的"may"就是在提醒你这一点:**返回 `True` 只说明"可能共享",不代表一定共享;返回 `False` 则可以放心——确实不共享**。真正需要精确答案(而不是"快速但保守"的答案)时,用更严格的 `np.shares_memory`(默认更慢,必要时可以设置 `max_work` 控制最坏情况下的计算开销)。

---

## 10. `np.set_printoptions`

**签名:**
```python
np.set_printoptions(precision=8, suppress=False, threshold=1000, linewidth=75)
```
- `precision`:打印浮点数保留几位小数(默认 8)
- `suppress`:是否禁止科学计数法(默认 `False`,允许极小/极大数值用科学计数法显示;设成 `True` 统一用普通小数显示)
- `threshold`:数组总元素数超过这个值时,打印会用 `...` 省略中间部分(默认 1000)
- `linewidth`:每行打印的字符宽度,超过会自动换行

**一句话:** 一个全局设置打印格式的调试工具——只改变"打印/`repr` 显示出来的样子",不改变数组本身存储的真实数值和精度。

**AI 研究场景:** 调试模型输出时,默认打印会用科学计数法(比如 `1.23456789e-05`)且大数组会自动省略中间部分(`[0. 1. 2. ... 997. 998. 999.]`),想看清楚某个具体位置的完整数值(比如确认两个几乎相等的数到底哪一位开始不一样)就需要调宽 `precision`、关掉科学计数法、调大 `threshold`——纯粹是调试时"看清楚数字"的工具,不参与任何计算。

**可运行例子:**
```python
import numpy as np

x = np.array([1/3, 2/3, 1e-10])
original_options = np.get_printoptions()      # 先记住原始设置,方便还原

np.set_printoptions(precision=3, suppress=True)
text = repr(x)
assert "0.333" in text
assert "e-" not in text          # suppress=True:不用科学计数法,极小值显示成接近0的小数

np.set_printoptions(precision=8, suppress=False)
assert "e-" in repr(x)           # 恢复默认后,1e-10 这种极小值又变回科学计数法显示

big = np.arange(2000)
np.set_printoptions(threshold=1000)
assert "..." in repr(big)         # 超过 threshold,自动折叠中间部分

np.set_printoptions(threshold=10000)
assert "..." not in repr(big)     # 调大 threshold 后,完整打印

np.set_printoptions(**original_options)   # 还原,不影响后面的代码

# 更推荐的写法:with np.printoptions(...) 上下文管理器,自动还原,不用自己记着调回去
with np.printoptions(precision=2):
    assert repr(np.array([2.0]) / 3) == "array([0.67])"
assert np.get_printoptions()["precision"] == original_options["precision"]   # 出了 with 块自动恢复
```

**常见坑:** `set_printoptions` 只改**打印显示**的样子,不改数组底层的真实精度——调小 `precision` 只是让你少看到几位小数,数组内部依然是完整的 float32/float64 精度,不会因此损失任何计算结果。这是**全局**设置,改了之后会一直影响后面所有代码的打印(包括不相关的地方),写 notebook/教程代码需要临时调整时,优先用 `with np.printoptions(...):` 这个上下文管理器——和第 8 节的 `errstate` 是同一种思路,离开 `with` 块自动恢复,不用自己操心"记得改回去"。

---

## 小结:这一批 10 个主题解决的问题

| 主题 | 解决的问题 |
|---|---|
| `save`/`load` | 单个数组的二进制存取(.npy),精确无损 |
| `savez`/`savez_compressed` | 多个数组打包存取(.npz),模型权重一类场景;后者额外压缩体积 |
| `savetxt`/`loadtxt` | 人类可读的文本存取,小数据/跨工具场景,注意精度可能被截断 |
| `allclose`/`isclose` | 浮点数"足够接近"的验证——assert 方法论的核心工具 |
| `array_equal`/`array_equiv` | 精确相等验证(整数/结构性比较,不允许浮点误差) |
| `.copy()` vs 视图 | 内存是否共享——数值 bug 的常见来源 |
| `.astype` | 已有数组的类型转换(区别于创建时的 `dtype=`) |
| `errstate` | 临时控制除零/溢出等数值警告的处理方式 |
| `may_share_memory` | 诊断两个数组是否共享内存的工具本身,及其"保守判断"的边界 |
| `set_printoptions` | 调试用的打印格式控制,不影响真实精度 |

---

numpy 系列到这里,从 [01-creation-and-init.md](01-creation-and-init.md) 的创建初始化,一路到本篇的 IO 与验证,一共覆盖了约 **120 个函数**——这也是本系列 numpy 部分的收官。下一步,同样"逐函数精读、AI 研究场景、assert 验证"这套方法论会延续到 torch,衔接 [02-pytorch-basics.md](../02-pytorch-basics.md):tensor 的创建、索引、自动求导、GPU 相关的函数,很多你已经在 numpy 里见过影子(比如这篇的 `allclose` 对应 `torch.allclose`,`.astype` 对应 `.to(dtype)`),torch 系列会把这些对应关系一个个讲清楚。这里先卖个关子,具体内容留到 torch 系列展开。

---

*更新:2026-07-07*
