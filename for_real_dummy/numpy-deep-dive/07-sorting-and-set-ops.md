# 07 · 排序与集合运算(Sorting & Set Ops)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决两类问题:**怎么把数组变得有序**(`sort`/`argsort`/`partition` 一家子),以及**把数组当"集合"来做交集/并集/差集/成员判断**(`intersect1d`/`union1d`/`setdiff1d`/`isin`)——外加一个建立在"有序"基础上的利器:二分查找 `searchsorted`。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `np.sort`

**签名:**
```python
np.sort(a, axis=-1, kind=None, order=None)
```
- `a`:要排序的数组
- `axis`:沿哪个轴排序,默认 `-1`(最后一个轴——对二维数组来说就是"每一行内部单独排序");传 `axis=None` 会先拉平成一维再整体排序
- `kind`:排序算法,`'quicksort'`(默认,内部其实是更快的 introsort 变体)/`'mergesort'`/`'heapsort'`/`'stable'`。绝大多数场景不用管,只有需要"相同值保持原始相对顺序"(稳定排序)时才需要显式传 `kind='stable'`(新版本 numpy 也支持更直接的 `stable=True` 关键字,两者不能同时传)
- `order`:给结构化数组(structured array)按字段名排序用的,研究代码里很少用到

**一句话:** 返回一个新的、排好序的数组(默认**升序**),不改变原数组本身。

**AI 研究场景:**
- 把一批样本的 loss/置信度排序后可视化或按阈值筛选(比如"看看 loss 最高的那些样本长什么样",排查数据是否有脏样本)。
- 中位数、分位数等统计量的底层实现基本都是先排序(后面 05 批的 `median`/`percentile` 本质上都基于排序)。
- 小规模、要求精确的最近邻检索:对一批候选和查询点的距离排序,取最近的几个(数据量大时会换成近似最近邻算法,但排序是理解这一整类问题的基础)。

**可运行例子:**
```python
import numpy as np

a = np.array([3, 1, 2])
s = np.sort(a)
assert list(s) == [1, 2, 3]
assert list(a) == [3, 1, 2]        # np.sort 返回副本,原数组不变

b = np.array([3, 1, 2])
b.sort()                            # ndarray.sort() 是原地排序方法,没有返回值
assert list(b) == [1, 2, 3]

# 降序:没有 ascending 参数,标准写法是负号技巧或反转切片
desc1 = np.sort(a)[::-1]
desc2 = -np.sort(-a)
assert list(desc1) == [3, 2, 1]
assert list(desc2) == [3, 2, 1]

# 二维数组:axis=-1(默认)是每一行内部单独排序
m = np.array([[3, 1], [2, 4]])
assert np.array_equal(np.sort(m, axis=-1), [[1, 3], [2, 4]])
assert np.array_equal(np.sort(m, axis=0), [[2, 1], [3, 4]])   # axis=0 是每一列单独排序
```

**常见坑:**
- `np.sort(a)` 返回**副本**,不改变 `a`;而 `a.sort()`(数组自带的方法,不是 `np.sort`)是**原地**排序、没有返回值——这两个经常被搞混,写成 `a = a.sort()` 是常见错误(得到的会是 `None`,因为原地方法不返回值)。
- numpy **没有** `ascending=False` 这种参数(传了直接 `TypeError`)。降序的标准写法是 `np.sort(a)[::-1]` 或者 `-np.sort(-a)`(对原数组取负、排序、再取负——两次取负抵消,升序变降序,数值上完全等价)。
- `kind` 参数控制的是**排序算法实现**(quicksort/mergesort/heapsort/stable),**不是排序方向**——如果因为想要降序而尝试给 `kind` 传奇怪的值(比如 `kind='descending'`),会直接报 `ValueError`,不会神奇地变成降序。`kind` 唯一常用的场景是需要"相同值保持原始相对顺序"的稳定排序,这时才传 `kind='stable'`。

---

## 2. `np.argsort`

**签名:**
```python
np.argsort(a, axis=-1, kind=None, order=None)
```
参数含义和 `np.sort` 完全一样,唯一区别在返回值。

**一句话:** 不返回排好序的值,而是返回"排序后每个位置对应原数组里的下标"——换句话说,告诉你"按什么顺序取原数组的元素,才能拿到排好序的结果"。

**AI 研究场景:**
- **按某一列排序整个表格/数组(最经典的写法):** 数据是 `(N, D)` 的表格,每行是一条样本,想按某一列(比如"分数")给所有行排序,同时保持每行内部其他字段的对应关系不变——这必须先对那一列 `argsort` 拿到"行的排列顺序",再用**花式索引**把整个数组按这个顺序整行重排。直接对某一列单独调用 `sort` 只会打乱行与行之间的对应关系(下面例子会展示这个反面教材)。
- 排序后仍要保留"原始是第几号样本"这个信息(比如打印"loss 最高的是第几个样本"),`argsort` 的返回值本身就是原始下标,不需要额外维护一张映射表。
- 配合切片取 top-k(下一节的 `partition` 是这个需求更快的版本)。

**可运行例子:**
```python
import numpy as np

scores = np.array([50, 10, 30])
order = np.argsort(scores)
assert list(order) == [1, 2, 0]          # 10在原数组下标1,30在下标2,50在下标0
assert list(scores[order]) == [10, 30, 50]

# 经典写法:按某一列给整个二维数组排序(花式索引)
# 每一行是 [sample_id, score]
data = np.array([
    [1, 88.5],
    [2, 71.2],
    [3, 95.0],
    [4, 60.3],
])
row_order = np.argsort(data[:, 1])        # 按第1列(score)升序排出"行的顺序"
sorted_data = data[row_order]             # 花式索引:整行整行地重排,不破坏行内对应关系
assert list(sorted_data[:, 0]) == [4, 2, 1, 3]     # id顺序,和score对应
assert list(sorted_data[:, 1]) == sorted(data[:, 1].tolist())

# 降序同理:对负值 argsort
row_order_desc = np.argsort(-data[:, 1])
sorted_desc = data[row_order_desc]
assert list(sorted_desc[:, 0]) == [3, 1, 2, 4]

# 反面教材:只对某一列调用 sort,行与行的对应关系被破坏(不报错,但语义错了)
broken = data.copy()
broken[:, 1] = np.sort(broken[:, 1])       # 只排了score这一列,id列没有跟着动
assert list(broken[:, 0]) == [1, 2, 3, 4]   # id还是原来的顺序
assert list(broken[:, 1]) == [60.3, 71.2, 88.5, 95.0]  # 但score已经被打乱重排
# 结果:id=1现在"对应"着60.3,然而原始数据里id=1对应的其实是88.5——数据关联已经错乱
```

**常见坑:** `argsort` 返回的是**下标数组**,不是排好序的值——拿到结果后忘记再做一次索引(`scores[order]`)是最常见的疏忽。更隐蔽的错误是上面反面教材展示的:只对某一列调用 `sort`(而不是 `argsort` 再花式索引整个数组),会把这一列的值和其他列的值错误地"对应"起来——这个错误**不会报错**,只会默默产生一个语义错误但形状完全正常的结果,排查起来很麻烦,值得反复留意。

---

## 3. `np.partition` / `np.argpartition`

**签名:**
```python
np.partition(a, kth, axis=-1)
np.argpartition(a, kth, axis=-1)
```
- `kth`:一个整数(或整数数组),表示"分界位置"的下标;可以是负数,和切片一样从末尾数(比如 `kth=-3` 表示"倒数第3个位置")
- `axis`:同 `sort`,默认沿最后一个轴

**一句话:** **"不完全排序"**——只保证排序完成后本该出现在第 `kth` 位置的那个值,真的被放到了第 `kth` 位置上,并且它左边的所有元素都 `<=` 它、右边的所有元素都 `>=` 它;但左边内部、右边内部都**不保证任何顺序**。`argpartition` 是它的下标版本,返回下标而不是值(和 `argsort`/`sort` 的关系一样)。

**AI 研究场景:**
- **Top-K 全家桶:** 计算 Top-5 准确率时,只需要知道"真实标签的 logit 是否排进了前5",完全不需要知道这前5个谁大谁小的相对顺序;推荐系统的召回阶段要从几百万候选里先粗筛出 Top-1000 送给下一阶段精排,这时候要的是"这1000个是谁",而不是"这1000个内部的精确排名"(排名交给下游更贵的精排模型)。这类场景下对全体做完整 `sort` 纯属浪费。
- **性能:** 完整排序是比较排序,复杂度 `O(n log n)`;`partition`/`argpartition` 内部用的是选择算法(numpy 里叫 introselect,快速选择的变体),平均复杂度只有 `O(n)`。数组越大、且只要 k 相对 n 比较小,这个差距越明显——下面例子里 500 万元素的数组上实测了这个差距。

**可运行例子:**
```python
import numpy as np
import time

# 基础语义验证
a = np.array([7, 2, 9, 1, 5, 8, 3])
k = 3
p = np.partition(a, k)
sorted_a = np.sort(a)
assert p[k] == sorted_a[k]                 # 第k位置的值 == 完整排序后第k位置的值
assert np.all(p[:k] <= p[k])               # 左边都更小(或相等)
assert np.all(p[k + 1:] >= p[k])           # 右边都更大(或相等)

# 常见误解验证:两侧内部并不是有序的(数组大一点才看得出来,小数组容易巧合排好)
np.random.seed(1)
big = np.random.randint(0, 100000, size=500)
k2 = 250
p2 = np.partition(big, k2)
assert list(p2[:k2]) != sorted(p2[:k2].tolist())        # 左边不是有序的!
assert list(p2[k2 + 1:]) != sorted(p2[k2 + 1:].tolist())  # 右边也不是!

# 切片技巧:最小的K个 / 最大的K个
K = 3
smallest_K = np.partition(a, K)[:K]
largest_K = np.partition(a, -K)[-K:]
assert set(smallest_K.tolist()) == {1, 2, 3}
assert set(largest_K.tolist()) == {7, 8, 9}

# argpartition + top-k 的标准recipe(推荐系统召回、top-k准确率场景)
scores = np.array([0.2, 0.9, 0.5, 0.95, 0.1, 0.7, 0.99, 0.3])
k3 = 3
topk_idx = np.argpartition(scores, -k3)[-k3:]           # 最大的3个,下标顺序不保证
assert set(scores[topk_idx].round(2).tolist()) == {0.9, 0.95, 0.99}
# 如果还需要这k个内部有序,再单独对这一小撮排序(仍然比对全体排序快,因为 k << n)
topk_sorted = topk_idx[np.argsort(-scores[topk_idx])]
assert list(scores[topk_sorted]) == [0.99, 0.95, 0.9]

# 实测速度差距:数组越大,partition相对sort的优势越明显
n = 5_000_000
arr = np.random.rand(n)


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


t_sort = timeit(lambda: np.sort(arr))
t_partition = timeit(lambda: np.partition(arr, 100))
assert t_partition < t_sort * 0.7        # 本机实测:partition耗时约为sort的30%~40%

# argpartition返回下标;a[argpartition结果] 和 partition(a,k) 性质相同但未必逐位置相同
idx = np.argpartition(big, k2)
via_idx = big[idx]
assert via_idx[k2] == p2[k2]                                   # 第k位置的值一致
assert set(via_idx[:k2].tolist()) == set(p2[:k2].tolist())     # 左侧集合一致
assert set(via_idx[k2 + 1:].tolist()) == set(p2[k2 + 1:].tolist())  # 右侧集合一致
```

**常见坑:**
- **两侧不是有序的,别被结果或者小数组的巧合误导。** 用很小的数组测试时(几个到二十来个元素)经常会"恰好"得到完全有序的结果(小规模子数组内部容易退化成插入排序),容易让人误以为 `partition` 的两侧默认也是有序的——数组大一点(几百个元素以上,如上面例子)这个假象就会消失。如果确实需要局部有序,必须自己再对那一部分单独 `sort` 一次。
- `argpartition` 返回下标,`a[np.argpartition(a, k)]` 和 `np.partition(a, k)` 满足的**性质相同**(第 k 位置的值相同、左右两侧的值集合相同),但**逐位置未必完全一样**——两侧内部具体是什么排列本来就没有定义,不要写单元测试去断言两者逐元素相等。
- `kth` 是"第几小"的**下标**,不是"要多少个"。配合切片记两个短句就够用:想要最小的 K 个,写 `np.partition(a, K)[:K]`;想要最大的 K 个,写 `np.partition(a, -K)[-K:]`。直接使用 `np.partition(a, K)` 而不切片,拿到的是一个和原数组等长的数组(只是"第K位置正确"),不是"前K个"。

---

## 4. `np.intersect1d`

**签名:**
```python
np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
```
- `ar1`/`ar2`:两个数组,会被当成"集合"处理(自动去重)
- `assume_unique`:如果确定输入本身没有重复值,设 `True` 能跳过内部去重步骤,略微提速
- `return_indices`:是否同时返回交集元素在 `ar1`、`ar2` 里各自的位置下标

**一句话:** 返回两个数组的**交集**——两边都出现过的元素,结果自动去重并升序排列。

**AI 研究场景:** 做消融实验或者复现实验时,经常需要确认"这次跑的验证集样本 id"和"上次跑的验证集样本 id"是不是同一批——直接拿两个 id 数组算交集,如果交集大小等于两边各自的大小,说明是同一批样本;差一点都可能意味着数据划分脚本被改动过、或者随机种子没对齐,这是复现实验时排查"数字对不上"问题的第一步。

**可运行例子:**
```python
import numpy as np

run1_ids = np.array([5, 3, 8, 1, 3, 9])   # 假设这是实验A用到的样本id(混入了重复,故意的)
run2_ids = np.array([9, 1, 2, 8])         # 实验B用到的样本id

common = np.intersect1d(run1_ids, run2_ids)
assert list(common) == [1, 8, 9]          # 结果自动去重 + 排序

# 判断两次实验的样本集合是否完全一致:交集大小是否等于两边各自去重后的大小
same_set = (len(common) == len(np.unique(run1_ids)) == len(np.unique(run2_ids)))
assert same_set is False                 # 不完全一致(下一节 setdiff1d 会告诉你具体差在哪)

# return_indices=True: 还能拿到交集元素在两个原数组里的位置
common2, idx1, idx2 = np.intersect1d(run1_ids, run2_ids, return_indices=True)
assert np.array_equal(run1_ids[idx1], common2)
assert np.array_equal(run2_ids[idx2], common2)
```

**常见坑:** 结果是**去重且排序**过的,不保留原始顺序,也不保留重复次数——如果两个数组里某个 id 各自出现了好几次,交集里只会出现一次。如果需要知道"这个共同元素在原数组的哪些位置",不要自己再手写一遍 `np.where` 查找,直接用 `return_indices=True` 更不容易出错。

---

## 5. `np.union1d`

**签名:**
```python
np.union1d(ar1, ar2)
```

**一句话:** 返回两个数组的**并集**——两边所有出现过的元素合在一起,去重并升序排列。

**AI 研究场景:** 两个数据源各自标注了一批类别/词表,想知道"合并起来一共覆盖了哪些类别"用来构建统一的类别表/词表;或者合并两次实验各自新增的样本 id,得到"这两次实验总共涉及的样本"全集。

**可运行例子:**
```python
import numpy as np

vocab_a = np.array([3, 1, 4, 1, 5])       # 数据源A里出现过的类别/词表id
vocab_b = np.array([9, 2, 6, 5, 3])       # 数据源B里出现过的类别/词表id

merged = np.union1d(vocab_a, vocab_b)
assert list(merged) == [1, 2, 3, 4, 5, 6, 9]   # 两边所有出现过的id,去重+排序

# union1d 自动去重,即使输入里有重复元素也不影响
assert len(merged) == len(set(vocab_a.tolist()) | set(vocab_b.tolist()))
```

**常见坑:** 和 `concatenate` 是两回事——`np.concatenate([vocab_a, vocab_b])` 只是把两个数组头尾拼起来,长度是两者之和,**不去重也不排序**;`union1d` 才是真正的集合并集语义。需要"合并成一个大数据集"用 `concatenate`,需要"合并成一个无重复的类别/id 集合"用 `union1d`,搞反了容易导致类别表里出现重复项。

---

## 6. `np.setdiff1d`

**签名:**
```python
np.setdiff1d(ar1, ar2, assume_unique=False)
```

**一句话:** 返回**差集**——在 `ar1` 里出现、但**没**在 `ar2` 里出现的元素(去重排序)。**注意不对称**:`setdiff1d(ar1, ar2)` 和 `setdiff1d(ar2, ar1)` 通常结果不同。

**AI 研究场景:** 承接上面 `intersect1d` 的场景——发现两次实验的样本集合对不上之后,用 `setdiff1d` 精确定位"这次实验里新增/多出来的样本 id 有哪些"、"上次有但这次漏掉的样本 id 有哪些",这是排查数据集划分不一致问题时最直接的工具。也常用于找出"标注了但类别表里没定义"的异常标签。

**可运行例子:**
```python
import numpy as np

run1_ids = np.array([5, 3, 8, 1, 3, 9])
run2_ids = np.array([9, 1, 2, 8])

only_in_run1 = np.setdiff1d(run1_ids, run2_ids)   # run1有但run2没有
assert list(only_in_run1) == [3, 5]

only_in_run2 = np.setdiff1d(run2_ids, run1_ids)   # 反过来!结果不对称
assert list(only_in_run2) == [2]

# 用两次 setdiff1d 精确定位"两次实验样本集合差在哪"
assert not (len(only_in_run1) == 0 and len(only_in_run2) == 0)   # 不对称说明两边确实不完全一样
```

**常见坑:** 参数顺序决定了结果,`setdiff1d(a, b)` 读作"a 减 b"、"a 有 b 没有的部分",不是对称操作。想要完整核对两个集合是否一致,通常需要**两个方向都算一遍**(正如上面例子),只算一个方向容易漏掉一半信息(比如只发现了"这次多出来的",却没发现"上次有的这次丢了")。

---

## 7. `np.in1d`(现在叫 `np.isin`)

**签名:**
```python
np.isin(element, test_elements, assume_unique=False, invert=False)
```
- `element`:要检查的数组(结果和它形状一致)
- `test_elements`:拿来做比对的"集合"(内部会被拉平,不要求和 `element` 形状一样)
- `invert`:`True` 的话结果整体取反,等价于但比手动取反更快

**一句话:** 判断 `element` 里的**每一个元素**是不是出现在 `test_elements` 这个集合里,返回一个和 `element` **形状相同**的布尔数组——本质上是 Python `in` 关键字的"数组广播版"。

**名字的历史:** 这个功能最早叫 `np.in1d`("in 1-D",只能处理一维输入,用之前得自己 `.ravel()`)。后来 numpy 引入了功能更通用的 `np.isin`(支持任意形状输入,不用先拉平,官方从很早就建议统一用它)。`np.in1d` 在 numpy 2.0(2024年)被标记为 deprecated,到 **2.4.0** 彻底移除——我们仓库用的正是移除之后的 2.4.6,下面例子里会实测证实这一点。**结论:新代码一律用 `np.isin`,`in1d` 只是历史名字,当遇到老代码里的 `in1d` 时知道它就是 `isin` 的前身即可。**

**AI 研究场景:**
- **按白名单/黑名单过滤:** 从一个大数据集的标签数组里,只挑出属于某几个感兴趣类别的样本,`mask = np.isin(labels, allowed_classes)` 一行搞定,比对每个类别写一次 `==` 再 `|` 起来干净得多。
- **判断两批数据是否有重叠:** 检查这一批新采样的样本 id 有没有意外混入之前训练集用过的 id(防止训练/测试集泄漏),`np.isin(new_ids, train_ids).any()` 就能报警。

**可运行例子:**
```python
import numpy as np

# 历史小知识实测:老名字 np.in1d 在本仓库的 numpy 2.4.6 里已经被彻底移除
in1d_removed = False
try:
    np.in1d(np.array([1, 2]), np.array([2, 3]))
except AttributeError:
    in1d_removed = True
assert in1d_removed        # numpy 2.0 标记deprecated,2.4.0起彻底移除,只能用 isin

labels = np.array([3, 1, 4, 1, 5, 9, 2, 6])
allowed_classes = [1, 2, 3]

mask = np.isin(labels, allowed_classes)
assert list(mask) == [True, True, False, True, False, False, True, False]

filtered = labels[mask]
assert list(filtered) == [3, 1, 1, 2]

# invert=True:反过来找"不在白名单里"的(比如要过滤掉的噪声类别)
mask_out = np.isin(labels, allowed_classes, invert=True)
assert list(labels[mask_out]) == [4, 5, 9, 6]

# isin保持输入的原始形状;这是它比in1d更通用的地方(in1d当年总是拉平成1维)
grid = np.array([[1, 7], [3, 8]])
mask2d = np.isin(grid, allowed_classes)
assert mask2d.shape == (2, 2)
assert mask2d.tolist() == [[True, False], [True, False]]

# 别和 array_equal / python自带的in 搞混:isin是"每个元素分别判断",返回同形状布尔数组
assert bool(np.isin(3, labels))                      # 标量也能用,返回0维数组
assert not np.array_equal(labels, allowed_classes)   # array_equal判断的是整体是否完全相等,语义不同
```

**常见坑:** `np.isin(a, b)` 判断的是"a **的每个元素**在不在 b 里",结果形状和 `a` 一样;不要跟 `np.array_equal(a, b)`(判断两个数组**整体**是否完全相等,返回单个布尔值)搞混,也不要用 Python 内置的 `in`(`3 in some_array` 只能判断单个标量在不在里面,不能像 `isin` 一样一次性广播到整个数组、给出逐元素的布尔结果)。如果看到老代码或老教程里的 `np.in1d`,记得它在新版 numpy 里已经不存在了,直接换成 `np.isin`,参数含义基本不变。

---

## 8. `np.searchsorted`

**签名:**
```python
np.searchsorted(a, v, side='left', sorter=None)
```
- `a`:**必须是已经排好序(升序)的数组**——这是使用前提,不是可选项
- `v`:要查找插入位置的值,可以是标量,也可以是数组(一次查一批)
- `side`:`'left'` 或 `'right'`,决定"查找值恰好等于数组里已有元素"时插入在该元素的左边还是右边
- `sorter`:如果 `a` 本身没排序,但你手上有一份"能让它变得有序"的下标数组(比如 `argsort` 的结果),可以传进来,内部按 `sorter` 重排后再查找,不用你自己先真的排一遍

**一句话:** 在一个有序数组里用**二分查找**,返回"如果要把 `v` 插入进 `a`,应该插入哪个位置才能让 `a` 继续保持有序"。

**AI 研究场景:**
- **学习率分段调度(带里程碑的 scheduler):** 给定一串里程碑步数(比如 `[1000, 5000, 10000]`),想知道当前训练步数落在第几段,`np.searchsorted(milestones, current_step, side='right')` 一行代码就替代了一长串 `if step < 1000: ... elif step < 5000: ...` 的判断,而且天然支持一次给一批 step 做向量化查询。
- **直方图分桶/连续值离散化:** 把一批连续的分数/特征值映射到预先定义好的区间编号(比如画直方图、或者把连续特征离散化成类别特征喂给某些模型),`np.searchsorted(bin_edges, values)` 直接给出每个值该落到哪个桶。
- **为什么比线性扫描快:** 二分查找是 `O(log n)`,挨个比较的线性扫描是 `O(n)`——当里程碑/分桶边界很多、或者要高频查询一个大的有序表(比如做检索)时,差距会很明显。

**可运行例子:**
```python
import numpy as np

sorted_arr = np.array([1, 3, 5, 7, 9])

pos = np.searchsorted(sorted_arr, 6)
assert pos == 3     # 6应该插入到下标3(在7前面),插入后数组仍然有序

# side='left' / 'right':当查找值等于数组里已有的元素时,插入位置不同
pos_left = np.searchsorted(sorted_arr, 5, side='left')
pos_right = np.searchsorted(sorted_arr, 5, side='right')
assert pos_left == 2     # 插到已有的5的左边(即5本身所在的下标)
assert pos_right == 3    # 插到已有的5的右边(5后面)

# 可以一次查一批值(向量化,不用写循环)
vals = np.array([0, 2, 5, 10])
positions = np.searchsorted(sorted_arr, vals)
assert list(positions) == [0, 1, 2, 5]

# AI场景1:学习率分段调度——给定里程碑步数,直接算出当前在第几段
milestones = np.array([1000, 5000, 10000])
steps = np.array([500, 1000, 3000, 5000, 8000, 12000])
stage = np.searchsorted(milestones, steps, side='right')
assert list(stage) == [0, 1, 1, 2, 2, 3]

# AI场景2:直方图分桶——把连续值离散化到区间编号
bin_edges = np.array([0, 10, 20, 30, 40])
values = np.array([5, 10, 15, 39.9, 40, 0])
bucket = np.searchsorted(bin_edges, values, side='right') - 1
assert list(bucket) == [0, 1, 1, 3, 4, 0]

# 常见坑1:a必须是有序的!传入没排序的数组,函数不报错,但结果没有意义
unsorted_arr = np.array([50, 10, 90, 30, 70, 20, 80, 5])
wrong = np.searchsorted(unsorted_arr, 45)          # 静默给出一个错误答案
correct = np.searchsorted(np.sort(unsorted_arr), 45)
assert wrong == 2
assert correct == 4
assert wrong != correct       # 同一个查找值,答案完全不同——因为输入前提被违反了

# 常见坑2的补救:如果数组本身没排序,但你有一份能让它排好序的下标(比如argsort的结果)
# 可以用 sorter 参数,不用真的把数组排一遍
sorter = np.argsort(unsorted_arr)
pos_with_sorter = np.searchsorted(unsorted_arr, 45, sorter=sorter)
assert pos_with_sorter == correct
```

**常见坑:**
- **`a` 必须有序,否则结果"静默"出错。** 这是 `searchsorted` 最危险的坑:传入一个没排序的数组,函数**不会报错**,而是若无其事地返回一个二分查找路径算出来的、但毫无意义的下标(上面例子里对同一个查找值 45,未排序数组给出 2,真正排序后是 4)。因为没有报错,这类 bug 往往要等到下游结果诡异了才会被发现。
- `side='left'` 和 `side='right'` 只在查找值**恰好等于**数组里已有元素时才会产生不同结果——分桶时用左闭右开 `[a, b)` 还是左开右闭 `(a, b]` 区间,取决于这个参数选对没选对,选错了通常只在边界值上出错(比如恰好等于某个 bin 边界的值被分错桶),平时测试很容易漏掉这类边界情况,建议专门测一下边界值。

---

## 小结:这一批 8 个函数解决的问题

| 函数 | 解决的问题 |
|---|---|
| `sort` | 返回排好序的副本(默认升序) |
| `argsort` | 返回排序用的下标(按某一列给整表排序的基础) |
| `partition`/`argpartition` | 不完全排序,`O(n)` 快速找 top-k |
| `intersect1d` | 两个数组的交集(去重排序) |
| `union1d` | 两个数组的并集(去重排序) |
| `setdiff1d` | 差集,`a`有`b`没有的部分(不对称) |
| `in1d`/`isin` | 判断每个元素是否在另一个集合里 |
| `searchsorted` | 有序数组里二分查找插入位置(分桶/调度) |

下一批:[08-broadcasting-and-ufunc.md](08-broadcasting-and-ufunc.md)

---

*更新:2026-07-07*
