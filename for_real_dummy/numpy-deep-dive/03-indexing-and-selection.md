# 03 · 索引与选择(Indexing & Selection)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**怎么从一个数组里精确挑出我要的那部分数据**——无论是切一段连续区间、按条件过滤 padding、用下标表做 embedding 查表,还是在 causal attention 里构造"看不到未来"的 mask。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. 基础切片 `a[i:j:k]`

**签名:**
```python
a[start:stop:step]                                # 一维
a[start1:stop1:step1, start2:stop2:step2, ...]     # 多维,每个维度一套独立的 start:stop:step
```
- `start`:起始下标(包含),不写默认 0
- `stop`:结束下标(**不包含**),不写默认到末尾
- `step`:步长,不写默认 1;可以是负数(反向遍历)
- 多维数组每个维度的切片用逗号分隔,互相独立
- 负数下标表示"从末尾往前数",`-1` 是最后一个元素

**一句话:** 用 `start:stop:step` 描述"取一段连续/等间隔的区间",多维时每个维度各写一套,负数下标表示"从末尾往前数"。

**AI 研究场景:**
- **语言模型最经典的"错位一位"构造:** `input_ids = tokens[:, :-1]`(去掉最后一个 token 作为输入)、`labels = tokens[:, 1:]`(去掉第一个 token 作为预测目标)——这是自回归语言模型训练数据构造的标准写法,[01-numpy-for-c-programmers.md](../01-numpy-for-c-programmers.md) 里已经提前打过照面。
- **反转序列:** `a[::-1]`,某些数据增强或双向处理需要把序列倒过来。
- **隔点降采样:** `signal[::2]` 每隔一个点采一个,常见于音频/时间序列下采样。
- **切一个小 batch 做快速调试:** `small_batch = dataset[:32]`,跑通流程比跑全量数据更重要的时候用。

**可运行例子:**
```python
import numpy as np

a = np.arange(10)

# 基础 start:stop:step
b = a[2:7:2]
assert b.tolist() == [2, 4, 6]

# 反转:step 为负数
r = a[::-1]
assert r.tolist() == list(range(9, -1, -1))

# 负数下标:从末尾数
assert a[-1] == 9
assert a[-3:].tolist() == [7, 8, 9]

# 多维:每个维度独立切
m = np.arange(1, 13).reshape(3, 4)
sub = m[0:2, 1:3]
assert sub.tolist() == [[2, 3], [6, 7]]

# 关键性质:切片返回的是视图(view),不是拷贝
view = a[2:5]
view[0] = 999
assert a[2] == 999          # 改视图,原数组跟着变
```

**常见坑:** 切片返回的是原数组的**视图**——`view = a[2:5]` 和 `a` 共享同一块底层内存,修改 `view` 会连带修改 `a`,这一点和下一节布尔索引、第 3 节花式索引完全不同(它们返回的是独立拷贝)。这是索引这个专题贯穿始终的核心对比:**"切片 = 视图,布尔索引 / 花式索引 = 拷贝"**,建议现在就记牢,后面每一节都会回过头呼应这个区别。如果想让切片后的数组和原数组彻底脱钩,要显式调用 `.copy()`。

---

## 2. 布尔索引 `a[mask]`

**签名:**
```python
a[condition]
```
- `condition`:一个和 `a` 形状相同(或可广播)的布尔数组,通常直接由比较运算符产生,比如 `a > 0`

**一句话:** 用一个"每个位置 True/False"的同形状布尔数组,把所有 `True` 对应的元素挑出来,**拍平成一维**返回(不保留原来的多维结构)。

**AI 研究场景:**
- **过滤 padding token:** `valid = input_ids[attention_mask.astype(bool)]`——用 attention mask 把真正有意义的 token 挑出来,padding 部分直接扔掉。
- **数据清洗:** `clean = data[~np.isnan(data)]`,去掉 NaN/异常值,`~` 表示取反。
- **挑正/负样本子集:** `positive_samples = features[labels == 1]`,分类任务里按标签筛出某一类的所有样本。
- **只对有效位置算 loss:** `loss[mask]` 只保留非 padding 位置的 loss 值再求平均,避免 padding 部分"稀释"了真实 loss。

**可运行例子:**
```python
import numpy as np

a = np.array([1, -2, 3, -4, 5])
mask = a > 0
assert mask.tolist() == [True, False, True, False, True]
positive = a[mask]
assert positive.tolist() == [1, 3, 5]

# 2D场景:过滤padding(1=有效token,0=padding)
tokens = np.array([[5, 8, 2, 0, 0], [3, 1, 0, 0, 0]])
attention_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=bool)
valid_tokens = tokens[attention_mask]           # 结果拍平成1维,不再是2D
assert valid_tokens.tolist() == [5, 8, 2, 3, 1]

# 关键性质:布尔索引(读取)返回的是拷贝,不是视图
b = np.array([1, 2, 3, 4])
copy = b[b > 2]
copy[0] = 999
assert b.tolist() == [1, 2, 3, 4]               # 原数组没被改,证明是独立拷贝

# 但"赋值"用法可以真正原地修改——走的是 __setitem__,不是"先读出来再改"
c = np.array([1, -2, 3, -4, 5])
c[c < 0] = 0
assert c.tolist() == [1, 0, 3, 0, 5]
```

**常见坑:** 和上一节的切片正相反,**布尔索引 `a[mask]`(读取)返回的是独立拷贝**——`copy = a[mask]` 之后再改 `copy` 不会影响 `a`。但要分清"读"和"写"两种用法:`a[mask] = value` 这种**赋值**写法可以真正原地修改原数组,因为它调用的是 `__setitem__`,直接对原数组底层内存按位置写入,并不经过"先拷贝出来再改"这条路。规律是:**想读一份子集另存,直接 `a[mask]`;想批量原地修改满足条件的元素,写成 `a[mask] = value`,不要写成 `tmp = a[mask]; tmp[:] = value` 这种以为能生效、实际上只改了拷贝的写法。**

---

## 3. 花式索引 `a[[i,j,k]]`

**签名:**
```python
a[index_array]
```
- `index_array`:一个**整数**数组或 list(不是布尔数组),元素是下标,可以乱序、可以重复、长度不必等于 `a` 的长度

**一句话:** 给一组"整数下标列表",按列表里写的顺序把对应位置的元素挑出来组成新数组——顺序由你给的下标列表决定,不是原数组的顺序。

**AI 研究场景:**
- **embedding lookup 的本质:** `embedding_matrix[token_ids]` 就是花式索引——`token_ids` 是一批整数 ID,每个 ID 对应查表取出 `embedding_matrix` 的一行,这正是 `nn.Embedding` 底层做的事情,理解了这一节基本就理解了 embedding 层最核心的计算。
- **打乱/重排样本:** `shuffled_data = dataset[permutation]`,配合 01 篇的 `np.random.choice`、09 篇的 `np.random.permutation` 生成的下标数组使用。
- **随机采一个 mini-batch:** `batch = dataset[np.random.randint(0, len(dataset), size=batch_size)]`——这里的随机下标数组就是花式索引的 `index_array`。

**可运行例子:**
```python
import numpy as np

a = np.array([10, 20, 30, 40, 50])

idx = [0, 2, 4]
assert a[idx].tolist() == [10, 30, 50]

# 顺序由下标列表决定,可以乱序、可以重复
idx2 = [4, 4, 0]
assert a[idx2].tolist() == [50, 50, 10]

# embedding lookup 的本质
embedding_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])   # 3个词,每个词2维向量
token_ids = np.array([2, 0, 2, 1])
looked_up = embedding_matrix[token_ids]
assert looked_up.shape == (4, 2)
assert looked_up.tolist() == [[0.5, 0.6], [0.1, 0.2], [0.5, 0.6], [0.3, 0.4]]

# 关键性质:花式索引也返回拷贝,不是视图(和布尔索引一致,和切片相反)
b = np.array([1, 2, 3, 4])
fancy = b[[0, 1]]
fancy[0] = 999
assert b.tolist() == [1, 2, 3, 4]

# 进阶陷阱:多维花式索引是"逐个配对",不是笛卡尔积子矩阵
mat = np.arange(1, 10).reshape(3, 3)          # [[1,2,3],[4,5,6],[7,8,9]]
pair = mat[[0, 1], [2, 0]]                     # 配对成坐标 (0,2) 和 (1,0)
assert pair.tolist() == [3, 4]                 # 不是"第0,1行 × 第2,0列"的子矩阵
```

**常见坑:** 花式索引和布尔索引一样返回**拷贝**——这是本篇要反复强调的对比:**切片是视图,布尔索引和花式索引都是拷贝。** 另外两个深坑:① 如果给花式索引传的是一个 Python bool 列表(比如 `a[[True, False, True]]`),numpy 会自动识别成布尔索引而不是"下标 0/1/0",因为 numpy 是按传入数组的 dtype 自动判断走哪条路径,int 列表和 bool 列表走的规则完全不同,混着写容易得到意料之外的结果;② 多维花式索引 `mat[[0,1],[2,0]]` 是把两个下标列表"逐位配对"成坐标 `(0,2)`、`(1,0)`,**不是**取"第 0、1 行和第 2、0 列"交叉出的子矩阵——想要那种子矩阵效果,得配合 `np.ix_` 或者分两步切片,这是多维花式索引最容易踩的坑。

---

## 4. `np.where`(条件选择器用法)

**签名:**
```python
np.where(condition)
```
- `condition`:布尔数组
- 本节只讲**单参数**用法(返回下标);`np.where(condition, x, y)` 这种"三元表达式/逐元素二选一"的用法留到 [04-elementwise-math.md](04-elementwise-math.md) 细讲,这里先只建立"用 where 找位置"这一层认知,不要和三元写法混在一起。

**一句话:** 传入一个条件,返回一个**元组**,元组里每个维度对应一个数组,告诉你"满足条件的元素都在哪些坐标上"——本质上和第 8 节的 `np.nonzero(condition)` 完全等价。

**AI 研究场景:**
- **挑正/负样本的下标(而不是值本身):** 先用 `np.where` 定位"哪些位置是正样本",再拿这些下标去另一个数组里取值——这比直接布尔索引更灵活,因为同一组下标可以复用在多个不同数组上(比如同时切 `features` 和 `sample_weights`)。
- 和 `nonzero` 一样,常用来定位"稀疏"信息,比如置信度矩阵里哪些位置超过阈值。

**可运行例子:**
```python
import numpy as np

labels = np.array([0, 1, 1, 0, 1, 0])

idx = np.where(labels == 1)
assert isinstance(idx, tuple)
assert len(idx) == 1                       # 1维输入,元组里只有1个数组(对应第0维)
assert idx[0].tolist() == [1, 2, 4]

# 用这组下标去另一个数组里取值——这是它比直接布尔索引更灵活的地方
features = np.array([10, 20, 30, 40, 50, 60])
positive_samples = features[idx]           # 元组可以直接当索引用,不用先解包
assert positive_samples.tolist() == [20, 30, 50]

# 单参数形式的 where 和 nonzero 完全等价
assert np.array_equal(np.where(labels == 1)[0], np.nonzero(labels == 1)[0])
```

**常见坑:** 最容易迷惑的地方是 `np.where` 有两套完全不同的调用方式:**只传条件**(本节讲的,返回下标元组)和**传条件+两个候选值**(`where(cond, x, y)`,04 篇讲,返回逐元素二选一的新数组)——两种用法名字一样,行为完全不同,一定要靠"参数个数"分辨,不要靠记忆猜。另外,单参数 `np.where(condition)` 返回的下标元组和第 8 节 `np.nonzero` 有一模一样的坑:元组长度等于维度数,不是满足条件的元素个数,详细展开见第 8 节。

---

## 5. `np.take`

**签名:**
```python
np.take(a, indices, axis=None, mode='raise')
```
- `indices`:整数下标(数组/list),可以是任意形状
- `axis`:沿哪个轴取;不写就先把 `a` 展平成1维再取
- `mode`:下标越界时怎么处理——`'raise'`(默认,直接报错)、`'wrap'`(取模绕回)、`'clip'`(夹到边界)

**一句话:** 花式索引 `a[indices]` 的函数版本,功能大部分重叠,但多了显式的 `axis` 参数和处理越界下标的 `mode` 参数。

**AI 研究场景:**
- 在高维 tensor(比如 `(batch, seq_len, hidden)`)上沿指定轴按下标查表,`np.take(a, idx, axis=1)` 比 `a[:, idx, :]` 这种要手动补齐冒号的花式索引写法更清晰,尤其轴数多的时候。
- `mode='clip'` 常用来处理"位置下标可能超出预设最大长度"这类场景——与其让程序崩溃,不如把超出范围的下标夹到最后一个合法位置继续跑。

**可运行例子:**
```python
import numpy as np

a = np.array([10, 20, 30, 40, 50])

result = np.take(a, [0, 2, 4])
assert result.tolist() == [10, 30, 50]
assert np.array_equal(result, a[[0, 2, 4]])          # 一维场景下和花式索引结果一致

# 显式 axis:按列取
m = np.arange(1, 13).reshape(3, 4)
col = np.take(m, [0, 2], axis=1)
assert col.tolist() == [[1, 3], [5, 7], [9, 11]]

# mode='clip':越界下标不报错,直接夹到边界
# 注意:clip模式下负数不再表示"从末尾数",而是被当成"太小"直接夹到0
clipped = np.take(a, [-1, 10], mode='clip')
assert clipped.tolist() == [10, 50]                  # -1→夹到0号位,10→夹到4号位(最后一个)

# 返回的同样是拷贝,不是视图
r2 = np.take(a, [0, 1])
r2[0] = 999
assert a.tolist() == [10, 20, 30, 40, 50]
```

**常见坑:** 不写 `axis` 时会先把整个数组**展平**再取,多维数组下极容易忘记这一点,取到的是"展平后的第 N 个元素"而不是"某一行/列",一定要显式传 `axis`。默认 `mode='raise'`,下标越界会直接抛 `IndexError`(这点和普通花式索引一致);`mode='clip'` 时负数下标的语义会变——不再是"从末尾数",而是被当成"越界的小值"夹到 0,和正常 Python/numpy 负数下标的直觉不一样,用之前最好实际打印验证一下。返回值和花式索引一样是拷贝,不是视图。

---

## 6. `np.take_along_axis`

**签名:**
```python
np.take_along_axis(arr, indices, axis)
```
- `indices`:下标数组,**维度数必须和 `arr` 相同**,在除 `axis` 以外的其他维度上要能和 `arr` 匹配/广播
- `axis`:必须显式指定,不能省略(这点和 `take` 不同)

**一句话:** "每一行/每一列取不同下标"版本的取值操作——`indices` 本身是一个和 `arr` 同维度数的数组,在 `axis` 这一维上告诉每个位置该取哪个下标,常和 `argmax`/`argsort` 的输出配套使用,是 numpy 里标准的"gather"写法(等价于 PyTorch 的 `torch.gather`)。

**AI 研究场景:**
- **经典组合 `argmax` + `take_along_axis`:** `argmax` 只能告诉你"每行最大值在哪一列",不会直接给你那个值;`idx = scores.argmax(axis=1, keepdims=True)` 拿到下标后,`np.take_along_axis(scores, idx, axis=1)` 才能把对应的值取出来——这是"先找位置、再按位置取值"这套 gather 模式的标准范例。
- **排序后同步重排"伴随数组":** 按某一列的分数 `argsort` 得到重排下标后,用 `take_along_axis` 把 label/权重等"伴随数组"按相同顺序同步打乱,保证数据和标签不会错位对应。

**可运行例子:**
```python
import numpy as np

scores = np.array([[0.1, 0.7, 0.2],
                    [0.6, 0.3, 0.1]])

# 标准组合:argmax 找位置 + take_along_axis 按位置取值
top1_idx = np.argmax(scores, axis=1, keepdims=True)     # keepdims=True 保持维度数一致!
assert top1_idx.tolist() == [[1], [0]]

top1_val = np.take_along_axis(scores, top1_idx, axis=1)
assert top1_val.tolist() == [[0.7], [0.6]]

# 排序下标同步重排"伴随数组"
values = np.array([[3, 1, 2]])
labels = np.array([['c', 'a', 'b']])
order = np.argsort(values, axis=1)
assert order.tolist() == [[1, 2, 0]]
sorted_labels = np.take_along_axis(labels, order, axis=1)
assert sorted_labels.tolist() == [['a', 'b', 'c']]

# 返回的是拷贝,不是视图
mutated = np.take_along_axis(scores, top1_idx, axis=1)
mutated[0, 0] = 999.0
assert scores[0, 1] == 0.7                              # 原数组没被改

# 常见坑的实锤:忘记 keepdims=True,indices 少一维,直接报错
bad_idx = np.argmax(scores, axis=1)                      # shape (2,),不是 (2,1)
try:
    np.take_along_axis(scores, bad_idx, axis=1)
    assert False, "应该报错但没报错"
except ValueError as e:
    assert "same number of dimensions" in str(e)
```

**常见坑:** 最常见的错误是忘记给 `argmax`/`argsort` 传 `keepdims=True`——`scores.argmax(axis=1)` 默认返回的下标数组会**少一维**(`(2,)` 而不是 `(2,1)`),直接传给 `take_along_axis` 会报 `ValueError: indices and arr must have the same number of dimensions`,上面的例子已经实际验证了这个报错。记住口诀:**`take_along_axis` 要求 `indices` 和 `arr` 维度数完全相同,`axis` 必须手写,不能省略。** 另外它和 `take`/花式索引一样返回拷贝,不是视图。

---

## 7. `np.argwhere`

**签名:**
```python
np.argwhere(a)
```
- `a`:数组(通常是布尔数组,或者会被当作"非零即True"处理的数值数组)
- 返回一个 `(N, a.ndim)` 形状的二维数组,`N` 是满足条件的元素个数,每一**行**是一个完整坐标

**一句话:** 把"哪些位置是 True/非零"的结果,组织成"每行一个完整坐标"的表格——比第 8 节 `nonzero` 的"每个维度一个数组"的元组形式更直观,适合直接打印查看或者逐点遍历。

**AI 研究场景:**
- **定位异常值:** `np.argwhere(np.isnan(data))` 一次性拿到所有 NaN 的完整坐标,方便打印/记录到日志里定位是哪一条样本的哪一个特征出了问题。
- **图像/注意力矩阵里找"被激活"的位置:** 比如某个二值化后的 mask,`np.argwhere(mask)` 直接给出所有前景像素的 `(row, col)` 坐标列表,方便画框或者统计。

**可运行例子:**
```python
import numpy as np

data = np.array([[1, 0, 3], [0, 5, 0]])
coords = np.argwhere(data > 2)
assert coords.tolist() == [[0, 2], [1, 1]]        # (0,2)位置是3,(1,1)位置是5

# 定位NaN坐标
arr = np.array([1.0, np.nan, 3.0, np.nan])
nan_coords = np.argwhere(np.isnan(arr))
assert nan_coords.tolist() == [[1], [3]]           # 1维数组,每个坐标退化成单个数字包一层

# 常见坑的实锤:argwhere的结果不能直接拿去索引原数组
try:
    data[coords]
    assert False, "应该报错但没报错"
except IndexError as e:
    assert "out of bounds" in str(e)
```

**常见坑:** numpy 官方文档原话强调:**`argwhere` 的输出不适合直接用来索引数组,想索引请用 `nonzero`。** 上面的例子已经验证了这一点——`data[coords]` 会直接抛 `IndexError`,因为 `coords` 是"坐标表"而不是"每个维度一个下标数组"的元组形式,直接拿去当索引会被 numpy 误解成花式索引的下标,凑不上形状就报错(即便凑巧凑得上形状,结果语义也是错的,不会是你想要的筛选结果)。记住分工:**`argwhere` 用来"看"/"数"/"打印"坐标,真正要索引数组时用下一节的 `nonzero`。**

---

## 8. `np.nonzero`

**签名:**
```python
np.nonzero(a)
```
- 返回一个**元组**,长度固定等于 `a.ndim`,元组里第 `i` 个数组是所有非零(或True)元素在第 `i` 维上的坐标

**一句话:** 和 `argwhere` 是同一份坐标信息的两种组织方式——`argwhere` 按"元素"分组(每行一个坐标),`nonzero` 按"维度"分组(每个维度一个数组),后者的形式可以直接喂给索引操作。

**AI 研究场景:**
- **直接索引取值:** `mask[np.nonzero(mask)]` 一步到位拿到所有非零值,这是 `nonzero` 相比 `argwhere` 的核心优势——上一节讲过 `argwhere` 的结果不能这样直接用。
- **稀疏结构定位:** 统计/定位一个大矩阵里"有效"(非零)位置,`nonzero` 这个名字的由来就是为了配合稀疏矩阵场景设计的。
- 常和 05 批要讲的 `np.count_nonzero` 前后搭配着用:一个数"有多少个",一个给"具体在哪"。

**可运行例子:**
```python
import numpy as np

mask = np.array([[0, 5, 0], [3, 0, 7]])
rows, cols = np.nonzero(mask)                      # 按维度拆开,常见写法
assert rows.tolist() == [0, 1, 1]
assert cols.tolist() == [1, 0, 2]

# 直接拿去索引(argwhere的结果做不到这一点)
values = mask[np.nonzero(mask)]
assert values.tolist() == [5, 3, 7]

# 和 argwhere 是同一份信息的两种组织形式,可以互相转换
coords = np.argwhere(mask)
assert np.array_equal(np.array(np.nonzero(mask)).T, coords)

# 常见坑的实锤:元组长度是维度数,不是"非零元素个数"!
a = np.array([0, 3, 0, 7])
result = np.nonzero(a)
assert isinstance(result, tuple)
assert len(result) == 1                            # 长度等于 a.ndim(=1),不是"有几个非零"
assert len(result[0]) == 2                          # 真正的"非零元素个数"要多解包一层再len

# 想对下标做运算,要先取出对应维度的数组,不能对元组本身操作
shifted = result[0] + 1
assert shifted.tolist() == [2, 4]

# 元组可以直接当索引用,不需要手动解包
assert a[result].tolist() == [3, 7]
```

**常见坑:** 最容易踩的坑就是开头强调的那个:**`np.nonzero` 返回的是一个元组,对它 `len()` 得到的是数组的维度数(`a.ndim`),不是"非零元素的个数"**——上面的例子里 `len(result) == 1`,但 `a` 实际有 2 个非零元素,这个"1"只是因为 `a` 是1维数组,换成2维数组 `len()` 就会变成2,和"有多少个非零"毫无关系。真想数"有多少个非零元素",要么 `len(result[0])`,要么直接用 05 批的 `np.count_nonzero(a)`。另外元组本身不能直接做数学运算(`result + 1` 会报错,元组不支持和整数相加),必须先用 `result[0]`、`result[1]` ... 取出具体某一维的数组再运算。

---

## 9. `np.isin`

**签名:**
```python
np.isin(element, test_elements)
```
- `element`:要检查的数组,返回结果和它**同形状**
- `test_elements`:拿来做比对的"集合"(数组/list),不要求和 `element` 同形状

**一句话:** 向量化版本的"在不在这个集合里"——相当于对 `element` 里每个元素分别做一次 Python 的 `x in test_elements`,但一次性对整个数组算完,不用写循环。

**AI 研究场景:**
- **识别特殊 token:** `np.isin(token_ids, special_token_ids)` 判断一批 token id 是否属于 `[BOS, EOS, PAD]` 这类特殊符号集合,生成 mask 后决定是否在 loss/生成阶段忽略它们。
- **数据泄漏排查:** 检查验证集的样本 id 是否有一部分混进了训练集——`np.isin(val_ids, train_ids)`,这是划分数据集之后必做的健全性检查(sanity check)。

**可运行例子:**
```python
import numpy as np

token_ids = np.array([101, 7592, 2088, 102, 0, 0])
special_tokens = [101, 102, 0]                    # 类似 [CLS]/[SEP]/[PAD]

is_special = np.isin(token_ids, special_tokens)
assert is_special.tolist() == [True, False, False, True, True, True]

# 结合布尔索引,过滤掉特殊token,只留正文token
content_tokens = token_ids[~np.isin(token_ids, special_tokens)]
assert content_tokens.tolist() == [7592, 2088]

# 数据泄漏排查:训练集和验证集有没有重叠
train_ids = np.array([1, 2, 3, 4, 5])
val_ids = np.array([4, 5, 6, 7])
overlap = val_ids[np.isin(val_ids, train_ids)]
assert overlap.tolist() == [4, 5]                 # 4和5同时出现在训练和验证集,数据泄漏!
```

**常见坑:** 返回数组的形状由**第一个参数** `element` 决定,和 `test_elements` 的形状无关——两个参数顺序搞反,得到的形状就会不对(比如本来想按 token 逐个判断,结果因为参数顺序反了变成按"集合"逐个判断)。另外 `np.isin` 内部对大数组是靠排序或哈希实现的,比 Python 原生 `in`/`set` 更适合"整个数组一次性批量判断",但如果是"同一个固定集合被反复高频查询"这种场景,不一定比提前转成 Python `set` 更快——它的定位是"向量化的批量成员判断",不是通用的高性能查找结构。

---

## 10. `np.diag` / `np.diagonal`

**签名:**
```python
np.diag(v, k=0)
np.diagonal(a, offset=0, axis1=0, axis2=1)
```
- `np.diag`:**双重身份**——传入的是1维数组时,构造一个以它为对角线的2维矩阵(其余位置补0);传入的是2维数组时,反过来提取它的对角线,返回1维数组。`k` 控制取哪一条对角线(0=主对角线,正数往右上偏移,负数往左下偏移)
- `np.diagonal`:只做"提取对角线"这一件事,不会反过来构造矩阵,接口更明确,`offset` 用法和 `diag` 的 `k` 相同,还支持更高维数组按指定的两个轴(`axis1`/`axis2`)取对角线

**一句话:** `diag` 是"一维数组 ↔ 对角矩阵"互相转换的双向工具(靠输入是1维还是2维自动切换方向);`diagonal` 专注"从已有矩阵抽对角线"这一个方向,不做反向构造。

**AI 研究场景:**
- **构造对角权重矩阵:** 把"每个类别的权重"这样一个一维向量变成对角矩阵去做加权线性变换,`np.diag(class_weights) @ X`;或者构造缩放矩阵。这是 01 篇 `np.eye` 那个"单位矩阵是特殊对角矩阵"话题的延伸——`np.eye(n)` 其实就等价于 `np.diag(np.ones(n))`。
- **提取协方差矩阵的方差:** 协方差矩阵的对角线就是每个维度自己的方差,`np.diagonal(cov_matrix)` 是数据分析/特征工程里常用的一步。
- **自注意力得分矩阵的对角线:** 代表"每个 token 关注自己"的强度,有时候会单独拿出来分析。

**可运行例子:**
```python
import numpy as np

# 1维 -> 2维:构造对角矩阵
v = np.array([1, 2, 3])
D = np.diag(v)
assert D.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

# 2维 -> 1维:反过来提取对角线
extracted = np.diag(D)
assert extracted.tolist() == [1, 2, 3]

# np.diagonal:只做提取,支持非方阵、支持偏移offset
m = np.arange(1, 13).reshape(3, 4)
main_diag = np.diagonal(m)
assert main_diag.tolist() == [1, 6, 11]

offset_diag = np.diagonal(m, offset=1)
assert offset_diag.tolist() == [2, 7, 12]

# AI场景:协方差矩阵对角线 = 各维度方差
data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])   # 3个样本,2个特征
cov = np.cov(data, rowvar=False)
variances = np.diagonal(cov)
assert np.allclose(variances, [data[:, 0].var(ddof=1), data[:, 1].var(ddof=1)])

# 构造方向(1维->2维)得到的是独立、可写的新数组
D2 = np.diag(v)
D2[0, 0] = 999
assert v.tolist() == [1, 2, 3]                    # v不受影响

# 提取方向(2维->1维)得到的是只读视图,不是普通拷贝——尝试写入会报错
extracted2 = np.diag(D)
try:
    extracted2[0] = 999
    assert False, "应该报错但没报错"
except ValueError as e:
    assert "read-only" in str(e)

d = np.diagonal(m)
try:
    d[0] = 999
    assert False, "应该报错但没报错"
except ValueError as e:
    assert "read-only" in str(e)

# 真想原地改对角线,用 np.fill_diagonal
m2 = np.zeros((3, 3))
np.fill_diagonal(m2, 7)
assert m2.tolist() == [[7, 0, 0], [0, 7, 0], [0, 0, 7]]
```

**常见坑:** `np.diag` 一个函数做两件相反的事,完全靠输入是1维还是2维自动切换,读代码时如果没看清变量定义,很容易看反是在"构造"还是"提取"。更深的坑在"提取"方向(`np.diag(2维数组)` 或者 `np.diagonal(...)`):它返回的既**不是独立拷贝、也不是普通可写视图**,而是一个**只读视图**——它和原数组共享同一块内存(通过原数组修改数据,提取出来的对角线也会跟着变),但直接对它本身赋值会报 `ValueError: assignment destination is read-only`,上面的例子已经验证。这是 numpy 从 1.9/1.10 版本之后专门加的保护(更早的版本会返回可写视图,悄悄改了对角线也不会报错,是个容易埋雷的历史遗留行为)。真想原地修改矩阵的对角线,应该用 `np.fill_diagonal(a, value)`。

---

## 11. `np.triu` / `np.tril`

**签名:**
```python
np.triu(m, k=0)
np.tril(m, k=0)
```
- `m`:输入矩阵(或更高维数组的最后两维)
- `k`:控制从哪条对角线开始保留,`k=0` 是主对角线本身,`k>0` 往右上偏移(排除更多主对角线附近的元素),`k<0` 往左下偏移

**一句话:** `triu` 保留矩阵的上三角部分(其余清零),`tril` 保留下三角部分(其余清零)——最经典的应用是构造 Transformer 的 causal attention mask。

**AI 研究场景:**
- **causal attention mask(自回归 Transformer 的核心机制):** `np.triu(np.ones((seq_len, seq_len)), k=1)` 生成一个"哪些位置代表未来、不能看"的 mask——严格上三角部分(不含主对角线)标记为1,代表"未来"位置;把这些位置对应的 attention score 设成 `-inf`,过 softmax 之后自然变成0,这样每个 token 就只能看到自己和之前的 token,看不到未来——这是 GPT 类自回归模型训练时最核心的一个 mask 构造技巧,没有之一。
- `tril` 是同一个思路的镜像版本,有时候用来表示"已经发生的历史"或因果关系矩阵。

**可运行例子:**
```python
import numpy as np

seq_len = 4
future_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
assert future_mask.tolist() == [
    [0., 1., 1., 1.],
    [0., 0., 1., 1.],
    [0., 0., 0., 1.],
    [0., 0., 0., 0.],
]

# 应用到 attention score 上:未来位置设成 -inf,softmax后趋近于0
scores = np.zeros((seq_len, seq_len))
scores[future_mask == 1] = -np.inf
assert scores[0].tolist() == [0., -np.inf, -np.inf, -np.inf]   # 第0个token只能看到自己
assert scores[3].tolist() == [0., 0., 0., 0.]                   # 最后一个token能看到全部历史

# tril 是镜像:保留下三角(含主对角线)
lower = np.tril(np.ones((3, 3)))
assert lower.tolist() == [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]

# 常见坑的实锤:k写成0会把主对角线也mask掉,自己都看不到自己
wrong_mask = np.triu(np.ones((seq_len, seq_len)), k=0)
assert wrong_mask[0, 0] == 1.0        # 对角线被标记成"禁止关注",这是一个隐蔽的逻辑bug
```

**常见坑:** `k` 的偏移方向和"要不要包含主对角线"是最容易踩的坑——causal mask 必须用 `k=1`(严格上三角,不含主对角线),因为每个 token 必须能关注"自己"(主对角线代表自己关注自己);如果偷懒写成 `k=0`,主对角线也会被标记成"禁止",相当于每个 token 连自己当前的信息都看不到,是一个初期很难发现的隐蔽 bug(模型依然能训练,只是效果会莫名其妙变差,调半天才发现是mask错了)。上面的例子已经验证了这个差异。记忆技巧:`triu(..., k=1)` 里的 `1` 就是在说"从主对角线**再往上数1条**开始保留",正好跳过主对角线本身。

---

## 12. `np.select`

**签名:**
```python
np.select(condlist, choicelist, default=0)
```
- `condlist`:一组布尔数组组成的 list,按**顺序**逐个检查
- `choicelist`:和 `condlist` 一一对应的取值 list,长度必须相同
- `default`:所有条件都不满足时的默认值

**一句话:** 多条件版本的 `where`——`where(cond, x, y)` 只能二选一,`select` 可以像 `if/elif/elif/.../else` 一样按顺序检查任意多个条件,命中第一个为 True 的条件就用对应的值。

**AI 研究场景:**
- **分段函数/分段调度:** 实现分阶段的学习率调度(warmup 阶段用一个公式、衰减阶段用另一个公式、最低阶段封顶)、或者某些激活函数的分段线性近似,`select` 比嵌套多层 `where` 可读性好得多。
- **连续值离散化打分:** 把模型输出的连续置信度按区间分成"高/中/低"几档离散标签,一次性搞定,不用写循环判断。

**可运行例子:**
```python
import numpy as np

scores = np.array([0.95, 0.4, 0.6, 0.1, 0.85])

condlist = [scores >= 0.8, scores >= 0.5]
choicelist = ['high', 'mid']
labels = np.select(condlist, choicelist, default='low')
assert labels.tolist() == ['high', 'low', 'mid', 'low', 'high']

# 常见坑的实锤:条件顺序反了,严格条件永远抢不到
wrong_condlist = [scores >= 0.5, scores >= 0.8]      # 宽松条件排在前面
wrong_choicelist = ['mid', 'high']
wrong_labels = np.select(wrong_condlist, wrong_choicelist, default='low')
assert wrong_labels.tolist() == ['mid', 'low', 'mid', 'low', 'mid']
assert 'high' not in wrong_labels.tolist()            # 0.95分本该是high,结果被mid抢先命中
```

**常见坑:** `select` 是按 `condlist` **顺序从前往后**找第一个为 `True` 的条件,不是"最严格匹配"也不是"最后一个为真的生效"——如果多个条件可能同时满足,**必须把更严格的条件写在前面**。上面的例子里如果把 `>=0.5` 写在 `>=0.8` 前面,`0.95` 分会被"抢先"归到宽松的那一档,`'high'` 这个标签永远不会出现,这是使用 `select` 最容易踩的顺序坑——而且它不会报错,只会默默给出错误结果,比直接报错的 bug 更难排查。

---

## 小结:这一批 12 个函数解决的问题

| 函数/写法 | 解决的问题 |
|---|---|
| 基础切片 `a[i:j:k]` | 连续/等间隔取子集,返回**视图** |
| 布尔索引 `a[mask]` | 按条件筛选(过滤padding/异常值/正负样本),返回**拷贝** |
| 花式索引 `a[[i,j,k]]` | 按下标列表取(embedding lookup本质),返回**拷贝** |
| `where`(选择器用法) | 定位满足条件的下标(元组形式,和nonzero等价) |
| `take` | 显式axis+越界模式(clip/wrap/raise)控制的取值 |
| `take_along_axis` | 逐行/逐列不同下标的取值(配合argmax/argsort的gather操作) |
| `argwhere` | 坐标表形式定位(适合看/数,**不适合**直接索引) |
| `nonzero` | 按维度分组的下标元组(适合直接索引,但len()是维度数不是个数) |
| `isin` | 向量化的集合成员判断(过滤特殊token/数据泄漏排查) |
| `diag`/`diagonal` | 一维↔对角矩阵互转 / 提取对角线(**只读视图**) |
| `triu`/`tril` | 上/下三角(causal attention mask的标准构造) |
| `select` | 多条件分段选择(if/elif/elif的向量化版本) |

下一批:[04-elementwise-math.md](04-elementwise-math.md) —— 数学与逐元素运算。

---

*更新:2026-07-07*
