# 04 · 数学与逐元素运算(Elementwise Math)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**给定一个(或两个)数组,怎么对它的每一个元素做数学/逻辑运算**——从加减乘除到 exp/log/sin,再到裁剪、比较、逻辑判断。这是整个 numpy 数值计算的地基,05 批的归约统计、06 批的线性代数,本质上都是在这些逐元素运算之上再做"聚合"。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,包括 `log(0)`、除以 0、负数开方这些边界情况的真实行为(是报错还是警告+`nan`/`inf`),全部是实测结果,不是凭记忆写的。

---

## 1. 四则运算 `+ - * /` 的 ufunc 本质

**签名:**
```python
np.add(x1, x2)
np.subtract(x1, x2)
np.multiply(x1, x2)
np.divide(x1, x2)
```
- `x1`, `x2`:两个数组(或数组和标量),形状需要能广播成一致(完整广播规则留到 08 批专题讲)
- 平时写代码时你几乎不会直接打这几个函数名——你写的 `a + b`、`a - b`、`a * b`、`a / b` 在底层**就是**在调用它们

**一句话:** `+ - * /` 这几个运算符不是 Python 内置的黑箱魔法,而是 numpy 给 `ndarray` 重载(overload)之后,分别转发给 `np.add`/`np.subtract`/`np.multiply`/`np.divide` 这四个 **ufunc**(universal function,通用函数)去执行的语法糖。ufunc 的定义就是"逐元素操作、自动支持广播"的 C 语言实现函数。

**AI 研究场景:** 这是本文乃至后面所有章节的地基——不管是 `loss = pred - target`、`grad * learning_rate`,还是任何看起来"像数学公式"的 numpy 代码,背后跑的都是这一层 ufunc 机制。理解了"运算符只是 ufunc 的马甲",才能明白下面第 13、14 节的比较/逻辑运算符是同一套逻辑,也能明白为什么 numpy 数组的默认运算永远是"逐元素"而不是像数学课本里整体运算(矩阵乘法 `@` 是例外,见 06 批线性代数)。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# 运算符和 ufunc 调用是完全等价的两种写法
assert np.array_equal(a + b, np.add(a, b))
assert np.array_equal(a - b, np.subtract(a, b))
assert np.array_equal(a * b, np.multiply(a, b))
assert np.allclose(a / b, np.divide(a, b))

assert type(np.add).__name__ == "ufunc"     # np.add 本身就是一个 ufunc 对象,不是普通函数

# 广播:形状不同也能逐元素运算(完整规则见 08 批)
mat = np.array([[1, 2, 3], [4, 5, 6]])      # shape (2, 3)
row = np.array([10, 20, 30])                # shape (3,)
assert np.array_equal(mat + row, np.add(mat, row))
```

**常见坑:** 最容易搞混的是 `*`——数学课本里矩阵乘法写作 `AB`,不少人第一反应会觉得 numpy 里 `A * B` 也是矩阵乘法,但 `*` 永远是逐元素相乘(Hadamard 积),真正的矩阵乘法是 `@` 或 `np.matmul`(06 批细讲)。这个坑在 [03-how-to-look-up-not-memorize.md](../03-how-to-look-up-not-memorize.md) 里也提到过,这里从 ufunc 原理再强调一次:`+ - * /` 四个符号刚好对应上面四个 ufunc,没有第五个"矩阵乘法符号"藏在里面。

---

## 2. `np.exp` / `np.log`

**签名:**
```python
np.exp(x)      # e^x
np.log(x)      # 自然对数(以 e 为底),log(e^x) == x
```

**一句话:** `exp` 是自然指数函数,`log` 是它的反函数(自然对数)。

**AI 研究场景:** 这两个函数是深度学习论文代码里出现频率最高的数学函数——
- **softmax:** 把一组任意实数("logits")转成一个和为 1 的概率分布,公式是 `exp(x_i) / sum(exp(x))`。
- **交叉熵损失(cross-entropy loss):** 分类任务最常用的损失函数,核心是 `-log(p_正确类别)`——预测概率越接近 1,loss 越接近 0;预测得越离谱,loss 越大(甚至趋向无穷)。
- **对数似然(log-likelihood):** 概率模型的训练目标基本都是"最大化对数似然",取 log 是为了把连乘变成连加,避免很多个小于 1 的数相乘导致数值下溢(underflow)。

**可运行例子:**
```python
import numpy as np

x = np.array([0.0, 1.0, 2.0])
assert np.allclose(np.exp(x), [1.0, np.e, np.e**2])
assert np.allclose(np.log(np.exp(x)), x)          # log 是 exp 的反函数

def softmax(v):
    e = np.exp(v - v.max())     # 减去最大值,原因见下面的坑
    return e / e.sum()

s = softmax(np.array([2.0, 1.0, 0.1]))
assert abs(s.sum() - 1.0) < 1e-6

# 交叉熵损失
p = np.array([0.7, 0.2, 0.1])     # 模型预测的概率分布
label = 0                          # 真实类别是第 0 类
loss = -np.log(p[label])
assert loss > 0
assert abs(loss - 0.3567) < 1e-3

# exp 会溢出(overflow)成 inf,朴素 softmax 直接失效;减最大值后正常
scores = np.array([1000.0, 1001.0, 1002.0])
naive = np.exp(scores) / np.exp(scores).sum()
assert np.isnan(naive).all()              # 实测:inf/inf 变成 nan,朴素实现整体失败
stable = softmax(scores)
assert not np.isnan(stable).any()         # 减最大值后完全正常
assert abs(stable.sum() - 1.0) < 1e-6

# log(0) 和 log(负数) 不报错,只是产生特殊值(不中断程序)
assert np.log(0.0) == -np.inf             # 实测:返回 -inf,伴随 RuntimeWarning,不抛异常
assert np.isnan(np.log(-1.0))             # 实测:返回 nan,伴随 RuntimeWarning,不抛异常
```

**常见坑:**
1. **`exp` 会溢出成 `inf`,不会报错。** 实测 `np.exp(1000.0)` 结果是 `inf`,并弹出一条 `RuntimeWarning: overflow encountered in exp`(不是 Exception,程序不会中断)。这就是为什么手写 softmax 必须先减去最大值——`exp(x - x.max())` 和数学上 `exp(x)` 归一化后的结果完全等价(exp 的比值不变),但把最大输入压到 0,彻底避免了溢出。上面代码实测验证了朴素实现在大数值输入下直接输出 `[nan, nan, nan]`(`inf / inf` 是未定义的,变成 `nan`),这是新手实现 softmax 最容易踩的坑,也是为什么所有正经实现(包括 PyTorch 内部)都会做这一步。
2. **`log(0)` 不报错,返回 `-inf`;`log(负数)` 不报错,返回 `nan`。** 两者都只弹一条 `RuntimeWarning`(不会让程序崩溃)——这意味着如果模型预测概率算出来是 0(比如 softmax 后某一项下溢成 0.0),交叉熵损失会悄悄变成 `inf` 或 `nan`,不会有任何报错提示你哪里出了问题。这是数值稳定性问题最常见的根源,下面第 3 节的 `log1p`、第 9 节的 `clip` 都是围绕这个问题的解决方案。

---

## 3. `np.log2` / `np.log1p`

**签名:**
```python
np.log2(x)     # 以 2 为底的对数
np.log1p(x)    # 数学上等于 log(1 + x),但数值上精确得多
```

**一句话:** `log2` 是换了底数的对数(工程/信息论里常用);`log1p` 数学上等于 `np.log(1 + x)`,但专门为 `x` 很接近 0 的场景做了数值优化。

**AI 研究场景:**
- **`log2`:** 信息论里"熵"和"信息量"的标准单位是"比特(bit)",公式里的对数就是以 2 为底;语言模型的评估指标 bits-per-character/bits-per-byte 用的也是 `log2`(而普通交叉熵 loss 常用自然对数,单位是 nat——这也是为什么同一个模型在不同论文里报告的 loss 数值可能不一样,只是对数底数不同)。
- **`log1p`:** 任何"1 加一个很小的数再取 log"的场景都该用它,比如 KL 散度、某些奖励塑形(reward shaping)公式里的 `log(1+x)` 项,或者对一个接近 0 的概率增量做数值稳定的对数变换。

**可运行例子:**
```python
import numpy as np

assert np.log2(8) == 3.0          # 2^3 = 8
assert np.log2(1) == 0.0

# log1p 的数值稳定性:x 非常小的时候
tiny = 1e-16
assert (1 + tiny) == 1.0           # 实测:加法这一步就已经损失精度了!

naive = np.log(1 + tiny)           # 先做加法,精度已经丢了
stable = np.log1p(tiny)            # 直接对 x 本身做数值稳定的计算,不走"先加法"这条路

assert naive == 0.0                # 精度全部丢失,1e-16 这一项"凭空消失"
assert stable == tiny              # 实测:log1p 精确保留了 1e-16(位级相等)
```

**常见坑:** 问题根源在浮点数精度:`float64` 大约只有 15-17 位有效十进制数字,当 `x` 小到 `1 + x` 在浮点表示下直接"塌缩"成 `1.0`(实测 `x=1e-16` 时就会发生)时,`np.log(1+x)` 从一开始拿到的输入就已经是错的,后面算得再精确也没用。`np.log1p` 不走"先加法再取 log"这条路,所以能保留这份精度。**判断标准很简单:凡是代码里出现 `log(1 + 变量)` 这种模式,且变量可能很小,就应该改写成 `log1p(变量)`。**

---

## 4. `np.sqrt` / `np.power`

**签名:**
```python
np.sqrt(x)          # 平方根
np.power(x1, x2)    # x1 的 x2 次方,等价于 x1 ** x2
```

**一句话:** `sqrt` 是开平方;`power` 是通用的"底数的指数次方",指数也可以是逐元素不同的数组。

**AI 研究场景:**
- **`sqrt`:** 计算 L2 范数(向量长度)`np.sqrt((x**2).sum())`、标准差 `std = sqrt(var)`、Adam/RMSProp 优化器更新公式分母里的 `sqrt(二阶矩估计) + eps`、Transformer 缩放点积注意力(scaled dot-product attention)里除以 `sqrt(d_k)` 防止点积数值过大导致 softmax 梯度消失。
- **`power`:** 多项式特征构造(`x**2`、`x**3`)、学习率衰减公式里的指数项——多数时候更常见的写法其实是 `**` 运算符,`np.power` 是它的具名 ufunc 版本,和第 1 节的加减乘除是同一套逻辑。

**可运行例子:**
```python
import numpy as np

assert np.sqrt(4.0) == 2.0
assert np.allclose(np.sqrt([1, 4, 9]), [1, 2, 3])

assert np.allclose(np.power([2, 3], [3, 2]), [8, 9])      # 2^3=8, 3^2=9
assert np.power(2.0, -1) == 0.5                            # 浮点数支持负指数

# L2 范数的手写实现
v = np.array([3.0, 4.0])
l2_norm = np.sqrt((v ** 2).sum())
assert l2_norm == 5.0

# Attention 缩放:除以 sqrt(d_k)
d_k = 64
raw_scores = np.array([10.0, 20.0, 30.0])
scaled = raw_scores / np.sqrt(d_k)
assert np.allclose(scaled, raw_scores / 8.0)

# 整数数组的负数次幂:直接报错,不是返回小数
try:
    np.power(np.array([2]), np.array([-1]))
    raise AssertionError("应该抛出 ValueError")
except ValueError:
    pass                        # 实测确认:整数负指数会抛 ValueError

# 负数底数配非整数指数:结果是 nan,不是复数
assert np.isnan(np.power(-8.0, 1 / 3))     # 实测:不会聪明地算出实数立方根 -2
assert np.isnan(np.sqrt(-1.0))             # 同理,负数开方也是 nan(不报错)
```

**常见坑:**
1. **整数数组的负数次幂会直接报错**,不是返回小数——实测 `np.power(np.array([2]), np.array([-1]))` 抛出 `ValueError: Integers to negative integer powers are not allowed.`(结果理论上是分数,但数组 dtype 是整数,存不下)。想算负指数,数组要先是浮点型。
2. **负数底数配非整数指数,结果是 `nan`,不是数学上"应该"算出来的实数。** 实测 `np.power(-8.0, 1/3)` 返回 `nan`(带 `invalid value encountered in power` 警告),不会像"负八的立方根是 -2"那样聪明地算出来——numpy 默认只在实数范围内计算,负数开非整数次方在实数范围内无定义。想要"负数的立方根"得自己处理符号,比如 `np.sign(x) * np.abs(x)**(1/3)`。
3. `np.sqrt(-1.0)` 同理返回 `nan` + 警告,不报错——这和上一节 `log(负数)` 的行为完全一致,都是"进了实数域算不出来的运算,numpy 选择返回 `nan` 而不是抛异常"的统一风格。

---

## 5. `np.abs`

**签名:**
```python
np.abs(x)     # 绝对值,别名 np.absolute
```

**一句话:** 逐元素取绝对值,负数变正数,正数不变。

**AI 研究场景:** L1 损失/L1 正则化(MAE, Mean Absolute Error;Lasso 回归的正则项)直接由 `np.abs` 构成;梯度裁剪前先要看梯度的量级,常用 `np.abs(grad).max()` 快速看一眼梯度有没有爆炸;调试训练时比较"预测值和真实值差多少",看误差的绝对值比看有符号的差值更直观。

**可运行例子:**
```python
import numpy as np

assert np.array_equal(np.abs([-1, 2, -3]), [1, 2, 3])

grad = np.array([-0.5, 0.3, -1.2])
assert np.allclose(np.abs(grad), [0.5, 0.3, 1.2])
assert np.abs(grad).max() == 1.2          # 梯度里绝对值最大的分量,常用来判断是否爆炸

# L1 loss(MAE)
pred = np.array([1.0, 2.0, 3.0])
target = np.array([1.5, 1.5, 3.5])
l1_loss = np.abs(pred - target).mean()
assert abs(l1_loss - 0.5) < 1e-6
```

**常见坑:** 和 Python 内置的 `abs()` 用法几乎一样(numpy 数组重载了 `__abs__`,`abs(arr)` 和 `np.abs(arr)` 效果一致),看到别人代码里裸写 `abs(arr)` 不用大惊小怪,它就是在调用这个 ufunc。真正要小心的是**复数数组**——`np.abs` 对复数返回的是模长而不是"负号翻正",本篇 AI 场景不涉及复数,知道这个特殊情况存在即可。

---

## 6. `np.sign`

**签名:**
```python
np.sign(x)    # 正数 -> 1,负数 -> -1,零 -> 0
```

**一句话:** 只保留数值的"方向"(符号),丢掉大小信息。

**AI 研究场景:** 对抗样本经典方法 FGSM(Fast Gradient Sign Method)的公式就是 `x_adv = x + epsilon * sign(梯度)`——只用梯度的方向、不用具体大小,朝"loss 增大最快"的方向走一小步来构造对抗样本;类似地,SignSGD 一类优化器只用梯度符号更新参数(丢弃梯度量级,只保留方向)。

**可运行例子:**
```python
import numpy as np

assert np.array_equal(np.sign([-5, 0, 5]), [-1, 0, 1])
assert np.sign(0.0) == 0.0

# FGSM 对抗样本构造(简化示意)
x = np.array([0.5, 0.5, 0.5])
grad = np.array([-0.3, 0.1, 0.0])       # 假设这是 loss 对 x 的梯度
epsilon = 0.05
x_adv = x + epsilon * np.sign(grad)
assert np.allclose(x_adv, [0.45, 0.55, 0.5])
```

**常见坑:** `np.sign(0)` 返回 `0`,不是 `1` 或 `-1`——这个第三种取值容易被忽略,如果后续代码假设 `sign` 的结果只有 `±1` 两种可能(比如拿来做除法或索引),遇到输入恰好是 0 时就会出 bug。

---

## 7. `np.round` / `np.floor` / `np.ceil`

**签名:**
```python
np.round(a, decimals=0)   # 四舍五入到指定小数位(默认整数)
np.floor(x)                # 向下取整(向负无穷方向)
np.ceil(x)                 # 向上取整(向正无穷方向)
```

**一句话:** 三种不同的"取整"方式——`round` 找最近的(可以留小数位),`floor`/`ceil` 分别是"往下"和"往上"钉死到整数。

**AI 研究场景:** `np.ceil(len(dataset) / batch_size)` 是计算"一个 epoch 有多少个 batch"的标准写法(数据集不能被 `batch_size` 整除时,最后一个不完整的 batch 也要算一个,所以用 `ceil` 而不是普通除法或 `floor`);把连续值离散化分桶(分位数分桶、直方图边界)常用 `floor`;`round` 常见于打印/记录日志时控制小数位数,让输出好看。

**可运行例子:**
```python
import numpy as np

assert np.floor(2.7) == 2.0
assert np.ceil(2.1) == 3.0
assert np.floor(-2.1) == -3.0        # 注意:负数下 floor 是"更小"的那个整数
assert np.ceil(-2.1) == -2.0

# 计算一个 epoch 有几个 batch(向上取整,不能漏掉尾巴)
dataset_size, batch_size = 100, 32
num_batches = int(np.ceil(dataset_size / batch_size))
assert num_batches == 4               # 32+32+32+4,最后一批不满也要算一个

# 银行家舍入(round half to even):四舍五入到"最近的偶数"
assert np.round(0.5) == 0.0
assert np.round(1.5) == 2.0
assert np.round(2.5) == 2.0
assert np.round(3.5) == 4.0

# 进阶:decimals 参数走"乘10^n再舍入再除回去",可能引入和内置 round 不一致的结果
assert round(2.675, 2) == 2.67        # Python 内置 round:老实按 2.675 的真实浮点值向下舍
assert np.round(2.675, 2) == 2.68     # 实测:numpy 的结果不一样(原因见下面的坑)
```

**常见坑:**
1. **`np.round` 用的是"round half to even"(银行家舍入),不是小学学的"四舍五入"(round half up)。** 实测 `0.5 -> 0`、`1.5 -> 2`、`2.5 -> 2`、`3.5 -> 4`——正好卡在 `.5` 上时,规则是"向最近的偶数走",而不是永远向上。这是 IEEE 754 浮点标准推荐的舍入方式(避免"永远进位"带来的系统性偏差),Python 内置的 `round()` 也是同一套规则,不是 numpy 独有,但确实和很多人的直觉不一样。
2. **C 语言背景要小心:`floor` 和"强制转 int 截断"是两回事。** C 里 `(int)(-2.1)` 是截断向零(结果 `-2`,numpy 对应的是 `np.trunc`,实测同样是 `-2.0`),但 `np.floor(-2.1)` 是向负无穷方向取整(结果 `-3.0`)——两者在正数上结果一样,在负数上会差 1,这是从 C 转过来最容易踩的一个符号坑。
3. **进阶细节:** `np.round` 处理小数位数(`decimals != 0`)时,内部大致是"乘 `10^n` → 按银行家舍入取整 → 再除回去",这个"乘"和"除"本身也是浮点运算,可能引入新的误差。实测 `np.round(2.675, 2)` 结果是 `2.68`,而 Python 内置 `round(2.675, 2)` 结果是 `2.67`——两者不一致,原因是 `2.675` 在二进制浮点下真实存的值是 `2.67499999999999982...`(比 2.675 略小),Python 的 `round` 老实按这个真实值向下舍;而 `2.675 * 100` 这一步浮点乘法运算的结果恰好被舍入成了正好等于 `267.5` 的值,银行家舍入下 `267.5` 舍向偶数 `268`,除回去就是 `2.68`。**结论:凡是涉及"精确小数"的场景(比如钱),不要相信任何浮点数的 round,该用 `decimal.Decimal`。**

---

## 8. `np.maximum` / `np.minimum`

**签名:**
```python
np.maximum(x1, x2)    # 逐元素取较大值
np.minimum(x1, x2)    # 逐元素取较小值
```

**一句话:** 两个数组(或数组与标量)按位置一一比较,每个位置各自留下较大(或较小)的那个——输出形状和输入一样。

**AI 研究场景:** **ReLU 激活函数的完整实现就是一行 `np.maximum(x, 0)`**——负数被压成 0,正数保持不变,这是神经网络里最基础的非线性单元;给一个可能是 0 的分母/概率设一个下限也常用 `np.maximum(x, eps)`,防止后续除法或取 log 时炸出 `inf`/`nan`(呼应第 2 节的坑)。

**这里必须做一个关键区分**(05 批会详细展开):`np.maximum`/`np.minimum` 是**逐元素**操作,两个输入数组形状要能广播、输出形状不变;而 `np.max`/`np.min` 是**归约(reduction)**操作,对**一个**数组沿某个轴"压扁"成更少的元素(甚至一个标量)。名字长得像,但一个是"比较两堆数",一个是"从一堆数里挑最大的",完全是两回事。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 5, 3])
b = np.array([4, 2, 6])
assert np.array_equal(np.maximum(a, b), [4, 5, 6])     # 每个位置各自取较大值
assert np.array_equal(np.minimum(a, b), [1, 2, 3])

def relu(v):
    return np.maximum(v, 0)

assert np.array_equal(relu(np.array([-1, 0, 2])), [0, 0, 2])

# 和归约操作的区别(05 批详解,这里只做对比)
assert np.max(a) == 5                       # 归约:一个数组 -> 一个标量
assert np.maximum(a, b).shape == a.shape    # 逐元素:形状不变
```

**常见坑:** 把 `np.maximum(a, b)`(两个数组的逐元素比较)和 `np.max(a)`(一个数组的归约)搞混,是这一节最容易犯的错——`maximum` 后面必须跟两个"东西"去比较,`max` 只对一个数组自己内部找最大值。函数名故意起得很像,查文档/敲代码时务必看清楚是几个参数。

---

## 9. `np.clip`

**签名:**
```python
np.clip(a, a_min, a_max)
```
- `a_min`/`a_max` 任一个传 `None` 表示"这一头不设限"(不能直接省略不写)

**一句话:** 把数组里所有小于 `a_min` 的值拉到 `a_min`,所有大于 `a_max` 的值压到 `a_max`,中间的值不变——"夹住"数值范围。

**AI 研究场景:**
- **梯度裁剪(gradient clipping):** 防止梯度爆炸,训练循环里常见 `grad = np.clip(grad, -clip_value, clip_value)`(这是按值裁剪的简化版本,实践中更常见按整体范数裁剪,但思路一致)。
- **PPO 的概率比裁剪:** 强化学习 PPO 算法的核心机制之一,是把新旧策略的概率比 `ratio` 裁剪到 `[1-epsilon, 1+epsilon]` 区间,防止单次更新步子迈得太大导致策略崩溃,公式就是 `np.clip(ratio, 1-epsilon, 1+epsilon)`。
- **裁剪概率再取 log:** 呼应第 2 节的坑——如果模型预测的概率可能算出 0 或 1,直接取 `log` 会产生 `-inf`,标准做法是先 `p = np.clip(p, 1e-12, 1 - 1e-12)` 再 `np.log(p)`,彻底避免数值爆炸。

**可运行例子:**
```python
import numpy as np

x = np.array([-2, 0.5, 3, 10])
assert np.array_equal(np.clip(x, 0, 5), [0, 0.5, 3, 5])

# 梯度裁剪
grad = np.array([-10.0, 0.5, 10.0])
clipped_grad = np.clip(grad, -1.0, 1.0)
assert np.array_equal(clipped_grad, [-1.0, 0.5, 1.0])

# PPO 概率比裁剪
ratio = np.array([0.7, 1.0, 1.5])
epsilon = 0.2
clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
assert np.array_equal(clipped_ratio, [0.8, 1.0, 1.2])

# 裁剪概率,避免 log(0) 产生 -inf
p = np.array([0.0, 0.5, 1.0])
p_safe = np.clip(p, 1e-12, 1 - 1e-12)
loss = -np.log(p_safe)
assert np.isfinite(loss).all()          # 裁剪后不会再出现 -inf
```

**常见坑:** 只想限制一头(比如只要下限,不要上限)时,另一个参数要显式传 `None`,不能直接省略不写——`np.clip(x, 0, 5)` 里两个参数都是位置参数,写成 `np.clip(x, 0)` 会报错缺参数,必须写 `np.clip(x, 0, None)`。另外 `clip` 默认返回一个新数组,不会改变原数组(除非显式传 `out=` 参数)。

---

## 10. `np.nan_to_num`

**签名:**
```python
np.nan_to_num(x, nan=0.0, posinf=None, neginf=None)
```
- `nan`:`nan` 要替换成的值,默认 `0.0`
- `posinf`/`neginf`:正/负无穷要替换成的值,默认分别替换成该 dtype 能表示的最大/最小有限值

**一句话:** 把数组里的 `nan`/`inf`/`-inf` 这些"非正常"浮点值,替换成正常的有限数字。

**AI 研究场景:** 训练大模型时 loss 偶尔"炸"出 `nan`(学习率设太大、梯度爆炸、混合精度溢出等原因),如果不处理,`nan` 会像病毒一样通过后续每一步计算污染所有相关参数(`nan` 参与任何运算,结果还是 `nan`)。`nan_to_num` 常被用作训练脚本里的"安全网",或者在对整个数据集做统计汇总时,过滤掉个别异常样本产生的 `nan`/`inf`,防止一颗老鼠屎坏了一锅汤(比如 `array.mean()` 只要有一个 `nan`,平均值就整体变成 `nan`)。

**可运行例子:**
```python
import numpy as np

x = np.array([1.0, np.nan, np.inf, -np.inf])
y = np.nan_to_num(x)
assert not np.isnan(y).any()
assert np.isfinite(y).all()

# 自定义替换值,更贴近实际训练场景的用法
y2 = np.nan_to_num(x, nan=-1.0, posinf=1e6, neginf=-1e6)
assert y2[1] == -1.0
assert y2[2] == 1e6
assert y2[3] == -1e6

# 典型场景:一个 nan 污染整体统计
polluted = np.array([1.0, 2.0, np.nan, 4.0])
assert np.isnan(polluted.mean())              # 直接算 mean,结果是 nan
cleaned = np.nan_to_num(polluted, nan=0.0)
assert cleaned.mean() == 1.75                 # 清洗后能正常统计
```

**常见坑:** `nan_to_num` 只是"掩盖症状",不是"治病"——训练里出现 `nan` 往往说明学习率、初始化或数值精度哪里出了真正的问题,把 `nan` 悄悄替换成 0 可能会让训练"看起来正常"但实际上已经跑偏。**排查阶段更推荐先用 `np.isnan(x).any()` 主动检测并报错定位**,确认问题根源后再考虑要不要在生产代码里加这层保护。

---

## 11. `np.sin` / `np.cos`

**签名:**
```python
np.sin(x)    # 正弦,x 是弧度制
np.cos(x)    # 余弦,x 是弧度制
```

**一句话:** 三角函数,逐元素计算,输入是弧度(不是角度)。

**AI 研究场景:** Transformer 的正弦位置编码(sinusoidal positional encoding)——原始论文《Attention Is All You Need》给每个 token 位置分配一个由 `sin`/`cos` 组成的向量,不同频率的正弦/余弦波交替填充向量的偶数/奇数维,让模型在没有循环结构的情况下也能感知 token 的顺序信息。这是除了 exp/log 之外,论文代码里第二常见的数学函数组合。

**可运行例子:**
```python
import numpy as np

assert np.allclose(np.sin(0), 0)
assert np.allclose(np.cos(0), 1)
assert np.allclose(np.sin(np.pi / 2), 1.0)

# 简化版 Transformer 位置编码
d_model, seq_len = 4, 10
pos = np.arange(seq_len)[:, None]              # shape (10, 1)
i = np.arange(d_model)[None, :]                # shape (1, 4)
angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
angles = pos * angle_rates                     # 广播成 (10, 4)

pe = np.zeros((seq_len, d_model))
pe[:, 0::2] = np.sin(angles[:, 0::2])          # 偶数维用 sin
pe[:, 1::2] = np.cos(angles[:, 1::2])          # 奇数维用 cos

assert pe.shape == (seq_len, d_model)
assert np.allclose(pe[0, 0::2], 0.0)           # 第 0 个位置,sin(0)=0
assert np.allclose(pe[0, 1::2], 1.0)           # 第 0 个位置,cos(0)=1
```

**常见坑:** numpy 的三角函数入参是**弧度**,不是角度——如果你的数据是角度(比如 0-360 度),要先手动乘 `np.pi / 180`(或者用 `np.deg2rad`)转成弧度,直接把角度数值传进去会得到完全错误的结果(比如指望 `sin(90度)=1`,`np.sin(90)` 实际算的是"90 弧度"的正弦,是个跟 1 不沾边的数)。

---

## 12. `np.mod` / `np.remainder`

**签名:**
```python
np.mod(x1, x2)          # 取余数,结果符号跟随 x2(除数)
np.remainder(x1, x2)    # 和 np.mod 是同一个函数的两个名字
```

**一句话:** 取模(余数)运算,`np.mod` 和 `np.remainder` 完全是同一个东西的两个别名。

**AI 研究场景:** 训练循环里"每隔 N 步做一次日志/保存 checkpoint"的判断 `if step % log_every == 0`;经验回放缓冲区(replay buffer)用循环下标写入新数据 `buffer[idx % capacity] = new_sample`,实现一个固定大小的环形缓冲区;K 折交叉验证按 `sample_idx % k` 把样本分配到不同的折。

**可运行例子:**
```python
import numpy as np

assert np.mod(5, 3) == 2
assert np.mod is np.remainder                # 实测确认:两者是同一个函数对象

# numpy 的 mod 符号跟随"除数"(和 Python 内置 % 一致)
assert np.mod(-5, 3) == 1
assert np.remainder(-5, 3) == 1
assert (-5) % 3 == 1                          # Python 内置 % 行为一致
assert np.fmod(-5, 3) == -2                   # 对比:C 风格的 fmod 符号跟随"被除数"

# 环形缓冲区下标(replay buffer 的典型写法)
capacity = 5
buffer = np.zeros(capacity)
for step in range(8):
    buffer[step % capacity] = step
assert list(buffer) == [5, 6, 7, 3, 4]        # 步数超过容量后,从头覆盖
```

**常见坑:** 如果你有 C 语言背景,要注意 numpy/Python 的取模和 C 的 `%` 在**负数**上的行为不一样——C 的 `%` 结果符号跟随"被除数"(实测 `np.fmod(-5, 3)` 结果是 `-2`,和 C 的 `%` 一致),而 numpy 的 `np.mod`/`np.remainder`/Python 内置 `%` 结果符号跟随"除数"(实测 `np.mod(-5, 3)` 是 `1`)。numpy 里 `np.fmod` 就是专门给 C 风格取模用的,两个函数长得像但结果不同——写循环下标这种"取余数必须非负"的场景,一定要用 `np.mod`/`%`,不要手滑用成 `np.fmod`。

---

## 13. 比较运算符 `> < ==` vs `np.greater`/`np.less` 等

**签名:**
```python
np.greater(x1, x2)         # 等价于 x1 > x2
np.less(x1, x2)             # 等价于 x1 < x2
np.equal(x1, x2)            # 等价于 x1 == x2
np.greater_equal(x1, x2)    # 等价于 x1 >= x2
np.less_equal(x1, x2)       # 等价于 x1 <= x2
np.not_equal(x1, x2)        # 等价于 x1 != x2
```

**一句话:** 和第 1 节的 `+ - * /` 一模一样的道理——`> < == >= <= !=` 这几个比较运算符,同样是 numpy 重载出来转发给对应 ufunc 的语法糖,不是什么独立的机制。

**AI 研究场景:** 几乎所有布尔掩码(mask)的构造起点都是这些比较运算符——阈值化预测结果(`preds > 0.5` 转成二分类标签)、筛选满足条件的样本、构造 attention 里的 padding mask(`token_ids != pad_id`)。[03-indexing-and-selection.md](03-indexing-and-selection.md) 会专门讲怎么用这些比较结果去做布尔索引(`a[a > 0]`),这里先建立"比较符号 = ufunc"这一层认识。

**可运行例子:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

assert np.array_equal(a > b, np.greater(a, b))
assert np.array_equal(a < b, np.less(a, b))
assert np.array_equal(a == b, np.equal(a, b))
assert np.array_equal(a >= b, np.greater_equal(a, b))
assert np.array_equal(a <= b, np.less_equal(a, b))
assert np.array_equal(a != b, np.not_equal(a, b))

# 典型用法:把预测概率阈值化成二分类标签
probs = np.array([0.2, 0.6, 0.9, 0.4])
labels = (probs > 0.5).astype(int)
assert list(labels) == [0, 1, 1, 0]
```

**常见坑:** 比较运算符返回的永远是一个**布尔数组**(dtype 是 `bool`,形状和输入一致),不是单个 `True`/`False`。如果直接把它塞进 Python 的 `if` 判断(比如数组元素超过一个时写 `if a > b: ...`),会报错——这个报错的原因和第 14 节的 `and`/`or` 报错是同一个机制,放在下一节一起讲清楚。

---

## 14. `np.logical_and` / `np.logical_or` / `np.logical_not`

**签名:**
```python
np.logical_and(x1, x2)    # 逐元素"与"
np.logical_or(x1, x2)     # 逐元素"或"
np.logical_not(x)          # 逐元素"非"
```

**一句话:** 布尔数组之间的逐元素逻辑运算,替代 Python 内置的 `and`/`or`/`not`(那三个关键字在多元素数组上根本用不了)。

**AI 研究场景:** 构造复合条件的 mask 是最常见场景——Transformer 的 attention mask 往往需要"causal mask(不能看未来)**且**padding mask(不能看 padding 位置)"两个条件同时满足,就是 `np.logical_and(causal_mask, padding_mask)`;筛选"损失异常大**或**梯度异常大"的样本做调试用 `np.logical_or`;反选一个 mask(比如"非 padding 位置")用 `np.logical_not`。

**可运行例子:**
```python
import numpy as np

la = np.array([True, True, False])
lb = np.array([True, False, False])

assert np.array_equal(np.logical_and(la, lb), [True, False, False])
assert np.array_equal(np.logical_or(la, lb), [True, True, False])
assert np.array_equal(np.logical_not(la), [False, False, True])

# 复合 mask:分数要同时满足两个区间条件
scores = np.array([0.1, 0.6, 0.9])
mask = np.logical_and(scores > 0.5, scores < 0.95)
assert np.array_equal(mask, [False, True, True])

# 运算符版本:& | ~ 对布尔数组等价于 logical_and/or/not,但优先级比比较运算符高,必须加括号
mask2 = (scores > 0.5) & (scores < 0.95)
assert np.array_equal(mask2, mask)

# Python 的 and/or/not 在多元素数组上直接报错(实测报错信息见下面的坑)
try:
    if la and lb:
        pass
    raise AssertionError("不应该执行到这里")
except ValueError as e:
    assert "truth value" in str(e)
```

**常见坑:**
1. **Python 内置的 `and`/`or`/`not` 不能用在(元素数 > 1 的)numpy 数组上**,实测报错信息是:`ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`。原因是 Python 的 `and`/`or`/`not` 要求操作数能被转换成单一的 `True`/`False`(调用 `__bool__`),但一个多元素数组"整体是真是假"是有歧义的(有的元素 True 有的 False,该算整体真还是假?)。**记住:数组按位置比较用 `&`/`|`/`~`(或 `np.logical_and` 等函数),数组的"整体真假判断"用 `.any()`/`.all()`,Python 的 `and`/`or`/`not` 只留给单个布尔值的场景(比如 `if x > 0 and y > 0:`,这里 `x`、`y` 都是标量)。**
2. **用 `&`/`|` 代替 `np.logical_and`/`np.logical_or` 时,一定要给每个比较表达式加括号**——因为 `&` 的运算符优先级比 `>`/`<` 更高,`scores > 0.5 & scores < 0.95` 会先算 `0.5 & scores`,实测直接抛 `TypeError`(浮点数不支持位运算)。必须写成 `(scores > 0.5) & (scores < 0.95)`,这是新手用运算符简写形式时最容易忽略的括号坑。

---

## 15. `np.where`(三元表达式用法)

**签名:**
```python
np.where(condition, x, y)
```
- `condition`:布尔数组
- `x`:`condition` 对应位置为 `True` 时取这里的值
- `y`:`condition` 对应位置为 `False` 时取这里的值
- 输出形状和 `condition`(经广播后)一致

**一句话:** 数组版本的三元表达式 `a if cond else b`——在整个数组上逐元素地做"if-else"。

**AI 研究场景:** 需要"满足条件用一个值、不满足用另一个值,但整体形状不能变"的场景——比如 attention 里把 padding 位置的分数替换成 `-inf`(`np.where(padding_mask, -np.inf, scores)`),让 softmax 之后这些位置的权重趋近于 0;分段函数的实现(比如 Huber Loss 误差小时用平方项、误差大时用线性项);比 Python 的 `if/else` 好在完全向量化,不需要写 for 循环逐元素判断。

**可运行例子:**
```python
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
y = np.where(x > 0, x, 0)                # 三元表达式:>0 保留原值,否则填 0(等价于 relu)
assert np.array_equal(y, [0, 0, 0, 1, 2])

# 两个数组之间做条件选择
p = np.array([1, 2, 3, 4, 5])
q = np.array([10, 20, 30, 40, 50])
cond = np.array([True, False, True, False, True])
r = np.where(cond, p, q)
assert np.array_equal(r, [1, 20, 3, 40, 5])

# np.where 只传条件时是完全不同的"选择器"用法(见下面的坑)
idx = np.where(x > 0)
assert isinstance(idx, tuple)                  # 返回下标元组,不是数值!
assert np.array_equal(x[idx], x[x > 0])        # 配合下标取值,效果等价于布尔索引
```

**常见坑:** `np.where` 其实有两种完全不同的调用方式,是本节最容易踩的坑——
- **三元表达式用法(本节主角):** `np.where(cond, x, y)`,传 3 个参数,返回一个和 `cond` 同形状的数组,把"选出满足条件的元素"和"结果形状不变"两件事同时做到。
- **只传条件的"选择器"用法([03-indexing-and-selection.md](03-indexing-and-selection.md) 的内容):** `np.where(cond)` 只传 1 个参数,返回值完全是另一回事——一个"每个维度一条"的下标数组组成的元组,效果上等价于对满足条件的位置做布尔索引 `a[cond]`,但拿到的是下标而不是值本身。

同一个函数名,传 1 个参数和传 3 个参数是两套完全不同的行为,是 numpy API 里少数几个"参数个数决定语义"的地方——用之前务必想清楚自己要的是"整体形状不变的条件替换"(3 参数)还是"满足条件的下标"(1 参数)。

---

## 小结:这一批 15 个知识点解决的问题

| # | 函数 | 解决的问题 |
|---|---|---|
| 1 | `+ - * /`(`add`/`subtract`/`multiply`/`divide`) | 逐元素四则运算的 ufunc 本质,后续一切的地基 |
| 2 | `exp`/`log` | softmax、交叉熵、对数似然;exp 溢出与 log(0) 的数值稳定性 |
| 3 | `log2`/`log1p` | 信息论比特单位;小数值取对数的精度保护 |
| 4 | `sqrt`/`power` | L2 范数、Adam 分母、attention 缩放;负数开方/负指数的边界行为 |
| 5 | `abs` | L1 loss/正则化、梯度量级检查 |
| 6 | `sign` | FGSM 对抗样本、SignSGD 只用梯度方向 |
| 7 | `round`/`floor`/`ceil` | 取整、batch 数计算(踩坑:银行家舍入) |
| 8 | `maximum`/`minimum` | ReLU 实现;逐元素比较(区别于归约的 `max`/`min`) |
| 9 | `clip` | 梯度裁剪、PPO 概率比裁剪、log 前的安全区间 |
| 10 | `nan_to_num` | 清洗训练中出现的 nan/inf,防止污染统计 |
| 11 | `sin`/`cos` | Transformer 正弦位置编码 |
| 12 | `mod`/`remainder` | 环形缓冲区下标、周期性步数判断 |
| 13 | 比较运算符(`greater` 等) | 阈值化、构造 mask 的起点 |
| 14 | `logical_and`/`or`/`not` | 复合 mask 条件(替代不能用的 Python `and`/`or`) |
| 15 | `where`(三元用法) | 无分支的逐元素条件替换(attention mask 等) |

下一批:[05-reduction-and-statistics.md](05-reduction-and-statistics.md) —— 归约与统计。

---

*更新:2026-07-07*
