# 06 · 线性代数(Linear Algebra)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这组函数解决一个问题:**怎么用 numpy 做真正的线性代数运算**——不是逐元素的 `+ - * /`,而是矩阵乘法、求逆、分解这些"整体作用在矩阵/向量上"的操作。这是 AI 研究代码里含金量最高的一批 numpy 函数:PCA、注意力机制、最小二乘、正则化、采样多元正态分布,底层全是这几页里的函数。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,不是凭空写的。

---

## 1. `np.dot`

**签名:**
```python
np.dot(a, b)
```
- `a`, `b`:数组,维度可以是标量、向量、矩阵,甚至更高维

**一句话:** 广义的"点乘"——1D 和 1D 是向量点积,2D 和 2D 是矩阵乘法,再往高维走语义会变复杂(下一节详细讲这个坑)。

**AI 研究场景:** 手写一个没有 batch 维的线性层 `y = np.dot(W, x)`、计算两个向量的相似度(点积是余弦相似度的分子)、验证反向传播手推的梯度公式——这些"教科书式"的线性代数场景很多人第一反应就是 `dot`。实际工程代码里 2D 及以下场景 `dot` 和下一节的 `@` 完全等价,选哪个更多是风格问题。

**可运行例子:**
```python
import numpy as np

# 1D · 1D -> 标量点积
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
s = np.dot(u, v)
assert np.allclose(s, 32.0)                 # 1*4 + 2*5 + 3*6

# 2D · 2D -> 矩阵乘法,和 @ 完全等价
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
assert np.allclose(np.dot(A, B), A @ B)

# 2D · 1D -> 矩阵作用在向量上,线性层没有 batch、没有 bias 的情形
x = np.array([1.0, 1.0])
y = np.dot(A, x)
assert np.allclose(y, [3.0, 7.0])
```

**常见坑:** `dot` 在 2D 以内和 `@`/`matmul` 完全一样,很多教程/老代码到处混用,这没问题。**但一旦数组变成 3D 及以上(比如加了 batch 维),`dot` 和 `matmul` 的结果会彻底不同**——这不是小细节,是真实项目里会让人怀疑 numpy 是不是有 bug 的坑,下一节专门拆解。

---

## 2. `@` / `np.matmul`

**签名:**
```python
a @ b
np.matmul(a, b)     # 和 @ 完全等价,@ 就是 __matmul__ 的语法糖
```

**一句话:** 矩阵乘法。2D 情形下和 `dot` 一样;3D 及以上时,`matmul` 把最后两个维度当成"一个矩阵",前面的维度当成**批量(batch)维**——这才是深度学习代码里"批量矩阵乘法"的正确语义。

**AI 研究场景:** 神经网络的核心操作 `output = input @ weight + bias` 绝大多数场景都带 batch 维,比如 `input` 的 shape 是 `(batch, seq_len, hidden)`,要对每个样本独立做矩阵乘法——这正是 `matmul`/`@` 存在的意义:**不用写 for 循环遍历 batch,一行代码对整个 batch 并行处理**。

**dot 和 matmul 在高维到底差在哪(必须搞清楚的坑):**

假设 `a.shape = (batch, n, m)`,`b.shape = (batch, m, p)`(典型的"一个 batch 的矩阵,配一个 batch 的矩阵"):

| 写法 | 结果 shape | 语义 |
|---|---|---|
| `np.matmul(a, b)` / `a @ b` | `(batch, n, p)` | **批量对齐**:第 `i` 个结果只用第 `i` 个 `a` 和第 `i` 个 `b`,和你直觉里的"batch matmul"一致 |
| `np.dot(a, b)` | `(batch, n, batch, p)` | **不对齐**:对 `a` 的最后一维和 `b` 的倒数第二维求和,但 `a` 的 batch 维和 `b` 的 batch 维各自独立保留,相当于把所有 batch 两两组合都算了一遍(笛卡尔积) |

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
batch = 4
a = np.random.randn(batch, 3, 4)     # 4 个 (3,4) 矩阵
b = np.random.randn(batch, 4, 2)     # 4 个 (4,2) 矩阵,每个 batch 配自己的一份

# --- matmul / @:批量对齐,逐个矩阵乘法 ---
r_matmul = np.matmul(a, b)
assert r_matmul.shape == (4, 3, 2)
for i in range(batch):
    assert np.allclose(r_matmul[i], a[i] @ b[i])     # 第 i 个结果只用第 i 组数据

# --- dot 在高维:不对齐,笛卡尔积式求和 ---
r_dot = np.dot(a, b)
assert r_dot.shape == (4, 3, 4, 2)                    # 形状完全不同!
# 真正对应"batch i 配 batch i"的结果,其实是挂在 r_dot 的对角线上
for i in range(batch):
    assert np.allclose(r_dot[i, :, i, :], r_matmul[i])

# 2D 情形下两者才是完全等价的(小例子里容易误以为永远等价)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
assert np.allclose(np.dot(A, B), np.matmul(A, B))
```

**常见坑:** 写批量矩阵乘法代码时,**只要数组是 3D 及以上,永远用 `@`/`matmul`,不要用 `dot`**——这是新手从教程(教程里全是 2D 小例子,两者表现一致)迁移到真实 batch 训练代码时最容易踩的坑,`dot` 不会报错,只会安静地给你一个形状完全不对、多算了一堆无意义交叉项的结果,调试起来非常痛苦。记忆窍门:**"matmul 里的 mul 让你联想到 batch 并行相乘;dot 高维语义反直觉,高维场景直接从选项里划掉。"**

---

## 3. `np.outer` / `np.inner`

**签名:**
```python
np.outer(a, b)
np.inner(a, b)
```
- `outer`:输入不论原始形状,先拉平成 1D,再产生 `(len(a), len(b))` 的矩阵
- `inner`:1D 情形下就是普通点积;2D 情形下等价于 `a @ b.T`

**一句话:** `outer` 是"外积"(两个向量生成一个矩阵,`a` 的每个元素乘 `b` 的整个向量);`inner` 是"内积"的推广(点积,或者矩阵按行两两点积)。

**AI 研究场景:**
- **outer:** 手推反向传播时,单样本线性层 `y = W @ x` 的梯度公式是 `dW = outer(dy, x)`——上游梯度 `dy`(shape 等于输出维度)和输入 `x`(shape 等于输入维度)做外积,直接得到形状和 `W` 一致的梯度矩阵,这是全连接层反传最核心的一行公式。
- **inner:** 一批 query 向量和一批 key/候选向量,想要"每个 query 对每个候选的相似度"一次性算出来(检索系统、推荐系统的核心操作),`np.inner(A, B)` 比手写 `A @ B.T` 更直接地表达"我要的是内积矩阵"。

**可运行例子:**
```python
import numpy as np

# outer: 单样本线性层反传梯度
delta = np.array([0.1, 0.2])          # 上游梯度,shape=(2,) 对应输出维度
x = np.array([1.0, 2.0, 3.0])         # 该层输入,shape=(3,) 对应输入维度

dW = np.outer(delta, x)               # dW[i,j] = delta[i] * x[j]
assert dW.shape == (2, 3)
assert np.allclose(dW, [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])

# inner: 1D 情形等价于 dot
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
assert np.allclose(np.inner(u, v), np.dot(u, v))

# inner: 2D 情形等价于 A @ B.T —— 一次算出所有行两两的相似度矩阵
A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # 3 个 query 向量
B = np.array([[1.0, 1.0], [1.0, 0.0]])               # 2 个 key 向量
sim = np.inner(A, B)
assert sim.shape == (3, 2)
assert np.allclose(sim, A @ B.T)
```

**常见坑:** `outer` 会先把输入拉平成 1D 再计算——如果传进去的是一个 `(2,3)` 的矩阵,不会报错,而是先摊平成长度 6 的向量再做外积,输出形状可能不是你以为的样子。`inner` 在维度 ≥2 时的求和规则(只对最后一维求和)和下面 `einsum`/`dot` 的高维规则不完全一样,超过 2D 的场景不确定就直接用 `einsum` 显式写清楚下标,不要依赖记忆。

---

## 4. `np.linalg.norm`

**签名:**
```python
np.linalg.norm(x, ord=None, axis=None, keepdims=False)
```
- `x`:向量或矩阵
- `ord`:范数类型——`None`(默认)向量是 L2、矩阵是 Frobenius;`1` 是 L1;`2` 是 L2;`np.inf` 是无穷范数
- `axis`:沿哪个轴计算(比如对一批向量逐行求范数,不拉平整体算)

**一句话:** 衡量向量"有多大"/矩阵"有多大"的标准工具,`ord` 决定"用什么规则衡量大小"。

**`ord` 对向量的含义(正则化里最常用到的三种):**

| `ord` | 名称 | 公式(向量 `x`) | 直觉 | 典型用途 |
|---|---|---|---|---|
| `1` | L1 范数 | `sum(\|x_i\|)` | 所有分量绝对值加起来 | Lasso/L1 正则化,鼓励**稀疏**(很多分量变成恰好 0) |
| `2`(默认) | L2 范数 | `sqrt(sum(x_i^2))` | 欧几里得距离/模长 | weight decay/L2 正则化,鼓励权重整体**变小但不一定为 0**;梯度裁剪的裁剪依据 |
| `np.inf` | 无穷范数 | `max(\|x_i\|)` | 绝对值最大的那个分量 | 衡量"最坏情况下单个分量有多大",比如逐元素误差上界 |

**AI 研究场景:**
- **L2 正则化 / weight decay:** loss 里加一项 `0.5 * lambda * ||W||_2^2`,抑制权重过大、缓解过拟合。
- **梯度裁剪(gradient clipping):** 训练 RNN/Transformer 时,按梯度的**全局 L2 范数**裁剪,防止梯度爆炸——如果范数超过阈值,就把梯度整体按比例缩小,方向不变。
- **向量归一化:** 对比学习(contrastive learning)、检索系统里算余弦相似度前,先把向量除以自己的 L2 范数变成单位向量。

**可运行例子:**
```python
import numpy as np

w = np.array([3.0, -4.0, 0.0, 1.0])

# 默认(ord=None)== ord=2,向量情形下是 L2 范数
assert np.allclose(np.linalg.norm(w), np.linalg.norm(w, ord=2))
assert np.allclose(np.linalg.norm(w), np.sqrt(np.sum(w**2)))

# ord=1: L1 范数,绝对值之和
assert np.allclose(np.linalg.norm(w, ord=1), 8.0)          # 3+4+0+1

# ord=inf: 绝对值最大的分量
assert np.allclose(np.linalg.norm(w, ord=np.inf), 4.0)

# axis 参数:对一批向量(比如一批 embedding)逐行求范数
embeds = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
row_norms = np.linalg.norm(embeds, axis=1)
assert np.allclose(row_norms, [5.0, 1.0, 5.0])

# 梯度裁剪的标准写法
grad = np.array([10.0, 10.0, 10.0])
max_norm = 5.0
grad_norm = np.linalg.norm(grad)
grad_clipped = grad * (max_norm / grad_norm) if grad_norm > max_norm else grad
assert np.allclose(np.linalg.norm(grad_clipped), max_norm)

# 向量归一化(单位向量化)
v = np.array([3.0, 4.0])
v_unit = v / np.linalg.norm(v)
assert np.allclose(np.linalg.norm(v_unit), 1.0)
```

**常见坑:** `ord=1`/`ord=np.inf` 对**矩阵**的含义和对**向量**完全不同——矩阵的 `ord=1` 是"绝对值列和的最大值"(不是所有元素绝对值加起来!),`ord=np.inf` 是"绝对值行和的最大值"。
```python
import numpy as np
M = np.array([[1.0, -7.0], [3.0, 4.0]])
assert np.allclose(np.linalg.norm(M, ord=1), np.abs(M).sum(axis=0).max())   # 11.0,不是整体绝对值和15.0
assert not np.allclose(np.linalg.norm(M, ord=1), np.abs(M).sum())
```
不确定就先打印 `x.shape` 看清楚自己在算向量范数还是矩阵范数。

---

## 5. `np.linalg.inv`

**签名:**
```python
np.linalg.inv(a)
```
- `a`:方阵(行数=列数),且必须可逆(非奇异)

**一句话:** 求矩阵的逆 `A⁻¹`,满足 `A @ A⁻¹ == I`。

**AI 研究场景:** 计算马氏距离(Mahalanobis distance)需要协方差矩阵的逆;某些解析解(比如小规模岭回归、二阶优化里近似 Hessian 求逆)会直接用到 `inv`。但**大多数"为了解方程"而用 `inv` 的场景都不是最佳实践**——下面第 8 节 `solve` 会讲为什么。

**可运行例子:**
```python
import numpy as np

A = np.array([[4.0, 7.0], [2.0, 6.0]])
A_inv = np.linalg.inv(A)
assert np.allclose(A @ A_inv, np.eye(2))
assert np.allclose(A_inv @ A, np.eye(2))

# 奇异矩阵(行/列线性相关)无法求逆,会抛出异常
singular = np.array([[1.0, 2.0], [2.0, 4.0]])   # 第二行是第一行的2倍
try:
    np.linalg.inv(singular)
    raised = False
except np.linalg.LinAlgError:
    raised = True
assert raised
```

**常见坑:** 矩阵"接近奇异"(病态,condition number 很大)但没有严格奇异到触发报错时,`inv` 不会报错,但结果可能已经严重失真——数值上不可靠,不代表逻辑上有 bug。这也是为什么"矩阵能不能求逆"不能只看有没有报错,该用 `np.linalg.cond` 或下面的 `matrix_rank` 提前检查。

---

## 6. `np.linalg.pinv`

**签名:**
```python
np.linalg.pinv(a)
```
- `a`:**任意形状**的矩阵,不要求是方阵,也不要求可逆

**一句话:** Moore-Penrose 伪逆——`inv` 的推广,对非方阵、不满秩的矩阵也能算出一个"最合理的逆"。

**AI 研究场景:** 线性回归/最小二乘问题 `min ||Ax - b||²` 的解析解是 `x = pinv(A) @ b`——当样本数和特征数不相等(几乎总是这样:样本数远多于特征数,即超定方程组)时,方阵的 `inv` 根本用不了,`pinv` 就是干这个的。这是从"线性代数里的解方程"过渡到"机器学习里的最小二乘拟合"最关键的一座桥。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
A = np.random.randn(5, 2)              # 5 个样本,2 个特征(超定:方程比未知数多)
x_true = np.array([1.0, 2.0])
b = A @ x_true + 0.001 * np.random.randn(5)   # 加一点观测噪声

x_est = np.linalg.pinv(A) @ b
assert np.allclose(x_est, x_true, atol=1e-2)

# 验证:pinv 最小二乘解和官方最小二乘求解器 lstsq 结果一致
x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
assert np.allclose(x_est, x_lstsq)

# 方阵且可逆时,pinv 退化成普通 inv(行为兼容,不冲突)
B = np.array([[4.0, 7.0], [2.0, 6.0]])
assert np.allclose(np.linalg.pinv(B), np.linalg.inv(B))
```

**常见坑:** `pinv` 内部是通过 SVD(第 11 节)算出来的,比直接 `inv` 计算量大得多——如果明确知道矩阵是方阵且可逆,直接用 `inv`(或者解方程用 `solve`)更高效,`pinv` 是"不确定矩阵形状/满秩性"时的通用兜底方案,不是默认首选。

---

## 7. `np.linalg.det`

**签名:**
```python
np.linalg.det(a)
```
- `a`:方阵

**一句话:** 计算行列式——一个标量,`det(A) == 0` 意味着矩阵不可逆(奇异)。

**AI 研究场景:**
- **可逆性检查:** 求逆之前先看 `det` 是否接近 0,提前发现"这个矩阵根本不该求逆"的问题(不过大矩阵直接算 `det` 开销不小,工程上更常用 `matrix_rank` 或 `cond`)。
- **多元高斯分布的归一化常数:** 多元正态分布的概率密度公式里有一项 `1/sqrt(det(Σ))`(`Σ` 是协方差矩阵),贝叶斯方法、高斯过程里经常要算。
- **归一化流(normalizing flows):** 生成模型里"变量替换公式"要求算雅可比矩阵的行列式(准确地说是 `log|det J|`),这是 AI 研究里 `det` 出现频率最高的场景之一。

**可运行例子:**
```python
import numpy as np

A = np.array([[4.0, 7.0], [2.0, 6.0]])
d = np.linalg.det(A)
assert np.allclose(d, 4.0*6.0 - 7.0*2.0)      # 2x2 行列式公式:ad - bc

singular = np.array([[1.0, 2.0], [2.0, 4.0]])
assert np.allclose(np.linalg.det(singular), 0.0, atol=1e-9)   # 行线性相关 -> 行列式为0

# 行列式和特征值的关系:det(A) == 所有特征值的乘积
eigvals, _ = np.linalg.eig(A)
assert np.allclose(np.prod(eigvals), d)
```

**常见坑:** 大矩阵的 `det` 数值上很容易溢出或下溢(行列式随矩阵增大可能是天文数字或趋近于 0),深度学习里如果真的需要算行列式(比如归一化流),几乎总是用 `np.linalg.slogdet` 算 `log|det|` 而不是直接算 `det` 本身,避免数值溢出——这是本文没专门列一节但值得提前知道的兄弟函数。

---

## 8. `np.linalg.solve`

**签名:**
```python
np.linalg.solve(a, b)
```
- `a`:方阵(系数矩阵),`b`:向量或矩阵(右端项)
- 求解线性方程组 `a @ x == b`,返回 `x`

**一句话:** 直接求解线性方程组 `Ax = b`,不显式计算 `A` 的逆。

**AI 研究场景:** 几乎所有"解线性方程组"的场景都该用 `solve` 而不是 `inv(A) @ b`——正规方程法的线性回归解析解、高斯过程回归里求解核矩阵方程、任何显式写出"解一个线性系统"需求的研究代码。

**为什么 `solve(A, b)` 比 `inv(A) @ b` 更好——数值稳定性:**

`inv(A) @ b` 要先精确计算出整个逆矩阵,再做一次矩阵乘法,中间多了一步"求逆"本身的舍入误差,这个误差会被放大传递到最终结果。`solve` 用的是 LU 分解直接对 `b` 做正向/反向回代,少了"显式求逆"这一步误差放大的机会。矩阵条件数(condition number)越大(越接近奇异),这个差距越明显——用经典的病态矩阵 Hilbert 矩阵实测:

**为什么更快——计算量:**

`solve` 只需要对**一组**右端向量做回代;`inv` 相当于要对**单位矩阵的每一列**都做一遍回代(等价于解 n 个方程组),再额外做一次矩阵乘法——多做了好几倍无意义的工作。

**可运行例子:**
```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")        # 固定单线程,避免 BLAS 多线程调度抖动干扰计时
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import time

# --- 数值稳定性:Hilbert 矩阵是线性代数里"病态矩阵"的经典教学例子 ---
n = 10
H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
x_true = np.ones(n)
b = H @ x_true

x_solve = np.linalg.solve(H, b)
x_inv = np.linalg.inv(H) @ b

err_solve = np.linalg.norm(x_solve - x_true)
err_inv = np.linalg.norm(x_inv - x_true)
assert err_solve < err_inv    # solve 误差更小(本机实测:inv 误差约是 solve 的 33 倍)

# --- 性能:同样规模下 solve 应该比 inv+matmul 更快 ---
def time_it(fn, repeat=5, loop=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(loop):
            fn()
        best = min(best, (time.perf_counter() - t0) / loop)
    return best

n2 = 1000
np.random.seed(1)
M = np.random.randn(n2, n2)
M = M @ M.T + n2 * np.eye(n2)     # 良态对称正定矩阵
rhs = np.random.randn(n2)

t_solve = time_it(lambda: np.linalg.solve(M, rhs))
t_inv = time_it(lambda: np.linalg.inv(M) @ rhs)
assert t_inv > t_solve    # 本机实测:inv+matmul 比 solve 慢约 3-4 倍
```
本机实测输出(仅供参考,具体倍数因机器而异):`solve 误差=2.74e-04, inv 误差=9.01e-03`(inv 误差约 33 倍);`solve≈19ms, inv+matmul≈70ms`(inv 约慢 3.6 倍)。

**常见坑:** 只需要解一次方程组时千万别写成 `np.linalg.inv(A) @ b`——这是从数学课本"先求逆再乘"的思维定式直接搬进代码的典型坑。**唯一需要显式算出 `inv(A)` 的场景,是你要用这个逆矩阵本身参与后续多次不同的计算**(比如同一个逆要在循环里反复乘不同向量、且量级很大到值得预先分解),否则 `solve` 永远是更好的默认选择。

---

## 9. `np.linalg.eig`

**签名:**
```python
np.linalg.eig(a)
```
- `a`:方阵(不要求对称)
- 返回 `(eigenvalues, eigenvectors)`——`eigenvectors` 每一**列**是一个特征向量,和 `eigenvalues` 按位置一一对应

**一句话:** 通用特征值分解,满足 `A @ v == lambda * v`。

**AI 研究场景:** 分析 RNN/线性动力系统权重矩阵的**谱半径**(最大特征值的绝对值)——谱半径大于 1,反复相乘后数值倾向于爆炸;小于 1 倾向于消失,这是理解 RNN 梯度爆炸/消失问题的经典线性代数直觉(虽然真实 RNN 有非线性激活,不能直接照搬结论,但谱半径分析是理解问题的第一步)。

**可运行例子:**
```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
eigvals, eigvecs = np.linalg.eig(A)

# 验证定义: A @ v == lambda * v,对每一列(每个特征向量)分别验证
for i in range(2):
    v = eigvecs[:, i]
    assert np.allclose(A @ v, eigvals[i] * v)

# 实数矩阵也可能有复数特征值——90度旋转矩阵就是经典例子(没有实数方向不改变)
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
r_eigvals, _ = np.linalg.eig(R)
assert np.iscomplexobj(r_eigvals)
assert np.allclose(np.abs(r_eigvals), 1.0)     # 旋转矩阵特征值模长恒为1(实测约为 ±i)

# 谱半径:RNN 权重矩阵反复相乘是否会数值爆炸的直觉指标
np.random.seed(0)
W_rnn = np.random.randn(4, 4) * 0.5
spectral_radius = np.max(np.abs(np.linalg.eig(W_rnn)[0]))
assert spectral_radius > 0
```

**常见坑:** **`eig` 不保证特征值按任何顺序排列**——用一个对角矩阵 `diag([5,1,3])` 测试就能验证(特征值显然就是对角线的 `[5,1,3]`),本机实测返回顺序恰好是 `[5,1,3]`(即按对角线出现顺序,但这是实现细节,不是 API 保证,换个矩阵结构或 numpy 版本可能不成立)。**永远不要假设 `eig` 返回的第一个特征值是最大的**——需要"最大特征值"就显式 `np.argmax(np.abs(eigvals))`,需要排序就自己 `np.argsort`。另外它对一般矩阵可能返回复数结果(哪怕输入是纯实数矩阵),下游代码如果没预期到复数类型容易报错或得到诡异结果。

---

## 10. `np.linalg.eigh`

**签名:**
```python
np.linalg.eigh(a)
```
- `a`:**对称矩阵**(实数)或**厄米矩阵**(复数,`a == a.conj().T`)
- 返回 `(eigenvalues, eigenvectors)`,和 `eig` 格式一样,但有额外保证

**一句话:** `eig` 的对称矩阵专用版——利用"对称"这个数学性质,换了一个更快、更稳定的算法,而且有 `eig` 没有的额外保证。

**`eigh` 相比 `eig` 多的保证(为什么 PCA/协方差矩阵分析都用它):**

| 保证 | `eig`(通用) | `eigh`(对称专用) |
|---|---|---|
| 特征值一定是实数 | 不保证(可能返回复数) | **保证**(对称矩阵的数学性质) |
| 排序 | 不保证任何顺序 | **保证升序排列** |
| 特征向量正交 | 不保证 | **保证**(标准正交基) |
| 速度/数值稳定性 | 通用算法,较慢 | 专用算法,更快更稳 |

**AI 研究场景:** PCA 的核心步骤是对**协方差矩阵**(必然对称)做特征分解,取最大的几个特征值对应的特征向量作为主成分方向——协方差矩阵天然对称,用 `eigh` 而不是 `eig` 是标准做法。同理:图神经网络里对**拉普拉斯矩阵**(对称)做谱聚类、优化中分析 **Hessian 矩阵**(对称)的曲率,都是 `eigh` 的场景。

**可运行例子:**
```python
import numpy as np

S = np.array([[4.0, 1.0, 0.0],
              [1.0, 3.0, 1.0],
              [0.0, 1.0, 2.0]])    # 对称矩阵

eigvals, eigvecs = np.linalg.eigh(S)

assert np.isrealobj(eigvals)                       # 保证实数
assert np.all(np.diff(eigvals) >= 0)               # 保证升序:实测 [1.268, 3.0, 4.732]
assert np.allclose(eigvecs.T @ eigvecs, np.eye(3)) # 保证特征向量正交(标准正交基)

for i in range(3):
    assert np.allclose(S @ eigvecs[:, i], eigvals[i] * eigvecs[:, i])

# PCA 套路:协方差矩阵特征分解后,升序排列意味着"最大主成分"在最后,取最后 k 个
k = 2
top_k_directions = eigvecs[:, -k:]        # 取最后两列(最大的两个特征值对应的方向)
assert top_k_directions.shape == (3, k)

# 对称矩阵上 eig 和 eigh 的特征值集合应一致(只是顺序可能不同)
eigvals_general, _ = np.linalg.eig(S)
assert np.allclose(sorted(eigvals_general.real), sorted(eigvals))
```

**常见坑:** 对一个"其实是对称"的矩阵误用 `eig` 不会报错,但会更慢、且可能因为浮点误差返回带极小虚部的"伪复数"结果(比如 `3.0+1e-16j`),下游代码做实数运算时莫名其妙报类型错误。反过来,**如果矩阵其实不对称却传给 `eigh`,它会默默只读取矩阵的下三角(或上三角)部分,当成对称矩阵处理,不会报错也不会提醒你**——这是最容易被忽视的坑:`eigh` 假设你保证了输入对称,它不做检查。

---

## 11. `np.linalg.svd`

**签名:**
```python
np.linalg.svd(a, full_matrices=True)
```
- `a`:任意形状矩阵(不要求方阵、不要求对称)
- 返回 `U, S, Vt`,满足 `a == U @ diag(S) @ Vt`(`S` 是一维数组,存奇异值,已按降序排列)
- `full_matrices=False`:"经济型" SVD,`U`/`Vt` 只保留必要的列/行,更省内存,更常用

**一句话:** 奇异值分解——把任意矩阵拆成"旋转 × 缩放 × 旋转"三部分,是线性代数里最通用、应用最广的分解,没有"矩阵必须方阵/对称"这些限制。

**AI 研究场景(SVD 几乎是"最重要的线性代数工具"):**
- **PCA 的另一种实现路径:** 对**中心化后的数据矩阵**直接做 SVD,不需要先显式算协方差矩阵(算协方差矩阵是 `X.T @ X`,会放大数值误差,`svd` 直接对数据做分解更稳定),`Vt` 的行就是主成分方向。
- **低秩近似/压缩:** 只保留最大的 k 个奇异值/对应的向量,就得到原矩阵的"最优 k 秩近似"(Eckart-Young 定理)——推荐系统里用户-物品评分矩阵的矩阵分解(matrix factorization)本质就是截断 SVD。
- **LoRA 类微调方法:** 大模型微调时,LoRA 假设"权重更新矩阵是低秩的",用两个小矩阵的乘积去近似一个大的更新矩阵——这个近似的最优解正是截断 SVD 给出的。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
A = np.random.randn(4, 3)

# 默认 full_matrices=True: U 是方阵(4,4),Vt 是方阵(3,3)
U, S, Vt = np.linalg.svd(A)
assert U.shape == (4, 4) and Vt.shape == (3, 3) and S.shape == (3,)
assert np.all(np.diff(S) <= 0)     # 奇异值降序排列

# 经济型 SVD(full_matrices=False):更常用,U 直接是 (4,3)
U2, S2, Vt2 = np.linalg.svd(A, full_matrices=False)
assert U2.shape == (4, 3)
A_reconstructed = U2 @ np.diag(S2) @ Vt2
assert np.allclose(A, A_reconstructed)

# --- 低秩近似:构造一个"本质秩为2"的评分矩阵(5用户 x 6物品,2维隐向量生成) ---
np.random.seed(1)
user_factors = np.random.randn(5, 2)
item_factors = np.random.randn(6, 2)
ratings = user_factors @ item_factors.T          # 理论秩为 2

Ur, Sr, Vtr = np.linalg.svd(ratings, full_matrices=False)
assert np.allclose(Sr[2:], 0.0, atol=1e-10)      # 第3个及以后的奇异值应接近0

# 只用前2个奇异值/向量重建,应该完美还原(因为矩阵本来就是秩2)
approx = Ur[:, :2] @ np.diag(Sr[:2]) @ Vtr[:2, :]
assert np.allclose(ratings, approx)

# 加噪声后,保留更多奇异值 -> 重建误差更小(奇异值排序的实际意义)
noisy = ratings + 0.01 * np.random.randn(5, 6)
Un, Sn, Vtn = np.linalg.svd(noisy, full_matrices=False)
err_k1 = np.linalg.norm(noisy - Un[:, :1] @ np.diag(Sn[:1]) @ Vtn[:1, :])
err_k2 = np.linalg.norm(noisy - Un[:, :2] @ np.diag(Sn[:2]) @ Vtn[:2, :])
assert err_k2 < err_k1
```

**常见坑:** `full_matrices=True`(默认)返回的 `U`/`Vt` 是完整方阵,想重建 `A` 时不能直接 `U @ diag(S) @ Vt`——`S` 只有 `min(m,n)` 个元素,和完整方阵形状对不上,必须先手动把 `S` 摆成正确形状的对角矩阵(补 0),或者干脆直接传 `full_matrices=False` 拿"经济型"的版本更省心。另外,**一个随机矩阵通常不是低秩的**——上面 LoRA 的近似效果好坏取决于"真实权重更新是否真的接近低秩",这是个经验假设,不是数学保证,对纯随机噪声矩阵做低秩截断只会引入很大误差。

---

## 12. `np.linalg.qr`

**签名:**
```python
np.linalg.qr(a)
```
- `a`:任意形状矩阵(常见于 `m >= n` 的"高瘦"矩阵)
- 返回 `Q, R`,满足 `a == Q @ R`,`Q` 的列是标准正交基,`R` 是上三角矩阵

**一句话:** QR 分解——把矩阵拆成"正交矩阵 × 上三角矩阵",实现上等价于对矩阵的列做 Gram-Schmidt 正交化。

**AI 研究场景:** **正交初始化(orthogonal initialization)**——RNN 类模型的循环权重矩阵如果初始化成正交矩阵,由于正交变换不改变向量长度(`||Qv|| == ||v||`),多个时间步反复相乘时不会像随机矩阵那样容易导致数值爆炸或消失,这是缓解 RNN 长序列梯度问题的经典技巧:先随机生成一个矩阵,取它 QR 分解的 `Q` 作为初始权重。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
A = np.random.randn(5, 3)
Q, R = np.linalg.qr(A)

assert Q.shape == (5, 3) and R.shape == (3, 3)      # 默认经济型分解
assert np.allclose(A, Q @ R)
assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)  # Q 的列标准正交
assert np.allclose(R, np.triu(R))                    # R 是上三角矩阵

# AI 场景:正交初始化,验证"正交矩阵不改变向量长度"
n = 4
random_mat = np.random.randn(n, n)
Q_ortho, _ = np.linalg.qr(random_mat)
assert np.allclose(Q_ortho.T @ Q_ortho, np.eye(n), atol=1e-10)

v = np.random.randn(n)
assert np.allclose(np.linalg.norm(Q_ortho @ v), np.linalg.norm(v))   # 长度不变
```

**常见坑:** 默认的"经济型"分解(`Q` 只有 `min(m,n)` 列)对"高瘦"矩阵(行多列少)才有意义;如果 `a` 是"矮胖"矩阵(列多行少),`Q` 不会是方阵,正交初始化这类需要"方阵、完整基"的场景要确保输入矩阵形状是方阵(比如 `n x n` 的随机矩阵),否则拿到的 `Q` 不满足预期的正交方阵性质。

---

## 13. `np.linalg.cholesky`

**签名:**
```python
np.linalg.cholesky(a)
```
- `a`:**对称正定(SPD)矩阵**——不满足条件会抛 `LinAlgError`
- 返回下三角矩阵 `L`,满足 `a == L @ L.T`

**一句话:** 专门给"对称正定矩阵"用的分解,是 `eigh`/`svd` 之外第三种利用矩阵特殊结构换取速度的分解方式——协方差矩阵、核矩阵这类"天生正定"的矩阵,`cholesky` 通常是最快的选择。

**AI 研究场景:**
- **采样多元正态分布:** 已知均值 `mu` 和协方差 `Sigma`,想采样服从 `N(mu, Sigma)` 的样本,标准做法是先对独立标准正态 `z ~ N(0, I)` 采样,再算 `x = mu + L @ z`(`L` 是 `Sigma` 的 Cholesky 分解)——这是贝叶斯方法、生成模型里从任意协方差的高斯分布采样的通用技巧。
- **高斯过程回归(Gaussian Process Regression):** 求解 `(K + sigma^2 I) alpha = y` 是 GP 回归的核心步骤,`K` 是核矩阵(天然对称正定),标准做法是先 `cholesky` 分解,再用分解结果求解,比直接 `solve` 或 `inv` 更快也更数值稳定。

**可运行例子:**
```python
import numpy as np

np.random.seed(0)
M = np.random.randn(3, 3)
Sigma = M @ M.T + 3 * np.eye(3)      # 构造一个对称正定矩阵(协方差矩阵的典型构造方式)

L = np.linalg.cholesky(Sigma)
assert np.allclose(L @ L.T, Sigma)
assert np.allclose(L, np.tril(L))    # L 是下三角矩阵

# 非对称正定矩阵会报错
not_spd = np.array([[1.0, 2.0], [3.0, 1.0]])
try:
    np.linalg.cholesky(not_spd)
    raised = False
except np.linalg.LinAlgError:
    raised = True
assert raised

# AI 场景:采样多元正态分布 N(mu, Sigma)
mu = np.array([1.0, 2.0, 3.0])
np.random.seed(42)
z = np.random.randn(200000, 3)              # 独立标准正态采样
samples = mu + z @ L.T                       # 等价于对每个样本算 mu + L @ z_i

assert np.allclose(samples.mean(axis=0), mu, atol=0.05)
assert np.allclose(np.cov(samples.T), Sigma, atol=0.1)

# AI 场景:用 Cholesky 分解求解正定线性系统(先解 Lw=y,再解 L^T alpha=w)
y = np.array([0.5, -0.2, 1.0])
alpha_direct = np.linalg.solve(Sigma, y)
alpha_via_L = np.linalg.solve(L.T, np.linalg.solve(L, y))
assert np.allclose(alpha_direct, alpha_via_L)
```

**常见坑:** `cholesky` 对"正定"的要求很严格——半正定(特征值里有恰好等于 0 的情况,比如某些退化的协方差矩阵)也会报错,这时候需要退化到 `eigh`(检查特征值是否 ≥ 0)或者在对角线上加一个小的扰动项(`Sigma + eps * I`,工程上非常常见的"稳定化"技巧,GP 回归代码里几乎总能看到这一行)。另外容易搞混方向:是 `L @ L.T`(下三角乘自己的转置),不是 `L.T @ L`。

---

## 14. `np.trace`

**签名:**
```python
np.trace(a)
```
- `a`:方阵

**一句话:** 矩阵对角线元素之和。

**AI 研究场景:**
- **Frobenius 范数的等价写法:** `||A||_F^2 == trace(A.T @ A)`——某些推导里(尤其是矩阵微积分求梯度时)用 `trace` 形式更方便做代数变换,这是论文推导里常见的写法转换。
- **KL 散度公式里的一项:** 两个多元高斯分布之间的 KL 散度公式包含 `trace(Σ2⁻¹ Σ1)` 这一项(VAE 等生成模型推导 loss 时会遇到)。
- **验证恒等式:** 矩阵的迹等于它所有特征值之和,这是检验特征分解代码是否正确的一个快速数值校验手段。

**可运行例子:**
```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
assert np.allclose(np.trace(A), 1.0 + 4.0)

# Frobenius 范数平方 == trace(A^T A)
assert np.allclose(np.linalg.norm(A)**2, np.trace(A.T @ A))

# 迹 == 特征值之和(对称矩阵用 eigh 验证更稳)
S = np.array([[4.0, 1.0], [1.0, 3.0]])
eigvals, _ = np.linalg.eigh(S)
assert np.allclose(np.trace(S), np.sum(eigvals))
```

**常见坑:** `np.trace` 只看**方阵**的主对角线;对非方阵调用不会报错(numpy 会取 `min(m,n)` 长度的对角线求和),这个"静默兼容"的行为容易掩盖"我传错矩阵形状了"的 bug,不确定时先 assert 一下 `a.shape[0] == a.shape[1]`。

---

## 15. `np.einsum`

**签名:**
```python
np.einsum(subscripts, *operands)
```
- `subscripts`:一个字符串,比如 `'bqd,bkd->bqk'`,逗号分隔每个输入数组的下标,`->` 后面是输出的下标
- `*operands`:参与运算的数组,数量要和 `subscripts` 里逗号分隔的段数一致

**一句话:** 用"爱因斯坦求和约定"的下标记法,一行表达式描述任意复杂的张量收缩/求和运算——矩阵乘法、批量矩阵乘法、转置、迹、attention score 全都是它的特殊情形。**这是 AI 研究代码里控制张量运算最灵活的工具,没有之一。**

**怎么读下标记法(核心规则,记住这三条就能读懂 90% 的 einsum):**
1. 每个输入数组的下标字母个数 = 该数组的维度数,字母和维度按顺序一一对应。
2. **同一个字母在多个输入里重复出现,但没出现在输出里** → 这个维度被"收缩"(对应位置逐元素相乘后再求和,矩阵乘法的本质)。
3. **同一个字母在输入和输出里都出现** → 这个维度被保留,且如果它在多个输入里同时出现,意味着"按这个维度对齐"(不是笛卡尔积)——这正是第 2 节里 `matmul` 对 batch 维的处理方式,`einsum` 能显式表达出来。

**AI 研究场景:** Transformer 的 attention 机制、多头注意力的分拆与合并、任何"输入形状里维度一多,普通 `matmul`/`transpose` 组合就开始搭积木搭得很痛苦"的场景——直接用 `einsum` 显式写下标,比拼接一堆 `transpose`+`reshape`+`matmul` 更不容易出错,也更接近论文公式本身的写法。

**可运行例子(3 个由浅入深的例子):**

```python
import numpy as np

np.random.seed(0)

# --- 例子 1: 批量矩阵乘法,和 matmul 完全等价 ---
# 'bij,bjk->bik' 读法: 输入1下标(batch,行,列)+ 输入2下标(batch,行,列)
#                       j 在输入里重复但不在输出里 -> 求和收缩(矩阵乘法的"内积"部分)
#                       b 同时在两个输入和输出里 -> 按 batch 对齐,不做笛卡尔积
A3 = np.random.randn(3, 4, 5)
B3 = np.random.randn(3, 5, 6)
r1 = np.einsum('bij,bjk->bik', A3, B3)
assert np.allclose(r1, np.matmul(A3, B3))

# --- 例子 2: attention score(单头)---
# q: (batch, seq_q, d)   k: (batch, seq_k, d)
# 'bqd,bkd->bqk': d 被求和收缩(query 和 key 逐维相乘再加总 = 点积)
#                 b 对齐(不同 batch 互不干扰),q 和 k 各自保留 -> 输出是"每个 query 对每个 key 的得分"
batch, seq_q, seq_k, d = 2, 3, 4, 8
q = np.random.randn(batch, seq_q, d)
k = np.random.randn(batch, seq_k, d)
scores = np.einsum('bqd,bkd->bqk', q, k)
assert scores.shape == (batch, seq_q, seq_k)
assert np.allclose(scores, q @ k.transpose(0, 2, 1))    # 等价于批量 q @ k^T

# --- 例子 3: 多头注意力完整流程(比单头多一个 heads 维,q/k/v 都要走一遍)---
heads, d_head = 4, 8
q4 = np.random.randn(batch, heads, seq_q, d_head)
k4 = np.random.randn(batch, heads, seq_k, d_head)
v4 = np.random.randn(batch, heads, seq_k, d_head)

# 第一步: 'bhqd,bhkd->bhqk' —— b 和 h 都对齐保留,d 收缩,q/k 保留成输出的两个维度
attn_scores = np.einsum('bhqd,bhkd->bhqk', q4, k4) / np.sqrt(d_head)
assert attn_scores.shape == (batch, heads, seq_q, seq_k)

exp_s = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
attn_weights = exp_s / exp_s.sum(axis=-1, keepdims=True)     # 对 seq_k 维做 softmax

# 第二步: 'bhqk,bhkd->bhqd' —— k(seq_k)被收缩掉(加权求和),换回 d_head 维
out = np.einsum('bhqk,bhkd->bhqd', attn_weights, v4)
assert out.shape == (batch, heads, seq_q, d_head)

# 附:einsum 表达"迹"和"转置"(帮助理解下标规则本身)
M = np.random.randn(4, 4)
assert np.allclose(np.einsum('ii->', M), np.trace(M))     # 重复下标+不出现在输出 -> 对角线求和
assert np.allclose(np.einsum('ij->ji', M), M.T)            # 输出下标顺序颠倒 -> 转置
```

**常见坑:** 下标字符串是纯字符串,**手滑打错一个字母 numpy 不一定会报错**,只会给你一个形状凑巧对得上、但数值完全错误的结果——尤其是 batch 维和被求和维用了容易混淆的相邻字母(比如 `q`/`k` 长得像但含义不同)时最容易出错。**养成习惯:写完 `einsum` 立刻用形状 assert 校验,能的话再用一个"笨办法"(比如 `matmul`+`transpose` 组合,或者手写循环)交叉验证数值,不要只凭感觉相信下标写对了。**

---

## 16. `np.kron`

**签名:**
```python
np.kron(a, b)
```
- `a`, `b`:任意形状数组

**一句话:** Kronecker 积——把 `a` 的每个元素替换成"该元素 × 整个 `b`"这一整块,拼成一个大矩阵。`a.shape=(m,n)`、`b.shape=(p,q)` 时,结果形状是 `(m*p, n*q)`。

**AI 研究场景:**
- **最近邻上采样(nearest-neighbor upsampling):** 把一个低分辨率特征图的每个像素"复制"成一个 `k×k` 的小方块,等价于和一个全 1 矩阵做 Kronecker 积——`np.kron(feature_map, np.ones((2,2)))` 就是最直接的 2 倍上采样。
- **K-FAC 等二阶优化方法:** 训练大模型时用二阶信息(近似 Hessian/Fisher 信息矩阵)能加速收敛,但完整二阶矩阵太大存不下、算不动——K-FAC 类方法的核心技巧就是用两个小矩阵的 Kronecker 积去近似这个大矩阵,因为 `(A⊗B)⁻¹ == A⁻¹⊗B⁻¹`,只需要分别求两个小矩阵的逆,不用直接对大矩阵求逆。

**可运行例子:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
K = np.kron(A, B)
assert K.shape == (4, 4)

# 最近邻上采样:每个像素放大成 2x2 的块
feature_map = np.array([[1, 2], [3, 4]])
upsampled = np.kron(feature_map, np.ones((2, 2), dtype=int))
assert upsampled.shape == (4, 4)
assert np.array_equal(upsampled, [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])

# K-FAC 类方法的数学基础:(A kron B)^-1 == A^-1 kron B^-1,分别求逆比整体求逆划算
A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
B2 = np.array([[3.0, 0.0], [0.0, 4.0]])
assert np.allclose(np.linalg.inv(np.kron(A2, B2)), np.kron(np.linalg.inv(A2), np.linalg.inv(B2)))
```

**常见坑:** Kronecker 积的结果大小是两个输入尺寸的**乘积**,`(100,100)` 和 `(100,100)` 的 Kronecker 积是 `(10000,10000)`——增长非常快,真实场景里几乎不会对大矩阵直接做 `kron` 再处理,而是像 K-FAC 那样**利用数学恒等式,避免真正生成那个大矩阵**,只在小矩阵上操作。

---

## 17. `np.linalg.matrix_rank`

**签名:**
```python
np.linalg.matrix_rank(a, tol=None)
```
- `a`:任意形状矩阵
- `tol`:数值容差,不写的话用一个基于矩阵尺寸和浮点精度自动算出的默认阈值

**一句话:** 计算矩阵的(数值)秩——底层是对 `a` 做 SVD,数一下"明显大于 0"的奇异值有几个。

**AI 研究场景:**
- **检查特征多重共线性:** 回归分析前检查特征矩阵是否满秩,如果不满秩,说明有特征是其他特征的线性组合,某些解析解(比如没加正则化的正规方程)会因为矩阵不可逆而失败。
- **验证 LoRA/低秩假设:** 微调产生的权重更新矩阵是否真的接近低秩,决定了低秩近似(第 11 节 SVD)的效果好不好,`matrix_rank` 是量化"有多低秩"的直接工具。

**可运行例子:**
```python
import numpy as np

full_rank = np.eye(2)
assert np.linalg.matrix_rank(full_rank) == 2

rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0]])   # 第二行是第一行的2倍
assert np.linalg.matrix_rank(rank_deficient) == 1

# matrix_rank 本质是"数有多少个奇异值大于阈值"
np.random.seed(0)
M = np.random.randn(5, 3) @ np.random.randn(3, 6)      # 中间维度是3,矩阵秩最多为3
assert np.linalg.matrix_rank(M) == 3

# --- 常见坑实测:默认阈值极严格,真实噪声很容易让"看起来低秩"的矩阵被判成满秩 ---
np.random.seed(2)
low_rank_update = np.random.randn(64, 4) @ np.random.randn(4, 64)   # 真实 rank=4

tiny_noise = low_rank_update + 1e-14 * np.random.randn(64, 64)      # 噪声远小于机器精度阈值
assert np.linalg.matrix_rank(tiny_noise) == 4                       # 符合预期

realistic_noise = low_rank_update + 1e-8 * np.random.randn(64, 64)  # 更"真实"的数值噪声量级
assert np.linalg.matrix_rank(realistic_noise) == 64                 # 默认 tol 下被误判成满秩!

assert np.linalg.matrix_rank(realistic_noise, tol=1e-4) == 4        # 手动给合理 tol 才能纠正回来
```

**常见坑:** 默认的 `tol` 正比于矩阵最大奇异值和机器精度(`~1e-16` 量级)的乘积,非常严格——真实数据里哪怕只有 `1e-8` 量级的数值噪声(浮点运算的正常误差水平,并不算大),也可能让一个"本质上明显低秩"的矩阵被默认阈值判定成满秩(上面例子里从秩 4 判成了 64)。**做"这个矩阵是不是低秩"这类判断时,不要无脑相信默认 `tol`,结合实际数据的噪声量级手动传一个合理的 `tol`**,否则结论可能完全是错的。

---

## 小结:这一批 17 个函数解决的问题

| 函数 | 解决的问题 |
|---|---|
| `dot` | 通用点乘(1D/2D 场景足够用,高维小心) |
| `@`/`matmul` | **批量矩阵乘法**——高维场景的正确选择,和 `dot` 语义不同 |
| `outer`/`inner` | 外积(反传梯度公式)/ 内积推广(批量相似度矩阵) |
| `linalg.norm` | L1/L2/inf 范数——正则化、梯度裁剪、归一化 |
| `linalg.inv` | 矩阵求逆(单纯为了解方程时不是最优选择) |
| `linalg.pinv` | 伪逆——非方阵/不满秩场景的最小二乘解 |
| `linalg.det` | 行列式——可逆性检查、高斯分布归一化常数、归一化流 |
| `linalg.solve` | **解方程组的正确姿势**——比 inv+乘 更快更稳 |
| `linalg.eig` | 通用特征分解(不保证排序,可能返回复数) |
| `linalg.eigh` | 对称矩阵专用特征分解——PCA/协方差矩阵的标准选择 |
| `linalg.svd` | 奇异值分解——PCA/降维/低秩近似/LoRA 的数学基础 |
| `linalg.qr` | QR 分解——正交初始化、正交化 |
| `linalg.cholesky` | 正定矩阵专用分解——采样多元高斯、GP 回归求解 |
| `trace` | 对角线之和——Frobenius 范数、KL 散度公式里的常见项 |
| `einsum` | **最灵活的张量运算工具**——批量矩阵乘法、attention score 的标准写法 |
| `kron` | Kronecker 积——上采样、K-FAC 类二阶优化的数学基础 |
| `linalg.matrix_rank` | 数值秩——多重共线性检查、低秩假设验证 |

下一批:[07-sorting-and-set-ops.md](07-sorting-and-set-ops.md)

---

*更新:2026-07-07*
