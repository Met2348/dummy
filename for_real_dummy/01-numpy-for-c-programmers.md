# NumPy 入门 —— 写给 C 程序员

> 你会 C，这是优势。本文用 C 的视角解释 numpy 的每个概念。
> 目标：能看懂仓库里所有用到 numpy 的代码。

---

## 0. numpy 是什么？

**一句话：** numpy 的 `ndarray` = C 的多维数组 + 自动管理内存 + 批量运算 API。

```c
// C：你自己管内存
float mat[3][4];
mat[1][2] = 5.0f;
```

```python
import numpy as np
mat = np.zeros((3, 4))   # 自动分配，自动释放
mat[1, 2] = 5.0
```

底层实现：numpy 的数组数据就是一块连续内存，和 C 数组一样。
Python 只是给你一个更方便的接口。

---

## 1. 创建数组

```python
import numpy as np

# 从列表创建（最常见）
a = np.array([1, 2, 3, 4])          # 1D，shape=(4,)
b = np.array([[1, 2], [3, 4]])       # 2D，shape=(2,2)

# 特殊数组
np.zeros((3, 4))      # 全 0，shape=(3,4)   ← C: memset(mat, 0, ...)
np.ones((2, 3))       # 全 1
np.eye(4)             # 4x4 单位矩阵
np.arange(0, 10, 2)   # [0,2,4,6,8]        ← C: for(i=0;i<10;i+=2)
np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
```

---

## 2. shape 和 dtype —— 最重要的两个属性

```python
a = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

print(a.shape)   # (2, 3)   → 2行3列
print(a.dtype)   # float64  → 类似 C 的 double
print(a.ndim)    # 2        → 几维
print(a.size)    # 6        → 总元素个数
```

**C 类比：**

| numpy | C |
|-------|---|
| `a.shape = (2, 3)` | `int a[2][3]` |
| `a.dtype = float32` | `float a[]` |
| `a.dtype = float64` | `double a[]` |
| `a.dtype = int32` | `int a[]` |

ML 代码里最常见的类型：
- `float32`（速度快，精度够）
- `float16`（更快，用于 GPU）
- `int64`（存 token 索引）

---

## 3. 索引和切片

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 单个元素（和 C 一样）
a[1, 2]        # → 6     ← C: a[1][2]

# 切片（Python 特有，但很重要）
a[0, :]        # → [1, 2, 3]   第0行，所有列
a[:, 1]        # → [2, 5, 8]   所有行，第1列
a[1:3, 0:2]    # → [[4,5],[7,8]]  子矩阵

# 负索引（从末尾数）
a[-1, :]       # → [7, 8, 9]   最后一行
a[:, -1]       # → [3, 6, 9]   最后一列
```

**仓库里常见的切片模式：**

```python
# 取 batch 的前 N 个
outputs = model_outputs[:N]

# 取序列的所有 token，除了最后一个（用于预测下一个词）
input_ids = tokens[:, :-1]
labels    = tokens[:, 1:]
```

---

## 4. 向量化操作 —— 最重要的思维转变

C 程序员的本能是写循环：

```c
// C：逐元素加法
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
```

numpy 的方式：直接对数组操作，**不写循环**：

```python
# numpy：一行搞定，而且底层是 C 实现的，比你写的循环快 100 倍
c = a + b
```

**所有基本运算都是逐元素的：**

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

a + b       # [5., 7., 9.]
a * b       # [4., 10., 18.]
a ** 2      # [1., 4., 9.]
np.sqrt(a)  # [1., 1.41, 1.73]
np.exp(a)   # [e^1, e^2, e^3]
```

---

## 5. 广播（Broadcasting）—— 没有 C 类比，要专门学

广播是 numpy 最神奇的特性：**形状不同的数组可以直接运算**。

```python
# 矩阵每行加同一个向量
mat  = np.ones((3, 4))    # shape: (3, 4)
bias = np.array([1,2,3,4]) # shape: (4,)

result = mat + bias        # shape: (3, 4)
# numpy 自动把 bias "广播" 成 3 行
```

**理解规则（从右对齐，逐维匹配）：**

```
mat:   (3, 4)
bias:     (4,)   ← 右对齐
              ↑ 维度相同，OK
           ↑ bias 这里没有维度，自动扩展成 3
```

**仓库里会见到的常见广播：**

```python
# 给每个 token 加位置编码
tokens     # shape: (batch, seq_len, hidden)
pos_embed  # shape: (       seq_len, hidden)
out = tokens + pos_embed  # 自动广播 batch 维
```

---

## 6. 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法（不是逐元素！）
C = A @ B             # 推荐写法
C = np.matmul(A, B)   # 等价

# 转置
A.T   # [[1,3],[2,4]]

# 向量点积
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
np.dot(u, v)   # 1*4 + 2*5 + 3*6 = 32
```

**为什么 ML 里全是矩阵乘法？**
神经网络的核心操作就是 `output = input @ weight + bias`，
其中 `@` 是矩阵乘法，`weight` 是模型的参数。

---

## 7. 常用统计和形状操作

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 统计
a.sum()          # 21（全部加）
a.sum(axis=0)    # [5, 7, 9]（每列求和）
a.sum(axis=1)    # [6, 15]（每行求和）
a.mean()         # 3.5
a.max()          # 6
a.argmax()       # 5（最大值的索引）

# 形状操作
a.reshape(3, 2)         # 重排成 3x2（总元素数不变）
a.reshape(-1)           # 展平成 1D，-1 表示"自动算"
a.reshape(2, -1)        # 2行，列数自动算

np.concatenate([a, a], axis=0)  # 沿行方向拼接 → shape(4,3)
np.stack([a, a], axis=0)        # 新增一个维度堆叠 → shape(2,2,3)
```

---

## 8. 仓库代码里最常见的 numpy 模式

```python
# 1. 初始化权重（随机）
W = np.random.randn(hidden_size, output_size) * 0.01

# 2. Softmax 实现（概率归一化）
def softmax(x):
    e = np.exp(x - x.max())   # 减最大值防止溢出
    return e / e.sum()

# 3. 逻辑回归（auto-research 里有完整实现）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

pred = sigmoid(X @ W + b)

# 4. 归一化
mean = data.mean(axis=0)
std  = data.std(axis=0)
data_norm = (data - mean) / std
```

---

## 练习：自己动手

在 Python 解释器（或 Jupyter）里完成以下任务：

### 练习 1：创建和操作
```python
import numpy as np
# 创建一个 4x4 的随机矩阵，打印它的 shape 和 dtype
# 取出第 2 行，第 3 列的元素
# 取出前两行组成的子矩阵
```

### 练习 2：向量化思维（关键！）
```python
# 不用 for 循环，用 numpy 计算：
# 给定 scores = np.array([2.0, 1.0, 0.1])
# 计算它的 softmax（先 exp，再除以 sum）
scores = np.array([2.0, 1.0, 0.1])
# 你的代码在这里
```

### 练习 3：矩阵乘法
```python
# 一个简单的线性层：output = input @ W + b
# input shape: (3, 4)  ← batch=3 个样本，每个 4 维
# W shape: (4, 2)      ← 输出 2 维
# b shape: (2,)
# 用 numpy 实现，打印 output 的 shape
```

---

## 下一步

完成本教程后，继续看 [02-pytorch-basics.md](02-pytorch-basics.md)。
PyTorch 的 Tensor 操作和 numpy **几乎完全一样**，只是多了 GPU 支持和自动求导。

---

*更新：2026-07-02*
