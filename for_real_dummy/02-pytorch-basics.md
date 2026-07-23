# PyTorch 基础 —— 从 numpy 到神经网络

> 前提：已看完 [01-numpy-for-c-programmers.md](01-numpy-for-c-programmers.md)
> 目标：能看懂仓库里所有模型定义和训练代码

---

## 0. PyTorch 是什么？

**一句话：** PyTorch = numpy + GPU 支持 + 自动求导（autograd）

```
numpy ndarray
    ↓ 加上 GPU 支持
    ↓ 加上自动计算梯度
PyTorch Tensor
```

你已经会 numpy 了，所以 PyTorch 的大部分操作你已经知道——只是换了名字。

---

## 1. Tensor 基础操作（和 numpy 对比）

```python
import torch
import numpy as np

# numpy 创建
a_np = np.array([[1.0, 2.0], [3.0, 4.0]])

# torch 创建（操作几乎一样！）
a_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 特殊 tensor
torch.zeros(3, 4)       # np.zeros((3,4))
torch.ones(2, 3)        # np.ones((2,3))
torch.randn(3, 4)       # np.random.randn(3,4)
torch.arange(0, 10, 2)  # np.arange(0,10,2)

# 属性
a_pt.shape     # torch.Size([2, 2]) ← 同 np.shape
a_pt.dtype     # torch.float32
a_pt.ndim      # 2

# 操作（和 numpy 一样！）
a_pt + a_pt
a_pt * 2
a_pt @ a_pt       # 矩阵乘法
a_pt.T            # 转置
a_pt.sum()
a_pt.mean(dim=0)  # numpy 里叫 axis，torch 里叫 dim
a_pt.reshape(4)
```

**唯一需要注意的不同：**
- numpy 用 `axis=...`，torch 用 `dim=...`
- 其他操作几乎相同

---

## 2. GPU 支持 —— `.to()` 方法

这是 PyTorch 相比 numpy 最核心的优势：

```python
# 检查 GPU 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 把 tensor 移到 GPU
a = torch.randn(1000, 1000)
a_gpu = a.to('cuda')    # 或 a.to(device)
a_gpu = a.cuda()        # 简写

# 在 GPU 上的运算自动在 GPU 执行（速度 >> CPU）
b_gpu = a_gpu @ a_gpu

# 移回 CPU（比如要打印或转 numpy）
b_cpu = b_gpu.cpu()
```

**为什么要 GPU？**
矩阵乘法 `A @ B` 在 GPU 上比 CPU 快 100-1000 倍。
训练一个大模型，CPU 要几个月，GPU 要几小时。

---

## 3. Autograd —— 自动求导（最重要！）

这是 PyTorch 存在的根本原因。训练神经网络需要计算**梯度**（微积分里的导数），autograd 自动帮你算。

### 3.1 直觉

数学上，如果 `L = (x - 3)²`，那 `dL/dx = 2(x-3)`。
当 `x=5` 时，梯度 = `2*(5-3) = 4`，意思是"x 往正方向走一步，L 增加 4"。

PyTorch 自动帮你算这个。

### 3.2 代码演示

```python
# requires_grad=True 告诉 PyTorch：请追踪这个 tensor 的梯度
x = torch.tensor(5.0, requires_grad=True)

# 正向计算（forward pass）
L = (x - 3) ** 2    # L = 4.0

# 反向传播（backward pass）—— 自动计算所有梯度
L.backward()

# 查看梯度
print(x.grad)   # tensor(4.)   ← 就是 dL/dx = 2*(5-3) = 4
```

### 3.3 为什么重要

神经网络的参数（weights）就是带 `requires_grad=True` 的 tensor。
每次训练：
1. 前向传播：用参数计算输出和 loss
2. 反向传播：`loss.backward()` 自动算每个参数的梯度
3. 更新参数：用梯度调整参数，让 loss 变小

---

## 4. nn.Module —— 神经网络的积木

仓库里几乎所有模型都是这个模式：继承 `nn.Module`，实现 `forward()`。

### 4.1 最小示例

```python
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()                              # 固定写法，必须调用父类
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        # nn.Parameter 就是"这个 tensor 是模型参数，需要训练"

    def forward(self, x):
        return x @ self.weight + self.bias
        # forward 定义：给定输入 x，怎么计算输出

# 使用
layer = LinearLayer(4, 2)
x = torch.randn(3, 4)   # batch=3，每个 4 维
out = layer(x)           # 调用 layer(x) 会自动执行 forward(x)
print(out.shape)         # torch.Size([3, 2])
```

**为什么必须用 `nn.Parameter` 包一层，不能直接写 `torch.randn(..., requires_grad=True)`？**
两者在"能不能算梯度"这件事上没有区别——区别在于"模型认不认识它是自己的参数"。`nn.Module.__init__()`（靠 `super().__init__()` 触发）内部会维护一份"这个模型都有哪些参数"的登记表；只有被 `nn.Parameter` 包过的 tensor，赋值给 `self.xxx` 时才会自动登记进这张表，而 `model.parameters()`（`torch.optim.Adam(model.parameters(), ...)` 依赖的正是这个）只会遍历这张登记表，不会去扫描每个属性判断"这个是不是一个会算梯度的 tensor"。如果偷懒直接写 `self.weight = torch.randn(4, 2, requires_grad=True)`，这个 tensor 依然会正确参与反向传播、算出梯度（`requires_grad`/`.backward()` 本身不关心你有没有用 `nn.Parameter` 包装），但它**不会出现在 `model.parameters()` 里**——优化器压根不知道有这个参数存在，`optimizer.step()` 永远不会更新它，而且不会有任何报错或警告，训练能正常跑、loss 甚至可能还在下降（其它参数在正常更新），只有这一个参数在"陪跑"，是非常隐蔽的坑。记住一条规矩就够了：**模型自己的可训练参数，只要是直接赋给 `self.xxx` 的 tensor，一律用 `nn.Parameter` 包起来。**

**C 类比：**
`nn.Module` 就像 C 里的 struct + 函数指针：
- 成员变量（`weight`、`bias`）= struct 里的字段
- `forward()` = 函数指针，定义"这个结构体怎么处理数据"

### 4.2 PyTorch 内置层（仓库里会大量看到）

```python
nn.Linear(4, 2)           # 全连接层：y = xW + b
nn.ReLU()                 # 激活函数：max(0, x)
nn.Dropout(p=0.1)         # 随机丢弃 10% 的神经元（防过拟合）
nn.LayerNorm(hidden_size) # 归一化层（Transformer 核心组件）
nn.Embedding(vocab, dim)  # 词向量表：把 token ID 映射到向量
nn.MultiheadAttention(...)# 注意力机制（Transformer 核心）
```

### 4.3 用 `nn.Sequential` 串联多层

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
# 等价于手写一个 Module，forward 依次经过这三层
```

---

## 5. 完整训练循环 —— 必须背下来的模板

这是 ML 代码的核心骨架，仓库里所有训练代码都是这个模式的变体：

```python
import torch
import torch.nn as nn

# --- 1. 定义模型 ---
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# --- 2. 定义优化器和损失函数 ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()   # 均方误差，回归问题常用

# --- 3. 训练循环 ---
for epoch in range(100):

    # 3a. 前向传播
    pred = model(X_train)          # 用模型预测
    loss = loss_fn(pred, y_train)  # 计算 loss

    # 3b. 反向传播
    optimizer.zero_grad()   # 清空上一步的梯度（必须！否则会累积）
    loss.backward()         # 计算所有参数的梯度
    optimizer.step()        # 根据梯度更新参数

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**三步缺一不可：**
1. `zero_grad()` — 清梯度
2. `backward()` — 算梯度
3. `step()` — 更新参数

---

## 6. 仓库代码里的常见模式

### 6.1 Adapter（仓库里大量使用）

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)  # 压缩
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, hidden_size)  # 还原

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # 残差连接
```

这是 `learning/adapter-tuning-family/` 里所有代码的核心模式。

### 6.2 加载预训练模型

```python
from transformers import AutoModel

# 仓库里大量用 HuggingFace 的 transformers 库
model = AutoModel.from_pretrained("bert-base-uncased")
```

### 6.3 冻结参数（只训练部分层）

```python
# 冻结所有参数（不更新）
for param in model.parameters():
    param.requires_grad = False

# 只解冻 adapter 的参数
for param in model.adapter.parameters():
    param.requires_grad = True
```

---

## 7. 练习题

### 练习 1：tensor 操作
```python
import torch
# 创建两个 shape=(3,4) 的随机 tensor
# 做矩阵乘法，打印结果的 shape
# 提示：需要转置其中一个
```

### 练习 2：autograd
```python
# 定义 f(x) = x^3 - 2*x^2 + x，用 autograd 计算 x=2 时的导数
# 理论答案：f'(x) = 3x^2 - 4x + 1，x=2 时 = 3*4 - 8 + 1 = 5
x = torch.tensor(2.0, requires_grad=True)
# 你的代码
# 验证 x.grad == 5.0
```

### 练习 3：写一个最小的线性模型（核心练习）
用 `nn.Module` 实现一个单层线性分类器：
- 输入：`(batch, 10)` 的 tensor
- 输出：`(batch, 2)` 的 tensor（二分类）
- 在随机数据上跑 10 步训练循环，打印 loss

---

## 总结：看懂仓库代码的关键

| 看到这个 | 意思是 |
|----------|--------|
| `x.shape` | 这个数据是几维的？每维多大？ |
| `model(x)` | 调用 `model.forward(x)` |
| `loss.backward()` | 计算所有梯度 |
| `nn.Linear(a, b)` | 全连接层，把 a 维变成 b 维 |
| `x + residual` | 残差连接（防梯度消失） |
| `.to(device)` | 把数据/模型移到 CPU 或 GPU |
| `for param in model.parameters()` | 遍历所有可训练参数 |

---

*更新：2026-07-02*
