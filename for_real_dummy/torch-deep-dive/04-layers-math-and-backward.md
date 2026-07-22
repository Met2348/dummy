# 04 · 常用层前向反向数学推导(Layers: Math and Backward)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批是全系列数学密度最高的一批,也是"手推公式"类面试题的核心储备——不是"这层是干什么的"这种表面介绍,而是**签名怎么写、前向公式怎么推、反向传播的每一个梯度公式怎么从链式法则推出来、推出来的公式是否真的和 PyTorch 自动求导算出来的数字对得上**。面试官问"手写一下 Linear 层的反向传播"时,期望看到的就是本篇第 1 节的推导过程和验证方式;问"BatchNorm 的反向传播为什么不简单"时,期望看到的就是第 4 节那个"天真推导 vs 完整推导"的对比实验。

**前提:** 建议先看完 [02-pytorch-basics.md](../02-pytorch-basics.md)(知道 `nn.Linear`/`nn.ReLU`/`nn.Dropout`/`nn.LayerNorm`/`nn.Embedding`/`nn.MultiheadAttention` 这些层的名字和大致用途,但还没学过内部前向反向的数学),以及 [01-tensor-memory-model.md](01-tensor-memory-model.md)(tensor 的内存/stride 心智模型,第 3 节 im2col、第 8 节多头拆分会直接用到 `view`/`transpose`/`contiguous` 的直觉)。本篇**不依赖** [02-autograd-internals.md](02-autograd-internals.md)/[03-nn-module-internals.md](03-nn-module-internals.md) 里更底层的 autograd 引擎机制(计算图、`grad_fn`、`retain_graph` 这些)——那两批讲"autograd 引擎怎么运转",本篇讲"具体每一层的前向/反向公式长什么样",两者互补,但本篇可以独立阅读。

本文所有代码例子已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用)下实际跑通验证。**每一节的数学推导都不是"写完公式就完了"**——推导出的理论梯度公式,全部用代码实现出来,和 PyTorch `.backward()` 自动算出的 `.grad` 做 `torch.allclose` 数值对比(不用 `==` 比较,浮点数精度问题——这是写 [01-tensor-memory-model.md](01-tensor-memory-model.md) 时踩过的坑,该文第 6 节结尾专门提醒过),对不上就不会写进本文。

**本篇统一结构(与 01 篇一致的 7 段式):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 数学推导为主,推导完必须用代码验证理论值和实际值吻合
4. AI 研究场景
5. 可运行例子
6. **面试怎么问 + 追问链** —— 这批内容天然适合"请手推一下 XX 的反向传播"这类问法,追问链会往"如果输入维度变了会怎样""这个公式在什么条件下不成立"这类方向深挖
7. 常见坑

---

## 1. `nn.Linear` 前向 + 反向手推——面试"手写线性层反向传播"的标准答案

**是什么:**
```python
nn.Linear(in_features, out_features, bias=True)
# 前向: y = x @ W.T + b
#   x: (N, in_features)
#   W: (out_features, in_features)   -- 注意形状,不是 (in_features, out_features)
#   b: (out_features,)
#   y: (N, out_features)
```

**一句话:** 全连接层的前向就是一次矩阵乘法加广播加法;它的反向传播是矩阵求导链式法则里最基础、最常考的一个例子——三条梯度公式(`dL/dW`、`dL/db`、`dL/dx`)全部可以从一个标量下标公式出发直接推出来,不需要任何"奇技淫巧"。

**底层机制/为什么这样设计:**

先写出单个元素的前向公式(batch 内第 `n` 个样本、第 `o` 个输出通道):

```
y[n,o] = Σ_i x[n,i] * W[o,i]  +  b[o]
```

反向传播的输入是上游传下来的梯度 `dL/dy`(形状和 `y` 完全一致,`(N, out_features)`),目标是求出 `dL/dW`、`dL/db`、`dL/dx`。用链式法则,对每个具体的参数元素求偏导:

**对 `W[o,i]`:** 对固定的 `(o,i)`,`W[o,i]` 只通过 `y[n,o]`(对所有 `n`)影响 loss,系数是 `x[n,i]`:

```
dL/dW[o,i] = Σ_n (dL/dy[n,o]) * (dy[n,o]/dW[o,i]) = Σ_n dL/dy[n,o] * x[n,i]
```

写成矩阵形式(这一步是很多人会卡住的地方——下标公式对,但矩阵乘法的转置方向经常写反):

```
dL/dW = (dL/dy).T @ x        # (out,N) @ (N,in) = (out,in),形状和 W 完全一致
```

**对 `b[o]`:** 同理,`b[o]` 只影响 `y[:,o]` 这一整列,且系数恒为 1:

```
dL/db[o] = Σ_n dL/dy[n,o]  →  dL/db = dL/dy.sum(dim=0)     # 对 batch 维求和,容易漏掉这一步
```

**对 `x[n,i]`:** `x[n,i]` 通过所有 `o` 影响 `y[n,o]`(每个输出通道都用到了这个输入分量):

```
dL/dx[n,i] = Σ_o dL/dy[n,o] * W[o,i]  →  dL/dx = dL/dy @ W    # (N,out) @ (out,in) = (N,in)
```

**一个好记的形状校验技巧(面试现场推公式时最实用):** 任何参数/输入的梯度,形状必须和它自己完全一致——`dL/dW` 形状必须是 `(out,in)`,`dL/dx` 形状必须是 `(N,in)`。给定两个要相乘的矩阵,通常只有一种转置/顺序组合能凑出目标形状,忘了公式时,现场把形状摆出来倒推乘法顺序,十有八九能推对。

**可运行例子(手写反向传播,和 `.backward()` 数值对比,不吻合就不算过关):**

```python
import torch

torch.manual_seed(0)
N, in_features, out_features = 5, 4, 3

x = torch.randn(N, in_features, requires_grad=True)
W = torch.randn(out_features, in_features, requires_grad=True)
b = torch.randn(out_features, requires_grad=True)
target = torch.randn(N, out_features)

# ---- PyTorch autograd 路径(真值) ----
y = x @ W.T + b
loss = 0.5 * ((y - target) ** 2).sum()      # 用 MSE 形式的 loss,dL/dy = y - target,不是简单的全1向量
loss.backward()

# ---- 手写反向传播,完全不碰 autograd ----
with torch.no_grad():
    y_manual = x @ W.T + b
    dL_dy = y_manual - target                # MSE 对 y 的梯度

    manual_dW = dL_dy.T @ x                   # dL/dW = (dL/dy)^T @ x
    manual_db = dL_dy.sum(dim=0)              # dL/db = sum over batch
    manual_dx = dL_dy @ W                     # dL/dx = dL/dy @ W

assert torch.allclose(manual_dW, W.grad, atol=1e-5)
assert torch.allclose(manual_db, b.grad, atol=1e-5)
assert torch.allclose(manual_dx, x.grad, atol=1e-5)
```

实测:`manual_dW.shape == torch.Size([3, 4])`,和 `W.grad` 形状一致;三个 `allclose` 全部通过。额外用同样的权重构造一个真正的 `nn.Linear` 做交叉验证,它的 `.weight.grad`/`.bias.grad`/输入 `.grad` 同样和手推版本完全一致——说明 `nn.Linear` 内部的反向传播就是这三条公式,没有任何隐藏的额外逻辑。

追加验证(面试追问"能不能用 `einsum` 统一写这三条公式"的标准答案,实测两者完全一致):

```python
ein_dW = torch.einsum('no,ni->oi', dL_dy, x)   # 等价于 (dL/dy).T @ x
ein_dx = torch.einsum('no,oi->ni', dL_dy, W)   # 等价于 dL/dy @ W
assert torch.allclose(ein_dW, manual_dW, atol=1e-6)
assert torch.allclose(ein_dx, manual_dx, atol=1e-6)
```

**AI 研究场景:** 这是几乎一切模型的地基组件——CNN 的分类头、Transformer 的 QKV 投影和 FFN、RNN 的门控,本质都是若干个 Linear 的组合。手推这三条公式的意义不只是应付面试:自定义 `autograd.Function`(下一批 [02-autograd-internals.md](02-autograd-internals.md) 会讲)要求你手写 `forward`/`backward`,写法就是这个套路——`ctx.save_for_backward` 保存前向用到的张量,`backward` 里用保存的张量按这三条公式算梯度。

**面试怎么问 + 追问链:**
- **Q:** "手推一下 `nn.Linear` 的反向传播,给出 `dL/dW`、`dL/db`、`dL/dx` 的公式。" —— 期望完整写出上面三条公式,并能说清楚形状。
- **追问 1:** "`W` 的形状是 `(out_features, in_features)` 还是 `(in_features, out_features)`?为什么?" —— 需要准确说出 `nn.Linear.weight.shape == (out, in)`,前向用 `x @ W.T` 而不是 `x @ W`。
- **追问 2(考察是否真的理解,而不是背公式):** "如果 `bias=False`,三条公式里哪一条会消失?其余两条会变吗?" —— 期望答"`dL/db` 不存在了,`dL/dW`、`dL/dx` 公式完全不变,因为 `b` 只是加法项,不影响 `dy/dW` 和 `dy/dx`"。
- **追问 3(深挖边界条件):** "如果 batch 维度是 0(空 batch),这三条公式还成立吗?" —— `dL/dW`、`dL/dx` 会退化成全 0(求和是空求和),不会报错但训练上毫无意义,能看出候选人是否真的把公式里 `Σ_n` 这个求和的边界条件想清楚了。
- **追问 4(拓展到高维输入):** "如果输入 `x` 不是 2 维,而是 `(batch, seq_len, in_features)` 这种 3 维呢?" —— 期望答出 `nn.Linear` 只对最后一维做变换,前面的维度都当成"批量"维度处理,公式本质不变,只是 `Σ_n` 现在是对 `(batch, seq_len)` 这个组合维度求和——这是 Transformer 里 Linear 层的真实使用方式。

**常见坑:**
- `dL/dW = (dL/dy).T @ x` 的转置方向写反,写成 `x.T @ dL/dy`——这样算出来形状是 `(in,out)`,和 `W` 的形状 `(out,in)` 对不上,现场用形状校验就能发现。
- 算 `dL/db` 时忘记对 batch 维求和,只保留最后一个样本的梯度——`bias` 是所有样本共享的,每个样本都对它的梯度有贡献,必须求和(这个坑本质上和第 4 节 BatchNorm 里 `dL/dγ`/`dL/dβ` 求和是同一类错误)。
- 认为"手写反向传播"必须避开 `torch.no_grad()`——恰恰相反,手写反向传播的计算过程本身不应该被 autograd 记录(不然就是在用 autograd 验证 autograd,失去了对比的意义),用 `torch.no_grad()` 包裹手写部分、只让"真值"那一路留在计算图里,才是正确的对比方式。

---

## 2. `F.relu` vs `nn.ReLU`——functional 无状态 API 和 Module 封装的关系

**是什么:**
```python
torch.nn.functional.relu(input, inplace=False)   # 函数:给定输入,直接算,不需要先构造任何对象
torch.nn.ReLU(inplace=False)                       # 类:先实例化成一个 nn.Module 对象,再像函数一样调用
```

**一句话:** `nn.ReLU()` 内部**就是**调用 `F.relu`——两者算的是完全相同的东西,区别只在"有没有被包装成一个可以注册进模型结构的对象"。

**底层机制/为什么这样设计:** 直接看 PyTorch 源码里 `nn.ReLU.forward` 的实现(不是转述,是现场 `inspect.getsource` 打印出来的):

```python
def forward(self, input: Tensor) -> Tensor:
    return F.relu(input, inplace=self.inplace)
```

一行代码,直接转发给 `F.relu`。这个模式在 PyTorch 里是通用设计:`torch.nn.functional` 下面是一层**无状态的纯函数**,`torch.nn` 下面的 `Module` 子类(`nn.Linear`、`nn.ReLU` 的 `forward` 内部同理)是**对这些函数的有状态包装**。"有状态"体现在两方面:
1. **可学习参数**:`nn.Linear` 包装 `F.linear` 时,把 `weight`/`bias` 存成 `nn.Parameter`,注册进模块,能被 `optimizer` 追踪、能被 `state_dict()` 保存;`nn.ReLU` 没有可学习参数,但依然继承了"有状态对象"这个身份——它虽然没参数,却**可以被注册**。
2. **可被模型结构感知**:`nn.Sequential`、`nn.ModuleList` 这些容器要求元素是 `nn.Module`,才能递归遍历、`.to(device)`、`.train()/.eval()` 联动切换。`F.relu` 只是个函数调用,写在 `forward` 方法体内部,不会出现在 `named_modules()`/`state_dict()` 里,容器也放不进去。

**AI 研究场景:** 两种写法在实际代码里都很常见,选择标准是"要不要把这一步算作模型结构的一部分":
- 用 `nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))` 搭积木时,必须用 `nn.ReLU()`(容器需要 `Module` 对象)。
- 自己写 `class MyBlock(nn.Module): def forward(self, x): ... x = F.relu(x) ...` 这种自定义 `Module` 内部,用 `F.relu` 更简洁,不需要在 `__init__` 里多注册一个没有参数的子模块。
- 需要用 hook(`register_forward_hook`,下一批 [03-nn-module-internals.md](03-nn-module-internals.md) 会讲)在某一层的输出上做文章时,必须是 `Module` 对象——纯函数调用没有"对象"可以挂 hook,这时候即使 ReLU 没有参数,也要用 `nn.ReLU()`。

**可运行例子:**
```python
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

print(inspect.getsource(nn.ReLU.forward))
# def forward(self, input: Tensor) -> Tensor:
#     return F.relu(input, inplace=self.inplace)

x1 = torch.randn(4, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)

y1 = nn.ReLU()(x1)          # Module 版本
y2 = F.relu(x2)               # functional 版本
assert torch.equal(y1, y2)     # 前向完全一致

y1.sum().backward(); y2.sum().backward()
assert torch.equal(x1.grad, x2.grad)   # 反向也完全一致

# 只有 Module 版本能被容器注册
seq = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
names = [n for n, _ in seq.named_modules()]
assert '1' in names                     # ReLU 被注册成子模块,序号 "1"
assert sum(p.numel() for p in nn.ReLU().parameters()) == 0   # 参数量为 0,印证"无状态计算"

# inplace 参数在两个接口里行为完全一致(共享同一实现)
x3 = torch.tensor([-1.0, 2.0, -3.0])
F.relu(x3, inplace=True)
assert x3.tolist() == [0.0, 2.0, 0.0]
```

**面试怎么问 + 追问链:**
- **Q:** "`F.relu` 和 `nn.ReLU()` 有什么区别?" —— 很多人只会说"一个是函数一个是类",更好的答案是"`nn.ReLU` 内部就是调用 `F.relu`,区别只在有没有被包装成可注册的 Module 对象"。
- **追问 1:** "既然 `nn.ReLU()` 没有可学习参数,为什么还要费劲包装成一个类?直接用 `F.relu` 不是更简单吗?" —— 期望答出"能被 `nn.Sequential` 这类容器接纳、能挂 forward hook、`.train()/.eval()` 能联动传播到它身上(虽然 ReLU 本身不区分训练/推理,但架构上统一)"。
- **追问 2(考察知识面):** 举一个"函数版本和 Module 版本行为不能简单划等号"的反例。 —— 期望答出 `nn.Dropout()` 和 `F.dropout()`:`F.dropout` 需要显式传 `training=True/False`,而 `nn.Dropout` 会自动读取 `self.training`(由外层 `model.eval()` 联动设置),如果在自定义 forward 里用 `F.dropout` 却忘了传 `training=self.training`,推理时依然会随机丢弃——这是第 6 节的内容,也是一个真实的工程陷阱。
- **追问 3:** "`inplace=True` 在两个接口里都存在吗?行为一样吗?" —— 期望答"一样,因为 `nn.ReLU.forward` 本来就是把 `self.inplace` 转发给 `F.relu` 的 `inplace` 参数,是同一个底层实现"。

**常见坑:** 在自定义 `Module` 里用 `F.dropout`/`F.batch_norm` 这类**对训练/推理模式敏感**的函数时,忘记手动传 `training=self.training`——`F.relu` 因为没有这个模式敏感性,这个坑不会出现在它身上,但会出现在 `F.dropout` 上(见第 6 节),初学者容易把"`F.xxx` 和 `nn.XXX` 永远等价"这个从 ReLU 上得到的印象错误地带到 Dropout/BatchNorm 上。

---

## 3. `Conv2d` 的 im2col 实现原理——卷积为什么能变成一次矩阵乘法

**是什么:**
```python
F.conv2d(input, weight, bias=None, stride=1, padding=0)
# input:  (N, C_in, H, W)
# weight: (C_out, C_in, kH, kW)
# output: (N, C_out, H_out, W_out)
```
`im2col`("image to column")是卷积的一种**等价实现方式**:把每个滑动窗口覆盖到的那块输入,展开拼成一个"列",所有滑动窗口的列拼在一起组成一个大矩阵,卷积核也拉平成一个矩阵,卷积运算就变成了**一次矩阵乘法**。

**一句话:** 卷积的数学本质是"滑动窗口做内积"——`im2col` 只是把这些内积**重新排布**成一次大矩阵乘法,数值上完全等价,但把"很多次小的滑窗内积"变成"一次大矩阵乘法",这正是卷积能在 GPU 上高效并行的根本原因(矩阵乘法是 GPU/cuBLAS/Tensor Core 优化得最极致的运算)。

**底层机制/为什么这样设计:**

直接卷积的定义(先看单样本、单输出通道):

```
out[o,i,j] = Σ_c Σ_u Σ_v  input[c, i*s+u, j*s+v] * weight[o,c,u,v]  + bias[o]
```

这是"每个输出位置 `(i,j)` 都要对一个 `(C_in,kH,kW)` 的小块和卷积核做一次内积"的运算。如果老老实实按定义写循环,GPU 完全发挥不出并行矩阵乘法的优势(大量小规模、形状不规则的内积,调度开销远大于计算量)。

`im2col` 的做法:
1. 把输入里每一个大小为 `(C_in,kH,kW)` 的滑动窗口,**展平成一个长度为 `C_in*kH*kW` 的列向量**。
2. 一共有 `H_out*W_out` 个滑动窗口位置,把这些列向量**横向拼接**,得到一个 `(C_in*kH*kW, H_out*W_out)` 的大矩阵,记作 `cols`。
3. 把卷积核 `weight` 从 `(C_out,C_in,kH,kW)` **拉平**成 `(C_out, C_in*kH*kW)`。
4. 卷积就变成了一次矩阵乘法:`weight_flat @ cols`,形状 `(C_out,C_in*kH*kW) @ (C_in*kH*kW,H_out*W_out) = (C_out,H_out*W_out)`,再 reshape 回 `(C_out,H_out,W_out)`。

代价是**显存**:`cols` 矩阵里,相邻滑动窗口之间有大量像素是重叠的(尤其 `stride < kernel_size` 时),`im2col` 会把这些重叠像素**重复存储**多份,`cols` 矩阵通常比原始输入大好几倍——这是"用显存换并行度"的典型工程取舍,也是为什么现代实现(cuDNN)在 `im2col` 之外还有 Winograd、FFT 等不需要展开重复数据的卷积算法,`im2col` 是最直观、最容易验证正确性,但不是显存最省的一种。

**可运行例子(手写 im2col + 矩阵乘法,和 `F.conv2d` 数值对比):**

```python
import torch
import torch.nn.functional as F

def im2col_manual(x, kernel_size, stride=1, padding=0):
    N, C, H, W = x.shape
    kH, kW = kernel_size
    x_padded = F.pad(x, (padding, padding, padding, padding))
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    cols = []
    for i in range(H_out):
        for j in range(W_out):
            h0, w0 = i * stride, j * stride
            patch = x_padded[:, :, h0:h0 + kH, w0:w0 + kW]     # (N,C,kH,kW) 一个滑动窗口
            cols.append(patch.reshape(N, -1))                    # 展平成一列
    return torch.stack(cols, dim=2), H_out, W_out                 # (N, C*kH*kW, L), L=H_out*W_out

def conv2d_via_im2col(x, weight, bias=None, stride=1, padding=0):
    C_out = weight.shape[0]
    cols, H_out, W_out = im2col_manual(x, weight.shape[2:], stride, padding)
    w_flat = weight.reshape(C_out, -1)
    out = torch.einsum('oc,ncl->nol', w_flat, cols)   # 核心:一次矩阵乘法替代所有滑窗内积
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out.reshape(-1, C_out, H_out, W_out)

# ---- 验证:和 F.conv2d 数值完全一致 ----
torch.manual_seed(0)
x = torch.randn(2, 3, 7, 7)
weight = torch.randn(4, 3, 3, 3)
bias = torch.randn(4)

out_manual = conv2d_via_im2col(x, weight, bias, stride=2, padding=1)
out_torch = F.conv2d(x, weight, bias, stride=2, padding=1)
assert out_manual.shape == out_torch.shape == (2, 4, 4, 4)
assert torch.allclose(out_manual, out_torch, atol=1e-4)   # 实测最大误差 1.907e-06
```

实测最大误差 `1.9073486328125e-06`,在 float32 浮点误差范围内,和 `F.conv2d` 完全等价。

**顺带验证手写 im2col 本身对不对:** PyTorch 内置了 `F.unfold`,就是 im2col 这个原语本身(不含后续矩阵乘法),可以直接拿它交叉验证:

```python
cols_manual, _, _ = im2col_manual(x, (3, 3), stride=2, padding=1)
cols_builtin = F.unfold(x, kernel_size=(3, 3), stride=2, padding=1)
assert cols_manual.shape == cols_builtin.shape == (2, 27, 16)   # 27 = 3(C_in) * 3 * 3(kernel)
assert torch.allclose(cols_manual, cols_builtin, atol=1e-6)
```

**反向传播:** 因为手写实现全程只用了 `slicing`/`reshape`/`einsum` 这些 autograd 原生支持的可微操作,`x`、`weight` 只要 `requires_grad=True`,`.backward()` 会自动算出正确梯度——不需要再手写一次反向传播。实测让 `x`、`weight`、`bias` 分别过一遍手写路径和 `F.conv2d` 路径,`dL/dx`、`dL/dW`、`dL/db` 全部对得上(`atol=1e-3`——卷积累加了大量元素,数值误差比 Linear 层稍大属正常)。

**面试深挖点:im2col 的反向传播是什么?** 既然前向是"展开成列 + 矩阵乘法",反向传播自然对应"矩阵乘法的反向(转置矩阵乘)+ 把梯度按原来的滑窗位置**加回去**"——注意是"加回去"不是"填回去":重叠的像素在前向被重复用了多次,反向就要把这些重叠位置上的梯度**累加**。这个"把展开的列按原位置累加回原图"的操作叫 **col2im**,PyTorch 里对应 `F.fold`(`F.unfold` 的伴随算子)。用内积不变性 `<unfold(x), g> == <x, fold(g)>`(对任意 `g` 成立)可以验证这个关系,实测两边数值相等,印证了"重叠区域反向要累加梯度"这个说法。

**AI 研究场景:** 现代框架的卷积算子在底层不会永远用最朴素的 `im2col`(会因为显存膨胀在大 kernel/高分辨率场景下变得很贵),但**理解 im2col 是理解"卷积如何映射到矩阵乘法硬件"的第一步**——无论是 cuDNN 的多种卷积算法选择,还是移动端推理引擎为省显存做的"隐式 GEMM"(不真正展开 `cols`,而是让矩阵乘法内核在读取时按需重新计算每个元素该从原图哪里取),都是在 `im2col` 这个基线思路上做空间换时间/时间换空间的权衡。

**面试怎么问 + 追问链:**
- **Q:** "卷积在 GPU 上是怎么实现高效并行的?" —— 期望答出"通过 im2col 把滑动窗口内积转换成一次大矩阵乘法,复用 GPU 上高度优化的 GEMM(矩阵乘法)kernel"。
- **追问 1:** "im2col 有什么代价?" —— 期望答出"显存膨胀,因为相邻滑动窗口有重叠像素,展开后会被重复存储,重叠程度由 `stride` 相对 `kernel_size` 的大小决定"。
- **追问 2(深挖):** "如果 `stride == kernel_size`(无重叠)呢,im2col 还会有显存膨胀问题吗?" —— 这时候滑动窗口彼此不重叠,`cols` 矩阵的元素总数和原图元素总数是同一个量级,没有"重复存储"的膨胀——好的候选人能现场意识到"膨胀程度和 `stride` vs `kernel_size` 的关系"这个连续的权衡,而不是死记"im2col 费显存"这一句话。
- **追问 3:** "im2col 的反向传播是什么?" —— 期望答出"col2im/`F.fold`,重叠位置的梯度要做累加,不是简单地把列塞回原位置"。
- **追问 4(拓展到 1x1 卷积):** "`kernel_size=1, stride=1, padding=0` 的卷积,用 im2col 会退化成什么?" —— 这时候每个"窗口"就是单个像素本身,`cols` 就是原始输入本身 reshape 一下(没有任何重叠、没有任何膨胀),卷积退化成一个**逐像素位置的矩阵乘法**,这正是为什么 1x1 卷积在实现上等价于对每个空间位置独立做一次 Linear 变换(ResNet/MobileNet 里大量使用 1x1 卷积做通道数变换,底层就是这个)。

**常见坑:**
- 以为 `im2col` 只是"教科书讲讲",实际框架不会这样实现——`F.unfold`/`F.fold` 是 PyTorch 内置的一等公民 API,某些需要"手工控制卷积展开方式"的场景(实现新的卷积变体、可视化每个位置用到的感受野)会直接用到。
- 混淆 `cols` 矩阵的维度顺序——不同资料/框架对 `(C*kH*kW, L)` 还是 `(L, C*kH*kW)` 的约定不一致,面试手推时**先明确自己的约定再推导矩阵乘法顺序**,否则转置方向很容易出错(这一点和第 1 节 `nn.Linear` 反向传播的转置陷阱是同一类错误)。
- 认为 im2col 只是"节省了 Python 循环的开销"——真正的收益是让计算映射到 GPU 上专门优化过的 GEMM kernel,如果在 CPU 上、且没有调用底层优化过的 BLAS 库,im2col 版本不一定比朴素循环快多少,"快"是"矩阵乘法在特定硬件上被极致优化"带来的,不是"矩阵乘法"这个数学形式本身自带的属性。

---

## 4. `BatchNorm` 训练/推理模式 + 反向传播公式推导——本篇难度最高的一节

**是什么:**
```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
# 训练模式 (.train()):用当前 batch 的均值/方差归一化,同时用它们更新 running_mean/running_var
# 推理模式 (.eval()) :用训练过程中累积的 running_mean/running_var 归一化,不再依赖当前输入的统计量
```

**一句话:** BatchNorm 对每个特征通道独立操作,训练时"看当下这批数据的均值方差",推理时"用训练全程攒下来的均值方差"——两种模式用的是不同的数字,同一个输入在两种模式下会算出不同的输出,这不是 bug,是设计如此。

**底层机制/为什么这样设计——训练/推理模式的公式:**

对某一个特征通道,batch 内有 `N` 个样本 `x_1,...,x_N`。

训练模式:
```
μ_B = (1/N) Σ_n x_n                        # 当前 batch 的均值
σ²_B = (1/N) Σ_n (x_n - μ_B)²               # 当前 batch 的方差(分母是 N,不是 N-1 —— 有偏估计)
x̂_n = (x_n - μ_B) / sqrt(σ²_B + ε)          # 归一化
y_n = γ * x̂_n + β                           # 可学习的缩放和平移,还原表达能力
```

同时,每次训练前向都会用指数滑动平均更新两个 buffer(不是可学习参数,是 `register_buffer` 注册的统计量,不参与梯度更新,下一批 [03-nn-module-internals.md](03-nn-module-internals.md) 会细讲 buffer 和 parameter 的区别):

```
running_mean = (1-momentum) * running_mean + momentum * μ_B
running_var  = (1-momentum) * running_var  + momentum * σ²_B,unbiased      # 注意:这里用的是无偏方差!
```

**一个非常容易被忽略、也非常容易被面试问到的细节:** 归一化当前 batch 用的 `σ²_B` 是**有偏**方差(分母 `N`),但更新 `running_var` 用的却是**无偏**方差(分母 `N-1`,Bessel 校正)——两处用了不同的方差估计,已用 `nn.BatchNorm1d` 实测验证(见下方代码)。这是 PyTorch 的实现细节,不是数学上必须如此,但足够偏门、足够适合用来判断候选人是不是真的读过源码或者认真做过实验,而不是只会背"BN 用均值方差做归一化"这种粗粒度描述。

推理模式:
```
y_n = γ * (x_n - running_mean) / sqrt(running_var + ε) + β
```
不再计算当前输入的 `μ_B`、`σ²_B`,也不再更新 `running_mean`/`running_var`。

**底层机制/为什么这样设计——反向传播推导(标量简化版本,针对单个特征通道):**

这是本篇故意留到最后强调的难点:**BN 的反向传播不是"对每个输出独立地应用链式法则"这么简单**,因为 `μ_B` 和 `σ²_B` 本身是**所有** `x_n` 的函数——改变任意一个 `x_i`,不仅 `y_i` 会变,`μ_B`、`σ²_B` 也会跟着变,进而**其他所有** `y_n`(`n≠i`)也会跟着变。也就是说,`y_i` 对 `x_j`(`j≠i`)存在一条"经过 `μ_B`/`σ²_B` 中转"的间接梯度路径,这是一个普通的逐元素函数(比如 ReLU、Sigmoid)完全不会有的现象。

**一个直接的反例,证明"把 `μ_B`、`σ²_B` 当常数求导"是错的:** 如果天真地把 `x̂_i=(x_i-μ_B)/sqrt(σ²_B+ε)` 里的 `μ_B`、`σ²_B` 当成与 `x_i` 无关的常数,会得到 `dx̂_i/dx_i = 1/sqrt(σ²_B+ε)`,进而 `dL/dx_i ≈ (dL/dy_i) * γ / sqrt(σ²_B+ε)`。下面的代码现场验证:**这个天真公式和 autograd 算出来的真实梯度对不上**(最大误差 `0.558`,量级上完全不对),必须把 `μ_B`、`σ²_B` 对 `x_i` 的依赖也考虑进链式法则,才能对上(误差 `~6e-8`,float32 精度范围)。

完整推导(仍然按标量链式法则一步步来,`i` 遍历 batch 内所有样本):

```
dL/dx̂_i = (dL/dy_i) * γ                                              # 第一步,普通链式法则

dL/dσ²_B = Σ_i (dL/dx̂_i) * (x_i - μ_B) * (-1/2) * (σ²_B+ε)^(-3/2)     # x̂_i 对 σ²_B 的依赖,对所有 i 求和

dL/dμ_B  = [Σ_i (dL/dx̂_i) * (-1/sqrt(σ²_B+ε))]
           + (dL/dσ²_B) * (-2/N) * Σ_i (x_i - μ_B)                    # 第二项因 Σ(x_i-μ_B)=0 恒为 0,但推导时先别急着省略

dL/dx_i  = (dL/dx̂_i) / sqrt(σ²_B+ε)                                   # 直接路径(和天真公式一样的部分)
           + (dL/dσ²_B) * 2*(x_i - μ_B) / N                            # 经过 σ²_B 中转的间接路径
           + (dL/dμ_B) / N                                             # 经过 μ_B 中转的间接路径
```

后两行正是"天真推导"漏掉的部分——`x_i` 不仅通过 `x̂_i` 自己的直接路径影响 loss,还通过"我是 batch 里的一员,我参与决定了这个 batch 的均值和方差"这条间接路径,影响了**所有** `y_n` 进而影响 loss。这也是为什么 BN 的反向传播不能像 ReLU/Linear 那样只看自己这一个输出——它天然是一个"batch 内样本互相耦合"的算子。

**可运行例子(天真推导 vs 完整推导,和 autograd 数值对比):**

```python
import torch
torch.manual_seed(0)
eps = 1e-5
N = 6
x = torch.randn(N, requires_grad=True)
gamma = torch.tensor(1.7, requires_grad=True)
beta = torch.tensor(-0.3, requires_grad=True)

# autograd 真值
mu = x.mean()
var = ((x - mu) ** 2).mean()
xhat = (x - mu) / torch.sqrt(var + eps)
y = gamma * xhat + beta
upstream = torch.randn(N)          # 任意上游梯度,不用标量 loss,更能测出公式是否普适
y.backward(upstream)
true_dx = x.grad.clone()

# 天真推导:把 mu, var 当常数
with torch.no_grad():
    mu_v, var_v = x.mean(), ((x - x.mean()) ** 2).mean()
    std_inv = 1.0 / torch.sqrt(var_v + eps)
    naive_dx = upstream * gamma.detach() * std_inv

# 完整推导:考虑 mu, var 也是 x 的函数
with torch.no_grad():
    dxhat = upstream * gamma.detach()
    dvar = (dxhat * (x - mu_v) * (-0.5) * (var_v + eps) ** (-1.5)).sum()
    dmu = (dxhat * (-std_inv)).sum() + dvar * (-2.0 / N) * (x - mu_v).sum()
    correct_dx = dxhat * std_inv + dvar * (2.0 * (x - mu_v) / N) + dmu / N

assert not torch.allclose(naive_dx, true_dx, atol=1e-3)     # 天真推导:错
assert torch.allclose(correct_dx, true_dx, atol=1e-4)        # 完整推导:对
```

实测:`naive_dx` 和真值最大误差 `0.558`(量级上完全对不上,证明天真推导是错的);`correct_dx` 和真值最大误差 `5.96e-08`,在 float32 精度范围内完全吻合。`dL/dγ`、`dL/dβ` 相对简单(`γ`、`β` 是逐样本共享的仿射参数,不存在"经过均值方差中转"的耦合),分别是 `(upstream * x̂).sum()` 和 `upstream.sum()`,同样实测和 autograd 完全一致。

**训练/推理模式差异 + 有偏/无偏方差差异,同样现场验证过:**
```python
bn = torch.nn.BatchNorm1d(4, momentum=0.1)
xb = torch.randn(8, 4)
out_train = bn(xb)                    # 用当前 batch 的有偏方差(除以N)归一化

batch_var_biased = xb.var(dim=0, unbiased=False)
batch_var_unbiased = xb.var(dim=0, unbiased=True)
assert torch.allclose(out_train, bn.weight * (xb - xb.mean(dim=0)) / torch.sqrt(batch_var_biased + bn.eps) + bn.bias, atol=1e-5)
assert torch.allclose(bn.running_var, 0.9 * torch.ones(4) + 0.1 * batch_var_unbiased, atol=1e-5)   # running_var 用无偏方差更新

bn.eval()
out_eval = bn(xb)                       # 同样的输入,用 running 统计量归一化
assert not torch.allclose(out_train, out_eval, atol=1e-3)   # 实测:确实不一样
```

**AI 研究场景:** BN 的"batch 内样本互相耦合"这个性质,是它在 Transformer/NLP 场景里几乎绝迹的根本原因之一(下一节详细展开)——推理时如果 batch size=1(比如线上单条请求),训练模式的 BN 直接**报错**(方差退化,实测复现报错信息 `Expected more than 1 value per channel when training, got input size torch.Size([1, 5])`),这逼着 BN 必须依赖 running 统计量才能处理任意 batch size。而"训练/推理用不同统计量"这个设计,也是很多"训练时表现很好、上线后掉点"排查案例的根源之一——一旦怀疑这类问题,第一反应应该是检查 `model.eval()` 有没有被正确调用、`running_mean`/`running_var` 是不是被污染了(比如某次忘记 `.eval()` 就跑了统计,或者数据分布线上线下不一致导致 running 统计量本身就不能代表线上数据),这个排查方法论会在 [11-debugging-and-common-errors.md](11-debugging-and-common-errors.md) 进一步展开。

**面试怎么问 + 追问链:**
- **Q:** "BatchNorm 训练模式和推理模式有什么区别?" —— 表层答案是"训练用 batch 统计量,推理用 running 统计量",这是几乎所有人都能答上来的第一层。
- **追问 1(过滤表层背诵):** "手推一下 BN 对输入 `x_i` 的梯度,和普通的逐元素激活函数(比如 Sigmoid)的反向传播,本质区别是什么?" —— 期望答出"`μ_B`、`σ²_B` 是所有 `x_n` 的函数,所以 `x_i` 有除了自身以外、经过均值方差中转的间接梯度路径,不能把 `μ_B`、`σ²_B` 当常数处理"。
- **追问 2(杀伤力很强,建议现场让候选人推公式或至少讲清楚思路):** "如果我们真的把 `μ_B`、`σ²_B` 当常数来求梯度,会发生什么?" —— 期望答出"公式会漏掉两条间接路径,数值上是错的",最好能提到"这是很多人自己实现 BN 时踩过的坑,反向传播算出来的梯度会系统性偏离真实梯度,导致训练不收敛或收敛异常"。
- **追问 3(细节,考察是否读过实现):** "归一化当前 batch 用的方差,和更新 `running_var` 用的方差,是同一个公式吗?" —— 期望答出"不是,归一化用有偏方差(除以 N),更新 running_var 用无偏方差(除以 N-1,Bessel 校正)"——这是本节里最容易被问倒的细节。
- **追问 4(边界条件):** "batch size=1 时,训练模式的 BatchNorm 会发生什么?" —— 期望答出"方差退化(单个样本谈不上'和均值的离散程度'),PyTorch 直接抛异常,这也是为什么很多需要单样本推理的场景不能直接用训练模式的 BN"。

**常见坑:**
- 只会背"BN 训练用 batch 统计、推理用 running 统计"这一句话,推不出反向传播为什么"不简单"——这是本节的考察重点,详见追问 1/2。
- 忽略"归一化用有偏方差、更新 running_var 用无偏方差"这个细节,以为整个 BN 内部只有一种方差计算方式。
- 以为 `momentum` 参数的含义和"优化器里的动量"是同一个东西——BN 里的 `momentum` 是指数滑动平均的更新步长(新统计量占多大权重),数学形式上和优化器动量([06-optimizer-internals.md](06-optimizer-internals.md) 会讲)不是一回事,只是刚好用了同一个词。
- 记错 PyTorch 里 `momentum` 的更新方向——实测确认公式是 `running=(1-momentum)*running+momentum*batch_stat`,而不是某些其他框架里 `running=momentum*running+(1-momentum)*batch_stat` 这种相反的语义约定;用错方向虽然不会报错,但会导致 running 统计量收敛速度和预期完全不一样,这类"跑起来不报错但数值不对"的坑最难排查,唯一办法是像本节一样现场验证公式,不能凭印象。

---

## 5. `LayerNorm` vs `BatchNorm`——归一化的是不同的维度,这个区别决定了 Transformer 的选择

**是什么:**
```python
nn.BatchNorm1d(num_features)     # 对同一个特征通道,在整个 batch 上求均值方差
nn.LayerNorm(normalized_shape)    # 对同一个样本,在它自己的所有特征上求均值方差
```

**一句话:** BN 和 LN 都是"减均值除标准差再缩放平移"这同一个数学操作,唯一但决定性的区别是**在哪个维度上求这个均值和方差**——BN 是"竖着看"(同一特征、跨样本),LN 是"横着看"(同一样本、跨特征)。

**底层机制/为什么这样设计:**

把一个 batch 的数据想象成一张 `(N,C)` 的表格,`N` 行是样本,`C` 列是特征:

- **BatchNorm**:对每一**列**(每个特征通道)单独求均值方差,统计量来自这一列的 `N` 个数字(跨样本)。计算 `y[:,c]` 的均值方差,只用到 `x[:,c]` 这一整列。
- **LayerNorm**:对每一**行**(每个样本)单独求均值方差,统计量来自这一行的 `C` 个数字(跨特征)。计算 `y[n,:]` 的均值方差,只用到 `x[n,:]` 这一整行。

这个"横着看还是竖着看"的区别,直接决定了三个对 Transformer 至关重要的性质:

**1. batch 独立性。** LN 的输出**只依赖这个样本自己**——把 batch 里其他任何样本换成别的值,这个样本的 LN 输出**完全不变**(下面代码现场验证)。BN 的输出**依赖整个 batch 里所有样本**——改变某一个样本,BN 会改变 batch 里**所有**样本的输出(包括没被改动的那些),因为它们共享同一个 `μ_B`、`σ²_B`。NLP 任务里,一个 batch 内的句子是相互独立的样本(第 3 句话的翻译不应该受第 7 句话内容的影响),LN 的 batch 独立性天然匹配这个假设,BN 破坏了它。

**2. 变长序列 / 小 batch / batch size=1 场景。** LN 逐样本工作,batch size 是 1 还是 1000 对它完全没有影响。BN 需要"batch 内有多个样本"才能算出有意义的方差(上一节已经验证 batch_size=1 时训练模式 BN 直接报错)——推理时单条请求(batch=1)、或者自回归解码时逐 token 生成,BN 完全不可用,LN 不受影响。

**3. 推理时不依赖 batch 统计量。** BN 需要额外维护 `running_mean`/`running_var` 两个 buffer,训练/推理用不同公式;LN **没有** `running_mean`/`running_var` 这两个 buffer(下面代码验证 `hasattr` 为 False),`.train()`/`.eval()` 两种模式算的是同一个公式——每次都用当前样本自己的均值方差,不存在"训练阶段统计量和推理时输入分布不匹配"这类问题。

**可运行例子:**

```python
import torch, torch.nn as nn
torch.manual_seed(0)
N, C = 4, 5
x = torch.randn(N, C) * 3 + 1

bn = nn.BatchNorm1d(C); ln = nn.LayerNorm(C)
out_bn, out_ln = bn(x), ln(x)

# BN: 每个通道(dim=0,跨样本)均值0方差1;LN: 每个样本(dim=1,跨特征)均值0方差1
assert torch.allclose(out_bn.mean(dim=0), torch.zeros(C), atol=1e-5)     # 按列(通道)看
assert torch.allclose(out_ln.mean(dim=1), torch.zeros(N), atol=1e-5)     # 按行(样本)看

# 决定性实验:扰动样本3,看样本0的输出变不变
x2 = x.clone(); x2[3] += 100.0
out_bn_before, out_bn_after = bn(x)[0], bn(x2)[0]
out_ln_before, out_ln_after = ln(x)[0], ln(x2)[0]

assert not torch.allclose(out_bn_before, out_bn_after, atol=1e-2)   # BN: 样本0的输出被"连坐"改变了!
assert torch.allclose(out_ln_before, out_ln_after, atol=1e-5)         # LN: 样本0完全不受影响

assert not hasattr(ln, "running_mean") and hasattr(bn, "running_mean")   # LN 没有 running 统计量
ln.eval()
assert torch.allclose(ln(x), out_ln, atol=1e-6)   # LN 的 train/eval 输出完全一样(同一个公式)
```

实测:扰动样本 3(加 100)之后,样本 0 的 BN 输出从 `[-0.5213,-1.4035,-1.6630,-1.4016,0.5463]` 变成 `[-0.6254,-0.6265,-0.6622,-0.6413,-0.5676]`——样本 0 的**输入**其实一个数字都没变,**输出**却整个变了;同一场景下样本 0 的 LN 输出前后完全一致,一位小数都不差。

**AI 研究场景:** 这是"为什么 Transformer 用 LayerNorm 不用 BatchNorm"这个经典问题的完整答案链条——NLP 序列长度可变(BN 要求在 batch 维上算统计量,变长序列的 padding 部分会污染统计量)、推理常见小 batch/单条请求(BN 在 batch=1 时数学上退化)、以及自回归生成场景每步只处理一个新 token(时间维上不存在"batch"概念)。反过来,CNN 处理的图像分类场景,batch 内样本之间没有这种强独立性要求(甚至有依赖 batch 统计量带来的正则化收益),且训练时通常能保证较大的 batch size,BN 在这个场景里工作得很好——这也是为什么"BN/LN 哪个更好"没有绝对答案,要看数据和任务的结构。

**面试怎么问 + 追问链:**
- **Q:** "LayerNorm 和 BatchNorm 有什么区别?" —— 期望第一句话就是"归一化的维度不同",而不是先讲"一个用在 CV 一个用在 NLP"这种结论,结论要从维度区别推出来,不是背出来。
- **追问 1(核心追问):** "为什么 Transformer 用 LayerNorm 不用 BatchNorm?给出至少两个原因。" —— 期望答出 batch 独立性、变长序列/小 batch 场景、推理不依赖 running 统计量 这几点中的至少两点,并能解释因果关系(不是"因为大家都这么用")。
- **追问 2(现场推理,考察是否真懂):** "如果输入是 `(batch, seq_len, hidden_dim)` 的 3 维张量,`nn.LayerNorm(hidden_dim)` 是在哪个维度上算均值方差?" —— 期望答出"只在最后一维 `hidden_dim` 上算,对每个 `(batch, seq_len)` 位置独立处理,和 batch 维、序列位置维完全无关"——这是 Transformer 里 LN 的真实使用方式。
- **追问 3(反问):** "既然 LN 这么多优点,BN 是不是已经过时了?" —— 期望答出"没有绝对优劣,CNN/图像任务里 batch 通常较大且样本间没有强独立性要求,BN works well,甚至提供了额外的正则化效果;LN 的优势是针对 NLP/序列/小 batch 场景的具体问题,不是全面碾压"。
- **追问 4(深挖到另一种归一化):** "如果既想保留 batch 独立性,又想对整个样本(不只是最后一维)做归一化,CV 领域有没有类似的设计?" —— 这里可以提一句 GroupNorm/InstanceNorm(不要求展开讲,是给候选人一个"归一化家族"的整体视野,能不能接住这个问题能看出候选人的知识广度)。

**常见坑:**
- 把"LN 在 NLP 用、BN 在 CV 用"当成死记硬背的结论,面试被追问"为什么"时说不出机制层面的原因。
- 以为 LayerNorm 是"对整个 batch 求一个均值方差,只是公式和 BN 不同"——实际上 LN 每个样本的统计量都是独立算的,`(N,C)` 输入会算出 `N` 组均值方差(每个样本一组),不是一组。
- 混淆 `nn.LayerNorm(normalized_shape)` 里 `normalized_shape` 的语义——它指定的是"从最后一维往前数,哪几维参与归一化",不是"batch 里有多少样本",这个参数在处理 `(N,seq_len,hidden_dim)` 这类多维输入时,写错很容易导致归一化维度选错而不报错(形状允许但语义错误,是最难排查的一类 bug)。

---

## 6. `Dropout` 的 inverted scale 技巧——为什么训练时要多做一次放大

**是什么:**
```python
nn.Dropout(p=0.5)
# 训练模式:以概率 p 把每个元素置0,同时把"存活"下来的元素除以 (1-p)
# 推理模式:什么都不做,原样输出(identity)
```

**一句话:** Dropout 训练时不只是"随机丢弃"——存活下来的激活值还要被**放大** `1/(1-p)` 倍,这个放大不是随意加的补偿,而是为了让"训练时的期望激活量级"和"推理时不丢弃、直接用全部激活值"的量级**对齐**,这样推理阶段才能什么额外处理都不用做,直接 pass through。

**底层机制/为什么这样设计:**

记 `keep_prob=1-p`,`mask` 是逐元素独立采样的伯努利随机变量(以概率 `keep_prob` 取 1,以概率 `p` 取 0)。

**如果不做缩放**(最朴素的写法):
```
output_train = x * mask
E[output_train] = E[mask] * x = keep_prob * x
```
存活元素直接保留原值,置 0 元素变成 0,取期望后,整体量级变成了原来的 `keep_prob` 倍——比如 `p=0.5` 时,训练阶段输出的期望量级只有输入的一半。但推理阶段不丢弃任何东西(`output_eval=x`),量级是输入的 100%。这就产生了一个**训练/推理量级不匹配**的问题:同一层,输出的期望量级在两个阶段差了 `keep_prob` 倍,下游所有层看到的"典型输入量级"在训练和推理时是不一样的,训练学到的权重尺度到了推理阶段不再适配。

**两种修正方式:**
1. **原始 Dropout 论文(Hinton et al. 2012)的做法**:训练时不缩放,**推理时**把权重/激活值乘以 `keep_prob` 补偿。问题是这要求推理阶段做额外计算,还容易被忘记(尤其是模型导出/部署到另一套推理引擎时,这个"隐藏的缩放步骤"很容易在转换过程中丢失)。
2. **inverted dropout(现在 `nn.Dropout` 实际采用的方式)**:训练时就把存活元素放大 `1/keep_prob` 倍:
```
output_train = (x * mask) / keep_prob
E[output_train] = keep_prob * x / keep_prob = x       # 和 output_eval = x 的量级完全对齐!
```
这样一来,推理阶段**不需要任何额外处理**——`Dropout` 在 `eval()` 模式下就是一个纯粹的恒等函数,这正是"inverted"这个名字的由来:把本该在推理时做的缩放,反过来挪到了训练时做。

**可运行例子:**
```python
import torch, torch.nn as nn
p = 0.3; keep_prob = 1 - p
drop = nn.Dropout(p=p)

x = torch.ones(200_000)          # 常数1,方便直接读出缩放系数
drop.train()
out = drop(x)

survived = out[out != 0]
assert abs((out != 0).float().mean().item() - keep_prob) < 0.01     # 存活比例 ≈ keep_prob(大数定律)
assert torch.allclose(survived, torch.full_like(survived, 1 / keep_prob), atol=1e-6)  # 存活值精确等于 1/keep_prob
assert abs(out.mean().item() - 1.0) < 0.01                            # 训练模式输出期望 ≈ 输入(1.0),没有偏移!

drop.eval()
assert torch.equal(drop(x), x)     # 推理模式:纯粹的恒等函数

# 对比:不做 inverted scale 的朴素版本,期望量级只有 keep_prob,和推理阶段的 1.0 对不上
naive_mean = (x * (torch.rand_like(x) > p).float()).mean().item()
assert abs(naive_mean - keep_prob) < 0.01
assert abs(naive_mean - 1.0) > 0.2    # 朴素版本和推理阶段量级差了整整 p
```

实测:`p=0.3` 时,训练模式下约 `70.0%` 的元素存活,存活值精确等于 `1/0.7=1.428571`;训练模式整体输出均值 `1.0000429`(≈ 输入的 1.0);不做缩放的朴素版本整体均值只有 `0.7012`,和推理阶段的 `1.0` 差了整整 `p=0.3`——这就是 inverted scale 要消除的那个"训练推理量级不匹配"。

**反向传播:** Dropout 的反向传播复用**同一个 mask**——`dL/dx_i=0`(如果 `x_i` 被丢弃)或 `dL/dx_i=(dL/dy_i)/keep_prob`(如果存活),已用 `.backward()` 现场验证:被丢弃的位置梯度精确为 0,存活位置梯度额外乘以 `1/keep_prob`,这一点很自然——前向是逐元素乘以 `mask/keep_prob`,链式法则下反向传播也是乘以同一个系数。

**AI 研究场景:** inverted dropout 这个设计模式(**训练时做额外工作,换取推理时零开销**)在深度学习工程里反复出现,不止 Dropout 一处——理解了这里的动机,再遇到类似"训练和推理路径不对称,但推理端被刻意设计成更简单"的模块时,会更容易猜到设计意图是什么。

**面试怎么问 + 追问链:**
- **Q:** "Dropout 训练时为什么要把存活的激活值除以 `1-p`?" —— 期望完整推出"不做缩放会导致训练/推理阶段期望激活量级不匹配"这个核心原因,而不是只说"为了不影响后面的层"这种模糊表述。
- **追问 1:** "如果不用 inverted dropout,而是用最早论文里'训练不缩放、推理时缩放'的方式,会有什么问题?" —— 期望答出"推理阶段需要额外乘法,容易在模型导出/部署时被遗漏,且如果模型要同时支持多套推理后端,每一套都要正确实现这个缩放,inverted 版本把这个负担在训练时一次性解决"。
- **追问 2(边界条件):** "`p=1.0`(全部丢弃)会发生什么?" —— `keep_prob=0`,`1/keep_prob` 除零,期望候选人能反应过来这是一个数学上的退化情形,体现"考虑边界条件"的习惯。
- **追问 3(深挖到训练稳定性):** "Dropout 在卷积层(`nn.Dropout2d`)和全连接层上的丢弃粒度一样吗?" —— 这是一个开放追问,期望候选人知道 `Dropout2d` 是按**整个通道**丢弃而不是逐像素丢弃(因为卷积特征图相邻像素高度相关,逐像素丢弃起不到有效的正则化作用),体现对"同一个思想在不同层类型上要不同实现方式"的敏感度(不要求现场推导,能提出这个区别本身就是加分项)。

**常见坑:**
- 自定义 `forward` 里用 `F.dropout(x, p=0.5)` 却忘记传 `training=self.training`——`F.dropout` 默认 `training=True`,如果不显式传,**推理阶段也会继续随机丢弃**,这是第 2 节提到过的"functional 版本不会自动感知 `model.eval()`"这个坑在 Dropout 上的具体体现,而且比 ReLU 的场景更危险——ReLU 在两种模式下行为本来就一样,Dropout 不一样,这个疏漏会让模型在推理时行为随机、结果不可复现。
- 以为 Dropout 的丢弃比例在小 batch 上也严格精确等于 `p`——伯努利采样是随机的,大数定律保证的是"样本量足够大时,比例趋近于 p",单次小规模前向,实际丢弃比例围绕 `p` 有统计波动,这也是本节代码用 20 万个元素而不是 20 个元素来验证比例的原因。

---

## 7. `Embedding` backward 是稀疏梯度——一个大词表矩阵,一个 batch 只碰到其中一小撮行

**是什么:**
```python
nn.Embedding(num_embeddings, embedding_dim, sparse=False)
# weight: (num_embeddings, embedding_dim) —— 词表大小 x 词向量维度
# forward(token_ids): 按下标"查表",等价于 one_hot(token_ids) @ weight,但绝不会真的构造 one_hot 矩阵
```

**一句话:** 词表动辄几万到几十万行,一个训练 batch 通常只覆盖其中几百到几千个不同的 token——反向传播时,`weight` 矩阵里**没被这个 batch 用到的行,梯度严格为 0**,这是"稀疏梯度"最直观、最容易现场验证的例子。

**底层机制/为什么这样设计:**

`Embedding` 的前向可以理解成 `output = one_hot(token_ids) @ weight`——用 token id 对应的 one-hot 行向量去"选出" `weight` 里的对应行。反向传播时:

```
dL/dweight[t,:] = Σ_{位置k, 满足token_ids[k]==t} dL/doutput[k,:]
```

也就是说,某一行 `t` 的梯度,只由**这个 batch 里所有等于 `t` 的 token 位置**贡献,累加求和——如果词表里某个 token `t` 在这个 batch 里一次都没出现,它对应的那一行梯度就是**精确的 0**,不是"很小接近0",是数学上严格等于 0(因为 `Σ` 里没有任何一项对应它)。

这个 `Σ` 求和还带来第二个容易被问到的细节:**如果同一个 token 在 batch 里出现多次**(常见,比如高频词"的"、"是"),它对应行的梯度是**每次出现贡献的梯度之和**,不是只取一次、也不是取平均——这一点已用代码验证:token `37` 在 batch 里出现 2 次,它那一行的梯度精确等于出现 1 次时的**2 倍**。

**两种"稀疏"的表现形式(容易混淆,面试常问区别):**
- `sparse=False`(默认):`.grad` 依然是一个**普通的 dense tensor**,形状和 `weight` 完全一样 `(num_embeddings, embedding_dim)`——只是这个 dense tensor 里,绝大多数行恰好全是 0。"稀疏"体现在**数值内容**上,不体现在**存储结构**上。
- `sparse=True`:`.grad` 是一个真正的 `torch.sparse_coo_tensor`——只存储非零行的**下标+数值**,不存储那些全零的行,"稀疏"体现在**存储结构**本身,词表越大、batch 触碰到的比例越小,省的内存和计算越可观。

**可运行例子:**
```python
import torch, torch.nn as nn
torch.manual_seed(0)
vocab_size, embed_dim = 10_000, 8
emb = nn.Embedding(vocab_size, embed_dim)          # sparse=False (默认)

token_ids = torch.tensor([37, 37, 512, 9999, 1, 512])   # 6个位置,只碰到4个不同token;37和512各出现2次
out = emb(token_ids)
out.sum().backward()

grad = emb.weight.grad
assert grad.is_sparse is False and grad.shape == (vocab_size, embed_dim)   # 默认:dense tensor
nonzero_rows = (grad.abs().sum(dim=1) != 0).nonzero().flatten()
assert set(nonzero_rows.tolist()) == {37, 512, 9999, 1}                     # 只有4行非零
assert (grad.abs().sum(dim=1) == 0).sum().item() == vocab_size - 4          # 其余9996行严格全零

assert torch.allclose(grad[37], torch.full((embed_dim,), 2.0))    # 出现2次 -> 梯度是1次的2倍(累加)
assert torch.allclose(grad[1],  torch.full((embed_dim,), 1.0))    # 只出现1次

# ---- sparse=True: 梯度是真正的 sparse_coo_tensor ----
emb_sparse = nn.Embedding(vocab_size, embed_dim, sparse=True)
emb_sparse(token_ids).sum().backward()
grad_s = emb_sparse.weight.grad
assert grad_s.is_sparse is True
coalesced = grad_s.coalesce()                          # 去重(相同下标的梯度合并求和)
assert coalesced._indices().shape[1] == 4                # 恰好4个不同的行有梯度条目

# ---- 延伸: Adam 不支持稀疏梯度,SparseAdam 支持 ----
opt = torch.optim.Adam(emb_sparse.parameters(), lr=1e-3)
try:
    opt.step()
    raised = False
except RuntimeError:
    raised = True
assert raised    # "Adam does not support sparse gradients, please consider SparseAdam instead"

opt2 = torch.optim.SparseAdam(list(emb_sparse.parameters()), lr=1e-3)
opt2.step()        # 不报错
```

实测:9996 / 10000 行严格全零(只有 4 个 token 被用到);`torch.optim.Adam` 对稀疏梯度直接抛出 `RuntimeError: Adam does not support sparse gradients, please consider SparseAdam instead`;换成 `torch.optim.SparseAdam` 处理同一个稀疏梯度,`.step()` 正常执行不报错。

**AI 研究场景:** 大词表场景(几十万甚至百万级 token 的多语言/多模态模型)下,`sparse=True` 配合 `SparseAdam`/`SGD` 能显著降低 embedding 层反向传播和优化器状态更新的计算量与显存占用——不需要对整张词表矩阵做梯度更新,只需要处理这个 batch 真正碰到的那一小撮行。但要注意:`sparse=True` 不是没有代价的万能开关,它要求优化器支持稀疏梯度(`Adam` 不支持,`SparseAdam`/`SGD` 支持),且稀疏张量的某些操作(和其他 dense 梯度混合、某些正则化项)实现起来比 dense 张量更麻烦,工程上是否值得切换到 `sparse=True`,要看词表规模和 batch 覆盖率的实际比例。

**面试怎么问 + 追问链:**
- **Q:** "`Embedding` 层的反向传播有什么特点?" —— 期望答出"梯度是稀疏的,一个 batch 只会让词表里被用到的那些行有非零梯度"。
- **追问 1:** "怎么现场验证这个说法,而不是嘴上说说?" —— 期望能想到检查 `.grad` 里有多少行是全零(或者直接看 `sparse=True` 时 `.grad.is_sparse`),这是本节代码的核心验证方式。
- **追问 2(容易漏答的细节):** "如果同一个 token 在一个 batch 里出现了 5 次,它对应的那一行梯度是什么?" —— 期望答出"5 次出现各自贡献的梯度**求和**",而不是"只算一次"或"取平均"——这个求和的来源就是本节推导里那个 `Σ_{位置k}`。
- **追问 3:** "`sparse=False`(默认)的梯度和 `sparse=True` 的梯度有什么本质区别?两者数值内容一样吗?" —— 期望答出"数值内容完全一样(非零的行、每行的值都相同),区别只是**存储格式**——dense 版本把那些 0 也老老实实存下来,sparse 版本只存非零条目",而不是以为两者算出的梯度数值本身不同。
- **追问 4(工程深挖):** "为什么 `torch.optim.Adam` 不支持稀疏梯度,需要专门的 `SparseAdam`?" —— 这是一个开放问题,期望候选人能想到"Adam 需要维护逐参数的一阶/二阶矩估计,如果梯度是稀疏的,动量项要不要对'这一步没出现梯度的参数'也做衰减,是一个语义选择——`SparseAdam` 选择只更新真正有梯度的那些行的动量状态,不去衰减未触达行的动量",不强求候选人复述精确实现,能说出"这是个语义设计问题,不是随便就能兼容"就已经体现了理解深度。

**常见坑:**
- 以为 `sparse=True` 会让训练**必然**更快——如果词表本身不大,或者 batch 覆盖率本来就很高(比如小词表语言模型),稀疏张量额外的索引开销可能得不偿失,是否使用要基于实际 profiling,不是无脑开关。
- 混淆"梯度稀疏"和"参数稀疏(剪枝)"——本节讲的是**这一次反向传播**里,哪些行的梯度非零(和具体这个 batch 用到的 token 有关,换一个 batch 非零的行就变了);"参数剪枝"是让**权重本身**永久性地变成 0/被移除,是两个完全不同的概念,不要因为都用了"稀疏"这个词就混为一谈。
- 忘记 `sparse=True` 之后,梯度裁剪(`clip_grad_norm_`,[06-optimizer-internals.md](06-optimizer-internals.md) 会讲)等一些常见训练技巧对稀疏张量的支持可能不完整,组合使用前需要现场验证,不能想当然照搬 dense 场景下的代码。

---

## 8. `MultiheadAttention` 内部 QKV 拆分与合并逻辑

**是什么:**
```python
nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
# in_proj_weight: (3*embed_dim, embed_dim)  -- Wq, Wk, Wv 拼在一起的大矩阵
# out_proj:        nn.Linear(embed_dim, embed_dim)  -- 多头拼接后的输出投影
```

**一句话:** 多头注意力不是"真的有多个独立的小模型并行跑"——它是把**同一个** `embed_dim` 维的向量,沿特征维**切成** `num_heads` 份,每一份独立算一次标准的 scaled dot-product attention,算完再**拼回**原来的 `embed_dim` 维,全程只是 reshape/transpose 的形状游戏,没有任何"真正的并行子网络"。

**在讲拆分之前:"标准的 scaled dot-product attention" 本身是什么、为什么这样设计(从零建立)**

> 这一段是本仓库多个系列(`long-context-deep-dive`、`kernel-gpu-deep-dive`、`alignment-algorithms-deep-dive` 等)反复依赖、但此前从未真正从零建立过的地基——2026-07-23 一次全仓库新手可读性排查发现这个缺口后补写于此,后续系列统一交叉引用这里,不再各自重复讲一遍。

**为什么需要这样一种机制。** 一个句子/序列里,某个 token 的含义经常要靠其他 token 才能确定(比如"它"到底指代前面哪个名词,取决于上下文)。我们想要一种机制:让每个 token 在计算自己的新表示时,能够"参考"序列里其他 token 的信息,而且**参考谁、参考多少,应该是模型学出来的,而不是提前写死的规则**(比如"只看前一个词"这种写死的规则,表达能力显然不够)。self-attention 就是解决这个问题的具体设计。

**类比:把它想成一次"模糊版"的字典查询。** Python 字典 `d[key]` 要求 `key` 和字典里存的键**精确匹配**,一旦找到就返回**唯一**对应的 value。Attention 做的是这件事的"模糊版本"——不要求精确匹配,而是:
1. 拿你手上的 **Query**(你想找什么),去和序列里**每一个** token 提供的 **Key**(它能提供什么线索)分别算一个"相似度分数"(不是 0/1 的匹不匹配,是一个连续的数字)。
2. 把这一串相似度分数变成一组**权重**(全部非负、加起来等于 1)。
3. 不是只取"最相似"的那一个,而是按这组权重,对每个 token 提供的 **Value**(它实际携带的内容)做**加权平均**,得到最终结果。

也就是说,普通字典查询是"精确匹配、只取一个";attention 是"按相似度打分、软性地融合所有人的贡献"。**Q(Query)、K(Key)、V(Value) 这三个名字就是直接照搬字典查询的术语来的**,只是从"精确匹配"换成了"可学习的相似度加权"。

**为什么用点积衡量"相似度"。** Q 和 K 都是向量,线性代数里两个向量的点积(`Σ q_i·k_i`)在向量长度固定的情况下,方向越接近点积越大、方向正交点积趋于 0、方向相反点积为负——点积本身就是一种现成的"对齐程度"度量,不需要发明新的相似度函数,这也是为什么这套机制被称为 **dot-product** attention(还存在用一个小型神经网络算相似度的"additive attention"等变体,但点积版本因为能直接用矩阵乘法批量算、在 GPU 上极快,是当前的主流选择)。

**为什么要 softmax,而不是直接用相似度分数加权。** 相似度分数(点积原始结果)可以是任意实数,直接拿来加权平均没有意义(权重可能是负的,或者加起来不等于 1)。`softmax` 把一组任意实数变成一组"非负、加起来等于 1"的权重,而且保留了"分数越大权重越大"的相对顺序——这正是"加权平均"这个操作需要的输入形式。

**为什么要除以 `sqrt(维度)`(这里先建立直觉,数值稳定性话题详见 [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md)):** 点积是多个乘积项相加,参与求和的维度越高,点积结果的数值量级往往越大;数值量级一旦过大,`softmax` 会趋于"赢者通吃"(几乎所有权重都挤到分数最大的那一项,其余全部接近 0),模型会失去"软性地综合多个来源"这个本来想要的效果,梯度也会跟着变得很小、难以训练。除以 `sqrt(维度)` 是一个简单有效的缩放,把点积的数值量级重新拉回一个 `softmax` 表现良好的范围。

**一个具体到能验证的最小例子(3 个 token,先不引入投影矩阵,直接看"点积衡量相似度"这件事本身):**

```python
import torch, torch.nn.functional as F, math
torch.manual_seed(0)

# 3 个 token 的玩具序列,每个 token 用一个 4 维向量表示(手工构造,便于验证直觉,不是真实embedding)
x = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],    # token 0
    [0.0, 0.0, 1.0, 1.0],    # token 1 —— 方向和 token 0 正交("不相关")
    [0.9, 1.1, 0.1, -0.1],   # token 2 —— 方向和 token 0 接近("相关")
])

# 先看最朴素的情形:Q=K=V=token自己(相当于 Wq=Wk=Wv=单位矩阵这个特例),
# 只为了纯粹展示"点积大小 -> 相似度 -> attention权重大小"这条因果链
scores = x @ x.transpose(0, 1)          # scores[i,j] = token i 的 query 和 token j 的 key 的点积
weights = F.softmax(scores, dim=-1)

# 验证直觉:token0 对"和自己方向接近"的 token2 的权重,应该明显高于对"正交"的 token1 的权重
assert weights[0, 2].item() > weights[0, 1].item()
assert abs(weights[0].sum().item() - 1.0) < 1e-6      # 权重确实加起来等于1

# 加上可学习的投影矩阵 Wq/Wk/Wv 和 1/sqrt(d) 缩放的完整版本,验证权重依然合法(非负、和为1)
d = 4
Wq, Wk, Wv = torch.randn(4, d) * 0.5, torch.randn(4, d) * 0.5, torch.randn(4, d) * 0.5
Q, K, V = x @ Wq, x @ Wk, x @ Wv
scores_full = (Q @ K.transpose(0, 1)) / math.sqrt(d)
weights_full = F.softmax(scores_full, dim=-1)
assert torch.allclose(weights_full.sum(dim=-1), torch.ones(3), atol=1e-6)

output = weights @ x     # 用权重对 V(这里 V=x)做加权平均,得到每个token的新表示
```
实测 `weights[0] = [0.468, 0.063, 0.468]`——token0 给自己和 token2 的权重接近且明显更大,给 token1 的权重最小,和"点积大小反映方向相似度"这条直觉完全吻合。

**用图示把这条链路串起来(以 token 0 为例):**
```
token 0 embedding   token 1 embedding   token 2 embedding
       │                    │                    │
       ▼                    ▼                    ▼
   [Q0 K0 V0]           [Q1 K1 V1]           [Q2 K2 V2]     ← 每个 token 各自生成一份 Q/K/V

token 0 的 Query 去和每一个 token 的 Key 算点积(相似度):

   Q0·K0        Q0·K1        Q0·K2
     │            │            │
     ▼            ▼            ▼
  [ 2.00         0.00         2.00 ]              ← scores(还没归一化)
     │
     ▼  softmax(转成"非负、加起来=1"的权重)
  [ 0.47         0.06         0.47 ]              ← weights:更关注自己和token2,几乎不理token1
     │
     ▼  按这组权重对 V0,V1,V2 做加权平均
  output_0 = 0.47·V0 + 0.06·V1 + 0.47·V2
```

有了这条从零建立的地基,下面第 8 节剩余部分(拆分成多个 head、再合并回去)要处理的问题就变成了:上面这一整套"标准 scaled dot-product attention"计算,`nn.MultiheadAttention` 在工程实现上是怎么被拆到多个更小的子空间里**并行**算多份、再合并回来的——这是一个纯粹的形状/工程问题,和"attention 本身为什么这样设计"是两个不同层次的问题。

**底层机制/为什么这样设计(拆分与合并的工程细节):**

设 `embed_dim=E`,`num_heads=h`,`head_dim=E/h`(要求整除,`nn.MultiheadAttention` 在构造时就会检查,除不尽直接报 `AssertionError`)。

**第一步,QKV 投影。** PyTorch 把 `Wq`、`Wk`、`Wv` 三个 `(E,E)` 矩阵**拼接**成一个 `(3E,E)` 的 `in_proj_weight`(自注意力场景下三者维度相同,可以合并成一次矩阵乘法,再切开,比算三次独立的 `F.linear` 更省一次 kernel launch 开销):
```
qkv = F.linear(x, in_proj_weight, in_proj_bias)     # (batch, seq, 3E)
q, k, v = qkv.split(E, dim=-1)                        # 按最后一维切成三份,各 (batch, seq, E)
```

**第二步,拆头。** 把 `E` 这一维重新解释成 `(num_heads, head_dim)` 两维,再把 `num_heads` 挪到 batch 维旁边,这样每个 head 的 attention 计算可以通过一次**批量矩阵乘法**(batched matmul)一起做完,不需要写 Python 循环遍历每个 head:
```
q.view(batch, seq, num_heads, head_dim).transpose(1, 2)   # -> (batch, num_heads, seq, head_dim)
```
这一步纯粹是形状重排(呼应第 3 节和 [01-tensor-memory-model.md](01-tensor-memory-model.md) 反复强调的 stride 机制——`view`/`transpose` 本身零拷贝或接近零拷贝),没有任何计算,`E` 维里的数字被重新分组解读成 `(head,head_dim)`,并不是重新计算出来的。

**第三步,每个 head 独立做标准 attention。** 因为多出来的 `num_heads` 维被放在了 batch 维旁边,`q @ k.transpose(-2,-1)` 这种矩阵乘法会自动在 `(batch,num_heads)` 这两维上广播/批量执行,每个 head 互不干扰:
```
scores = q @ k.transpose(-2,-1) / sqrt(head_dim)     # (batch, num_heads, seq, seq)
attn = softmax(scores, dim=-1)
out = attn @ v                                          # (batch, num_heads, seq, head_dim)
```

**第四步,合并头+输出投影。** 把 `(num_heads,head_dim)` 这两维合并回 `E`,做一次输出投影:
```
out.transpose(1, 2).reshape(batch, seq, E)   # 合并回 embed_dim(第2节提到的 contiguous+view 套路)
out_proj(merged)                               # 最后一次 Linear,让不同 head 的信息重新混合
```

**为什么要"切成多个头"而不是直接在完整的 `E` 维上做一次 attention?** 直接在 `E` 维上做一次 attention,`softmax` 只能学出**一种**"关注模式"(一组注意力权重分布)。切成 `h` 个头之后,每个头在自己的 `head_dim` 子空间里独立算一套注意力权重,相当于让模型同时维护 `h` 组不同的"关注模式"(有的头可能学会关注局部临近 token,有的头学会关注句法上的主谓关系),这是"多头"这个名字真正的含义——**头之间共享输入,但注意力权重完全独立**。最后的 `out_proj` 再把这些不同视角的信息重新线性组合。

**可运行例子(手写拆分/合并逻辑,和 `nn.MultiheadAttention` 用同一组权重对比):**

```python
import math, torch, torch.nn.functional as F, torch.nn as nn
torch.manual_seed(0)
embed_dim, num_heads = 8, 2
head_dim = embed_dim // num_heads
batch, seq_len = 3, 5

mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)
mha.eval()
x = torch.randn(batch, seq_len, embed_dim)

ref_out, ref_weights = mha(x, x, x, need_weights=True, average_attn_weights=False)

# ---- 手写:用 mha 内部同一组权重,复现拆分/attention/合并全过程 ----
Wq, Wk, Wv = mha.in_proj_weight.split(embed_dim, dim=0)
bq, bk, bv = mha.in_proj_bias.split(embed_dim, dim=0)
q, k, v = F.linear(x, Wq, bq), F.linear(x, Wk, bk), F.linear(x, Wv, bv)

def split_heads(t):
    b, s, e = t.shape
    return t.view(b, s, num_heads, head_dim).transpose(1, 2)     # (batch, heads, seq, head_dim)

qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
scores = qh @ kh.transpose(-2, -1) / math.sqrt(head_dim)
attn = F.softmax(scores, dim=-1)
out_heads = attn @ vh

def merge_heads(t):
    b, h, s, d = t.shape
    return t.transpose(1, 2).contiguous().view(b, s, h * d)

manual_out = F.linear(merge_heads(out_heads), mha.out_proj.weight, mha.out_proj.bias)

assert torch.allclose(manual_out, ref_out, atol=1e-5)      # 实测最大误差 ~1.2e-7
assert torch.allclose(attn, ref_weights, atol=1e-5)          # 每个头的注意力权重也完全一致
```

实测最大误差 `1.19e-07`(输出和逐头注意力权重都对得上,float32 精度范围内)。

**"如果输入维度变了会怎样"这个面试追问的现成答案(现场验证):** 固定 `embed_dim=8`、`seq_len=5`,把 `num_heads` 分别取 `1,2,4,8`(对应 `head_dim` 分别是 `8,4,2,1`),现场验证:**输出形状永远是 `(batch,seq,embed_dim)`,不随 `num_heads` 变化**;真正变化的是 `head_dim`(每个头能看到的子空间维度)和 attention 权重张量的头数这一维 `(batch,num_heads,seq,seq)`。这个不变性是"多头"设计的一个重要工程性质——换 `num_heads` 不需要改下游任何接口。

**AI 研究场景:** 拆头/合并头这套 `view+transpose` 的写法,是几乎所有 Transformer 实现(from-scratch 版本、HuggingFace `transformers`、各类论文复现代码)里高频出现的样板代码,读懂这四步就能看懂任何一份多头注意力实现,不管它变量名怎么起。理解"多头共享输入、注意力权重独立"这一点,也是理解后续像 Grouped-Query Attention / Multi-Query Attention(多个 query 头共享同一组 K/V,用于推理加速)这类变体的基础——那些变体本质上是在"每个头独立 K/V"和"所有头共享 K/V"这两个极端之间做取舍。

**面试怎么问 + 追问链:**
- **Q:** "手写一下多头注意力里,`embed_dim` 是怎么拆成多个头,又是怎么合并回去的?" —— 期望完整说出 `view` 拆分+`transpose` 挪位置+每头独立算 attention+`transpose`+`view` 合并回去 这四步,最好能提到这全程是"reshape 游戏,没有任何'切开的小网络'"。
- **追问 1(考察是否真的算过,不是背图):** "如果 `embed_dim=512,num_heads=8`,每个头的 `head_dim` 是多少?这个数字在哪一步的张量形状里出现?" —— `head_dim=64`,出现在 `view(batch,seq,8,64)` 这一步,以及 `scores=q@k.T/sqrt(64)` 的缩放因子里。
- **追问 2(深挖 scaling factor):** "attention 公式里为什么要除以 `sqrt(head_dim)`,而不是除以 `sqrt(embed_dim)`?" —— 期望答出"缩放是为了控制 `q·k` 点积的方差,点积是在 `head_dim` 维的子空间里做的(每个头各自算自己的 `q@k.T`),用于缩放的应该是**参与点积求和的维度数**,也就是 `head_dim`,不是整个 `embed_dim`"——这是一个经常被问出来判断候选人是否真的理解拆头之后维度含义变化的问题(这个缩放因子背后的方差论证会在 [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md) 附近的数值稳定性话题里进一步展开)。
- **追问 3:** "`embed_dim` 必须能被 `num_heads` 整除吗?为什么?" —— 期望答出"是的,`head_dim=embed_dim/num_heads` 必须是整数,`nn.MultiheadAttention` 构造时会用 assert 检查,除不尽直接报错(实测报错信息就是 `embed_dim must be divisible by num_heads`),因为拆分逻辑要求每个头拿到的子空间维度相等"。
- **追问 4(开放,考察知识广度):** "如果 `num_heads=1`,多头注意力退化成什么?" —— 退化成标准的单头 scaled dot-product attention,`head_dim==embed_dim`,这是一个很好的"用极端情况检验自己理解是否自洽"的思维方式,值得在面试里主动提及。

**常见坑:**
- 拆头之后忘记 `transpose(1,2)` 把 `num_heads` 挪到 batch 维旁边,直接对 `(batch,seq,num_heads,head_dim)` 形状做矩阵乘法——批量矩阵乘法默认在**最后两维**上做矩阵乘、其余维度当 batch 广播,`num_heads` 留在中间会导致矩阵乘法在错误的维度上进行,这是本节最容易在手写实现时犯的错误,表现为形状能凑出来但数值完全不对。
- 合并头时忘记 `.contiguous()`——`transpose` 之后张量不再连续(第 01 篇第 3/4 节讲过这个机制),直接对不连续张量调用 `.view()` 会报错,必须先 `.contiguous()` 或者改用 `.reshape()`。
- 以为"多头"意味着参数量或计算量比单头 attention 大很多——实际上总参数量和总计算量基本不变(`embed_dim` 不变,只是拆成更小的子空间并行处理),多头的代价主要是**多了几次形状重排的开销**,不是参数量的成倍增长,这也是为什么"增加 `num_heads` 数量"在工程上是一个相对"便宜"的架构调整。

---

## 9. 残差连接(`x + F(x)`)对梯度传播的影响

**是什么:**
```python
def forward(self, x):
    return x + self.block(x)     # 而不是 return self.block(x)
```
输出是"输入本身"和"输入经过一个子网络 `F` 变换后的结果"两者相加,而不是单纯地把 `x` 送进 `F` 就完事。

**一句话:** 加法节点在反向传播时,会把上游梯度**原封不动**地同时传给两个分支——残差连接因此给梯度提供了一条可以**跳过 `F(x)` 直接传回输入**的"高速通道",不管 `F` 内部的梯度多小,这条通道传回去的梯度都至少有"原样复制"这一份,这是残差连接缓解深层网络梯度消失的核心机制。

**底层机制/为什么这样设计:**

先看最基础的事实:加法节点 `c=a+b` 的反向传播,`dc/da=1`,`dc/db=1`——上游梯度 `dL/dc` 原样传给 `a` 和 `b` 两个分支,不缩放、不衰减(这是加法求导的必然结果,`∂(a+b)/∂a=1` 是个常数,不依赖 `a`、`b` 的具体值)。

把这个事实应用到 `y=x+F(x)`:
```
dy/dx = 1 + dF(x)/dx
```
`dy/dx` 由两部分组成——**恒为 1 的直接项**,和 `F` 自身的局部梯度项。哪怕 `F` 是一个梯度很小的子网络(比如内部权重初始化得很小,或者用了 `tanh`/`sigmoid` 这类容易饱和的激活函数,局部梯度天然 <1),`dy/dx` 也不会小于这个恒为 1 的直接项太多——**残差连接把"最坏情况下梯度衰减到 0"变成了"最坏情况下梯度也还有一份原样保留"**。

**堆叠很多层之后,这个差异会指数级放大。** 假设堆叠 `L` 层,每层局部梯度的量级是 `g`(`g<1` 代表梯度衰减):
- **没有残差**(纯前馈堆叠):总梯度 ≈ `g^L`——`g<1` 时随层数 `L` **指数衰减**,`L` 稍微大一点就会衰减到浮点数精度都无法区分的量级(下面代码现场验证,60 层之后梯度分量已跌入 float32 次正规数区间)。
- **有残差**:每一层的局部梯度是 `(1+g)` 量级而不是 `g`,总梯度里包含"全程都走恒等分支"这一条路径,贡献恰好是 `1^L=1`,不随层数衰减,再加上其他"部分走 `F`、部分走恒等"的路径贡献——总梯度不会消失到 0,这是残差网络能训练几百层而不退化的核心原因。

**可运行例子:**

```python
import torch, torch.nn as nn
torch.manual_seed(0)

# 最小验证:加法节点上游梯度原样传给两个分支
a = torch.tensor(2.0, requires_grad=True); b = torch.tensor(3.0, requires_grad=True)
(a + b).backward(torch.tensor(7.0))
assert a.grad.item() == 7.0 and b.grad.item() == 7.0     # 精确相等,不是近似

# y = x + F(x),即使 F 的局部梯度极小,dy/dx 也被恒等项"兜底"
x = torch.tensor(1.5, requires_grad=True)
y = x + torch.tanh(x * 0.01)          # d(tanh(0.01x))/dx ≈ 0.01,非常小
y.backward()
assert abs(x.grad.item() - 1.0) < 0.02   # dy/dx ≈ 1 + 一个小量,主导项是恒等分支的1

# 深层堆叠对比:60层的 plain 网络 vs 60层的残差网络,梯度传回第一层输入的量级
hidden_dim, num_layers = 16, 60

def make_layers(std=0.05):
    layers = nn.ModuleList()
    for _ in range(num_layers):
        lin = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(lin.weight, std=std)   # 故意用很小的初始化,模拟局部梯度很小的情形
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    return layers

torch.manual_seed(0); plain_layers = make_layers()
torch.manual_seed(0); res_layers = make_layers()      # 同一组初始权重,公平对比

h0_plain = torch.randn(hidden_dim, requires_grad=True)
h0_res = h0_plain.detach().clone().requires_grad_(True)

h = h0_plain
for lin in plain_layers:
    h = torch.tanh(lin(h))                # 没有残差
h.sum().backward()

h = h0_res
for lin in res_layers:
    h = h + torch.tanh(lin(h))            # 有残差
h.sum().backward()

assert h0_plain.grad.norm().item() < 1e-3     # plain: 梯度范数几乎完全消失
assert h0_res.grad.norm().item() > 0.1          # residual: 梯度依然是 O(1) 量级
```

实测:60 层之后,plain 网络传回输入的梯度分量本身已经跌到 float32 的**次正规数(subnormal)**范围——量级在 `1e-44` 左右(比如 `1.121e-44, 8.408e-45, ...`),不是数学上精确的 0,但已经小到基本等同于"没有梯度信号";对它们计算 L2 范数时(范数计算要先平方,`(1e-44)^2≈1e-88`,远低于 float32 能表示的最小正数),平方这一步直接下溢成 0.0,所以 `h0_plain.grad.norm()` 报告出来的结果精确是 `0.0`。同样 60 层、同样的初始权重,只是每层多了个 `+` 号的残差网络,传回输入的梯度范数是 `5.503`——两者差了几十个数量级,残差版本的梯度完全在正常可用的范围内。这个对比直接展示了残差连接到底在多大程度上改变了深层网络的梯度传播行为。

**AI 研究场景:** 残差连接是从 ResNet(图像分类,2015)开始,到几乎所有现代深层架构(Transformer 的每个 attention block/FFN block 前后都套了残差)的标配设计,原因就是本节推导的这条"梯度高速通道"——没有它,单纯堆叠更多层往往不会让模型更强,反而会因为梯度消失(或者说,深层部分根本学不动)导致训练更困难、效果更差(这在原始 ResNet 论文里被称为"退化问题",不是过拟合,是**优化**本身变难了)。理解这一点后,再看 Transformer 的架构图里那些反复出现的"Add & Norm",就能明白"Add"这一步不是随便加的,是整套深层网络能训练起来的必要条件之一。

**面试怎么问 + 追问链:**
- **Q:** "残差连接为什么能缓解梯度消失?" —— 期望完整推出"加法节点梯度原样传递 + `dy/dx=1+dF/dx` + 恒等项不随深度衰减"这条逻辑链,不是只说"因为有一条捷径"这种模糊表述。
- **追问 1(考察是否真的做了实验,而不是背结论):** "能不能现场估算一下,60 层的普通前馈网络和 60 层的残差网络,梯度量级能差多少?" —— 期望能提出"没有残差近似 `g^L` 指数衰减,有残差至少保留 `1^L=1` 这条路径"的量级估算思路,即使数字记不准,方法论要对。
- **追问 2(反问,考察是否会盲从):** "残差连接是不是让梯度消失问题被'完全解决'了?" —— 期望答出"不是完全解决,只是提供了一条不会衰减的路径,`F(x)` 分支自身内部依然可能有梯度消失问题(比如 `F` 内部本身很深);另外配合 LayerNorm、合适的初始化,才是深层网络能稳定训练的完整答案,残差只是其中很重要的一块"。
- **追问 3(拓展):** "如果 `F(x)` 的输出和 `x` 形状不一样(比如通道数变了),残差连接还能直接相加吗?" —— 期望答出"不能直接相加,需要一个转换(ResNet 里叫'projection shortcut',常见做法是用一个 `1x1` 卷积或者 Linear 把 `x` 投影到和 `F(x)` 一样的形状,再相加)",这是"如果输入维度变了会怎样"这类追问的具体例子。
- **追问 4(深挖到数值层面):** "反向传播时,残差连接的梯度公式 `dy/dx=1+dF/dx` 在什么条件下会失效或者需要重新考虑?" —— 一个开放性的好答案方向是"如果 `x` 和 `F(x)` 之间还有其他非加法的耦合(比如某些变体用 `x*F(x)` 或者门控残差 `α*x+(1-α)*F(x)`),这条'恒等项系数为1'的简单结论就不再成立,需要重新推导——这提醒我们'残差连接'不是只有 `x+F(x)` 一种形式,具体公式必须回到定义重新求导,不能套用记忆中的结论"。

**常见坑:**
- 把"残差连接解决了梯度消失"理解成"残差连接让所有梯度问题都不存在了"——只解决了"跨层传播时被连续多次相乘导致衰减"这一种机制,不是万能药(参见追问 2)。
- 忘记残差连接要求两个分支形状一致,直接对形状不匹配的 `x` 和 `F(x)` 相加,PyTorch 广播机制有时会"救"一部分形状不匹配的情况(广播把小张量悄悄扩展成大张量),产生的结果在语义上是错的,但不一定报错,排查起来容易被忽略。
- 只在"最外层"加残差,却以为"网络越深,残差的效果越明显"——残差连接必须**在每一层(或每个 block)都加**,才能提供逐层不衰减的梯度路径;只在整个网络的输入输出之间加一次残差(相当于只有一层残差 block),对中间几十层的梯度传播没有帮助,深层部分该消失还是会消失。

---

## 10. ReLU 死亡神经元问题(dead ReLU)

**是什么:** 一个用 ReLU 激活的神经元,如果它的**输入(pre-activation,也就是激活函数之前的线性组合结果)对训练集里几乎所有样本都恒为负**,那么无论输入怎么变,这个神经元的输出恒为 0,并且——这是问题的核心——它的梯度也恒为 0,意味着这个神经元的权重**再也不会被更新**,永久停止学习。

**一句话:** ReLU 在负半轴上导数严格为 0,一旦某个神经元"陷进"负半轴出不来,它的权重梯度就被这个 0 掐断,不管后面喂多少数据、训练多少步,它都不会再变化——"死"是永久性的,不是"暂时学得慢"。

**底层机制/为什么这样设计:**

ReLU 的定义和导数:
```
ReLU(z) = max(0, z)
ReLU'(z) = 1  (z > 0)
ReLU'(z) = 0  (z < 0)
ReLU'(z) = 未定义,PyTorch 约定取 0  (z = 0,次梯度的一种选择)
```

对一个神经元 `z=w·x+b`,`y=ReLU(z)`,链式法则:
```
dL/dw = dL/dy * dy/dz * dz/dw = dL/dy * ReLU'(z) * x
dL/db = dL/dy * dy/dz * dz/db = dL/dy * ReLU'(z) * 1
```
只要 `z<0`,`ReLU'(z)=0`,不管 `dL/dy` 是多少、`x` 是多少,这两个梯度都被这个 0 因子"乘没了",精确地等于 0——不是"很小",是数学上严格的 0(下面代码用 `torch.equal` 而不是 `allclose` 验证,因为这就该是位级精确的 0)。

**"死"为什么是永久性的?** 因为优化器的更新公式是 `w ← w - lr*dL/dw`——如果 `dL/dw` 恒为 0,`w` 就永远不会变,`b` 同理。而 `w`、`b` 不变,意味着下一步、下下一步……只要输入的分布没有剧烈变化,`z=w·x+b` 依然大概率是负的,`ReLU'(z)` 依然是 0,梯度依然是 0——这是一个**自我锁死**的循环,没有任何机制能让它自己走出来(除非输入数据分布本身发生了让 `z` 变正的剧变,但这通常不会在正常训练里自动发生)。

**神经元实际是怎么"死掉"的?** 最常见的触发场景是**学习率过大**或**遇到异常/离群 batch**,导致某一步的梯度更新幅度过大,把 `bias`(或者权重整体)一把拽进"对几乎所有正常输入都产生负 pre-activation"的区域——下面代码复现了这个过程:一个原本健康的神经元,一次超大学习率的更新后,`bias` 被拽到 `-1600` 量级,此后对任意正常量级的输入,pre-activation 都稳定为负,神经元从此死亡。

**可运行例子:**

```python
import torch, torch.nn as nn
torch.manual_seed(0)
in_dim, batch = 4, 32

# ---- 构造一个已经"死"的神经元:bias 很负,对任意正常输入 pre-activation 都是负的 ----
neuron = nn.Linear(in_dim, 1)
with torch.no_grad():
    neuron.weight.fill_(0.1); neuron.bias.fill_(-10.0)

x = torch.randn(batch, in_dim)
pre = neuron(x)
assert (pre < 0).all()                        # 每个样本的 pre-activation 都是负的
y = torch.relu(pre)
assert torch.equal(y, torch.zeros_like(y))      # 输出恒为0

y.sum().backward()
assert torch.equal(neuron.weight.grad, torch.zeros_like(neuron.weight.grad))   # 梯度精确为0
assert torch.equal(neuron.bias.grad, torch.zeros_like(neuron.bias.grad))

# ---- 20步 SGD,权重/偏置一步都不会动 ----
opt = torch.optim.SGD(neuron.parameters(), lr=0.5)
w0, b0 = neuron.weight.clone(), neuron.bias.clone()
for _ in range(20):
    opt.zero_grad()
    torch.relu(neuron(torch.randn(batch, in_dim))).sum().backward()
    opt.step()
assert torch.equal(neuron.weight, w0) and torch.equal(neuron.bias, b0)   # 位级不变,永久死亡

# ---- 对比:健康神经元(bias≈0),部分样本 pre-activation 为正,梯度非零,会正常更新 ----
healthy = nn.Linear(in_dim, 1)
with torch.no_grad():
    healthy.weight.fill_(0.1); healthy.bias.fill_(0.0)
yh = torch.relu(healthy(x))
yh.sum().backward()
assert not torch.equal(healthy.weight.grad, torch.zeros_like(healthy.weight.grad))

# ---- 复现"怎么死的":健康神经元被一次超大学习率更新拽入永久死亡 ----
victim = nn.Linear(in_dim, 1)
with torch.no_grad():
    victim.weight.fill_(0.1); victim.bias.fill_(0.5)
victim.weight.requires_grad_(False)   # 冻结权重,只让 bias 变化,便于干净地观察这一个机制

x0 = torch.ones(batch, in_dim)         # 一个把更新推向极端的离群 batch
opt_v = torch.optim.SGD(victim.parameters(), lr=50.0)   # 故意用一个夸张的大学习率
torch.relu(victim(x0)).sum().backward()
opt_v.step()                            # 一次更新之后 bias 被打到 -1599.5

x1 = torch.randn(batch, in_dim)         # 换回正常量级的随机数据
pre1 = victim(x1)
assert (pre1 < 0).all()                  # 神经元已经死亡:任何正常输入都产生负 pre-activation
```

实测:死神经元的 `weight.grad`/`bias.grad` 就是字面意义的 `tensor([0.,0.,0.,0.])`/`tensor([0.])`;20 步 SGD(甚至用了 `lr=0.5` 这种较大学习率)之后权重、偏置**逐 bit 完全没变**;健康神经元(约 40.6% 的 batch 样本 `pre-activation>0`)的梯度是非零向量,一步 SGD 后权重确实发生了变化。复现"怎么死的"的实验里,一次 `lr=50` 的超大更新把 `bias` 从 `0.5` 打到 `-1599.5`,此后哪怕换回正常量级的随机输入,pre-activation 依然稳定在 `-1600` 附近,神经元从这一步之后再没救回来过。

**AI 研究场景:** "dead ReLU 比例"是诊断一个用 ReLU 的网络训练是否健康的常见指标——统计每一层里"对当前验证集,pre-activation 恒为负"的神经元占比,如果这个比例异常高(远超预期,比如超过 30%~50%),往往意味着学习率设置不当、初始化不合理、或者某次训练中遇到了破坏性的梯度爆炸。常见的缓解手段包括:降低学习率、更保守的权重初始化(比如 Kaiming 初始化专门为 ReLU 设计)、梯度裁剪(防止单步更新过大,[06-optimizer-internals.md](06-optimizer-internals.md) 会讲 `clip_grad_norm_`)、以及换用负半轴梯度不为 0 的激活函数(下一节的 GELU,或者 LeakyReLU/ELU 这类变体)。

**面试怎么问 + 追问链:**
- **Q:** "什么是 dead ReLU 问题?" —— 期望完整答出"pre-activation 恒为负 → 输出恒为0 → 梯度恒为0 → 权重永远不更新 → 死亡是永久性的"这条完整链条,不是只说"ReLU 会让一些神经元死掉"这种表层描述。
- **追问 1(考察是否理解"永久性"这个关键词):** "如果只是某一个 batch 让梯度变成了 0,下一个 batch 神经元还有机会'活过来'吗?" —— 期望答出"取决于权重有没有变化——如果这一步梯度是0,权重就不会更新,那么只要输入数据的整体分布没有剧变,下一个 batch 大概率还是负的,梯度大概率还是0,这是一个自我强化的锁死状态,不是'运气好下一步就能恢复'"。
- **追问 2(深挖成因):** "什么样的训练配置容易导致大量神经元死亡?" —— 期望提到"学习率过大"这个最常见原因,加分项是能提到"糟糕的权重初始化"、"某次异常的梯度爆炸/离群数据"也可能触发。
- **追问 3(解决方案,开放题):** "怎么缓解 dead ReLU 问题?" —— 期望至少提出两种:降低学习率/梯度裁剪(治标,减少触发概率)、换用负半轴梯度非零的激活函数如 LeakyReLU/ELU/GELU(治本,从机制上让神经元不会被"锁死",下一节详细讲 GELU 的这个性质)。
- **追问 4(反问深挖):** "死掉的神经元对模型最终效果一定是坏事吗?" —— 一个好的开放性回答方向是"少量死亡神经元某种程度上类似稀疏激活/隐式正则化,不一定致命;但大面积死亡(比如整个网络一半以上神经元都死了)意味着模型的**有效容量**大幅下降,这才是真正需要担心的信号——'有没有死神经元'和'死亡比例是否健康'是两个不同粒度的问题"。

**常见坑:**
- 以为 dead ReLU 是"梯度很小、学得很慢",实际上是**精确的 0**,是质变不是量变——如果面试时说"梯度会变得很小导致学得慢",会被追问"是很小还是精确为 0"当场问出破绽。
- 只在权重上找"死亡"的证据,忘记偏置 `bias` 同样会梯度为 0、同样不会更新——死亡神经元是"这一整个神经元(权重+偏置)都不再更新",不是只有权重的问题。
- 把"输出恒为 0"和"死亡"划等号——输出为 0 只是必要不充分条件,真正的判断标准是"对当前及未来可预见的输入分布,pre-activation 是否稳定为负",如果只是训练早期偶然某几步输出为 0,后续正常又变正,不算真正意义上的"死亡"。

---

## 11. GELU 等现代激活函数的梯度特性——为什么"处处光滑"很重要

**是什么:**
```python
nn.GELU()                        # 默认 approximate='none',精确公式
F.gelu(x, approximate='none')      # GELU(x) = x * Φ(x),Φ 是标准正态分布的累积分布函数(CDF)
```

**一句话:** ReLU 在 `x=0` 处导数突然从 0 跳到 1(处处分段线性,不是处处光滑),GELU 处处无限次可导——这个"光滑"特性直接对应上一节 dead ReLU 问题的一种缓解方式,也是现代 Transformer(BERT、GPT 系列的 FFN 层)普遍选择 GELU 而不是 ReLU 的重要原因之一。

**底层机制/为什么这样设计:**

GELU 的精确定义(不是近似):
```
GELU(x) = x * Φ(x)
Φ(x) = 标准正态分布的累积分布函数 = 0.5 * (1 + erf(x/√2))
```
直觉解释:`Φ(x)` 可以理解成"以标准正态分布为标准,这个输入 `x` 有多大概率应该被保留"的一个平滑权重(`x` 越大,`Φ(x)` 越接近 1,越应该整个保留;`x` 越负,`Φ(x)` 越接近 0,越应该被抑制)——ReLU 可以看成这个思想的一个**硬性、非黑即白**版本(`x>0` 全保留,`x<0` 全部归零,权重只能是 0 或 1);GELU 是它的**平滑、概率化**版本,权重 `Φ(x)` 可以取 0~1 之间的任意值,且随 `x` 连续变化。

**求导(乘积法则,标准正态分布的 PDF 记作 `φ`):**
```
d/dx [x * Φ(x)] = Φ(x) + x * φ(x)
φ(x) = (1/√(2π)) * exp(-x²/2)       # 标准正态分布的概率密度函数
```
这个导数公式里,`Φ(x)` 和 `φ(x)` 都是处处光滑(无限次可导)的函数,乘积、加和之后 `GELU'(x)` 自然也处处光滑——不存在 ReLU 在 `x=0` 处那种"左导数是 0、右导数是 1,两边对不上"的跳变。

**"光滑"具体体现在哪:**
- **ReLU**:`ReLU'(-0.0001)=0`,`ReLU'(+0.0001)=1`——一步之隔,导数直接跳变 `1`,是一个不连续点。
- **GELU**:`GELU'(-0.0001)≈0.49992`,`GELU'(+0.0001)≈0.50008`——同样跨过 0 点,导数只变化了 `0.00016`,几乎感觉不到跳变,`GELU'` 是一条连续光滑的曲线,不是分段函数。

**这个光滑性对训练稳定性的实际意义:**
1. **不存在"恰好卡在 0 点附近梯度突变"的病态情况。** ReLU 的输入如果频繁在 0 附近来回摆动(常见于训练早期,权重还没稳定下来),梯度会在 0 和 1 之间反复横跳,给优化过程带来额外的噪声;GELU 在 0 附近梯度变化平缓,不会有这种剧烈跳变。
2. **负半轴不是硬性 0,缓解(不是消除)dead-neuron 风险。** 上一节讲过,ReLU 在负半轴梯度**精确为 0**,一旦陷进去出不来。GELU 在负半轴梯度是**小但非零**的(甚至可能是负值,因为 GELU 在中等负值区域会先略微下探到 0 以下再回升到 0——它不是单调函数),意味着即使当前 pre-activation 是负的,这个神经元依然能接收到(哪怕很小的)梯度信号,有机会被"拉回"正常区域,不会像 ReLU 那样彻底锁死。

**可运行例子:**

```python
import math, torch, torch.nn.functional as F
torch.manual_seed(0)

# ---- 手推导数公式,和 autograd 数值对比 ----
x = torch.linspace(-4, 4, 17, requires_grad=True)
F.gelu(x).sum().backward()
autograd_grad = x.grad.clone()

with torch.no_grad():
    Phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    phi = torch.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
    manual_grad = Phi + x * phi

assert torch.allclose(manual_grad, autograd_grad, atol=1e-6)   # 实测最大误差 ~1.19e-7

# ---- 0点附近:ReLU 跳变 vs GELU 平滑 ----
def grad_at(fn, v):
    t = torch.tensor([v], requires_grad=True); fn(t).backward(); return t.grad.item()

eps = 1e-4
relu_jump = grad_at(F.relu, eps) - grad_at(F.relu, -eps)
gelu_jump = grad_at(F.gelu, eps) - grad_at(F.gelu, -eps)
assert relu_jump == 1.0            # ReLU: 精确跳变1
assert abs(gelu_jump) < 1e-3        # GELU: 几乎感觉不到跳变

# ---- 负半轴:ReLU 精确为0 vs GELU 小但非零(甚至可能是负的)----
for v in (-0.5, -1.0, -2.0):
    assert grad_at(F.relu, v) == 0.0
    assert grad_at(F.gelu, v) != 0.0

# ---- GELU 非单调:在中等负值区域会跌破0(不是简单地贴着0)----
xs = torch.linspace(-3, 0, 200)
assert F.gelu(xs).min().item() < 0
```

实测:`x=-0.5` 处 `GELU'=0.1325`(正);`x=-1.0` 处 `GELU'=-0.0833`(负!);`x=-2.0` 处 `GELU'=-0.0852`(仍为负,量级和 `x=-1.0` 接近)——`ReLU'` 在这三处都精确是 `0.0000`。`0` 点附近,`ReLU` 导数跳变恰好是 `1.0`,`GELU` 导数跳变只有 `0.00016`,人眼看几乎是一条连续曲线。`GELU(x)` 在 `[-3,0]` 区间的最小值是 `-0.1700`,证实曲线确实跌破了 0(不是像 ReLU 那样贴着 0 封底)。

**AI 研究场景:** GELU 是 BERT、GPT 系列、以及绝大多数现代 Transformer FFN 层的默认激活函数;此外 SiLU/Swish(`x*sigmoid(x)`,和 GELU 数学形态很相似,同样处处光滑、同样负半轴非单调)在 LLaMA 等模型的 FFN 里也被广泛使用——它们共享同一套设计动机:用光滑、概率化的软门控,替代 ReLU 的硬截断。工程上还有一个实际考量:`approximate='tanh'` 提供了一个用 `tanh` 近似 `erf` 的更快版本(避免精确计算 `erf`,在早期硬件/某些算子库上更快),现代 GPU 上这个性能差异已经不明显,`approximate='none'`(精确公式)通常是默认且推荐的选择。

**面试怎么问 + 追问链:**
- **Q:** "GELU 和 ReLU 相比,梯度性质上有什么本质区别?" —— 期望答出"ReLU 分段线性、0点导数不连续;GELU 处处光滑(无限可导)",而不是只说"GELU 效果更好"这种没有机制支撑的结论。
- **追问 1(要求手推):** "手推一下 GELU 的导数。" —— 期望能写出 `GELU(x)=xΦ(x)`,用乘积法则得到 `Φ(x)+xφ(x)`,如果记不住 `Φ`/`φ` 的具体公式扣分会少一些,但乘积法则这一步必须能推出来。
- **追问 2(深挖负半轴):** "GELU 在负半轴的梯度和 ReLU 相比,对训练有什么实际影响?" —— 期望连回上一节 dead ReLU:"GELU 负半轴梯度非零,不会出现 ReLU 那种'一旦 pre-activation 恒负就永久锁死梯度'的情况,理论上更不容易出现大面积死亡神经元"。
- **追问 3(细节,容易漏答):** "GELU 是单调函数吗?" —— 期望答出"不是,GELU 在中等负值区域会先降到 0 以下,形成一个局部最小值,再回升趋近于 0"——很多人只记得"GELU 长得像 ReLU 但更平滑",忽略了这个非单调的细节,这也是 `GELU'(x)` 在某些负值点会是**负数**(而不是"小的正数")的原因。
- **追问 4(反问,防止盲目吹捧新方法):** "GELU 一定比 ReLU 好吗?有没有代价?" —— 期望答出"计算量更大(涉及 `erf`/`exp`,比 `max(0,x)` 昂贵得多,虽然现代硬件上这个差距已经很小),而且'处处光滑'不是免费的午餐,ReLU 依然因为极简单、极快、且在很多 CNN 场景里效果依旧很好而被广泛使用;选择哪个激活函数最终要看具体任务和架构的经验结果,不是'新的一定更好'"。

**常见坑:**
- 把 GELU 记成"ReLU 的平滑版本"就止步于此,说不出"具体怎么个平滑法"(乘积法则推导)、也说不出"平滑对训练稳定性的具体好处是什么"(0点无跳变、负半轴非零梯度)——这类"知道名词、说不出机制"的回答在深挖面试里很容易露怯。
- 以为 GELU 在负半轴的梯度总是正的——中等负值区域梯度可能是**负数**(见追问 3),不是"从 0 慢慢增大到 0.5"这么简单的单调关系。
- 混淆 `approximate='none'` 和 `approximate='tanh'` 两个版本——两者数值上有细微差异(tanh 版本是近似),混用/切换时如果依赖了"数值完全一致"的假设(比如做数值调试对比),会引入难以察觉的小误差来源。

---

## 小结:这一批 11 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | `nn.Linear` 前向反向 | `dL/dW=(dL/dy)^T@x`,`dL/db=(dL/dy).sum(0)`,`dL/dx=dL/dy@W`——梯度形状永远和参数形状一致 |
| 2 | `F.relu` vs `nn.ReLU` | `nn.ReLU.forward` 内部就是调用 `F.relu`,区别只在"能不能被注册进模型结构" |
| 3 | `Conv2d` 的 im2col | 滑窗内积 = 展开成列 + 一次矩阵乘法,是卷积能在 GPU 高效并行的关键,代价是显存膨胀 |
| 4 | `BatchNorm` 训练/推理 + 反向 | 训练用当前 batch 有偏方差,推理用无偏方差更新的 running 统计量;反向传播必须把 μ、σ² 对 x 的依赖也考虑进去,不能当常数 |
| 5 | `LayerNorm` vs `BatchNorm` | 归一化维度不同(跨样本 vs 跨特征)决定了 batch 独立性、小 batch 场景可用性、是否需要 running 统计量 |
| 6 | `Dropout` inverted scale | 训练时把存活值放大 `1/keep_prob`,让训练/推理期望激活量级一致,推理才能零开销 pass through |
| 7 | `Embedding` 稀疏梯度 | 一个 batch 只让词表里被用到的行有非零梯度,重复 token 的梯度会累加求和 |
| 8 | `MultiheadAttention` 拆分/合并 | `view+transpose` 把 embed_dim 拆成多个 head 独立算 attention,再合并回去,全程是形状游戏 |
| 9 | 残差连接与梯度传播 | 加法节点梯度原样传给两个分支,`dy/dx=1+dF/dx` 提供不随深度衰减的梯度高速通道 |
| 10 | ReLU 死亡神经元 | pre-activation 恒负 → 梯度精确为0 → 权重永不更新,是自我锁死的永久性故障,不是"学得慢" |
| 11 | GELU 梯度特性 | 处处光滑,0点无跳变,负半轴梯度非零(甚至可负),缓解 dead-neuron 风险,但计算开销比 ReLU 大 |

下一批:[05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md)

---

*更新:2026-07-07*
