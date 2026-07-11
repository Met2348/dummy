# 05 · 常用层前向反向数学推导(Layers: Math and Backward)—— TensorFlow/Keras 视角

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批覆盖 Dense / Conv2D / BatchNormalization / LayerNormalization / Dropout / Embedding / MultiHeadAttention 七个知识点,是 [torch-deep-dive/04-layers-math-and-backward.md](../torch-deep-dive/04-layers-math-and-backward.md)(以下简称"torch04")的 TensorFlow 对照篇。**这两篇讲的是同一套数学**——前向公式、反向传播的每一条梯度公式、BatchNorm 那个"不能把 μ/σ² 当常数"的经典陷阱,数学推导不会因为换了框架而改变。所以本篇**不重新推导这些数学**,每一节开头都会用一句话交叉引用 torch04 对应小节确认"数学相同",然后把全部篇幅集中在真正有信息增量的地方:**TF/Keras 的参数命名、默认值、张量轴序约定,和 PyTorch 具体在哪里不一样、不一样到什么程度**——这些差异不是"换个名字"那么简单,`momentum` 语义相反、kernel 轴序相反、padding 从数值变成字符串,都是带着 PyTorch 直觉直接照搬就会踩坑的真实陷阱,本篇每一条差异结论背后都有现场跑出来的真实数字或真实报错文本支撑,不是转述文档。

**前提:** 建议先看完 [torch-deep-dive/04-layers-math-and-backward.md](../torch-deep-dive/04-layers-math-and-backward.md) 全文(本篇假设你已经知道 `nn.Linear`/`Conv2d` 反向传播怎么推、BatchNorm 为什么不能把均值方差当常数、多头注意力怎么拆头合并),以及 [numpy-deep-dive/06-linear-algebra.md](../numpy-deep-dive/06-linear-algebra.md) 第 15 节 `np.einsum`(本篇第 7 节会大量用到 einsum 表达式,不重讲下标记法本身怎么读,只讲它在 attention 里具体怎么用)。

**环境:** 本文所有代码在 [00-roadmap.md](00-roadmap.md) 声明的 WSL2 + `~/tf-venv`(TensorFlow 2.21.0,GPU 直通,`TF_USE_LEGACY_KERAS=1` 让 `tf.keras` 解析回经典 Keras 2 实现)环境下实际跑通验证。如果你在自己电脑上复现,必须先按 00 篇的说明装好 `tf_keras` 并设置环境变量,否则部分依赖"经典 tf.keras 内部结构"的例子(尤其第 7 节内省 `EinsumDense` 子层的部分)可能和你观察到的不一致。**"AI 研究/工程场景"这一段的诚实声明**(全系列统一,这里不重复展开):仓库里没有真实的 TensorFlow 代码可引用,下面每节的场景是根据真实训练/部署问题重构的,不是仓库文件引用。

**本篇统一结构(与 00 篇一致的 7 段式):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 先一句话确认与 torch04 数学相同,然后聚焦 TF API 和 PyTorch 的具体差异
4. AI 研究/工程场景
5. 可运行例子(带 `assert`,WSL2 `~/tf-venv` 实测)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. `Dense`——和 `nn.Linear` 同一个数学,权重存储形状是转置关系

**是什么:**
```
tf.keras.layers.Dense(
    units,                               # 输出维度,对应 nn.Linear 的 out_features
    activation=None,
    use_bias=True,                       # 对应 nn.Linear 的 bias(bool),关键字参数名不同
    kernel_initializer='glorot_uniform',  # 默认 Xavier/Glorot 均匀分布
    bias_initializer='zeros',
)
# 前向: y = x @ kernel + bias
#   x:      (..., input_dim)   -- 只变换最后一维,前面所有维度当"批量"维,和 nn.Linear 完全一致
#   kernel: (input_dim, units)  -- 注意:这是 (in,out),不是 nn.Linear.weight 的 (out,in)
#   bias:   (units,)
```

**一句话:** Dense 和 `nn.Linear` 做的是同一个矩阵乘法 + 广播加法,前向反向的三条梯度公式(`dL/dW`、`dL/db`、`dL/dx`)在数学上完全等价于 torch04 第 1 节推导的结果,这里不重复推导;唯一但很容易踩坑的差异是**权重矩阵的存储形状**——TF 的 `kernel` 形状是 `(in,out)`,PyTorch 的 `weight` 形状是 `(out,in)`,两者互为转置。

**底层机制/为什么这样设计:**

torch04 第 1 节推导的 `dL/dW=(dL/dy)^T@x`、`dL/db=dL/dy.sum(0)`、`dL/dx=dL/dy@W` 三条公式,对 TF 的 Dense 层同样成立(把公式里的 `W` 换成 `kernel.T` 即可,因为 TF 前向用的是 `x@kernel` 不是 `x@kernel.T`)——这里不重复推导,重点是把"转置在哪里发生"这件事讲清楚。

`nn.Linear` 的前向是 `y = x @ W.T + b`,`W.shape == (out,in)`——每次前向都要"转置"一下(逻辑上如此,实际执行框架会做优化)。TF 的 Dense 反过来,**把这个转置提前烤进了存储形状里**:`kernel.shape == (in,out)`,前向直接 `y = x @ kernel + bias`,不需要再转置。这不是数学差异(两边算出来的 `y` 数值完全一样),纯粹是"转置这一步在存储的时候做一次,还是在每次前向的时候做"的工程选择——但这个选择直接影响**跨框架迁移权重**这件真实工程任务:把一个 PyTorch 训练好的 `nn.Linear.weight`(形状 `(out,in)`)搬到 TF 的 `Dense.kernel` 里,必须先 `.T` 转置一次,形状对不上会直接报错,形状"凑巧"对上(比如 `in==out` 的方阵)则会得到一个能跑但数值完全错误的模型。

第二个差异是**延迟构建(lazy build)**:`Dense(units=3)` 构造时**没有传入输入维度**——`kernel` 要等到第一次真正被调用、看到输入的最后一维大小,才会被创建出来。构造完成但还没调用过的 `Dense` 层,访问 `.kernel` 会直接 `AttributeError`(下面代码现场验证)。这是 Keras 层的通用设计(`build()` 机制,[04-keras-api-internals.md](04-keras-api-internals.md) 会展开讲),`nn.Linear` 则要求构造时就显式给出 `in_features`——PyTorch 里最接近的等价物是 `nn.LazyLinear`,但那是一个特意存在的"额外"类,不是 `nn.Linear` 的默认行为。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
dense = tf.keras.layers.Dense(units=3, use_bias=True)
x = tf.constant(np.random.randn(5, 4).astype("float32"))

# 构造时还没有 kernel(延迟构建)
dense_lazy = tf.keras.layers.Dense(units=3)
assert dense_lazy.built is False
try:
    _ = dense_lazy.kernel
    raised = False
except AttributeError:
    raised = True
assert raised
_ = dense_lazy(x)
assert dense_lazy.built is True

y = dense(x)  # 第一次调用触发 build,kernel 形状此时才确定
assert dense.kernel.shape == (4, 3)          # (in_features, units) —— 和 nn.Linear.weight 的 (out,in) 是转置关系
assert dense.bias.shape == (3,)

# 前向确实是 x @ kernel + bias,不是 x @ kernel.T + bias
manual_y = x.numpy() @ dense.kernel.numpy() + dense.bias.numpy()
assert np.allclose(manual_y, y.numpy(), atol=1e-5)

# 如果错误地当成 PyTorch 的 (out,in) 去转置着用,形状直接对不上,TF不会静默给错误结果
try:
    _ = x.numpy() @ dense.kernel.numpy().T
    shape_mismatch_raised = False
except ValueError:
    shape_mismatch_raised = True
assert shape_mismatch_raised   # (5,4) @ (3,4) 形状不合法

# use_bias=False -> 没有 bias 变量(不是"全零的bias",是 None)
dense_nobias = tf.keras.layers.Dense(units=3, use_bias=False)
_ = dense_nobias(x)
assert dense_nobias.bias is None

# 3D 输入:只变换最后一维,前面维度当批量维,和 nn.Linear 行为一致
x3 = tf.constant(np.random.randn(2, 7, 4).astype("float32"))
dense3 = tf.keras.layers.Dense(units=6)
y3 = dense3(x3)
assert y3.shape == (2, 7, 6)
assert dense3.kernel.shape == (4, 6)

# 默认初始化器和 PyTorch 默认不是同一分布
assert type(dense.kernel_initializer).__name__ == "GlorotUniform"
# PyTorch nn.Linear 默认是 kaiming_uniform_(a=sqrt(5)),两者是不同的分布族,不是"同一个初始化换了个名字"

print("TF version:", tf.__version__)
```

**AI 研究/工程场景:** 把一个在 PyTorch 里训练好、要部署到 TFLite/TF Serving 的模型手动"抄"一遍权重(常见于双端团队各自用擅长的框架,推理侧统一到 TF 生态)时,`kernel = W.T` 这一步转置几乎是抄权重脚本里第一个会漏掉的地方——如果只对了 `in==out` 的方阵层,漏转置不会报错,只会让这一层数值完全错误,后续所有层的输出都跟着错,且不一定表现为报错(可能只是精度指标全面下滑,排查成本很高)。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 Dense 层和 PyTorch 的 `nn.Linear` 在实现上有什么本质区别?" —— 期望先说"数学上没有区别,前反向公式完全一样",再说出"权重存储形状是转置关系"这个核心差异,而不是只停留在"参数名不一样"这种表面观察。
- **追问 1:** "如果要把一个 PyTorch 训练好的 `nn.Linear` 权重加载进 TF 的 `Dense`,要做哪一步转换?" —— 期望准确答出 `kernel = W.numpy().T`,并能说明如果忘记转置、且形状恰好允许(方阵)会发生什么(不报错,数值错误)。
- **追问 2:** "`Dense` 的 `kernel` 是什么时候被创建的?为什么 PyTorch 的 `nn.Linear` 不需要这一步?" —— 期望答出 `build()` 延迟构建机制,构造时不需要预先指定 `input_dim`,第一次调用才推断;PyTorch 需要显式 `in_features`(除非用 `nn.LazyLinear`)。
- **追问 3(呼应 torch04 追问4):** "3D 输入 `(batch,seq,features)` 喂给 Dense 会发生什么?" —— 只变换最后一维,和 `nn.Linear` 完全一致,这一点两个框架没有差异。

**常见坑:**
- 手动迁移权重时忘记转置 `kernel`/`weight`,方阵层(`in_features==out_features`)不会报错,只会让这一层数值全错。
- 以为 `Dense(units=3)` 构造完就能访问 `.kernel` 查看形状——必须先做过一次前向调用(或显式 `.build(input_shape)`)。
- 想当然认为两边默认初始化器是同一个分布(TF 默认 Glorot 均匀,PyTorch `nn.Linear` 默认 Kaiming 均匀且用了一个特殊的 `a=sqrt(5)`),跨框架复现同一个实验、又没有显式指定初始化器时,起始权重分布就已经不同,训练早期的行为不能指望完全对齐。

---

## 2. `Conv2D`——同一个 im2col 数学,`data_format` 默认值和 kernel 轴序都和 PyTorch 相反

**是什么:**
```
tf.keras.layers.Conv2D(
    filters,                        # 对应 PyTorch 的 out_channels
    kernel_size,
    strides=(1, 1),
    padding='valid',                 # 字符串:'valid' 或 'same',不是数值
    data_format='channels_last',      # 默认 NHWC —— 和 PyTorch 固定的 NCHW 相反
)
# TF 输入默认形状: (N, H, W, C_in)          PyTorch 固定: (N, C_in, H, W)
# TF kernel 形状:  (kH, kW, C_in, C_out)     PyTorch weight 固定: (C_out, C_in, kH, kW)
```

**一句话:** 卷积的数学本质——滑窗内积能等价改写成 im2col + 一次矩阵乘法——和 torch04 第 3 节完全相同,这里不重复推导;TF 和 PyTorch 在 Conv2D 上的差异全部是"表达同一件事的不同约定",但恰好集齐了三处:输入张量的轴序默认值(`data_format`)、卷积核张量的轴序、以及 padding 的写法(字符串 vs 数值),是实操中最容易踩坑的层之一。

**底层机制/为什么这样设计:**

`im2col` 把滑动窗口内积改写成一次矩阵乘法这件事,TF 和 PyTorch 底层都是这么做的(或用等价的 Winograd/隐式 GEMM 等算法,数值结果一致),torch04 第 3 节的推导和验证在这里原样成立,不再重复。下面是三处真正需要现场验证才能讲清楚的 API 差异。

**差异 1:`data_format` 默认值相反。** TF 默认 `channels_last`(`NHWC`,通道在最后一维),PyTorch 的 `nn.Conv2d` **没有** `data_format` 这个参数——它固定假设输入是 `NCHW`(通道在第二维),没有切换选项(PyTorch 有 `memory_format=torch.channels_last`,但那只是**内存布局**的性能优化,tensor 的**逻辑 shape 顺序**依然是 `(N,C,H,W)`,和这里说的 `data_format` 不是一回事)。直接把一份 TF 的 `(N,H,W,C)` 数据丢给期望 `(N,C,H,W)` 的 PyTorch 模型,如果 `H==C` 或 `W==C` 凑巧形状对得上,不会报错,只会把空间维度和通道维度**语义错位**,模型能跑但学到的东西毫无意义。

**差异 2:卷积核的轴序也不一样,这是和 `data_format` 独立的另一个坑。** 很多人以为"卷积核轴序"和"输入数据轴序"是绑在一起的同一件事,现场验证:TF 的 `kernel.shape == (kH,kW,C_in,C_out)`,PyTorch 的 `weight.shape == (C_out,C_in,kH,kW)`——这是**权重张量**的轴序差异,不管输入用的是 `NHWC` 还是 `NCHW`,权重的轴序约定都不会变。跨框架搬运一个卷积网络,数据的轴序要转一次,权重的轴序**要再转一次**,是两次独立的形状体操,漏了哪一次都会导致跑起来不报错但结果错误。

**差异 3:`padding` 从数值变成字符串,且字符串语义暗藏一个"隐藏公式"。** PyTorch 的 `padding` 是一个具体数字(每边填充多少像素);TF 的 `padding='valid'`(完全不填充)或 `padding='same'`(自动计算填充量,让输出空间尺寸满足 `out=ceil(in/stride)`)。`'same'` 背后的公式已现场验证:

```
out = ceil(in / stride)
total_pad = max((out-1)*stride + kernel_size - in, 0)
pad_before = total_pad // 2
pad_after  = total_pad - pad_before      # stride>1 或 kernel 是偶数时,pad_before != pad_after(不对称!)
```

只有 `stride==1` 且 `kernel_size` 是奇数时,`'same'` 才退化成 PyTorch 里那种最直觉的"两边对称各填 `kernel_size//2`"。`stride>1` 或 `kernel_size` 是偶数时,`'same'` 会算出**不对称**的填充量(一边比另一边多填一格),已现场验证(`H=9,k=4,s=2` 时 `pad_before=1,pad_after=2`)。PyTorch 自 1.9 起也支持 `padding='same'` 这个字符串,但**只支持 `stride=1`**——`stride>1` 时直接抛 `RuntimeError: padding='same' is not supported for strided convolutions`(已用真实 PyTorch 2.11.0+cu128 触发验证),TF 的 `'same'` 则对任意 stride 都适用(靠的就是上面那个可能不对称的填充公式)。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np
import math

np.random.seed(0)

# ---- 差异1: 默认 data_format ----
conv = tf.keras.layers.Conv2D(filters=4, kernel_size=3)
assert conv.data_format == "channels_last"
x = tf.constant(np.random.randn(2, 8, 8, 3).astype("float32"))   # NHWC
y = conv(x)
assert y.shape == (2, 6, 6, 4)

# ---- 差异2: kernel 轴序,和 PyTorch (C_out,C_in,kH,kW) 完全不同 ----
assert conv.kernel.shape == (3, 3, 3, 4)     # (kH, kW, C_in, C_out)

# ---- 差异3: padding 公式 ----
def out_valid(H, k, s):
    return (H - k) // s + 1

def out_same(H, s):
    return math.ceil(H / s)

for k, s, pad, H in [(3, 1, "valid", 9), (3, 1, "same", 9), (3, 2, "valid", 9), (4, 2, "same", 9)]:
    c = tf.keras.layers.Conv2D(filters=2, kernel_size=k, strides=s, padding=pad)
    out = c(tf.constant(np.random.randn(1, H, H, 1).astype("float32")))
    expect = out_valid(H, k, s) if pad == "valid" else out_same(H, s)
    assert out.shape[1] == expect, (k, s, pad, out.shape[1], expect)

# same + stride=1 + 奇数kernel == PyTorch式对称 padding=k//2
k = 3
c_same = tf.keras.layers.Conv2D(filters=2, kernel_size=k, strides=1, padding="same",
                                  use_bias=False, kernel_initializer="ones")
xin = tf.constant(np.random.randn(1, 5, 5, 1).astype("float32"))
y_same = c_same(xin)
x_padded = tf.pad(xin, [[0, 0], [k // 2, k // 2], [k // 2, k // 2], [0, 0]])
c_valid = tf.keras.layers.Conv2D(filters=2, kernel_size=k, strides=1, padding="valid",
                                   use_bias=False, kernel_initializer="ones")
_ = c_valid(xin)
c_valid.kernel.assign(c_same.kernel)
assert np.allclose(y_same.numpy(), c_valid(x_padded).numpy(), atol=1e-4)

# same + stride=2 + 偶数kernel: 填充不对称(pad_before != pad_after)
H, k2, s2 = 9, 4, 2
out = out_same(H, s2)
total_pad = max((out - 1) * s2 + k2 - H, 0)
pad_before, pad_after = total_pad // 2, total_pad - total_pad // 2
assert (pad_before, pad_after) == (1, 2)     # 现场验证:确实不对称

print("TF version:", tf.__version__)
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):所有 `(kernel_size,stride,padding)` 组合的输出尺寸公式全部吻合;`same+stride=1+奇数kernel` 和手工对称填充版本数值一致;`H=9,k=4,s=2` 时 `same` 算出的不对称填充确认是 `pad_before=1,pad_after=2`。另在 Windows 侧用真实 PyTorch 2.11.0+cu128 触发验证:`F.conv2d(x,w,stride=2,padding='same')` 精确抛出 `RuntimeError: padding='same' is not supported for strided convolutions`,`stride=1` 时则正常工作。

**AI 研究/工程场景:** 数据预处理管线用 OpenCV/PIL 起手(通常产出 `HWC` 排列的图像数组)。如果下游模型是 PyTorch,常规操作是显式 `.transpose(2,0,1)` 把 `HWC` 转成 `CHW`;如果下游换成 TF/Keras,这一步转置往往被(正确地)省略——因为 TF 默认的 `NHWC` 恰好和 OpenCV/PIL 的原始排列一致。这也是为什么"从 PyTorch 迁移到 TF 的图像预处理代码"经常会把这行 transpose 误留下来,反而引入了一个不必要且错误的轴序变换。模型转换工具(`onnx-tf`、`tf2onnx` 之类)要负责的核心工作之一,就是正确处理这里的 `data_format` 和 kernel 轴序转换,理解这两处差异是排查"模型转换后精度掉了但结构看起来一样"这类问题的必备知识。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 Conv2D 默认输入格式是什么?和 PyTorch 一样吗?" —— 期望准确说出 TF 默认 `NHWC`,PyTorch 固定 `NCHW`,且能说清楚"固定"和"默认"的区别(TF 可以切换,PyTorch 没有切换的参数)。
- **追问 1:** "如果要把一个 TF 训练好的 Conv2D 权重搬到 PyTorch 里用,要做几步转换?" —— 期望答出两步独立操作:输入数据的轴序转换(如果输入也要复用)和卷积核张量的轴序转置(`kernel.transpose(3,2,0,1)` 从 `(kH,kW,C_in,C_out)` 转到 `(C_out,C_in,kH,kW)`),并说清楚这是两件独立的事。
- **追问 2:** "`padding='same'` 在 TF 和 PyTorch 里是完全等价的概念吗?" —— 期望答出"不完全等价":数学目标一致(输出尺寸等于 `ceil(input/stride)`),但 PyTorch 的 `'same'` 字符串只支持 `stride=1`,`stride>1` 直接报错,TF 的 `'same'` 对任意 stride 都能工作(靠可能不对称的填充)。
- **追问 3(深挖):** "什么情况下 `'same'` 算出来的填充是不对称的?" —— 期望答出"`stride>1` 或者 `kernel_size` 是偶数时",并能现场说出不对称填充公式的大致形状。

**常见坑:**
- 把"`data_format` 决定输入轴序"和"kernel 轴序"当成同一件事——两者独立,搬运一个卷积网络必须两处都转换,只转其中一处会得到一个能跑但语义错误的模型。
- 假设 `padding='same'` 在任意 `stride` 下都等价于 PyTorch `padding=kernel_size//2`——只有 `stride=1` 且 `kernel_size` 为奇数时才成立,`stride>1` 时 TF 会自动算出可能不对称的填充,而 PyTorch 的 `padding='same'` 字符串在 `stride>1` 时**根本不支持**,直接报错。
- 混用两边的输入数据不做轴序转换,当空间维度和通道数数值上凑巧能对上形状检查时,模型不会报错但训练/推理结果毫无意义,这是最难排查的一类坑,因为它不产生任何异常信号。

---

## 3. `BatchNormalization`——归一化公式相同,但 `momentum` 语义和 PyTorch **相反**(全系列最经典的跨框架陷阱)

**是什么:**
```
tf.keras.layers.BatchNormalization(
    axis=-1,           # 沿哪个轴是"特征通道"(默认最后一维,呼应 data_format='channels_last')
    momentum=0.99,      # !! 语义和 PyTorch 的 momentum 相反,默认值也不同(PyTorch 默认 0.1)
    epsilon=1e-3,
    center=True, scale=True,
)
# 训练模式: moving_mean = moving_mean * momentum + batch_mean * (1 - momentum)
# PyTorch:  running_mean = running_mean * (1-momentum) + batch_mean * momentum   <- 两个系数的位置反了
```

**一句话:** 训练/推理两套统计量的存在、归一化公式本身(用当前 batch 的有偏方差)——这些和 torch04 第 4 节完全相同,这里不重复推导;但两个框架都叫 `momentum` 的这个参数,数学含义**互为相反数的角色**,是跨框架代码里最经典、最容易被"同名参数"这个假象坑到的一个陷阱。

**底层机制/为什么这样设计:**

归一化公式(减 batch 均值除以 batch 标准差,`γ`/`β` 仿射还原)和 torch04 第 4 节的推导完全一致,已现场验证 TF 的输出确实等于用**有偏**方差(`ddof=0`)算出的手工公式,和 PyTorch 的归一化方式相同,这一点不再重复。下面是四处需要现场验证才能讲清楚的数值行为差异。

**差异 1(全系列最经典的坑):`momentum` 语义相反。** TF 的公式是 `moving_mean_new = moving_mean_old * momentum + batch_mean * (1-momentum)`——`momentum` 是"保留多少旧值"的权重,越接近 1,新 batch 对 `moving_mean` 的影响越小(更新得越"慢")。PyTorch(torch04 第 4 节已验证)的公式是 `running_mean_new = running_mean_old * (1-momentum) + batch_mean * momentum`——`momentum` 是"采用多少新值"的权重,越接近 1,更新得越"快"。**两个公式里 `momentum` 和 `1-momentum` 的位置完全对调**,同一个数字代入两边,含义正好相反。下面代码用同一组输入数据、同一个 `momentum=0.9`,现场对比 TF 真实运行结果和(torch04 已用真实 `torch.nn.BatchNorm1d` 验证过的)PyTorch 公式,两边算出的 `moving/running_mean` 完全不同;并验证一个更能说明问题的对称关系:**TF 的 `momentum=0.1` 和 PyTorch 的 `momentum=0.9` 算出完全相同的数字**——因为两边真正决定更新速度的都是"保留旧值的那个系数",TF 里这个系数就是 `momentum` 本身,PyTorch 里这个系数是 `1-momentum`,想要两边行为一致,必须显式换算 `momentum_TF = 1 - momentum_PyTorch`,不能直接照抄数字。

**差异 2:不只是语义反,默认值本身也不一样。** TF 默认 `momentum=0.99`(保留 99% 旧值,更新非常慢);PyTorch 默认 `momentum=0.1`(保留 90% 旧值)。这意味着"知道语义反了、自作聪明填个 `1-x`"也救不了直接照搬默认值的场景——因为压根没人显式设置过 `momentum`,两边默认值本身就不是互为镜像的数字(`1-0.99=0.01 != 0.1`)。

**差异 3:`moving_variance` 更新用的方差,有偏/无偏也不一样。** torch04 第 4 节验证过一个容易被忽略的 PyTorch 细节:归一化用**有偏**方差(`ddof=0`),但更新 `running_var` 用的是**无偏**方差(`ddof=1`,Bessel 校正)。这里现场验证 TF 的对应行为:**TF 归一化和更新 `moving_variance` 用的是同一个有偏方差**,不像 PyTorch 那样"归一化一个公式、更新 buffer 又换一个公式"。这是除 `momentum` 反转之外,第二个真实存在、容易被面试问到细节的差异点。

**差异 4:`batch_size=1` 时的健壮性不同。** torch04 第 4 节验证过 PyTorch 训练模式下 `BatchNorm1d` 遇到 `batch_size=1` 会直接抛 `RuntimeError`(方差在单样本下退化,PyTorch 主动拦截)。TF **不会报错**——现场验证:单样本喂进训练模式的 `BatchNormalization`,方差算出来精确是 0,输出精确是全 0 向量,`moving_mean`/`moving_variance` 依然会被"正常"更新(只是这一步统计量毫无代表性)。这不是"TF 更健壮",而是 TF 选择不对这个退化情况报错——如果代码不小心用 `batch_size=1` 跑了训练模式的 BN,PyTorch 会用一个响亮的报错提醒你,TF 会默默给一个没有意义的输出,排查起来反而更麻烦。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

x = tf.constant([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
batch_mean = x.numpy().mean(axis=0)

# ---- 差异1+2: momentum 语义反 + 默认值不同 ----
bn = tf.keras.layers.BatchNormalization(momentum=0.9)
_ = bn(x, training=True)
tf_moving_mean = bn.moving_mean.numpy()
manual_tf_formula = 0.0 * 0.9 + batch_mean * (1 - 0.9)          # TF: moving = old*momentum + batch*(1-momentum)
assert np.allclose(tf_moving_mean, manual_tf_formula, atol=1e-5)
assert np.allclose(tf_moving_mean, [0.4, 0.5], atol=1e-4)

# 同一 momentum=0.9,如果套用 PyTorch 的公式(torch04第4节已用真实 torch.nn.BatchNorm1d 验证过这条公式;
# 这里额外在 Windows 侧用真实 torch 2.11.0+cu128、同一批输入重新跑过一遍,golden 值是 [3.6, 4.5])
pytorch_golden_momentum_0_9 = np.array([3.5999999046325684, 4.5])
assert not np.allclose(tf_moving_mean, pytorch_golden_momentum_0_9, atol=0.5)   # 同一个数字,含义完全不同

# 对称关系现场验证: TF的momentum=0.1 和 PyTorch的momentum=0.9 算出同一个数字
# (两边"保留旧值系数"都是0.9: TF里这个系数就是momentum本身,PyTorch里是1-momentum)
bn_sym = tf.keras.layers.BatchNormalization(momentum=0.1)
_ = bn_sym(x, training=True)
assert np.allclose(bn_sym.moving_mean.numpy(), pytorch_golden_momentum_0_9, atol=1e-4)

bn_default = tf.keras.layers.BatchNormalization()
assert bn_default.momentum == 0.99            # PyTorch 默认是 0.1,不是 1-0.99=0.01,默认值本身不对称

# ---- 差异3: moving_variance 用有偏方差更新(PyTorch 用无偏方差,torch04第4节已验证) ----
bn_var = tf.keras.layers.BatchNormalization(momentum=0.9)
_ = bn_var(x, training=True)
batch_var_biased = x.numpy().var(axis=0, ddof=0)     # 5.0
batch_var_unbiased = x.numpy().var(axis=0, ddof=1)    # 6.6667
manual_update_biased = 1.0 * 0.9 + batch_var_biased * (1 - 0.9)      # moving_variance 初始值是1.0(Ones)
assert np.allclose(bn_var.moving_variance.numpy(), manual_update_biased, atol=1e-4)
manual_update_unbiased = 1.0 * 0.9 + batch_var_unbiased * (1 - 0.9)
assert not np.allclose(bn_var.moving_variance.numpy(), manual_update_unbiased, atol=1e-2)

# ---- 差异4: batch_size=1, training=True, TF 不报错 ----
bn_bs1 = tf.keras.layers.BatchNormalization()
y_bs1 = bn_bs1(tf.constant([[1., 2., 3.]]), training=True)
assert np.allclose(y_bs1.numpy(), 0.0)          # 方差退化为0,输出精确是全0,不报错

print("TF version:", tf.__version__)
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):TF `momentum=0.9` 算出 `moving_mean=[0.4,0.5]`;真实 PyTorch `momentum=0.9`(2.11.0+cu128 实测)算出 `running_mean=[3.6,4.5]`——同一个数字、同一批输入,结果天差地别;TF `momentum=0.1` 精确复现了 PyTorch `momentum=0.9` 的 `[3.6,4.5]`,印证了"保留旧值系数"才是两边真正对应的量。`moving_variance` 用有偏方差更新确认为 `1.4`,如果套用无偏方差会是 `1.5667`,和实际观测值对不上。

**AI 研究/工程场景:** 复现一篇论文或者移植一份 PyTorch 训练脚本到 TF 时,BatchNorm 的 `momentum` 几乎是被"直接抄数字"最频繁的一个超参数——因为参数名字面意义相同、默认值又都是"接近 1 的一个小数",容易让人误以为只是数值大小不同、不是方向相反。这类 bug 的典型症状不是训练崩溃或报错,而是**推理阶段(`training=False`)的效果明显差于训练阶段**,且差距随训练步数增加而恶化或迟迟不收敛——因为 `moving_mean`/`moving_variance` 要么几乎不更新(错配了"应该快"的语义和"要慢"的默认值组合)、要么更新过猛导致统计量在训练后期还在大幅波动,拿一个不稳定的统计量去做推理,效果自然不稳定。排查这类问题时,第一反应应该是**现场打印 `moving_mean`/`moving_variance` 在训练过程中的变化曲线**,而不是先怀疑模型结构或数据。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `BatchNormalization` 和 PyTorch 的 `BatchNorm` 都有一个叫 `momentum` 的参数,它们的含义一样吗?" —— 期望第一句话就是"不一样,而且是相反的",然后能说出两条公式的差异(哪个系数配的是 `momentum` 本身,哪个系数配的是 `1-momentum`)。
- **追问 1(杀伤力很强):** "如果我把一份 PyTorch 代码里 `momentum=0.1` 的 BN 层直接照抄成 TF 的 `momentum=0.1`,会发生什么?" —— 期望答出"完全不是等价的迁移——PyTorch 的 0.1 表示'保留 90% 旧值,更新较慢但不算太慢',TF 的 0.1 表示'只保留 10% 旧值,几乎每个 batch 都在往新统计量大幅跳',要正确迁移必须换算成 `momentum_TF = 1 - momentum_PyTorch = 0.9`"。
- **追问 2(考察是否只知道表面结论):** "除了 `momentum` 语义相反,BatchNorm 在两个框架里还有其他数值行为差异吗?" —— 期望能提到 `moving_variance`/`running_var` 更新时用的方差是否无偏这一点(TF 用有偏,PyTorch 用无偏),这是一个更细节、更容易分辨"背过这条坑" vs "真正系统验证过"的追问。
- **追问 3:** "`batch_size=1` 时,两边分别会发生什么?" —— 期望答出 PyTorch 训练模式下直接 `RuntimeError`,TF 不报错但会静默给出全 0 输出且用这个没有意义的统计量更新 `moving_mean`/`moving_variance`。

**常见坑:**
- 跨框架迁移代码时直接照抄 `momentum` 数值,不做 `1-x` 换算——这是本节的核心陷阱,而且不会报错,只会让训练/推理表现出诡异的不一致。
- 知道要换算 `1-x`,但忘记两边**默认值本身也不同**,只对显式设置过的层做了换算,漏了依赖默认值的层。
- 只记住"momentum 反了"这一条坑,以为 BatchNorm 在两个框架里其他地方都完全一致——`moving_variance` 的有偏/无偏差异同样是真实存在、容易被面试追问出来的第二层细节。
- 误以为 TF 在 `batch_size=1` 时"没报错"等于"这样用是安全的"——它只是没有主动拦截,静默产出的统计量依然是没有意义的。

---

## 4. `LayerNormalization`——同一个归一化,`epsilon` 默认值差 100 倍,`center`/`scale` 独立开关

**是什么:**
```
tf.keras.layers.LayerNormalization(
    axis=-1,            # 可以是单个int,也可以是一个list(多个轴一起归一化)—— 比 PyTorch 更灵活
    epsilon=1e-3,        # !! 默认值和 PyTorch 的 1e-5 差 100 倍
    center=True,          # 是否有 beta(平移),可以独立于 scale 关闭
    scale=True,            # 是否有 gamma(缩放),可以独立于 center 关闭
)
# PyTorch: nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
#          normalized_shape 必须是"从最后一维往前数"的连续若干维,eps 默认 1e-5
#          elementwise_affine 只有一个开关,同时控制 gamma 和 beta,不能只要一个
```

**一句话:** "在哪个维度上求均值方差"这个决定性质(横着看、跨特征、batch 独立)和 torch04 第 5 节完全相同,这里不重复推导;API 层面三处差异都已现场验证:默认 `epsilon` 相差 100 倍、`center`/`scale` 可以独立开关(PyTorch 只有一个联合开关)、`axis` 可以是任意轴甚至任意一组轴(PyTorch 的 `normalized_shape` 只能是连续的尾部若干维)。

**底层机制/为什么这样设计:**

归一化公式本身(对每个样本自己的若干维求均值方差,`(x-μ)/sqrt(σ²+ε)` 再仿射)和 torch04 第 5 节完全一致,已现场验证 TF 用的也是有偏方差(`ddof=0`),和 torch04 描述的 PyTorch 行为一致,这里不重复推导。

**差异 1:`epsilon` 默认值不同,而且差距不小。** TF `LayerNormalization` 默认 `epsilon=1e-3`,PyTorch `nn.LayerNorm` 默认 `eps=1e-5`,**相差 100 倍**(`1e-3/1e-5=100`,已现场验证)。`epsilon` 是加在方差上防止除零的小常数,通常量级远小于方差本身、影响可以忽略——但如果输入某一维本身方差就很小(比如某些初始化阶段、或者输入特征高度相关的场景),`epsilon` 的选择会变得可观测,跨框架比较两边输出、或者想在 TF 里精确复现一个 PyTorch 模型的数值行为,必须显式对齐 `epsilon`,不能依赖默认值"反正都很小,应该没关系"。

**差异 2:`center`/`scale` 独立开关,PyTorch 做不到。** PyTorch 只有一个 `elementwise_affine` 布尔值,要么同时有 `weight`(对应 `gamma`)和 `bias`(对应 `beta`),要么两个都没有。TF 现场验证:`center=True, scale=False` 时,`beta` 存在但 `gamma` 不存在——这是 PyTorch 单一开关**结构性做不到**的组合(如果只想要平移、不想要缩放,PyTorch 需要手动把 `weight` 冻结成全 1 常量并从优化器里排除,TF 一个参数就能表达)。

**差异 3:`axis` 可以是任意轴(甚至任意一组轴),不要求是"尾部连续若干维"。** PyTorch 的 `normalized_shape` 语义是"从最后一维往前数 `len(normalized_shape)` 维",天然只能描述**连续的尾部维度**。TF 的 `axis` 参数可以传一个单独的轴(哪怕是中间某一维,不是最后一维),也可以传一个 list(比如 `axis=[1,2]` 同时在两个维度上求统计量)。已现场验证两种情形:`axis=1`(3 维输入的中间维,不是最后一维)一样能正常工作,归一化结果和手工按 `axis=1` 求均值方差算出的公式完全吻合;`axis=[1,2]` 多轴同时归一化同样吻合。这是 PyTorch 的 `normalized_shape` 无法直接表达的场景——要归一化一个不在尾部的维度,PyTorch 那边必须先 `permute` 把目标维度换到最后,算完再换回来。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

np.random.seed(0)
ln = tf.keras.layers.LayerNormalization()

# ---- 差异1: epsilon 默认值相差100倍 ----
assert ln.epsilon == 1e-3          # PyTorch nn.LayerNorm 默认 eps=1e-5,相差100倍
assert ln.axis == -1

x = tf.constant(np.random.randn(4, 5).astype("float32") * 3 + 1)
out = ln(x)
mu = x.numpy().mean(axis=-1, keepdims=True)
var = x.numpy().var(axis=-1, keepdims=True, ddof=0)      # 有偏方差,和 torch04 第5节描述的PyTorch行为一致
manual = (x.numpy() - mu) / np.sqrt(var + ln.epsilon)
assert np.allclose(out.numpy(), manual, atol=1e-4)

names = {w.name.split("/")[-1].split(":")[0] for w in ln.trainable_weights}
assert names == {"gamma", "beta"}       # 参数名是 gamma/beta,不是 weight/bias

# ---- 差异2: center/scale 独立开关,PyTorch 做不到 ----
ln_only_center = tf.keras.layers.LayerNormalization(center=True, scale=False)
_ = ln_only_center(x)
has_beta = any("beta" in w.name for w in ln_only_center.trainable_weights)
has_gamma = any("gamma" in w.name for w in ln_only_center.trainable_weights)
assert has_beta is True and has_gamma is False    # 只有beta没有gamma,PyTorch的elementwise_affine做不到这个组合

# ---- 差异3: axis 可以不是最后一维,甚至可以是多个轴 ----
x3 = tf.constant(np.random.randn(2, 3, 4).astype("float32"))

ln_mid = tf.keras.layers.LayerNormalization(axis=1)     # 归一化"中间"这一维,不是最后一维
out_mid = ln_mid(x3)
mu_mid = x3.numpy().mean(axis=1, keepdims=True)
var_mid = x3.numpy().var(axis=1, keepdims=True, ddof=0)
assert np.allclose(out_mid.numpy(), (x3.numpy() - mu_mid) / np.sqrt(var_mid + ln_mid.epsilon), atol=1e-4)

ln_multi = tf.keras.layers.LayerNormalization(axis=[1, 2])   # 同时在两个轴上求统计量
out_multi = ln_multi(x3)
mu_multi = x3.numpy().mean(axis=(1, 2), keepdims=True)
var_multi = x3.numpy().var(axis=(1, 2), keepdims=True, ddof=0)
assert np.allclose(out_multi.numpy(), (x3.numpy() - mu_multi) / np.sqrt(var_multi + ln_multi.epsilon), atol=1e-4)

print("TF version:", tf.__version__)
```

**AI 研究/工程场景:** 把一个 HuggingFace/PyTorch 训练好的 Transformer 权重转换到 TF 版本做推理服务(常见于同一个模型要同时提供 PyTorch 训练 + TF Serving 部署两条链路)时,如果只对齐了 `gamma`/`beta`(`weight`/`bias`)权重数值,却没有显式把 TF `LayerNormalization` 的 `epsilon` 改成和源模型一致的 `1e-5`,两边推理输出会有一个小但可测量的数值偏差——单层看误差很小,但 Transformer 有几十层 LayerNorm,层层传递下这个偏差会被放大,最终可能表现为"模型转换后指标掉了 0.1~0.5 个点,但结构、权重都看起来完全对齐",非常难排查,因为没有任何报错,只有一个说不清楚哪里来的小数值漂移。

**面试怎么问 + 追问链:**
- **Q:** "LayerNorm 的默认 `epsilon` 在 TF 和 PyTorch 里一样吗?" —— 期望准确说出不一样(`1e-3` vs `1e-5`),这是一个纯粹靠"有没有实际读过/验证过两边默认值"决定能不能答上来的细节题。
- **追问 1:** "这个 epsilon 差异什么时候会真正影响到数值结果?" —— 期望答出"当归一化维度上的方差本身很小、和 epsilon 量级接近时影响明显;方差远大于 epsilon 时,两边结果几乎没有可观测差异",不是死记"有影响"或"没影响",而是能说出影响产生的条件。
- **追问 2:** "PyTorch 的 `elementwise_affine=False` 和 TF 的哪个参数组合等价?" —— 期望答出 `center=False, scale=False` 同时关闭。
- **追问 3(深挖到轴设计):** "如果我想对一个 `(batch,H,W,C)` 的张量,只在 `H` 这一个维度上做归一化(不动 `W`、`C`),TF 和 PyTorch 分别怎么写?" —— TF 直接 `axis=1`(如果 `H` 是第 1 维);PyTorch 的 `normalized_shape` 只能表达尾部连续维度,必须先 `permute` 把 `H` 换到最后一维,算完再换回来,是一个结构性限制,不是"不会写"的问题。

**常见坑:**
- 跨框架对比/迁移模型时,只对齐了可训练参数(gamma/beta),忽略了 `epsilon` 这种"看起来是无关紧要的默认值",造成难以定位的小数值偏差。
- 想用 TF 的 `center`/`scale` 独立开关表达 PyTorch 语义时,忘记 PyTorch 端根本没有对应的"只要一半"的写法,如果要在 PyTorch 复现"只有 beta 没有 gamma"的效果,需要手动把 `weight` 固定成全 1 常量并从优化器参数里排除。
- 遇到需要归一化非尾部维度的场景,不知道 TF 的 `axis` 能直接指定,绕了一圈用 `permute`/`transpose` 手动实现,徒增代码复杂度和出错概率。

---

## 5. `Dropout`——inverted scale 技巧相同,但 `training` 参数没有隐式状态

**是什么:**
```
tf.keras.layers.Dropout(
    rate,              # 丢弃概率,对应 PyTorch 的 p —— 语义相同,只是参数名不同(这里没有反!)
    noise_shape=None,   # TF 特有:自定义 mask 的广播形状,可以用来实现"整个通道一起丢弃"
    seed=None,
)
# layer(x, training=True)  -- 训练模式,丢弃+放大
# layer(x, training=False) -- 推理模式,恒等
# layer(x)                 -- 不传 training 时,默认 False(推理行为,不丢弃!)
```

**一句话:** inverted-dropout 的 scale 技巧(丢弃后把存活值放大 `1/keep_prob`)和 torch04 第 6 节完全相同,已现场验证 TF 这边的缩放系数分毫不差;这一节真正的差异不在数学,而在"训练/推理模式是怎么被决定的"这件事上——PyTorch 靠 `self.training` 隐式状态,TF 的 `training` 参数没有默认的"当前模式",必须显式传入。

**底层机制/为什么这样设计:**

先说清楚**没有反的地方**:TF 的 `rate` 就是丢弃概率,和 PyTorch 的 `p` 语义完全一致(不要因为刚学完 BatchNorm 的 `momentum` 教训就怀疑这里是不是也反了——这里两个框架的参数名不同,但语义方向相同)。inverted scale 的动机和公式(`output = mask * x / keep_prob`,让训练/推理阶段的期望激活量级对齐)和 torch04 第 6 节完全相同,已现场用大数定律验证过缩放系数,这里不重复推导。

真正的差异在于**"现在是训练模式还是推理模式"这件事是怎么被这个层知道的**。PyTorch 的 `nn.Dropout` 是一个有状态的 `Module`,`self.training` 这个布尔属性由外层调用 `.train()`/`.eval()` 时**自动联动设置到模型树里的每一个子模块**,`forward` 里直接读 `self.training`,不需要调用者每次手动传。TF 的 `Dropout.call(inputs, training=None)` **没有这样一个由 `.train()`/`.eval()` 联动的隐式状态**——如果直接单独调用 `dropout_layer(x)`(不传 `training`),`training` 默认是 `None`/`False`,行为退化成推理模式的恒等函数,哪怕主观上"正在训练"。只有当这个 `Dropout` 层是某个 `Model`(`Sequential`/`Functional`/子类化)的一部分,并且通过 `model.fit()` 或显式 `model(x, training=True)` 调用整个模型时,Keras 才会把外层传入的 `training` 状态自动传播给每一个子层——这个自动传播机制本身会在 [04-keras-api-internals.md](04-keras-api-internals.md) 展开讲,这里只需要知道结论:**脱离了这套传播机制单独调用 Dropout 层,或者在自定义 `call()` 里调用 `self.dropout(x)` 却忘记转发 `training` 参数,都会静默变成推理模式**,和 PyTorch 里"用 `F.dropout` 忘记传 `training=self.training`"是同一类坑(torch04 第 2/6 节提到过),但 TF 这边更容易在"手写训练循环 + 自定义 `call()`"场景里被忽略,因为不传参数不会报错,行为看起来"正常"(只是没有真的在丢弃)。

**TF 特有的 `noise_shape`:** PyTorch 要实现"整个通道一起丢弃"(而不是逐像素独立丢弃)需要换用专门的 `nn.Dropout2d`/`nn.Dropout3d` 类(torch04 第 6 节追问 3 提到过)。TF 用同一个 `Dropout` 类,通过 `noise_shape` 参数控制 mask 在哪些维度上"共享"(广播):比如 `(N,H,W,C)` 的输入,`noise_shape=[N,1,1,C]` 让同一个样本、同一个通道内,所有空间位置共用同一个 0/1 决定,等价于 PyTorch `Dropout2d` 的效果(已现场验证:同一通道内要么全部存活要么全部置零)——这是"一个参数化的通用机制"和"几个专门类"两种设计哲学的差异,TF 也另外提供了 `SpatialDropout1D/2D/3D` 作为这个思路的现成封装,内部就是预置了对应的 `noise_shape`。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

p = 0.3
keep_prob = 1 - p
drop = tf.keras.layers.Dropout(rate=p)
x = tf.ones([200_000])

# ---- inverted scale 和 torch04 第6节完全相同 ----
out = drop(x, training=True)
survive_frac = (out.numpy() != 0).mean()
assert abs(survive_frac - keep_prob) < 0.01
survived = out.numpy()[out.numpy() != 0]
assert np.allclose(survived, 1 / keep_prob, atol=1e-5)
assert abs(out.numpy().mean() - 1.0) < 0.01

# ---- training 参数没有隐式状态,必须显式传 ----
assert np.array_equal(drop(x, training=False).numpy(), x.numpy())     # 显式 False -> 恒等
assert np.array_equal(drop(x).numpy(), x.numpy())                       # 不传 training -> 默认 False,同样是恒等!
# 对比 PyTorch: nn.Dropout 由外层 .train()/.eval() 自动联动设置 self.training,不需要每次调用手动传

# ---- noise_shape: 整个通道一起丢弃,等价于 PyTorch 的 nn.Dropout2d ----
x_img = tf.ones([2, 4, 4, 3])                                    # (N,H,W,C)
drop_channel = tf.keras.layers.Dropout(rate=0.5, noise_shape=[2, 1, 1, 3])
out_img = drop_channel(x_img, training=True).numpy()
for b in range(2):
    for c in range(3):
        channel_vals = out_img[b, :, :, c]
        assert np.all(channel_vals == 0) or np.all(channel_vals != 0)   # 同一通道内,要么全丢要么全留

print("TF version:", tf.__version__)
```

**AI 研究/工程场景:** 用 Keras 子类化 API(`Model` 子类,自定义 `call()`)手写一个非标准结构的网络,并且没有用 `model.fit()`(比如强化学习场景需要手写训练循环、和环境交互),在 `call(self, x, training=None)` 里调用了 `self.dropout(x)` 但忘记写成 `self.dropout(x, training=training)`——这个模型训练几十个 epoch 后效果始终不如预期(因为 Dropout 从头到尾都没真正生效,正则化效果为零,和一个没加 Dropout 的模型没有区别),但代码本身完全不报错、看起来"结构正确",是一类需要专门经验才能快速定位的隐蔽 bug。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `Dropout` 层怎么知道现在是训练还是推理模式?" —— 期望答出"不像 PyTorch 靠 `self.training` 自动联动,TF 的 `training` 参数需要在调用时显式传入,没有传的话默认是 False(推理行为)"。
- **追问 1(本节最容易考出经验的问题):** "如果在自定义 `Model` 的 `call()` 方法里写 `x = self.dropout(x)`,不传 `training`,会发生什么?" —— 期望准确答出"这个 Dropout 层永远表现为推理模式(不丢弃),不会报错,模型会正常训练,只是失去了 Dropout 应有的正则化效果",并能提到这是一个真实、隐蔽、只能靠代码审查或现场验证发现的 bug。
- **追问 2:** "如果想对卷积层的输出做'整个通道一起丢弃'而不是逐像素丢弃,TF 有几种写法?" —— 期望至少提到 `noise_shape` 参数和 `SpatialDropout2D` 这个现成的类,并能说出两者本质是同一个机制的不同封装层级。
- **追问 3(确认没有掉入过度泛化的陷阱):** "`rate` 和 PyTorch 的 `p` 含义完全一致吗?会不会也像 BatchNorm 的 momentum 一样反过来?" —— 期望候选人能明确说"不会反,这里就是同一个语义换了个名字",体现出候选人不是在"背结论"而是对每个参数都独立核实过,不会把 BatchNorm 那个特例过度泛化成"TF 和 PyTorch 什么都是反的"。

**常见坑:**
- 在自定义 `call()` 里调用子层 Dropout 时忘记转发 `training` 参数,静默退化成推理模式,没有任何报错信号。
- 单独调用一个 Dropout 层做单元测试/调试(比如在 notebook 里直接 `dropout_layer(x)`)时,因为没有传 `training=True`,会观察到"Dropout 好像没生效",容易误判成层本身有 bug,而实际上只是调用方式的问题。
- 把 BatchNorm 的 `momentum` 反转教训过度泛化,怀疑 `rate`/`p` 是不是也语义相反——这里没有反,过度怀疑同样会导致跨框架迁移代码时引入不必要的换算错误。

---

## 6. `Embedding`——稀疏梯度默认开启,`mask_zero` 不是 `padding_idx` 的等价物

**是什么:**
```
tf.keras.layers.Embedding(
    input_dim,                           # 词表大小,对应 PyTorch 的 num_embeddings
    output_dim,                           # 词向量维度,对应 PyTorch 的 embedding_dim
    embeddings_initializer='uniform',      # 默认 RandomUniform(-0.05,0.05),PyTorch 默认 N(0,1)
    mask_zero=False,                        # !! TF特有,不是 PyTorch padding_idx 的等价物
)
```

**一句话:** 稀疏梯度的数学本质——一个 batch 只让词表里出现过的行有非零梯度、重复 token 的梯度是求和不是取一次或平均——和 torch04 第 7 节完全相同,这里不重复推导;TF 不需要任何类似 `sparse=True` 的开关,`GradientTape` 对 Embedding 变量算出的梯度**默认且无条件**就是稀疏表示(`tf.IndexedSlices`),这是和 PyTorch 最大的 API 设计差异,而 `mask_zero` 这个名字最容易让人望文生义成 PyTorch 的 `padding_idx`,实际上是两个不同的机制。

**底层机制/为什么这样设计:**

"一个 batch 只碰到词表的一小撮行,反向传播时没碰到的行梯度精确为 0,重复出现的 token 梯度是所有出现位置的和"这条结论和 torch04 第 7 节完全相同,已现场验证:`token_ids=[37,37,512,9999,1,512]` 这个 batch 里,词表 1 万行中只有 4 行有非零梯度,且 token 37 出现两次,梯度精确是出现一次时的 2 倍——这里不重复推导。

**差异 1:稀疏梯度不需要开关,是默认且唯一的行为。** PyTorch 的 `nn.Embedding` 有 `sparse=False`(默认,梯度内容稀疏但存储稠密的普通 `Tensor`)和 `sparse=True`(梯度存储也稀疏的 `sparse_coo_tensor`,需要配合 `SparseAdam` 等专门优化器)两种模式,需要显式选择。TF 这边**只有一种行为**:`tape.gradient()` 对 Embedding 层的变量求出来的梯度,类型永远是 `tf.IndexedSlices`——已现场验证 `g.indices` 保存了这个 batch 里实际出现过的 token id(**不去重**,重复出现的 token 在 `indices` 里也重复出现,和 PyTorch `sparse_coo_tensor` 在 `.coalesce()` 之前的原始存储形态是同一个概念),`g.values` 保存每次出现各自贡献的梯度值,真正需要"这一行总共的梯度"时,用 `tf.convert_to_tensor()` 转成稠密张量(TF 会自动完成"按 indices 分组求和"这一步),Keras 的优化器都原生支持直接处理 `IndexedSlices`,不需要专门换一个"SparseXxx"优化器类。

**差异 2:默认初始化分布不同。** 已现场验证:TF 默认 `embeddings_initializer='uniform'`,具体是 `RandomUniform(minval=-0.05, maxval=0.05)`;PyTorch `nn.Embedding` 默认是标准正态分布 `N(0,1)`(已用真实 PyTorch 采样确认)。不仅分布形状不同(均匀 vs 正态),数值量级也差出一个数量级(`±0.05` vs 标准差 `1`),跨框架复现同一个实验、又没有显式指定初始化器时,起始的词向量在两边完全不是"同一个分布采样出来的"。

**差异 3(最容易踩的坑):`mask_zero` 不是 `padding_idx` 的等价物。** 这是两个解决不同问题的机制,名字和"padding token 相关"这个直觉容易让人误以为是一回事,现场验证结论:

- PyTorch 的 `padding_idx`:**强制**把这一行 embedding **初始化成精确的全 0 向量**,并且**冻结**它的梯度——已现场验证 `nn.Embedding(...,padding_idx=0)` 反向传播后 `weight.grad[0]` 精确是全 0,`optimizer.step()` 之后 `weight[0]` 依然精确是全 0,这一行参数**永远不会被更新**。
- TF 的 `mask_zero=True`:**不会**把 `input_dim` 里第 0 行的 embedding 向量置零——已现场验证,`mask_zero=True` 的 `Embedding` 层,查询 token id 0 依然返回一个正常的、非零的、可训练的随机初始化向量。`mask_zero=True` 真正做的事情是让这个层多出一个 `compute_mask()` 方法,产生一个**独立的布尔张量**(标记哪些位置的 token id 是 0),这个布尔张量会被**下游**支持 masking 的层(比如 `LSTM`/`GRU`/`Attention`)自动读取、用来决定"计算的时候要不要跳过这个时间步",但 `Embedding` 层自己的输出**并不会**因为 `mask_zero=True` 而在 padding 位置变成 0 向量。

如果需要的其实是 PyTorch `padding_idx` 那种"这一行参数永远锁定为 0"的效果,`mask_zero` 做不到,需要自己额外处理(比如初始化后手动把第 0 行清零,并且在每次更新后重新清零,或者自定义一个梯度 mask)。

**可运行例子:**
```python
import tensorflow as tf
import numpy as np

vocab_size, embed_dim = 10_000, 8
emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

token_ids = tf.constant([37, 37, 512, 9999, 1, 512])
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(emb(token_ids))
g = tape.gradient(loss, emb.trainable_variables)[0]

# ---- 差异1: 不需要sparse=True开关,默认就是 IndexedSlices ----
assert isinstance(g, tf.IndexedSlices)
assert sorted(g.indices.numpy().tolist()) == [1, 37, 37, 512, 512, 9999]   # indices 不去重,原始出现顺序/次数

dense_g = tf.convert_to_tensor(g)     # 转稠密时自动按 indices 分组求和
assert dense_g.shape == (vocab_size, embed_dim)
nonzero_rows = sorted(np.nonzero(np.any(dense_g.numpy() != 0, axis=1))[0].tolist())
assert nonzero_rows == [1, 37, 512, 9999]
assert np.allclose(dense_g.numpy()[37], 2.0)      # 出现2次 -> 梯度是1次的2倍(求和,不是平均)

# ---- 差异2: 默认初始化分布不同 ----
init_config = emb.embeddings_initializer.get_config()
assert init_config["minval"] == -0.05 and init_config["maxval"] == 0.05     # TF: 均匀分布[-0.05,0.05]
# PyTorch nn.Embedding 默认是 N(0,1) 标准正态分布 —— 分布形状、数值量级都不一样

# ---- 差异3: mask_zero 不等于 padding_idx,不会把 embedding 向量置零 ----
emb_mask = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
row0 = emb_mask(tf.constant([0])).numpy()
assert not np.allclose(row0, 0.0)              # index 0 的向量依然是正常的非零随机初始化!

ids = tf.constant([[5, 7, 0, 0]])
mask = emb_mask.compute_mask(ids)
assert mask.numpy().tolist() == [[True, True, False, False]]   # mask_zero 只产生一个独立的布尔mask
out = emb_mask(ids)
assert not np.allclose(out.numpy()[0, 2], 0.0)    # 对应 padding 位置的输出向量本身依然不是0

print("TF version:", tf.__version__)
```

实测(WSL2 `~/tf-venv`,TF 2.21.0):`type(g)` 精确是 `tensorflow.python.framework.indexed_slices.IndexedSlices`;另在 Windows 侧用真实 PyTorch 2.11.0+cu128 验证 `padding_idx=0`:反向传播后 `weight.grad[0]` 精确全 0,`optimizer.step()` 之后 `weight[0]` 依然精确全 0(冻结),而同一次 PyTorch 实验里不设 `padding_idx` 的默认初始化采样值形如 `[0.6186,-0.1108,-0.0696,-0.4646]`,明显是标准正态分布量级,和 TF 默认的 `[-0.05,0.05]` 均匀分布完全不是同一回事。

**AI 研究/工程场景:** 一个变长文本分类任务,序列 pad 到定长后丢给 `Embedding(mask_zero=True) -> LSTM -> Dense`,团队里有成员误以为"设了 `mask_zero=True` 就等价于 padding 位置的信息已经被清零、后面怎么处理都安全",于是在 `LSTM` 之后接了一个不支持 mask 传播的自定义层(比如某个手写的 pooling 操作忽略了 `mask`),padding 位置那些"看似无害、其实非零"的 embedding 向量就会实际参与计算、污染最终结果——这类 bug 的排查关键就是分清楚"`mask_zero` 只是传递了一个'建议忽略这里'的信号,不代表数据本身被清空了",下游每一层是否真正遵守这个信号,要逐层确认。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `Embedding` 层反向传播,梯度是稀疏的吗?需要像 PyTorch 一样显式打开一个开关吗?" —— 期望答出"梯度本质上和 PyTorch 一样是稀疏的(没被 batch 碰到的行梯度为0),但 TF 不需要开关,`GradientTape` 算出来的默认就是 `tf.IndexedSlices`"。
- **追问 1(本节最关键的追问):** "`mask_zero=True` 和 PyTorch 的 `padding_idx` 是一回事吗?" —— 期望明确答"不是",并能说清楚两者的具体区别(一个只产生下游可选遵守的布尔 mask,一个强制锁定参数本身为 0)。这是一个专门用来分辨"读过官方文档 vs 望文生义"的问题。
- **追问 2(开放题):** "如果确实需要 PyTorch `padding_idx` 那种'这一行参数永远是 0,梯度也永远不更新'的效果,在 TF 里怎么实现?" —— 期望候选人能想到手动方案的方向(初始化时清零第 0 行,每次 `optimizer.apply_gradients` 之后手动把第 0 行重新置零,或者自定义一个梯度 mask 在反向传播阶段就拦截掉第 0 行的梯度),不强求现场写出完整代码,但要能说清楚"TF 没有一个内置参数能一步做到,需要自己在训练循环里额外处理"。
- **追问 3:** "同一个 token 在 batch 里出现 5 次,它的梯度是什么?" —— 呼应 torch04 第7节:5 次出现各自贡献的梯度**求和**,不是只算一次也不是平均,TF 这边转成稠密张量之后已现场验证同样成立。

**常见坑:**
- 把 `mask_zero=True` 当成"padding 位置已经被清零、可以放心处理"的保证——实际上只是给下游层提供了一个可选遵守的信号,`Embedding` 层自己的输出没有被清零,数据依然"活着"。
- 以为 TF 需要类似 PyTorch `sparse=True` 的显式开关才能拿到稀疏梯度——不需要,这是默认且唯一的行为,反而是"想要稠密梯度"在 TF 里没有直接对应的开关(要拿到稠密表示需要自己 `tf.convert_to_tensor`)。
- 跨框架复现实验时忽略默认初始化分布的差异,复现出来的"起始条件"其实并不一致。

---

## 7. `MultiHeadAttention`——同一套拆头数学,三个独立 `EinsumDense` 子层 + `einsum`

**是什么:**
```
tf.keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,                 # 每个头里 Q/K 的维度,必须显式指定(PyTorch 是 embed_dim/num_heads 自动算出)
    value_dim=None,            # 默认等于 key_dim,但可以独立设置 —— PyTorch 做不到
    output_shape=None,          # 默认等于输入最后一维,但可以独立设置 —— PyTorch 输出维度固定等于 embed_dim
    attention_axes=None,         # TF特有:只在指定的轴上做attention,PyTorch没有这个概念
)
# 调用: mha(query, value, key=None, ...)   -- key 默认等于 value(不是等于 query,注意参数顺序)
```

**一句话:** Q/K/V 拆分、缩放点积、多头独立算 attention 再合并这套数学和 torch04 第 8 节完全相同,这里不重复推导;TF 的内部实现结构和 PyTorch 有一个根本性差异——PyTorch 用一个融合的 `in_proj_weight` 大矩阵一次算完 QKV 再切开,TF 用三个独立的 `EinsumDense` 子层分别投影 Q/K/V,并且大量依赖 `einsum`(交叉引用 [numpy-deep-dive/06-linear-algebra.md](../numpy-deep-dive/06-linear-algebra.md) 第 15 节,这里不重讲下标记法怎么读,只讲这几个具体的 einsum 表达式在这里干了什么)。

**底层机制/为什么这样设计:**

拆头的原理(把 `embed_dim` 切成 `num_heads` 份、每份独立算 attention、缩放因子为什么是 `sqrt(head_dim)`)和 torch04 第 8 节完全相同,这里不重复推导。下面是三处结构性差异。

**差异 1:三个独立 `EinsumDense` 子层,不是一个融合矩阵。** 已现场内省真实层的内部结构:`mha._query_dense`/`_key_dense`/`_value_dense`/`_output_dense` 都是 `tf.keras.layers.EinsumDense` 的实例(一种"自带 einsum 表达式的 Dense"),而不是 PyTorch 那种拼在一起的单个 `(3*embed_dim,embed_dim)` 大矩阵。`_query_dense` 的 `.equation` 属性直接就是 `'abc,cde->abde'`——`a,b,c` 对应输入的 `(batch,seq,embed_dim)`,`c,d,e` 对应 kernel 的 `(embed_dim,num_heads,head_dim)`,`c` 这个字母在两个输入里重复但不出现在输出里,被求和收缩掉(numpy06 第 15 节讲过的"重复字母收缩"规则,不是新知识);输出 `abde` 保留了 `(batch,seq,heads,head_dim)` 四个维度——**一次 einsum 调用同时完成了"投影 + reshape 成多头形状"两件事**,对比 torch04 第 8 节 PyTorch 手写实现里"先 `F.linear` 投影、再手动 `.view().transpose()` 拆头"的两步走,是同一个数学操作的不同工程实现路径。已现场验证:虽然实现结构不同,Q+K+V 三个 `EinsumDense` 的参数总量,和 PyTorch 融合矩阵 `3*embed_dim*embed_dim` 的参数量**完全相等**——只是"要不要把三个矩阵拼成一个"的存储实现选择,不影响数学和参数量。

**差异 2:`key_dim`/`value_dim` 可以独立设置,PyTorch 的 Q/K/V 被迫共享同一个 `head_dim`。** PyTorch 的 `nn.MultiheadAttention` 只接受一个 `embed_dim`,`head_dim=embed_dim/num_heads` 是唯一值,Q/K/V 三者的每头维度必须相等。TF 允许 `key_dim`(Q 和 K 的每头维度)和 `value_dim`(V 的每头维度,默认等于 `key_dim`,但可以单独指定)独立设置——已现场验证 `key_dim=4,value_dim=6` 可以同时工作:`_query_dense`/`_key_dense` 的 kernel 最后一维是 4,`_value_dense` 的 kernel 最后一维是 6,`_output_dense` 负责把拼接后的多头 value(总维度 `num_heads*value_dim`)重新投影回目标输出维度,形状能对上是因为 attention score 的计算只依赖 Q/K 的维度(两者要一致才能做点积),不依赖 V 的维度,V 的维度只在最后加权求和、拼接、输出投影这几步里起作用——这是 Transformer 原始论文里 `d_k`/`d_v` 本来就是两个独立超参数的自由度,PyTorch 的实现选择把它们锁死为相等,TF 保留了这个自由度。

**差异 3:`output_shape` 独立于输入 `embed_dim`,`attention_axes` 是纯 TF 特有能力。** PyTorch 的输出维度永远等于输入的 `embed_dim`(因为 `out_proj` 固定是方阵)。TF 允许显式指定 `output_shape`,让输出投影到任意维度——已现场验证 `embed_dim=8` 的输入配合 `output_shape=16` 能正常工作,输出最后一维是 16 不是 8。`attention_axes` 则解决了一个 PyTorch 完全没有对应能力的问题:输入如果是图像/视频这类高维张量(比如 `(batch,H,W,C)`),只想在某一个空间轴(比如 `W`)上做 attention、其他轴保持独立,PyTorch 的 `nn.MultiheadAttention` 只理解"一个序列维度",必须手动 reshape/permute 把想要的维度捏成三维 `(batch,seq,embed)` 才能塞进去;TF 直接传 `attention_axes=(2,)` 就能表达"只在第 2 维上做 attention"——已现场验证:`(2,4,6,8)` 的 4D 输入(可以理解成 `batch=2,H=4,W=6,C=8`),`attention_axes=(2,)` 之后输出形状保持 `(2,4,6,8)` 不变,返回的 attention 权重形状是 `(2,4,2,6,6)`(`batch,H,heads,W,W`)——`H` 这一维被当成了额外的批量维,只有 `W` 这一维真正参与了 attention 的 `query x key` 组合,这是 PyTorch 原生 API 结构性做不到、必须手写 reshape 才能实现的能力。

**可运行例子:**
```python
import math
import tensorflow as tf
import numpy as np

np.random.seed(0)
embed_dim, num_heads = 8, 2
head_dim = embed_dim // num_heads
batch, seq_len = 3, 5

mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)
x = tf.constant(np.random.randn(batch, seq_len, embed_dim).astype("float32"))
out = mha(x, x, x)
assert out.shape == (batch, seq_len, embed_dim)

# ---- 差异1: 三个独立 EinsumDense,不是一个融合矩阵;总参数量相等 ----
assert type(mha._query_dense).__name__ == "EinsumDense"
assert mha._query_dense.equation == "abc,cde->abde"
assert mha._query_dense.kernel.shape == (embed_dim, num_heads, head_dim)
assert mha._output_dense.equation == "abcd,cde->abe"
total_qkv_params = (mha._query_dense.kernel.numpy().size + mha._key_dense.kernel.numpy().size
                     + mha._value_dense.kernel.numpy().size)
assert total_qkv_params == 3 * embed_dim * embed_dim   # 和 PyTorch 融合 in_proj_weight 的参数量完全相等

# ---- 差异2: key_dim / value_dim 独立设置,PyTorch 做不到 ----
mha_diff = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, value_dim=6)
out_diff = mha_diff(x, x, x)
assert mha_diff._query_dense.kernel.shape[-1] == 4     # key_dim
assert mha_diff._value_dense.kernel.shape[-1] == 6     # value_dim,独立于key_dim
assert out_diff.shape == (batch, seq_len, embed_dim)     # 最终仍投影回embed_dim(由_output_dense负责)

# ---- 差异3: output_shape 独立于输入embed_dim;attention_axes 是纯TF能力 ----
mha_outshape = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, output_shape=16)
out_outshape = mha_outshape(x, x, x)
assert out_outshape.shape == (batch, seq_len, 16)          # PyTorch 做不到:输出维度固定等于embed_dim

x_img = tf.constant(np.random.randn(2, 4, 6, 8).astype("float32"))    # (batch, H, W, C)
mha_axes = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, attention_axes=(2,))
out_axes, weights_axes = mha_axes(x_img, x_img, x_img, return_attention_scores=True)
assert out_axes.shape == (2, 4, 6, 8)             # 形状不变
assert weights_axes.shape == (2, 4, 2, 6, 6)        # (batch, H, heads, W, W) -- 只在W轴上做了attention

# ---- einsum 交叉验证:手写复现层内部的等价计算,和真实层输出数值吻合(不重讲einsum语法,只用它) ----
Wq, bq = mha._query_dense.kernel.numpy(), mha._query_dense.bias.numpy()
Wk, bk = mha._key_dense.kernel.numpy(), mha._key_dense.bias.numpy()
Wv, bv = mha._value_dense.kernel.numpy(), mha._value_dense.bias.numpy()
Wo, bo = mha._output_dense.kernel.numpy(), mha._output_dense.bias.numpy()
xin = x.numpy()
q = np.einsum('abc,cde->abde', xin, Wq) + bq            # (batch,seq,embed) x (embed,heads,head_dim)
k = np.einsum('abc,cde->abde', xin, Wk) + bk
v = np.einsum('abc,cde->abde', xin, Wv) + bv
scores = np.einsum('aecd,abcd->acbe', k, q) / math.sqrt(head_dim)   # (batch,heads,seq_q,seq_k)
weights = tf.nn.softmax(scores, axis=-1).numpy()
attn_out = np.einsum('acbe,aecd->abcd', weights, v)
manual_out = np.einsum('abcd,cde->abe', attn_out, Wo) + bo          # 合并多头 + 输出投影,一步到位
assert np.allclose(manual_out, out.numpy(), atol=1e-4)

print("TF version:", tf.__version__)
```

**AI 研究/工程场景:** 把一份 HuggingFace/PyTorch 训练好的 Transformer 权重迁移到 TF 侧做部署(常见于同一个模型要提供两套推理服务)时,`MultiHeadAttention` 是迁移脚本里最容易出错的一层:PyTorch 融合的 `in_proj_weight`(形状 `(3*embed_dim,embed_dim)`)必须先按 `embed_dim` 切成 Q/K/V 三段,再分别 reshape 成 TF 每个 `EinsumDense` 期望的 `(embed_dim,num_heads,head_dim)` 形状(不是简单的 reshape,还涉及轴的重新排列),`out_proj` 同理要从 `(embed_dim,embed_dim)` 重新排列成 TF `_output_dense` 期望的 `(num_heads,head_dim,embed_dim)`——任何一步轴序排错,模型都能正常跑、输出形状也完全正确,但数值是错的,且错误往往只在做端到端精度回归测试时才会被发现。理解本节内省出来的这几个 `.equation` 字符串和 kernel 形状,是写对这类权重迁移脚本、以及排查"迁移后精度下降"问题的必备知识。

**面试怎么问 + 追问链:**
- **Q:** "TF 的 `MultiHeadAttention` 内部是怎么实现 Q/K/V 拆分的?和 PyTorch 的实现方式有什么本质不同?" —— 期望先说"数学上和 PyTorch(torch04第8节)完全相同",再说出"TF 用三个独立的 EinsumDense 子层,PyTorch 用一个融合矩阵切开"这个结构性差异,并能提到两边总参数量其实相等。
- **追问 1(深挖 einsum 的实际作用):** "`_query_dense` 的 einsum 表达式 `'abc,cde->abde'` 具体在做什么?" —— 期望能对照 numpy06 讲过的下标规则现场解读:`c` 重复但不出现在输出里,是被收缩的维度(对应 `embed_dim` 上的矩阵乘法收缩);`a,b`(batch,seq)和 `d,e`(heads,head_dim)都在输入输出里保留,是这次投影同时完成"线性变换+reshape成多头形状"的原因。
- **追问 2:** "TF 允许 `key_dim` 和 `value_dim` 不一样,这在数学上意味着什么?PyTorch 能做到吗?" —— 期望答出"意味着 attention score 的计算(依赖 Q/K 维度必须相等做点积)和最终 value 的加权求和(不要求和 Q/K 维度相等)是两件独立的事,V 的维度可以自由选择;PyTorch 做不到,`embed_dim/num_heads` 是唯一值,Q/K/V 被迫共享"。
- **追问 3(考察知识广度):** "`attention_axes` 解决了一个什么样的实际问题?举一个 PyTorch 原生 API 做不到、必须手动 reshape 才能实现的场景。" —— 期望举出图像/视频这类高维输入,只想在某个特定的空间轴上做 attention(比如只在宽度方向、不影响高度方向)的场景。
- **追问 4:** "如果要把一个 PyTorch 训练好的 `nn.MultiheadAttention` 权重迁移到 TF 的 `MultiHeadAttention`,需要做哪几步形状变换?" —— 期望说出"融合的 `in_proj_weight` 要先按 `embed_dim` 切成三段(Q/K/V),再各自 reshape 成 `(embed_dim,num_heads,head_dim)`;`out_proj` 要 reshape 成 `(num_heads,head_dim,embed_dim)`",体现出候选人真的理解两边的 kernel 形状差异,不是只停留在"知道两边实现不同"这个表层结论。

**常见坑:**
- 以为 TF 三个独立子层的实现方式意味着参数量或计算量比 PyTorch 更大——实际总参数量完全相等,只是存储成一个矩阵还是三个矩阵的工程选择。
- 想当然认为 PyTorch 也支持类似 `key_dim`/`value_dim` 独立设置——它不支持,`embed_dim/num_heads` 是唯一值,需要自己手写或用第三方实现才能得到这个自由度。
- 手动迁移权重时,只做了"简单 reshape"而没有意识到还需要正确的轴重新排列(尤其是 `_output_dense` 的 `(num_heads,head_dim,embed_dim)` 这种三维 kernel,和 PyTorch 二维 `out_proj.weight` 之间不是一次 `.reshape()` 能对上的,需要先 reshape 再 transpose 到正确的轴序),导致迁移后的模型形状正确但数值错误。

---

## 小结:这一批 7 个知识点解决的问题

| # | 知识点 | 与 PyTorch 的核心差异 |
|---|------|---------|
| 1 | `Dense` | `kernel` 形状 `(in,out)`,和 `nn.Linear.weight` 的 `(out,in)` 互为转置;`build()` 延迟构建,不需要预先指定输入维度 |
| 2 | `Conv2D` | 默认 `data_format='channels_last'`(NHWC),kernel 轴序 `(kH,kW,C_in,C_out)`,均与 PyTorch 固定的 NCHW/`(C_out,C_in,kH,kW)` 相反;`padding='same'` 对任意 stride 都适用(可能不对称填充),PyTorch 的 `'same'` 只支持 stride=1 |
| 3 | `BatchNormalization` | `momentum` 语义和 PyTorch **相反**(TF 是"保留旧值"权重,PyTorch 是"采用新值"权重),默认值也不同;`moving_variance` 用有偏方差更新(PyTorch 用无偏);`batch_size=1` 训练模式不报错(PyTorch 会报错) |
| 4 | `LayerNormalization` | 默认 `epsilon` 相差 100 倍(1e-3 vs 1e-5);`center`/`scale` 可独立开关(PyTorch 只有一个联合开关);`axis` 可以是任意轴甚至任意一组轴(PyTorch `normalized_shape` 只能是尾部连续维度) |
| 5 | `Dropout` | `rate`/`p` 语义相同(没有反);`training` 参数没有隐式状态,必须显式传入,否则默认表现为推理模式;`noise_shape` 用一个通用机制实现 PyTorch 需要专门类(`Dropout2d`)才能做到的整通道丢弃 |
| 6 | `Embedding` | 反向传播梯度默认且无条件是 `tf.IndexedSlices`,不需要 `sparse=True` 开关;默认初始化是均匀分布(PyTorch 是正态分布);`mask_zero` 不等价于 `padding_idx`,不会把 embedding 向量置零或冻结梯度 |
| 7 | `MultiHeadAttention` | Q/K/V 用三个独立 `EinsumDense` 子层实现(PyTorch 用一个融合矩阵),总参数量相等;`key_dim`/`value_dim` 可独立设置;`output_shape`/`attention_axes` 是 PyTorch 原生 API 没有的能力 |

**这一批的核心方法论,也是全系列的方法论:** 数学推导交给 torch04(它已经用代码验证过每一条梯度公式),本篇每一个"差异"结论背后都有现场跑出来的真实数字或真实报错文本支撑——尤其是 BatchNorm 的 `momentum` 反转,只有现场用同一组输入代入两边公式算出两个不同的数字、再验证 `momentum_TF=1-momentum_PyTorch` 时数值重合,才真正"证明"了这个反转,而不是"听说反了"。

下一批:[06-loss-functions-and-numerical-stability.md](06-loss-functions-and-numerical-stability.md)

---

*更新:2026-07-09*
