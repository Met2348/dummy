# 12 · 手把手实战:从零搭一个迷你图像处理管线

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 12 个"函数专题",不计入"约 120 个函数"的统计——和 [11-advanced-interview-depth.md](11-advanced-interview-depth.md)是同一挂,但风格不一样:11 号文件里,你是**旁观者**,跟着三个多级追问案例把"这个坑是怎么被现场发现的"过一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真实能跑的小工具。这个格式最早在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证过,这里是同一套方法论第一次搬到 numpy-deep-dive。

本文所有代码例子已在仓库 `.venv`(numpy 2.4.6)下实际跑通验证,包括计时结果和 ASCII 可视化输出,不是凭空写的、也不是手算出来再倒推 assert 的。

## 为什么是"图像处理管线"

不是要发明新知识点,是把三个你已经学过的知识点串成一个真实有用的东西——**手写一个只用 numpy 的迷你图像处理管线:灰度化 → 均值模糊(卷积)→ Sobel 边缘检测**。不用 PIL、不用 opencv、不需要真实图片文件,全程自己用 numpy 画一张小的合成"图像"。选这个题目还有一个额外的好处:图像天然是"看得见"的数据,每一步的效果不仅能用 `assert` 验证,还能直接打印出来用肉眼确认——这篇教程会顺手写一个几行的 ASCII 可视化函数,让每个阶段的输出既有断言、也有图。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 把一张合成的彩色图,变成一张灰度图 | [02-shape-and-structure.md](02-shape-and-structure.md) 数组形状与 strides(图像的内存布局)、[04-elementwise-math.md](04-elementwise-math.md) 逐元素运算、[06-linear-algebra.md](06-linear-algebra.md) `dot`/`@` |
| 阶段 2 | 给灰度图做一次均值模糊,先用最笨的写法跑对 | [04-elementwise-math.md](04-elementwise-math.md) 逐元素运算(暴力循环里的加权求和) |
| 阶段 3 | 同一个模糊操作换成向量化写法,真实测出速度差多少 | [08-broadcasting-and-ufunc.md](08-broadcasting-and-ufunc.md) 广播机制、[06-linear-algebra.md](06-linear-algebra.md) 矩阵化(卷积核展平成向量 + 矩阵乘法) |
| 阶段 4 | 换一套卷积核,同一份代码从"模糊"变成"边缘检测" | [04-elementwise-math.md](04-elementwise-math.md) `sqrt`/`power`(梯度幅值) |
| 阶段 5 | 把前四步拼成一个完整的 `MiniImagePipeline` 类,在没见过的新图案上跑一次端到端 | 阶段 1-4 全部组装 |

每个阶段的代码都能独立运行(本文件用仓库统一的 `_verify_md.py` 校验——把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的函数时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:从像素到灰度图——图像本质上就是个数组

"图像"这个词听起来很视觉化,但对 numpy 来说,一张彩色图片就是一个形状为 `(高, 宽, 3)` 的三维数组——最后一维的 3 个数分别是这个像素的 R、G、B 分量。先不管真实图片文件怎么读(那是 PIL/opencv 的活),自己用 numpy 画一张最简单的合成图:背景一个颜色,中间画一个矩形块。

```python
import numpy as np

H, W = 20, 20
img = np.zeros((H, W, 3), dtype=np.uint8)
img[:, :] = (20, 20, 80)          # 背景：偏暗的蓝紫色
img[6:14, 6:14] = (200, 120, 40)  # 中间画一个 8x8 的矩形块：暖橙色

assert img.shape == (20, 20, 3)   # (高, 宽, 通道数)，最后一维是 RGB 三个通道
assert img.itemsize == 1          # uint8，每个通道占1字节

# .strides 是字节数，不是元素个数——02-shape-and-structure.md 讲过这套机制，
# 这里换成图像也是同一个推导：C连续（行优先）内存布局下，
# 沿高度方向挪一步要跨过"一整行"的字节数，沿宽度方向挪一步要跨过"一个像素(3通道)"的字节数，
# 沿通道方向挪一步只跨1字节(uint8每个元素本来就是1字节)
assert img.strides == (60, 3, 1)
assert img.strides[0] == W * 3 * img.itemsize   # 一行 = 20 个像素 * 3 通道 * 1 字节 = 60
assert img.strides[1] == 3 * img.itemsize        # 一个像素 = 3 通道 * 1 字节 = 3
assert img.strides[2] == img.itemsize            # 一个通道 = 1 字节

# 用 strides 手动验证任意一个像素/通道的地址公式，和 02 类验证 2D 数组是同一个思路，
# 只是这里多了一维(通道)：地址 = i*strides[0] + j*strides[1] + c*strides[2]
i, j, c = 8, 9, 1   # 矩形块内部某个像素的 G 通道
byte_offset = i * img.strides[0] + j * img.strides[1] + c * img.strides[2]
flat = img.reshape(-1)
assert flat[byte_offset // img.itemsize] == img[i, j, c] == 120
print("byte_offset:", byte_offset, "-> value:", int(flat[byte_offset]))
print("stage1a strides check ok")
```

`.strides` 告诉 numpy "沿每一维挪一步要跨多少字节",这和 [02-shape-and-structure.md](02-shape-and-structure.md) 里用 `.strides` 手推 2D 数组地址公式是同一套逻辑——只是从"矩阵"换成了"图像",本质没有任何区别:图像说到底就是一段摊平的一维内存,`shape`/`strides` 只是"怎么切开解读"它的说明书。

有了合成的 RGB 图,第一件真正要"实现"的事是**灰度化**:把 3 个通道压缩成 1 个亮度值。业界标准公式(ITU-R BT.601 亮度公式)不是简单平均,而是加权和:`gray = 0.299*R + 0.587*G + 0.114*B`——权重不是随便定的,是因为人眼对绿色最敏感、对蓝色最不敏感,绿色权重最大。

```python
import numpy as np

def make_rect_rgb(H=20, W=20):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 80)
    img[6:14, 6:14] = (200, 120, 40)
    return img

def ascii_art(gray, levels=".:-=+*#%@"):
    lo, hi = gray.min(), gray.max()
    span = hi - lo if hi > lo else 1.0
    lines = []
    for row in gray:
        chars = [levels[int((v - lo) / span * (len(levels) - 1))] for v in row]
        lines.append("".join(chars))
    return "\n".join(lines)

rgb = make_rect_rgb()
weights = np.array([0.299, 0.587, 0.114])

# 写法一：逐元素乘法 + 沿通道轴(axis=-1)求和——04-elementwise-math.md 的逐元素运算
gray_elementwise = (rgb.astype(np.float64) * weights).sum(axis=-1)

# 写法二：把 weights 看成一个向量，(H,W,3) @ (3,) 会沿数组最后一维做矩阵-向量乘法，
# 等价于对每个像素的 RGB 向量和 weights 向量做一次点积——06-linear-algebra.md 的 dot/@
gray_linalg = rgb.astype(np.float64) @ weights

assert np.allclose(gray_elementwise, gray_linalg)   # 两种写法，同一个结果
assert gray_elementwise.shape == (20, 20)             # 通道维被"点掉"了，从3D变回2D

assert np.isclose(gray_elementwise[0, 0], 26.84)      # 背景像素的灰度值
assert np.isclose(gray_elementwise[8, 8], 134.8)      # 矩形块像素的灰度值

print(ascii_art(gray_elementwise))
print("stage1b grayscale ok, background=%.2f rectangle=%.2f" % (gray_elementwise[0, 0], gray_elementwise[8, 8]))
```

真实打印出来的 ASCII 图长这样(`.` 是背景,`@` 是矩形块,肉眼就能确认灰度化做对了):

```
....................
....................
....................
....................
....................
....................
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
......@@@@@@@@......
....................
....................
....................
....................
....................
....................
```

**这一步让程序多会一件事:** 从"一张彩色合成图"变成"一张灰度图",而且用两种独立写法(逐元素求和 / 线性代数点积)算出了完全一致的结果——这不是巧合,`(H,W,3) @ (3,)` 本来就是"对每个像素做一次加权求和"的矩阵乘法表达方式,两条路径殊途同归。

---

## 阶段 2:暴力卷积——先把"卷积"这件事用最笨的写法跑对

均值模糊、边缘检测,本质上都是同一个操作:**卷积**——拿一个小的数字方阵(卷积核/kernel),在图像上每个位置都和周围的像素做一次"加权求和"。最直接的实现方式是四层嵌套循环:两层遍历输出图像的每个像素,两层遍历卷积核的每个位置。先写这个最笨的版本,理由和 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 阶段 1 一样——不是因为笨的版本没用,是因为**正确性要先在一个容易验证对错的实现上钉死,再谈性能优化**,不然优化后的版本"变快了但也变错了"根本发现不了。

```python
import numpy as np

def make_gray_rect():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 80)
    img[6:14, 6:14] = (200, 120, 40)
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights

def convolve2d_naive(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2      # 3x3 核 -> 上下左右各补1圈，输出和输入同样大小
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)   # 零填充(zero padding)
    padded[ph:ph + H, pw:pw + W] = img
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):              # 输出的每一行
        for j in range(W):          # 输出的每一列
            acc = 0.0
            for ki in range(kh):    # 核的每一行
                for kj in range(kw):  # 核的每一列
                    acc += kernel[ki, kj] * padded[i + ki, j + kj]
            out[i, j] = acc
    return out

mean_kernel = np.ones((3, 3)) / 9.0     # 3x3 均值核：9个格子各占1/9权重
gray = make_gray_rect()
blurred = convolve2d_naive(gray, mean_kernel)

assert blurred.shape == gray.shape

# 矩形内部远离边界的像素：3x3窗口全部落在矩形内，模糊前后应该完全不变
assert np.isclose(blurred[10, 10], gray[10, 10]) and np.isclose(blurred[10, 10], 134.8)

# 矩形边界上的像素，模糊前是"非黑即白"的硬跳变，模糊后应该出现过渡值——
# 这才是均值模糊真正在做的事：抹平突变，让相邻像素互相"渗透"
before = gray[6, 4:9]
after = blurred[6, 4:9]
assert len(set(before.tolist())) == 2               # 模糊前：只有背景值和矩形值两种
assert len(set(np.round(after, 4).tolist())) > 2      # 模糊后：出现了中间过渡值
print("before:", before)
print("after:", after)
print("stage2 naive convolution ok")
```

真实跑出来的过渡值是 `[26.84, 50.83, 74.82, 98.81, 98.81]`——从纯背景色 26.84 到矩形色 134.8,中间被均值模糊拉出了一段渐变,这正是"模糊"这个词的字面含义。

**这一步让程序多会一件事:** 图像多了一个"卷积"操作,能做均值模糊了。但这个实现有一个明显的问题还没有暴露——它到底有多慢?下一阶段先测出来,再决定要不要优化。

---

## 阶段 3:向量化滑窗卷积——广播、矩阵化,真实测速

暴力版本的四层循环里,最内层的两层(遍历卷积核的 3x3=9 个位置)其实可以整体挪到 numpy 层面去做:与其对每个输出像素单独算"9 个数加权求和",不如反过来想——**对卷积核的每一个位置,一次性把它加权乘到整张图像对应的位置上**。这样 Python 层面的循环从"图像的每个像素"(H×W 次)缩小成"卷积核的每个位置"(kh×kw 次,3x3 核只有 9 次),循环体内做的是整块数组的运算,交给 numpy 的 C 层去跑。

```python
import numpy as np
import time

def make_gray_rect(H, W):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 80)
    r0, r1 = H // 3, 2 * H // 3
    c0, c1 = W // 3, 2 * W // 3
    img[r0:r1, c0:c1] = (200, 120, 40)
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights

def convolve2d_naive(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += kernel[ki, kj] * padded[i + ki, j + kj]
            out[i, j] = acc
    return out

def convolve2d_vectorized(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    out = np.zeros((H, W), dtype=np.float64)
    # 只在"核的3x3=9个位置"上做 Python 循环，不是在"图像的每个像素"上循环；
    # 循环体内 kernel[ki,kj] 是一个标量，padded[ki:ki+H, kj:kj+W] 是一整块二维数组切片，
    # 标量乘数组、数组加数组，靠的正是广播规则（08-broadcasting-and-ufunc.md）——
    # 形状 () 的标量和形状 (H,W) 的数组能直接逐元素运算，不用手写循环去"拉伸"标量
    for ki in range(kh):
        for kj in range(kw):
            out += kernel[ki, kj] * padded[ki:ki + H, kj:kj + W]
    return out

mean_kernel = np.ones((3, 3)) / 9.0

# 先在小图上对齐结果——两种写法必须先证明"算的是同一件事"，比速度才有意义
small = make_gray_rect(20, 20)
assert np.allclose(convolve2d_naive(small, mean_kernel), convolve2d_vectorized(small, mean_kernel))

# 再在大图上真实测速
big = make_gray_rect(150, 150)

def best_of(fn, trials=3):
    best = None
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best

t_naive = best_of(lambda: convolve2d_naive(big, mean_kernel))
t_vec = best_of(lambda: convolve2d_vectorized(big, mean_kernel))
print(f"naive best-of-3: {t_naive * 1000:.2f} ms")
print(f"vectorized best-of-3: {t_vec * 1000:.4f} ms")
print(f"ratio: {t_naive / t_vec:.1f}x")

# 容差放宽到20倍(本机实测在270~330倍之间)，这里要验证的是数量级差距，不是精确复现某个具体倍数
assert t_naive > t_vec * 20
print("stage3a broadcasting speedup confirmed")
```

本机实测:150x150 的图像上,暴力版本要 84 毫秒左右,向量化版本只要 0.3 毫秒左右,**差了两百多倍**——和 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 阶段 1 里"倒排索引比线性扫描快几十万倍"是同一个道理:不是常数系数的差距,是"Python 解释器一个像素一个像素地跑"和"C 语言层面整块内存一次性算完"的量级差距。

广播不是唯一的向量化角度。回到"卷积到底在算什么"这个问题:**输出图像上的每一个像素,本来就是"卷积核"和它对应的那一小块窗口做的一次点积**。既然单个像素是点积,那么"对所有像素同时做这件事",自然可以想到线性代数的角度——把所有窗口摊平堆成一个大矩阵,卷积核也摊平成一个向量,一次矩阵-向量乘法就把全部输出算出来了:

```python
import numpy as np

def make_gray_rect():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 80)
    img[6:14, 6:14] = (200, 120, 40)
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights

def convolve2d_vectorized(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    out = np.zeros((H, W), dtype=np.float64)
    for ki in range(kh):
        for kj in range(kw):
            out += kernel[ki, kj] * padded[ki:ki + H, kj:kj + W]
    return out

def convolve2d_matmul(img, kernel):
    # 把卷积核"矩阵化"：每个输出像素本来就是 kernel 和它对应窗口的点积，
    # 那么"对所有像素同时做这件事"，就是把所有窗口摊平堆成一个大矩阵，
    # 再和摊平的 kernel 做一次矩阵-向量乘法——06-linear-algebra.md 的 @/matmul
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    # 依然只循环 kh*kw=9 次，不是 H*W 次；每次循环摊平一整块窗口切片，作为矩阵的一列
    cols = []
    for ki in range(kh):
        for kj in range(kw):
            cols.append(padded[ki:ki + H, kj:kj + W].reshape(-1))
    patches = np.stack(cols, axis=1)     # (H*W, 9)：每一行是一个像素对应的3x3窗口摊平后的向量
    flat_kernel = kernel.reshape(-1)      # (9,)：卷积核本身摊平成一个向量
    out_flat = patches @ flat_kernel       # 一次矩阵-向量乘法，算出全部 H*W 个输出
    return out_flat.reshape(H, W)

mean_kernel = np.ones((3, 3)) / 9.0
gray = make_gray_rect()

out_vec = convolve2d_vectorized(gray, mean_kernel)
out_mm = convolve2d_matmul(gray, mean_kernel)
assert np.allclose(out_vec, out_mm)     # 两种向量化写法，结果完全一致

# 验证"矩阵化"背后的直觉：单个像素的卷积结果，本来就是 kernel 和它窗口的点积
H, W = gray.shape
ph = pw = 1
padded = np.zeros((H + 2 * ph, W + 2 * pw))
padded[ph:ph + H, pw:pw + W] = gray
window_00 = padded[0:3, 0:3]
assert np.isclose(np.dot(mean_kernel.reshape(-1), window_00.reshape(-1)), out_vec[0, 0])

print("stage3b matmul convolution ok, patches shape:", (H * W, 9))
```

两种向量化写法(广播 shift-and-add / 矩阵化 im2col+matmul)算出完全一致的结果,这不是偶然——它们是同一个数学操作的两种不同实现路径,一个用"标量广播乘 + 累加"表达,一个用"矩阵乘法"表达,殊途同归。真实深度学习框架实现卷积层时,`im2col`(把窗口摊平成矩阵再乘)正是历史上最早、至今仍在用的经典实现方式之一。

**这一步让程序多会一件事:** 同一个模糊操作,从"能跑但慢"变成"跑得快,而且有两种独立验证过的向量化写法互相印证"。

---

## 阶段 4:换一套卷积核就是边缘检测——Sobel

均值模糊用的是"每个位置权重相等"的核。**卷积这个操作本身不关心核里装的是什么数字**——换一套数字,同一份 `convolve2d` 代码不用改一行,就能从"模糊"变成完全不同的另一种操作:边缘检测。Sobel 算子是最经典的边缘检测卷积核,分为两个方向:

```python
import numpy as np

def make_gray_rect():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 80)
    img[6:14, 6:14] = (200, 120, 40)
    weights = np.array([0.299, 0.587, 0.114])
    return img.astype(np.float64) @ weights

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
    padded[ph:ph + H, pw:pw + W] = img
    out = np.zeros((H, W), dtype=np.float64)
    for ki in range(kh):
        for kj in range(kw):
            out += kernel[ki, kj] * padded[ki:ki + H, kj:kj + W]
    return out

def ascii_art(gray, levels=".:-=+*#%@"):
    lo, hi = gray.min(), gray.max()
    span = hi - lo if hi > lo else 1.0
    lines = []
    for row in gray:
        chars = [levels[int((v - lo) / span * (len(levels) - 1))] for v in row]
        lines.append("".join(chars))
    return "\n".join(lines)

# Sobel 算子：两个 3x3 卷积核，分别对"水平方向"和"竖直方向"的亮度变化敏感
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

gray = make_gray_rect()
gx = convolve2d(gray, SOBEL_X)
gy = convolve2d(gray, SOBEL_Y)
mag = np.sqrt(gx ** 2 + gy ** 2)     # 梯度幅值：04-elementwise-math.md 的 power/sqrt

print(ascii_art(gray))
print()
print(ascii_art(mag))

# 矩形内部(远离所有边界)：亮度完全不变，梯度应该是0
assert mag[10, 10] < 1e-6
# 背景内部(远离矩形、也远离图像边框)：同样应该接近0
# 注意不能写 == 0——浮点加减法不保证严格抵消为0，这里量级在1e-15，用小阈值判断"足够接近0"更稳妥
assert mag[1, 1] < 1e-6

# 矩形边界：真实有亮度突变的地方，梯度幅值应该远大于flat区域
assert mag[6, 10] > 400

# 不完美的真实结果，如实记录而不是回避：图像最外面一圈(row0/row19/col0/col19)
# 梯度幅值也不是0，而是100+——这不是矩形的边，是"零填充"本身制造出来的假边缘。
# 卷积需要给图像边界补一圈数，这里补的是0，而真实图像边界的像素值是26.84(背景色)，
# 0和26.84之间存在"人为制造"的落差，Sobel会把这个落差也当成一条边缘检测出来
assert mag[0, 10] > 50          # 图像上边框：真实存在的zero-padding伪影
assert mag[0, 0] > 50           # 图像四角：zero-padding伪影的角效应更明显（两条边同时产生落差）
print("stage4 sobel edge detection ok (zero-padding border artifact confirmed, not a bug)")
```

真实打印出来的边缘图长这样(`:` 是零填充伪影围出的外框,中间那圈 `%`/`*` 才是矩形真正的边):

```
::::::::::::::::::::
:..................:
:..................:
:..................:
:..................:
:....-*%%%%%%*-....:
:....*%%%%%%%%*....:
:....%%......%%....:
:....%%......%%....:
:....%%......%%....:
:....%%......%%....:
:....%%......%%....:
:....%%......%%....:
:....*@%%%%%%@*....:
:....-*%%%%%%*-....:
:..................:
:..................:
:..................:
:..................:
::::::::::::::::::::
```

**这里必须诚实说明一个不完美的地方**(仿照 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 对 tie-break 排序结果的处理方式,不回避):这张边缘图最外面一整圈本不该出现,它不对应原图里任何真实的边缘,纯粹是"零填充"这个简化选择带来的副作用——真实图像的边界之外并不存在"亮度为 0 的一圈像素",这是卷积为了能在边界也输出结果而人为补上的,补出来的 0 和背景色 26.84 之间存在真实的数值落差,Sobel 忠实地把这个人为落差报告成了一条边。真实图像处理库(比如 `scipy.ndimage`、opencv)默认往往用"镜像"或"复制边缘像素"的填充方式而不是补 0,就是为了避免这个伪影——这篇教程为了"从零实现"的简单性选了最容易写的零填充,代价就是这一圈假边缘,如实记录在这里,不是本文的 bug,是这个简化选择的已知代价。

**这一步让程序多会一件事:** 同一份 `convolve2d`,换一套卷积核数字,从"模糊"变成了"边缘检测"——顺带诚实暴露了一个从零实现邊界处理时不可避免会遇到的真实瑕疵。

---

## 阶段 5:组装成一个完整的 `MiniImagePipeline`

把前四阶段拼进一个类,在一张**之前完全没出现过**的合成图案(对角线条纹,不再是矩形)上跑一次完整的端到端流程:灰度化 → 均值模糊(先去掉一点噪声,边缘检测前的标准做法)→ Sobel 边缘检测。

```python
import numpy as np

class MiniImagePipeline:
    """从 RGB 合成图到边缘检测结果，三步一次跑完：灰度化 -> 均值模糊 -> Sobel边缘检测。"""

    GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])
    MEAN_KERNEL = np.ones((3, 3)) / 9.0
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    @staticmethod
    def _convolve2d(img, kernel):
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        H, W = img.shape
        padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=np.float64)
        padded[ph:ph + H, pw:pw + W] = img
        out = np.zeros((H, W), dtype=np.float64)
        for ki in range(kh):
            for kj in range(kw):
                out += kernel[ki, kj] * padded[ki:ki + H, kj:kj + W]
        return out

    @classmethod
    def to_grayscale(cls, rgb):
        return rgb.astype(np.float64) @ cls.GRAY_WEIGHTS

    @classmethod
    def blur(cls, gray):
        return cls._convolve2d(gray, cls.MEAN_KERNEL)

    @classmethod
    def edges(cls, gray):
        gx = cls._convolve2d(gray, cls.SOBEL_X)
        gy = cls._convolve2d(gray, cls.SOBEL_Y)
        return np.sqrt(gx ** 2 + gy ** 2)

    @classmethod
    def process(cls, rgb):
        gray = cls.to_grayscale(rgb)
        blurred = cls.blur(gray)
        mag = cls.edges(blurred)
        return gray, blurred, mag

def ascii_art(gray, levels=".:-=+*#%@"):
    lo, hi = gray.min(), gray.max()
    span = hi - lo if hi > lo else 1.0
    lines = []
    for row in gray:
        chars = [levels[int((v - lo) / span * (len(levels) - 1))] for v in row]
        lines.append("".join(chars))
    return "\n".join(lines)

# 一张全新的合成图——对角线条纹，之前四个阶段都没见过这个图案
H, W = 20, 20
diag_rgb = np.zeros((H, W, 3), dtype=np.uint8)
diag_rgb[:, :] = (10, 40, 30)             # 背景：暗青色
for i in range(H):
    for j in range(W):
        if abs(i - j) <= 1:
            diag_rgb[i, j] = (230, 210, 40)   # 条纹：亮黄色，3像素宽

gray, blurred, mag = MiniImagePipeline.process(diag_rgb)

print(ascii_art(gray))
print()
print(ascii_art(mag))

# 远离对角线、也远离图像边框的纯背景像素：梯度应该接近0
assert mag[10, 2] < 1e-6

# 条纹两侧的过渡带(真正的亮暗跳变发生的地方)：梯度应该很大
assert mag[10, 8] > 400
assert mag[10, 12] > 400

# 反直觉但真实的发现：对角线正中间那一格(i==j)，梯度幅值算出来几乎是0，
# 不是"线越亮的地方梯度越大"这种直觉猜测。真实原因：条纹本身宽度是3像素、左右对称，
# 正中间那一格的3x3窗口里，左右两侧对称位置的亮度变化互相抵消了——
# Sobel测的是"相邻像素之间的跳变"，对称条纹的正中心恰好没有净跳变，
# 真正的跳变发生在条纹和背景的交界处(条纹两侧的过渡带)，不是条纹本身最亮的地方
assert mag[10, 10] < 1e-6
assert mag[5, 5] < 1e-6
assert mag[15, 15] < 1e-6

print("stage5 end-to-end pipeline ok on an unseen pattern")
print("flat bg:", mag[10, 2], "| flank (edge):", mag[10, 8], "| on-stripe center:", mag[10, 10])
```

真实打印出来的边缘图,能清楚看到条纹两侧各有一条独立的亮线,条纹正中间那条虚线反而是暗的:

```
%@%*=:::::::::::::::
@-*#+-.............:
%*.+#+:............:
*#+.+#+:...........:
=+#+.+#+:..........:
:-+#+.+#+:.........:
:.:+#+.+#+:........:
:..:+#+.+#+:.......:
:...:+#+.+#+:......:
:....:+#+.+#+:.....:
:.....:+#+.+#+:....:
:......:+#+.+#+:...:
:.......:+#+.+#+:..:
:........:+#+.+#+:.:
:.........:+#+.+#+-:
:..........:+#+.+#+=
:...........:+#+.+#*
:............:+#+.*%
:.............-+#*-@
:::::::::::::::=*%@%
```

`+`/`#` 组成的两条平行细线才是条纹真正的两条边,中间那条由 `:`/`.` 组成的虚线恰好是条纹本身最亮的地方——这个"最亮处梯度反而最低"的结果一开始是反直觉的,但把 Sobel 的定义("测量相邻像素之间的跳变",不是"测量亮度本身")重新想一遍,就能推导出这正是对称条纹应该有的行为,不是 bug。

**这一步让程序多会一件事:** 从"四段各自独立的代码"变成"一个类,输入任意 RGB 合成图,一次调用跑完整条流水线"——而且在一张全新图案上验证了它不是只对某一张训练时见过的图像凑巧生效。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **更精细的边缘检测:** Sobel 只是 Canny 边缘检测算法的第一步(算梯度),完整的 Canny 还要做非极大值抑制(把粗边细化成一像素宽)和双阈值滞后判断(把弱边缘和强边缘区分对待)——这篇教程只做到 Sobel 这一层,没有继续实现后两步。
- **更真实的边界处理:** 本文用零填充,阶段 4 已经诚实展示了它带来的边框伪影。真实库常用"镜像填充"或"复制边缘像素"来避免这个问题,实现上只是 `padded` 数组边缘几行/几列的取值方式不同,原理不难,这里不展开。
- **真实图片输入:** 把"自己画的合成数组"换成 PIL/opencv/`matplotlib.image` 读进来的真实照片——数据来源变了,但灰度化、卷积、Sobel 这几个核心函数完全不用改,因为它们本来就只依赖"输入是一个二维/三维 numpy 数组"这个约定。
- **更快的卷积:** 真实深度学习框架(PyTorch/TensorFlow)实现卷积层时,除了本文用到的 im2col 思路,还有 FFT 卷积、Winograd 算法等更复杂的加速手段,并且直接跑在 GPU 上并行——本文的 numpy 版本只是把"卷积是什么"这件事的原理跑通,不是工业级实现。
- 真要往这几个方向深挖,每一个都够单独写一整篇——numpy 之外更完整的图像/信号处理库(比如 `scipy.ndimage`、opencv、scikit-image)会是自然的下一站,这里只指方向,不展开。

---

## 这篇教程展示的方法论

和 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 同一套模式:挑几个关联的已学知识点(这里是形状/strides、逐元素运算、线性代数、广播机制)→ 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果(包括真实的计时数字和真实的 ASCII 可视化输出),而不是一次性甩出完整代码。遇到不完美的真实结果——零填充带来的边框伪影、对称条纹正中心梯度反而是0——如实记录并解释原因,不回避、也不为了让结果"看起来更漂亮"而悄悄换一张更配合的测试图案。这是这个"教程体"格式第一次从 dsa-deep-dive 搬到 numpy-deep-dive,后续要不要在其余系列继续推广,是留给后续单独决定的问题,这里不展开。

---

*创建:2026-07-24*
