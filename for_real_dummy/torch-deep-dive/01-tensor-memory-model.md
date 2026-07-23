# 01 · Tensor 内存模型与基础操作深挖(Tensor Memory Model)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批是全系列的地基,也是"面试第二三轮"最容易被问到、却最少人能讲清楚原理的一块——不是"这个函数怎么调",而是"一个 tensor 在内存里到底长什么样"。想清楚这一点,后面 autograd 的 in-place 报错、`.detach()`/`.clone()` 的区别、显存优化的很多技巧,都会变成"顺理成章",而不是死记硬背的规则列表。

**本文定位和 numpy-deep-dive 系列的关系:** 数组的创建/形状变换/索引这些操作,torch 和 numpy 几乎是同一套心智模型(建议先看完 [numpy-deep-dive/02-shape-and-structure.md](../numpy-deep-dive/02-shape-and-structure.md)),这里不重复,只讲 **torch 独有、且是面试重灾区**的部分:tensor 的内存到底怎么组织、什么时候共享内存什么时候不共享、以及这些底层事实如何和 autograd 交互产生一整类经典报错。

本文所有代码例子已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用)下实际跑通验证。凡是"底层机制"部分给出的结论,都配了能直接跑的内省代码(打印 `data_ptr()`/`stride()`/`_version` 等),不是转述文档或凭经验断言。

**本篇统一结构(比 numpy 系列多两块,专门服务"面试深度"这个要求):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究/工程场景
5. 可运行例子(能内省的地方,现场打印内部状态,不用你相信文字描述)
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

---

## 1. `Tensor.storage()`(已弃用)/ `Tensor.untyped_storage()` —— tensor 的底层数据到底存在哪

**是什么:**
```python
x.untyped_storage()   # 现在推荐的写法,返回一块不带类型信息的连续内存
x.storage()            # 老写法,torch 2.x 起会打印 UserWarning(实测确认,见下)
```

**一句话:** 一个 tensor 对象本身只是"元数据"(形状、步长、数据类型、指向哪块内存),真正存数字的是它背后的 `storage`——一块**一维、连续**的内存缓冲区。多个 tensor 完全可以指向同一个 storage,只是"看"的方式(形状/步长)不同。

**底层机制/为什么这样设计:** 把"数据"和"怎么看这份数据"拆成两层,是几乎所有数组库(numpy 也一样)的核心设计——好处是"改变形状"这种操作可以做到零拷贝(只改元数据,不搬内存),这是本篇后面 `view`/`transpose`/`expand` 一系列"零拷贝操作"的根本原因。`storage()` 之所以被弃用,是因为它返回的 `TypedStorage` 带了 dtype 信息,和"storage 本来就应该是无类型的一段字节缓冲区"这个设计理念冲突,新版本统一收敛到 `untyped_storage()`。

**AI 研究场景:** 平时写模型代码几乎不会直接操作 storage,但排查"两个 tensor 是不是意外共享了内存"(典型场景:一个函数返回的"新" tensor 其实是输入的视图,调用方无意中修改了输入)时,`data_ptr()`(storage 起始地址)是最快的诊断手段。

**可运行例子:**
```python
import torch
import warnings

x = torch.arange(6)

# 老写法会警告,新写法不会 —— 现场验证,不是转述
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s_old = x.storage()
    assert len(w) == 1
    assert "TypedStorage is deprecated" in str(w[0].message)

s_new = x.untyped_storage()
assert len(s_new) == x.numel() * x.element_size()   # 字节数 = 元素个数 * 每个元素多少字节

# 两个不同形状的 tensor,共享同一个 storage
y = x.view(2, 3)
assert x.data_ptr() == y.data_ptr()          # 起始地址一样
assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
assert x.shape != y.shape                     # 但"看"的形状不一样
```

**面试怎么问 + 追问链:**
- **Q:** "tensor 的 `.shape` 和它底层数据存储是什么关系?"—— 期望答出"storage 是一维连续内存,shape/stride 是附加的视图信息"。
- **追问 1:** "那 `x.view(2,3)` 之后,原来的 `x` 和新的 `y` 是两份数据吗?" —— 期望能答出"共享同一个 storage,`data_ptr()` 相同",最好能现场说出怎么验证。
- **追问 2(容易被问倒):** "既然存储是无类型的一段字节,PyTorch 怎么知道该把每 4 个字节还是 2 个字节解释成一个数?"—— 答案是 dtype 是 tensor 元数据的一部分,不是 storage 的属性,`untyped_storage()` 这个名字本身就是这层设计的证据。

**常见坑:** 把 `.storage()` 当成"拷贝了一份数据"来用——它返回的对象仍然和原 tensor 共享底层内存,修改 storage 会影响所有指向它的 tensor。新代码一律用 `untyped_storage()`,老代码遇到 `.storage()` 的警告不用慌,行为暂时不变,只是将来会移除。

---

## 2. `stride()` —— 用步长在同一块内存上表达"不同形状"

**是什么:**
```python
x.stride()          # 返回一个 tuple,每个维度"走一步"要跨过底层 storage 里多少个元素
x.stride(dim)        # 只看某一维
```

**一句话:** `stride[i]` 表示"沿着第 `i` 维挪动一个下标,在底层一维内存里要跳过多少个元素"——这是 tensor 从"一维内存"重建出"多维语义"的数学映射。

**底层机制/为什么这样设计:** 一个 `(3,4)` 的 tensor,访问 `x[i,j]` 实际算的是 `storage[i*stride[0] + j*stride[1] + offset]`。**标准(C 顺序、连续)** 布局下 `stride = (4, 1)`(第 0 维每挪一步跳 4 个元素,因为每"行"有 4 个数;第 1 维每挪一步跳 1 个元素)。这个映射公式是理解本篇后面几乎所有"零拷贝"操作的钥匙:`transpose`/`permute`/`expand` 全部只是**改写 stride/shape 这几个数字**,storage 里的字节一个都不挪。

画出来看会直观很多——以下面例子里的 `a = torch.arange(12).reshape(3, 4)` 为例,storage 自始至终是同一段**一维**内存,`shape`/`stride` 只是"怎么切开看待"这段内存的说明书:

```text
storage(物理上真实存在的一维内存,下标 0~11,从头到尾没有任何"行/列"的概念):
[ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11]

a 的逻辑形状 (3,4),stride=(4,1) 表示"每隔 4 个元素换一'行'、每隔 1 个元素换一'列'"
这样切出来看待上面这同一段内存:
  行0 → [ 0][ 1][ 2][ 3]
  行1 → [ 4][ 5][ 6][ 7]
  行2 → [ 8][ 9][10][11]

a[i,j] 的物理地址 = storage[i*stride[0] + j*stride[1]]
例如 a[2,3]:  2*4 + 3*1 = 11  →  storage[11] = 11,和 a[2,3] 的值精确对上(下面代码会现场验证)
```

**AI 研究场景:** 判断"这次 reshape 会不会触发隐藏的内存拷贝"(进而影响性能),第一反应应该是看 stride 是否连续,而不是猜。Transformer 里 `q.transpose(1,2)` 这种操作在大模型推理代码里高频出现,理解它是"零拷贝"还是"隐藏拷贝"直接影响你对代码性能的判断力。

**可运行例子:**
```python
import torch

a = torch.arange(12).reshape(3, 4)
assert a.stride() == (4, 1)          # 标准连续布局:第0维跳4个,第1维跳1个

# 手动验证 stride 公式:a[i,j] == storage[i*stride0 + j*stride1]
flat = a.reshape(-1)
i, j = 2, 3
assert a[i, j].item() == flat[i * a.stride(0) + j * a.stride(1)].item()

b = a.T                               # 转置:形状变了,但完全没挪动内存
assert b.stride() == (1, 4)           # stride 顺序跟着反过来,shape 也反过来
assert b.data_ptr() == a.data_ptr()   # 同一块内存,只是"看"的方式变了
assert not b.is_contiguous()          # 但已经不是"标准顺序"了,见下一节
```

**面试怎么问 + 追问链:**
- **Q:** "`a.T` 之后为什么这么快?一个 `(1000,1000)` 的矩阵转置几乎不耗时,这是为什么?"—— 期望答出"只改 stride/shape 两个数字,不搬数据"。
- **追问:** "那既然内存没动,`a.T[i][j]` 是怎么算出正确的值的?"—— 需要现场推一下 stride 公式,而不是只会说"底层帮你处理了"。
- **深挖追问(区分度很高):** "PyTorch 怎么知道一个 tensor 是不是`contiguous`?给你 shape 和 stride,你能不能手写一个判断函数?"—— 这是下一节的内容,能连起来答说明理解是成体系的,不是背知识点。

**常见坑:** 把 stride 和"字节偏移"搞混——`stride()` 返回的单位是**元素个数**,不是字节数;要换算成字节需要再乘以 `element_size()`。另外新手常以为"转置了就该重新排列内存",看到 `a.T.stride()` 和直觉的"新连续布局"不一样时会困惑——这正是本篇要建立的核心认知:形状变了不代表内存动了。

---

## 3. `is_contiguous()` / `contiguous()`

**是什么:**
```python
x.is_contiguous()     # 判断:当前 shape+stride 是否等价于"标准C顺序连续布局"
x.contiguous()         # 如果已经连续,直接返回自己(no-op);不连续就拷贝出一份新的连续内存
```

**一句话:** "连续"(contiguous)指的是当前 tensor 的 stride,恰好等于"按 shape 从左到右紧密排列"该有的 stride——`contiguous()` 是"如果不满足这个条件,就老老实实拷贝一份满足条件的数据"的兜底操作。

**底层机制/为什么这样设计:** 很多底层算子(尤其是调用 C++/CUDA kernel——即跑在 GPU 上实际执行计算的那一段代码,由 CPU 发起调用、GPU 负责真正执行——的运算)要求输入内存是连续的,因为连续内存能用简单的指针步进遍历,不连续则需要额外处理每一维的 stride,实现更复杂、有些老 kernel 干脆不支持。`contiguous()` 存在的意义就是"在需要连续内存的地方,提供一个安全的转换出口"——本身设计成 no-op 优先(已经连续就不浪费一次拷贝),体现的是 PyTorch 一贯的"能不拷贝就不拷贝"的性能哲学。

**可运行例子:**
```python
import torch

a = torch.randn(3, 4)
assert a.is_contiguous()
c1 = a.contiguous()
assert c1.data_ptr() == a.data_ptr()      # 已连续:no-op,同一块内存

b = a.T
assert not b.is_contiguous()               # 转置后不再连续(上一节验证过 stride 变了)
c2 = b.contiguous()
assert c2.data_ptr() != b.data_ptr()       # 真的拷贝了一份新内存
assert c2.shape == (4, 3)                  # b 是 a 转置后的形状 (4,3)
assert c2.stride() == (3, 1)               # 新内存按 (4,3) 这个形状重新排成标准顺序
assert torch.equal(c2, b)                  # 数值不变,只是内存布局变了
```

**面试怎么问 + 追问链:**
- **Q:** "`.contiguous()` 一定会发生内存拷贝吗?"—— 标准答案是"不一定,已经连续时是 no-op",能现场验证 `data_ptr()` 是否相等的候选人明显更扎实。
- **追问:** "什么场景下你会主动调用 `.contiguous()`?"—— 期望举出"transpose/permute 之后紧接着要 `.view()`""某些老算子/kernel 报错要求连续输入"这类真实场景。

**常见坑:** 滥用 `.contiguous()`"以防万一"——每次调用虽然在已连续时是 no-op,但一旦不确定就无脑加,容易掩盖"其实我应该用 `reshape()` 让它自动处理"这个更简洁的写法(见下一节),而且在真正不连续的路径上会引入不必要的额外拷贝,增加显存峰值。

---

## 4. `view()` vs `reshape()`

**是什么:**
```python
x.view(*shape)       # 要求内存足够"规整"能直接复用,不满足条件直接报错
x.reshape(*shape)     # 能复用就复用(等价于 view);不能就自动 contiguous() 再 view
```

**一句话:** `view` 是"只做零拷贝这一件事,做不到就报错,把决定权交给你";`reshape` 是"帮你把零拷贝和兜底拷贝都安排好,你只要结果"。

**底层机制/为什么这样设计:** 上一节讲了 `is_contiguous`,这里的"内存足够规整"指的是更宽松的条件——并不要求严格连续,而是要求新 shape 能通过调整 stride、不改变底层内存的方式表达出来(严格连续必然满足这个条件,但反过来不一定)。`view` 存在的意义是**给性能敏感代码一个"零拷贝断言"**:用 `view` 就是在说"我确信这里不该有拷贝,如果有说明我的假设错了,请报错让我知道",而不是让拷贝静默发生。

**可运行例子:**
```python
import torch

a = torch.arange(12).reshape(3, 4)
b = a.view(2, 6)
assert b.data_ptr() == a.data_ptr()          # view 永远零拷贝(能成功的话)

t = a.T                                        # 不连续
try:
    t.view(12)
    assert False
except RuntimeError as e:
    assert "view size is not compatible" in str(e)   # 实测报错信息,明确提示"改用 reshape"

r = t.reshape(12)                              # reshape 自动兜底:失败就 contiguous() 再 view
assert r.data_ptr() != t.data_ptr()            # 这次确实拷贝了(和上一节 contiguous() 的例子是同一件事)
assert torch.equal(r, t.contiguous().view(12))  # reshape 在内部做的事,验证等价
```

**面试怎么问 + 追问链:**
- **Q:** "什么时候用 `view`,什么时候用 `reshape`?"—— 很多人只会说"reshape 更安全",更好的答案是"写性能敏感代码、需要显式保证零拷贝时用 view(拷贝了就该报错让你发现);不确定内存布局、只关心结果正确时用 reshape"。
- **追问:** "`view` 报错的判断条件具体是什么,只有转置过的 tensor 会报错吗?"—— 期望知道核心条件是"新 shape 能否用同一份内存 + 新的 stride 表达",而不是死记"transpose 之后不能 view"这一条具体案例。

**常见坑:** 把 `reshape` 当成"永远安全、无脑用"——它确实不会报错,但会**悄悄引入一次你没预料到的拷贝**,在显存紧张或者性能敏感的循环里,这种"隐藏成本"比一次明确的报错更难排查。

---

## 5. `.T` / `transpose()` / `permute()` —— 只改 stride,不挪数据

**是什么:**
```python
x.T                          # 所有维度整体逆序(和 numpy 的 .T 语义一致,见 numpy-deep-dive/02 第2节)
x.transpose(dim0, dim1)       # 只交换指定的两个维度
x.permute(*dims)              # 任意重排所有维度顺序
```

**一句话:** 三者都是"改 shape/stride 元数据,不碰底层内存"的零拷贝操作——区别只在"允许你精确控制交换/重排到什么程度",数学本质和上面两节讲的机制完全一样。

**底层机制/为什么这样设计:** 延续第 2 节的核心认知——既然维度顺序只是"怎么解读 stride"的问题,重排维度自然可以做到零成本。这也解释了为什么 attention 实现里 `k.transpose(-1,-2)` 这种操作几乎不产生性能开销(是本篇第 2 节内容在 Transformer 场景的直接应用)。

**可运行例子:**
```python
import torch

x = torch.randn(2, 3, 4)   # 假装是 (batch, seq, dim)
y = x.permute(1, 0, 2)      # 换成 (seq, batch, dim),某些老代码/库要求这个顺序
assert y.shape == (3, 2, 4)
assert y.data_ptr() == x.data_ptr()     # 零拷贝
assert not y.is_contiguous()             # 但和转置一样,不再连续了

# transpose 是 permute 的特例:只交换两维
z = x.transpose(0, 1)
assert torch.equal(z, x.permute(1, 0, 2))
```

**面试怎么问 + 追问链:**
- **Q:** "attention 里 `Q @ K.transpose(-1,-2)` 这一步,`transpose` 会不会成为性能瓶颈?"—— 期望答"不会,transpose 本身零拷贝,真正的开销在后面的矩阵乘法";如果候选人担心"transpose 很慢",说明对内存模型理解不到位。

**常见坑:** 见第 2 节结尾——`.T` 对 2 维以上的 tensor 是"整体维度逆序",不是"只转置最后两维",这个坑在 numpy 系列已经讲过一次,torch 里同样成立,想要更精确的控制要用 `transpose`/`permute`。

---

## 6. in-place 操作(`_` 后缀)与内存复用

**是什么:** 方法名以下划线结尾的操作(`add_`、`mul_`、`relu_`……)直接在原 tensor 的 storage 上修改数据,不分配新内存,没有返回值意义上的"新 tensor"(虽然语法上仍返回 `self`)。

**一句话:** `y = x.add(1)` 会新分配一块内存存结果;`x.add_(1)` 直接把 `x` 底层那块内存的数字改掉,`data_ptr()` 前后完全一样。

**底层机制/为什么这样设计:** 省掉一次内存分配和拷贝,在显存紧张、或者需要频繁原地更新的场景(最典型的就是优化器更新参数)是有意义的性能优化。但这个"直接改内存"的特性,会在下一节和 autograd 产生本篇最经典的一类冲突——这里先建立"in-place = 复用同一块内存"这一层纯内存事实。

**可运行例子:**
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
ptr_before = x.data_ptr()

y = x.add(10)             # 非 in-place:新内存
assert y.data_ptr() != ptr_before

x.add_(10)                 # in-place:原地修改
assert x.data_ptr() == ptr_before
assert x.tolist() == [11.0, 12.0, 13.0]

# 优化器更新参数的本质就是一堆 in-place 操作(简化示意)
param = torch.tensor([1.0, 2.0])
grad = torch.tensor([0.1, 0.2])
lr = 0.1
param.sub_(lr * grad)      # 等价于 SGD 的一步更新,原地完成,不新分配 param 的内存
assert torch.allclose(param, torch.tensor([0.99, 1.98]))   # float32 精度,不能用 ==(呼应 numpy 系列反复强调的浮点数比较原则)
```

**面试怎么问 + 追问链:**
- **Q:** "为什么 PyTorch 的优化器(比如 `optimizer.step()`)内部大量使用 in-place 操作?"—— 期望答"避免每一步更新都重新分配一份和参数一样大的内存,大模型场景下这个开销不可忽视"。这个问题会在 [06-optimizer-internals.md](06-optimizer-internals.md) 详细展开。

**常见坑:** 见下一节——不是所有 in-place 操作都安全,和 autograd 一起用时有明确的规则,不能凭"能跑就是对的"来判断。

---

## 7. in-place 操作与 autograd 的版本计数器机制(本篇重点,常被讲错)

**是什么:** 每个 tensor 内部维护一个 `._version` 计数器,初始为 0,**每次 in-place 修改该 tensor 都会 +1**。autograd 在反向传播时,如果某个 op 的 backward 需要用到"当年保存的某个 tensor 的值",会检查这个 tensor 当前的 `_version` 是否还等于保存时的版本——不等,直接报错。

**一句话:** "in-place 操作会破坏 autograd"这句话**不完全准确**——精确的说法是:in-place 操作会让被修改的 tensor 的 `_version` 增加,而只有当反向传播时**恰好需要用到这个 tensor 被修改前的值**,才会报错;如果没人需要那个旧值,in-place 操作完全不受影响,能正常跑完 `.backward()`。

**底层机制/为什么这样设计(附带一个很多教程会讲错的细节,已实测验证):**

不是所有反向传播都需要"保存前向的中间值"。比如 `add` 的反向传播 `d(a+b)/da = 1`,是个常数,根本不需要知道 `a` 或 `b` 原来是多少;但 `exp` 的反向传播 `d(exp(x))/dx = exp(x)`,**必须**用到自己的输出值。所以:

```python
import torch

# 场景一:y = w*2 之后原地修改 y —— 不报错!因为 mul 对 w 的反向传播不需要保存 y
w = torch.tensor([1.0, 2.0], requires_grad=True)
y = w * 2
y.add_(1.0)              # 原地改 y,y 的 _version 变成 1
y.sum().backward()        # 正常跑完,不报错
assert w.grad.tolist() == [2.0, 2.0]

# 场景二:y = w.exp() 之后原地修改 y —— 报错!因为 exp 的反向传播需要保存的输出值(即 y)已经被改了
w2 = torch.tensor([1.0, 2.0], requires_grad=True)
y2 = w2.exp()
assert y2._version == 0
y2.add_(1.0)              # 原地改,_version 变成 1
assert y2._version == 1
try:
    y2.sum().backward()
    assert False
except RuntimeError as e:
    # 实测报错信息,精确点出"是 ExpBackward0 的第0个输出,版本对不上"
    assert "modified by an inplace operation" in str(e)
    assert "ExpBackward0" in str(e)
    assert "is at version 1; expected version 0" in str(e)
```

**AI 研究场景:** 手写自定义训练循环时,"能不能对某个中间激活值做 in-place 操作(比如 `relu_()` 而不是 `relu()` 省显存)"是真实的工程决策——答案取决于这个激活值有没有被后续某个反向传播依赖它的**原始**值,而不是一刀切"能省则省"或"能不用就不用"。

**面试怎么问 + 追问链(这是本篇"面试深度"要求的典型体现):**
- **Q:** "为什么 `y = w * 2; y.add_(1); y.sum().backward()` 不报错,但换成 `y = w.exp()` 就报错?"—— 这是一道能立刻分出候选人是"背过结论"还是"理解机制"的问题。只会说"in-place 不能用在 autograd 里"的候选人会被这个反例问住。
- **追问 1:** "PyTorch 是怎么知道'这个值被改过'的?"—— 期望答出"每个 tensor 有版本计数器,in-place 操作会递增它,反向传播会核对版本号"。
- **追问 2(深挖):** "报错信息里的 `ExpBackward0` 是什么?"—— 这是下一批([02-autograd-internals.md](02-autograd-internals.md))的 `grad_fn` 概念,能连起来答说明知识体系是打通的。
- **追问 3(开放题):** "如果你要写一个显存优化技巧,大量使用 in-place 操作省显存,你会怎么系统性地判断哪些地方安全?"—— 没有标准答案,考察的是"版本计数器机制"这个原理能不能被灵活应用到新场景。

**常见坑:** 网上很多资料把这条规则简化成"训练代码里不要用 in-place 操作",这个说法过于保守且不准确——`optimizer.step()` 内部到处都是 in-place 操作(第 6 节已经看到),因为参数更新这一步**不在**任何反向传播需要追溯的路径里(它发生在 `backward()` 之后)。真正需要小心的,是"某个 tensor 的值被后续反向传播依赖,又在依赖发生前被原地改掉"这一种特定情形,而不是任何 in-place 操作。

---

## 8. `.detach()` vs `.clone()` vs `.data` —— 三者的精确区别(面试经典陷阱题)

**是什么:**
```python
x.detach()    # 新 tensor,共享 storage,requires_grad=False,没有 grad_fn(彻底脱离计算图)
x.clone()     # 新 tensor,独立 storage(真拷贝),但保留在计算图里(有 grad_fn,梯度能传回 x)
x.data        # 新 tensor,共享 storage,requires_grad=False —— 但对它的修改完全不被 autograd 感知
```

**一句话:** 三者的核心差异是两个独立的维度——"**存储是否独立**"(clone 独立,detach/.data 共享)和"**是否还在计算图里**"(clone 在图里、能反传;detach/.data 都不在图里),`.data` 比 `.detach()` 多一层"危险":用它修改数据不会被 autograd 的版本计数器追踪。

**底层机制/为什么这样设计:**

| | 共享 storage? | 在计算图里(有 grad_fn)? | 修改会被版本计数器感知? |
|---|---|---|---|
| `.detach()` | ✅ 共享 | ❌ 不在(叶子,`requires_grad=False`) | ✅ 会(现代、推荐的做法) |
| `.clone()` | ❌ 独立(真拷贝) | ✅ 在(`CloneBackward0`,梯度能传回原 tensor) | 不适用(改的是自己的独立内存) |
| `.data` | ✅ 共享 | ❌ 不在 | ❌ **不会**(历史遗留,绕过安全检查) |

`.data` 是 PyTorch 早期(Variable 和 Tensor 还没合并的年代)遗留下来的底层访问方式,`.detach()` 是后来专门设计出来、行为上"更安全"的替代品——两者共享存储的效果一样,但 `.detach()` 返回的新 tensor **仍然会正确参与版本计数器机制**(修改它,版本号照样 +1,如果原 tensor 的某处反向传播需要这个值,依然会正确报错提醒你);`.data` 则完全绕开了这层保护,修改它连报错的机会都没有,梯度计算会**静默地**用一个已经过时的计算关系继续算下去。下面这个例子是本篇除版本计数器外最值得记住的一个陷阱,已实测验证:

**可运行例子:**
```python
import torch

p = torch.tensor([1.0, 2.0], requires_grad=True)
q = p * 3

d = q.detach()
cl = q.clone()

assert d.requires_grad is False and d.data_ptr() == q.data_ptr()
assert cl.requires_grad is True and cl.data_ptr() != q.data_ptr()
assert cl.grad_fn is not None and "CloneBackward" in str(type(cl.grad_fn))

# --- 陷阱实锤:通过 .data 修改,梯度会"静默地"算出一个过时的结果 ---
p2 = torch.tensor([5.0], requires_grad=True)
q2 = p2 * 2                    # q2 = 10.0,计算图记录的关系是 "q2 = p2 * 2"
q2.data += 100                  # 通过 .data 原地改成 110.0 —— 不报错,不留痕迹
assert q2.item() == 110.0

loss = (q2 * 3).sum()           # loss 用的是"篡改后"的 110.0 参与前向计算
loss.backward()                  # 不报错!因为 .data 修改没有让 _version 变化
# 但 backward 算出来的梯度,反映的是计算图里记录的 "q2 = p2*2" 这个关系,
# 完全不知道 q2 后来被偷偷加了 100 —— 梯度对应的是一个已经不存在的计算过程
assert p2.grad.item() == 6.0   # d(loss)/d(p2) = d(3*q2)/d(q2) * d(q2)/d(p2) = 3*2 = 6
                                  # 这个 6 只在"q2 真的等于 p2*2"时才正确,但 q2 现在其实是 110
```

**面试怎么问 + 追问链:**
- **Q:** "`.detach()`、`.clone()`、`.data` 有什么区别,分别在什么场景用?"—— 基础版问法,期望答出上面表格的两个维度。
- **追问 1(区分度很高):** "既然 `.detach()` 和 `.data` 都共享内存、都脱离计算图,它们有什么实质区别?"—— 只有理解版本计数器机制(第7节)的候选人才能准确答出"`.detach()` 的修改仍会被追踪,`.data` 不会"。
- **追问 2:** "为什么现在的代码规范建议用 `.detach()` 而不是 `.data`?"—— 期望连回"静默产生错误梯度"这个具体危害,而不是只说"官方推荐"。
- **追问 3(工程场景):** "什么时候你会用 `.clone()` 而不是 `.detach()`?"—— 期望举出"需要一份独立副本、但仍希望梯度能传回原 tensor"的场景,比如某些正则化/数据增强需要保留一份"干净"输入同时还要计算梯度。

**常见坑:** 以为 `.detach()` 会拷贝数据(它不会,仍然共享 storage,修改 `.detach()` 出来的 tensor 会影响原 tensor 的值,只是不影响梯度追踪);以为 `.data` 已经被淘汰不能用(它还能用,只是不安全,新代码应默认选 `.detach()`)。

---

## 9. `.expand()` vs `.repeat()` —— 内存语义完全不同(比 numpy 的 tile/repeat 更危险)

**是什么:**
```python
x.expand(*sizes)    # 零拷贝:把大小为1的维度"虚拟地"扩展,底层仍是同一份数据(stride=0)
x.repeat(*sizes)     # 真拷贝:把整个 tensor 真实复制多份,拼成一个更大的、独立的新 tensor
```

**一句话:** `expand` 是"假装变大了"(多个逻辑位置指向同一个物理内存),`repeat` 是"真的变大了"(每个位置都是独立内存)——这个区别比 numpy 的 `tile`/`repeat`(那两个都会真实复制)更危险,因为 `expand` 允许你**写入**,而写入会产生反直觉的连锁效应。

**底层机制/为什么这样设计:** `expand` 的实现方式,是把待扩展的维度 stride 设成 **0**——回忆第 2 节的 stride 公式 `storage[i*stride0 + j*stride1]`,如果某一维的 stride 是 0,那么这一维不管下标 `i` 取多少,算出来的偏移量都一样,也就是"事实上都在读同一个内存位置"。这是纯粹的零拷贝技巧(表示"batch 维度共享同一份权重/mask"这类场景效率极高),但如果你对结果做**原地写入**,会出现下面这个很反直觉、已实测验证的现象:

**可运行例子:**
```python
import torch

v = torch.tensor([1, 2, 3])
e = v.expand(3, 3)              # "假装"变成 3x3,stride=0 那一维
assert e.stride() == (0, 1)      # 第0维 stride=0,证实"虚拟扩展"
assert e.data_ptr() == v.data_ptr()

r = v.repeat(3, 1)                # 真实复制成 3x3
assert r.data_ptr() != v.data_ptr()
assert torch.equal(e, r)           # 读出来的值一样,但内存语义完全不同

# --- 陷阱实锤:对 expand 结果做原地写入,会"跨行"产生诡异效果 ---
e[0, 0] = 999
# 直觉上应该只改"第0行第0列",但由于第0维stride=0,三"行"其实是同一份内存,
# 逻辑上不存在独立的"第0行"——写入位置(0,0)会让所有行的第0列都变成999
assert e.tolist() == [[999, 2, 3], [999, 2, 3], [999, 2, 3]]
assert v.tolist() == [999, 2, 3]     # 原始 v 也被改了!因为共享同一块内存

# repeat 则完全独立,写入不会有这种连锁反应
r[0, 0] = 999
assert r.tolist() == [[999, 2, 3], [1, 2, 3], [1, 2, 3]]   # 只改了真正的第0行
```

**AI 研究场景:** 给一个 batch 里的每个样本"共享同一份"attention mask 或位置编码模板时,`expand` 比 `repeat` 省内存、更快(不用真的复制 batch 份),这是训练大模型时的常见优化;但前提是**只读不写**——这也是为什么很多框架代码里 `expand` 出来的结果后面紧跟着的操作都是纯计算(比如加到别的 tensor 上),而不是原地修改。

**面试怎么问 + 追问链:**
- **Q:** "`expand` 和 `repeat` 有什么区别?"—— 基础答案是"expand 零拷贝,repeat 真拷贝"。
- **追问(杀伤力很强):** "如果我对 `expand` 出来的结果做 `+=` 或者索引赋值,会发生什么?"—— 期望答出"由于 stride=0,写入会跨越所有'虚拟重复'的位置,而且会连带修改原始 tensor",最好能现场推出这是 stride 机制的直接后果,而不是"就是不能这样用"的死记硬背。
- **追问:** "所以 `expand` 之后能安全做哪些操作?"—— 期望答"只读操作、参与不改变自身的计算(比如加法产生新 tensor)都安全,任何原地修改都危险"。

**常见坑:** 以为 `expand` 和 `repeat` 只是"写法不同、结果一样",在性能优化时随手把 `repeat` 换成 `expand` 却忘了后面代码对结果做了原地修改——这类 bug 的症状通常是"某个看起来无关的 tensor 的值莫名其妙变了",排查起来非常隐蔽,是本篇里最值得单独拎出来记住的一条。

---

## 10. `.to(device)` 的 no-op 判定

**是什么:** `x.to(device=..., dtype=..., ...)` 是"搬到某个设备/转成某个精度"的统一入口——但如果目标设备和 dtype**和当前完全一致**,它不会做任何拷贝,直接返回原 tensor 本身。

**一句话:** `.to()` 内部先检查"要不要搬",答案是"不需要"时就是纯粹的 no-op(和 numpy-deep-dive 系列反复出现的"能不拷贝就不拷贝"是同一个设计哲学)。

**底层机制/为什么这样设计:** 训练代码里经常出现 `x = x.to(device)` 这种防御性写法(不确定 `x` 现在在哪个设备),如果每次都无脑拷贝,当 `x` 已经在目标设备上时就是纯粹的浪费——尤其是在训练循环内部高频调用的代码路径上,这种"重复搬运"的隐藏开销会累积得很可观。

**可运行例子:**
```python
import torch

x = torch.randn(3)
y = x.to('cpu')                          # 已经在 cpu,目标还是 cpu
assert x.data_ptr() == y.data_ptr()        # no-op,同一块内存

if torch.cuda.is_available():
    g = torch.randn(3, device='cuda')
    g2 = g.to('cuda')                      # 已经在 cuda,目标还是 cuda
    assert g.data_ptr() == g2.data_ptr()    # 同样是 no-op

    c = g.to('cpu')                         # 真正跨设备,必须拷贝
    assert c.data_ptr() != g.data_ptr()
```

**面试怎么问 + 追问链:**
- **Q:** "训练循环里写 `batch = batch.to(device)`,如果 `batch` 已经在目标 device 上,这行代码有没有额外开销?"—— 期望答"没有,是 no-op",这是一个考察"你是不是真的理解 .to() 而不是无脑照抄样板代码"的好问题。

**常见坑:** 误以为 `.to(device)` 每次调用都会产生同步等待或者内存分配开销,因此过度设计"手动缓存设备判断"这类不必要的优化——大多数情况下相信 `.to()` 自己的 no-op 判定就够了,真正需要关心开销的是**确实**跨设备的那一次拷贝(下一批 [08-memory-and-performance.md](08-memory-and-performance.md) 会讲 `non_blocking=True` 怎么优化这一次真实拷贝)。

---

## 11. `pin_memory()` / `non_blocking=True`

**是什么:**
```python
x.pin_memory()                       # 把 CPU tensor 拷贝到"页锁定"(page-locked)内存
x.to(device, non_blocking=True)       # 配合 pinned 内存使用,发起异步拷贝,不等拷贝完成就返回
```

**一句话:** 普通 CPU 内存(pageable)可能被操作系统换出到磁盘,GPU 没法直接从这种内存做 DMA(直接内存访问)搬数据,必须先经过一次"锁在物理内存里"的中转拷贝;`pin_memory()` 提前把这次中转做好,`non_blocking=True` 则让"CPU 到 GPU"这一步搬运可以和其他 CPU 计算重叠,不用傻等。

**底层机制/为什么这样设计:** 这是操作系统内存管理和 GPU DMA 机制共同决定的——`pin_memory()` 本身**不是零拷贝**(下面会验证它确实分配了新内存),它的价值在于"提前把这次不可避免的拷贝做掉",换来后续"pinned 内存 → GPU"这一步可以用 `non_blocking=True` 发起异步传输。`DataLoader(pin_memory=True)` 背后就是在数据加载阶段提前做这件事,让真正的训练循环里"数据搬到 GPU"这一步更快、更能和计算重叠。

**可运行例子:**
```python
import torch
import time

x = torch.randn(1000, 1000)
assert x.is_pinned() is False

xp = x.pin_memory()
assert xp.is_pinned() is True
assert xp.data_ptr() != x.data_ptr()    # pin_memory 本身就是一次拷贝,不是 no-op

if torch.cuda.is_available():
    big = torch.randn(50_000_000)        # 约 200MB
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g1 = big.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    t_pageable = time.perf_counter() - t0

    big_pinned = big.pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g2 = big_pinned.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    t_pinned = time.perf_counter() - t0

    print(f"pageable: {t_pageable*1000:.1f}ms, pinned+non_blocking: {t_pinned*1000:.1f}ms")
    # 本机实测:pinned 版本明显更快(具体倍数受机器/负载影响,方向稳定)
    assert t_pinned < t_pageable
```

**面试怎么问 + 追问链:**
- **Q:** "`DataLoader` 里的 `pin_memory=True` 是做什么用的?"—— 期望答"让 CPU→GPU 的数据搬运可以异步进行、和计算重叠",而不是只会背"能加速"三个字。
- **追问:** "`pin_memory()` 这一步本身是不是免费的?"—— 期望答"不是,它自己就是一次真实的内存拷贝(pageable→pinned),只是通常放在数据加载的 worker 进程里提前做,不占用主训练循环的时间"。
- **追问(容易漏答):** "只调用 `non_blocking=True` 但源 tensor 不是 pinned 内存,会发生什么?"—— 期望答"这个参数会被静默忽略,退化成同步拷贝,不会报错,但也不会有异步的收益"——这是一个实践中很容易踩、却不会报错提醒你的坑。

**常见坑:** 见上面追问 3——`non_blocking=True` 不是"万能加速开关",只有配合 pinned 内存(且目标是 GPU)才真正生效,源数据是普通 pageable 内存时这个参数不会报错也不会警告,只是安静地不起作用,容易让人误以为"用了就一定更快"。

---

## 12. dtype 转换:`.float()` / `.half()` / `.to(dtype=...)`

**是什么:** 把 tensor 转换成另一种数值精度——`.float()` 是 `.to(torch.float32)` 的简写,`.half()` 是 `.to(torch.float16)` 的简写,以此类推。

**一句话:** 和"跨设备"拷贝的判断逻辑完全一致(第 10 节)——**目标 dtype 和当前一致时是 no-op**,不一致时必须真实拷贝+转换(因为不同精度的比特位表示方式本来就不同,不可能通过改 stride/shape 这种"元数据戏法"实现)。

**底层机制/为什么这样设计:** 和"改变形状"不同,"改变精度"是要动**每一个数字的比特表示**的,float32 转 float16 涉及真实的数值截断/舍入,没有办法像 view/transpose 那样只调整解读方式,所以这里没有"零拷贝"这个选项——这一点是本篇前面几节"能不拷贝就不拷贝"哲学的边界:凡是数据本身(不是解读方式)要变,就必须付出实际拷贝+计算的代价。

**可运行例子:**
```python
import torch

x = torch.tensor([1.5, 2.5], dtype=torch.float32)

x_same = x.float()                      # 目标就是 float32,已经是了
assert x_same.data_ptr() == x.data_ptr()  # no-op

x_half = x.half()                        # 真的要转换比特表示
assert x_half.dtype == torch.float16
assert x_half.data_ptr() != x.data_ptr()   # 必须拷贝,没有零拷贝的可能
assert x_half.element_size() == 2          # float16 每个数占2字节(float32是4字节)

# 混合精度训练场景:模型权重通常保持 float32(数值稳定性),前向计算时转成 float16
weight_fp32 = torch.randn(4, 4)
weight_fp16 = weight_fp32.half()
assert weight_fp32.dtype != weight_fp16.dtype
assert not torch.equal(weight_fp32, weight_fp16.float())   # 转换有精度损失,不完全相等
assert torch.allclose(weight_fp32, weight_fp16.float(), atol=1e-3)  # 但足够接近
```

**面试怎么问 + 追问链:**
- **Q:** "`.float()`/`.half()` 这类 dtype 转换,和 `.to(device)` 一样有 no-op 优化吗?"—— 期望答"有,目标 dtype 相同时是 no-op,不同则必须拷贝",能看出候选人是否把"设备转换"和"精度转换"背后统一的"变不变就不拷贝"逻辑串起来了,而不是当成两条孤立的规则记。
- **追问:** "为什么混合精度训练(AMP)要保留一份 float32 的权重,而不是全程用 float16?"—— 这是 [07-training-loop-internals.md](07-training-loop-internals.md) 会展开的内容,这里先埋一个伏笔:float16 表示范围和精度有限,直接用它累积梯度更新容易出现下溢/精度丢失。

**常见坑:** 频繁在训练循环内部反复做不必要的 dtype 转换(比如每个 step 都 `.float()` 一个本来就是 float32 的 tensor)——虽然这种情况下是 no-op 不产生额外开销,但如果转换目标每次都不一样(比如某个中间结果 dtype 不稳定,导致 `.half()` 每次都触发真实拷贝),会成为容易被忽略的性能隐患,建议用 `x.dtype` 显式检查而不是盲目转换。

---

## 小结:这一批 12 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|---|---|
| 1 | `storage()`/`untyped_storage()` | tensor = 元数据 + 指向一块一维连续内存;多个 tensor 可共享同一 storage |
| 2 | `stride()` | `storage[i*stride0+j*stride1+...]` 是多维语义到一维内存的映射公式,后面一切零拷贝操作的根 |
| 3 | `is_contiguous()`/`contiguous()` | 判断当前 stride 是否等于标准顺序;不连续时 `contiguous()` 兜底拷贝 |
| 4 | `view()` vs `reshape()` | view=零拷贝断言(失败就报错),reshape=零拷贝优先+自动兜底拷贝 |
| 5 | `.T`/`transpose()`/`permute()` | 只改 shape/stride,不挪数据,零成本 |
| 6 | in-place(`_`后缀) | 直接改原内存,不新分配,是显存优化和优化器更新参数的基础 |
| 7 | in-place + autograd 版本计数器 | 报错与否取决于"反向传播是否需要保存前的值",不是所有 in-place 都危险 |
| 8 | `.detach()` vs `.clone()` vs `.data` | 共享内存吗?在计算图里吗?两个独立维度;`.data` 修改不受版本计数器保护,最危险 |
| 9 | `.expand()` vs `.repeat()` | expand 零拷贝(stride=0)但写入会跨"虚拟行"污染;repeat 真拷贝,写入安全 |
| 10 | `.to(device)` no-op | 目标设备/dtype 相同就不拷贝 |
| 11 | `pin_memory()`/`non_blocking` | pin_memory 本身是一次真实拷贝,换来后续异步传输的资格 |
| 12 | dtype 转换 | 和设备转换同一逻辑:变了必须拷贝,不变就是 no-op |

下一批:[02-autograd-internals.md](02-autograd-internals.md) —— Autograd 核心机制(全系列的重中之重)。

---

*更新:2026-07-07*
