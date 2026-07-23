# 11 · 调试与常见报错精解(Debugging and Common Errors)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这是 torch-deep-dive 系列的最后一批,主题从"这个机制是怎么工作的"切换成"这个机制坏掉的时候,报错长什么样、怎么读、怎么定位"——前 10 批建立的底层认知,在这里派上真正的用场:很多报错信息本身就是"这个机制被违反了"的直接体现,读得懂 02 篇的版本计数器机制,就能一眼看穿"modified by an inplace operation"这句话到底在说什么,而不是每次都靠 Google 报错文本、复制粘贴 Stack Overflow 上的"解决方案"却不知道为什么有效。

**本篇和前几批的关系:** 本篇大量交叉引用 01(内存模型)、02(autograd)、03(hook 机制)、06(参数过滤机制)、08(显存机制)——不是巧合,而是"常见报错"本来就是"底层机制被违反"的症状,理解症状离不开先理解病理。

**验证方法论(与前几批一致,这里格外重要):** 本文每一个报错信息都是**现场真实触发**出来的精确文本(不是凭记忆转述、也不是从网上搜来的旧版本文本),因为报错文本本身经常随 PyTorch 版本演进变化,本机版本是 2.11.0+cu128,如果你用的版本不同,文本可能有出入,但报错背后的机制原理是稳定的。

**本篇统一结构(同前几批):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场触发真实报错,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. CUDA out of memory —— 报错信息本身就是排查线索,不用只会调小 batch size

**是什么:** `RuntimeError: CUDA out of memory. ...` —— 显存不够用,分配失败时的报错。

**一句话:** 这条报错远不止"显存不够"四个字,它把 [08-memory-and-performance.md](08-memory-and-performance.md) 第 1 节讲过的 `allocated`/`reserved` 拆分直接打印在报错文本里——认真读这几个数字,往往能在改代码之前就分清楚"是真的显存不够"还是"碎片化导致明明够用但分配失败"这两种完全不同、解法也不同的情况。

**底层机制/为什么这样设计:**

现场触发一次真实 OOM,看报错文本的完整结构:

```python
import torch
try:
    torch.cuda.empty_cache()
    x = torch.randn(1000, 1000, 1000, 200, device='cuda')   # 故意申请一个天文数字大小
except RuntimeError as e:
    print(e)
```

实测报错原文:
```
CUDA out of memory. Tried to allocate 745.06 GiB. GPU 0 has a total capacity of 16.00 GiB
of which 14.88 GiB is free. Of the allocated memory 0 bytes is allocated by PyTorch,
and 0 bytes is reserved by PyTorch but unallocated. If reserved but unallocated memory
is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

这句话里至少 4 个数字值得逐个读:"Tried to allocate"(这次请求到底要多大)、"total capacity"(卡的物理总显存)、"free"(driver 层面当前空闲多少)、"allocated by PyTorch"/"reserved by PyTorch but unallocated"(呼应 08 篇第 1 节的 `memory_allocated()`/`memory_reserved()`)。**排查的第一步不是立刻调小 batch size,而是看这几个数字的关系:**

- 如果 "Tried to allocate" 远大于 "free"(就像上面的例子,745GB vs 14.88GB 空闲):是真的显存不够,要么减小模型/batch/输入尺寸,要么用 [07-training-loop-internals.md](07-training-loop-internals.md) 讲过的梯度累加/checkpoint/混合精度这类省显存手段。
- 如果 "reserved by PyTorch but unallocated" 这个数字**很大**、"Tried to allocate" 其实没有超过 free 太多:很可能是显存碎片化(08 篇第 1 节结尾提过的伏笔)——缓存池里攥着足够多的显存,但被拆成了很多和这次请求大小不匹配的小块,拼不出连续的一大块。报错文本自己就给出了应对建议:`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,这是本节验证时读到的、很多人不知道有这个选项的一条真实提示。

**可运行例子(区分"真的不够"和"碎片化导致的假性不够",用 `memory_summary()` 现场诊断):**

```python
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 制造碎片化:交替分配大小差异很大的tensor,再释放中间那些,让剩下的block不连续
blocks = []
for size_mb in [50, 5, 50, 5, 50, 5, 50]:
    n = size_mb * 1024 * 1024 // 4
    blocks.append(torch.randn(n, device='cuda'))
# 释放偶数下标的大block,只留下不连续的小碎片和空隙
for i in range(0, len(blocks), 2):
    blocks[i] = None
torch.cuda.empty_cache()   # 注意:empty_cache只会把"完全空闲的segment"还给driver,
                            # 不能把"segment内部混杂着活跃小block"的部分腾出来给别的分配复用

allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
print(f"allocated={allocated/1024**2:.1f}MB reserved={reserved/1024**2:.1f}MB")
# 用 torch.cuda.memory_summary() 能看到更细粒度的分池情况(large pool/small pool分别的
# 已分配/空闲/碎片统计),这是比只看两个汇总数字更细致的排查工具
print(torch.cuda.memory_summary(abbreviated=True)[:500])
```

上面这 7 个 block 按分配顺序,释放偶数下标(4 个 50MB 的)之后,物理布局(不是数值,是"这一段内存现在处于什么状态")长这样:

| 位置(分配顺序) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 大小 | 50MB | 5MB | 50MB | 5MB | 50MB | 5MB | 50MB |
| 释放后状态 | 已释放 ░ | 仍在用 ■ | 已释放 ░ | 仍在用 ■ | 已释放 ░ | 仍在用 ■ | 已释放 ░ |

已释放的部分加起来是 `4*50=200MB`,数字上看起来很宽裕;但这 200MB 被 3 个依然存活的 5MB block 拦腰截成 **4 段互不相邻**的 50MB 碎片。如果接下来想申请一块**连续**的 100MB,即使全场"已释放"的总量有 200MB,这次请求依然会失败(或者被迫向 driver 重新申请一大块全新的,不能拼凑这 4 段碎片)——这正是"`reserved` 够大、`allocated` 也不高,分配却还是失败"这类 OOM 报错背后的真实物理图像。

**AI 研究场景:** 云端多租户 GPU 集群/Colab 这类共享环境里,"同一个模型/batch size 昨天能跑,今天突然 OOM"是常见现象,原因往往不是显存"变小了",而是同一进程反复分配/释放不同大小 tensor(比如做超参搜索时循环 new 不同 batch size 的模型)累积出的碎片化——这也是为什么长时间跑多组实验的脚本,有时会在训练脚本外层每组实验之间加一次进程级别的重启,而不是指望进程内的 `empty_cache()` 彻底解决碎片问题。

**面试怎么问 + 追问链:**
- **Q:** "遇到 CUDA OOM,你的排查思路是什么?"—— 期望不是马上说"调小 batch size",而是先读报错信息本身的几个数字。
- **追问 1(核心):** "报错信息里 'Tried to allocate' 和 'reserved but unallocated' 这两个数字分别说明什么?"—— 期望能对应上 08 篇的 `memory_allocated`/`memory_reserved` 概念,并说出"reserved but unallocated 很大往往提示碎片化,不是真的显存不够"。
- **追问 2(工程向):** "如果确认是碎片化导致的,有什么办法缓解?"—— 期望提到 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 这个环境变量(报错信息本身就给出的建议),或者更粗暴但有效的"避免在同一进程里反复创建/销毁差异很大尺寸的显存分配模式"。

**常见坑:** 看到 OOM 报错就条件反射式地调小 batch size 重跑,不看报错文本给出的具体数字——如果根因是碎片化,调小 batch size 可能"恰好绕过去了",但没有真正解决问题,更大规模的训练任务遇到同样的碎片化模式时还会复发。

---

## 2. "modified by an inplace operation" —— 报错好读,但很难定位到底是哪一行

**是什么:** `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation...`

**一句话:** [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 7 节和 [02-autograd-internals.md](02-autograd-internals.md) 已经讲透了这条报错背后的版本计数器机制(某个 backward 需要用到的 tensor,在 forward 结束之后、backward 真正执行之前被原地修改过,版本号对不上),本节不重复推导——这里要解决一个机制篇没有解决的实际问题:**这条报错默认只会告诉你是哪个算子(比如 `TanhBackward0`)出了问题,不会告诉你这是在你代码的哪一行导致的**,在几百行的真实模型代码里,光凭这条报错文本经常无从下手。

**底层机制/为什么这样设计:**

先现场触发一次真实报错,看默认信息有多"贫瘠":

```python
import torch

def buggy_forward(x):
    y = torch.tanh(x)      # tanh的backward需要保存y本身(d(tanh)/dx = 1-y^2)
    z = y + 1
    y.add_(100)              # bug:原地修改了y,而y的原始值是TanhBackward需要的
    loss = (z * y).sum()
    return loss

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = buggy_forward(x)
try:
    loss.backward()
except RuntimeError as e:
    print(e)
```

实测报错原文:
```
one of the variables needed for gradient computation has been modified by an inplace
operation: [torch.FloatTensor [3]], which is output 0 of TanhBackward0, is at version 1;
expected version 0 instead. Hint: enable anomaly detection to find the operation that
failed to compute its gradient, with torch.autograd.set_detect_anomaly(True, check_nan=False).
```

只看这段文本,你知道"某个 `tanh` 的输出被原地改过了",但如果模型里有好几处 `tanh`,不知道具体是哪一行代码里的那个 `y.add_(100)`。**报错信息自己给出了解法**——这正是下面第 5 节要讲透的 `detect_anomaly()`;这里先剧透一下效果:开着它重新跑一遍同样的代码,会在报错之前额外打印出"Traceback of forward call that caused the error",精确指到 `y = torch.tanh(x)` 这一行(不是 `y.add_(100)` 那一行,而是"产出这个被污染的 tensor"的那一行——找到这一行之后,再顺着它的使用去找后续对它做了原地修改的代码,通常就不难了)。

**面试怎么问 + 追问链:**
- **Q:** "这条报错是什么意思?"—— 期望能讲出版本计数器机制(呼应01/02篇),不要求现场重新推导,但要能讲清楚"为什么会有这个检查"。
- **追问 1(承接机制,考察debug能力而不只是背知识点):** "光看这条报错,你怎么在几百行代码里定位到底是哪一行出的问题?"—— 期望直接说出 `torch.autograd.set_detect_anomaly(True)`,这是报错信息自己给出的提示,能不能想到"报错信息本身就是文档"也是一种工程素养。
- **追问 2:** "为什么 PyTorch 不直接默认开启这种更详细的追踪,而是要额外手动开启?"—— 期望能联系到第5节会讲到的性能代价(本节验证是"数倍"量级的运行时开销),默认关闭是"日常训练不为一个调试功能买单"的权衡。

**常见坑:** 只盯着报错文本里提到的算子类型(比如 "TanhBackward0")去猜测代码位置,而不是直接开 `detect_anomaly()` 让它自动定位——尤其模型里同一种算子出现多次时,靠猜的效率远低于让工具直接告诉你。

---

## 3. "element 0 of tensors does not require grad and does not have a grad_fn" —— 三种最常见的触发原因

**是什么:** 对一个 `requires_grad=False` 或者压根没有 `grad_fn` 的 tensor 调用 `.backward()`。

**一句话:** 这条报错字面意思很直白("这个 tensor 不需要梯度、也没有计算图"),但**触发它的真实原因往往不是"我以为需要梯度但漏标了"这么简单**——本节验证了三种真实场景,只有第一种是"忘记标 requires_grad"这种直觉原因,后两种更隐蔽。

**底层机制/为什么这样设计:** `.backward()` 要求被调用的 tensor 本身 `requires_grad=True` 且存在 `grad_fn`(意味着它是某次可微计算的输出)——只要这两个条件有一个不满足,直接触发这条报错。

**可运行例子(三种真实触发场景):**

```python
import torch, torch.nn as nn

# 场景1:最直觉的原因——leaf tensor忘了标 requires_grad=True
w = torch.tensor([1.0, 2.0])            # 忘记 requires_grad=True
loss = (w * torch.tensor([3.0, 4.0])).sum()
try:
    loss.backward()
except RuntimeError as e:
    assert str(e) == "element 0 of tensors does not require grad and does not have a grad_fn"

# 场景2:参数本身没问题,但计算过程被包在 no_grad() 里,切断了计算图(呼应02篇)
w2 = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    y = w2 * 2                           # 在no_grad里算出来的y,没有grad_fn
loss2 = y.sum()
try:
    loss2.backward()
except RuntimeError as e:
    assert str(e) == "element 0 of tensors does not require grad and does not have a grad_fn"

# 场景3:更隐蔽——想冻结"部分"参数,结果不小心把"全部"参数都冻结了,
# 导致loss本身连着的所有leaf都requires_grad=False,整条计算图从头到尾都不带梯度
model = nn.Linear(3, 3)
for p in model.parameters():
    p.requires_grad_(False)    # 本意可能是"先冻结,后面再解冻部分",但如果忘了解冻……
loss3 = model(torch.randn(1, 3)).sum()
try:
    loss3.backward()
except RuntimeError as e:
    assert str(e) == "element 0 of tensors does not require grad and does not have a grad_fn"
```

**AI 研究场景:** 场景 2 在"用预训练模型提特征、只训练新加的头"这类迁移学习代码里很常见——如果不小心把提特征这一步也包进了外层的 `with torch.no_grad():`(本来只想省显存/加速这一步的推理),连同下游本该训练的头的梯度也一起被切断了;场景 3 在"先冻结全部参数、再选择性解冻几层做微调"的两段式代码里高发,如果解冻那一步的判断条件写错(比如层名字拼写不匹配导致一个都没解冻成功),会在看似"代码逻辑没问题"的情况下,训练一开始就直接报错。

**面试怎么问 + 追问链:**
- **Q:** "这条报错,你能想到几种可能的原因?"—— 期望不止说出"忘记requires_grad=True"这一种,面试官通常会追问还有没有别的可能。
- **追问 1(区分度高):** "如果确认参数本身 `requires_grad=True` 没问题,还有什么原因会导致这个错?"—— 期望能提到"某段计算被包进了 `no_grad()` 上下文,切断了计算图",这是最容易被忽视的一种。
- **追问 2(工程向,联系06篇):** "如果只冻结了模型的一部分,为什么最终 loss 反而完全不能 backward?"—— 期望能推理出"如果'一部分'不小心变成了'全部'(比如条件判断写错导致该解冻的没解冻成功),整条从输入到loss的路径上就没有任何一个requires_grad=True的leaf,loss自然也不会有grad_fn"。

**常见坑:** 看到这条报错就本能地在最外层加 `requires_grad=True`,却没有意识到问题可能出在中间某处的 `no_grad()` 或者冻结逻辑——建议的排查方法是从 loss 往回顺着计算图检查:`loss.requires_grad`、组成 loss 的中间量的 `requires_grad`,一步步网上游追,直到找到"从这一步开始就是 False"的分界点。

---

## 4. size/shape 不匹配 —— 系统性排查方法论,不是死记每种报错文本

**是什么:** 矩阵乘法/逐元素运算/拼接等操作,因为参与运算的 tensor 形状不满足对应的规则而报错。

**一句话:** 不同类型的运算,形状不匹配报出来的错误文本和错误原因都不一样——矩阵乘法要求内维精确相等,逐元素运算走的是 broadcast 规则(不相等的维度必须有一方是1),搞清楚"当前这个报错属于哪一类规则被违反",比死记某一条报错文本更重要。

**底层机制/为什么这样设计 + 可运行例子(现场触发两类最常见的形状报错,对比措辞差异):**

```python
import torch

# 矩阵乘法:要求 mat1 的最后一维 == mat2 的第一维(不是broadcast规则)
a = torch.randn(3, 4)
b = torch.randn(5, 6)
try:
    a @ b
except RuntimeError as e:
    assert str(e) == "mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)"
    # 报错精确点出两个操作数各自的完整形状,重点检查"第一个的最后一维"和"第二个的第一维"

# 逐元素运算:走 broadcast 规则,要求从最后一维往前比,每一维要么相等、要么其中一个是1
a2 = torch.randn(3, 4)
b2 = torch.randn(3, 5)
try:
    a2 + b2
except RuntimeError as e:
    assert str(e) == "The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1"
    # 报错精确点出"在哪一维"、"两边分别是多少"——直接对着这一维去检查数据管道哪里出的形状偏差
```

**系统性排查方法论(比单纯读报错文本更重要的是养成的习惯):**

1. **先分清运算类型**:是矩阵乘法(`@`/`matmul`/`nn.Linear`,内维精确匹配规则)还是逐元素运算(`+`/`*`/大多数激活函数,broadcast 规则)——两类规则不同,排查方向也不同。
2. **报错文本里的维度下标是从0开始还是有其它含义**,精确定位到是哪一维出了问题,不要凭"大概是维度不对"就开始瞎试。
3. **养成在关键节点打印 `.shape` 的习惯**,尤其是数据从 dataloader 出来、进 embedding/conv、进 attention 这几个经典的"形状最容易在这里第一次出错"的位置。

**面试怎么问 + 追问链:**
- **Q:** "矩阵乘法和逐元素运算的形状报错,有什么本质区别?"—— 期望能说出"矩阵乘法要求内维精确匹配,逐元素运算走broadcast规则(允许某一维是1)",而不是笼统地说"形状不对"。
- **追问 1(工程向):** "在一个几十层的模型里,某处报了形状不匹配,你怎么最快定位到底是哪一层的问题?"—— 期望能提到"在forward里分段打印shape(或者用hook,呼应03篇register_forward_hook)、二分法排查"这类系统性方法,而不是逐行读代码猜。

**常见坑:** 把 broadcast 规则和矩阵乘法规则搞混——比如以为逐元素加法也要求"内维匹配",或者以为矩阵乘法也能像 broadcast 一样"某一维是1就能自动适配"(矩阵乘法不支持这种放宽,只有真正的 batch 矩阵乘法 `torch.bmm`/`@` 在支持的维度上有自己的一套 broadcast 规则,和"内维必须精确相等"这条核心约束不是一回事,不能混着理解)。

---

## 5. `detect_anomaly()` —— 用运行时开销换"精确定位到出问题的那一行"

**是什么:**
```python
with torch.autograd.detect_anomaly(check_nan=True):   # check_nan默认就是True(已验证签名)
    loss = model(x)
    loss.backward()
```

**一句话:** 开启之后,`autograd` 在 forward 阶段会记录每个算子对应的 Python 调用栈;一旦 backward 阶段真的出问题(inplace 版本冲突,或者产出了 nan),就能把"这个算子是在哪一行 forward 代码里创建的"完整打印出来,不再只是给你一个抽象的算子类型名字。

**底层机制/为什么这样设计:**

**必须完整包住 forward + backward,只包 backward 拿不到定位信息(现场验证这条使用细节):**

```python
import torch

def buggy_forward(x):
    y = torch.tanh(x)
    z = y + 1
    y.add_(100)
    loss = (z * y).sum()
    return loss

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

with torch.autograd.detect_anomaly():
    loss = buggy_forward(x)     # forward也必须在detect_anomaly上下文里,才能记录调用栈
    try:
        loss.backward()
    except RuntimeError as e:
        print(e)
# 额外打印的warning(不是异常,是backward报错之前的一条warning)精确包含:
#   "Traceback of forward call that caused the error:
#      File ..., line N, in buggy_forward
#        y = torch.tanh(x)"
# 精确指到产生这个"后来被污染"的tensor的那一行forward代码
```

**`check_nan=True`(默认值)额外的能力:检测 backward 过程中产生的 nan,不只是inplace版本冲突:**

```python
x2 = torch.tensor([0.0], requires_grad=True)
y2 = torch.sqrt(x2)          # sqrt在0处的导数是inf
loss2 = (y2 * 0).sum()        # inf * 0 = nan,会出现在梯度链条里

with torch.autograd.detect_anomaly():
    try:
        loss2.backward()
    except RuntimeError as e:
        assert "returned nan values" in str(e)
        # 实测: "Function 'SqrtBackward0' returned nan values in its 0th output."
```

**代价(现场实测,不是泛泛地说"会变慢"):**

```python
import torch, torch.nn as nn, time

device = 'cuda'
model = nn.Sequential(*[nn.Linear(512, 512) for _ in range(10)]).to(device)
x = torch.randn(256, 512, device=device, requires_grad=True)
for _ in range(5):
    model(x).sum().backward()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    model(x).sum().backward()
torch.cuda.synchronize()
t_normal = (time.perf_counter() - t0) / 20

t0 = time.perf_counter()
with torch.autograd.detect_anomaly():
    for _ in range(20):
        model(x).sum().backward()
torch.cuda.synchronize()
t_anomaly = (time.perf_counter() - t0) / 20

print(f"normal: {t_normal*1000:.3f}ms  with detect_anomaly: {t_anomaly*1000:.3f}ms  "
      f"slowdown: {t_anomaly/t_normal:.2f}x")
# 实测(独立重跑5次,含一次紧接在另外两次detect_anomaly调用之后的"热态"测量):
#   5.0x / 5.8x / 6.0x / 7.1x / 2.9x —— 量级稳定在"数倍",绝对倍数本身对测量条件
# (是否紧邻其它detect_anomaly调用、系统负载)比较敏感,不应该当成一个精确常数记,
# 但"明显更慢"这个方向在所有重跑里都成立
assert t_anomaly > t_normal * 1.5
```

**AI 研究场景:** 训练过程中偶发性地(比如每几千步一次)出现 nan/inf 导致训练崩掉,是大模型训练里排查成本很高的一类问题——`detect_anomaly()` 因为开销大(本节独立重跑5次实测都在3~7倍这个量级),通常不会全程开着,而是先复现问题、缩小到能稳定复现的最小场景(更小的模型/更少的数据),再开 `detect_anomaly()` 精确定位,而不是在大规模训练任务上全程开启。

**面试怎么问 + 追问链:**
- **Q:** "`detect_anomaly()` 具体能帮你做什么?"—— 期望说出"定位inplace版本冲突具体发生在forward的哪一行"和"检测backward过程中的nan"这两个能力,不是只知道其中一个。
- **追问 1(容易漏答的使用细节):** "只在 `backward()` 调用外面包一层 `detect_anomaly()`,不包 `forward()`,能定位到出问题的forward代码行吗?"—— 期望答"不能,必须连 forward 一起包进去,因为调用栈是在forward执行时被记录的,只在backward阶段打开这个模式,已经错过了记录调用栈的时机"。
- **追问 2(工程判断):** "既然这么有用,为什么不干脆一直开着?"—— 期望能给出量化的理由(本节多次实测都在3~7倍这个量级的开销),并且能说出合理的使用策略(问题复现后针对性开启,不是全程开启)。

**常见坑:** 在大规模/长时间训练任务里全程开着 `detect_anomaly()` "以防万一"——本节实测的数倍开销在小规模调试时不明显,但在真实大模型训练的时间/算力成本尺度上是不可接受的,应该只在主动排查具体问题时短暂开启。

---

## 6. nan/inf 定位方法 —— 从"哪里能看到 nan"到"哪一层最先产生 nan"

**是什么:** `torch.isnan(x)` / `torch.isinf(x)` —— 逐元素检查是否为 nan/inf,返回同形状的布尔 tensor。

**一句话:** 知道"某个 tensor 里有 nan"只是第一步,真正有用的排查是"在一个几十层的模型里,第一次出现 nan/inf 是在哪一层"——手动到处插 `print(torch.isnan(x).any())` 效率很低,`register_forward_hook`(呼应 [03-nn-module-internals.md](03-nn-module-internals.md) 第 6 节)能把这件事自动化。

**底层机制/为什么这样设计:**

**先现场制造一条真实的 nan 传播链条(不是构造一个抽象例子):**

```python
import torch

x = torch.tensor([1000.0, 1.0, 2.0])
h1 = torch.exp(x)                 # exp(1000) 直接数值溢出成 inf
assert torch.isinf(h1).any()
print("h1:", h1.tolist())          # [inf, 2.718, 7.389]

h2 = h1 - h1[0]                     # inf - inf = nan(经典的nan生成模式之一)
assert torch.isnan(h2).any()
print("h2:", h2.tolist())          # [nan, -inf, -inf]
```

这是理解 nan 从哪来的关键:**nan 几乎不是凭空出现的,通常是"某处先出现了 inf(数值溢出),这个 inf 后续参与了 `inf-inf`/`inf*0`/`0/0` 这类运算才变成 nan"**——排查时如果只搜"第一次出现 nan 的地方",经常已经晚了一步,真正的病根(第一次出现 inf 的地方)在更早的层。

**用 `register_forward_hook` 自动定位"第一层出问题"的层(比手动逐层print规模化得多):**

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
first_bad_layer = {"idx": None}

def make_hook(idx):
    def hook(module, inp, out):
        if first_bad_layer["idx"] is None and (torch.isnan(out).any() or torch.isinf(out).any()):
            first_bad_layer["idx"] = idx
    return hook

for i, layer in enumerate(model):
    layer.register_forward_hook(make_hook(i))

model[0].weight.data.fill_(1e20)     # 故意制造一个会在第0层就溢出的权重
model[0].bias.data.fill_(1e20)
model(torch.tensor([[1e20, 1e20, 1e20]]))
assert first_bad_layer["idx"] == 0
print("first layer producing nan/inf:", first_bad_layer["idx"])
```

**AI 研究场景:** 训练大模型时"loss 在第N步突然变成 nan"是经典排查场景,标准流程通常是:①先确认是 forward 阶段还是 backward 阶段先出现 nan(forward 阶段用上面的 hook 方法;backward 阶段配合第 5 节的 `detect_anomaly(check_nan=True)`);②定位到具体层之后,再结合这一层的输入数据范围、权重初始化、学习率等,分析是数值溢出(呼应05篇log-sum-exp稳定性)、梯度爆炸(呼应06篇`clip_grad_norm_`)还是别的原因。

**面试怎么问 + 追问链:**
- **Q:** "训练中途出现 nan loss,你怎么排查?"—— 期望能提出系统性方法,而不是"调小学习率试试"这种没有诊断过程的猜测。
- **追问 1(核心):** "nan 一般是怎么产生的?"—— 期望能说出"通常先有 inf(数值溢出),inf 参与某些运算(inf-inf、inf*0、0/0)之后变成 nan",这决定了排查时要往"更早"的地方找,而不是只看nan第一次出现的位置。
- **追问 2(工程向,联系03篇):** "在一个几十层的模型里,怎么高效定位是哪一层先产生的异常值,而不是每层手动加print?"—— 期望能想到用 `register_forward_hook` 批量给每一层挂检测逻辑,这是 03 篇 hook 机制的一个实际应用场景。

**常见坑:** 只检查"loss 是不是 nan",不往前追查是模型内部哪一层先产生的异常值——直接调小学习率、加梯度裁剪这类"通用止血"手段有时能让训练不崩,但没有找到真正的根因(比如某一层权重初始化不合理、某处确实存在真实的数值溢出风险),类似问题可能在换个数据分布/换个超参之后再次复发。

---

## 7. 设备不一致报错 —— 两种长得不一样、但本质相同的报错文本

**是什么:** 参与同一次运算的多个 tensor 分布在不同设备(比如一个在 `cuda:0`、一个在 `cpu`)。

**一句话:** 同样是"设备不一致",不同算子报出来的文本长得不太一样——逐元素运算和 `nn.Linear`/矩阵乘法给出的报错措辞有明显差异,如果只认过其中一种,遇到另一种容易一时反应不过来是同一类问题。

**可运行例子(现场触发两种典型场景,对比报错文本):**

```python
import torch, torch.nn as nn

a = torch.randn(3, 3, device='cuda')
b = torch.randn(3, 3, device='cpu')
try:
    a + b
except RuntimeError as e:
    assert str(e) == "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"

model = nn.Linear(3, 3).to('cuda')
x_cpu = torch.randn(2, 3)
try:
    model(x_cpu)
except RuntimeError as e:
    # 措辞明显不同:点名是matmul内部的哪个操作数在哪个设备,而不是笼统地说"发现两个设备"
    assert "Expected all tensors to be on the same device" in str(e)
    assert "mat1 is on cpu" in str(e)
```

**AI 研究场景:** 最高发的场景是"模型 `.to('cuda')` 了,但从 dataloader 出来的 batch 忘记 `.to(device)`"——这在刚开始学 PyTorch 时几乎人人踩过,但更隐蔽的版本是:模型内部某个中间变量是在 `__init__` 里用 `torch.tensor(...)` 直接创建的常量(没有注册成 `nn.Parameter` 或 `register_buffer`,呼应 [03-nn-module-internals.md](03-nn-module-internals.md) 第 4 节),这类"游离"在模块自动设备管理之外的 tensor,模型整体 `.to(device)` 时不会跟着一起移动,训练到用它的那一步才会报错,报错位置和真正的根因(`__init__` 里的写法)距离较远,不容易一眼看出。

**面试怎么问 + 追问链:**
- **Q:** "'设备不一致'这类报错,你遇到过哪些不同的触发场景?"—— 期望不只说"忘记 .to(device)"这一种,能提到"模块内部有游离在自动设备管理之外的普通tensor"这类更隐蔽的情况。
- **追问 1(联系03篇):** "为什么用 `register_buffer` 注册的 tensor 不会有这个问题,普通的 `self.x = torch.tensor(...)` 会?"—— 期望能直接引用03篇的结论:`register_buffer` 注册的 tensor 会被 `nn.Module.__setattr__` 自动纳入 `_buffers`,`.to(device)` 时会一并处理;普通属性赋值的 tensor 不会被这套机制感知到,自然不会跟着模型一起搬设备。

**常见坑:** 看到"设备不一致"报错就无脑在报错发生的那一行加 `.to(device)` 让代码"能跑起来",而不去追查为什么会有一个 tensor 忘了放到正确设备上——这种"头痛医头"的补丁容易在代码重构之后再次出现同类问题,治本的做法是确保模块里所有需要跟着模型走的 tensor 都通过 `nn.Parameter`/`register_buffer` 正确注册。

---

## 8. `requires_grad` 误用 —— 三个容易混淆的相关陷阱

**是什么(一组相关但不同的坑,不是单一报错):** 围绕 `requires_grad` 的几种典型误用模式。

**一句话:** `requires_grad` 相关的坑不止"忘记设置"这一种——本节验证的三个场景分别对应"直接在 leaf 上做原地操作"(报错,但和第2节的报错文本不同)、"用 `.detach()` 想要一份'可修改的副本'结果共享了存储"(不报错,但悄悄破坏原tensor)、"部分冻结参数的正确写法"(不报错,行为正确,用来对比前两种错误)。

**底层机制/为什么这样设计 + 可运行例子:**

**陷阱1:对一个 `requires_grad=True` 的 leaf tensor 直接做原地操作,报错文本和第2节不是同一条:**

```python
import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
try:
    w.add_(1.0)          # 直接原地改一个requires_grad=True的leaf
except RuntimeError as e:
    assert str(e) == "a leaf Variable that requires grad is being used in an in-place operation."
    # 注意:这条和第2节"modified by an inplace operation"文本完全不同——
    # 这里是"压根不允许对requires_grad=True的leaf做原地操作"(更早期的、更直接的拦截),
    # 第2节是"允许原地操作,但backward发现被改过的值和它需要的对不上"(更晚期才发现的问题)
```

**陷阱2:`.detach()` 不是"深拷贝",拿到的 tensor 和原 tensor 共享存储,原地修改会互相影响:**

```python
w2 = torch.tensor([1.0, 2.0], requires_grad=True)
w2_detached = w2.detach()
assert w2_detached.requires_grad is False   # detach出来的确实不需要梯度

w2_detached.add_(100)          # 这个操作本身不报错(detached tensor不是"requires_grad的leaf")
assert w2.tolist() == [101.0, 102.0]   # 但w2本身也被改了!因为共享同一块存储
assert w2.requires_grad is True         # w2的requires_grad属性没变,但数值已经被污染
```

**陷阱3(正确用法,用来对比):部分冻结参数的标准写法——只对想冻结的那部分设 `requires_grad=False`,其余不变,`optimizer` 仍可以把全部参数传进去:**

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
for p in model[0].parameters():
    p.requires_grad_(False)     # 只冻结第0层

opt = torch.optim.SGD(model.parameters(), lr=0.1)   # 仍然可以把全部参数传给optimizer
w0_before = model[0].weight.detach().clone()
w1_before = model[1].weight.detach().clone()

loss = model(torch.randn(1, 3)).sum()
loss.backward()
assert model[0].weight.grad is None        # 冻结层:没有梯度
assert model[1].weight.grad is not None     # 未冻结层:正常有梯度
opt.step()                                    # 不报错(06篇讲过:.grad is None的参数被自动跳过)
assert torch.equal(model[0].weight, w0_before)      # 冻结层权重没变
assert not torch.equal(model[1].weight, w1_before)    # 未冻结层权重正常更新
```

**AI 研究场景:** 陷阱2("以为detach是深拷贝")在"想要一份模型输出用来做后处理/可视化,又不想影响原计算图"这类场景很容易踩到——如果后续对这份"detach 出来的副本"做了原地修改,原 tensor 会被静默污染,而且这类 bug 经常要等到很久之后"为什么这个不该变的中间结果变了"才会被发现;陷阱3(正确的部分冻结写法)是几乎所有微调/迁移学习代码的标配模式。

**面试怎么问 + 追问链:**
- **Q:** "`.detach()` 返回的 tensor,和原 tensor 是什么关系?"—— 期望答"共享底层存储,只是切断了梯度追踪(requires_grad=False、没有grad_fn),不是数据上的深拷贝"。
- **追问 1(陷阱题):** "如果我对 `.detach()` 出来的 tensor 做原地修改,原 tensor 会受影响吗?"—— 期望答"会,因为共享存储",能现场举出上面的验证例子是加分项。
- **追问 2:** "如果真的需要一份完全独立、不共享存储的副本,应该怎么写?"—— 期望答 `.detach().clone()`(或者 `.clone().detach()`,顺序不影响结果),而不是只用 `.detach()`。
- **追问 3(呼应第2节,考察能否区分相似报错):** "'a leaf Variable that requires grad is being used in an in-place operation' 和 'modified by an inplace operation' 这两条报错,是同一个问题吗?"—— 期望能说出"不是,前者是直接禁止对requires_grad=True的leaf做原地操作(更早拦截),后者是允许原地操作发生,但backward时发现某个被保存的值已经不对了(更晚才发现)",体现对这两条相似报错的精确区分。

**常见坑:** 混淆"陷阱1"和第2节的报错,把两条不同的报错当成同一个问题去排查,浪费时间;另外把 `.detach()` 误当成 `.clone()` 的同义词使用,这在"只是想读取一下数值、不打算修改"的场景下没有影响,但一旦后续代码不小心对这份"detach 结果"做了原地操作,就会追着一个"看似无关"的 bug 排查很久。

---

## 9. reshape/view 报错 —— 报错信息里已经写好了修复方案

**是什么:** `RuntimeError: view size is not compatible with input tensor's size and stride...`

**一句话:** [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 3、4 节已经讲透了 `view()` 要求底层内存连续、`reshape()` 在不连续时会自动退化成拷贝这套机制,本节要强调一个容易被忽略的细节:**这条报错信息本身,就已经把"该怎么修"写在最后一句了**。

**底层机制/为什么这样设计 + 可运行例子:**

```python
import torch

c = torch.randn(4, 6)
ct = c.T                    # transpose之后不再连续(呼应01篇)
try:
    ct.view(24)
except RuntimeError as e:
    print(e)
    # 实测报错原文: "view size is not compatible with input tensor's size and stride
    #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead."
    assert "Use .reshape(...) instead" in str(e)   # 解决方案已经写在报错信息的最后一句!

result = ct.reshape(24)      # 按报错提示直接换成reshape,确实work
assert result.shape == torch.Size([24])
assert result.data_ptr() != ct.data_ptr()   # 呼应01篇:这里reshape退化成了拷贝,不是view那种zero-copy
```

**这条报错值得多想一步的地方:** `reshape()` 能解决"报错消失"这个表面问题,但如果这段代码在一个性能敏感的热路径里被反复调用,每次都触发一次隐藏的拷贝(而不是零拷贝的 `view`),累积开销可能不小——遇到这条报错时,更完整的排查思路除了"改成 reshape 让它先跑起来",还应该问一句"这个 tensor 为什么会变成不连续的(是不是前面有不必要的 `transpose`/`permute`),能不能从源头避免,而不是每次都靠 reshape 兜底"。

**面试怎么问 + 追问链:**
- **Q:** "遇到这条 view 报错,你会怎么处理?"—— 期望不只说"换成reshape",能提一句"报错信息本身已经建议了这个方案"体现出真的读过报错文本,而不是习惯性地搜索报错关键词。
- **追问 1(深挖,呼应01篇,考察是否只满足于'能跑'):** "只是无脑把所有 `view()` 都换成 `reshape()`,有没有隐藏代价?"—— 期望能说出"`reshape()` 在能zero-copy时行为和view一样,但遇到不连续输入会自动退化成拷贝,如果这种情况在热路径里频繁发生,会有不必要的额外拷贝开销,更彻底的做法是排查为什么会变得不连续、能否从源头避免"。
- **追问 2:** "什么操作最容易让一个 tensor 变得不连续?"—— 期望能直接引用01篇内容:`transpose()`/`permute()`/一些高级索引操作,这些都只是改变了 stride 元数据,不会重排底层内存。

**常见坑:** 把这条报错当成"随手加 `.contiguous()` 或者换成 `.reshape()` 就完事"的机械操作,不去想一想数据流经过了哪一步导致了不连续——大多数场景下 `.reshape()` 确实是又快又对的解法,但如果同一个不连续 tensor 在一个循环里被反复 reshape,值得回头看看是不是从设计上就能避免制造出这个不连续的中间结果。

---

## 小结:这一批 9 个知识点解决的问题

| # | 知识点 | 核心结论 |
|---|------|---------|
| 1 | CUDA OOM | 报错文本自带allocated/reserved/free等排查线索,先读数字再决定是"真不够"还是"碎片化" |
| 2 | inplace报错定位难题 | 默认只报算子类型不报代码行号,需要配合第5节的`detect_anomaly()`才能精确定位到forward哪一行 |
| 3 | "does not require grad"三种成因 | 忘记requires_grad(直觉)、被no_grad()切断计算图、误将部分冻结做成全部冻结(隐蔽) |
| 4 | shape不匹配方法论 | 矩阵乘法(内维精确匹配) vs 逐元素运算(broadcast规则)是两类不同规则,报错测辞也不同 |
| 5 | `detect_anomaly()` | 必须连forward一起包住才能记录调用栈;check_nan默认True还能抓nan;实测开销稳定在3~7倍量级,只用于主动排查 |
| 6 | nan/inf定位 | nan通常由更早的inf经过inf-inf/inf*0等运算产生;`register_forward_hook`可批量自动定位第一个出问题的层 |
| 7 | 设备不一致 | 不同算子报错措辞不同(笼统 vs 精确点名matmul操作数);根因常是模块内游离于`register_buffer`外的tensor |
| 8 | requires_grad三个陷阱 | leaf原地操作报错(区别于第2节)、`.detach()`共享存储不是深拷贝、正确的部分冻结写法(用于对比) |
| 9 | view/reshape报错 | 报错信息自带修复方案("Use .reshape(...) instead");但要多想一步排查不连续的源头,不是无脑套用 |

---

## torch-deep-dive 系列完结:约 100 个知识点回顾

这是 torch-deep-dive 系列的第 11 批,也是最后一批。11 个文件加起来正好 **100 个知识点**(12+14+10+11+7+10+7+8+6+6+9),从最基础的 tensor 内存模型,一路铺到分布式训练、序列化部署、报错排查——覆盖了"训练一个模型,从写第一行 `import torch` 到把 checkpoint 部署出去"这条完整链路上,大厂技术面试二三四面真正会深挖的那些"为什么"。

| # | 分类 | 文件 | 知识点数 |
|---|------|------|--------|
| 01 | Tensor 内存模型与基础操作深挖 | [01-tensor-memory-model.md](01-tensor-memory-model.md) | 12 |
| 02 | Autograd 核心机制 | [02-autograd-internals.md](02-autograd-internals.md) | 14 |
| 03 | nn.Module 系统内核 | [03-nn-module-internals.md](03-nn-module-internals.md) | 10 |
| 04 | 常用层前向反向数学推导 | [04-layers-math-and-backward.md](04-layers-math-and-backward.md) | 11 |
| 05 | 损失函数与数值稳定性 | [05-loss-functions-and-numerical-stability.md](05-loss-functions-and-numerical-stability.md) | 7 |
| 06 | 优化器内部机制 | [06-optimizer-internals.md](06-optimizer-internals.md) | 10 |
| 07 | 训练循环深层机制 | [07-training-loop-internals.md](07-training-loop-internals.md) | 7 |
| 08 | 内存与性能 | [08-memory-and-performance.md](08-memory-and-performance.md) | 8 |
| 09 | 分布式训练基础机制 | [09-distributed-training-basics.md](09-distributed-training-basics.md) | 6 |
| 10 | 序列化与部署基础 | [10-serialization-and-deployment.md](10-serialization-and-deployment.md) | 6 |
| 11 | 调试与常见报错精解 | [11-debugging-and-common-errors.md](11-debugging-and-common-errors.md) | 9 |
| | **合计** | | **100** |

**这个系列贯穿始终的一条原则,比任何单个知识点都重要:凡是能验证的,现场跑代码验证,不转述、不凭记忆。** 这条原则在写作过程中不是一句空话——系列里至少有 4 处真实的自查修正,都是先写了"看起来合理"的内容、再实际运行代码时被拆穿的:

- 01 篇:两处浮点数/stride 的具体数值错误(复制粘贴时没有针对新场景重新计算)。
- 06 篇:一处硬编码的、依赖未设种子随机数的"示例数值"(重新运行会得到完全不同的数字)。
- 08 篇:一处实验设计缺陷(把不必要的矩阵乘法重新计算放进了计时循环内,导致噪声掩盖了真实想测量的信号)。
- 09 篇:一处 API 使用错误(按"空格拼接"的整句去匹配一段实际用换行符折行排版的官方 docstring)。

这些修正本身,连同系列里反复出现的"float32 别用 `==`,要用 `torch.allclose`"、"随机数据要设种子保证可复现"、"跨小节共享一个进程时,库的持久化内部状态会污染'从零开始'的假设"这几条教训,可能比任何一个单独的 API 知识点都更值得带走——它们不是 PyTorch 的知识,是"怎么可靠地验证一个技术论断"这件事本身的方法论,在 PyTorch 之外一样适用。

---

*更新:2026-07-07*
