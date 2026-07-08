# 08 · 内存与性能深挖(Memory and Performance)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批讲的不是"新概念",而是把前面几篇建立的底层认知(尤其是 [01-tensor-memory-model.md](01-tensor-memory-model.md) 的 stride/内存布局、pin_memory)接到"性能"这个维度上——一个训练脚本跑得慢、显存爆了、`nvidia-smi` 和代码里查到的数字对不上,原因几乎总能落到本篇这 8 个知识点里的某一个。这也是"面试二三四面"最喜欢挖的一类问题:面试官通常不问"这个 API 怎么用",而是甩给你一个"训练脚本莫名其妙 OOM 了/变慢了",看你有没有系统性的排查思路。

**本文定位和前面几篇的关系:** [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 10、11 节已经讲过 `.to(device)` 的 no-op 判定和 `pin_memory()`/`non_blocking` 的基本机制,本篇第 8 节会在 `DataLoader` 整体吞吐的视角下重新用到这些结论,不重复基础概念;本篇也是 [07-training-loop-internals.md](07-training-loop-internals.md)(`autocast`/梯度检查点等训练循环机制)和 [09-distributed-training-basics.md](09-distributed-training-basics.md)(分布式场景下显存与通信的进一步权衡)的地基。

本文所有代码例子已在仓库 `.venv`(torch 2.11.0+cu128,CUDA 可用,GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU,16GB 显存,compute capability 8.6)下实际跑通验证。**本篇对"性能/内存有没有差异"这类问题格外较真**——凡是给出的数字,都是当场用 `torch.cuda.memory_allocated()`/`memory_reserved()`/`time.perf_counter()` + `torch.cuda.synchronize()` 实测得到的,不是转述博客或文档里的"经验数字";文中会如实标注"这个数字在我机器上多次重跑有多大波动",而不是挑一次好看的结果贴上去。

**本篇统一结构(与 01 篇一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(现场测量,不转述)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. CUDA 缓存分配器(caching allocator)—— 为什么 `nvidia-smi` 和代码里的数字对不上

**是什么:**
```python
torch.cuda.memory_allocated(device=None)   # 当前有多少字节"正被某个 tensor 实际引用"
torch.cuda.memory_reserved(device=None)    # 缓存分配器手上攥着的显存池总大小(已用 + 缓存着没用的)
torch.cuda.empty_cache()                    # 把缓存池里当前没有 tensor 在用的部分,真正还给 CUDA driver(第6节详讲)
```

**一句话:** PyTorch 不会每次分配/释放显存都调用一次昂贵的 `cudaMalloc`/`cudaFree`,而是自己维护一个显存池(caching allocator)——`memory_allocated()` 是"逻辑上正被使用"的部分,`memory_reserved()` 是"这个池子当前从 driver 那里攥着的总量",两者之差就是"缓存着、随时能复用、但当前没有 tensor 在用"的显存,这部分 `nvidia-smi` 会算进你的进程占用,但 PyTorch 自己的"已分配"统计不会。

**底层机制/为什么这样设计:** `cudaMalloc`/`cudaFree` 涉及和 GPU 驱动的同步通信,是重量级调用——如果训练循环里每个中间 tensor 的产生和释放都真实触发一次 driver 调用,性能会非常差。下面会用实测量化这个"重"到底有多重。PyTorch 的解法是自己管理一个"显存池":第一次需要显存时,以较大粒度(实测:向上取整到 **2MB** 的整数倍,例子里会验证)向 driver 申请一大块;之后同大小的显存请求直接从池子里"切"出来复用,tensor 被释放时,这块内存"还给池子"而不是还给 driver。`nvidia-smi` 是从操作系统/驱动视角统计"这个进程从 driver 要走了多少显存",它看到的自然是 `reserved`(池子总大小),不是 `allocated`(池子里当前真正被引用的部分)——这就是"我明明 `del` 了一堆 tensor,`nvidia-smi` 里显存却没降"的真正原因,不是内存泄漏,是缓存生效了。

**AI 研究场景:** 排查"两个实验之间切换,显存好像没完全释放"时,第一反应应该是查 `memory_reserved()` vs `memory_allocated()`,而不是只看 `nvidia-smi`——如果 `allocated` 已经很低但 `reserved` 还很高,说明只是缓存没还给 driver,不是真的泄漏(真泄漏见第5节,`empty_cache()` 何时真的有用见第6节);反过来如果 `allocated` 本身在持续上涨,才是需要认真排查的信号。

**可运行例子:**
```python
import torch

def mb(x):
    return x / 1024**2

torch.cuda.empty_cache()
print(f"基线: allocated={mb(torch.cuda.memory_allocated()):.1f}MB reserved={mb(torch.cuda.memory_reserved()):.1f}MB")
# 实测: allocated=0.0MB reserved=0.0MB

x = torch.randn(100_000_000, device='cuda')       # 100M * 4B ≈ 381.5MB (逻辑大小)
print(f"分配后: allocated={mb(torch.cuda.memory_allocated()):.1f}MB reserved={mb(torch.cuda.memory_reserved()):.1f}MB")
# 实测: allocated=382.0MB reserved=382.0MB  (分配器向上取整到 2MB 粒度: 191*2=382,验证了取整规则)
assert torch.cuda.memory_reserved() == 191 * 2 * 1024**2

del x
print(f"del后(未 empty_cache): allocated={mb(torch.cuda.memory_allocated()):.1f}MB reserved={mb(torch.cuda.memory_reserved()):.1f}MB")
# 实测: allocated=0.0MB reserved=382.0MB  <- 关键: reserved 完全没降,证明"缓存"确实发生了
assert torch.cuda.memory_allocated() == 0
assert torch.cuda.memory_reserved() > 0

y = torch.randn(100_000_000, device='cuda')        # 再分配一个同尺寸 tensor
print(f"复用缓存分配: allocated={mb(torch.cuda.memory_allocated()):.1f}MB reserved={mb(torch.cuda.memory_reserved()):.1f}MB")
# 实测: allocated=382.0MB reserved=382.0MB  <- reserved 没有增长,说明是从缓存池里直接切出来的,没有新的 driver 调用
```

再验证一层:`nvidia-smi` 报的数字,到底是跟着 `allocated` 走还是跟着 `reserved` 走?

```python
import subprocess

def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

# 本机实测三个时间点(这台机器只有本进程在用 GPU):
#   分配 381.5MB tensor 后:            nvidia-smi=553MiB, torch allocated=382.0MB, reserved=382.0MB
#   del 掉、不 empty_cache:            nvidia-smi=553MiB(不变!), torch allocated=0.0MB, reserved=382.0MB
#   empty_cache() 之后:                nvidia-smi=171MiB,        torch allocated=0.0MB, reserved=0.0MB
# 结论: nvidia-smi 的数字紧跟 reserved,和 allocated 完全不相关;171MiB 是"just有个CUDA context"的固定开销
# (553 - 382 = 171,精确对上,不是巧合)
```

再用实测量化"缓存到底省了多少": 连续 200 次分配/释放同尺寸(~76.3MB)tensor,一组命中缓存,一组每次都强制 `empty_cache()` 逼出一次真实 driver 分配:
```python
# 命中缓存(不清空):       45.1ms / 200次 = 0.2255 ms/次
# 每次强制走真实driver分配:  765.2ms / 200次 = 3.8262 ms/次
# 慢了 17.0 倍 —— 这就是"caching allocator 为什么存在"的量化答案,不是空口说"cudaMalloc 很慢"
```

**面试怎么问 + 追问链:**
- **Q:** "`torch.cuda.memory_allocated()` 和 `torch.cuda.memory_reserved()` 有什么区别?为什么 `nvidia-smi` 显示的显存经常和这两个数字都对不上又都对得上?"—— 期望讲清楚"缓存池"这个中间层,并能指出 `nvidia-smi` 实际上跟的是 `reserved`。
- **追问 1:** "为什么 PyTorch 不在 tensor 被 `del` 之后立刻把显存还给 driver?"—— 期望答出 `cudaMalloc`/`cudaFree` 是重量级调用,最好能报出量级(本节实测约 17 倍)。
- **追问 2(深挖,区分度高):** "如果我的代码反复分配、释放很多**不同大小**的 tensor,会不会导致这个缓存池碎片化,即使 `reserved` 总量够用,某次分配还是失败?"—— 期望答"会",这是真实存在的问题(经典症状:`CUDA out of memory. Tried to allocate ... but PyTorch reserved ... free ...` 报错里 reserved 和 free 加起来明明够,还是失败),现代 PyTorch 提供 `PYTORCH_CUDA_ALLOC_CONF` 环境变量(比如 `expandable_segments:True`)来缓解,这是本节故意留的一个"往深了问"的口子,能接上说明候选人不止会背诵基础结论。

**常见坑:** 把 `nvidia-smi` 的数字当成"我的模型实际用了多少显存"来做容量规划——它包含了缓存池里的冗余部分,`memory_allocated()` 才是"tensor 真正占用"的数字;反过来只看 `memory_allocated()` 判断"还能分配多少"也不严谨,因为哪怕 `allocated` 很低,如果 `reserved` 已经攥住了大部分显存又发生了碎片化,新的、大小不匹配的分配请求依然可能失败——这正是 `empty_cache()`(第6节)存在意义的伏笔。

---

## 2. `.item()`/`.cpu()` 触发的同步开销

**是什么:**
```python
tensor.item()   # 单元素 tensor -> Python 标量,必须先同步再读值
tensor.cpu()     # 把 tensor 搬回 CPU 内存(生成新 tensor),必须先同步再拷贝
```

**一句话:** GPU 上的计算默认异步执行——CPU 把"请算这个"这条指令丢进 CUDA stream 队列就立刻返回、不等真正算完;但 `.item()`/`.cpu()`/`print(tensor)` 这类需要把 GPU 上的值搬到 CPU、让 Python 能直接读到的操作,必须先等这个 tensor 依赖的所有排队中的 GPU 操作全部执行完(这个等待动作就是"同步"),才能安全地把值读出来——这会打断本该异步重叠的流水线。

**底层机制/为什么这样设计:** CUDA 编程模型里,几乎所有"发起计算"的调用(kernel launch)本身只是往命令队列(stream)里插入一条指令,由 GPU 按顺序异步取出执行,CPU 线程立刻拿回控制权——这是"GPU 计算和 CPU 逻辑重叠"的根本机制,PyTorch 默认执行模式完整继承了它。但 Python 的 `float`/`int` 是 CPU 内存里的对象,`.item()` 要返回一个真正的 Python 标量,这个数值必须先在物理上被 GPU 算完、通过 PCIe 搬回 host 内存——`.item()` 内部因此会强制发起一次等价于 `torch.cuda.synchronize()` 的等待,再做一次(通常很小的)device-to-host 拷贝。`.cpu()` 同理必须先同步,但如果搬的是一个大 tensor,还要**另外**付出真实的 PCIe 传输时间——"同步等待"和"数据传输"是两件不同的事,下面实测会把这两部分拆开看。

**AI 研究场景:** 训练循环里为了打印/记录 loss、给 tqdm 进度条更新数字、往 wandb/tensorboard 里塞一个标量,几乎总要调用一次 `.item()`——如果这一步放在每个 iteration 都执行,相当于人为地把本该异步流水线执行的 GPU 计算,每一步都拉回来强制排队等待。规范做法是攒够 N 步再读一次(或只在真正要打印/log 的那一步读),这也是为什么训练框架的 `Trainer` 类普遍有 `log_every_n_steps` 这样的参数。

**可运行例子:**

先拆解"同步开销"和"传输开销"这两件不同的事(4096×4096 fp32 矩阵乘法,充分 warmup 后测量,每组 30 次取均值):
```python
import torch, time

device = 'cuda'
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
for _ in range(30):            # 预热:让 cuBLAS 完成算法选择等一次性开销,排除干扰
    _ = a @ b
torch.cuda.synchronize()

# 连续发射30次不同步,只在最后统一 sync 一次 -> 单次"发射"的均摊耗时
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(30):
    c = a @ b
t_dispatch_only = (time.perf_counter() - t0) / 30
torch.cuda.synchronize()
t_with_final_sync = (time.perf_counter() - t0) / 30
# 实测: 纯发射 0.0313 ms/次;算上最后一次同步等待 28.5665 ms/次
# -> 发射耗时只占真实计算耗时的 0.11%,证明 CPU 发指令确实"几乎不用等 GPU"
assert t_dispatch_only < t_with_final_sync / 50

## ★ 复验时发现的问题:上面"每次都重新matmul再读取"的设计测不准

用这个写法(每次循环都重新 `c = a @ b` 再读取)独立重跑 5 轮,`t_cpu_full > t_item` 只出现 1/5 次——不是偶然噪声,是实验设计本身有缺陷:矩阵乘法自己的运行时抖动(受 GPU 频率/功耗状态影响,4096×4096 fp32 matmul 在这台笔记本卡上一次要跑约 28~30ms,轮次间波动可以有 1~2ms)和要测量的信号("多传 64MB 数据"应该多花的时间)是同一个量级,噪声把信号淹没了。

正确做法:把 matmul 挪到计时循环**之外**,只对同一个已经算好、已经同步过的 `c` 反复读取——这样"传输时间"这个效应才能和"计算时间的运行时抖动"干净地分离开:

```python
c = a @ b
torch.cuda.synchronize()          # c 已经算好并同步,后面两个循环不再重新触发矩阵乘法

t_item = 0.0
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50):
    val = c.sum().item()          # 反复读同一个已算好的 tensor
t_item = (time.perf_counter() - t0) / 50

t_cpu_full = 0.0
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50):
    c_host = c.cpu()               # 反复搬整个 64MB 矩阵回host
t_cpu_full = (time.perf_counter() - t0) / 50
# 实测(独立5轮,方向全部一致,不是单次幸运结果):
#   t_item  ≈ 0.25~0.58 ms/次(标量读取,几乎不含真实传输)
#   t_cpu   ≈ 9.0~10.6 ms/次(64MB PCIe传输的真实成本)
#   比值约 20~40x —— 隔离掉matmul噪声后,这才是干净、稳定可复现的信号,
#   比"混在matmul抖动里"的原始 1.78ms/1.05x 差值可靠得多
assert t_cpu_full > t_item        # 大张量的 .cpu() 比取标量的 .item() 更贵,多出的是传输,不是同步
```

这本身就是一条值得记住的方法论教训(呼应本篇开头"如实标注多次重跑的波动"的承诺):**测量一个小效应时,如果外面套了一层量级相近或更大的噪声源(这里是"没必要的重复计算"),很容易把真实存在、但本身很小的效应测成"看运气"——先想办法把混杂变量挪出计时区间,而不是加大量重复次数硬平均。**
```

再看一个更贴近训练循环的场景(256×256 矩阵乘 + relu + mean,共 500 步,对比"每步都 `.item()`" vs "每 50 步 `.item()` 一次"):
```python
import torch, time

device = 'cuda'
N, size = 500, 256

def one_step(x, w):
    return torch.relu(x @ w).mean()

x = torch.randn(size, size, device=device)
w = torch.randn(size, size, device=device)
for _ in range(20):
    one_step(x, w)
torch.cuda.synchronize()

t0 = time.perf_counter()
for i in range(N):
    loss = one_step(x, w)
    val = loss.item()            # 每步都同步
torch.cuda.synchronize()
t_every_step = time.perf_counter() - t0

t0 = time.perf_counter()
for i in range(N):
    loss = one_step(x, w)
    if i % 50 == 0:
        val = loss.item()        # 每50步同步一次
torch.cuda.synchronize()
t_every_50 = time.perf_counter() - t0

print(f"每步.item(): {t_every_step*1000:.1f}ms  每50步.item(): {t_every_50*1000:.1f}ms  慢 {t_every_step/t_every_50:.2f}x")
# 本机三次实测(GPU负载有波动,方向稳定): 2.47x / 3.84x / 4.12x —— 每步都同步稳定慢 2.5~4 倍
assert t_every_step > t_every_50
```

**面试怎么问 + 追问链:**
- **Q:** "为什么训练循环里不建议每一步都 `.item()` 打印 loss?"—— 期望讲出"异步流水线被打断",最好能举出量级(本节实测 2.5~4 倍)。
- **追问 1:** "是不是只要不调用 `.item()` 就不会有同步?"—— 期望答出"任何需要把 GPU 数据变成 CPU 可读值的操作都会同步",能举出 `.cpu()`、`print(tensor)`、`if gpu_tensor > 0`(把 tensor 当 Python bool 用)、`.numpy()` 这些同样触发同步的例子,而不是只知道 `.item()` 这一个。
- **追问 2(深挖):** "`.item()` 和 `.cpu()` 的开销是不是完全一回事?"—— 期望答出"都要同步,但 `.cpu()` 如果搬的是大 tensor,还要另外付出和数据量成正比的 PCIe 传输时间,`.item()` 因为只取一个标量,传输部分可以忽略"——这是本节实测特意拆开的点,能答出说明真的理解机制而不是背了一条规则。
- **追问 3(工程场景):** "分布式训练里,如果某一张卡的训练循环里有一次不必要的 `.item()`,只会拖慢这张卡自己吗?"—— 开放题,期望联系到 all-reduce 等集合通信需要所有 rank 互相等待,一张卡本地变慢会拖慢整个迭代(为09篇埋伏笔)。

**常见坑:** 只警惕 `.item()`/`.cpu()` 这两个名字,却没意识到 `print(gpu_tensor)`、把 GPU tensor 直接放进 `if` 判断、`.numpy()`(且会额外报错,GPU tensor 不能直接转 numpy)都是同样性质的隐式同步点——这些往往藏在看似无害的调试代码里,一行 `print` 就可能把整个训练循环的 GPU 利用率拖低。排查时如果发现 `nvidia-smi` 的 `Volatile GPU-Util` 有规律地跳动(忙-闲交替),值得检查循环体里是否有同步调用。

---

## 3. `memory_format` channels_last —— 卷积更喜欢的内存排布

**是什么:**
```python
x.to(memory_format=torch.channels_last)                 # NCHW tensor -> 物理上 NHWC 排布(.shape 打印出来仍是 NCHW)
x.is_contiguous(memory_format=torch.channels_last)        # 判断是否已经是这种排布
```

**一句话:** 默认的 4 维图像 tensor `(N,C,H,W)` 在内存里是"同一通道的所有像素连续存放"(NCHW 连续);`channels_last` 把物理排布换成"同一个空间位置的所有通道连续存放"(数学上等价于 NHWC),`.shape` 不变,变的只是底层内存的物理排列顺序——这是 [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 2 节 stride 机制在卷积网络场景的直接应用。

**底层机制/为什么这样设计:** 回忆 01 篇的 stride 公式:NCHW 连续布局下 `stride=(C·H·W, H·W, W, 1)`,最后一维(W)相邻元素在内存里也相邻;`channels_last` 把 stride 变成 `(C·H·W, 1, W·C, C)`——这时 C(通道)维度的 stride 变成 1,意味着同一个空间位置 `(h,w)` 上所有通道的数值在物理内存里挨在一起。这个排布对卷积更友好,原因是卷积运算本质是"对同一空间位置的所有输入通道做加权求和"——如果这些通道数值本来就连续存放,GPU 做内存合并访问、以及把数据凑成 Tensor Core 需要的小矩阵块都更直接,cuDNN 针对 NHWC 布局有专门优化过的 kernel 实现路径(尤其配合 fp16/bf16 精度)。**这不是"改改 stride 元数据"就能实现的零拷贝操作**——下面会验证 `data_ptr()` 确实变了,是一次真实的物理重排。

**AI 研究场景:** 训练/推理 CNN(ResNet、EfficientNet、Stable Diffusion U-Net 里的卷积层)时,`model.to(memory_format=torch.channels_last)` + 输入也转成 `channels_last` + 配合 `torch.autocast` 用 fp16/bf16,是官方推荐的标准组合,尤其在有 Tensor Core 的 GPU(Ampere/Hopper 等)上有意义。收益幅度因模型/分辨率/硬件差异很大,不能直接照抄网上的数字,必须在自己的场景里实测——这正是本节要用真实卷积做 benchmark 的原因。

**可运行例子:**

先验证 stride 变化和"这是真拷贝不是零拷贝":
```python
import torch

x = torch.randn(2, 3, 4, 4)             # N,C,H,W
assert x.stride() == (48, 16, 4, 1)      # 标准 NCHW 连续
assert x.is_contiguous(memory_format=torch.channels_last) is False

xc = x.to(memory_format=torch.channels_last)
assert xc.stride() == (48, 1, 12, 3)     # 通道维 stride 变成 1 —— NHWC 语义
assert xc.shape == x.shape                # 逻辑形状不变
assert xc.is_contiguous(memory_format=torch.channels_last) is True
assert xc.is_contiguous() is False         # 但按"标准 NCHW连续"的定义看,它不连续
assert xc.data_ptr() != x.data_ptr()       # 关键: data_ptr 变了,是真拷贝,不是像 transpose 那样的零拷贝
assert torch.equal(xc, x)                  # 数值读出来完全一样,只是物理排布不同

# 等价于这个 permute+contiguous+permute 的组合(验证了它的实现原理是"物理转置",不是元数据戏法)
equiv = x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
assert equiv.stride() == xc.stride()

# 模块的 .to(memory_format=...) 会同时转换权重(卷积权重也要是 channels_last,cudnn 才会真正选中优化 kernel)
conv = torch.nn.Conv2d(4, 8, 3)
w_before = conv.weight.stride()             # (36, 9, 3, 1)
conv2 = conv.to(memory_format=torch.channels_last)
assert conv2.weight.stride() == (36, 1, 12, 4)
assert conv.weight is conv2.weight           # 原地修改,返回的是同一个 Parameter 对象
```

再实测卷积性能差异(fp16,batch=32, 256通道, 56×56, 3×3 卷积,充分 warmup + cudnn.benchmark):
```python
import torch, time

torch.backends.cudnn.benchmark = True
device = 'cuda'
N, C, H, W, OC = 32, 256, 56, 56, 256
conv = torch.nn.Conv2d(C, OC, kernel_size=3, padding=1).to(device).half()

def bench(memory_format, iters=50, warmup=20):
    inp = torch.randn(N, C, H, W, device=device, dtype=torch.float16).to(memory_format=memory_format)
    conv_mf = conv.to(memory_format=memory_format)
    for _ in range(warmup):
        out = conv_mf(inp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = conv_mf(inp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

t_nchw = bench(torch.contiguous_format)
t_chl = bench(torch.channels_last)
print(f"NCHW: {t_nchw*1000:.3f}ms/iter, channels_last: {t_chl*1000:.3f}ms/iter, speedup={t_nchw/t_chl:.2f}x")
# 本机实测(重跑两次确认稳定): 4.673ms vs 3.981ms (1.17x) ; 4.743ms vs 4.032ms (1.18x)
# 换一组更大的配置(64,512,28,28)->512通道再验证一次方向: 10.383ms vs 8.578ms (1.21x)
assert t_chl < t_nchw   # 方向稳定:这台 Ampere 笔记本卡上 channels_last 确实更快,幅度是中等的 ~1.2x,不是夸张的"数倍"
```

**面试怎么问 + 追问链:**
- **Q:** "`channels_last` 是做什么的,为什么对卷积更快?"—— 期望讲出 NHWC 排布让同一空间位置的通道连续存放,更贴合卷积的访存模式。
- **追问 1:** "这个转换是不是和 `transpose` 一样零拷贝?"—— 期望答"不是",并能说出原因(不是单纯改 stride,底层数据物理位置真的变了,`data_ptr()` 会变)。
- **追问 2(深挖,区分度高):** "为什么 Tensor Core 尤其偏爱这种排布?"—— 期望能讲到 Tensor Core 做的是小块矩阵乘法,需要把数据凑成特定形状的连续小块送进去,NHWC 让"一个像素的所有通道"天然连续,更容易被硬件高效地打包成这样的小块,这是"访存模式匹配硬件设计"的具体例子,而不是泛泛地说"能加速"。
- **追问 3:** "如果只转换了输入没转换模型权重,会发生什么?"—— 期望答出"不会报错,但也吃不到性能红利",见下面常见坑。

**常见坑:** 只转换输入 tensor 却忘了 `model.to(memory_format=torch.channels_last)` 转换模型本身——卷积的 weight 也需要是 channels_last 排布,cuDNN 才会真正选中 NHWC 优化路径,否则输入和权重格式不匹配时 PyTorch 会静默插入一次额外的格式转换,不报错但也没有性能收益,这类"配置对了一半"的坑很难从报错信息里发现,只能靠实测对比或者检查 `out.stride()` 确认格式有没有真的传播下去。另外不是所有算子都保证会保持 channels_last 格式,链路中一旦有个不支持它的算子,格式会被静默转回 contiguous,后面的层就白转换了。

---

## 4. `torch.jit.script` vs `torch.jit.trace`

**是什么:**
```python
traced = torch.jit.trace(module, example_inputs)   # 拿样例输入实际跑一遍,记录下走过的算子序列
scripted = torch.jit.script(module)                  # 直接解析 forward 的 Python 源码(受限子集),生成静态图
```

**一句话:** `trace` 是"录像"——给一个具体输入实际执行一遍前向,把实际发生的算子调用序列固化下来,任何依赖输入*具体数值*的分支(`if`/`for`/`while`)都只会记录下这一次真正走过的那条路径;`script` 是"编译"——直接分析 Python 源码的语法结构,把 `if`/`for` 真正翻译成 TorchScript 里的控制流指令,能正确处理所有分支,代价是只支持一个受限的 Python 语法子集。

**底层机制/为什么这样设计:** `torch.jit.trace` 的实现方式是:用一批会记录"自己被如何操作过"的特殊 tensor 跑一次真实前向,过程中每一个被调用到的 ATen 算子都被记录进一张图里——这个机制天生只能看见"这一次实际发生了什么"。Python 层面的 `if x.sum() > 0` 这种判断,在 trace 阶段会被立即求值成一个具体的 `True`/`False`(决定这一次实际往哪个分支走),而这个判断本身**不会**被记录成图里的一个条件节点——所以 trace 出来的图里根本不存在分支结构,不管以后传入什么输入,永远执行"当时录像录到的"那条路径。`torch.jit.script` 不执行代码,而是解析函数体的抽象语法树,把 Python 的 `if` 语句翻译成 TorchScript 图里真正的 `prim::If` 节点(下面例子会直接打印图结构验证),因此天然能正确处理依赖运行时数值的控制流——代价是解析器必须能静态理解你写的每一行代码,支持面比 trace 窄。

**AI 研究场景:** 把研究阶段的模型导出成 TorchScript 用于 C++/移动端部署(第10篇会展开)时,很多人图省事直接用 `torch.jit.trace`——如果模型里有看起来无害的控制流(比如"batch 里有 padding 就走一条路径、没有就走另一条""训练/推理模式下行为不同的自定义模块"),trace 出来的模型在生产环境遇到没被样例输入覆盖到的分支时,会**安静地**算出错误结果而不是报错。这是 TorchScript 相关 bug 里最隐蔽的一类,因为本地用样例输入测一次"看起来完全正常"。

**可运行例子:**
```python
import torch

class BranchyModel(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        else:
            return x * -1

m = BranchyModel()
pos_input = torch.tensor([1.0, 2.0, 3.0])     # sum=6>0 -> eager 结果是 x*2 = [2,4,6]
neg_input = torch.tensor([-1.0, -2.0, -3.0])  # sum=-6<0 -> eager 结果是 x*-1 = [1,2,3]

assert m(pos_input).tolist() == [2.0, 4.0, 6.0]
assert m(neg_input).tolist() == [1.0, 2.0, 3.0]

# trace: 用 pos_input 录制 —— 录到的是 "*2" 这条分支
traced = torch.jit.trace(m, pos_input)
# 实测会打印 TracerWarning:
# "Converting a tensor to a Python boolean might cause the trace to be incorrect.
#  ... this value will be treated as a constant ... the trace might not generalize to other inputs!"
# 这不是无关噪音,是 PyTorch 在明确提醒你"接下来的行为可能不对"

assert torch.equal(traced(pos_input), pos_input * 2)          # 用回录制时的输入:对
assert not torch.equal(traced(neg_input), m(neg_input))        # 用另一个该走不同分支的输入:错!
assert torch.equal(traced(neg_input), neg_input * 2)            # 实测结果是 [-2,-4,-6]:trace 把 "*2" 分支焊死了

# script: 正确处理两个分支
scripted = torch.jit.script(m)
assert torch.equal(scripted(pos_input), m(pos_input))
assert torch.equal(scripted(neg_input), m(neg_input))           # 两个分支都对

# 直接看图结构差异,而不是只看输出数值
print(traced.graph)
# 实测输出里只有: %8 = prim::Constant[value={2}]() ; %9 = aten::mul(%x, %8) ; return (%9)
# 完全没有分支结构,不管输入什么,永远是 mul-by-2

print(scripted.graph)
# 实测输出里有: %6 = aten::gt(...) ; %25 = prim::If(%8)
#   block0(): aten::mul(%x.1, 2) -> 对应 if 分支
#   block1(): aten::mul(%x.1, -1) -> 对应 else 分支
# 真正保留了两条路径
```

**面试怎么问 + 追问链:**
- **Q:** "`torch.jit.trace` 和 `torch.jit.script` 有什么区别,你会怎么选?"—— 期望讲出"录像 vs 编译"这个核心比喻,以及"trace 遇到依赖输入数值的控制流会出问题"。
- **追问 1(杀伤力强):** "给我一个具体例子,trace 会在什么场景下悄悄给出错误结果?"—— 期望能现场构造一个类似上面的分支模型,而不是只会背"trace 有控制流问题"这句话。
- **追问 2:** "怎么发现自己的模型 trace 错了?"—— 期望答"用样例输入之外、会触发不同分支的输入去验证,而不能只看用来 trace 的那个输入跑出来对不对"——因为用来验证的输入如果就是当初 trace 用的那个,结果必然"看起来是对的"。
- **追问 3(深挖):** "如果模型大部分是纯计算、只有一小段有控制流,有没有折中方案?"—— 期望知道两者可以混用(比如给某个子模块或方法单独打 `@torch.jit.script_if_tracing`/`@torch.jit.export` 之类的标记,或者把有控制流的部分单独 script、其余部分照常 trace/eager),不是非此即彼的选择。

**常见坑:** 只用"能不能成功导出、导出后跑一次样例输入结果对不对"来验证 trace 的正确性——这恰好是 trace 最会骗人的地方,用来验证的输入本身就是走的被记录的那条路径,当然"看起来是对的"。`torch.jit.trace` 执行时如果代码里有依赖 tensor 值的 Python 控制流,PyTorch 会打印 `TracerWarning`(本节实测原文:"Converting a tensor to a Python boolean might cause the trace to be incorrect"),这是一个必须重视的信号,不是可以忽略的噪音警告。

---

## 5. 显存泄漏常见成因 —— 忘记 `.item()` 到底在什么条件下才真的泄漏

**是什么(这是一个代码模式,不是具体 API):**
```python
# 反模式:把还带着计算图的 tensor(而不是 .item() 取出的纯数值)存进一个长期存活的容器
epoch_losses = []
for x, y in dataloader:
    loss = criterion(model(x), y)
    epoch_losses.append(loss)          # 危险:存的是 tensor,不是数值
```

**一句话:** 常见的说法是"忘记 `.item()` 会导致整个计算图和它依赖的中间激活值都被这个 list 引用着,永远不会被回收"——**这句话只在一个更精确的条件下成立:这个 tensor 背后的计算图还没有被 `backward()` 消费/释放过**。如果 `backward()` 已经以默认参数(`retain_graph=False`)正常跑过一次,反向传播过程中会主动清空为它保存的中间值,这时候哪怕 list 里还留着这个 tensor 的引用,也**不会**持续占用显存——这是一个很多资料会讲得过于简化(甚至讲错)的细节,下面用四组实验把"真泄漏"和"不泄漏"的边界现场做出来。

**底层机制/为什么这样设计:** 每个 autograd 追踪产生的 tensor,它的 `grad_fn` 会用 `save_for_backward` 保存反向传播需要用到的中间值(呼应 [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 7 节的版本计数器机制,是同一层设计的另一面)。`backward()` 执行时,引擎按拓扑逆序遍历计算图,每用完一个节点保存的值就会**主动释放**它——`retain_graph=False` 这个默认值的含义正是"这张图只打算用一次,用完主动清空,不等 Python 的垃圾回收"。这个清空动作不看"还有没有人拿着 `grad_fn` 的引用",所以哪怕你把已经 `backward()` 过的 loss tensor 存进一个 list,图早就被清空了,list 里只剩一个"空壳"(轻量的图节点结构 + 一个标量数值,不含中间激活值)。**真正会导致显存持续增长的,是"某个还没被 `backward()` 释放过的、带梯度的 tensor,被一个生命周期很长的容器一直引用着"**——最常见的两种写法:①验证/统计循环里,只是想"记个总数"就写 `total_loss = total_loss + loss`(而不是 `.item()`),但从没在这个累加结果上调用过 `backward()`;②验证循环忘了 `torch.no_grad()`,又忘了 `.item()`,每个 batch 的 loss 全部带着计算图堆进一个 list。

**AI 研究场景:** 长跑训练任务"跑到第 N 个 epoch 突然 OOM"是经典排查场景,思路通常是打点看 `memory_allocated()` 是不是在稳定爬升;验证循环忘记 `torch.no_grad()` 又忘记 `.item()`,导致悄悄攒住一整个 epoch 的计算图,是这类 bug 里最常见的一种,因为"验证阶段不需要梯度"这件事很容易在写代码时被忽略。

**可运行例子:**
```python
import torch

def mb(x): return x / 1024**2
device = 'cuda'
model = torch.nn.Sequential(torch.nn.Linear(2000, 2000), torch.nn.ReLU(), torch.nn.Linear(2000, 2000)).to(device)

# --- 场景1: 每步都 backward()(默认 retain_graph=False),然后把裸 loss tensor 存进 list ---
torch.cuda.empty_cache()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
losses, readings = [], []
for i in range(30):
    x = torch.randn(256, 2000, device=device)
    y = model(x).sum()
    opt.zero_grad(set_to_none=True)
    y.backward()                 # 图已经被消费/释放
    opt.step()
    losses.append(y)              # 没写 .item(),但……
    readings.append(mb(torch.cuda.memory_allocated()))
# 实测: readings 从头到尾稳定在 83.2MB,30步增长 0.0MB —— 不泄漏!backward() 已经释放了中间激活值
assert max(readings) - min(readings) < 1.0

# --- 场景2: 从不调用 backward(),纯前向后把裸 tensor 存进 list(典型:验证循环忘记 no_grad+item) ---
torch.cuda.empty_cache()
losses2, readings2 = [], []
for i in range(30):
    x = torch.randn(256, 2000, device=device)
    y = model(x).sum()
    losses2.append(y)             # 从未 backward 过这个图,一直占着中间激活值
    readings2.append(mb(torch.cuda.memory_allocated()))
# 实测: readings2 从 85.2MB 涨到 200.7MB,30步增长 115.5MB —— 真泄漏,而且是线性增长
assert readings2[-1] - readings2[0] > 100

# --- 场景3: 经典 "epoch累加器" 写法,total_loss += loss 忘记 .item(),同样从未 backward ---
torch.cuda.empty_cache()
total_loss, readings3 = 0.0, []
for i in range(30):
    x = torch.randn(256, 2000, device=device)
    y = model(x).sum()
    total_loss = total_loss + y   # total_loss 变成一个 tensor,累积了30步的整条计算图
    readings3.append(mb(torch.cuda.memory_allocated()))
# 实测: 205.0MB -> 320.7MB,增长 115.6MB,和场景2几乎同样的增长速率(~3.85MB/step)
assert readings3[-1] - readings3[0] > 100
assert isinstance(total_loss, torch.Tensor) and total_loss.requires_grad

# --- 场景4: 隔离验证机制 —— 场景1的写法,但故意 retain_graph=True,不让 backward 释放缓存 ---
torch.cuda.empty_cache()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
losses4, readings4 = [], []
for i in range(15):
    x = torch.randn(256, 2000, device=device)
    y = model(x).sum()
    opt.zero_grad(set_to_none=True)
    y.backward(retain_graph=True)   # 关键区别:图不会被自动释放
    opt.step()
    losses4.append(y)
    readings4.append(mb(torch.cuda.memory_allocated()))
# 实测: 85.2MB -> 140.8MB,15步增长 55.6MB(~3.97MB/step,和场景2/3速率一致)
# 证明:同一段"append裸tensor"代码,retain_graph=False时不泄漏,retain_graph=True时真的泄漏
# 泄漏与否的开关是"backward有没有释放保存的中间值",不是"有没有人拿着这个tensor的引用"
assert readings4[-1] - readings4[0] > 40
```

**面试怎么问 + 追问链:**
- **Q:** "你写训练循环时怎么记录每一步的 loss?会不会有内存问题?"—— 基础期望是"用 `.item()` 取纯数值再存,不要存 tensor 本身"。
- **追问 1(核心,能筛出真懂假懂):** "如果我已经对这个 loss 调用了 `backward()`,还留着一个引用在 list 里,会不会内存泄漏?"—— 只会背"不要保留 tensor 引用"这条规则的候选人会答"会",但实测结果是**默认参数下不会**——这道题专门用来分辨"背过结论"还是"理解机制"。
- **追问 2(深挖):** "那什么情况下,即使调用了 `backward()`,这个 bug 依然会发生?"—— 期望答出 `retain_graph=True` 的场景(比如一个 loss 需要 backward 两次、GAN 训练里生成器和判别器共享一段计算图、RNN 手动展开),这正是本节场景4现场复现的情况。
- **追问 3(工程开放题):** "怎么在训练脚本里自动发现这类问题,而不是等 OOM 了才排查?"—— 没有标准答案,考察能不能想到"定期打点 `memory_allocated()`,如果它随 step 数近似线性增长就该报警"这类系统性思路,呼应下面第7节。

**常见坑:** 网上很多资料把这条规则简化成"永远不要保留带 `grad_fn` 的 tensor 引用"——这个说法过于保守,真正的判断标准是"这个 tensor 背后的计算图有没有被 `backward()` 释放过",不是"有没有人拿着它的引用"。反过来也要小心:一旦训练代码里用了 `retain_graph=True`(有合理理由,比如共享子图的多头 loss),那"存了忘记 `.item()` 的 tensor"这条经典 bug 就会真的发生,因为 `backward()` 不会主动清空图——写这类代码时要格外小心日志记录的地方有没有不小心存了裸 tensor。

---

## 6. `torch.cuda.empty_cache()` 的作用与常见误解

**是什么:**
```python
torch.cuda.empty_cache()   # 把缓存分配器里"当前没有 tensor 在用、但还没归还 driver"的部分,真正还给 CUDA driver
```

**一句话:** 很多人以为这个函数能让程序"腾出更多可用显存",但结合第1节的机制看,它做的事情精确来说是"把 `reserved` 池里超出 `allocated` 的那部分还给 driver"——这**不会**让当前进程能用的显存总量变多(缓存池从来不是"能用显存"的上限),只在"需要把这部分显存暂时让给同一块 GPU 上的另一个进程"这种场景下真的有意义。

**底层机制/为什么这样设计:** 直接承接第1节——缓存分配器平时故意不把释放的显存还给 driver,就是为了避免反复走慢速的 `cudaMalloc`/`cudaFree`(第1节实测:走一次真实 driver 分配比命中缓存慢约 17 倍)。`empty_cache()` 存在的意义,是在**明确知道**"接下来一段时间不需要这么大的缓存池,且这块显存对别的进程有用"时,主动把这层缓存优化关掉一次——这是一个"用未来的分配速度,换取现在让别的进程能跑起来"的权衡,不是一个无成本的"释放内存"按钮。

**可运行例子:**
```python
import torch

def mb(x): return x / 1024**2

torch.cuda.empty_cache()
free0, _ = torch.cuda.mem_get_info()     # driver 视角的"整卡空闲显存"

x = torch.randn(100_000_000, device='cuda')   # ~381.5MB
free1, _ = torch.cuda.mem_get_info()

del x
free2, _ = torch.cuda.mem_get_info()
print(f"del后(未empty_cache): allocated={mb(torch.cuda.memory_allocated()):.1f}MB "
      f"reserved={mb(torch.cuda.memory_reserved()):.1f}MB driver_free={mb(free2):.1f}MB")
# 实测: allocated=0.0MB reserved=382.0MB driver_free=14855.0MB (和分配后一样,没有变化)
assert free2 == free1        # 关键: 只 del 不 empty_cache,driver 层面看到的空闲显存完全不变

torch.cuda.empty_cache()
free3, _ = torch.cuda.mem_get_info()
print(f"empty_cache后: allocated={mb(torch.cuda.memory_allocated()):.1f}MB "
      f"reserved={mb(torch.cuda.memory_reserved()):.1f}MB driver_free={mb(free3):.1f}MB")
# 实测: allocated=0.0MB reserved=0.0MB driver_free=15237.0MB
assert torch.cuda.memory_allocated() == 0          # allocated 本来就是0,empty_cache 前后都一样 —— 它不影响"正在用"的显存
assert torch.cuda.memory_reserved() == 0             # reserved 真的降为0了
assert free3 > free2                                  # driver 层面的空闲显存增加了,意味着"别的进程现在能用这块了"
print(f"empty_cache 让 driver 层面多出 {mb(free3-free2):.1f}MB 空闲显存(约等于之前缓存的 382MB)")
```

**AI 研究场景:** Jupyter notebook 连续跑多个实验、同一张卡上同时开着训练脚本和一个临时的推理/debug 脚本、Kaggle/Colab 这类共享 GPU 环境——这些场景下,当前进程手上有一大块"缓存着没用"的显存,另一个进程却因为显存不够起不来,这时候在当前进程(确保确实已经 `del` 掉大 tensor 之后)调用一次 `empty_cache()`,能把这部分显存真正腾给别的进程用。但如果只有自己一个训练任务在跑,循环里调用 `empty_cache()` 不但没用,还会让下一次分配重新触发慢速的 `cudaMalloc`(缓存被清空了),是常见的"自己给自己找性能麻烦"反模式。

**面试怎么问 + 追问链:**
- **Q:** "`empty_cache()` 能解决 OOM 吗?"—— 期望答"通常不能",如果峰值显存需求本来就超过显卡容量,清缓存不会变出更多显存。
- **追问 1:** "那它到底有什么用?"—— 期望讲出"多进程共享 GPU,把暂时不用的显存让给别的进程"这个场景,而不是含糊地说"清理内存"。
- **追问 2(深挖,反常识):** "'reserved 但没在用'的显存,为什么不能被同一进程接下来的分配直接复用,非要 `empty_cache()`?"—— 期望答"通常是可以复用的"(这才是缓存分配器存在的意义),只有当新请求的大小/连续性和已缓存的 block 不匹配(碎片化)时才复用不了——这时候 `empty_cache()` 反而可能帮上忙(把碎片化的小块都还给 driver,下次整块重新申请),但也可能把好不容易攒的缓存全清空导致后面变慢,是一个需要权衡的操作,不是"越干净越好"。

**常见坑:** 在训练循环内部"以防万一"每隔几步调用一次 `empty_cache()`——这是本节最该记住的反模式,不会帮你避免 OOM,只会强制后续分配重新走一遍慢速路径(第1节实测约慢 17 倍),白白拖慢训练。

**★ 复验时发现的一个真实现象(和"allocated 的零点"有关):** 如果在同一个进程里,`empty_cache()` 之前已经调用过 cuBLAS(如 `matmul`/`@`)或 cuDNN(如卷积)算子,`memory_allocated()` 在 `del` 掉所有 tensor、且 `empty_cache()` 之后,并不会精确回到 0——本机实测 cuBLAS 会为自己的 handle 保留一块约 8MB 的持久 workspace(卷积/cuDNN 场景实测这次没有触发同样的现象,但同一类机制在不同 cuDNN 版本/算法上是否保留workspace不能一概而论),这块内存不属于任何 Python tensor 对象,不会被 `del` 或垃圾回收触发释放,`empty_cache()` 也不会碰它——因为它根本不是"缓存池里没用的部分",而是 cuBLAS 库自己长期持有、生命周期和进程绑定的资源。实际影响:如果你的脚本已经做过任何矩阵乘法,`torch.cuda.memory_allocated()==0` 就不再是一个可靠的"确实什么都没有"的判据,想验证"是否真的什么都没有"更可靠的做法是记录一个训练/实验开始前的基线读数,后续都看相对这个基线的增量,而不是死认绝对值 0。

---

## 7. 显存 profiling 基础方法 —— 定位显存到底涨在哪一步

**是什么:**
```python
torch.cuda.memory_allocated() / torch.cuda.memory_reserved()           # 当前时刻的快照
torch.cuda.max_memory_allocated() / torch.cuda.max_memory_reserved()    # 自上次 reset 以来出现过的历史峰值
torch.cuda.reset_peak_memory_stats()                                     # 把"历史峰值"清零,重新开始统计一段代码的峰值
```

**一句话:** 单看某一时刻的 `memory_allocated()` 只能告诉你"现在用了多少",真正决定"这个训练任务需要多大显存的卡才跑得起来"的是**峰值**——`max_memory_allocated()` 能直接拿到从某个起点到现在出现过的最大值,不需要在代码里到处插 `print` 人工找最大值;配合在关键节点(创建模型后、forward 前后、backward 前后、optimizer.step 前后)打点读数,可以把"显存到底花在参数、激活值,还是优化器状态上"拆解清楚。

**底层机制/为什么这样设计:** 训练一步里显存占用不是单调递增的——`backward()` 一边在为每个参数分配新的梯度 tensor(和参数同样大小),一边在消费/释放前向过程中保存的激活值(第5节讲过这个释放机制),两者同时发生,最终净效果取决于谁的量级更大、以及某个瞬间"还没释放完的激活值"和"已经分配出来的梯度"是否短暂共存出现了局部峰值。只在 step 开始和结束各测一次,会完全错过这类"中间瞬间"的峰值——这就是 `max_memory_allocated()` 这个"历史最大值"接口存在的意义:它不要求你精确知道峰值出现在哪一行,只要求你在关心的区间前后调用 `reset_peak_memory_stats()`/`max_memory_allocated()`。

**可运行例子:**

分阶段打点,拆解"参数 vs 输入 vs 前向激活值 vs 梯度 vs 优化器状态"分别占多少:
```python
import torch

def mb(x): return x / 1024**2
device = 'cuda'
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

layers = []
dims = [4096]*6
for i in range(len(dims)-1):
    layers += [torch.nn.Linear(dims[i], dims[i+1]), torch.nn.ReLU()]
model = torch.nn.Sequential(*layers).to(device)

step1 = torch.cuda.memory_allocated()
n_params = sum(p.numel() for p in model.parameters())
print(f"1. 建好模型: allocated={mb(step1):.1f}MB (理论参数量 {n_params:,} * 4B = {mb(n_params*4):.1f}MB)")
# 实测: allocated=320.1MB,和理论参数大小 320.1MB 精确吻合

x = torch.randn(2048, dims[0], device=device)
step2 = torch.cuda.memory_allocated()
print(f"2. 建好输入batch: allocated={mb(step2):.1f}MB (delta={mb(step2-step1):.1f}MB)")
# 实测: allocated=352.1MB (delta=32.0MB,就是这个 batch 本身的大小)

out = model(x)
step3 = torch.cuda.memory_allocated()
print(f"3. forward完成: allocated={mb(step3):.1f}MB (delta={mb(step3-step2):.1f}MB, 这就是前向激活值的净开销)")
# 实测: allocated=521.2MB (delta=169.1MB) —— 注意不是 10层*32MB=320MB,因为很多中间张量
# 在被下一层消费后立刻被缓存分配器复用了(呼应第1节:分配器复用不只发生在你手动 del 的时候,
# 前向过程内部也在持续发生),真正"必须留到 backward 才能释放"的只有一部分

loss = out.sum()
step4 = torch.cuda.memory_allocated()
print(f"4. 算出scalar loss: allocated={mb(step4):.1f}MB (delta={mb(step4-step3):.1f}MB,标量,几乎不增加)")
# 实测: allocated=521.2MB (delta=0.0MB)

loss.backward()
step5 = torch.cuda.memory_allocated()
print(f"5. backward完成: allocated={mb(step5):.1f}MB (delta={mb(step5-step4):.1f}MB)")
# 实测: allocated=721.4MB (delta=200.2MB) —— 新分配了和参数同样大小的梯度张量,
# 同时前向激活值在backward过程中被逐步释放,净增量是两者相抵后的结果

peak = torch.cuda.max_memory_allocated()
print(f"6. 目前为止的峰值: {mb(peak):.1f}MB")
# 实测: 797.4MB —— 比 backward 结束后的 721.4MB 更高,说明峰值出现在某个中间瞬间
# (前向激活值还没释放完 + 梯度已经在分配)而不是backward算完之后那一刻

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.step()
step7 = torch.cuda.memory_allocated()
print(f"7. optimizer.step()后: allocated={mb(step7):.1f}MB (delta={mb(step7-step5):.1f}MB)")
# 实测: allocated=1361.6MB (delta=640.2MB) —— Adam 第一次 step 要为每个参数分配
# exp_avg + exp_avg_sq 两个状态,理论上是 2倍参数大小 = 640.2MB,精确吻合

final_peak = torch.cuda.max_memory_allocated()
print(f"8. 最终峰值(含优化器状态): {mb(final_peak):.1f}MB")
# 实测: 1681.6MB —— 这才是这个训练配置真正需要多大显存卡的答案,不是任何一个"稳定态"读数
```

**AI 研究场景:** 决定 batch size 上限、判断该上 gradient checkpointing(如果峰值主要来自前向激活值)还是需要模型并行/换更大显存的卡(如果峰值主要来自参数+优化器状态),都需要先把"显存花在哪"定量拆解清楚,而不是靠反复 OOM 报错去试探——尤其是提交到共享集群的大训练任务,一次因为显存预估错误的 OOM 可能浪费数小时排队等到的机器时间。

**面试怎么问 + 追问链:**
- **Q:** "训练脚本 OOM 了,你怎么排查是模型参数太大还是中间激活值太大?"—— 期望讲出"在 forward 前后、backward 前后分别打点 `memory_allocated()`,看增量主要发生在哪一步"这个方法论,而不是只会说"调小 batch size 试试"。
- **追问 1(容易漏答):** "为什么 `backward()` 之后 `allocated` 反而涨了,不是应该在释放前向的激活值吗?"—— 期望答出"backward 确实在释放激活值,但同时也在分配新的梯度张量,两者同时发生,净变化取决于谁的量级更大",本节例子里两者相抵后依然净增 200.2MB,说明梯度分配这部分更大。
- **追问 2(深挖):** "只在 `backward()` 前后各打点一次够吗?"—— 期望答"不一定",峰值可能出现在两次打点之间的某个瞬间(本节例子峰值 797.4MB 比 backward 结束时的 721.4MB 更高),这正是 `max_memory_allocated()` 这个接口存在的原因——它不要求你猜到峰值具体在哪一行。
- **追问 3(工程开放题):** "优化器状态占的显存能不能提前预知,不用真跑一遍?"—— 期望能推出"Adam/AdamW 是参数量的 2 倍(fp32),SGD+momentum 是 1 倍,不带 momentum 的 SGD 是 0 倍",这个话题会在 [06-optimizer-internals.md](06-optimizer-internals.md) 详细展开。

**常见坑:** 只看训练跑起来之后某一次 `print` 的 `memory_allocated()` 就下结论"这个模型只需要 XXX 显存"——真正决定需要多大显卡的是峰值,而峰值经常出现在一个容易被忽略的瞬间(前向激活值还没释放完、新的梯度张量已经在分配的交界点),不是训练稳定跑起来之后的某个"平稳值"。另外要注意 `max_memory_allocated()` 是"自上次 reset 以来"的累计峰值,不重置就一直测,会把好几个不同阶段(比如第一次 optimizer.step() 分配状态的那次峰值)混在一起,想单独测某一段代码的峰值,一定要先 `reset_peak_memory_stats()`。

---

## 8. `num_workers`/`pin_memory` 对 `DataLoader` 吞吐的影响

**是什么:**
```python
DataLoader(dataset, batch_size=B, num_workers=0, pin_memory=False, persistent_workers=False)
```

**一句话:** `num_workers>0` 让"CPU 准备下一个 batch"这件事放到后台子进程里做,和"GPU 算当前 batch"在时间上重叠,而不是主进程算完一个 batch 才回头处理下一批的 CPU 预处理;`pin_memory=True` 让 worker 进程产出的 batch 提前放进页锁定内存,配合训练循环里 `.to(device, non_blocking=True)`,让"CPU→GPU"这一步真正的搬运也能异步/更快完成——这是 [01-tensor-memory-model.md](01-tensor-memory-model.md) 第 11 节 `pin_memory()`/`non_blocking` 机制在 `DataLoader` 整体吞吐视角下的应用。

**底层机制/为什么这样设计:** `num_workers=0` 时,`__getitem__` 里的所有 CPU 工作(读文件、decode、数据增强)都发生在主进程,且发生在"当前这个 batch 被主进程请求的那一刻"——如果这部分 CPU 工作耗时比 GPU 算一个 batch 还长,GPU 会有大段时间在干等下一个 batch 准备好。`num_workers=N>0` 时,PyTorch 会启动 N 个独立子进程,每个都在后台不停地准备接下来的若干个 batch,主进程只需要从一个进程间队列里取一个"已经准备好"的 batch——只要 CPU 预处理总吞吐能跟上 GPU 消耗速度,GPU 几乎不用等待。`pin_memory` 部分的机制 01 篇已经讲过(操作系统的 pageable 内存 GPU 不能直接 DMA 读取,必须先搬进锁页内存);这里的新增点是:如果这次锁页内存搬运也放进 `DataLoader` 的后台流程提前做好,主训练循环拿到的 batch 已经是 pinned 的,才能真正用上 `non_blocking=True` 的异步传输——本机隔离测量显示,这一步单独就有约 2.8 倍的差异(见下)。

**AI 研究场景:** 大规模图像/视频训练里,数据预处理(JPEG decode、resize、数据增强)是明显的 CPU 密集型工作,如果 `num_workers=0`,即使模型和 GPU 本身很快,`nvidia-smi` 的利用率也会经常掉到很低——这是"GPU 利用率低但训练还在跑"最常见的病因之一,诊断思路就是本节这种对比实验:固定模型和 batch size,只改 `num_workers`,看总耗时变化有多大。

**可运行例子:**

构造一个有真实 CPU 开销的数据集(不是用 `sleep` 模拟,而是真的做几次 numpy 运算,更贴近真实的图像预处理开销),对比不同 `num_workers`/`pin_memory` 组合跑 20 个 batch 的总耗时(Windows 下 `DataLoader` 多进程用 `spawn`,必须放在 `if __name__ == "__main__":` 里跑):
```python
import time, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class SlowDataset(Dataset):
    def __init__(self, n=2000, feat=3*224*224):
        self.n, self.feat = n, feat
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        x = rng.standard_normal(self.feat, dtype=np.float32)
        for _ in range(3):                     # 模拟真实的CPU端预处理(标准化等)
            x = (x - x.mean()) / (x.std() + 1e-5)
        return torch.from_numpy(x), idx % 10

def run(num_workers, pin_memory, n_batches=20, batch_size=64):
    dl = DataLoader(SlowDataset(), batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    it = iter(dl)
    for _ in range(2):                          # warmup: worker 进程启动开销不计入计时
        xb, _ = next(it); xb = xb.to('cuda', non_blocking=pin_memory)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        xb, _ = next(it)
        xb = xb.to('cuda', non_blocking=pin_memory)
        y = (xb * 2).sum()                       # 一点GPU端计算,让"重叠"有意义
    torch.cuda.synchronize()
    return time.perf_counter() - t0

if __name__ == "__main__":
    for nw in [0, 2, 4]:
        for pm in [False, True]:
            dt = run(nw, pm)
            print(f"num_workers={nw} pin_memory={pm} -> {dt/20*1000:.1f} ms/batch")

# 本机实测(20个batch的均摊耗时,重跑两次,数值有正常波动但方向一致):
#   第一次: nw=0,pm=False: 297.93ms | nw=0,pm=True: 219.73ms
#           nw=2,pm=False: 124.91ms | nw=2,pm=True:  89.23ms
#           nw=4,pm=False:  68.95ms | nw=4,pm=True:  63.07ms   (nw:0->4 提速 4.72x)
#   第二次: nw=0,pm=False: 208.32ms | nw=0,pm=True: 201.27ms
#           nw=2,pm=False: 103.39ms | nw=2,pm=True: 102.85ms
#           nw=4,pm=False:  62.28ms | nw=4,pm=True:  63.78ms   (nw:0->4 提速 3.35x)
# 结论: num_workers 的影响巨大且非常稳定(两次重跑都是 3~5倍);pin_memory 在"整条DataLoader
# 循环"里的增量效果这次测量并不稳定(有时明显有时几乎没有)——这是诚实的观察,不是想当然地断言
# "pin_memory=True 一定让 DataLoader 循环快很多",下面单独把 H2D 拷贝这一步隔离出来看
```

单独隔离"batch → GPU"这一步拷贝(排除掉多进程通信噪声,只测 `.to(device)` 本身,40个batch取均值):
```python
# 固定 num_workers=4,只改 pin_memory,只计时 xb.to('cuda', non_blocking=pin_memory) 这一行,
# 每次都手动 torch.cuda.synchronize() 确保测到真正传输完成的时间
# 本机实测:
#   pin_memory=False: xb.is_pinned()=False, 单次拷贝均值 43.474 ms/batch
#   pin_memory=True:  xb.is_pinned()=True,  单次拷贝均值 15.507 ms/batch
# 拷贝这一步本身: 2.80x 的差距,清晰、稳定,符合 01篇pin_memory机制的预期
```

**面试怎么问 + 追问链:**
- **Q:** "`DataLoader` 读数据很慢,可能是什么原因,怎么排查?"—— 期望讲出"CPU 预处理跟不上 GPU 消耗速度"这个核心矛盾,以及用 `nvidia-smi` 利用率 + 改 `num_workers` 对比总耗时的排查思路。
- **追问 1:** "`num_workers` 是不是越大越好?"—— 期望答"不是",受 CPU 物理核心数限制,超过之后进程调度/进程间通信开销会抵消并行收益,而且每个 worker 都要占用额外内存(尤其 Windows/macOS 的 `spawn` 方式下,每个子进程要重新 import 主模块)。
- **追问 2(深挖):** "`pin_memory=True` 是不是没有代价的纯收益?"—— 期望呼应 01 篇:pinned 内存本身是操作系统里稀缺、不可换出的资源,`pin_memory()` 这一步本身就是一次真实拷贝(pageable→pinned),配合过多 worker 或过大 batch,可能把宿主机的锁页内存配额用满,不是"设了就是纯赚"的开关。
- **追问 3(工程场景):** "如果 GPU 计算本身就很慢(比如一个很大的模型),`num_workers`/`pin_memory` 还有意义吗?"—— 期望答"意义变小",这两个优化解决的是"CPU 预处理成为瓶颈"的问题,如果 GPU 计算时间远大于 CPU 预处理时间,数据加载早就被计算完全掩盖掉了,优化的重点应该转移到 GPU 计算本身(第3节 channels_last、混合精度等)。

**常见坑:** 把 `num_workers` 直接设成一个很大的数(比如 CPU 核心数的好几倍)指望"越大越快"——超过物理核心数后收益会饱和甚至反而变慢,本节实测里 `num_workers` 从 2 到 4 的提升幅度就已经明显小于 0 到 2;另外 `persistent_workers=False`(默认值)会导致每个 epoch 结束后 worker 进程被销毁、下一个 epoch 重新启动,如果 epoch 很短、worker 启动开销占比不小,建议打开 `persistent_workers=True`(本节例子里已经这样做),但要注意它会让 worker 进程常驻,占用的内存不会在 epoch 之间释放。

---

## 小结:这一批 8 个知识点解决的问题

| # | 知识点 | 核心结论(均为本机实测) |
|---|------|-----------------------|
| 1 | CUDA 缓存分配器 | `nvidia-smi`/`memory_reserved()` 是缓存池总量,`memory_allocated()` 是真正在用的量;命中缓存比强制走真实 driver 分配快 **17.0x** |
| 2 | `.item()`/`.cpu()` 同步开销 | GPU 异步执行,同步点会打断流水线;训练循环每步 `.item()` 比每50步一次慢 **2.5~4x**;`.cpu()` 大张量比 `.item()` 标量多付出真实 PCIe 传输时间 |
| 3 | `memory_format` channels_last | 真实物理重排(非零拷贝),`data_ptr()`会变;fp16 卷积在本机 Ampere GPU 上实测提速 **~1.17-1.21x** |
| 4 | `jit.script` vs `jit.trace` | trace 固化实际走过的分支,遇到依赖数值的 `if` 会安静地算错;script 解析源码,正确处理控制流 |
| 5 | 显存泄漏(忘记 `.item()`) | 只有当计算图**未被 `backward()` 释放**时才真泄漏;已默认 `backward()` 过的 tensor 留引用不泄漏(实测差异:30步 0MB 增长 vs 115.5MB 增长) |
| 6 | `empty_cache()` | 只把 `reserved` 还给 driver,不增加"能用"的显存;`allocated` 前后不变,`reserved`/驱动可见空闲显存会变化 |
| 7 | 显存 profiling | 用 `max_memory_allocated()` + 分阶段打点定位峰值来源;本例参数320.1MB+激活值169.1MB+梯度200.2MB+Adam状态640.2MB,峰值1681.6MB |
| 8 | `num_workers`/`pin_memory` | `num_workers` 让CPU预处理和GPU计算重叠,实测提速 **3.3~4.7x**;隔离测量下 `pin_memory` 让H2D拷贝本身提速 **2.80x** |

下一批:[09-distributed-training-basics.md](09-distributed-training-basics.md) —— 分布式训练基础机制(`DistributedDataParallel`、all-reduce 梯度同步)。

---

*更新:2026-07-07*
