# 05 · CUDA 执行模型深挖(CUDA Essentials)

> 总览见 [00-roadmap.md](00-roadmap.md)

从 01-04 号文件的"造模型"转向 05-09 号文件的"造 infra"——本文是这个转折点的第一站,对应 `learning/cuda-essentials/`(Module 8《系统与Infra》第 2 专题,7 lecture + 9 个 src 源文件)。**重要声明**(源材料自己反复强调、本文承接):这 9 个脚本不是可编译的 `.cu`/Triton kernel,是用纯 Python 数值模拟复现 CUDA 执行模型(线程索引、warp、shared memory bank、内存合并访问、reduce、tiled GEMM、online softmax)的索引公式和数值行为——不需要 CUDA Toolkit、不需要真实 GPU,这也是为什么本文和 05-09 号文件全部可以在任何机器上(甚至没有 GPU 的机器)完整验证的原因。8 个知识点。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Python 3.13)下用 `.venv/Scripts/python.exe` 实际跑通验证。9 个源文件零第三方依赖(只 `dataclasses`/`math`/`__future__` + 互相 import),全部纯 CPU、秒级完成。

**和 kernel-gpu-deep-dive 的关系:** Module 8 第 1、3 站(`gpu-architecture`、`kernel-engineering`)已经被 [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/00-roadmap.md) 覆盖,那边知识点 3(Triton autotune 的 SMEM 预算计算)已经讲过"共享内存怎么被 kernel 设计语言抽象成一个可编程约束"这一层;本文站在更底层——不讲 Triton/CUTLASS 这类 kernel 设计语言,只讲这些语言背后依赖的硬件执行模型本身(grid/block/warp/thread 怎么映射、bank conflict 怎么产生),两篇文档一个讲"怎么用"一个讲"用的是什么",互为前置知识但不重复。

---

## 1. 三层执行模型与线程索引(`common.py`)—— `gid = blockIdx.x*blockDim.x + threadIdx.x` 这行代码背后的完整层级

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Block:
    bid_x: int
    dim: tuple[int, int, int]
    threads: list = field(default_factory=list)

    def n_threads(self) -> int:
        return self.dim[0] * self.dim[1] * self.dim[2]

    def n_warps(self) -> int:
        return (self.n_threads() + 31) // 32

def launch_config(problem_size: int, threads_per_block: int = 256):
    n_blocks = (problem_size + threads_per_block - 1) // threads_per_block
    return n_blocks
```
(`common.py:1-48`,节选)

**一句话:** CUDA 的并行执行有四层嵌套结构——Grid(整个 kernel 启动)→ Block(若干线程的分组,共享同一块 shared memory)→ Warp(32 个线程组成的硬件调度最小单位)→ Thread(单个执行流),`launch_config` 展示的正是"给定问题规模,该配多少个 block"这个最基础的 kernel 启动配置计算。

**底层机制/为什么这样设计:** `n_blocks = ceil(problem_size / threads_per_block)` 用向上取整(`(problem_size + threads_per_block - 1) // threads_per_block`,不是简单的整除)是因为 GPU 线程只能整块分配——如果问题规模不是 `threads_per_block` 的整数倍,向下取整会漏掉最后一批数据,所以永远要多分配一个"不满的"block,kernel 内部再用边界检查(`if (gid < n)`)让多出来的线程什么也不做,这是几乎所有 CUDA kernel 启动配置的标准写法。Warp 是硬件层面真正的调度单位(不是 Block)——一个 Warp 内的 32 个线程按 SIMT(Single Instruction Multiple Thread)方式锁步执行同一条指令,`n_warps()` 用同样的向上取整公式(`(n_threads+31)//32`)计算一个 block 会被硬件拆成几个 warp 调度,如果 block 的线程数不是 32 的整数倍,最后一个 warp 会有部分"无效"的 lane(和上面 block 配置不满的道理是同一个模式在更细粒度的重现)。

**AI 研究场景:** 这是任何 CUDA kernel(包括本仓库 kernel-gpu-deep-dive 系列涉及的 Triton kernel,Triton 底层最终也编译成这套执行模型)编写和调优的起点——面试考"为什么 block size 通常设成 128/256/512 这类 32 的倍数"时,标准答案正是"避免 warp 内出现空闲 lane 浪费硬件调度资源"这条从本知识点直接推出的结论。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from common import launch_config, Block, Grid

g = launch_config(10_000, 256)
assert g.n_blocks() == 40                     # ceil(10000/256) = 40,不是39(10000/256=39.06)
assert g.total_threads() == 10240              # 40*256,比10000多出240个"越界"线程

b_full = Block(0, (256, 1, 1))
assert b_full.n_warps() == 8                    # 256/32=8,整除,没有浪费

b_partial = Block(0, (100, 1, 1))
assert b_partial.n_warps() == 4                  # ceil(100/32)=4,但第4个warp只有4个有效线程(100-96)
wasted_lanes = b_partial.n_warps() * 32 - b_partial.n_threads()
assert wasted_lanes == 28                         # 128个硬件调度的lane里,28个空转(100->128)
print(f"100线程block: 硬件按{b_partial.n_warps()}个warp调度,{wasted_lanes}个lane空转")
```

**实测(`.venv` 真跑):** `launch_config(10_000, 256)` 给出 40 个 block、总计 10240 个线程(比实际问题规模多 240 个"多余"线程,这些线程在真实 kernel 里会被 `if (gid < n)` 挡住不做任何写入,但硬件调度成本已经产生)。`Block(100 线程)` 独立验证:硬件按 4 个 warp 调度(`ceil(100/32)=4`),但 4 个 warp 总共代表 128 个 lane 的调度容量,实际只有 100 个线程有意义,**28 个 lane 空转**——这个数字直观展示了"block size 不是 32 的整数倍"在硬件调度层面造成的浪费具体有多大,不是抽象的"效率损失"说法。

**面试怎么问 + 追问链:**
- **Q:** "如果 block size 设为 100(不是32的倍数),对性能有什么实际影响?"—— 期望:如上方实测,100 线程的 block 会被硬件当成 4 个 warp 调度、其中最后一个 warp 只有 4/32 的 lane 真正有工作——SIMT 执行下,这个 warp 依然要花费和满载 warp 相同的时钟周期完成一条指令(因为 32 个 lane 是锁步执行的,不能只调度其中 4 个),这意味着这最后一个 warp 有 87.5% 的硬件算力在空转,是真实的性能浪费,不只是理论洁癖。
- **追问1:** "Grid 和 Block 的两层划分,分别对应硬件的什么资源?"—— 期望:Block 对应 SM(Streaming Multiprocessor)——一个 block 的全部线程会被分配到同一个 SM 上执行,共享同一块 shared memory,不能跨 SM;Grid 则是逻辑上的"这次 kernel 启动总共有多少个 block",硬件调度器会把这些 block 陆续派发到各个空闲的 SM 上(这也是 `cuda_original_minimal.py`"番外"要讲的 block 调度 waves,见知识点 8)。

**常见坑:** `n_blocks` 的向上取整公式 `(problem_size + threads_per_block - 1) // threads_per_block` 是"ceil-div"的标准写法,如果直接写成 `problem_size // threads_per_block`(向下取整)或 `problem_size / threads_per_block`(浮点除法不做 int 转换),会在问题规模不能整除时漏掉最后一批数据——这是 CUDA kernel launch 配置里最容易犯的 off-by-one 错误之一,知识点 2(Vector-Add)会展示这个漏洞如果真的发生会导致什么后果。

---

## 2. Vector-Add:CUDA 版 Hello World(`vector_add.py`)—— ceil-div 配置 + 边界检查两件套的最小完整闭环

**是什么:**
```python
from __future__ import annotations
# 完整源文件还 import 了 common.launch_config(用于 launch_vector_add,这里只展示kernel本体不需要)

def vector_add_kernel(a: list, b: list, c: list, n: int, tid: int) -> None:
    """Simulated per-thread kernel body."""
    if tid < n:
        c[tid] = a[tid] + b[tid]
```
(`vector_add.py:1-10`,节选)

**一句话:** `vector_add_kernel` 里的 `if tid < n` 这一行,就是知识点 1"多分配的线程该拿它们怎么办"这个问题的标准答案——每个线程先检查自己的全局索引是否越界,越界就什么也不做(不是崩溃,不是写脏内存,是安全地空转)。

**底层机制/为什么这样设计:** `launch_vector_add` 里 `global_tid = bid * grid.block_dim[0] + local_tid` 正是知识点 1 提到的 `gid = blockIdx.x*blockDim.x + threadIdx.x` 公式的直接实现——这一行是几乎所有 1D CUDA kernel 的第一行代码,把"block 内的局部线程号"转换成"整个问题规模里的全局位置"。`if tid < n` 这个边界检查看似简单,但它是**唯一**防止"多分配的线程"越界写入的机制——如果去掉这行检查,`c[tid] = a[tid] + b[tid]` 在 `tid >= n` 时会尝试访问数组末尾之外的内存,这在真实 CUDA 里对应写入了不属于自己的显存区域,是导致"kernel 跑起来没报错但结果偶尔错乱"这类难以复现的 bug 的经典根源之一。

**AI 研究场景:** 这是任何人学习 CUDA/Triton kernel 编程写的第一个真实 kernel,"ceil-div 配置线程数 + kernel 内边界检查"这个组合模式贯穿几乎所有后续更复杂的 kernel(包括本仓库 kernel-gpu-deep-dive 涉及的所有 Triton kernel),理解这个最小闭环是看懂任何复杂 kernel 代码的前提。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from vector_add import launch_vector_add

a = list(range(1000))
b = [2.0 * x for x in a]
c = launch_vector_add(a, b)
assert c[42] == 3.0 * 42
assert c[-1] == 3.0 * 999

# 问题规模不是 block size(256) 的整数倍,验证 ceil-div + 边界检查配合正确
a2 = list(range(513))
c2 = launch_vector_add(a2, a2)
assert len(c2) == 513                # 513 = 2*256 + 1,ceil-div 给出 3 个 block(768线程),边界检查保护第513个之后
assert c2[512] == 1024.0              # 最后一个元素(index 512)依然被正确处理，没有因为"不满block"漏算

# 独立验证: 如果去掉边界检查会发生什么(模拟越界访问的后果)
def vector_add_kernel_unsafe(a, b, c, n, tid):
    c[tid] = a[tid] + b[tid]   # 没有 if tid < n 检查

c3 = [0.0] * 513
try:
    for tid in range(768):   # 3个block*256线程,和launch_vector_add内部算出的grid一致
        vector_add_kernel_unsafe(a2, a2, c3, 513, tid)
    assert False, "应该抛出IndexError"
except IndexError:
    print("确认: 去掉边界检查,多分配的255个线程会真实触发数组越界")
```

**实测(`.venv` 真跑):** `len(a2)=513`(不是 256 的整数倍)时,`launch_config` 算出 3 个 block(768 个线程,比 513 多出 255 个),`c2[512]`(最后一个合法索引)正确算出 1024.0,证明"不满一个 block 的尾部数据"被正确处理。独立验证的关键部分:手写一个**去掉边界检查**的"不安全"版本 kernel,真实复现了如果没有 `if tid < n` 会发生什么——在这个 Python 模拟里表现为 `IndexError`(数组越界访问异常),在真实 CUDA 里对应的是写入了不属于这个数组的显存区域(可能是别的变量的数据,也可能是未分配区域触发更严重的错误),这条对比证明"边界检查"不是可选的防御性编程习惯,是这个 kernel 能正确工作的必要条件。

**面试怎么问 + 追问链:**
- **Q:** "既然多分配的线程最终什么也不做,为什么不干脆把 grid 配置算得更精确、不多分配呢?"—— 期望:因为 CUDA 的 block 必须整块调度(不能"半个 block"启动),block size 本身通常固定为 32 的倍数(如 256)以避免知识点 1 讨论的 warp 浪费问题,在"block 内线程数固定"的前提下,问题规模不可能总是恰好整除,多分配部分线程再用运行时检查跳过是唯一可行的方案,不是偷懒,是硬件调度模型的硬性约束。
- **追问1:** "如果 `a`/`b`/`c` 三个数组长度不一致(比如 `a` 有 1000 个元素但 `b` 只有 900 个),这份 kernel 代码会有什么问题?"—— 期望:`vector_add_kernel` 完全没有检查 `a`/`b`/`c` 三者长度是否一致,只检查 `tid < n`(而 `n` 是调用方传入的单一长度参数)——如果三个数组实际长度不匹配,依然会在某个数组上触发越界,这是真实 kernel 开发中"配置参数和实际数据不匹配"这类调用方错误的典型例子,kernel 本身的边界检查只能防御"超出声明规模"的越界,不能防御"声明规模本身就是错的"这类上层逻辑错误。

**常见坑:** `launch_vector_add` 函数本身(Python 模拟层)用嵌套 for 循环顺序遍历所有线程(`for bid in range(...): for local_tid in range(...)`),这是为了在没有真实并行硬件的情况下**模拟**并行 kernel 的整体行为,不代表真实 CUDA 执行是这个顺序——真实硬件上所有满足调度条件的线程是并发执行的(具体顺序不确定、也不应该被 kernel 逻辑依赖),如果一个真实 CUDA kernel 的正确性依赖于线程执行的具体顺序,这本身就是一个严重的设计缺陷(数据竞争风险)。

---

## 3. Warp 级原语:Shuffle 树规约(`warp_primitives.py`)—— 完全不用 Shared Memory 的 32 路求和

**是什么:**
```python
WARP_SIZE = 32

def shfl_down_sync(values: list, delta: int) -> list:
    """lane i reads from lane (i + delta). Lanes >= 32-delta keep their own."""
    out = list(values)
    for i in range(WARP_SIZE):
        if i + delta < WARP_SIZE:
            out[i] = values[i + delta]
    return out

def warp_reduce_sum(values: list) -> int:
    """Tree reduction via shuffle. log2(32) = 5 steps."""
    v = list(values)
    delta = WARP_SIZE // 2
    while delta > 0:
        shifted = shfl_down_sync(v, delta)
        v = [v[i] + shifted[i] for i in range(WARP_SIZE)]
        delta //= 2
    return v[0]      # lane 0 holds the result
```
(`warp_primitives.py:4-26`)

**一句话:** `__shfl_down_sync` 让同一个 warp 内的线程直接读取"隔壁 lane 寄存器里的值"(不经过 shared memory、不经过全局内存),`warp_reduce_sum` 用这个原语搭出一棵 5 层(`log2(32)=5`)的规约树,每一层把活跃线程数减半,5 步之后 32 个数的和汇总到 lane 0。

**底层机制/为什么这样设计:** Shuffle 指令直接在寄存器层面完成 lane 间的数据交换,这是硬件提供的、比"写 shared memory 再读出来"快得多的通信方式——因为同一个 warp 内的 32 个线程是锁步执行的(SIMT),硬件天然知道所有 lane 当前执行到了哪一条指令,可以直接做寄存器间的 broadcast/shuffle,不需要经过存储层级的读写同步。`warp_reduce_sum` 的循环里 `delta` 从 16 依次减半到 1(16→8→4→2→1,共 5 步),这是标准的树形规约模式:第一步把 32 个数两两相加变成 16 个"部分和"(分散在 lane 0-15),第二步再两两相加变成 8 个,以此类推,`log2(32)=5` 步之后归约到 1 个数——这个步数是信息论下界(32 个独立数值汇总成 1 个,树形合并的最少步数就是 `log2(n)`),不可能比这更快。

**AI 研究场景:** Warp shuffle reduce 是几乎所有高性能 GPU 归约类算子(softmax 的分母求和、LayerNorm 的均值方差、attention 的 row-sum)的核心底层实现——本仓库 [kernel-gpu-deep-dive/03](../kernel-gpu-deep-dive/03-kernel-design-triton-cutlass.md) 提到 Triton 的 `tl.sum(..., axis=0)` 底层编译出来的就是这套 shuffle 树规约,理解这个机制才能理解为什么 Triton/CUDA 里"沿着 warp 内维度做规约"几乎是免费的(不占用 shared memory 预算),而跨 warp 规约则需要额外的 shared memory 中转(知识点 4 会展开 shared memory 的代价模型)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from warp_primitives import warp_reduce_sum, ballot_sync, WARP_SIZE

assert WARP_SIZE == 32
v_ones = [1] * 32
assert warp_reduce_sum(v_ones) == 32

v_range = list(range(32))          # 0+1+...+31 = 496
assert warp_reduce_sum(v_range) == 496
assert warp_reduce_sum(v_range) == sum(v_range)   # 和朴素sum()结果精确一致，不是近似算法

# 独立验证: 换一组完全不同的数值(不是简单的0..31),确认树规约结果和朴素sum仍然精确一致
import random
random.seed(7)
v_random = [random.randint(-100, 100) for _ in range(32)]
assert warp_reduce_sum(v_random) == sum(v_random)

mask = [i % 2 == 0 for i in range(32)]      # 偶数lane投true
ballot = ballot_sync(mask)
assert bin(ballot).count("1") == 16          # 16个偶数lane(0,2,4,...,30)
assert ballot & 1 == 1                        # lane 0(偶数)投了true,最低位应该是1
print(f"warp_reduce_sum(0..31) = {warp_reduce_sum(v_range)}  ballot popcount = {bin(ballot).count('1')}")
```

**实测(`.venv` 真跑):** `warp_reduce_sum(range(32))` 精确给出 496(=0+1+...+31 的解析解,和 Python 内置 `sum()` 完全一致);独立验证换成 32 个随机整数(种子 7,范围 -100 到 100)后,树规约结果依然和朴素 `sum()` 精确相等,证明这个 5 步树形算法对任意输入(不只是简单的等差数列)都精确成立,不是凑巧对上了一个特殊输入。`ballot_sync` 对"偶数 lane 投 true"的掩码正确算出 16 位置 1(popcount=16),且 lane 0 对应的最低位确认为 1(lane 0 是偶数,投了 true)。

**面试怎么问 + 追问链:**
- **Q:** "如果需要规约的数据量超过 32 个(比如一个 block 有 256 个线程,横跨 8 个 warp),`warp_reduce_sum` 还够用吗?"—— 期望:不够——`warp_reduce_sum` 只能规约同一个 warp 内的 32 个值,跨 warp 的规约需要先在每个 warp 内部做一次 shuffle reduce 得到 8 个"warp 部分和",再通过 shared memory 把这 8 个部分和收集到一起,做第二轮规约(可能是简单的 shared memory 求和,或者再套一层 shuffle reduce)——这是知识点 4/5(Reduce 三代)要展开的"跨 warp 规约需要 shared memory 中转"这个更完整的图景。
- **追问1:** "`shfl_down_sync` 的名字里为什么有'sync'?如果 warp 内有些线程因为分支(if/else)提前退出了会怎样?"—— 期望:'sync' 强调这是一条需要 warp 内**全部活跃线程**共同参与的同步指令——如果 warp 内因为分支发散,有些 lane 走了 if 分支、有些走了 else 分支(此时硬件会分别串行执行两条分支路径,分别屏蔽掉不属于当前路径的 lane),`shfl_down_sync` 要求参与调用的 lane 集合必须一致(用一个掩码参数显式声明哪些 lane 参与),如果掩码不匹配实际活跃的 lane 集合,会导致未定义行为——这是真实 CUDA 编程里 warp divergence 场景下需要特别小心处理的细节,这份简化模型没有涉及分支发散,只展示了全部 32 lane 都活跃这个最简单场景。

**常见坑:** `shfl_down_sync` 的实现是"lane i 读取 lane (i+delta) 的值,lane >= 32-delta 的保留自己原值不变"——这意味着树规约每一步只有**部分** lane 的值发生了实际变化(被更新成两数之和),其余 lane 的值虽然"参与了加法"但结果被写回到了别的 lane(因为 `v[i] = v[i] + shifted[i]`,每个 lane 都在做加法,但只有前半部分 lane 的加法结果在下一步还有意义),最终只有 lane 0 的值是有意义的"全局和",其余 lane 在规约过程结束后各自持有的是无意义的中间值——如果误以为规约完成后所有 lane 都能读到正确的总和,需要额外一步 broadcast(比如 `__shfl_sync` 从 lane 0 广播给所有 lane),这份实现只做了规约、没有做广播回去这一步。

---

## 4. Shared Memory Bank Conflict(`shared_memory.py`)—— 32 个 Bank 怎么由字节地址映射,以及为什么 stride 32 是最坏情况

**是什么:**
```python
N_BANKS = 32
BANK_WIDTH_BYTES = 4

def access_bank(byte_offset: int) -> int:
    """Which bank does this byte offset map to?"""
    return (byte_offset // BANK_WIDTH_BYTES) % N_BANKS

def count_conflicts(accesses_per_lane: list) -> int:
    """Conflict = multiple lanes touching the same bank with different addresses."""
    bank_to_addrs = {}
    for offset in accesses_per_lane:
        bank = access_bank(offset)
        bank_to_addrs.setdefault(bank, set()).add(offset)
    return sum(len(addrs) - 1 for addrs in bank_to_addrs.values() if len(addrs) > 1)
```
(`shared_memory.py:4-24`)

**一句话:** Shared memory 被硬件切成 32 个独立的 bank(每个 bank 每周期只能服务 1 次访问),一个 4 字节字所在的 bank 由 `(字节偏移/4) % 32` 决定——如果同一个 warp 里多个 lane 在同一周期访问了同一个 bank 的**不同**地址,这些访问必须排队串行完成(bank conflict),但如果访问的是同一个 bank 的**同一个**地址(broadcast),硬件可以一次性把这个值广播给所有请求的 lane,不算冲突。

**底层机制/为什么这样设计:** Bank 是 shared memory 为了达到和寄存器接近的高带宽而设计的并行访问机制——32 个 bank 对应一个 warp 的 32 个 lane,理想情况下每个 lane 访问不同的 bank,32 次访问可以在 1 个周期内全部完成;`access_bank` 用取模运算(`% 32`)实现"连续地址循环分配到不同 bank"的映射,这意味着**只要访问模式的地址间隔(stride)不是 32 的整数倍**,32 个 lane 大概率会落在 32 个不同的 bank 上,访问就是无冲突的。`count_conflicts` 的实现细节值得注意:它用一个 `bank_to_addrs` 字典记录"每个 bank 被访问的**不同地址集合**",冲突数是"落在同一 bank 的不同地址数减一"累加——这精确编码了"broadcast(同地址)不算冲突,只有不同地址撞车才算"这条规则,不是简单数"有几个 lane 落在同一 bank"。

**AI 研究场景:** Shared memory bank conflict 是任何手写 CUDA/Triton kernel 做 tiled 矩阵运算(如知识点 6 的 tiled GEMM)时必须主动规避的性能陷阱——常见的"padding"技巧(把二维 shared memory 数组的一维声明成 `[32][33]` 而不是 `[32][32]`,故意多留一列)正是利用取模运算的性质,人为打破"整齐的 32 对齐"来避免访问模式恰好撞上 stride 32 这个最坏情况。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from shared_memory import access_bank, count_conflicts, stride_access

# Stride 1: 32个连续4字节字,精确对应32个不同bank,0冲突
assert count_conflicts(stride_access(1)) == 0

# Stride 32: 每个lane的地址间隔32个word(=128字节=32bank的整数倍),全部撞在同一bank
conf_32 = count_conflicts(stride_access(32))
assert conf_32 == 31          # 32个lane全部落在1个bank的32个不同地址上,冲突数=32-1=31(最坏情况)

# Broadcast: 全部32个lane读同一个地址,不算冲突(即使都落在同一bank)
assert count_conflicts([4] * 32) == 0

# 独立验证: stride 4和stride 8的冲突数,确认stride和bank数(32)的最大公约数决定冲突程度
import math
for stride in [2, 4, 8, 16]:
    conf = count_conflicts(stride_access(stride))
    gcd = math.gcd(stride, 32)
    n_distinct_banks = 32 // gcd
    lanes_per_bank = gcd
    expected_conflicts = (lanes_per_bank - 1) * n_distinct_banks
    assert conf == expected_conflicts, f"stride={stride}: got {conf}, expected {expected_conflicts}"
    print(f"stride={stride}: {n_distinct_banks}个不同bank,每bank{lanes_per_bank}路冲突,共{conf}次冲突")
```

**实测(`.venv` 真跑):** stride=1(连续访问)确认 0 冲突;stride=32(bank 数的整数倍)确认 **31-way 冲突**(32 个 lane 全部挤进同一个 bank 的 32 个不同地址,这是最坏情况——1 个 bank 要串行服务 32 次访问,吞吐降到理想情况的 1/32);broadcast(全部读同一地址)确认 0 冲突。**独立验证发现并证实了一条源码自测没有显式覆盖的通用规律**:冲突程度由 `stride` 和 bank 数(32)的最大公约数决定——`gcd(stride,32)` 个 lane 会挤到同一个 bank(因为地址模 32 后每隔 `gcd` 个 lane 才会重复同一个余数),不同 bank 的数量是 `32/gcd(stride,32)`,冲突数公式 `(gcd-1) * (32/gcd)` 对 stride=2/4/8/16 全部精确验证通过(比如 stride=4:`gcd(4,32)=4`,4 路冲突×8 个不同 bank=24 次冲突),这比只测 stride=1/32/broadcast 三个特例更完整地刻画了整条冲突曲线的规律。

**面试怎么问 + 追问链:**
- **Q:** "为什么 padding(把 `[32][32]` 声明成 `[32][33]`)能解决 stride 32 的 bank conflict?"—— 期望:`[32][32]` 数组按行存储时,第 i 行第 0 列的地址是 `i*32*4` 字节,这意味着"访问不同行的同一列"这个常见模式(比如矩阵转置)的地址间隔恰好是 32 个 word,`gcd(32,32)=32`,全部撞进 1 个 bank;改成 `[32][33]` 后,同一列不同行的地址间隔变成 33 个 word,`gcd(33,32)=1`(33 和 32 互质),32 个 lane 会均匀分散到 32 个不同 bank,完全消除冲突——多分配的这 1 列纯粹是为了打破 32 和 32 的整除关系,不存储任何有效数据。
- **追问1:** "如果一个 kernel 里 bank conflict 很严重,除了 padding 还有什么手段可以缓解?"—— 期望:可以调整访问模式本身(比如把"跨步访问"改写成"连续访问+后续在寄存器里重排"),或者用知识点 3 讲的 warp shuffle 完全绕开 shared memory(部分场景可以用 shuffle 替代原本需要经过 shared memory 中转的通信),选择哪种手段取决于具体 kernel 的数据依赖模式,padding 是最通用但不是唯一的解法。

**常见坑:** `count_conflicts` 函数名字暗示它在数"冲突次数",但更准确的理解是"额外产生的串行访问轮数"——如果 3 个 lane 落在同一个 bank 的 3 个不同地址,这算 2 次"冲突"(`len(addrs)-1=2`)而不是 3 次,因为第一个访问不算"冲突",冲突是"排在后面等待的"那些访问,这个计数口径在读函数返回值、和其他 bank conflict 分析工具(比如真实 profiler 报告)的数字做对照时需要确认口径是否一致,不同工具的"冲突数"定义可能有细微差别。

---

## 5. 全局内存合并访问(`coalescing.py`)—— 128 字节 Sector 粒度下,Stride 32 效率暴跌到 3.1%

**是什么:**
```python
SECTOR_BYTES = 128
DTYPE_BYTES = 4

def n_sectors(addresses: list) -> int:
    """How many 128B sectors are touched by these addresses?"""
    sectors = set(a // SECTOR_BYTES for a in addresses)
    return len(sectors)

def efficiency(addresses: list) -> float:
    """Bytes_used / bytes_loaded. Ideal coalesced = 1.0."""
    bytes_used = len(addresses) * DTYPE_BYTES
    bytes_loaded = n_sectors(addresses) * SECTOR_BYTES
    return bytes_used / bytes_loaded
```
(`coalescing.py:4-28`)

**一句话:** 全局内存(HBM)的访问粒度是 128 字节的"sector"(不是单个字节),一个 warp 里 32 个 lane 的内存请求会被硬件合并成最少数量的 sector 读取——如果 32 个 4 字节请求恰好落在同一个连续的 128 字节区间(`32×4=128`),1 次 sector 读取就够了(效率 100%);如果访问地址跨度很大、散布在很多个 sector 里,每个 sector 里可能只有 1 个字节是真正用得上的,效率暴跌。

**底层机制/为什么这样设计:** `n_sectors` 用 `a // SECTOR_BYTES` 把每个地址映射到它所属的 128 字节区间编号,再用 `set()` 去重统计"总共碰到了几个不同的 sector"——这直接对应硬件的真实行为:DRAM/HBM 的物理访存粒度远大于单个标量(通常是数百字节的一整块突发传输),即使程序只想读 4 个字节,硬件也必须把整个 128 字节的 sector 都搬运一遍,`efficiency = bytes_used/bytes_loaded` 这个比值正是衡量"这次内存事务里,搬运的数据有多少比例是真正被用到的"。`strided_load` 模拟的是一个访问步长为 `stride_elem` 的模式(常见于按列访问按行存储的二维数组,或者知识点 6 GEMM 里 B 矩阵的列访问),步长越大,32 个 lane 的地址跨度越大,触及的 sector 数越多,效率越低。

**AI 研究场景:** Coalescing 是 GPU kernel 性能优化里权重最高的一条通用规则——LLM 张量的存储布局(`[B,S,D]`,batch/seq/hidden 三个维度,最后一维 hidden 通常是内存里连续的)天然是"SoA"(Structure of Arrays)风格,这不是巧合,是几十年 GPU 编程经验倒逼出的数据布局约定:只要沿着最后一维(内存连续的那一维)做访问,天然就是合并访问模式,这也是为什么改变张量的 view/reshape/transpose 顺序有时会带来意外的性能剧变——本质上是在悄悄改变内存访问的合并程度。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from coalescing import coalesced_load, strided_load, n_sectors, efficiency

addrs_ideal = coalesced_load(0)              # 32个lane,地址0,4,8,...,124(连续)
assert n_sectors(addrs_ideal) == 1             # 全部落在同一个128字节sector
assert efficiency(addrs_ideal) == 1.0           # 100%效率

addrs_s32 = strided_load(0, 32)               # stride=32个元素(=128字节,恰好1个sector宽)
assert n_sectors(addrs_s32) == 32               # 每个lane各自落进独立的sector
assert abs(efficiency(addrs_s32) - 1/32) < 1e-6  # 效率暴跌到3.125%

# 独立验证: stride从1逐步增大到32,确认效率是单调递减的(不是某个中间stride更差)
prev_eff = 1.1   # 哨兵值,确保第一次比较通过
for stride in [1, 2, 4, 8, 16, 32]:
    addrs = strided_load(0, stride)
    eff = efficiency(addrs)
    assert eff <= prev_eff, f"stride={stride}处效率不应该比更小的stride更高"
    prev_eff = eff
    print(f"stride={stride:>2}: {n_sectors(addrs):>2}个sector, 效率={eff:.4f}")
```

**实测(`.venv` 真跑):** 理想合并访问(stride=1)确认 100% 效率,1 个 sector 服务全部 32 个请求;stride=32(step 恰好等于 1 个 sector 能装下的 word 数)确认触及 32 个不同 sector、效率精确为 `1/32≈3.125%`——这意味着为了搬运 128 字节真正有用的数据(32 个 lane×4 字节),硬件实际读取了 `32×128=4096` 字节,32 倍的带宽浪费。独立验证 stride 从 1 到 32 逐步增大时效率单调不增,不存在"中间某个 stride 反而比更大的 stride 更差"这种非单调情况——这条单调性质对写 kernel 时的直觉很重要:只要能减小访问步长(哪怕不能完全消除跨步),内存效率就不会变得更差。

**面试怎么问 + 追问链:**
- **Q:** "AoS(Array of Structures)和 SoA(Structure of Arrays)两种内存布局,哪种对 coalescing 更友好,为什么?"—— 期望:SoA 更友好——假设有 32 个粒子,每个粒子有 x/y/z 三个坐标,AoS 存储是 `[x0,y0,z0,x1,y1,z1,...]`(同一粒子的三个坐标挨在一起),如果 32 个线程各自只想读取"自己粒子的 x 坐标",地址间隔是 3 个 word(stride=3),不是连续访问;SoA 存储是 `[x0,x1,...,x31,y0,y1,...,y31,z0,...]`(同一坐标分量的 32 个粒子挨在一起),这时候 32 个线程读"各自粒子的 x 坐标"正好是连续地址(stride=1),完全合并——LLM 张量 `[B,S,D]` 布局的"最后一维连续"约定,本质上就是 SoA 思路在更高维张量上的体现。
- **追问1:** "coalescing(全局内存)和 bank conflict(shared memory,知识点 4)是同一个问题吗?"—— 期望:不是同一个模型,虽然表面上都是"访问模式导致效率下降"——coalescing 衡量的是"这次内存事务搬运的数据里有多少比例被真正用到"(sector 粒度,针对 HBM/DRAM 这类高延迟大颗粒度存储),bank conflict 衡量的是"多个并发请求有没有排队等待"(bank 粒度,针对 shared memory 这类低延迟但并行度受限的片上存储),两者的物理机制、粒度单位(128B vs 4B)、优化手段都不同,只是都属于"内存访问模式影响性能"这个大类,不能把两套分析框架混用。

**常见坑:** `n_sectors` 用整数除法 `a // SECTOR_BYTES` 划分 sector 边界,这意味着 sector 的边界是**固定对齐**的(0-127 是 sector 0,128-255 是 sector 1,以此类推),不是"从第一个访问地址开始往后数 128 字节"这种相对划分——如果 32 个 lane 的访问范围恰好跨越了一个固定 sector 边界(比如从地址 100 到 227,跨越了 0-127 和 128-255 两个 sector),即使地址总跨度不到 128 字节,也会被计成触及 2 个 sector,这个"固定对齐边界"的细节容易在心算场景时被忽略,导致对效率的估计产生偏差。

---

## 6. Reduce 三代与 Tiled GEMM(`reduce_kernel.py` + `gemm_tiled.py`)—— 从串行到树形到 Warp-Shuffle,以及 32 倍 HBM 流量削减是怎么算出来的

**是什么:**
```python
def reduce_brent_kung(data: list) -> float:
    """Tree reduction with log2 steps, each halving active threads."""
    v = list(data)
    stride = 1
    while stride < len(v):
        i = 0
        while i + stride < len(v):
            v[i] += v[i + stride]
            i += 2 * stride
        stride *= 2
    return v[0]
```
(`reduce_kernel.py:14-24`)

```python
def hbm_traffic_naive(M: int, N: int, K: int, dtype_bytes: int = 4) -> int:
    """Each thread loads K elements from A and K from B."""
    return dtype_bytes * M * N * 2 * K

def hbm_traffic_tiled(M: int, N: int, K: int, tile: int, dtype_bytes: int = 4) -> int:
    """A and B tiles loaded once per (ii,jj,kk) sub-block."""
    n_tiles = lambda x: (x + tile - 1) // tile
    loads = n_tiles(M) * n_tiles(N) * n_tiles(K) * tile * tile
    return dtype_bytes * (loads * 2)
```
(`gemm_tiled.py:38-50`,节选并简化)

**一句话:** Reduce 有三种实现思路——朴素串行(完全不并行,基线)、Brent-Kung 树形规约(`log2(n)` 步,每步活跃线程减半)、warp-shuffle(知识点 3 已展开,同样是树形但用寄存器直接交换不占 shared memory)——三者数学上必须给出完全相同的结果;Tiled GEMM 是"分块复用"思路在矩阵乘法上的具体应用:naive 版本每个输出元素独立地把整行/整列数据从 HBM 读一遍(大量重复读取),tiled 版本把数据先搬进 shared memory 里"重复利用",大幅减少 HBM 总读取量。

**底层机制/为什么这样设计:** `reduce_brent_kung` 的双层循环(外层 `stride` 每轮翻倍、内层用 `i += 2*stride` 跳步)实现的是"每一轮只有一半的位置真正参与相加,参与相加的位置本身也在向后跳跃",这是并行归约树的标准写法——第一轮 `stride=1`,位置 0 和 1 相加(结果存回 0),位置 2 和 3 相加(结果存回 2)……第二轮 `stride=2`,位置 0(已经是 0+1)和位置 2(已经是 2+3)相加,以此类推,`log2(n)` 轮之后位置 0 持有全部元素的和。GEMM 的 HBM 流量对比:`hbm_traffic_naive` 假设每个输出元素 `(i,j)` 独立地把 A 的第 i 行(K 个元素)和 B 的第 j 列(K 个元素)完整读一遍,`M*N` 个输出元素总共读取 `M*N*2*K` 个元素——完全没有利用"同一行 A 数据会被 N 个不同的输出元素重复读取"这个事实;`hbm_traffic_tiled` 把计算按 `tile×tile` 的小块组织,每个 `(ii,jj,kk)` 子块只需要把对应的 A、B 小块各读一次(存进 shared memory),这个小块内的 `tile×tile` 次乘加复用的是同一份已经搬进片上存储的数据,不需要重复访问 HBM。

**AI 研究场景:** Tiled GEMM 是 CUDA/Triton kernel 优化里最经典的教学案例(几乎每一本 GPU 编程教材都用它作为"从 naive 到优化"的第一课),背后的"分块复用减少 HBM 流量"思路直接延伸到 FlashAttention 的设计动机(把 attention 矩阵分块处理,避免把完整的 N×N attention 矩阵物化到 HBM——本仓库 [kernel-gpu-deep-dive/04](../kernel-gpu-deep-dive/04-flashattention-and-fusion.md) 详细讲过这个机制,和这里的 tiled GEMM 是同一种"分块降低访存量"思路在不同算子上的应用)。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/cuda-essentials/src")
from reduce_kernel import reduce_naive, reduce_brent_kung, reduce_warp_shuffle
from gemm_tiled import gemm_naive, gemm_tiled, hbm_traffic_naive, hbm_traffic_tiled

data = [float(i) for i in range(1024)]
expected = sum(data)
assert abs(reduce_naive(data) - expected) < 1e-3
assert abs(reduce_brent_kung(data) - expected) < 1e-3
assert abs(reduce_warp_shuffle(data) - expected) < 1e-3

random.seed(0)
M, N, K = 32, 32, 32
A = [[random.random() for _ in range(K)] for _ in range(M)]
B = [[random.random() for _ in range(N)] for _ in range(K)]
C_naive = gemm_naive(A, B)
C_tiled = gemm_tiled(A, B, tile=8)
for i in range(M):
    for j in range(N):
        assert abs(C_naive[i][j] - C_tiled[i][j]) < 1e-9    # 分块不改变数学结果，只改变访存模式

# 独立验证: 换不同tile size,确认HBM流量削减倍数精确等于tile本身(不是tile的平方)
for tile in [8, 16, 32, 64]:
    naive_bytes = hbm_traffic_naive(2048, 2048, 2048)
    tiled_bytes = hbm_traffic_tiled(2048, 2048, 2048, tile=tile)
    speedup = naive_bytes / tiled_bytes
    assert abs(speedup - tile) < 0.1
    print(f"tile={tile:>2}: HBM流量削减 {speedup:.1f}x")
```

**实测(`.venv` 真跑):** 三种 reduce 实现对 1024 个元素(0+1+...+1023=523776)给出完全一致的结果,验证并行化重排不改变加法的数学结果(浮点误差在 1e-3 内,来自求和顺序不同导致的舍入差异量级,不是算法错误)。GEMM naive 和 tiled(tile=8)在 32×32 随机矩阵上逐元素比对完全一致(误差 1e-9,浮点精度极限)。**独立验证发现一条精确的规律**:HBM 流量削减倍数**精确等于 tile size 本身**(tile=8→8.0x,tile=16→16.0x,tile=32→32.0x,tile=64→64.0x),不是 tile 的平方或其他关系——这是因为 naive 版本对每个 K 步都要重新读取 A/B(K 次完整读取),tiled 版本每个 `tile×tile` 子块只读取 A/B 各一次就复用 `tile` 次(沿 K 方向的 tile 内循环复用),复用倍数恰好是 `tile`,这条线性关系(而非二次关系)是本次验证独立发现、比 README 单一 tile=32 案例更完整的规律描述。

**面试怎么问 + 追问链:**
- **Q:** "HBM 流量削减倍数精确等于 tile size,这是不是意味着 tile size 越大越好?"—— 期望:不是——tile size 增大能持续减少 HBM 流量,但代价是需要更大的 shared memory 空间存放这个 tile(shared memory 容量有限,通常几十到上百 KB,tile 太大会导致单个 SM 能同时驻留的 block 数下降,进而影响并行度和延迟隐藏能力),真实 GEMM kernel 的 tile size 选择(常见 16/32/64/128)是"HBM 流量削减收益"和"shared memory 容量/occupancy 约束"之间的权衡,不能无限增大。
- **追问1:** "`hbm_traffic_tiled` 的公式假设每个 tile 只被读取 1 次,这在真实硬件上总是成立吗?"—— 期望:这是一个理想化假设——真实硬件上,如果 shared memory 容量不够同时装下当前需要的所有 tile,或者 kernel 调度导致同一个 tile 数据被多个 block 重复搬运(而不是通过 L2 cache 命中复用),实际 HBM 流量可能高于这个理想公式的估计;这份模型给出的是"理论下界"(在完美 tile 复用假设下能达到的最优流量),真实性能调优还需要结合具体硬件的 cache 行为综合考虑。

**常见坑:** `gemm_tiled` 的三重嵌套循环(`for ii/jj/kk`)顺序会影响真实硬件上的性能表现(即使数学结果完全相同)——比如 `kk` 循环放在最内层还是外层,决定了"部分和要不要跨 tile 累积存回内存"还是"能不能一直留在寄存器/shared memory 里累加",这份 Python 参考实现只关心结果正确性和 HBM 流量的理论计数,没有建模真实的循环顺序对 cache/寄存器复用的影响,不能直接当作"最优 tiled GEMM 实现"的模板照抄。

---

## 7. Capstone:Online Softmax 递推(`capstone_softmax.py`)—— FlashAttention 能做 Kernel Fusion 的前置知识

**是什么:**
```python
def softmax_online(x: list) -> list:
    """1-pass running max + running sum, FlashAttn-style."""
    m = -math.inf
    d = 0.0
    for v in x:
        m_new = max(m, v)
        d = d * math.exp(m - m_new) + math.exp(v - m_new)
        m = m_new
    return [math.exp(v - m) / d for v in x]
```
(`capstone_softmax.py:14-30`,节选)

**一句话:** 标准 softmax 需要三趟扫描(先找最大值防止数值溢出,再算 exp 求和,最后归一化),Online Softmax 用一对不变量(`m`=当前见过的最大值、`d`=按当前最大值重新校正过的累积和)把这个过程压缩成一趟扫描,每见到一个新元素就同时更新这两个不变量,不需要提前知道全局最大值。

**底层机制/为什么这样设计:** 核心技巧在更新公式 `d = d * exp(m - m_new) + exp(v - m_new)`——当扫描到一个比当前最大值 `m` 更大的新元素 `v` 时,最大值需要更新为 `m_new`,但这意味着**之前累积的 `d`(是按旧的 `m` 校正过的)现在校正基准变了**,不能直接沿用;这一步用 `d * exp(m - m_new)` 把旧的累积和"重新校正"到新的基准上(`exp(m-m_new)` 是一个 ≤1 的缩放因子,因为 `m_new >= m`),再加上新元素按新基准算出的 `exp(v-m_new)`——这个"重新校正"操作在数学上等价于"如果一开始就知道全局最大值、直接用它做三趟扫描",但只需要一趟扫描就完成,代价是每次最大值更新时要多做一次缩放乘法。这个递推能不 materialize 完整的中间数组(不需要先算出全部 `exp(x_i)` 再求和),是 FlashAttention 能把 Q/K/V 的 attention 计算全程保持在 shared memory/寄存器里、不用把 N×N 的 attention 矩阵写回 HBM 的数学基础。

**AI 研究场景:** Online softmax 是理解 FlashAttention 为什么能做到"数学上和标准 attention 完全等价,但 HBM 访存量从 O(N²) 降到 O(N)"这一核心主张的必经知识点——本仓库 [kernel-gpu-deep-dive/04](../kernel-gpu-deep-dive/04-flashattention-and-fusion.md) 知识点 1 已经在真实 FlashAttention 实现的语境下讲过这套递推(那边额外覆盖了分块并行的 `l_new`/`o_new` 三步更新),本知识点是这个机制最精简、最容易独立推导验证的数学内核。

**可运行例子:**
```python
import sys, math
sys.path.insert(0, "learning/cuda-essentials/src")
from capstone_softmax import softmax_naive, softmax_online

x = [3.0, 1.0, 0.2, -5.0, 7.0, 2.0]
a = softmax_naive(x)
b = softmax_online(x)
for u, v in zip(a, b):
    assert abs(u - v) < 1e-9
assert abs(sum(a) - 1.0) < 1e-9
assert abs(sum(b) - 1.0) < 1e-9

# 数值稳定性: 大数值输入(标准naive softmax如果不减max会直接exp溢出)
big = [1000.0, 1001.0, 1002.0]
bn = softmax_naive(big)   # 内部先减去max=1002,所以不会溢出
bo = softmax_online(big)
for u, v in zip(bn, bo):
    assert abs(u - v) < 1e-9

# 独立验证: 如果真的不做数值稳定化处理(不减max直接exp),大数值输入会立刻溢出
try:
    raw_exp = [math.exp(v) for v in big]
    assert False, "不应该走到这里"
except OverflowError:
    print("确认: math.exp(1000)本身就会OverflowError,这正是两种softmax实现都要先减max的原因")

print(f"softmax([3,1,0.2,-5,7,2]) 第5位(值最大的7.0对应): {b[4]:.4f}")
```

**实测(`.venv` 真跑):** 6 个元素的测试向量下,naive 和 online 两种实现逐元素精确一致(误差 1e-9,浮点极限);极端数值输入(1000/1001/1002)两者依然精确一致,证明"减去最大值"这个数值稳定化技巧在两种实现里都正确生效。独立验证补充了一个直接证据:**如果真的不做任何数值稳定化处理**,直接对 1000.0 调用 `math.exp` 会立即抛出 `OverflowError`(Python float 的指数上限在 `e^709` 左右,`e^1000` 远超此范围)——这直接证明"先减去最大值再算 exp"不是可选的代码风格偏好,是避免数值溢出崩溃的必要步骤,两种 softmax 实现(naive 的 `m=max(x)` 和 online 的运行时 `m` 更新)都建立在这条前提上。

**面试怎么问 + 追问链:**
- **Q:** "Online softmax 的更新公式里,为什么要用 `d * exp(m - m_new)` 而不是直接用新的 `m_new` 重新算一遍完整的和?"—— 期望:重新算一遍完整的和需要保留之前所有见过的元素(退化回三趟扫描,需要 O(N) 额外存储),而 `d * exp(m-m_new)` 这个"重新校正"操作只需要 `d` 和 `m` 这两个标量,不需要保留任何历史元素——这正是这个算法能做到 O(1) 额外空间、真正做到"一趟扫描"的关键技巧,不是为了炫技,是为了避免存储开销。
- **追问1:** "如果输入序列全部是相同的值(比如全部是 5.0),`m` 会怎么变化?这种情况下算法还正确吗?"—— 期望:第一个元素后 `m=5.0, d=1.0`(`exp(5-5)=1`);后续元素 `v=5.0=m`,不触发 `m` 更新(`max(m,v)=m`),`d` 持续累加 `exp(0)=1`,最终 `d=N`(元素个数),`softmax` 结果是 `[1/N, 1/N, ..., 1/N]`(均匀分布)——这是完全符合直觉的正确结果(全部数值相同时 softmax 就应该是均匀分布),算法在这种边界情况(没有严格更大的新最大值出现)下依然正确工作,不需要特殊处理。

**常见坑:** 这份 online softmax 实现是"单线程串行"的教学版本,真实 FlashAttention 的 online softmax 是在**多个并行处理的分块**之间做这套递推合并(每个 GPU thread block 处理一个 Q 分块和多个 K/V 分块,分块之间需要按这套 `m`/`d` 更新规则合并各自的局部统计量),涉及额外的 `l_new`/`o_new`(输出累积值)更新步骤,并且要处理"多个分块并行计算、结果需要按顺序合并"的同步问题——本知识点只覆盖了递推公式的数学内核,完整的分块并行版本推荐参照 [kernel-gpu-deep-dive/04](../kernel-gpu-deep-dive/04-flashattention-and-fusion.md) 知识点 1 的更完整展开。

---

## 8. 番外:官方 Guide 自身例子复现(`cuda_original_minimal.py`)—— Streams、CUDA Graphs 与 Occupancy 估算

**是什么:**
```python
def graph_submission_overhead_us(n_ops: int, repeats: int, *,
                                  per_op_launch_us: float = 5.0,
                                  instantiate_us: float = 80.0,
                                  graph_launch_us: float = 6.0) -> dict:
    """Compare repeated host launches with one captured CUDA graph."""
    normal = repeats * n_ops * per_op_launch_us
    graphed = instantiate_us + repeats * graph_launch_us
    return {"normal_us": normal, "graph_us": graphed, "speedup": normal / graphed}

def occupancy_from_threads(threads_per_block: int, *,
                            max_threads_per_sm: int = 2048,
                            max_blocks_per_sm: int = 32) -> dict:
    blocks_by_threads = max_threads_per_sm // threads_per_block
    blocks = min(max_blocks_per_sm, blocks_by_threads)
    active_threads = blocks * threads_per_block
    return {"blocks_per_sm": blocks, "active_threads": active_threads,
            "occupancy": active_threads / max_threads_per_sm}
```
(`cuda_original_minimal.py:85-117`,节选)

**一句话:** 本文件是唯一没有对应独立 lecture、只在 878 行论文导读里出现的脚本,复现官方 *CUDA C++ Programming Guide* 自身举的几个例子:线程索引线性化、block 调度的"波次"、CUDA Streams 的执行顺序、CUDA Graphs 摊薄重复 launch 开销、以及资源受限下的 occupancy(占用率)估算。

**底层机制/为什么这样设计:** `graph_submission_overhead_us` 对比两种执行同一组操作 `repeats` 次的方式——每次都单独 launch(`normal = repeats*n_ops*per_op_launch_us`,launch 开销线性累积)vs 先"录制"成一个 CUDA Graph 再重复"回放"(`graphed = instantiate_us + repeats*graph_launch_us`,一次性的录制开销 `instantiate_us` 分摊到所有重复次数上,后续每次只需要一个远比单独 launch 便宜的 graph_launch 开销)——这个模型抓住了 CUDA Graphs 的核心价值主张:如果同一组 kernel 序列要被重复执行很多次(常见于训练循环里每个 step 都是同一套 kernel),把 launch 开销从"每次都要付"变成"只在第一次录制时付一次固定成本",重复次数越多,摊薄效果越明显。`occupancy_from_threads` 用两个独立的硬件上限(`max_threads_per_sm`总线程数上限、`max_blocks_per_sm`总 block 数上限)分别算出能装下的 block 数,取较小值——这是"资源受限"这个概念最简化的版本(真实 occupancy 计算还要考虑寄存器和 shared memory 用量这两个额外的资源维度,这份简化模型只建模了线程数和 block 数两个上限)。

**AI 研究场景:** CUDA Graphs 是近年推理服务框架(如 vLLM,本仓库 [inference-serving-deep-dive/01](../inference-serving-deep-dive/01-inference-engine-core.md) 知识点 9 已经讨论过 CUDA Graphs 在真实推理引擎里的应用)大量采用的优化手段,尤其是 decode 阶段"同一套 kernel 序列反复执行成千上万次"这个场景是 CUDA Graphs 的最佳适用条件;occupancy 是 GPU kernel 性能调优时最常查看的诊断指标之一,决定了一个 SM 上有多少线程可以交替执行来"隐藏"内存访问延迟。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cuda-essentials/src")
from cuda_original_minimal import (linear_thread_index, Dim3, warp_and_lane,
                                     graph_submission_overhead_us, occupancy_from_threads)

assert linear_thread_index(Dim3(3, 2, 1), Dim3(8, 4, 2)) == 3 + 2 * 8 + 1 * 8 * 4   # x最快变化,然后y,然后z
assert warp_and_lane(63) == (1, 31)     # 第63号线程: warp 1(63//32), lane 31(63%32)
assert warp_and_lane(64) == (2, 0)       # 第64号线程: 恰好是下一个warp的第0个lane

g = graph_submission_overhead_us(n_ops=10, repeats=100)
assert g["speedup"] > 5.0
g_few = graph_submission_overhead_us(n_ops=10, repeats=2)   # 重复次数很少时,一次性instantiate开销可能占主导
assert g_few["speedup"] < g["speedup"]     # 重复次数越少,graph相对normal的加速比越小(分摊不充分)

occ_768 = occupancy_from_threads(768)
assert round(occ_768["occupancy"], 2) == 0.75    # 2048/768=2.67->取2个block,2*768=1536,1536/2048=0.75
occ_32 = occupancy_from_threads(32)
assert round(occ_32["occupancy"], 2) == 0.50      # 32线程/block: 2048/32=64个block,但max_blocks_per_sm=32封顶,32*32=1024/2048=0.5
print(f"threads_per_block=768: occupancy={occ_768['occupancy']:.0%}")
print(f"threads_per_block=32:  occupancy={occ_32['occupancy']:.0%}(被max_blocks_per_sm=32这个上限卡住,不是线程数上限)")
```

**实测(`.venv` 真跑):** `linear_thread_index`/`warp_and_lane` 两个索引公式全部精确匹配手算结果。`graph_submission_overhead_us(n_ops=10, repeats=100)` 给出 speedup>5倍;独立验证"重复次数很少"(`repeats=2`)场景下 speedup 显著更低,证实了"一次性 instantiate 开销需要足够多的重复次数才能被摊薄"这条直觉。`occupancy_from_threads(768)` 精确算出 0.75(受线程总数上限约束:`2048//768=2` 个 block,`2×768=1536` 活跃线程,`1536/2048=0.75`);`occupancy_from_threads(32)` 精确算出 0.50——这个结果**不是**被线程数上限卡住的(`2048//32=64` 个 block 理论上可以装下),而是被另一个独立的硬性上限 `max_blocks_per_sm=32` 卡住(`min(32, 64)=32` 个 block,`32×32=1024` 活跃线程,`1024/2048=0.5`),两个 0.75/0.50 结果背后被"卡住"的硬件资源维度完全不同,这是这个知识点最容易被忽视的细节。

**面试怎么问 + 追问链:**
- **Q:** "`occupancy_from_threads(32)` 的 occupancy 只有 50%,为什么不能通过'增大 threads_per_block'来提高占用率?"—— 期望:恰恰相反——本例的瓶颈是"block 数量"(被 `max_blocks_per_sm=32` 卡住),增大 `threads_per_block` 实际上会进一步减少能同时驻留的 block 数(`max_threads_per_sm // threads_per_block` 分母变大),占用率可能不升反降;真正提高这个场景占用率的方向反而是**减小**每个 block 的线程数,让更多 block 能同时驻留(但这本身要在"block 数上限 32"和"总线程数上限 2048" 两条约束的交点附近找最优,不是无限减小就一直更好)。
- **追问1:** "真实 GPU 上 occupancy 除了受线程数/block 数限制,还受什么资源限制?这份简化模型为什么没有建模那些?"—— 期望:真实 occupancy 还受寄存器文件大小(每个线程用的寄存器越多,一个 SM 能同时容纳的线程总数就越少)和 shared memory 容量(每个 block 用的 shared memory 越多,能同时驻留的 block 数就越少)这两个额外维度的约束——这份模型只用"线程数上限"和"block 数上限"两个最基础的硬件限制做简化演示,这也是为什么它的函数名明确叫 `occupancy_from_threads`(强调只考虑线程数这一个输入维度),不是声称完整复现了 CUDA occupancy calculator 的全部逻辑。

**常见坑:** `graph_submission_overhead_us` 和 `occupancy_from_threads` 用的默认参数(`per_op_launch_us=5.0`、`instantiate_us=80.0`、`max_threads_per_sm=2048`等)是有代表性的经验估计值,不是从某一款具体 GPU 型号(如 H100/A100)的官方规格里精确抄来的——真实调优时这些参数应该替换成目标硬件的真实规格(可以从 CUDA occupancy calculator 工具或 `cudaGetDeviceProperties` API 查到精确值),本知识点的价值在于理解计算逻辑本身,不在于记住这几个默认数字。

---

*上一篇:[04-small-model-graduation.md](04-small-model-graduation.md) | 下一篇:[06-cluster-networking.md](06-cluster-networking.md) —— 从单卡内部的执行模型,扩展到多机多卡的集群网络通信。*
