# 04 · FlashAttention 与 Kernel Fusion 深挖(FlashAttention & Kernel Fusion)

> 总览见 [00-roadmap.md](00-roadmap.md)

**先落地一遍全系列开头就强调过的差异化声明,具体到这一批的四个源文件上:** `learning/kernel-engineering/src/{flash_attention,fused_mlp,rmsnorm_kernel,capstone_attn_speedup}.py` 不是可编译的真实 CUDA/Triton kernel——这台 Windows 工作站没有 Linux CUDA 工具链,真编译不在这两个 `learning/` 模块、也不在本系列任何一篇的范围内。这四个脚本用**可断言验证的纯 Python 数值/机制模拟**去复现 FlashAttention 的分块递推和 kernel fusion 省下的访存往返,练的是"HBM 流量具体怎么算""为什么这样切块在数学上站得住脚"这套第一性原理,不是背某个框架的 API 怎么调。全部代码只依赖 `math`/`__future__`,纯 CPU、零第三方依赖,已在仓库根目录 `.venv`(Windows 原生,Python 3.13.9)下逐个实测跑通——下面每个"可运行例子"的输出都是这次重新独立跑出来的,包括撰写过程中新增的全部 assert,不是转述文档或者凭记忆断言。

本文覆盖 `learning/kernel-engineering/` 六个脚本里的后四个:`flash_attention.py`(知识点 1)、`capstone_attn_speedup.py`(知识点 2 的流量公式 + 知识点 5 的完整曲线,两个知识点共用同一对函数,分别看"怎么算"和"算出来的具体数字有多夸张")、`fused_mlp.py`(知识点 3)、`rmsnorm_kernel.py`(知识点 4)。这四个脚本彼此**零 import**,谁先跑都不影响谁。知识点 1 涉及的 online softmax 数学,和姊妹系列 [long-context-deep-dive/02-long-context-attention.md](../long-context-deep-dive/02-long-context-attention.md) 里 Ring Attention 用的是同一套递推(那边是把 block 换成"某张卡负责的一段 K/V",这里是单卡内部按 block 扫描),核心数学完全相同,那边已经写过的证明细节这里不重复,只在知识点 1 结尾点一句关联。

**本篇统一结构(七步,和 torch-deep-dive/huggingface-deep-dive 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(带 assert,真在 `.venv` 里跑过)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. Online Softmax 递推(`flash_attention.py`)—— 分块算 softmax,为什么和标准 softmax 精确等价

**是什么:**
```python
def attention_flash(Q: list[list[float]], K: list[list[float]],
                    V: list[list[float]], block_n: int = 4) -> list[list[float]]:
    """FlashAttention-2: outer loop over Q rows, inner loop over K/V blocks.

    Online running stats per Q row:
      m: running max
      l: running normalizer
      O: running output (un-normalized in numerator)
    """
    N, d = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(d)
    O = [[0.0] * d for _ in range(N)]

    for i in range(N):
        m = -math.inf
        l = 0.0
        o = [0.0] * d
        for j_start in range(0, N, block_n):
            j_end = min(j_start + block_n, N)
            s_block = [sum(Q[i][k] * K[j][k] for k in range(d)) * scale
                       for j in range(j_start, j_end)]
            m_new = max(m, max(s_block))
            rescale = math.exp(m - m_new) if m != -math.inf else 0.0
            l = l * rescale
            o = [v * rescale for v in o]
            for jj, s in enumerate(s_block):
                p = math.exp(s - m_new)
                l += p
                j = j_start + jj
                for k in range(d):
                    o[k] += p * V[j][k]
            m = m_new
        O[i] = [v / l for v in o]
    return O
```
(源码见 `learning/kernel-engineering/src/flash_attention.py:27-62`,核心递推在 `40-61` 行;`m`/`l`/`o` 分别是"目前见过的最大分数""目前的归一化分母""目前的未归一化输出累加值"这三个 running state,注意源码里没有单独的 `m_new`/`l_new`/`o_new` 变量对——`l`、`o` 是直接原地更新覆盖旧值,只有 `m_new` 在赋给 `m` 之前短暂存在。)

这段代码里 `Q[i][k] * K[j][k]` 这个点积、`scale = 1/sqrt(d)` 这个缩放、以及最后要对结果做 softmax,合起来就是标准 scaled dot-product attention 那套公式本身——如果还没搞清楚 Q/K/V 到底是什么、点积为什么能衡量"相似度"、为什么偏偏要除以 `sqrt(d)`,见 [torch-deep-dive/04-layers-math-and-backward.md](../torch-deep-dive/04-layers-math-and-backward.md) 第 8 节"在讲拆分之前"那一段从零建立的内容(Python 字典查询的类比、3 个 token 的可验证玩具例子)。本节和姊妹系列 `long-context-deep-dive` 一样,直接假设你已经懂 attention 本身,只讲这套计算"怎么分块、还能保证结果精确不变"这一层新东西,不重复推导 attention 公式本身。

**一句话:** 逐个 K/V block 扫过去,每看完一个新 block 就用三步刷新状态——`m_new = max(m, max(s_block))` 把"目前见过的最大值"更新到位,`rescale = exp(m - m_new)` 算出"旧的累积值现在应该按什么系数收缩"(因为它们当初是按旧的、偏小的 `m` 做的指数),`l`/`o` 先乘上这个 `rescale` 再加上新 block 的贡献——全程只需要同时装得下 1 个 block 的 `s_block`,从不需要在内存里铺开整行的分数。

**底层机制/为什么这样设计:** 先问一个最笨的问题——标准 softmax 为什么要减去一个 `max`?纯粹是数值稳定性:`softmax(s)_i = exp(s_i-m)/Σexp(s_j-m)` 对任意常数 `m` 恒成立(分子分母同时乘 `exp(-m)`,比值不变),选 `m=max(s)` 只是让指数的输入永远 `<=0`,不会上溢。这个恒等式的关键在于"`m` 可以是任意常数",于是第一个天然的问题就来了:如果还没扫完整行,不知道最终的全局最大值是多少,怎么办?

`attention_naive` 的答案是"先把这一整行的 N 个分数全部算出来、存起来,再一次性求 max"(见下一节,这一步正是需要把 N×N 分数矩阵摆进内存的根源)。`attention_flash` 的答案是"边扫边修正":每来一个新 block,先把"目前见过的最大值"更新成 `m_new`,但手头已经按旧的 `m` 算好的 `l`(分母)和 `o`(分子的加权和)不能直接沿用——它们的每一项当初是按 `exp(s_i-m)` 算的,现在基准变成了 `m_new`,需要把每一项都"换底"。

这里最值得手推一遍的是指数的可加性恒等式:`exp(s_i-m_new) = exp(s_i-m) * exp(m-m_new)`。这意味着"用新基准重算所有旧项"完全等价于"把旧的累积结果整体乘上一个和具体的 `s_i` 无关的公共系数 `exp(m-m_new)`"——因为这个系数对当前 block 之前所有的 `i` 都相同,可以直接乘在累积量 `l`、`o` 上,完全不需要回头重新遍历每一个旧的 `s_i`。这就是 `rescale=exp(m-m_new)` 这一行在做的事,`l=l*rescale`、`o=[v*rescale for v in o]` 把"旧状态换算到新基准下应该是什么值"一步做完。数学上可以对 block 数做归纳:处理完前 `t` 个 block 后,`(l,o)` 恒等于"只用这 `t` 个 block 里出现过的分数、以当前的 `m`(这 `t` 个 block 的最大值)做标准 softmax 未归一化"的结果——归纳的每一步都是精确的代数恒等变形,不涉及任何截断意义上的"近似";最后一个 block 处理完,`m` 就是全局最大值,`o/l` 因此就是标准 softmax 的精确输出,和 block 怎么切、切多大完全无关。

（补充一个从源码里读到、值得诚实记录的小细节:`rescale = math.exp(m - m_new) if m != -math.inf else 0.0` 里的 `-inf` 特判,在当前这份纯 Python 实现下其实是多余的——`math.exp(-math.inf - m_new)`(`m_new` 有限)本身就会干净地返回 `0.0`,不会抛异常也不会得到 `nan`,下面的可运行例子会现场验证这一点。这不是 bug,只是一处不影响结果的防御性冗余;了解"这一行技术上可以删"不代表"应该删",这恰恰是读代码时要分清"必要逻辑"和"防御性但冗余的逻辑"的一个具体例子。）

**AI 研究/工程场景:** 这套"边看边修正、永远不用回头"的递推,是 FlashAttention 敢把 attention 计算拆成能塞进 SRAM/寄存器的小块、而不用退回"先算完整行再 softmax"的数学基础——没有这个精确等价的保证,分块计算出来的 attention 就只是一个近似算法,精度经不住训练场景的检验。这也是为什么同一套 online softmax 递推能被复用到分布式场景:[long-context-deep-dive/02-long-context-attention.md](../long-context-deep-dive/02-long-context-attention.md) 的 Ring Attention 把"block"换成了"某张卡负责的一段 K/V",本质是同一套 `m`/`l`/`o` 递推,只是把"下一个 block 从哪来"从"下一次循环迭代"换成了"环形通信收到的下一份数据"——这里不重复那边已经写过的证明细节,只强调核心数学复用的这一层关系。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from flash_attention import attention_naive, attention_flash
import math, random

# Part A: 手工构造最小 trace(1 个 query、4 个 key,拆成 2 个 block),核心恒等式现场验证
s = [2.0, 0.5, 3.0, 1.0]        # 假装是这一行 Q 和 4 个 K 算出来的 raw score(已含 1/sqrt(d) 缩放)
v = [10.0, 20.0, 30.0, 40.0]     # 对应的 V(标量简化,d=1,方便手算)

m_std = max(s)
p_std = [math.exp(x - m_std) for x in s]
o_std = sum(p * vi for p, vi in zip(p_std, v)) / sum(p_std)

m, l, o = -math.inf, 0.0, 0.0
for sb, vb in [(s[0:2], v[0:2]), (s[2:4], v[2:4])]:      # 模拟 block_n=2
    m_new = max(m, max(sb))
    rescale = math.exp(m - m_new) if m != -math.inf else 0.0
    l = l * rescale
    o = o * rescale
    for si, vi in zip(sb, vb):
        p = math.exp(si - m_new)
        l += p
        o += p * vi
    m = m_new
o_online = o / l

assert o_std == o_online == 25.694766183237302   # 位级相等,不是"足够接近"

# Part B: 用真实源码在 N=8 的矩阵上验证,block_n 故意不整除 N(最后一个 block 只有 2 行)
random.seed(7)
N, d = 8, 4
Q = [[random.random() for _ in range(d)] for _ in range(N)]
K = [[random.random() for _ in range(d)] for _ in range(N)]
V = [[random.random() for _ in range(d)] for _ in range(N)]
O1 = attention_naive(Q, K, V)
O2 = attention_flash(Q, K, V, block_n=3)

max_diff = max(abs(O1[i][k] - O2[i][k]) for i in range(N) for k in range(d))
assert max_diff < 2e-16    # 本机实测 1.11e-16,是 float64 机器精度量级(~2.22e-16),不是算法误差
print(f"naive vs flash (N=8,d=4,block_n=3) max diff: {max_diff:.3e}")

# Part C: 验证 "-inf 特判" 在 Python 里确实是多余的(不是 bug,只是冗余防御)
assert math.exp(-math.inf - 3.5) == 0.0
```
本机实测输出:`naive vs flash (N=8,d=4,block_n=3) max diff: 1.110e-16`。

**面试怎么问 + 追问链:**
- **Q:** "FlashAttention 为什么要分块算 softmax?分块之后的结果和一次性算完整行的标准 softmax 是完全一样,还是一种近似?" —— 期望第一句话就答"精确一样,不是近似",再展开怎么做到的。
- **追问 1:** "处理第一个 block 时不知道全局最大值,用的是这个 block 内的局部最大值,这样得到的中间结果不就是'错'的吗,后面怎么纠正回来?" —— 期望答出"错"的只是中间状态相对于最终目标的基准点,`rescale` 会在每次更新 `m` 时把所有历史累积值重新换算到新基准下,永远保持"目前的 `(l,o)` 精确等于只用已扫过的 block 做标准 softmax(未归一化)"这个不变量。
- **追问 2(深挖):** "能不能不看代码,只用一个数学式子说明这个递推为什么精确成立?" —— 期望现场写出 `exp(s_i-m_new)=exp(s_i-m)*exp(m-m_new)`,并指出右边第二项和 `i` 无关,所以能被提到累加符号外面,整体乘在旧的 `(l,o)` 上。
- **追问 3:** "如果把 block 的切分方式换一种(比如从 4 个一组换成 3 个一组),或者把 block 顺序打乱,最终结果会变吗?" —— 期望答"不会,只要每个元素恰好被访问一次,`m` 最终都会收敛到全局最大值,递推的正确性和 block 大小、切分边界、扫描顺序都无关"——可以现场提到刚才 `block_n=3` 在 `N=8` 上不整除也通过了验证。

**常见坑:**
- 把"分块计算"和"近似算法"划等号——受 sparse attention / linear attention 这类真正牺牲精度换速度的方法影响,容易先入为主地认为 FlashAttention 也是用近似换效率;实际上它是纯粹的工程优化(怎么安排访存和计算顺序),数学上和标准 softmax 精确等价,这也是它能直接用在训练(而不仅是推理)里、不需要额外考虑精度损失的原因。
- 只记住"要减 max 防止上溢"这个结论,却说不出"为什么减去任意常数 softmax 的值不变"——这是能不能推出整个 online softmax 合法性的地基,面试里被追问"为什么这样做在数学上是对的"时最容易卡住。
- 误以为 `rescale` 只在 `m` 真的变化时才需要乘;实际上代码里每个 block 都会算一次 `rescale`(哪怕这次 `m_new==m`,此时 `rescale=exp(0)=1`,乘了等于没乘),这是为了让代码逻辑保持统一,不需要为"要不要更新"单独写分支判断。

---

## 2. Naive vs Flash 的 HBM 流量对比(`capstone_attn_speedup.py`)—— 那个从不被 materialize 的 N×N 矩阵

**是什么:**
```python
def hbm_naive_attn(N: int, d: int, dtype_bytes: int = 2) -> int:
    """Q, K, V read + S=N×N matrix written + P=N×N written + output written."""
    qkv = 3 * N * d
    s_matrix = N * N      # written and re-read
    p_matrix = N * N
    out = N * d
    return dtype_bytes * (qkv + 2 * s_matrix + 2 * p_matrix + out)


def hbm_flash_attn(N: int, d: int, dtype_bytes: int = 2) -> int:
    """Q, K, V read + output written. No N×N materialized to HBM."""
    qkv = 3 * N * d
    out = N * d
    return dtype_bytes * (qkv + out)
```
(源码见 `learning/kernel-engineering/src/capstone_attn_speedup.py:5-18`——这两个函数是知识点 5 capstone 的地基,这里先看它们在算什么、为什么这么算。)

**一句话:** naive attention 要把 `S=QK^T`(N×N 分数矩阵)和 `P=softmax(S)`(N×N 概率矩阵)都完整写入 HBM 一次、再读出来一次(分别对应 `attention_naive` 里"先算完整行的 `S`"和"再算完整行的 `P`"这两步各自的物化),flash attention 全程只读 Q/K/V、写最终输出 `O`,这两个 N×N 矩阵**从头到尾不会以完整矩阵的形态出现在 HBM 里**——这正是 `attention_flash` 每次只在局部变量里(对应真实硬件的 SRAM/寄存器)构造一个 `block_n` 大小的 `s_block`,用完即弃、不回写整行的原因。

**底层机制/为什么这样设计:** 先回到 `attention_naive` 的源码读一遍(`flash_attention.py:6-24`):`S = [[... for j in range(N)] for i in range(N)]` 先把整个 N×N 的 `S` 算出来、存成一个 Python 列表;紧接着 `P = []` 循环对 `S` 的每一行做 `max`/`exp`/归一化,又产出一个完整的 N×N 的 `P`;最后 `O = [[sum(P[i][j]*V[j][k] ...` 才用 `P` 算出输出。这是纯 Python,`S`、`P` 只是内存里的列表,没有真的写"HBM";但 `hbm_naive_attn` 这个函数把同样的数据流程"翻译"成了真实 GPU kernel 会发生的事:如果 `S`、`P` 是分开的 kernel(或分开的 kernel 阶段)算出来的,每一个都得先整体写回 HBM,下一步才能把它整体读回来接着算——这就是代码里 `2*s_matrix + 2*p_matrix` 里那个"×2"的来源(写一次、读一次)。而 `hbm_flash_attn` 对应的是 `attention_flash` 的真实行为:`s_block` 只是循环体内部的局部变量,离开这次循环迭代就不再需要,不需要、也从来没有整体地写回 HBM。

从最笨的想法讲起——为什么 naive 的写法一定要把整个 `S` 摆出来?因为标准 softmax 要对一整行求 max 和求和做归一化,如果不能保证"这一行的所有数已经在手边",就没法算出正确的分母。`attention_flash` 能避开这个限制,靠的正是知识点 1 讲的 online softmax 递推——用"随时可修正的 running state"替代"必须先看到全部才能归一化"这个前提。这两个知识点其实是同一个优化的两面:**知识点 1 是"数学上为什么能这样算",知识点 2 是"省下来的访存量具体是多少"**。

**AI 研究/工程场景:** N×N 矩阵这个量级有多夸张,决定了这个优化在真实场景里是"锦上添花"还是"生死攸关"——训练/推理常见的序列长度动辄几千到几十万 token,`N²` 在 `N` 变大时增长速度远超 `N*d`(`d` 通常固定在 64~256 这个量级,不随上下文变长而变),这正是知识点 5 capstone 要展开算的东西。更直接的后果是:如果没有 FlashAttention 这类优化,想训练长上下文模型往往连显存都放不下这个 N×N 中间矩阵(而不仅仅是"变慢"),这也是这一类优化在实际大模型训练里几乎是标配、而不是可选项的原因。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from capstone_attn_speedup import hbm_naive_attn, hbm_flash_attn

# 小规模手算核对公式(N=128, d=64, dtype_bytes=2 即 fp16/bf16)
N, d, db = 128, 64, 2
naive = hbm_naive_attn(N, d, db)
flash = hbm_flash_attn(N, d, db)

# naive: QKV 读 3Nd + S 矩阵写+读 2N² + P 矩阵写+读 2N² + 输出写 Nd = db*(4Nd + 4N²)
manual_naive = db * (4 * N * d + 4 * N * N)
# flash: 只有 QKV 读 3Nd + 输出写 Nd = db*4Nd,全程不 materialize N×N 矩阵
manual_flash = db * 4 * N * d

assert naive == manual_naive == 196608
assert flash == manual_flash == 65536
assert naive / flash == 3.0

# N 越大,S/P 两个 N×N 矩阵的占比越夸张(N² 项主导),d 不变时优势持续扩大
N2 = 4096
naive2 = hbm_naive_attn(N2, d, db)
flash2 = hbm_flash_attn(N2, d, db)
assert round(naive2 / flash2, 1) == 65.0
```

**面试怎么问 + 追问链:**
- **Q:** "naive attention 实现里,显存/HBM 的瓶颈具体在哪一步?" —— 期望明确指出是 `S`(`QK^T`)和 `P`(softmax 之后)这两个 N×N 矩阵的写入和读出,而不是笼统地说"attention 显存开销大"。
- **追问 1:** "这两个矩阵为什么各自要算'写一次、读一次'共两份流量,而不是各算一份?" —— 期望答出"算完要先落到 HBM(因为下一步计算不是在同一个寄存器/SRAM 生命周期内完成的),下一步开始时要再读回来",呼应公式里的"×2"。
- **追问 2(深挖):** "FlashAttention 省的是访存,那它的总浮点计算量(FLOPs)比 naive attention 更少吗?" —— 期望答"基本没有减少,甚至因为要在每个 block 内重复做归一化的 rescale 计算,FLOPs 略有增加;收益完全来自访存量的减少,这是一个典型的把 memory-bound 问题往 compute-bound 方向搬的优化",这道题最容易被问倒,能连回 02 批次 roofline 系列的 compute-bound/memory-bound 分类。
- **追问 3:** "如果 GPU 的 HBM 带宽突然变得无限大,这个优化还有意义吗?" —— 开放题,期望答出"意义会大幅下降,因为 FlashAttention 的核心收益就是省访存;但 SRAM/寄存器容量仍然有限,即使带宽不是瓶颈,naive 方式要求整行 N 个数同时可寻址这件事本身,在 N 很大时也可能超出片上存储容量",引导候选人把"带宽"和"容量"这两个不同维度的约束分开讨论。

**常见坑:**
- 把 `hbm_naive_attn`/`hbm_flash_attn` 公式里的 `qkv=3*N*d` 误当成"也要写一次、读一次"——Q/K/V 是外部输入,两个函数都只统计了"读一次"(它们不是中间结果,不存在"先写后读"的往返),容易和 `s_matrix`/`p_matrix` 的"写+读"两份混淆。
- 以为 `attention_naive` 源码里的 `S`、`P` 两个 Python 列表本身就是"HBM 流量的证据"——源码是纯 Python 模拟,`S`/`P` 只是普通内存里的 list,`hbm_naive_attn` 是另一个独立的函数,用手写公式去"翻译"这套算法在真实 GPU 上会产生多少 HBM 流量,两者是概念上的对应关系,不是同一段代码在被测量。

---

## 3. Kernel Fusion 的 HBM 流量核算(`fused_mlp.py`)—— 省下的是隐藏层那一次完整往返

**是什么:**
```python
def mlp_unfused(x: list[list[float]], W1: list[list[float]],
                W2: list[list[float]]) -> list[list[float]]:
    """3 HBM round-trips: matmul1 → write h, read h → gelu → write, read → matmul2."""
    N, D = len(x), len(x[0])
    H = len(W1[0])
    h = [[sum(x[i][d] * W1[d][k] for d in range(D)) for k in range(H)] for i in range(N)]
    h2 = [[gelu(v) for v in row] for row in h]
    out = [[sum(h2[i][k] * W2[k][d] for k in range(H)) for d in range(D)] for i in range(N)]
    return out


def mlp_fused(x: list[list[float]], W1: list[list[float]],
              W2: list[list[float]]) -> list[list[float]]:
    """Fused: per-row keep activation in registers, never write h to HBM."""
    N, D = len(x), len(x[0])
    H = len(W1[0])
    out = [[0.0] * D for _ in range(N)]
    for i in range(N):
        h_row = [gelu(sum(x[i][d] * W1[d][k] for d in range(D))) for k in range(H)]
        for d_out in range(D):
            out[i][d_out] = sum(h_row[k] * W2[k][d_out] for k in range(H))
    return out


def hbm_traffic(N: int, D: int, H: int, fused: bool, dtype_bytes: int = 2) -> int:
    """Bytes transferred to/from HBM."""
    weights = dtype_bytes * (D * H + H * D)        # W1 + W2 read once
    io = dtype_bytes * (N * D + N * D)             # x read, out write
    if fused:
        return weights + io
    # Unfused: hidden h written + read once (2× N*H)
    return weights + io + 2 * dtype_bytes * N * H
```
(源码见 `learning/kernel-engineering/src/fused_mlp.py:11-43`。)

**一句话:** `mlp_unfused` 是三段分开的计算(matmul1 → GeLU → matmul2),中间的隐藏层激活值 `h`(形状 `N×H`)要在段与段之间完整写入 HBM 再读出来;`mlp_fused` 把这三步塞进同一层循环,每一行的 `h_row` 只活在 Python 局部变量里(对应真实 kernel 里的寄存器/SRAM),算完立刻被下一步的 matmul2 消费掉,`h` 作为一个 N×H 大小的完整矩阵**从未在 HBM 上存在过**。

**底层机制/为什么这样设计:** 先问一个最笨的问题——如果 matmul1、GeLU、matmul2 是三个互相不知道对方存在的独立 kernel,中间结果要怎么才能从第一个"传"到第二个、第三个?唯一的办法是:第一个 kernel 把结果放到一个大家都能访问的地方(HBM),下一个 kernel 从那里读、算完再放回去,再下一个 kernel 再读一次——这正是 `mlp_unfused` 这个名字里"unfused"(没有融合)的字面意思:三段计算互相独立,只能靠 HBM 传数据。回到 `mlp_unfused` 的三行代码,看它们怎么对应"三次 HBM 往返"——第一行 `h=[[sum(...)...` 算出完整的 `N×H` 矩阵 `h`(matmul1 的结果);第二行 `h2=[[gelu(v)...` 单独对 `h` 的每个元素做逐点的 GeLU;第三行才用 `h2` 去做 matmul2。如果这是三个分开的 kernel(或者用某个高层框架不自动做 fusion 时天然产生的三次 kernel launch),`h` 要先被 matmul1 的 kernel 写回 HBM,GeLU 的 kernel 才能读到它、算完再写回一份 `h2`,matmul2 的 kernel 再读一次 `h2`——这中间"写 h、读 h(算 gelu)、写 h2、读 h2"的过程,恰好是 `hbm_traffic()` 公式里 `2*dtype_bytes*N*H` 这一项的来源(这里的代码把 GeLU 这一步在计数时和它的输入/输出合并处理,只用一次"写+读"共 `2×N×H` 计量,是这个教学脚本对"中间激活值可能被多算几次逐点 kernel"这一细节的简化,核心结论——多出来一次完整的 `N×H` 矩阵往返——是成立的)。

`mlp_fused` 的写法直接在源码层面消灭了这个中间产物:`h_row=[gelu(sum(...)) for k in range(H)]` 把"matmul1 的这一行结果"和"这一行的 GeLU"写在同一个列表推导式里,算出来的 `h_row` 只是一个长度为 `H` 的 Python list(对应真实 kernel 里一小片寄存器/SRAM),没有为完整的 `N×H` 矩阵单独分配存储、更没有把它整体写回过 HBM——它算完这一行立刻被内层循环拿去做 `out[i][d_out]=sum(h_row[k]*W2[k][d_out]...)`,用完即弃。**Fusion 省的不是计算,而是"中间结果要不要在 HBM 上完整现身一次"这个选择**,这和知识点 2 里 naive/flash attention 的 `S`/`P` 矩阵是不是被 materialize,是完全同构的问题。

**关键验证结论(今天重新独立复验):** `_self_test()` 用 `N=2048, D=4096, H=16384`(典型 Transformer FFN 规模,`H` 是 4 倍展开的隐藏维)跑出来的节省比例是 **30.8%**(精确值 `30.76923076923077%`,是一个确定性计算结果,不是随机 benchmark 的波动数字)。手动核对省下来的字节数,恰好等于 `2*dtype_bytes*N*H`(隐藏层的完整一次写+一次读),和公式定义完全吻合。

**AI 研究/工程场景:** 真实 Transformer 的 FFN 子层(`GeLU(x@W1)@W2`,或者更常见的 SwiGLU 变体)是逐 token 重复执行次数最多的模块之一,`H`(隐藏维)通常是 `D`(模型维)的 4 倍甚至更多,`fused_mlp.py` 里选的 `H=16384, D=4096` 正是这个"4 倍展开"的真实比例。这也是为什么 xFormers、FlashAttention 官方仓库、Triton 教程里都会提供"fused bias+GeLU""fused MLP"这类 kernel——把逐点激活函数焊死在紧邻的 matmul 里,是除了 attention 本身之外,transformer 推理/训练 kernel 优化最常见的第二类目标。`learning/kernel-engineering/lectures/05-fusion-patterns.md` 把这类"pointwise → matmul"的融合归成一整类模式,并指出更复杂的 SwiGLU(3 个 matmul)通常只能融合出"matmul+elementwise+matmul"这一段,不会把 3 个 matmul 全部揉进一个 kernel——这是一个值得知道的边界,不在这份代码要实现的范围内,这里只做提及。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from fused_mlp import hbm_traffic, mlp_unfused, mlp_fused
import random

# 数值等价性:fused 和 unfused 应该算出同一个结果(fusion 不改数学)
random.seed(13)
N, D, H = 4, 6, 12
x = [[random.random() for _ in range(D)] for _ in range(N)]
W1 = [[random.random() for _ in range(H)] for _ in range(D)]
W2 = [[random.random() for _ in range(D)] for _ in range(H)]
a = mlp_unfused(x, W1, W2)
b = mlp_fused(x, W1, W2)
max_diff = max(abs(a[i][d] - b[i][d]) for i in range(N) for d in range(D))
assert max_diff == 0.0   # 位级相等:两条路径的规约都用内置 sum(),运算顺序完全一致(对照知识点 4)

# HBM 流量核算:典型 FFN 规模 N=2048(batch*seq), D=4096(model dim), H=16384(4x 展开)
N, D, H = 2048, 4096, 16384
unfused_bytes = hbm_traffic(N, D, H, fused=False)
fused_bytes = hbm_traffic(N, D, H, fused=True)

manual_weights = 2 * (D * H + H * D)
manual_io = 2 * (N * D + N * D)
manual_fused = manual_weights + manual_io
manual_unfused = manual_fused + 2 * 2 * N * H
assert fused_bytes == manual_fused
assert unfused_bytes == manual_unfused

savings = (unfused_bytes - fused_bytes) / unfused_bytes
assert round(savings * 100, 1) == 30.8

# 省下来的正好是隐藏层 h 的一次完整写 + 一次完整读
saved_bytes = unfused_bytes - fused_bytes
assert saved_bytes == 2 * 2 * N * H
```

**面试怎么问 + 追问链:**
- **Q:** "`GeLU(x@W1)@W2` 这种两层 MLP,fuse 成一个 kernel 具体省的是哪一部分访存?" —— 期望明确答出"省的是中间隐藏层激活值 `h`(形状 `N×H`)的一次完整写入+读出",而不是笼统地说"省了访存"。
- **追问 1:** "为什么这个例子选 `H=16384, D=4096`,而不是两者相等?这个比例对收益有什么影响?" —— 期望能结合公式推导:省下来的固定是 `2*N*H`,分母(unfused 总流量)里还包含和 `D*H` 相关的权重项,`H` 相对 `D` 越大,`2*N*H` 这一项在总流量里的占比通常也越明显。
- **追问 2(深挖):** "如果 batch×seq(也就是 `N`)特别小,比如单 token 推理,fusion 还值得做吗?" —— 期望答"仍然值得,因为省下来的 `2*N*H` 和权重读取量 `4*D*H` 相比,`N` 很小时权重读取反而成了流量大头(推理阶段经典的'weight-bound'场景),这时候减少权重的重复读取(比如让多个请求共享一次权重读取)比单纯 fusion 更关键"——这道题在考察候选人是否能跳出"fusion 总是好的"这种绝对化认知。
- **追问 3:** "SwiGLU 有 3 个 matmul(gate、up、down)和 2 个逐点操作,能不能把 3 个 matmul 也融合成 1 个 kernel?" —— 期望答"通常不会把 3 个 matmul 完全揉进一个 kernel,因为中间维度太大,过多中间结果要么撑爆 SMEM,要么退化成和不融合差不多的分块重复读写;实践中常见的是融合出'matmul+elementwise+matmul'这一段,而不是三个 matmul 全部融合"——能答到这一层说明理解了"fusion 不是越多越好,要看中间结果能不能真的塞进片上存储"这条边界。

**常见坑:**
- 把 fusion 的收益理解成"减少了计算量(FLOPs)"——`mlp_fused` 和 `mlp_unfused` 做的浮点乘加次数完全一样(源码验证过数值位级相等),fusion 省的完完全全是访存,不是算力。
- 以为 `hbm_traffic()` 里的 `weights` 项(`W1`+`W2` 的读取)在 fused/unfused 之间也会有差异——两条路径都只读一次权重(见公式,`weights` 项在 `if fused` 分支前就已经算好,两个分支共享),差异只体现在隐藏层 `h` 这一项,权重读取量本身不受 fusion 影响。
- 把这次的"30.8%"当成一个"kernel fusion 通用的收益幅度"记下来——这个百分比是 `N=2048,D=4096,H=16384` 这一组具体形状算出来的确定性结果,换一组 `N/D/H` 比例,收益百分比会随之变化(可以现场用 `hbm_traffic()` 代入不同参数验证),不存在一个放之四海而皆准的"融合能省 30% 流量"的经验数字。

---

## 4. RMSNorm + Linear Fusion(`rmsnorm_kernel.py`)—— "融合不改数学,只改访存"的具体例证

**是什么:**
```python
def rmsnorm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """y = x / rms(x) * weight, where rms(x) = sqrt(mean(x²) + eps)."""
    n = len(x)
    s = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(s + eps)
    return [x[i] * inv * weight[i] for i in range(n)]


def fused_rmsnorm_linear(X: list[list[float]], norm_weight: list[float],
                          W: list[list[float]]) -> list[list[float]]:
    """Fuse RMSNorm + linear: never materialize normalized intermediate."""
    N, D = len(X), len(X[0])
    out_d = len(W[0])
    out = [[0.0] * out_d for _ in range(N)]
    for i in range(N):
        s = sum(v * v for v in X[i]) / D
        inv = 1.0 / math.sqrt(s + 1e-6)
        for j in range(out_d):
            acc = 0.0
            for d in range(D):
                acc += X[i][d] * inv * norm_weight[d] * W[d][j]
            out[i][j] = acc
    return out
```
(源码见 `learning/kernel-engineering/src/rmsnorm_kernel.py:6-32`。)

**一句话:** unfused 路径先算出完整的归一化结果 `norm=rmsnorm_batch(X,weight)`(一个 `N×D` 的中间矩阵),再拿它去做矩阵乘法;`fused_rmsnorm_linear` 把"除以 RMS、乘 `weight`、乘 `W` 做矩阵乘法累加"这几步写进同一层循环,归一化后的值只以标量 `X[i][d]*inv*norm_weight[d]` 的形式短暂出现在一次乘法链里,从未攒成一个完整的 `N×D` 矩阵——这和知识点 3 是同一个"别把中间结果落地"的原则,只是这次的重点是验证"融合前后数学结果是否真的一样"。

**底层机制/为什么这样设计:** 先问一个最笨的问题——为什么 RMSNorm 和 Linear 能融合,但两个普通的 matmul(比如 `A@B@C`)通常不能像这样轻松融合?差别在于要不要"跨行看数据"。RMSNorm 本身是逐行独立的操作(`rmsnorm(x,weight)` 只依赖这一行自己的数据,不需要跨行通信),这个性质决定了它天然适合和后面的 Linear 层融合:算完第 `i` 行的 `inv`(RMS 的倒数)之后,这一行需要的全部信息就集齐了,可以直接把"归一化"和"矩阵乘法的这一行"焊在同一个循环体里算完,不需要等其他行、也不需要一个完整的中间矩阵撑住"归一化"和"矩阵乘法"这两个阶段之间的边界。

这里最重要的验证不是"融合更省访存"(知识点 3 已经讲透 fusion 省访存的机制),而是"融合前后,算出来的是不是同一个数学函数"——`_self_test()` 用 `abs(Y1[i][j]-Y2[i][j])<1e-9` 断言两者足够接近,但这个阈值本身留了一个问题:**真实差异到底有多大,这个 `1e-9` 是不是过于宽松、掩盖了什么?** 今天重新独立跑出来的真实数字是:两条路径的最大绝对差是 `8.88e-16`——比 `1e-9` 的阈值小了整整 7 个数量级,说明这条断言虽然能过,但用来"验证融合数学正确"这件事上留了很大的松弛空间,没有卡在真实精度的边界上。

进一步深挖这 `8.88e-16` 到底从哪来,能发现一个和"fusion 本身"完全无关的真实原因:**unfused 路径的矩阵乘法用的是 Python 内置 `sum()`,而 `fused_rmsnorm_linear` 里用的是手写的 `acc += ...` 循环。** Python 3.12 起,`sum()` 对浮点数序列内部使用了 Neumaier(改进版 Kahan)补偿求和算法,精度显著高于逐项直接累加;仓库这次用的 `.venv` 是 Python 3.13.9,已经带有这个特性。把两条路径涉及的 8 个乘积项单独拿出来验证:手写循环累加的结果和 `math.fsum`(完全补偿求和,数学上最接近真值的 ground truth)有 `8.88e-16` 的差距,而内置 `sum()` 算出来的结果和 `math.fsum` **位级完全相等**——也就是说,这个差异不是"fusion 引入了近似",而是"两条路径恰好选用了两种精度不同的求和算法";如果把 `fused_rmsnorm_linear` 内部的手写循环换成 `sum(...)`,这个差异会直接归零(下面的可运行例子现场验证了这个替换)。这是一个比"floating point 不满足结合律,所以肯定有点误差"更精确、更能追根溯源的结论,也呼应了知识点 3 结尾提过的"不要把每一处数值差异都笼统归咎于'浮点误差',具体差多少、差在哪一步,是可以现场查清楚的"。

**AI 研究/工程场景:** LLaMA 系列开始,RMSNorm 取代 LayerNorm 成为主流 transformer 的归一化层,几乎每个 transformer block 里都会出现"RMSNorm 接一个 Linear(QKV 投影或者 FFN 的第一层)"这个模式,真实推理框架(vLLM、TensorRT-LLM 一类)会把 RMSNorm 的 rescale 操作直接揉进紧邻的 GEMM kernel 的 prologue(前处理)里,原因和这里验证的一致:norm 之后的中间结果如果不融合,要以 `N×D` 矩阵的形式完整走一次 HBM 往返,而 `D` 在大模型里动辄几千到上万,这一次往返在整个前向传播的访存预算里不是小数目。这里验证的"融合后数值等价"这件事,在生产环境中同样是新写一个 fused kernel 时必须补的正确性测试项——不能只测"跑得通、跑得快",数值一致性(哪怕允许 ULP 级别的浮点差异,也要能解释差异来自哪里)是能不能把这个 kernel 换上生产环境的前提。

**可运行例子:**
```python
import sys, math
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from rmsnorm_kernel import rmsnorm, rmsnorm_batch, fused_rmsnorm_linear
import random

# Part A: 单行 sanity check,手算 rms 核对公式
y = rmsnorm([1, 2, 3, 4], [1, 1, 1, 1])
rms = math.sqrt((1 + 4 + 9 + 16) / 4)
assert abs(y[0] - 1 / rms) < 1e-6

# Part B: fused vs unfused 数值等价性,拿到真实 diff(不是只看 <1e-9 通过)
random.seed(11)
N, D, OD = 3, 8, 5
X = [[random.random() for _ in range(D)] for _ in range(N)]
w = [1.0] * D
W = [[random.random() for _ in range(OD)] for _ in range(D)]

norm = rmsnorm_batch(X, w)
Y1 = [[sum(norm[i][d] * W[d][j] for d in range(D)) for j in range(OD)] for i in range(N)]
Y2 = fused_rmsnorm_linear(X, w, W)
max_diff = max(abs(Y1[i][j] - Y2[i][j]) for i in range(N) for j in range(OD))
assert 0 < max_diff < 1e-9    # 确实有微小差异,但比 self-test 的 1e-9 阈值紧了 7 个数量级

# Part C: 揪出这个差异的真正来源 —— 不是 fusion 改了数学,是 sum() 内置补偿求和 vs 手写循环
i, j = 0, 4    # 已定位这是最大的一组 diff
s = sum(v * v for v in X[i]) / D
inv = 1.0 / math.sqrt(s + 1e-6)
terms = [X[i][d] * inv * w[d] * W[d][j] for d in range(D)]

acc = 0.0
for t in terms:
    acc += t                        # 手写循环,复现 fused_rmsnorm_linear 内部真正在做的事
via_sum = sum(terms)                # unfused 路径实际调用的内置 sum()
truth = math.fsum(terms)            # 完全补偿求和,数学上最接近真值的 ground truth

assert acc == Y2[i][j]
assert via_sum == Y1[i][j] == truth    # 内置 sum() 和 fsum 完全一致(Python 3.12+ 的 Neumaier 补偿求和)
assert acc != via_sum                   # 同样的 8 个 term,只是换了个"怎么加总"的算法,结果就不再位级相等
```

**面试怎么问 + 追问链:**
- **Q:** "RMSNorm 后面接一个 Linear 层,融合成一个 kernel,数学上的输出应该和分开算完全相同吗?" —— 期望第一反应答"应该相同(同一个公式),但浮点运算不满足结合律,实践中可能有极小的数值差异",而不是想当然认为"融合就是纯粹的性能优化,数值上绝对零差异"。
- **追问 1:** "如果我告诉你实测确实有大约 `1e-15` 量级的差异,这说明 fusion 这个优化不安全吗?" —— 期望答"不能这么下结论,要先查清楚这个差异具体来自哪一步";这道题在考察候选人遇到"数值对不上"时,第一反应是"归咎于笼统的浮点误差就算了",还是"愿意继续往下查"。
- **追问 2(深挖,本知识点验证时独立发现的真实案例):** "你会怎么验证这个差异到底是算法本身导致的,还是别的原因?" —— 期望能提出类似"把两条路径的中间结果拆开,逐项对比""换一种求和方式看看差异会不会变化"这类系统性排查方法;可以现场引导到"发现是 Python 内置 `sum()` 用了比手写循环更精确的补偿求和算法,把手写循环换成 `sum()` 之后两者位级相等"这个具体结论,展示排查过程比背答案更重要。
- **追问 3:** "生产环境的 fused RMSNorm+Linear kernel,验证正确性的时候,应该用什么样的容差(tolerance)?" —— 期望答"不能拍脑袋定一个'看起来够小'的数字(比如这里源码用的 `1e-9` 其实比实际达到的精度松了 7 个数量级),更严谨的做法是理解清楚数值差异的来源(比如求和顺序、是否使用补偿求和)之后,把容差定在'能容纳这类已知、良性的浮点误差来源,但足够紧,能抓出真正的算法性错误'这个区间"。

**常见坑:**
- 看到两条路径输出"约等于但不完全相等",就直接归因于"浮点数误差,正常现象",不再深究——这次验证展示了这类差异往往有精确、可查明的具体来源(这里是"两种求和算法精度不同"),盲目归因会错过真正有价值的发现,也可能在差异来源其实是算法性 bug 时被误判为"无害的浮点误差"而放过真正的问题。
- 把 `_self_test()` 里 `<1e-9` 这类断言的通过,理解成"精度已经验证到 `1e-9` 这个量级"——阈值只是"不超过这个上限",不代表实际精度就在这个量级附近;这次实测的真实精度(`8.88e-16`)比阈值紧了 7 个数量级,阈值本身是一个相当宽松的上界,不是精确刻画。
- 想当然认为"手写循环"和"调用内置 `sum()`"在数值上必然等价——两者在数学上算的是同一个和式,但既是同一个和式,浮点计算机上的实现细节(尤其是 Python 3.12+ 给 `sum()` 加上的补偿求和优化)会让二者出现真实、可复现、可解释的位级差异,这是"同一个数学公式,不同的浮点实现路径,结果不保证位级相同"这条更普遍原则的一个具体例子。

---

## 5. Capstone:128k 序列的 HBM 节省曲线(`capstone_attn_speedup.py`)—— O(N²) vs O(N·d) 的渐近关系

**是什么:**
```python
def speedup_curve(d: int = 128) -> list[dict]:
    rows = []
    for N in [512, 2048, 8192, 32768, 131072]:
        n_b = hbm_naive_attn(N, d)
        f_b = hbm_flash_attn(N, d)
        rows.append({
            "seq_len": N,
            "naive_mb": round(n_b / 1e6, 1),
            "flash_mb": round(f_b / 1e6, 1),
            "speedup": round(n_b / f_b, 1),
        })
    return rows
```
(源码见 `learning/kernel-engineering/src/capstone_attn_speedup.py:21-32`,复用的正是知识点 2 讲过的 `hbm_naive_attn`/`hbm_flash_attn` 这两个函数,`d` 固定在 128,只让序列长度 `N` 变化。)

**一句话:** 把知识点 2 的两个流量公式应用到 5 个从 512 到 131072(128k)依次翻约 4 倍的序列长度上,画出一条"节省倍数"曲线——因为 naive 的流量里有 `O(N²)` 项、flash 的流量整体是 `O(N·d)`,序列越长这两条曲线拉开的差距越夸张,不是等比例增长,是随 `N` 近似线性、但斜率由 `1/d` 决定的增长。

**底层机制/为什么这样设计:** 先问一个最笨的问题——不看任何表格,能不能只靠知识点 2 的两个公式,自己推出"序列越长优势越大"这个结论?直接把知识点 2 验证过的公式展开:`speedup(N,d) = naive/flash = (4Nd+4N²)/(4Nd) = 1+N/d`。这是一个关于 `N` 的线性函数,但斜率是 `1/d`——`d`(每个 attention head 的维度,这里固定 128)决定了这条线涨得多快。这个公式解释了为什么表格最后一行(`N=131072`)的加速比会跳到千倍量级:`N/d=131072/128=1024`,加上常数项 `1`,正好是 `1025`。**这不是一个拍脑袋的"大模型都很夸张"式结论,是把知识点 2 的流量公式代入具体数字算出来的确定性结果**,今天重新独立跑出来的完整表格如下(`.venv` 实测,和 `00-roadmap.md` 里记录的今天早些时候的验证结果一致,没有出现回归):

| 序列长度 N | Naive (MB) | Flash (MB) | Speedup |
|---|---|---|---|
| 512 | 2.6 | 0.5 | 5.0x |
| 2048 | 35.7 | 2.1 | 17.0x |
| 8192 | 545.3 | 8.4 | 65.0x |
| 32768 | 8623.5 | 33.6 | 257.0x |
| 131072 | 137573.2 | 134.2 | 1025.0x |

值得留意 flash 那一列(`flash_mb`)几乎不随 `N` "爆炸式"增长——从 512 到 131072(256 倍),`flash_mb` 只从 0.5MB 涨到 134.2MB(约 256 倍,线性关系,符合 `O(N·d)`);而 `naive_mb` 从 2.6MB 涨到 137573.2MB(约 5 万倍,符合 `O(N²)`,`256²≈65536`,量级吻合)。两条曲线增长速度的本质差异,就是 `speedup` 一路飙升到 1025x 的根本原因。

**AI 研究/工程场景:** `learning/kernel-engineering/lectures/04-flashattention.md` 把这个结论总结成一句话:"长序列训练成为可能(32k → 128k → 1M)"——如果没有把 attention 从 `O(N²)` 的 HBM 流量降到 `O(N·d)`,行业不可能把上下文长度从早期的 2k/4k 一路推到今天 128k 甚至百万级别,单纯堆更多算力和更大带宽的硬件也追不上 `N²` 的增长速度。但这条曲线只解决了"attention 计算本身"的访存问题,不代表长上下文的显存压力被完全解决——同一份 lecture note 也点出"KV cache 仍是 O(N) 容量瓶颈",存 KV cache 本身(不是"访问"它产生的流量,是"装下"它需要的显存容量)仍随序列长度线性增长,后续需要 PagedAttention、MLA(Multi-head Latent Attention)这类技术接力解决——这里只点出这层关联,不展开(不在本篇/这几个源文件的范围内)。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from capstone_attn_speedup import hbm_naive_attn, hbm_flash_attn, speedup_curve

rows = speedup_curve()   # d=128 默认值
expected = [
    {"seq_len": 512,    "naive_mb": 2.6,      "flash_mb": 0.5,   "speedup": 5.0},
    {"seq_len": 2048,   "naive_mb": 35.7,     "flash_mb": 2.1,   "speedup": 17.0},
    {"seq_len": 8192,   "naive_mb": 545.3,    "flash_mb": 8.4,   "speedup": 65.0},
    {"seq_len": 32768,  "naive_mb": 8623.5,   "flash_mb": 33.6,  "speedup": 257.0},
    {"seq_len": 131072, "naive_mb": 137573.2, "flash_mb": 134.2, "speedup": 1025.0},
]
assert rows == expected

speedups = [r["speedup"] for r in rows]
assert speedups == sorted(speedups)    # 单调递增:序列越长,优势越夸张
assert speedups[-1] == 1025.0

# 渐近关系验证:speedup ≈ 1 + N/d,不是拍脑袋的经验数字
N, d = 131072, 128
assert round(hbm_naive_attn(N, d) / hbm_flash_attn(N, d), 1) == round(1 + N / d, 1) == 1025.0

# 换一个 d 验证同一条公式(面试追问会问:d 减半,speedup 会怎样变化?)
assert round(hbm_naive_attn(N, 64) / hbm_flash_attn(N, 64), 1) == 2049.0   # 约等于翻倍
```

**面试怎么问 + 追问链:**
- **Q:** "为什么序列长度从 512 涨到 128k(256 倍),HBM 流量的节省倍数会从 5x 一路涨到 1025x,不是等比例涨的吗?" —— 期望第一句话点出"naive 流量里有 `N²` 项,flash 是 `N` 的线性项,两者增长速度不同,比值当然不是等比例"。
- **追问 1:** "能不能写出这个 speedup 关于 N、d 的封闭形式公式?" —— 期望能现场从 `4Nd+4N²` 和 `4Nd` 推出 `speedup=1+N/d`,而不是只记得表格里的具体数字。
- **追问 2(深挖):** "如果 attention head 的维度 `d` 从 128 降到 64(比如换一种切分 head 的方式),同样 128k 序列的 speedup 会变成多少?" —— 期望能用 `1+N/d` 现场估算出约等于 `1+131072/64=2049`,接近翻倍;这道题在检验候选人是不是真的理解了公式,还是只背了"128k → 1025x"这一个具体数字。
- **追问 3(工程延伸):** "所以只要上了 FlashAttention,128k 上下文训练就没有显存问题了吗?" —— 期望答"不是,这条曲线只覆盖 attention 计算本身的 HBM 访存量,推理时的 KV cache 存储本身仍然是 `O(N)` 随序列线性增长的显存占用,是另一个需要 PagedAttention/MLA 之类技术解决的问题,不要把'访存流量被优化'和'显存占用被优化'混为一谈"。

**常见坑:**
- 把"1025x"当成一个和具体 `N`、`d` 无关的"FlashAttention 通用加速倍数"记下来到处套用——这个数字只在 `N=131072, d=128` 这一组具体参数下成立,换一组参数(尤其是 `d`)结果会明显不同,上面的追问 2 已经现场验证过。
- 把这条曲线的"speedup"理解成"运行时间快了多少倍"——`speedup_curve()` 算的完全是 HBM **字节流量**的比值,不是实测 wall-clock 时间;真实 kernel 的实际加速比还要受计算量、并行度、kernel launch 开销等因素影响,流量比值是一个上限性质的理论指标,不能直接等价于实测性能提升的倍数。
- 误以为这条曲线证明了"长上下文训练的显存问题已经被 FlashAttention 完全解决"——见追问 3,KV cache 的存储容量问题不在这条曲线覆盖的范围内,这是很多初学者容易过度引申的一个结论。

---
