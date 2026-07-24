# 06 · 手把手实战:搭一个迷你 Roofline 计算器 + 一个简化版 Online Softmax

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 6 个"知识点",不计入"19 个知识点"的统计——和 [05 类](05-advanced-interview-depth.md)是同一挂"不计入正式知识点统计的额外内容",但风格完全不一样:05 号文件里,你是**旁观者**,跟着调研出来的追问链把几个真实案例的推理过程看一遍;这一篇里,你是**动手的人**——从空文件开始,一步步敲代码,每写一段就跑一次、看到真实输出,最后独立搭出两个真实能用的小工具。这个"教程体"格式最早在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证过,这一篇是把它推广到本系列的第一次尝试。

## 为什么是"Roofline 计算器 + Online Softmax"

不是要发明新知识点,是把 [02 类](02-roofline-model.md)、[04 类](04-flashattention-and-fusion.md)里已经验证过的公式,自己重新手写一遍、拼成两个真实能跑的小工具。选这两个知识点组队不是随便拼凑:Roofline 计算器回答的是"一个算子的瓶颈在哪",Online Softmax 回答的是"怎么把 FlashAttention 这类算子的瓶颈往好的方向搬"——阶段 5 会把两者接在一起,亲手用计算器称出 Online Softmax 到底把瓶颈搬动了多少,这不是巧合,是刻意设计的收尾。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 给定 flops、访存字节数、硬件峰值算力/带宽,算出 arithmetic intensity 和 ridge point,判断计算瓶颈还是内存瓶颈 | [02 类知识点 1-2](02-roofline-model.md) AI 与 ridge point 公式 |
| 阶段 2 | 把计算器套在真实算子上(向量加法 vs GEMV vs 大矩阵乘法),在多块 GPU 上验证"向量加法总是内存瓶颈、大矩阵乘法总是计算瓶颈"这个直觉 | 阶段 1 + [02 类知识点 1](02-roofline-model.md) 的 GEMM 公式 |
| 阶段 3 | 输入一组数值,分块处理(不需要一次性拿到全部数据),用 running max / running 分母递推正确算出 softmax 归一化结果 | [04 类知识点 1](04-flashattention-and-fusion.md) online softmax 递推 |
| 阶段 4 | 给 online softmax 配上按 V 加权的输出累加,组装成"单行 mini flash attention",和"先材料化再算"的参照组数值比对 | 阶段 3 + [04 类知识点 1](04-flashattention-and-fusion.md) 的 m/l/o 三步递推 |
| 阶段 5 | 用阶段 1 的计算器,分析阶段 3-4 省下的访存量到底有多大意义——同一个 attention 算子,naive 方式核算的字节数 vs flash 方式核算的字节数代入计算器,直接看到 roofline 分类翻转 | 阶段 1-2 + [04 类知识点 2](04-flashattention-and-fusion.md) HBM 流量公式,两部分在这里真正"组装"到一起 |

每个阶段的代码都能独立运行(本文件用仓库统一的 `_verify_md.py` 校验方式:每个 ` ```python ` 代码块单独起一个新的 Python 子进程执行,块与块之间不共享任何变量,后面阶段用到前面阶段的函数会重新贴一遍,不是偷懒复制)。

**这篇教程和 02/04 号文件的一个重要区别:** 02/04 号文件里的可运行例子用 `sys.path.insert` 导入 `learning/gpu-architecture/src/`、`learning/kernel-engineering/src/` 下的真实源码(比如 `roofline.py`、`flash_attention.py`);这一篇不这样做——"手写"的重点是把公式自己重新敲一遍来验证是否真的理解了,而不是导入别人写好的实现跑一遍就算数。下面所有函数都是全新独立写的,但公式和数值特意和 02/04 号文件里已经验证过的结果逐条交叉核对,确保一致,不是另起炉灶编一套可能对不上的版本。全部代码是纯 Python 标准库(只用到 `math`、`random`),刻意不引入 numpy——`roofline.py`/`flash_attention.py` 这两个被参照的源模块自己也是纯标准库实现,这里选择保持一致,继续贯彻 [00-roadmap.md](00-roadmap.md)"全系列纯 CPU、零第三方依赖"的环境声明。

---

## 阶段 1:Roofline 计算器的内核——一个比值 + 一条分界线

[02 类知识点 1-2](02-roofline-model.md) 已经把公式讲透了:Arithmetic Intensity(AI)= flops / bytes_moved,Ridge Point = 峰值算力 / 峰值带宽,AI 和 ridge point 谁大谁小决定一个算子是计算瓶颈还是内存瓶颈。这一步要做的事很直接:把这几行公式真正敲成三个函数,而不是停留在"看懂了"这个感觉上。

```python
def arithmetic_intensity(flops, bytes_moved):
    """AI = 总浮点运算次数 / 总内存搬运字节数，单位 FLOP/byte。"""
    return flops / bytes_moved

def ridge_point(peak_flops_per_s, peak_bytes_per_s):
    """两条生产线的交叉点：算力峰值 / 带宽峰值，单位 FLOP/byte。"""
    return peak_flops_per_s / peak_bytes_per_s

def classify(ai, ridge):
    """AI 落在 ridge 左边（更小）是内存瓶颈，右边（大于等于）是计算瓶颈——
    这里用 >= 而不是 >，和 02-roofline-model.md 的 analyze() 用同一个约定。"""
    return "compute" if ai >= ridge else "memory"

# 一个假想的最小算子：flops=100，bytes=50，AI 应该是 2.0
assert arithmetic_intensity(100, 50) == 2.0

# 用 H100 的真实规格交叉核对：02-roofline-model.md 知识点2已经验证过 ridge point = 295.2 FLOP/byte
h100_peak_flops = 989.0e12   # bf16 峰值算力，989 TFLOPS
h100_peak_bw = 3.35e12       # 峰值显存带宽，3.35 TB/s
rp_h100 = ridge_point(h100_peak_flops, h100_peak_bw)
assert round(rp_h100, 1) == 295.2

# 恰好卡在 ridge point 上、以及左右各偏移一点，三种情况分类都要正确
assert classify(rp_h100, rp_h100) == "compute"
assert classify(rp_h100 - 1, rp_h100) == "memory"
assert classify(rp_h100 + 1, rp_h100) == "compute"

print("stage1 H100 ridge point:", round(rp_h100, 1), "FLOP/byte")
print("stage1 ok")
```

三个 assert 分别验证:①一个最简单的手算案例(flops=100,bytes=50 应该得到 AI=2.0);②用 H100 的真实规格(989 TFLOPS、3.35 TB/s)算出的 ridge point 是不是精确复现了 [02 类知识点 2](02-roofline-model.md) 已经验证过的 295.2 FLOP/byte——这一步是把自己的实现拿去对照已经验证过的真实数字,不是自说自话;③ridge point 左边、右边、恰好压线三种情况分类是否正确,其中"恰好压线"用的是 `>=`(判给 compute),这个边界约定和 `roofline.py::analyze()` 完全一致,不是随手挑的,后面阶段的分类结果要能对上,边界约定必须先统一。

## 阶段 2:把计算器套在真实算子上——向量加法 vs GEMV vs 大矩阵乘法

阶段 1 的计算器还只是三个孤立的函数,没有真正喂过任何算子。这一步选三个直觉上就该有明显差异的算子代入:向量加法(最典型的"读得多、算得少"场景)、GEMV([02 类知识点 1](02-roofline-model.md) 已经验证过的 LLM 推理 decode 阶段单 token 形状)、大矩阵乘法(训练阶段最常见的大 GEMM)。GEMM 的 flops/bytes 公式直接沿用 [02 类知识点 1](02-roofline-model.md) 里 `gemm_profile` 的约定(`flops=2*m*n*k`,`bytes_moved=dtype_bytes*(m*k+k*n+m*n)`);向量加法是这篇教程新引入的算子,公式自己推:`c[i]=a[i]+b[i]` 对每个元素做 1 次加法,同时要读 `a`、读 `b`、写 `c` 三个长度为 n 的数组,所以 `flops=n`,`bytes_moved=dtype_bytes*3*n`。

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved

def ridge_point(peak_flops_per_s, peak_bytes_per_s):
    return peak_flops_per_s / peak_bytes_per_s

def classify(ai, ridge):
    return "compute" if ai >= ridge else "memory"

def gemm_flops_bytes(m, n, k, dtype_bytes=2):
    """和 02-roofline-model.md 的 gemm_profile 同一套约定：
    flops=2*m*n*k（乘加各算一次），bytes_moved=dtype_bytes*(A+B+C 三个矩阵各读/写一次)。"""
    flops = 2 * m * n * k
    bytes_moved = dtype_bytes * (m * k + k * n + m * n)
    return flops, bytes_moved

def vector_add_flops_bytes(n, dtype_bytes=2):
    """c[i]=a[i]+b[i]：n 次加法，读 a、读 b、写 c 各 n 个元素。"""
    flops = n
    bytes_moved = dtype_bytes * 3 * n
    return flops, bytes_moved

# 两块真实GPU的规格（learning/gpu-architecture/src/common.py::GPUS，02-roofline-model.md已验证）
GPU_SPECS = {
    "A100": (312.0e12, 2.039e12),
    "H100": (989.0e12, 3.35e12),
}

OPS = {
    "vector_add(n=1e6,fp16)": vector_add_flops_bytes(1_000_000),
    "GEMV(1x4096x4096)": gemm_flops_bytes(1, 4096, 4096),
    "GEMM(4096^3)": gemm_flops_bytes(4096, 4096, 4096),
}

results = {}
for gpu_name, (peak_flops, peak_bw) in GPU_SPECS.items():
    ridge = ridge_point(peak_flops, peak_bw)
    for op_name, (flops, bytes_moved) in OPS.items():
        ai = arithmetic_intensity(flops, bytes_moved)
        bound = classify(ai, ridge)
        results[(gpu_name, op_name)] = (ai, bound)
        print(f"{gpu_name} | {op_name:24s} AI={ai:10.4f}  bound={bound}")

# 向量加法：不管哪块GPU，AI都远低于ridge point，稳定是内存瓶颈——符合直觉
assert results[("A100", "vector_add(n=1e6,fp16)")][1] == "memory"
assert results[("H100", "vector_add(n=1e6,fp16)")][1] == "memory"

# 大矩阵乘法：不管哪块GPU，AI都远高于ridge point，稳定是计算瓶颈——同样符合直觉
assert results[("A100", "GEMM(4096^3)")][1] == "compute"
assert results[("H100", "GEMM(4096^3)")][1] == "compute"

# GEMV（decode单token的形状）：AI≈1.0，同样稳定是内存瓶颈
assert results[("A100", "GEMV(1x4096x4096)")][1] == "memory"
assert results[("H100", "GEMV(1x4096x4096)")][1] == "memory"

# 交叉核对02-roofline-model.md已经验证过的具体数字，确认没有回归
assert round(results[("H100", "GEMM(4096^3)")][0], 2) == 1365.33
assert round(results[("H100", "GEMV(1x4096x4096)")][0], 2) == 1.0
assert round(results[("H100", "vector_add(n=1e6,fp16)")][0], 4) == 0.1667

# 换成fp32（dtype_bytes=4），同一个向量加法，字节数翻倍，AI应该精确减半
fp32_flops, fp32_bytes = vector_add_flops_bytes(1_000_000, dtype_bytes=4)
ai_fp32 = arithmetic_intensity(fp32_flops, fp32_bytes)
ai_fp16 = results[("H100", "vector_add(n=1e6,fp16)")][0]
assert round(ai_fp16 / ai_fp32, 6) == 2.0
assert classify(ai_fp32, ridge_point(*GPU_SPECS["H100"])) == "memory"   # 精度变了，结论没变：AI差ridge point太远，翻不了盘

print("stage2 ok")
```

本机实测输出(节选):`H100 | vector_add(n=1e6,fp16)   AI=    0.1667  bound=memory`、`H100 | GEMM(4096^3)             AI= 1365.3333  bound=compute`——向量加法和大矩阵乘法在两块 GPU 上的分类结果完全稳定,不随 GPU 变化,这不是巧合:向量加法的 AI(0.1667)和大矩阵乘法的 AI(1365.33)分别比 A100/H100 的 ridge point(153.0/295.2)低了三个数量级、高了近一个数量级,差距太大,换成任何一块量级相近的现代 GPU 都翻不了盘。**这也是这两个算子经常被拿来当"内存瓶颈/计算瓶颈"典型例子的真正原因**——不是因为它们恰好是这两类的代表,而是因为它们的 AI 离 ridge point 太远,分类结果几乎不可能被"换一块 GPU"这种扰动打破。真正会因为换 GPU 而翻盘的,是 AI 恰好卡在不同 GPU ridge point 之间的算子——[02 类知识点 3](02-roofline-model.md) 用 32k 上下文的 attention(AI=252.06)在 H100(ridge=295.2)和 H200(ridge=206.0)之间验证过这类翻盘,阶段 5 会用同样的思路,但翻盘的原因换成"同一个算子换一种访存方式核算",而不是"换一块 GPU"。

## 阶段 3:Online Softmax——分块处理一组数值,不需要一次性拿到全部数据

标准 softmax 要求"两趟遍历":第一趟要看完整组数字才能求出全局最大值(所以要求整组数据必须已经全部在手边——不能是一个只能往前走、不能回头重看的数据流);第二趟才能拿这个最大值去做归一化。[04 类知识点 1](04-flashattention-and-fusion.md) 已经验证过 online softmax 只需要一趟——数据以 block 为单位陆续到达时就能同步更新 `m`(running max)和 `l`(running 分母),处理完最后一个 block 立刻就有正确答案,不需要"先看完全部,再回头算"这个前提。这一步先只处理"一组数值本身"(不引入 V 加权),对应 [04 类知识点 1](04-flashattention-and-fusion.md) 里 `flash_attention.py` 每一行内层循环的 `m`/`l` 这两个 running state,先不管第三个 state `o`(阶段 4 再补上)。

为了让"不需要一次性拿到全部数据"这句话不只是嘴上说说,下面用一个生成器模拟"数据分批到达、读过的 block 不会被完整重新收集"这个约束——`softmax_online_stream` 的入参是一个只能顺序遍历一次的流,不是一个已经摆在手边的完整数组。

```python
import math

def softmax_naive(x):
    """先把整组数据全部拿到手里，一次性算标准softmax（减去最大值做数值稳定）——参照组。"""
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = sum(e)
    return [v / z for v in e]

def block_generator(data, block_size):
    """模拟数据分批到达：调用方每次只能拿到一个block，读过的block不会被完整重新收集。"""
    for start in range(0, len(data), block_size):
        yield data[start:start + block_size]

def softmax_online_stream(block_stream):
    """只扫一遍block_stream，用running max(m)/running分母(l)递推，
    从不要求一次性把整组原始数值摆在手边——这正是FlashAttention online softmax的核心技巧，
    这里先不引入V加权，只对"数值本身"做归一化（对应04类知识点1里m/l这两个state，o留到阶段4）。"""
    m = -math.inf
    l = 0.0
    numerators = []   # 只增长，不需要事先知道数据总长度
    for block in block_stream:
        block_max = max(block)
        m_new = max(m, block_max)
        rescale = math.exp(m - m_new) if m != -math.inf else 0.0
        l = l * rescale
        numerators = [v * rescale for v in numerators]   # 已经算出的分子，换算到新基准m_new下
        for s in block:
            p = math.exp(s - m_new)
            numerators.append(p)
            l += p
        m = m_new
    return [v / l for v in numerators]

x = [2.0, 0.5, 3.0, 1.0, -1.0, 4.0, 0.0, 2.5]

standard = softmax_naive(x)
online = softmax_online_stream(block_generator(x, block_size=3))   # 8个数，block_size=3不整除，最后一个block只有2个

max_diff = max(abs(a - b) for a, b in zip(standard, online))
assert max_diff < 1e-12
assert abs(sum(online) - 1.0) < 1e-12   # softmax结果必须归一化到1

# 两个边界场景：block_size=1（完全逐个到达，最极端的流式场景）和block_size远大于数据长度（退化成一次性看完）
online_bs1 = softmax_online_stream(block_generator(x, block_size=1))
assert max(abs(a - b) for a, b in zip(standard, online_bs1)) < 1e-12
online_bs_all = softmax_online_stream(block_generator(x, block_size=100))
assert max(abs(a - b) for a, b in zip(standard, online_bs_all)) < 1e-12

print("stage3 max diff (block_size=3):", max_diff)
print("stage3 ok")
```

本机实测输出:`stage3 max diff (block_size=3): 2.7755575615628914e-17`——这个量级比 float64 机器精度(约 2.22e-16)还小一个数量级,基本就是"位级相等"。三种 block_size(1、3、100)全部通过同一个断言阈值,说明这套递推的正确性和"数据具体怎么分块"无关——这也呼应 [04 类知识点 1 追问 3](04-flashattention-and-fusion.md) 已经验证过的结论:block 大小、切分边界、扫描顺序都不影响最终结果精确等价这件事。

## 阶段 4:给 Online Softmax 配上 V 加权——组装成"单行 mini flash attention"

阶段 3 只做到"分块算出归一化后的数值本身",还不是真正的 attention。真正的 attention 需要再做一步:把 softmax 权重和对应的 V 向量做加权平均,得到最终输出。[04 类知识点 1](04-flashattention-and-fusion.md) 的 `attention_flash` 对每一行 Q 维护三个 running state(`m`/`l`/`o`),这一步就是把阶段 3 缺的第三个 state `o` 补上——`o` 每次遇到新 block 也要按同一个 `rescale` 系数收缩,再加上新 block 的加权贡献,这是 [04 类知识点 1"底层机制"](04-flashattention-and-fusion.md) 里推过的同一个指数可加性恒等式(`exp(s_i-m_new)=exp(s_i-m)*exp(m-m_new)`)在起作用,这里不重复推导,只重新手写一遍验证。

```python
import math, random

def softmax_naive(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = sum(e)
    return [v / z for v in e]

def weighted_softmax_naive(scores, values):
    """先把整行score算出softmax权重，再和values做加权平均——"先materialize再算"的参照组。"""
    weights = softmax_naive(scores)
    d = len(values[0])
    out = [0.0] * d
    for w, v in zip(weights, values):
        for k in range(d):
            out[k] += w * v[k]
    return out

def paired_block_generator(scores, values, block_size):
    """和阶段3的block_generator同一个模拟：scores和values配对分批到达。"""
    n = len(scores)
    for start in range(0, n, block_size):
        yield scores[start:start + block_size], values[start:start + block_size]

def weighted_softmax_online_stream(block_stream):
    """block_stream 每次吐出 (scores_block, values_block) 一对，在阶段3的m/l递推上
    再加一个按分量累加的 o —— 这正是 flash_attention.py 里 attention_flash 对每一行 Q 做的事，
    这里只保留"一行"的情形，聚焦online softmax本身，不重复Q·K点积那一层。"""
    m = -math.inf
    l = 0.0
    o = None
    for scores_block, values_block in block_stream:
        if o is None:
            d = len(values_block[0])
            o = [0.0] * d
        block_max = max(scores_block)
        m_new = max(m, block_max)
        rescale = math.exp(m - m_new) if m != -math.inf else 0.0
        l = l * rescale
        o = [v * rescale for v in o]
        for s, v in zip(scores_block, values_block):
            p = math.exp(s - m_new)
            l += p
            for k in range(len(v)):
                o[k] += p * v[k]
        m = m_new
    return [v / l for v in o]

random.seed(42)
N, d = 11, 4     # N不是block_size的整数倍，专门验证边界
scores = [random.uniform(-3, 3) for _ in range(N)]
values = [[random.random() for _ in range(d)] for _ in range(N)]

naive_out = weighted_softmax_naive(scores, values)
online_out = weighted_softmax_online_stream(paired_block_generator(scores, values, block_size=4))

max_diff = max(abs(a - b) for a, b in zip(naive_out, online_out))
assert max_diff < 1e-12

# 再换block_size=1（最极端的逐个到达）确认结论不依赖分块方式
online_out_bs1 = weighted_softmax_online_stream(paired_block_generator(scores, values, block_size=1))
assert max(abs(a - b) for a, b in zip(naive_out, online_out_bs1)) < 1e-12

# 04类知识点1已验证过的同一个事实：-inf特判在Python里其实是多余的（不是bug，只是防御性冗余）
assert math.exp(-math.inf - 3.5) == 0.0

print("stage4 max diff (block_size=4):", max_diff)
print("stage4 ok")
```

本机实测输出:`stage4 max diff (block_size=4): 2.220446049250313e-16`——量级正好落在 float64 机器精度(约 2.22e-16)附近,和 [04 类知识点 1](04-flashattention-and-fusion.md) 报告的"1.110e-16"是同一个量级(具体数值不同是因为这里的随机种子、数据规模和 block 切法都不一样,但"精确到机器精度、不是近似算法"这个结论完全一致)。到这里,`weighted_softmax_online_stream` 已经是一个真实能用的"单行 mini flash attention"——给一组 K 分数和对应的 V 向量,分块喂进去,不需要一次性拿到全部数据,结果和"先把所有分数算完、materialize 成完整权重、再加权平均"精确一致(机器精度以内)。

## 阶段 5:组装——用 Roofline 计算器量化 Online Softmax 到底省下多少访存

前四个阶段搭了两个各自独立的小工具:阶段 1-2 的 Roofline 计算器能判断"一个算子是不是内存瓶颈";阶段 3-4 的 online softmax 证明了"分块算 attention 在数学上精确等价"。这一步把两者接上:[04 类知识点 2](04-flashattention-and-fusion.md) 已经验证过,naive attention 要把 N×N 的 `S`、`P` 两个矩阵完整写入/读出 HBM,flash attention(阶段 3-4 手写的这套 online softmax 正是它的核心)全程不 materialize 这两个矩阵——同一个算子,FLOPs 几乎不变,但 `bytes_moved` 差出一大截。喂进阶段 1 的计算器,直接看它是否把 roofline 分类判决翻了盘。

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved

def ridge_point(peak_flops_per_s, peak_bytes_per_s):
    return peak_flops_per_s / peak_bytes_per_s

def classify(ai, ridge):
    return "compute" if ai >= ridge else "memory"

def attn_flops(seq_len, head_dim):
    """单头attention的浮点运算量，公式和02-roofline-model.md的attention_profile一致（b=h=1时）：
    flops = 4 * b * h * s * s * d。naive和flash两种访存方式做的浮点运算量相同，这里只用一个函数。"""
    return 4 * seq_len * seq_len * head_dim

def hbm_naive_attn(seq_len, head_dim, dtype_bytes=2):
    """公式和04-flashattention-and-fusion.md知识点2的hbm_naive_attn完全一致：
    QKV读 + S矩阵写读 + P矩阵写读 + 输出写。"""
    N, d = seq_len, head_dim
    qkv = 3 * N * d
    s_matrix = N * N
    p_matrix = N * N
    out = N * d
    return dtype_bytes * (qkv + 2 * s_matrix + 2 * p_matrix + out)

def hbm_flash_attn(seq_len, head_dim, dtype_bytes=2):
    """公式和04-flashattention-and-fusion.md知识点2的hbm_flash_attn完全一致：
    只有QKV读 + 输出写，S/P两个N×N矩阵全程不落HBM——这正是阶段3-4手写的online softmax递推能做到的事。"""
    N, d = seq_len, head_dim
    qkv = 3 * N * d
    out = N * d
    return dtype_bytes * (qkv + out)

H100_PEAK_FLOPS = 989.0e12
H100_PEAK_BW = 3.35e12
ridge = ridge_point(H100_PEAK_FLOPS, H100_PEAK_BW)

N, d = 2048, 128
flops = attn_flops(N, d)
naive_bytes = hbm_naive_attn(N, d)
flash_bytes = hbm_flash_attn(N, d)

# 交叉核对04-flashattention-and-fusion.md知识点5表格里N=2048那一行：naive 35.7MB，flash 2.1MB
assert round(naive_bytes / 1e6, 1) == 35.7
assert round(flash_bytes / 1e6, 1) == 2.1

ai_naive = arithmetic_intensity(flops, naive_bytes)
ai_flash = arithmetic_intensity(flops, flash_bytes)
bound_naive = classify(ai_naive, ridge)
bound_flash = classify(ai_flash, ridge)

# 同一个flops，naive/flash公用——FlashAttention不省计算，只省访存（04类知识点2常见坑已强调过）
assert round(ai_naive, 2) == 60.24
assert ai_flash == 1024.0
assert bound_naive == "memory"     # 按naive方式核算的访存量，在H100上是内存瓶颈
assert bound_flash == "compute"    # 同样的计算量，按flash方式核算的访存量，在H100上翻转成计算瓶颈

# 交叉核对04文件speedup_curve的N=2048那一行：speedup=17.0
speedup = naive_bytes / flash_bytes
assert round(speedup, 1) == 17.0

print(f"N={N}, d={d}, H100 ridge={ridge:.1f} FLOP/byte")
print(f"naive-style AI={ai_naive:.2f} -> {bound_naive}")
print(f"flash-style AI={ai_flash:.2f} -> {bound_flash}")
print("stage5 ok")
```

本机实测输出:`naive-style AI=60.24 -> memory`、`flash-style AI=1024.00 -> compute`。**这里有一处诚实的返工**:最初手算 `2048*128/(2*(128+2048))` 心算成了 60.22,真正在 `.venv` 里跑出来是 `60.235294...`,四舍五入到小数点后两位是 **60.24**,不是手算猜的 60.22——这正是整篇教程"先跑出真实输出、再誊写成断言"这条纪律要防的事:哪怕只是一次除法,手算和真实浮点运算也可能对不上,连教程作者自己都在这道题上犯过。`ai_flash` 恰好是整数 `1024.0` 不是巧合:`flash_bytes` 只有 `dtype_bytes*4*N*d` 这一项,`flops` 是 `4*N*N*d`,两者相除,`N*N` 和 `N` 先约掉一个 `N`,剩下 `flops/flash_bytes = N/dtype_bytes`——代入 `N=2048,dtype_bytes=2` 正好等于 `1024`,这是一个和 `d` 完全无关的干净结果,读者可以自己代另一组 `N`、`dtype_bytes` 验证这个规律是否成立。

**这就是这篇教程真正的组装时刻**:阶段 1-2 搭的计算器和阶段 3-4 手写的算法,原本是两个互不相关的小工具,分别验证过各自正确;这一步用同一个计算器分析同一个算子在两种访存方式下的 AI,亲眼看到 roofline 判决从 `memory` 翻成 `compute`——online softmax 省下来的不是抽象的"效率提升",是具体、可测量、会让 roofline 分类结论改变的字节数。

---

## 可以怎么继续扩展(只指方向,不实现)

- **把阶段 1-2 的计算器扩成一张真实"算子清单"**:[02 类知识点 4](02-roofline-model.md) 的 capstone 已经示范了"10 个算子 × 4 款 GPU = 40 组"的批量分析,阶段 1-2 手写的计算器同样可以套上更大的算子清单(LayerNorm、attention、不同形状的 GEMM),自动统计有多少比例是 memory-bound——这是把玩具计算器变成真正调优工具的第一步。
- **阶段 3-4 的 online softmax 扩展到多行 Q**:这篇教程为了聚焦 online softmax 本身,只处理了"一行"的情形(对应 `attention_flash` 外层循环里的一次迭代);扩展到多行 Q、多个 attention head、批量维度,就是 [04 类知识点 1](04-flashattention-and-fusion.md) `attention_flash` 的完整版本,那边已经验证过、这里不重复实现。
- **给阶段 5 的翻盘分析换一块 GPU**:阶段 5 只在 H100 上验证了一次翻盘,[02 类知识点 3](02-roofline-model.md) 已经验证过 H100/H200 的 ridge point 差出 30%,同一个 `N=2048,d=128` 换到 H200 上,`bound_naive`/`bound_flash` 会不会又不一样,是一个可以自己动手代公式验证的开放问题。
- **真实序列长度扫描**:阶段 5 只固定了 `N=2048`,[04 类知识点 5](04-flashattention-and-fusion.md) 的 capstone 已经验证过从 512 到 131072 的完整曲线(speedup 从 5.0x 涨到 1025.0x)——把阶段 5 的 `ai_naive`/`ai_flash` 也按同样的 `N` 序列扫一遍,能看到"AI 差距"这条曲线和"HBM 流量节省倍数"那条曲线是不是同一个形状。

这几个方向都不实现,是为了让这篇教程聚焦在"两个已学知识点怎么手写、怎么组装"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这一篇额外验证了一件事:教程体不要求硬凑成"一个类、一次性组装"的形态([dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 是三个技巧拼进一个 `MiniSearchEngine` 类)——两个概念上独立的小工具(性能预测计算器 + 算法技巧),只要最后能找到一个真实、不牵强的连接点(阶段 5 用计算器称出算法技巧的收益),同样能组成一篇完整的教程,不必强行拼成同一个类或同一个数据结构。哪个模式更合适,取决于要组装的知识点之间原本是"同一个系统里的不同角色分工"(适合拼成一个类),还是"两个独立地各自成立、但能互相验证价值"的关系(适合分开写、最后用一个共同的度量连接)。

---

*创建:2026-07-24*
