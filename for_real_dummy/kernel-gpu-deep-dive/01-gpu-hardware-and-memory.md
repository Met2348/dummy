# 01 · GPU 硬件与存储层次深挖(GPU Hardware & Memory Hierarchy)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 本篇是 `kernel-gpu-deep-dive` 系列的第一批,回答"一张 GPU 卡长什么样":算力/显存/带宽这些参数怎么组织成可编程查询的数据,寄存器→共享内存→L2→HBM 这条存储金字塔怎么选,Tensor Core 为什么比通用 ALU 快一个数量级,一个 SM 上能塞下多少个线程块,以及多卡互联带宽怎么建模。后面几批(roofline 性能建模、kernel 设计语言、FlashAttention)全部建立在这一批的概念之上。

**这一批和 `learning/gpu-architecture/` 源码模块是什么关系(必须先说清楚):** `learning/gpu-architecture/README.md` 已经有非常详尽的专题概览、4 代 GPU 横向对比表、cheatsheet,但那份文档是"研究者对研究者"的密度——不铺垫、不做面试向的追问链。本篇讲**同一份代码**,但换一种写法:每个知识点从"最笨的问题"讲起(比如先问"为什么普通矩阵乘法在 GPU 上会算力打折",再引出 Tensor Core 的解释),并且额外多两块——**底层机制/为什么这样设计** 和 **面试怎么问 + 追问链**。

**必须显著重申一件事(源码模块 README 已经反复强调,这里是本系列正文第一次出现,再显著说一遍):** `learning/gpu-architecture/src/` 下的代码**不是可以编译的真实 CUDA/Triton/CUTLASS kernel**,是用可以现场 `assert` 验证的纯 Python 数值/机制模拟,去复现"专用矩阵单元为什么快""存储层次怎么选""SM 占用率怎么算""多卡带宽差多少"这些真实系统的行为规律——这是在 Windows 工作站上、不装 GPU 驱动/不装 CUDA 工具链也能把"为什么"讲透的设计取舍,不是造轮子偷懒。真正能跑在 GPU 上的 CUDA/Triton 代码需要 Linux + CUDA 工具链,不在本系列、也不在 `learning/` 任何模块的范围内。

**环境:** 仓库根目录 `.venv`(Windows 原生,Python 3.13.9)。全篇纯 CPU、零第三方依赖——`learning/gpu-architecture/src/` 下的脚本只依赖标准库的 `dataclasses`/`math`/`__future__`,秒级跑完,不需要 GPU、不需要装包。这些脚本不是通过 `pip install` 装进 `.venv` 的包,例子里统一用 `sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")` 从文件系统路径直接导入——这和仓库 `python-idioms`/`long-context-deep-dive` 系列已经在用的是同一种写法,不是本篇自己发明的技巧。

**本篇统一结构(七步,和 torch-deep-dive/numpy-deep-dive 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制/为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(带 assert,真在 `.venv` 里跑过)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. GPU 规格数据表 —— 把"型号参数表"变成可编程查询的对象(`common.py::GPUS`)

**是什么:**
```python
from common import GPUS, GPUSpec

GPUS: dict[str, GPUSpec]          # 型号简称(str) -> 规格对象,O(1) 查询
GPUS["H100"].bf16_tflops           # 直接按字段取数,不用翻 spec sheet
GPUS["H100"].ridge_point_bf16()    # 规格对象自带的一个计算方法
```
> 提醒:`00-roadmap.md` 里简写成 `GPU_CATALOG`,但读源码确认真实变量名是 `GPUS`——下文一律按源码原名,这也是这份系列反复强调的纪律:字段名/变量名以读到的源码为准,不凭型号名或计划文档里的简写去猜。

**一句话:** `GPUSpec` 是一个 `@dataclass(frozen=True)`,把一张 GPU 卡的算力(BF16/FP8/FP4 峰值 TFLOPS)、显存容量、显存带宽、NVLink 带宽、功耗打包成一个不可变对象;`GPUS` 是以型号简称为 key 的字典,覆盖 `A100`/`H100`/`H200`/`B200`/`GB200` 共 5 款,是 `learning/gpu-architecture/` 系列后续所有脚本(roofline、capstone)统一复用的"事实来源"。

**底层机制/为什么这样设计:** 真实字段(源码 `common.py:6-20`):
```python
@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_tflops: float          # dense BF16
    fp8_tflops: float           # E4M3
    fp4_tflops: float           # Blackwell only
    hbm_gb: int
    hbm_tb_s: float             # memory bandwidth
    nvlink_tb_s: float          # per-GPU bidirectional
    tdp_w: int

    def ridge_point_bf16(self) -> float:
        return (self.bf16_tflops * 1e12) / (self.hbm_tb_s * 1e12)
```
`frozen=True` 让规格对象运行时不可变——这些数字代表"物理事实"(vendor published peak numbers),不应该在程序运行过程中被意外改写。`GPUS` 用 `dict[str, GPUSpec]` 而不是"写一堆 if/elif 判断型号名"的过程式代码,是"数据和逻辑分离"的直接体现:后面 `capstone_roofline_zoo.py` 这类脚本可以直接 `for name, spec in GPUS.items()` 批量遍历所有型号,不需要为每加一款新 GPU 就多写一个分支。`ridge_point_bf16()` 这个方法绑定在数据对象自己身上而不是写成外部函数,是因为它需要的两个字段(`bf16_tflops`、`hbm_tb_s`)本来就都在这个对象上,"让数据自己知道怎么算"比"外部函数再重新解构数据"更内聚。

**AI 研究/工程场景:** 写"该用哪款 GPU 训练/推理这个模型"这类分析脚本时,不用东翻西找 spec sheet 把数字硬编码好几处,直接查表;`GPUS` 是这一个专题所有脚本共享的"single source of truth",改一次(比如某天 vendor 更新了标定数字)全系列跟着更新,不会出现"代码里同一个 H100 带宽数字在 3 个地方写了 3 遍、改的时候漏了一处"这种典型的复制粘贴 bug。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
import dataclasses
from common import GPUS, roofline_flops

assert set(GPUS.keys()) == {"A100", "H100", "H200", "B200", "GB200"}

h100 = GPUS["H100"]
h200 = GPUS["H200"]
b200 = GPUS["B200"]

# 真实规格数字(实测,不是背出来的)
assert h100.bf16_tflops == 989.0 and h100.hbm_tb_s == 3.35 and h100.hbm_gb == 80
assert h200.hbm_gb == 141 and h200.hbm_tb_s == 4.8            # H200 只加显存容量/带宽
assert h200.bf16_tflops == h100.bf16_tflops == 989.0           # 算力和 H100 完全一样,同一颗芯片
assert b200.fp4_tflops == 9000.0 and h100.fp4_tflops == 0.0    # FP4 是 Blackwell 独占,H100 建模为 0

# frozen=True: 规格是"物理事实",运行时不允许改写
try:
    h100.hbm_gb = 999
    assert False
except dataclasses.FrozenInstanceError:
    pass

# 查询表驱动计算: ridge_point 是"数据 + 行为"绑在一起的方法
rp_h100 = h100.ridge_point_bf16()
rp_h200 = h200.ridge_point_bf16()
assert round(rp_h100) == 295
assert round(rp_h200) == 206
assert rp_h200 < rp_h100     # H200 算力没涨、显存带宽涨更多 -> ridge point 反而更低(02 篇细讲这个反直觉发现)

# 常见坑现场复现: fp4_tflops=0.0 是"不支持"的哨兵值,不是"支持但很慢"
# 如果不检查 dtype 是否被这款 GPU 支持,直接传进 roofline_flops 会静默算出 0.0,不会报错提醒你
r = roofline_flops(h100, ai=500.0, dtype="fp4")
assert r == 0.0
```

**面试怎么问 + 追问链:**
- **Q:** "如果让你设计一个存储多种 GPU 硬件规格的数据结构,你会怎么设计?这里为什么用 `dataclass` 而不是 `dict of dict`?"—— 期望答出"字段有固定 schema、需要类型提示和不可变性,dataclass 比裸 dict 更适合",而不仅仅是"看起来更整洁"。
- **追问 1:** "为什么设成 `frozen=True`?不设会有什么风险?"—— 期望能想到"这些数字是外部事实,运行时被意外改写会导致后续所有基于它的计算都错,而且很难排查"。
- **追问 2(区分度很高):** "H100 和 H200 的 `bf16_tflops` 字段值完全一样(都是 989.0),这是数据表凑出来的巧合吗?"—— 期望答出"不是巧合,H200 是 H100 的显存升级版(HBM3→HBM3e,容量 80GB→141GB,带宽 3.35→4.8 TB/s),计算核心是同一颗芯片,算力字段理应相等"——能不能把"数据表里两个字段恰好相等"这个现象和背后的真实硬件产品线联系起来,是检验候选人是不是真的懂硬件,还是只会读数字的好问题。

**常见坑:**
- `fp4_tflops=0.0`(A100/H100/H200 都不支持 FP4)是一个"哨兵值",不是"支持但吞吐是 0"——上面例子已经实测验证:直接拿不支持的 dtype 去调用 `roofline_flops` 不会报错,只会静默返回 `0.0`,这种"看起来是合法结果、实际是无意义结果"的静默失败,比直接崩溃更危险,批量脚本处理多个 GPU × 多个 dtype 组合时,一定要显式检查这款 GPU 是否支持该 dtype,不能依赖"0.0 反正也不会被选中"这种偷懒假设。
- `name` 字段(`"H100-SXM5"`)和 `GPUS` 的 key(`"H100"`)是两个不同的字符串——`name` 是给人看的全称,key 是给程序查表用的简称,打印/日志里两者容易被搞混。

---

## 2. 存储层次与 tier 推荐 —— `memory_hierarchy.py::recommend_tier`

**是什么:**
```python
from memory_hierarchy import recommend_tier, H100_HIERARCHY, MemTier, cost_to_load

recommend_tier(working_set_bytes: int, reuse_count: int) -> MemTier
# H100_HIERARCHY: list[MemTier],5 层,从快到慢:
# registers -> shared_memory -> L1_cache -> L2_cache -> HBM3
```

**一句话:** 给定"这块数据多大"和"会被同一个 kernel 重用几次"两个信息,`recommend_tier` 用一串 if 阶梯,推荐把它放进 5 层存储金字塔里的哪一层——本质上是把"写 kernel 时该怎么分块、该不该显式声明 `__shared__` 缓存"这个工程决策,简化成一个可以现场跑分类的小函数。

**底层机制/为什么这样设计:** `H100_HIERARCHY`(源码 `memory_hierarchy.py:14-20`)5 层结构,实测(见下方可运行例子)延迟单调递增、带宽单调递减:

| 层 | 容量(每 SM) | 延迟(cycles) | 带宽 |
|---|---|---|---|
| registers | 256.0 KB | 1 | 2000.0 TB/s |
| shared_memory | 228.0 KB | 30 | 228.0 TB/s |
| L1_cache | 128.0 KB | 30 | 128.0 TB/s |
| L2_cache | 0(per-GPU,60MB 共享) | 200 | 12.0 TB/s |
| HBM3 | 0(per-GPU,80GB) | 600 | 3.35 TB/s |

`recommend_tier` 的判断顺序(源码 `memory_hierarchy.py:28-36`):先问"数据放得下寄存器吗,且被重用超过 10 次吗",不满足再问"放得下 SMEM 吗,且重用超过 2 次吗",再不满足看"是否 ≤60MB 能进 L2",最后兜底 HBM。`reuse_count` 门槛的设计意图很直接:寄存器/SMEM 是 SM 上极度稀缺的资源,只有会被反复读的数据才值得为它腾出黄金地段——如果一块数据只用一次,直接从 HBM/L2 读一次就够了,占着寄存器反而挤占了真正需要复用的数据。

但仔细读判断顺序会发现两处这个简化模型自身的粗糙之处:
1. **L2 这一档完全不检查 `reuse_count`**——不管一块 ≤60MB 的数据被重用 1 次还是 100 万次,只要装得下就统统推荐 L2,这和寄存器/SMEM 两档"必须重用够多次才推荐"的逻辑不一致,是这个模型没有细化的地方(真实 L2 命中率还取决于访问模式、有没有被别的 kernel/block 冲刷掉,不是单纯看数据大小)。
2. **`L1_cache`(在 `H100_HIERARCHY` 里排第 3 位)从未被 `recommend_tier` 引用**——它只用来展示"延迟从 1 cycle 到 600 cycle 单调递增"这条阶梯事实(`_self_test()` 里 `assert lats == sorted(lats)` 验证的正是这个),决策函数本身直接从 SMEM 判断跳到 L2,完全跳过了它。读代码不能只看"这个列表定义了哪些元素",还要确认"哪个函数真的用到了哪个元素"。

**这一节的重点(今天 2026-07-12 验证 `learning/gpu-architecture/` 时修过、现场核实过的死代码):** `_self_test()` 里原来这一行(见 `git show c1568f0 -- learning/gpu-architecture/src/memory_hierarchy.py`):
```python
t = recommend_tier(1024, reuse=100) if False else recommend_tier(1024, 100)
```
`if False` 恒为假,Python 的条件表达式(`A if COND else B`)是惰性求值——只会执行被选中的那一支,`recommend_tier(1024, reuse=100)` 这半支永远不会被求值、更不会被调用。已经简化成:
```python
t = recommend_tier(1024, 100)
```
行为完全不变,因为原来那半支从来没有真正执行过。有意思的是,如果这半支真的被执行,会立刻报错——`recommend_tier` 的第二个形参名是 `reuse_count`,不是 `reuse`(见上面签名),`if False` 把这个写错的关键字参数连同它所在的分支一起彻底藏了起来,不会在任何一次真实调用里暴露,直到有人像今天这样把死分支单独摘出来读、或者手滑把 `False` 改成 `True`。这是"死代码不会自己跳出来提醒你"最直接的例证:一段代码能在 `_self_test()` 全绿的情况下安然存在,不代表它里面每一行都是对的,只代表**被执行到的**那些行是对的。

**AI 研究/工程场景:** 这个函数模拟的决策,正对应写 Triton/CUDA kernel 时"tile 开多大、要不要显式把数据搬进 `__shared__`"的核心考量。FlashAttention 的关键优化之一,就是把 Q/K/V 的一个 block 显式搬进 SMEM 反复参与内积计算(重用次数高),而不是每次都从 HBM 读——这正对应 `recommend_tier` 里 `shared_memory` 分支的判断依据(数据小 + 重用次数多)。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from memory_hierarchy import recommend_tier, H100_HIERARCHY, cost_to_load

# 四级判断阶梯,按定义顺序验证
assert recommend_tier(1024, 100).name == "registers"           # 小 + 重用多 -> 寄存器
assert recommend_tier(100 * 1024, 5).name == "shared_memory"    # 100KB 重用 5 次 -> SMEM
assert recommend_tier(10 * 1024 * 1024, 1).name == "L2_cache"   # 10MB -> L2
assert recommend_tier(100 * 1024 * 1024, 1).name == "HBM3"      # 100MB -> HBM

# 关键细节: 同样 1KB 的数据,只重用 1 次(没过寄存器/SMEM 的 reuse 门槛)
# 不会进寄存器,而是一路落到"只看装不装得下"的 L2 分支
assert recommend_tier(1024, 1).name == "L2_cache"

# L2 分支完全不看 reuse_count: 重用 1 次和重用 99 万次,只要 <=60MB,结果一样
assert recommend_tier(10 * 1024 * 1024, 1).name == recommend_tier(10 * 1024 * 1024, 999_999).name == "L2_cache"

# H100_HIERARCHY 里排第 3 位的 L1_cache,从未被 recommend_tier 返回过
tier_names_in_hierarchy = [t.name for t in H100_HIERARCHY]
assert tier_names_in_hierarchy == ["registers", "shared_memory", "L1_cache", "L2_cache", "HBM3"]
assert "L1_cache" not in {
    recommend_tier(1024, 100).name, recommend_tier(100 * 1024, 5).name,
    recommend_tier(10 * 1024 * 1024, 1).name, recommend_tier(200 * 1024 * 1024, 1).name,
}

# 延迟单调递增(_self_test 里验证过的阶梯事实)
lats = [t.latency_cycles for t in H100_HIERARCHY]
assert lats == sorted(lats) == [1, 30, 30, 200, 600]

# 今天验证时修过的死代码分支,现场复现证明那半支确实是坏的:
# recommend_tier 的真实形参名是 reuse_count,不是 reuse
try:
    recommend_tier(1024, reuse=100)
    assert False
except TypeError as e:
    assert "unexpected keyword argument 'reuse'" in str(e)
```

**面试怎么问 + 追问链:**
- **Q:** "给你一块 100KB 的数据,在一个 kernel 里会被重用 5 次,你会放在哪一级存储?为什么不放寄存器?"—— 期望答"共享内存"——100KB 超过寄存器容量上限(256KB 但门槛还要求重用 >10 次这里只有 5 次),但装得下 228KB 的 SMEM 且重用 >2 次。
- **追问 1:** "如果同样 100KB,只被用 1 次呢?"—— 期望答"不值得放 SMEM,应该直接留在更低一级(L2/HBM),读一次的开销远小于'为它腾出稀缺 SMEM 空间'的机会成本"。
- **追问 2(区分度很高):** "L2 cache 这一档,不管数据被重用 1 次还是 100 万次,这个函数都推荐 L2——你觉得这是不是一个合理的简化?现实中 L2 是否命中还依赖什么?"—— 期望能提到"实际命中率取决于访问模式、有没有被同时运行的其它 block/kernel 挤出去,不是只看数据大小就能保证命中",能看出候选人是真的理解真实 L2 行为,还是只会背这个简化模型的判断阶梯。

**常见坑:**
- 死代码分支里的 bug 不会被 `assert` 覆盖到——`_self_test()` 全部通过,只能证明**被执行到**的代码是对的,不能证明整个文件没有问题;没被执行到的分支哪怕参数名都写错了,也不会在任何一次运行里暴露。
- **(顺手验证到的额外发现,不是本节要求验证的重点,但很能说明同一个道理)** 同文件里的 `cost_to_load(bytes_needed, tier)` 函数,`_self_test()` 完全没有测过它。实测发现它有一处单位换算错误:`(bytes_needed / 1e9) / tier.bandwidth_tb_s * 1e6` 把字节数除以 `1e9` 当成"转换成 GB",却直接拿去除以一个以 **TB/s** 为单位的带宽值,少换算了 1000 倍。真实验证:`cost_to_load(1_000_000, H100_HIERARCHY[4])`(HBM3,3.35 TB/s 读 1MB)返回 `298.5075` 微秒,但按 3.35 TB/s 真实带宽换算,1MB 应该约 `0.2985` 微秒传完,两者相差整整 1000 倍(`298.5075 / 0.2985 ≈ 1000`)。这不影响 `recommend_tier` 本身(它不调用 `cost_to_load`),但再次印证同一条规律:**没有被 `assert` 验证过的函数,和"这个函数存在数量级错误"经常是同一件事的两面。**

---

## 3. Tensor Core 吞吐建模 —— `tensor_core.py::flops_per_cycle` / `pick_shape`

**是什么:**
```python
from tensor_core import MMAShape, H100_WGMMA, B200_TCGEN05, flops_per_cycle, pick_shape

flops_per_cycle(shape: MMAShape) -> int
# 一条 MMA(Matrix Multiply-Accumulate)指令,一个"概念 cycle"编码了多少 FLOP

pick_shape(target_m: int, target_n: int, dtype: str, isa: list[MMAShape]) -> MMAShape
# 从一组指令形状里,挑出最贴合 target_m x target_n 这个 tile 尺寸的那一条
```

**一句话:** Tensor Core 比通用 ALU(CUDA Core)在矩阵乘法上快一个数量级的根本原因,不是"晶体管跑得更快",而是**一条硬件指令编码的计算量,从"一次标量乘加"变成了"一整块 m×n×k 的矩阵乘加"**——`flops_per_cycle(shape) = 2 * shape.m * shape.n * shape.k` 这一行代码,直接建模的就是"指令粒度变粗"这件事。

**底层机制/为什么这样设计:** 源码定义了两组 ISA(`tensor_core.py:16-26`):

| ISA | shape | m | n | k | dtype | flops_per_cycle |
|---|---|---|---|---|---|---|
| H100_WGMMA | wgmma.m64n8k16.bf16 | 64 | 8 | 16 | bf16 | 16,384 |
| H100_WGMMA | wgmma.m64n256k16.bf16 | 64 | 256 | 16 | bf16 | 524,288 |
| H100_WGMMA | wgmma.m64n256k32.fp8 | 64 | 256 | 32 | fp8 | 1,048,576 |
| B200_TCGEN05 | tcgen05.m128n256k32.bf16 | 128 | 256 | 32 | bf16 | 2,097,152 |
| B200_TCGEN05 | tcgen05.m128n256k64.fp8 | 128 | 256 | 64 | fp8 | 4,194,304 |
| B200_TCGEN05 | tcgen05.m128n256k128.fp4 | 128 | 256 | 128 | fp4 | 8,388,608 |

`flops_per_cycle = 2*m*n*k` 的三个因子分别是:`m*n` 是这一条指令一次算完的输出 tile 里有多少个元素,`k` 是每个输出元素要累加多深(规约维度),`2` 是"每一步累加 = 一次乘法 + 一次加法 = 2 FLOP"。三者相乘,就是"一条指令总共完成了多少次乘加"。相比之下,一个通用 ALU(CUDA Core)一条指令通常只做一次标量 FMA(2 FLOP)——Tensor Core 把 `m*n*k` 个标量乘加压缩进一条硬件指令并行完成,这正是专用矩阵单元和通用 ALU 数量级差距的来源:**不是频率差距,是"一条指令做多少事"的差距**。

dtype 精度如何参与建模:同一份 ISA 列表里,精度越低,同一档位的 `k`(规约维度)越大——H100 上 bf16 是 `k=16`,fp8 是 `k=32`(翻倍);B200 上 bf16 是 `k=32`,fp8 是 `k=64`,fp4 是 `k=128`,一路倍增。因为 `flops_per_cycle` 正比于 `k`,精度每降一档,同一条指令能吃进的规约深度翻倍,直接换来吞吐翻倍——这是"低精度训练/推理更快"这条行业共识,在这份代码里被量化成具体倍数关系的原因。

`pick_shape` 的打分逻辑(`tensor_core.py:39-42`):
```python
def score(s: MMAShape) -> tuple[int, int]:
    m_waste = (target_m % s.m) if target_m >= s.m else (s.m - target_m)
    n_waste = (target_n % s.n) if target_n >= s.n else (s.n - target_n)
    return (m_waste + n_waste, -flops_per_cycle(s))
```
先比 `waste`(tuple 第一个元素,优先级最高,越小越好);waste 相同再比 `flops_per_cycle`(取负数配合 `sorted` 升序实现"选最大")。`waste` 区分两种情况:目标尺寸比 shape 这一维大,`waste` 是除不尽的余数(比如 1000 行数据按 256 一组分 tile,会剩 `1000 % 256 = 232` 行凑不满一组,这部分开销无论如何都要付);目标尺寸比 shape 这一维小,`waste` 是"差多少才能凑够一整个 tile"(比如只有 32 行数据却要用 `m=64` 的 tile,直接浪费了 32 行的算力)。

**(独立验证发现,建议记入本节"常见坑")** `_self_test()` 里有一行注释 `# FP4 has 4x throughput of FP8 at same shape`,但紧跟着的断言只写了 `assert flops_per_cycle(s4) >= 2 * flops_per_cycle(fp8_b200)`。实测这两个值——`fp4_b200` 的 `flops_per_cycle` 是 8,388,608,`fp8_b200` 是 4,194,304,比值精确等于 **2.0**,不是注释说的 4:因为 fp4 相对 fp8,在 `m128n256` 这一档只有 `k` 从 64 翻倍到 128(m、n 都没变),`flops_per_cycle=2mnk` 里只翻了 `k` 这一个因子,自然是 2 倍而不是 4 倍。断言用的 `>=` 又恰好在"只有 2 倍"时也能通过,所以这句和实际计算结果对不上的注释,不会被任何测试戳穿。

**AI 研究/工程场景:** 这就是为什么"矩阵维度最好是 64/128 这类数的整数倍"这条经验规则,在训练大模型的代码里反复出现——维度没对齐 Tensor Core 的 MMA tile 形状,`pick_shape` 这样的调度逻辑就要么浪费算力补齐 `waste`,要么退化到更小、更低效的 shape。FP8/FP4 训练与推理(Blackwell 时代的重要话题)本质上是在用这里建模的"精度换算力"做交易:精度每降一档,单指令能吃的规约深度越深、`flops_per_cycle` 越高,但数值表示范围更窄,对 scaling/量化策略的要求也更高。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from tensor_core import H100_WGMMA, B200_TCGEN05, flops_per_cycle, pick_shape

# flops_per_cycle = 2*m*n*k: 一条指令编码的计算量
biggest_bf16 = H100_WGMMA[1]   # wgmma.m64n256k16.bf16
assert biggest_bf16.name == "wgmma.m64n256k16.bf16"
assert flops_per_cycle(biggest_bf16) == 2 * 64 * 256 * 16 == 524288

# pick_shape: 目标 tile 越贴合、waste 越小越优先;waste 相同再选吞吐更大的
chosen = pick_shape(target_m=64, target_n=256, dtype="bf16", isa=H100_WGMMA)
assert chosen.name == "wgmma.m64n256k16.bf16"     # 正好对齐,waste=0

# 目标比最大 shape 还小(只有 32 行、8 列)时,两个 bf16 候选的 m_waste 都是 64-32=32(打平)
# 决胜点在 n_waste: m64n8k16 的 n_waste=8-8=0,m64n256k16 的 n_waste=256-8=248,前者完胜
tiny = pick_shape(target_m=32, target_n=8, dtype="bf16", isa=H100_WGMMA)
assert tiny.name == "wgmma.m64n8k16.bf16"

# dtype 决定可用的最大 k(规约维度): 精度越低,同一档位 k 越大
fp8_h100 = pick_shape(target_m=64, target_n=256, dtype="fp8", isa=H100_WGMMA)
assert fp8_h100.k == 32 > biggest_bf16.k == 16

# B200: bf16 -> fp8 -> fp4,k 依次 32 -> 64 -> 128,flops_per_cycle 同步倍增
bf16_b200 = pick_shape(128, 256, "bf16", B200_TCGEN05)
fp8_b200 = pick_shape(128, 256, "fp8", B200_TCGEN05)
fp4_b200 = pick_shape(128, 256, "fp4", B200_TCGEN05)
assert (bf16_b200.k, fp8_b200.k, fp4_b200.k) == (32, 64, 128)

ratio_fp4_fp8 = flops_per_cycle(fp4_b200) / flops_per_cycle(fp8_b200)
ratio_fp4_bf16 = flops_per_cycle(fp4_b200) / flops_per_cycle(bf16_b200)
assert ratio_fp4_fp8 == 2.0      # 源码注释写"4x",实测精确等于 2.0 —— 见下方常见坑
assert ratio_fp4_bf16 == 4.0

# _self_test() 自己的断言用的是 >=,在"实际只有 2x"时依然成立,不会暴露注释和数字的出入
assert flops_per_cycle(fp4_b200) >= 2 * flops_per_cycle(fp8_b200)
```

**面试怎么问 + 追问链:**
- **Q:** "为什么 Tensor Core 比 CUDA Core 在矩阵乘法上快一个数量级?差距是从哪来的?"—— 期望答到"指令粒度":一条指令编码一整块矩阵乘加,而不是频率或晶体管数量的差距。
- **追问 1:** "如果我的矩阵维度(比如 m=100)不是 Tensor Core 某个 MMA shape 的整数倍,会发生什么?"—— 期望说出 `waste` 的概念:要么补零浪费算力,要么退化用更小的 shape,吞吐打折。
- **追问 2(区分度很高):** "FP8 相比 BF16,为什么吞吐能提升?提升的来源具体是什么?"—— 期望候选人能精确说出"是同一条指令能处理的规约维度 `k` 变大了,不是频率或并行度变化",这需要真正看过 `flops_per_cycle=2mnk` 这个公式,而不是泛泛而谈"精度低了所以快"。

**常见坑:**
- `cycles=32` 这个字段,6 条 shape 全部一样,注释写的是"per warp-group on H100, conceptual on B200"——B200 那半是作者自己标注的"概念性"数字,不能当成真实硬件延迟去做精确的时间估算,只能用来做同代内、不同 shape 之间的相对比较。
- 上面"独立验证发现"提到的 `# FP4 has 4x throughput of FP8` 注释,和实际算出来的比值(精确 2.0 倍)不符,断言用的 `>=` 阈值又恰好盖住了这个出入——读到代码注释里的倍数关系,自己动手用公式重新算一遍再下结论,不要直接采信文字描述,这和上一节 `memory_hierarchy.py` 的死代码是同一类陷阱的不同变体:上次是"从不执行的分支",这次是"断言的宽松阈值掩盖了注释的过度承诺"。

---

## 4. SM Occupancy —— `sm_occupancy.py::occupancy` 的五重瓶颈模型

**在进入 occupancy 公式之前,先把 GPU 的执行模型从零建立起来:** 本篇开头(第 4 行总览句)就写过"一个 SM 上能塞下多少个线程块",第 2 节存储层次表的表头也写着"容量(每 SM)"——但全篇到这里为止,还没有正面解释过"SM 是什么""线程块(thread block)是什么""它们和'线程'到底是什么关系"。这一节要密集使用 `SM`/`thread block`/`warp`/`threadIdx` 讲五重瓶颈模型,不先把这套执行模型搭起来,下面的五个约束条件会看起来像是在摆弄一堆抽象符号。

先用一个类比建立直觉:**一块 GPU 芯片,相当于很多个相对独立的"小处理器"拼成的一个集群**,每个"小处理器"就是一个 **SM(Streaming Multiprocessor,流式多处理器)**——H100 上一共有 132 个 SM(`sm_occupancy.py` 模块开头的注释原话是"H100 132 SM, 2048 threads/SM"),每个 SM 自带一整套寄存器堆、共享内存(SMEM)、warp 调度器,基本可以类比成一颗微型 CPU 核心,能独立调度和执行指令。当你在 host(CPU)端发起一次 **kernel 启动**(调用一次 Triton/CUDA kernel),GPU 不会把整份工作丢给某一个 SM 从头做到尾,而是先把总的工作量拆成很多个大小相等的 **线程块(thread block)**——像把一批任务卡片分发给车间里的各个工位一样,由硬件调度器把这些 block 分发到各个空闲的 SM 上并行执行。一次 kernel 启动产生的全部 block,合起来叫一个 **grid**;block 总数通常远多于 132,一个 SM 同一时刻能同时驻留几个 block,取决于这个 SM 的资源(寄存器/SMEM/线程槽位)够不够分——这正是下面 `occupancy()` 要回答的问题。block 一旦被分配给某个 SM,就会一直留在这个 SM 上直到执行完,不会中途被搬到另一个 SM。

线程块内部还有一层切分:SM 并不会把 block 里的线程逐个独立调度,而是**每 32 个连续编号的线程打包成一组,叫一个 warp**,SM 硬件以 warp 为最小调度单位,同一个 warp 里的 32 个线程在同一个时钟周期执行同一条指令。这也是全系列反复出现"32"这个数字的根源(后面 03 篇讲 SMEM bank conflict、内存合并访问时还会再遇到)。一个 256 线程的 block,内部会被切成 `256/32 = 8` 个 warp——下面可运行例子里第一个 `occupancy(256, 32, 16.0)` 算出 `active_warps=64`、`blocks_per_sm=8`,两者相除正好是 `64/8=8`,验证的就是这个切分关系(本机 `.venv` 实测核对过,见下方可运行例子)。

画成一张层级示意图(以一个 256 线程/block 的 kernel、Block 0 被分配到 SM #0 为例):

```
kernel launch (host 端发起一次调用)
                               |
                               v
+-------------------------------------------------------------+
|                             Grid                            |
+------------+------------+------------+--------+-------------+
|  Block 0   |  Block 1   |  Block 2   |  ...   |  Block N-1  |
+------------+------------+------------+--------+-------------+
       |            |            |          |          |
       v            v            v          v          v
     SM #0        SM #1        SM #2       ...      SM #131

(H100 一共有 132 个 SM;Grid 是这次 kernel 启动产生的全部线程块;调度器决定每个 Block 分给哪个 SM;
 Block 0 一旦分配到 SM #0 就固定在那里,不会中途换到别的 SM —— 往下拆开 Block 0 内部:)

+--------------------------------------------------------------+
|                    Block 0 (256 threads)                     |
+------------+------------+------------+--------+--------------+
|   Warp 0   |   Warp 1   |   Warp 2   |  ...   |    Warp 7    |
+------------+------------+------------+--------+--------------+
|thread 0-31 |thread 32-63|thread 64-95|  ...   |thread 224-255|
+--------------------------------------------------------------+
```

`threadIdx`(CUDA/Triton 里的内置变量)标的就是这张图最底层的编号——**线程在它所属 block 内部的编号**,不是全局唯一编号:Block 0 里 Warp 0 的 `threadIdx.x` 取值是 0..31,Block 1 里 Warp 0 的 `threadIdx.x` **同样**是 0..31(每个 block 各自独立编号,互不知道对方的存在,也不知道自己是第几个 block)。如果 kernel 代码需要算一个全局唯一的位置,通常要写 `blockIdx.x * blockDim.x + threadIdx.x`,把"当前是第几个 block"(`blockIdx.x`)和"block 内第几个线程"(`threadIdx.x`)拼起来——这里先记住 `threadIdx` 具体指什么就够了,03 篇会再用到这个词。

有了这套模型,下面 `occupancy()` 要回答的问题就有了具体画面:**一个 SM 那一整块资源(寄存器堆、SMEM、warp 槽位、block 数上限),同时最多能塞得下几个像 Block 0 这样的线程块?**

**是什么:**
```python
from sm_occupancy import occupancy, SMLimits, H100_SM

occupancy(threads_per_block: int, regs_per_thread: int,
          smem_per_block_kb: float, limits: SMLimits = H100_SM) -> dict
# 返回 {"blocks_per_sm": .., "occupancy": .., "active_warps": .., "bottleneck": ..}
```

**一句话:** 一个 SM(Streaming Multiprocessor)上能同时驻留多少个线程块(thread block),被 5 个互相独立的硬件预算同时卡住(线程总数/warp 总数/寄存器总量/共享内存总量/硬件规定的 block 数上限),`occupancy()` 分别按每个预算算出"这个预算下最多能塞几个 block",取**最小值**——谁给出的数最小,谁就是瓶颈(`bottleneck`)。

**底层机制/为什么这样设计:** `H100_SM = SMLimits(max_threads_per_sm=2048, max_warps_per_sm=64, max_blocks_per_sm=32, max_regs_per_sm=65536, max_smem_per_sm_kb=228)`。五个约束,对应 `occupancy()` 里五行计算(`sm_occupancy.py:20-27`):

1. `blk_by_threads = max_threads_per_sm // threads_per_block` —— 线程总数约束,最朴素的"分母不能超过分子"。
2. `blk_by_warps = max_warps_per_sm // warps_per_block`,其中 `warps_per_block = (threads_per_block + 31) // 32` —— 硬件调度以 warp(32 线程一组)为单位,即使 block 线程数不是 32 的整数倍,也要**向上取整**占用一整个 warp 槽位(`(x+31)//32` 是"向上取整除法"的标准写法),这意味着 block size 没对齐 32 时,会有线程在空转但依然占着 warp 槽位。
3. `blk_by_regs = max_regs_per_sm // (regs_per_thread * threads_per_block)` —— 寄存器是 SM 上一块固定大小的寄存器堆,按线程独占分配,一个 block 用掉 `regs_per_thread * threads_per_block` 个,这是最容易被写 kernel 的人忽略、却最常见的占用率杀手(局部变量、循环展开都会推高 `regs_per_thread`)。
4. `blk_by_smem = max_smem_per_sm_kb // smem_per_block_kb` —— 共享内存同理,是显式声明的稀缺资源。
5. `blk_by_max = max_blocks_per_sm` —— 硬件本身还有一个"同时最多驻留多少个 block"的独立上限(H100 是 32),这一条和资源用量无关,哪怕前四条算出来的数字再宽松,也不能超过这个天花板。

`occupancy = active_warps / max_warps_per_sm`,其中 `active_warps = blocks_per_sm * warps_per_block`——占用率是"按 warp 算的百分比",不是"按 block 数算的百分比",这是容易读错的一个细节。

**AI 研究/工程场景:** Nsight Compute 这类 profiler 报出来的 "achieved occupancy" 和 "limiting factor",本质上就是这个函数在做的事情的真实硬件版本——调 Triton/CUDA kernel 的 launch config 时,发现"占用率上不去",第一反应就是对照这五个维度逐个排查:是不是 `regs_per_thread` 太高(减少局部变量/循环展开)?还是 SMEM 声明太大(调小 tile)?

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from sm_occupancy import occupancy, H100_SM

# 好 kernel: 256 线程/block, 32 寄存器/线程, 16KB SMEM -> 被"线程总数"卡住,占用率 100%
good = occupancy(256, 32, 16.0)
assert good == {"blocks_per_sm": 8, "occupancy": 1.0, "active_warps": 64, "bottleneck": "threads"}
assert H100_SM.max_threads_per_sm // 256 == 8    # 手动验证 blocks_per_sm 的来源

# 寄存器爆炸: 128 寄存器/线程(其它不变) -> regs 变成瓶颈,blocks_per_sm 从 8 掉到 2,占用率掉到 0.25
bad_regs = occupancy(256, 128, 16.0)
assert bad_regs["bottleneck"] == "regs"
assert bad_regs["blocks_per_sm"] == 2 and bad_regs["occupancy"] == 0.25

# SMEM 爆炸: 100KB/block -> smem 变成瓶颈(228KB/SM 只够放 2 个 block 多一点)
bad_smem = occupancy(256, 32, 100.0)
assert bad_smem["bottleneck"] == "smem"
assert bad_smem["blocks_per_sm"] == int(228 // 100) == 2

# 硬件上限本身可以是瓶颈: block 很小(32 线程=1 warp)、几乎不占寄存器/SMEM 时,
# 前四条约束都很宽松,是"最多同时驻留 32 个 block"这条硬件天花板接管
tiny_kernel = occupancy(32, 1, 0.1)
assert tiny_kernel["bottleneck"] == "max_blocks"
assert tiny_kernel["blocks_per_sm"] == H100_SM.max_blocks_per_sm == 32
assert tiny_kernel["occupancy"] == 0.5    # 32 block * 1 warp/block = 32 active warps / 64 max = 0.5
```

**面试怎么问 + 追问链:**
- **Q:** "为什么给每个线程多分配几个寄存器,会导致整体 kernel 的并行度(occupancy)下降?"—— 期望答"寄存器总量是 SM 上的固定预算,单线程用得越多,同时能驻留的线程/block 就越少"。
- **追问 1:** "SMEM 用量和寄存器用量都很小,但 occupancy 还是上不去,可能是什么原因?"—— 期望能想到硬件本身的 `max_blocks_per_sm` 上限(专门用来测试候选人是不是只知道"寄存器/SMEM 是瓶颈"这两个最常见的答案,漏掉了和资源用量完全无关的硬件硬上限)。
- **追问 2(区分度很高):** "occupancy 是不是越高越好?能不能举一个 occupancy 低但性能反而更好的例子?"—— 期望答"不一定,高 occupancy 只是'有更多 warp 可以在等内存时切换,掩盖延迟',不等于计算吞吐最优";一些 GEMM kernel 会故意用更多寄存器换更少的重复访存/更多的指令级并行(ILP),occupancy 只有 25% 也可能比占用率 100% 但频繁 stall 的版本快——这是 Volkov《Better Performance at Lower Occupancy》的核心论点(`learning/gpu-architecture/lectures/04-sm-occupancy.md` 也引用了这篇),是区分"背过概念"和"真正做过 kernel 调优"候选人的分水岭问题。

**常见坑:**
- 把 occupancy 当成唯一的性能指标,无脑追求 100%——见上面追问 2,高占用率只是"延迟掩盖能力强",不是"算力用得最满"。
- 混淆 block 和 warp 两个粒度:occupancy 的定义是按 warp 算的百分比(`active_warps / max_warps_per_sm`),不是"用了几个 block / 最多能放几个 block"这个更直观但不准确的理解。
- `threads_per_block` 没对齐 32 的整数倍时,`(threads_per_block+31)//32` 这个向上取整会让最后不满一个 warp 的线程依然占用整个 warp 槽位,变相浪费——这是"block size 最好是 32 的整数倍"这条经验规则的量化依据。

---

## 5. NVLink 拓扑带宽 —— `nvlink_topology.py::allreduce_time_ms`

**在讲拓扑和带宽之前,先补一句"为什么多卡训练需要梯度同步"的动机说明:** 多卡训练最常见的并行方式是**数据并行**——一个大 batch 被切成几份,每张卡分到不同的数据切片,各自独立跑完前向 + 反向,算出一份**只基于自己那份数据**的梯度。但所有卡上的模型参数必须始终保持完全一致(不然每张卡都在朝不同方向更新参数,训练几步之后模型就分裂成好几个不同的版本),所以每一步更新参数之前,都要把所有卡各自算出的梯度**加总、再除以卡数取平均**,让每张卡实际用来更新参数的是"全体数据切片的平均梯度",而不是只看到自己那一份数据算出的梯度——这个"把各卡梯度加总平均"的操作,就叫 **all-reduce**。下文(包括下面可运行例子代码注释里)出现的"梯度同步"和"all-reduce",指的都是同一件事;这一节要建的模型,是这个操作在不同硬件互联拓扑下大概要花多长时间,不展开 all-reduce 具体怎么实现。

**是什么:**
```python
from nvlink_topology import Topology, TOPOLOGIES, allreduce_time_ms, compare

allreduce_time_ms(topo: Topology, bytes_total: int) -> float    # ring all-reduce 耗时(ms)
compare(bytes_total: int) -> list[dict]                          # 遍历所有预设拓扑给出对比表
```

**一句话:** 多卡互联带宽建模的核心,是把"N 张卡一起做一次 all-reduce(梯度同步)要多久"归结成一个和卡数、单卡链路带宽都有关的公式——`time = 2*(N-1)/N * bytes / per_gpu_link_BW`,卡数越多,系数越接近 2(不会继续线性增长),真正决定耗时的是分母的链路带宽。

**底层机制/为什么这样设计:** `TOPOLOGIES` 预设 4 组(`nvlink_topology.py:15-20`):

| key | 卡数 | 单卡链路带宽(双向) | bisection 带宽 |
|---|---|---|---|
| dgx_a100_8 | 8 | 0.6 TB/s | 4.8 TB/s |
| dgx_h100_8 | 8 | 0.9 TB/s | 7.2 TB/s |
| gb200_nvl72 | 72 | 1.8 TB/s | 129.6 TB/s |
| pcie_8 | 8 | 0.064 TB/s | 0.5 TB/s |

Ring all-reduce 的通信量模型是分布式训练里的经典结果:环形拓扑下,不管有 N 张卡,每张卡发送+接收的总数据量趋近于原始数据量的 **2 倍**(系数是 `2*(N-1)/N`,N 越大这个系数越逼近 2,不会随卡数继续显著增长)——这也是"理论上可以持续加卡做数据并行,通信开销不会线性爆炸"的数学依据。真正决定 `allreduce_time_ms` 结果量级的是分母 `per_gpu_link_tb_s`:PCIe(0.064 TB/s)和 NVSwitch(0.9~1.8 TB/s)差了一到两个数量级,这是"训练大模型必须用 NVLink/NVSwitch 互联,不能只靠 PCIe"这条工程共识在代码里的量化依据。

**(读代码时发现的细节,建议记入本节"常见坑")** `allreduce_time_ms()` 函数体(`nvlink_topology.py:23-28`)只用到了 `topo.n_gpus` 和 `topo.per_gpu_link_tb_s` 两个字段,`diameter_hops` 和 `bisection_tb_s` 都**没有**被这个函数引用——它们只出现在 `compare()` 拼出来的对比表里,是纯粹的"展示型"字段,不参与耗时的实际计算。而且 4 个预设拓扑的 `diameter_hops` 全部是 1(包括 72 卡的 GB200 NVL72)——这个简化模型把"任意两张卡之间通信"都当成一跳可达(NVSwitch/网状全互联的抽象),没有真的建模更大规模集群里"跨机柜要经过多级交换机"这种多跳场景。这是一次很好的读代码练习:一个 dataclass 字段出现在数据定义里、甚至出现在某个函数的输出表里,不代表它真的参与了你正在看的那个计算逻辑,要具体去函数体里确认它是否真被读取。

**AI 研究/工程场景:** 这一条只需要点出和 `learning/cluster-networking/` 的关联——那边在这个基础上继续深入 NVLink/InfiniBand/NCCL 集群互联的细节(真实的 all-reduce/all-gather 集合通信实现、跨节点网络拓扑),这里的 `allreduce_time_ms` 只是一个刻画"链路带宽差距量级"的最简化模型,不重复展开。

**可运行例子:**
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
import inspect
from nvlink_topology import TOPOLOGIES, allreduce_time_ms, compare

# ring all-reduce 公式: 2*(N-1)/N * bytes / 单卡带宽
h100_8 = TOPOLOGIES["dgx_h100_8"]
t = allreduce_time_ms(h100_8, bytes_total=int(1e9))     # 1GB 梯度同步
assert round(t, 3) == 1.944

# 手动按公式重新算一遍,验证不是凑出来的数字
ring_factor = 2 * (8 - 1) / 8                             # = 1.75
bw_gb_s = 0.9 * 1000                                       # per_gpu_link_tb_s -> GB/s
expected = (1e9 / 1e9) / bw_gb_s * 1000 * ring_factor
assert abs(expected - t) < 1e-9

# PCIe 退化: 同样 8 卡,链路从 0.9TB/s 掉到 0.064TB/s,耗时暴涨 14 倍以上
pcie_8 = TOPOLOGIES["pcie_8"]
t_pcie = allreduce_time_ms(pcie_8, bytes_total=int(1e9))
assert t_pcie > 10 * t                                     # 实测比值 ~14.06x

# GB200 NVL72: 72 卡一个域,ring_factor 趋近 2(比 8 卡的 1.75 更接近上限),
# 但单卡带宽也翻了倍(1.8TB/s),综合下来比 8 卡 H100 反而更快
gb200 = TOPOLOGIES["gb200_nvl72"]
t_gb200 = allreduce_time_ms(gb200, bytes_total=int(1e9))
ring_factor_72 = 2 * (72 - 1) / 72
assert round(ring_factor_72, 3) == 1.972
assert t_gb200 < t

# 更真实的规模: 175B 参数模型的梯度,fp16(2 bytes/参数)全量同步要多久
bytes_175b_fp16 = 175_000_000_000 * 2       # 350GB
rows = compare(bytes_total=bytes_175b_fp16)
by_name = {r["topo"]: r for r in rows}
assert by_name["DGX-H100 8x NVSwitch"]["allreduce_ms"] < by_name["DGX-A100 8x NVSwitch"]["allreduce_ms"]
assert by_name["PCIe-only 8 GPUs"]["allreduce_ms"] > 9000   # 实测 9570.312ms,近 10 秒同步一次梯度

# 常见坑现场验证: diameter_hops/bisection_tb_s 只是展示字段,没有参与 allreduce_time_ms 的计算
assert all(topo.diameter_hops == 1 for topo in TOPOLOGIES.values())
src = inspect.getsource(allreduce_time_ms)
assert "bisection_tb_s" not in src and "diameter_hops" not in src
```

**面试怎么问 + 追问链:**
- **Q:** "为什么大模型多卡训练要用 NVLink/NVSwitch 而不是走 PCIe 互联?差距大概有多少?"—— 期望能说出"差一到两个数量级",最好能现场估算或回忆出具体倍数(这里实测是 ~14 倍)。
- **追问 1:** "ring all-reduce 的通信量为什么是 `2*(N-1)/N` 而不是简单地和 N 成正比?这个系数在卡数很多的时候会怎么变化?"—— 期望理解"趋近于 2,不会无限增长",这是"为什么理论上可以持续加卡做数据并行、通信开销不会线性爆炸"的数学基础。
- **追问 2(区分度很高):** "GB200 NVL72 把 72 张卡当成'一个超级 GPU'用,这句话具体体现在这个通信模型的哪个字段上?"—— 期望能提到 `diameter_hops=1`(72 张卡两两之间都被建模成一跳可达)和相应更高的 `per_gpu_link_tb_s`(1.8 TB/s),而不是泛泛地说"因为很厉害"。

**常见坑:**
- 见上面"底层机制":`diameter_hops` 和 `bisection_tb_s` 字段存在于 `Topology` 数据类、也出现在 `compare()` 的输出表里,但 `allreduce_time_ms()` 的计算完全不引用它们——不要以为"表里列出来的字段都参与了计算",具体要看函数体读了哪些字段。
- 把 ring all-reduce 的 `2*(N-1)/N` 系数错误理解成"和卡数成正比,卡越多通信开销越大"——恰恰相反,这个系数是**递减并收敛到 2** 的,真正的规模瓶颈通常在别处(比如更大集群的多跳网络延迟、bisection 带宽是否够用),不是这个 ring 系数本身。
