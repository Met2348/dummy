# 02 · Roofline 性能建模深挖(Roofline Performance Model)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批要回答的问题是:同一段 GPU 代码,为什么有的"跑不满算力"不是因为写得烂,而是天生就不该指望算力用满?面试官问"你的 kernel 峰值利用率只有 3%,是不是哪里写挂了",能不能不心虚地答上来——这四个知识点就是那套判断标准。

**本文与 `learning/gpu-architecture/` 的关系(差异化声明,复述自 00-roadmap.md,必须先明确):** 本文讲的是 [`learning/gpu-architecture/src/roofline.py`](../../learning/gpu-architecture/src/roofline.py)、[`roofline_original_minimal.py`](../../learning/gpu-architecture/src/roofline_original_minimal.py)、[`common.py`](../../learning/gpu-architecture/src/common.py) 和 [`capstone_roofline_zoo.py`](../../learning/gpu-architecture/src/capstone_roofline_zoo.py) 这**同一份代码**,但换一种讲法:每个知识点从"最笨的想法"讲起(比如先问"GPU 明明标了很高的算力,为什么很多算子实测利用率只有个位数",再引入 AI/ridge point 的解释),并且额外补两块源模块 README 没有的东西——**底层机制/为什么这样设计** 和 **面试怎么问+追问链**。再重申一遍源模块 README 已经强调过的事:**这些 `src/*.py` 不是可编译的真实 CUDA/Triton/CUTLASS kernel,是用可断言验证的纯 Python 数值/机制模拟去复现 roofline 这套性能模型的行为**——不需要 GPU、不需要 CUDA 工具链,这是 Windows 工作站上刻意的设计取舍,不是偷懒抄近道。

**环境声明:** 全部代码在仓库根目录 `.venv`(Windows 原生,Python 3.13)下实际跑通验证,纯 CPU、零第三方依赖——`roofline.py`/`common.py`/`roofline_original_minimal.py`/`capstone_roofline_zoo.py` 只依赖 `dataclasses`/`__future__`,秒级跑完。**注意一个和 torch-deep-dive 不同的地方:** 这几个脚本之间是用 `from common import ...` 这种**相对文件 import**(不是 pip 包),所以下面每个"可运行例子"如果你要复制出去跑,必须让当前工作目录是 `learning/gpu-architecture/src/`(或者把代码存成该目录下的一个 `.py` 文件再跑)——这一点和 `roofline.py` 自己的 `_self_test()` 能直接跑的原理相同:Python 解释器会自动把脚本所在目录插进 `sys.path[0]`。

**本篇统一结构(七步,与 torch-deep-dive/huggingface-deep-dive 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计**
4. AI 研究/工程场景
5. 可运行例子(带 assert,真在 `.venv` 里跑过)
6. **面试怎么问 + 追问链**
7. 常见坑

---

## 1. Arithmetic Intensity 与 Roofline 分类 —— 一个比值决定谁是瓶颈

**是什么:**
```python
# learning/gpu-architecture/src/roofline.py
@dataclass
class OpProfile:
    name: str
    flops: int           # 这个算子总共要做多少次浮点运算
    bytes_moved: int      # 这个算子总共要往/从 HBM 搬多少字节

    def ai(self) -> float:
        return self.flops / max(1, self.bytes_moved)


def analyze(op: OpProfile, gpu: GPUSpec) -> dict:
    ai = op.ai()
    achievable = roofline_flops(gpu, ai)
    bound = "compute" if ai >= gpu.ridge_point_bf16() else "memory"
    ...
```

**一句话:** Arithmetic Intensity(简称 AI,2009 年原论文里叫 *operational intensity*)= 总浮点运算次数 ÷ 总内存访问字节数——这个比值回答的问题是"每从 HBM 搬 1 字节数据,这个算子配得上做几次浮点运算",AI 低说明数据搬运跟不上,AI 高说明算力才是天花板。

**底层机制/为什么这样设计:** GPU 干活时同时占用两条独立的"生产线":一条是计算单元按峰值算力 `peak_FLOPS` 的速度算数字,另一条是 HBM 按峰值带宽 `peak_BW` 的速度搬字节。两条线路理想情况下并行重叠,一个算子实际耗的时间取决于哪条线路更慢:`time = max(flops/peak_FLOPS, bytes/peak_BW)`。倒过来看"每秒能算多少":`achievable_FLOPS = flops/time = min(peak_FLOPS, peak_BW × (flops/bytes)) = min(peak_FLOPS, peak_BW × AI)`——这就是 roofline 模型的核心不等式,下一节详细展开。AI 的价值在于,它把"这个算子对两条生产线各自的需求"压缩成**一个和绝对规模无关的比值**:不需要知道 flops、bytes 具体是多少,只要知道它们的比例,就能判断这个算子在某块 GPU 上到底被哪条线路拖累。这个"operational intensity"的说法,是本仓库 [`roofline_original_minimal.py`](../../learning/gpu-architecture/src/roofline_original_minimal.py) 特意保留的 2009 年原论文用词——原论文强调这是"过滤掉 cache 命中之后,真正打到 DRAM 上的强度",和今天泛泛而谈的"arithmetic intensity"略有语境差别,但公式和用法完全一致,下一节会用它复现原论文两台 AMD 机器的真实数字来交叉验证。

**AI 研究/工程场景:** LLM 推理的 decode 阶段(一次只生成 1 个新 token,矩阵乘法退化成"矩阵乘向量",即 GEMV)几乎必然是 memory-bound——要把整份权重从 HBM 完整读一遍,却只配上很少的乘加,这也是为什么"用 FP8/FP4 量化权重"对 decode 阶段特别划算:bytes 直接减半甚至减到 1/4,flops 完全不变,AI 成倍上涨。而训练阶段的大矩阵乘法(GEMM,方阵边长几千起步)AI 轻松上千,几乎必然 compute-bound,优化方向完全不同(要抠 tensor core 利用率,而不是抠内存带宽)。

**可运行例子:**
```python
from common import GPUS
from roofline import gemm_profile, attention_profile, layernorm_profile, analyze

h100 = GPUS["H100"]

# GEMV(decode 单 token 的典型形状):m=1,几乎是"读一大块权重,只做很少计算"
gemv = gemm_profile(1, 4096, 4096)
assert gemv.flops == 2 * 1 * 4096 * 4096          # 2*m*n*k,乘和加各算一次
assert round(gemv.ai(), 2) == 1.0                  # 每读1字节,只配了约1次浮点运算
r_gemv = analyze(gemv, h100)
assert r_gemv["bound_by"] == "memory"
assert r_gemv["utilization_pct"] < 1.0             # H100 峰值989 TFLOPS,这里连1%都用不上

# 大方阵 GEMM:同一个公式,形状一变,AI 立刻从个位数冲到上千
big_gemm = gemm_profile(4096, 4096, 4096)
assert round(big_gemm.ai(), 2) == 1365.33
r_big = analyze(big_gemm, h100)
assert r_big["bound_by"] == "compute"
assert r_big["utilization_pct"] == 100.0            # 峰值 TFLOPS 完全打满

# LayerNorm:flops 和 bytes 同阶增长(都正比于 n*hidden),AI 是个和矩阵大小无关的常数
ln = layernorm_profile(4096, 4096)
assert ln.ai() == 2.0                                 # 8*n*hidden / (2*2*n*hidden) = 2,恒定
r_ln = analyze(ln, h100)
assert r_ln["bound_by"] == "memory"
assert r_ln["utilization_pct"] < 5.0                  # 全篇里最惨的一类算子,怎么调都优化不动
```

**面试怎么问 + 追问链:**
- **Q:** "什么是 Arithmetic Intensity?它和一个算子能达到的 achievable FLOPS 是什么关系?"—— 期望答出"flops/bytes 的比值,决定这个算子会被哪条生产线(计算或内存)拖后腿"。
- **追问 1:** "AI 很高是不是就意味着这个算子跑得快?"—— 容易被问倒的一点:AI 高只代表**有可能**是 compute-bound,真正跑多快还要看这块 GPU 的 `peak_FLOPS` 本身大不大——AI 上千但 GPU 峰值算力很低,绝对速度依然可能很慢,AI 只回答"被谁限制",不回答"限制到多少"。
- **追问 2(容易被问倒):** "LayerNorm 为什么 AI 恒等于 2,和输入矩阵开多大完全没关系?"—— 需要现场推公式:`flops = 8*n*hidden`,`bytes = 2*dtype_bytes*n*hidden`,分子分母都正比于 `n*hidden`,这一项直接约掉,剩下纯常数,这是"读写次数和数据量同阶"这类算子(LayerNorm/Softmax/elementwise)的共性,不是 LayerNorm 独有。
- **追问 3(深挖):** "同一个算子(比如这里的 attention),换一块 GPU 跑,AI 会变吗?"—— 期望答"不会,AI 是算子自身的属性,纯靠 flops/bytes 公式在跑之前就能算出来;会变的是 `bound_by` 和 `utilization_pct`,因为这两个依赖 GPU 的 `ridge_point`"——这是下一节的核心,能提前说到说明理解是连贯的。

**常见坑:** 把 AI 和 utilization(利用率)混为一谈——AI 是算子固有属性(不依赖具体硬件),utilization 是"这个算子在某块具体 GPU 上,实际能用上多少比例的峰值算力",同一个算子换一块 GPU,utilization 会变但 AI 不变(下一节会看到具体例子)。另外容易忽略 FLOPs 的计算约定:这里 GEMM 用 `2*m*n*k`(乘法和加法各算一次),如果拿网上别的资料的数字直接比对,先确认对方是不是用了同一套约定,不然数字对不上不代表谁算错了。

---

## 2. Ridge Point 计算 —— 算力和带宽这两条"生产线"的交叉点

**是什么:**
```python
# learning/gpu-architecture/src/common.py
@dataclass(frozen=True)
class GPUSpec:
    bf16_tflops: float
    hbm_tb_s: float
    ...
    def ridge_point_bf16(self) -> float:
        """FLOP/byte at which compute and memory roofs meet (BF16)."""
        return (self.bf16_tflops * 1e12) / (self.hbm_tb_s * 1e12)


def roofline_flops(spec: GPUSpec, ai: float, dtype: str = "bf16") -> float:
    peak = {"bf16": spec.bf16_tflops, ...}[dtype]
    mem_bound = spec.hbm_tb_s * ai
    return min(peak, mem_bound)
```

**一句话:** ridge point(脊线/山脊点)= `peak_FLOPS / peak_bandwidth`,单位是 FLOP/byte——它是一条分界线:算子的 AI 没跨过这个数,不管这块 GPU 标称算力多强,都会先被内存带宽耗尽;跨过了,内存就绰绰有余,算力封顶。

**底层机制/为什么这样设计:** 上一节推出了 `achievable = min(peak_FLOPS, peak_BW × AI)`。把这两个分支画在以 AI 为横轴、achievable FLOPS 为纵轴的 log-log 图上,是两条线:一条水平线 `y = peak_FLOPS`(算力屋顶),一条过原点的斜线 `y = peak_BW × AI`(带宽屋顶,斜率就是带宽)。两条线的交点满足 `peak_FLOPS = peak_BW × AI_ridge`,解出 `AI_ridge = peak_FLOPS / peak_BW`——这正是"ridge"(屋脊)这个名字的来源:两片屋顶相交的那条线。

光靠文字描述这两条线不够直观,画出来是这样(纵横两轴都是对数刻度,`achievable = min(peak_FLOPS, peak_BW×AI)` 这条折线本身就是"roofline"这个名字字面对应的形状——像屋顶一样,一段斜坡爬到顶,之后就封顶变平):

```
achievable
 FLOPS
 (log轴)
   ^
   |                         ______________________________
   |                        /      <- 算力屋顶(水平线,平顶部分):y = peak_FLOPS
   |                       /           AI 再怎么涨,achievable 也被钉死在这条线上,
   |                      /            这个区域是 compute-bound(算力是天花板)
   |                     *  <- ridge point:两片屋顶的交点,AI = peak_FLOPS / peak_BW
   |                    /
   |                   /   <- 带宽屋顶(斜坡部分):y = peak_BW × AI
   |                  /        AI 每往右一格,achievable 跟着线性往上涨,
   |                 /         这个区域是 memory-bound(带宽是天花板)
   |                /
   +----------------+-----------------------------------------> AI = flops/bytes (log轴)
              ridge point 在这里
      (左边:AI小,memory-bound)      (右边:AI大,compute-bound)
```

这张图把"AI 和 ridge point 谁大谁小"这个纯数字比较,翻译成了"这个算子的坐标点落在斜坡上还是平顶上"——斜坡上(左边)意味着这个算子哪怕给它再强的算力也没用,因为它自己都还没跑到能喂饱这块算力的数据搬运速度;平顶上(右边)意味着数据搬运早就绰绰有余,是算力这一侧先见顶。AI 落在 ridge 左边,斜线更低、起决定作用(memory-bound);AI 落在 ridge 右边,水平线更低、起决定作用(compute-bound)。这套公式不是 GPU 专属的——[`roofline_original_minimal.py`](../../learning/gpu-architecture/src/roofline_original_minimal.py) 原样复现了 2009 年原论文里两台真实 AMD 服务器(Opteron X2/X4)的 ridge point,数字量级和 GPU 完全不同(个位数 FLOP/byte,而不是几百),但公式一模一样,说明 roofline 是"任何有独立计算单元和独立内存带宽的系统"通用的性能模型,GPU 只是这套模型今天最常见的应用场景。

**AI 研究/工程场景:** 拿到一块新 GPU 的参数表,第一件事算出它的 ridge point,能立刻建立一个"分界直觉"——不用真的把每个算子都跑一遍 benchmark,先看算子的 AI 落在 ridge 哪一侧,就能预判优化该往哪使劲:AI 远小于 ridge,该做的是减少 HBM 流量(融合算子、量化、改数据布局);AI 远大于 ridge,该做的是提升计算效率(更大的 tile、tensor core 利用率、减少 warp divergence)。这也是为什么 `GPUSpec` 里这个方法特意叫 `ridge_point_bf16` 带精度后缀——换一档精度(FP8/FP4),`peak_FLOPS` 会变,ridge point 也会跟着变,不是 GPU 的单一固定属性。

**可运行例子:**
```python
from common import GPUS, roofline_flops

h100 = GPUS["H100"]

# ridge point 就是 peak_flops / peak_bandwidth,单位 FLOP/byte
rp = h100.ridge_point_bf16()
assert rp == (h100.bf16_tflops * 1e12) / (h100.hbm_tb_s * 1e12)
assert round(rp, 1) == 295.2

# 恰好卡在 ridge point 上:两个分支(peak 和 BW×AI)理论上应该相等
assert abs(roofline_flops(h100, rp) - h100.bf16_tflops) < 1e-6

# ridge point 左边(AI 更小):内存这条"水管"先耗尽,achievable 打不满 peak
assert roofline_flops(h100, rp - 50) < h100.bf16_tflops

# ridge point 右边(AI 更大):内存水管早就够用,峰值被计算这条水管封顶
assert roofline_flops(h100, rp + 50) == h100.bf16_tflops

# 交叉验证:同一个公式,换成 2009 年原论文的两台 AMD 机器,数量级完全不同,但公式不变
from roofline_original_minimal import OPTERON_X2, OPTERON_X4
assert round(OPTERON_X2.ridge_point(), 2) == 1.17     # 17.6 GFLOPS / 15.0 GB/s
assert round(OPTERON_X4.ridge_point(), 2) == 4.46     # 74.0 GFLOPS / 16.6 GB/s
```

**面试怎么问 + 追问链:**
- **Q:** "Ridge point 是怎么算出来的?物理含义是什么?"—— 期望现场推出 `peak_FLOPS = peak_BW × AI_ridge` 这个交点方程,而不是死背 `peak/BW` 这个公式。
- **追问 1:** "如果同一块 GPU 换成用 FP8 而不是 BF16,ridge point 会变吗?"—— 期望答"会变,FP8 的 `peak_FLOPS` 通常比 BF16 高一截(比如 H100 是 989→1979),HBM 带宽这个硬件属性不随精度变,所以 ridge point 会跟着精度往上/往下移"。
- **追问 2(容易被问倒):** "Ridge point 的单位为什么是 FLOP/byte,这个单位有什么讲究?"—— 期望推出 `peak_FLOPS` 单位是 FLOP/s,`peak_BW` 单位是 byte/s,相除单位正好是 FLOP/byte——**这和 AI 的单位完全一致**,不是巧合,是两者能够直接比大小、判断左右的前提,单位对不上的话"比较 AI 和 ridge point 谁大谁小"这件事本身就没有意义。
- **追问 3(开放,考察泛化):** "Roofline 模型是不是只能用来分析 GPU?"—— 期望能提到"这套模型本质是任何'计算单元 + 独立内存带宽'的系统都适用,`roofline_original_minimal.py` 复现的就是 2009 年论文里两台 CPU 服务器,不是 GPU",说明理解的是模型本身而不是死记 GPU 场景。

**常见坑:** 把左右记反——有人凭直觉觉得"AI 大 = 数据量大 = 更依赖内存 = 该在 ridge 左边",实际相反:AI 是"计算量/数据量"的比值,AI 越大代表相对于读的字节数,做的计算越多,是**更**compute-bound,不是更 memory-bound。另外容易忘记 ridge point 是"和精度绑定"的一个值,不是 GPU 的单一固定属性——`B200` 同时有 BF16/FP8/FP4 三档峰值,理论上对应三个不同的 ridge point,`common.py` 里只实现了 `ridge_point_bf16()` 这一档,不代表其他精度不存在对应的 ridge point。

---

## 3. Ridge Point 非单调的反直觉发现(本节重点):H200 比 H100 反而更低

**是什么:** 直接对比 `GPUS["H100"].ridge_point_bf16()` 和 `GPUS["H200"].ridge_point_bf16()` 的真实输出——这是我在 46 模块 runbook 验证任务里亲自跑 `learning/gpu-architecture/` 时独立发现、并写进该模块 README 的一条真实洞察,本节重新独立验证一遍,不照抄任何记忆数字。

**一句话:** H100 → H200 这次代际升级里,ridge point **没有**像"新一代=更强"的直觉那样上升,反而从 295.2 FLOP/byte 降到 206.0 FLOP/byte(降幅约 30.2%)——因为这次升级动的是显存带宽(HBM3→HBM3e),不是计算单元。

**底层机制/为什么这样设计(独立验证过程,数字均为 `.venv` 实跑输出):**

先看 `common.py::GPUS`(GPU 规格数据表,00-roadmap.md 里称它 `GPU_CATALOG` 是概念上的叫法,源码里实际变量名是 `GPUS`)里这两行的原始规格:

| GPU | BF16 TFLOPS(峰值算力) | HBM 带宽 | Ridge Point |
|---|---:|---:|---:|
| H100-SXM5 | 989.0 | 3.35 TB/s | 295.2 FLOP/byte |
| H200-SXM | 989.0 | 4.80 TB/s | 206.0 FLOP/byte |
| 变化幅度 | **+0%(一个数字都没变)** | **+43.3%** | **−30.2%** |

推理过程,不跳步骤:
1. `ridge_point = peak_FLOPS / peak_bandwidth`,这是一个**比值**,不是绝对值。
2. 分子(`bf16_tflops`)这一项,H100 和 H200 完全相等,都是 989.0——实跑验证:`h100.bf16_tflops == h200.bf16_tflops == 989.0` 成立。这不是四舍五入巧合,是 `common.py` 数据表里这两行 `bf16_tflops` 字段本来就填的同一个数字。
3. 分母(`hbm_tb_s`)这一项,H200 从 3.35 涨到 4.80,涨幅 `4.80/3.35 = 1.4328...`,即 **+43.3%**。
4. 分子不动、分母涨了 43.3%,比值必然跌:`1/1.4328 = 0.6979`,跌到原来的 69.8%,即 **−30.2%**——实跑验证 `rp_h100 / bw_ratio` 和 `rp_h200` 在浮点误差范围内精确相等,这不是"大概如此",是纯粹的除法关系。
5. 背后的硬件原因:H100 和 H200 用的是**同一颗 Hopper 计算核心**(SM 数量、tensor core 设计完全没有重新设计),H200 真正换的是显存本体——从 HBM3 换成容量更大、带宽更高的 HBM3e(容量 80GB→141GB,带宽 3.35→4.80 TB/s)。这是一次"同代内存刷新"(refresh),不是"新一代计算架构跃迁",所以算力峰值这一项原地不动是符合硬件事实的,不是数据表打错了。

顺手用同一份数据表交叉检查了另外两代,确认这不是 H200 一个孤例,而是"ridge point 由分子分母各自涨多少决定"这个规律的普遍体现:B200 相对 H100,算力涨了 2.275 倍(989→2250),带宽涨了 2.388 倍(3.35→8.00),带宽依然涨得比算力快一点点,所以 B200 的 ridge point(281.2)也比 H100(295.2)略低——虽然 B200 从任何一个绝对指标看都比 H100 强得多。而 GB200(算力再往上提到 2500,带宽和 B200 一样是 8.00)算力涨幅重新超过带宽涨幅,ridge point 回升到 312.5,反而超过了 H100。**结论:ridge point 不是随"代数"单调变化的量,纯粹取决于这一次具体升级里,算力和带宽两个维度各自涨了多少、谁涨得更快。**

**AI 研究/工程场景:** 这不是纯数字游戏,直接影响"要不要升级到 H200"这类真实采购/选型决策。如果工作负载是 compute-bound(比如大 batch 训练的方阵 GEMM,AI 动辄上千,远超两者的 ridge point),H100 换成 H200 对速度**没有任何帮助**,因为算力峰值一个数字都没变;H200 真正的价值在两处:①绝对显存容量更大(141GB vs 80GB),能放下更大的 KV cache/batch;②ridge point 更低意味着"更容易触达 compute-bound"——原本卡在 H100 ridge point(295.2)和 H200 ridge point(206.0)之间的算子,升级后会从 memory-bound 翻转成 compute-bound。这不是纸上谈兵:用上一节 `analyze()` 实跑了一个 32k 超长上下文的 attention(`attention_profile(1, 32, 32768, 128)`,AI=252.06)——这个 AI 恰好卡在两个 ridge point 中间,在 H100 上是 memory-bound(252.06 < 295.2,利用率 85.4%),同一个算子换到 H200 上直接翻转成 compute-bound(252.06 ≥ 206.0,利用率 100%)。GPU 没有把这个算子"加速"多少倍,只是 H200 的 ridge point 本身更低,够到 compute-bound 门槛更容易——下一节的 capstone 会看到这类翻转在批量结果里同样成立。

**可运行例子:**
```python
from common import GPUS
from roofline import attention_profile, analyze

h100 = GPUS["H100"]
h200 = GPUS["H200"]

# 算力这一项:H100 -> H200 一个数字都没变
assert h100.bf16_tflops == h200.bf16_tflops == 989.0

# 带宽这一项:H200 从 3.35 涨到 4.80 TB/s,涨了 43.3%
bw_ratio = h200.hbm_tb_s / h100.hbm_tb_s
assert round(bw_ratio, 3) == 1.433

# ridge point = 算力/带宽,分子不动、分母涨,比值必然跌
rp_h100 = h100.ridge_point_bf16()
rp_h200 = h200.ridge_point_bf16()
assert round(rp_h100, 1) == 295.2
assert round(rp_h200, 1) == 206.0
assert rp_h200 < rp_h100                               # 反直觉:更新的卡,ridge point 反而更低
assert abs(rp_h100 / bw_ratio - rp_h200) < 1e-9         # 精确验证这就是纯粹的除法关系,不是巧合

# 不只是数字游戏:同一个 attention 算子,在两款卡上的 bound_by 分类结果直接翻转
attn_32k = attention_profile(1, 32, 32768, 128)          # 32k 超长上下文
r_h100 = analyze(attn_32k, h100)
r_h200 = analyze(attn_32k, h200)
assert round(attn_32k.ai(), 2) == 252.06
assert r_h100["bound_by"] == "memory"                    # 252.06 < 295.2(H100 ridge)
assert r_h200["bound_by"] == "compute"                   # 252.06 >= 206.0(H200 ridge)
```

**面试怎么问 + 追问链:**
- **Q:** "H100 升级到 H200,对训练和推理分别有什么影响?"—— 这是一道能立刻分辨"背过参数表"还是"理解 ridge point"的问题。
- **追问 1(杀伤力很强):** "H200 的 BF16 算力(FLOPS)相比 H100 涨了多少?"—— 正确答案是**完全没涨,一个数字都没变**,只有显存容量和带宽涨了;很多候选人会下意识回答"肯定也涨了不少",因为"新一代"这个标签本身就在暗示"全面更强"。
- **追问 2(深挖):** "能不能用 ridge point 这个概念,量化说明 H200 这次升级的价值主要体现在哪?"—— 期望答出 ridge point 从 295.2 降到 206.0,意味着"原本 AI 落在 206~295 这个区间、在 H100 上是 memory-bound 的算子",到了 H200 上会翻转成 compute-bound——升级的价值不是"让已有的 compute-bound 算子更快"(算力没变,不会更快),而是"让更多原本卡在中间地带的算子更容易打满算力"。
- **追问 3(开放/工程判断):** "如果你所在的团队要采购新一代 GPU,只看官方宣传的'算力提升 X 倍'这一个数字,够不够做决策?你会建议额外看什么?"—— 没有标准答案,考察的是候选人有没有"分别拆开算力和带宽两个维度、自己算一下 ridge point 和自己的工作负载 AI 关系"这个工程直觉,而不是被市场宣传数字牵着走。

**常见坑:** 想当然认为"新一代 = 所有指标都变好",看到"H200"这种听起来更新的型号就默认它"全面碾压"H100——真实情况是这次升级只精准针对了显存子系统。另一个坑是把"ridge point 更低"直接等同于"这块卡更差"——恰恰相反,ridge point 更低说明这块卡**更容易触达 compute-bound**,是不是好事完全取决于你的工作负载 AI 落在哪个区间,对纯 memory-bound 的算子(比如 LayerNorm,AI 恒为 2,离两个 ridge point 都远得很),真正起作用的是 H200 绝对带宽(4.8 TB/s)比 H100 高,而不是 ridge point 这个比值本身变没变。还有一个更隐蔽的坑:直接拿四舍五入后的 295.2 和 206.0 做二次计算(比如手算比例)会有精度损失,应该像上面例子里那样用未取整的原始浮点数做比例验证。

---

## 4. Capstone:10 op × 4 GPU roofline zoo —— 把公式批量套在真实算子清单上

**是什么:**
```python
# learning/gpu-architecture/src/capstone_roofline_zoo.py
WORKLOADS = [
    ("gemv-1x4k-4k", gemm_profile(1, 4096, 4096)),
    ("gemm-2k-2k-2k", gemm_profile(2048, 2048, 2048)),
    ("gemm-8k-8k-8k", gemm_profile(8192, 8192, 8192)),
    ("attn-b1-h32-s2k", attention_profile(1, 32, 2048, 128)),
    ("attn-b1-h32-s32k", attention_profile(1, 32, 32768, 128)),
    ("layernorm-4k-4k", layernorm_profile(4096, 4096)),
    ("layernorm-4k-8k", layernorm_profile(4096, 8192)),
    ("gemm-4k-4k-128", gemm_profile(4096, 4096, 128)),
    ("gemm-128-4k-4k", gemm_profile(128, 4096, 4096)),
    ("gemm-32k-32k-32k", gemm_profile(32768, 32768, 32768)),
]

def run() -> list[dict]:
    results = []
    for gpu_name in ["A100", "H100", "H200", "B200"]:
        gpu = GPUS[gpu_name]
        for op_name, op in WORKLOADS:
            r = analyze(op, gpu)
            ...
    return results   # 10 op × 4 GPU = 40 组
```

**一句话:** 把前三节的公式(AI、ridge point、`analyze()`)批量套在一张"10 个真实 LLM 算子 × 4 款 GPU = 40 个组合"的表格上,一次性跑出每个组合的 AI、achievable TFLOPS、利用率、是 compute-bound 还是 memory-bound——不引入新公式,是前三节方法论的自动化、规模化验证。

**底层机制/为什么这样设计:** 真实的模型训练/推理不会只有一种算子——一次前向传播里同时有 attention 的 `QK^T`/`softmax·V`、MLP 的大 GEMM、LayerNorm,形状(batch/seq/hidden)也各不相同。"我的模型整体是不是内存受限"这个问题没有单一答案,必须对每个算子分别套公式再汇总。这份 `WORKLOADS` 清单(读源码确认)覆盖了几类典型退化形状:1 个 GEMV(`m=1`,模拟 decode 单 token)、5 个不同形状的 GEMM(2048³/8192³/32768³ 三个方阵,外加 `4096×4096×128` 和 `128×4096×4096` 两个"瘦"GEMM,模拟 batch 太小或 K 维度太小的场景)、2 个 attention(seq=2048 和 seq=32768,模拟短/长上下文)、2 个 LayerNorm(不同 hidden size)——刚好 10 个算子,乘上 `["A100", "H100", "H200", "B200"]` 4 款 GPU,共 40 组,和源码 `run()` 的双重循环完全对应。

**AI 研究/工程场景:** 实跑 40 组里 25 组(62.5%)是 memory-bound——这个比例是工业界"为什么要拼命做算子融合(kernel fusion)、拼命上量化"的数据支撑:多数 LLM 算子的瓶颈根本不在算力,单纯换算力更强的 GPU 对这些算子几乎没用,真正有效的杠杆是减少 HBM 流量(相邻算子融合、降精度、改数据布局)。同时,三个方阵 GEMM(2048³/8192³/32768³)在全部 4 款 GPU 上清一色 compute-bound、利用率接近或等于 100%——这是训练大模型 backbone 时最理想的情况,也是为什么 kernel 工程里会特意避免"瘦" GEMM 形状(小 K 维度、小 batch)。

**可运行例子:**
```python
from capstone_roofline_zoo import run, summarize, WORKLOADS

assert len(WORKLOADS) == 10                              # 10 个算子

results = run()                                            # 10 op x 4 GPU = 40 组
s = summarize(results)
assert s["total"] == 40
assert s["memory_bound"] == 25
assert s["compute_bound"] == 15
assert s["memory_bound"] > s["compute_bound"]              # 多数 LLM 算子是内存受限的
assert s["h100_big_gemm_util"] > 80.0                       # 8192^3 大 GEMM 在 H100 上几乎打满峰值

# 呼应上一节:同一个算子(32k 超长 attention),在 capstone 的批量结果里同样是
# H100 memory-bound、H200 compute-bound —— 不是巧合,是 ridge point 数字的直接后果
r_h100 = [r for r in results if r["gpu"] == "H100" and r["op"] == "attn-b1-h32-s32k"][0]
r_h200 = [r for r in results if r["gpu"] == "H200" and r["op"] == "attn-b1-h32-s32k"][0]
assert r_h100["bound_by"] == "memory"
assert r_h200["bound_by"] == "compute"
```

**面试怎么问 + 追问链:**
- **Q:** "如果给你一个新模型完整的前向传播算子清单,你会怎么系统性判断这个模型在某款 GPU 上的性能瓶颈在哪?"—— 期望答出"给每个算子按公式算 flops/bytes 得到 AI,和这块 GPU 的 ridge point 比较,逐个分类再汇总统计",而不是只跑一次端到端 benchmark 看总耗时、或者凭经验猜。
- **追问 1:** "清单里 `gemm-4k-4k-128` 和 `gemm-128-4k-4k` 这两个'瘦' GEMM 的 AI 是多少?和方阵 GEMM 比说明了什么?"—— 实跑两者 AI 完全相等(120.47),因为 `bytes_moved` 只取决于 `{m×k, k×n, m×n}` 这个无序三元组,两次调用只是交换了 m 和 k 的角色,三元组的值集合不变;这个 AI 比方阵 GEMM(千级)低一个数量级,说明"batch 太小"和"K 维度太小"是两种表现不同、但同样会拖累 AI 的退化模式,不是只有"矩阵整体偏小"这一种情况。
- **追问 2(深挖,串联知识点 3):** "同一个算子,换一块 GPU 会不会分类结果不一样?"—— 期望能举出 capstone 里两个真实例子:①`attn-b1-h32-s2k`(AI=204.80)在 A100 上是 compute-bound(A100 ridge=153.0,204.80≥153.0),换到 H100 上反而变成 memory-bound(H100 ridge=295.2,204.80<295.2);②上一节验证过的 `attn-b1-h32-s32k` 在 H100/H200 上的翻转。说明分类结果永远是"算子 AI"和"GPU ridge point"两者共同决定的,任何一边变了结论都可能变,不存在"这个算子天生就是 memory-bound"这种脱离具体硬件的绝对结论。
- **追问 3(开放题):** "40 组里 25 组 memory-bound,能不能就此得出'当前一代 GPU 算力过剩'的结论?"—— 期望能反驳:这个比例高度依赖 `WORKLOADS` 这份清单本身的构成(比如清单里刻意放了 2 个 LayerNorm 和 1 个 GEMV,天然 AI 就低),capstone 给出的是"方法论怎么批量应用"的演示,不是"放之四海皆准的行业结论",真实结论要看自己模型实际的算子分布去跑一遍同样的流程。

**常见坑:** 把"利用率(util)低"直接等同于"这个算子的 kernel 实现写得差"——util 低更多时候是算子的数学本质决定的(LayerNorm 这类"读多少数据只配得上做常数次运算"的算子,AI 理论上限就是 2,无论 kernel 实现多精妙,util 都不可能高),而不是实现质量问题,面试里被问到"为什么这个算子利用率这么低"时,先看 AI 和 ridge point 的关系,而不是先怀疑代码写挂了。另一个坑是混淆"这批 40 组结果"和"通用行业规律"——`avg_util_pct=59.2%` 这类汇总数字只对这份特定的 10-op 清单成立,换一批算子形状(比如加几个 batch 很大的 GEMM)整体比例会明显不同,capstone 演示的是"怎么算",不是"标准答案该是多少"。
