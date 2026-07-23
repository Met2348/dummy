# 06 · 集群网络深挖(Cluster Networking)

> 总览见 [00-roadmap.md](00-roadmap.md)

05 号文件讲的是单卡内部——线程怎么调度、内存怎么访问。本文把粒度从"卡内"升到"卡间":千卡训练时,梯度怎么在 GPU 之间搬运,直接决定了堆再多卡是否真的能线性加速。对应 `learning/cluster-networking/`(Module 8 第 4 专题,6 lecture + 7 个 src 源文件,核心论文 Hu et al. *Demystifying NCCL*)。7 个知识点。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Python 3.13)下用 `.venv/Scripts/python.exe` 实际跑通验证。7 个源文件零第三方依赖(`dataclasses`/`math`/`__future__`+互相import),纯 CPU、秒级完成——**这里必须再次强调本模块的定位**:`src/` 下没有 `torch.distributed`、没有 socket/多进程通信,是用可断言验证的纯 Python 时延/带宽解析模型复现集群网络的关键工程决策(链路选型、all-reduce 算法选择、拓扑设计、SHARP 硬件卸载),不是真实的分布式训练代码。

---

## 1. 链路时延带宽建模(`common.py`)—— 一张表里的 7 条互联技术,以及为什么 IB 比 NVLink 慢 9 倍不止

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Link:
    name: str
    bw_gb_s: float       # one direction
    latency_us: float

LINKS = {
    "nvlink4":    Link("NVLink 4 (per GPU)", 450.0,   0.5),
    "ib_ndr":     Link("IB NDR 400G",         50.0,   1.5),
    "eth_100g":   Link("Ethernet 100G",       12.5,   5.0),
    # ... 完整表还有pcie5_x16/nvlink5/ib_xdr/roce_400g,共7条
}

def time_to_send(bytes_total: int, link: Link) -> float:
    """Time in microseconds, including 1-way latency."""
    transfer_us = (bytes_total / 1e9) / link.bw_gb_s * 1e6
    return link.latency_us + transfer_us
```
(`common.py:1-27`,节选)

**一句话:** 任何一次数据传输的耗时都是"固定延迟+数据量/带宽"这个最基本的公式,`LINKS` 表把 7 种真实存在的互联技术(卡内 NVLink、节点内 PCIe、节点间 InfiniBand/RoCE/以太网)的真实带宽/延迟数量级摆在一起,数字本身就是"为什么大模型训练要用 NVLink+IB 而不是普通以太网"这个问题最直接的证据。

**底层机制/为什么这样设计:** `time_to_send` 公式里 `latency_us`(固定开销,和数据量无关)和 `transfer_us`(随数据量线性增长)是两个独立的项——这意味着传小消息时延迟项主导(带宽再高也没用,因为数据量太小体现不出带宽优势),传大消息时带宽项主导(此时延迟的那零点几微秒可以忽略不计)。7 条链路的具体数字差距揭示了硬件层级的现实:NVLink(卡间直连,450GB/s)比 IB NDR(节点间网络,50GB/s)带宽高 9 倍,延迟也更低(0.5μs vs 1.5μs)——这不是厂商随意定价,是物理走线距离和协议栈复杂度的直接体现:NVLink 是同一台服务器内部的专用高速互联,IB 需要经过网卡、交换机等更长的物理路径和更复杂的协议处理。即便物理路径更长,IB(以及后面知识点会提到的 RoCE)的延迟依然远低于普通以太网 TCP/IP,原因是两者都实现了 RDMA(Remote Direct Memory Access,远程直接内存访问)——网卡直接读写远程主机的内存,数据搬运绕过对方的 CPU 和操作系统内核网络协议栈,不需要像普通 TCP/IP 那样经过内核多次拷贝和上下文切换,这正是 RDMA 类链路(IB、RoCE)比普通以太网延迟低一个数量级的核心原因,也是知识点 5 SHARP 交换机内聚合技术能够成立的前提之一——没有 RDMA 这种"网卡/交换机直接访问内存"的能力,就没办法把规约计算下沉到网络设备里、绕开 CPU 参与。

**AI 研究场景:** 这张表是任何"应该把哪些并行策略放在哪个通信层级"决策的数据基础——02 号文件讨论的张量并行(TP)对通信延迟极度敏感,只能放在 NVLink 直连的同节点内(见知识点 2 会展开这个具体权衡),数据并行的梯度同步可以容忍节点间 IB 网络的更高延迟,这个层级划分的物理原因就是这张表里的数字差距。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from common import LINKS, time_to_send

t_nvlink = time_to_send(int(1e9), LINKS["nvlink4"])   # 1GB via NVLink
t_ib = time_to_send(int(1e9), LINKS["ib_ndr"])          # 1GB via IB NDR
assert 2000 < t_nvlink < 2500       # 约2.2ms
assert t_ib > 8 * t_nvlink            # IB比NVLink慢至少8倍

# 独立验证: 延迟主导 vs 带宽主导两种场景下,"IB比NVLink慢多少倍"这个比值本身会变化
tiny_nvlink = time_to_send(100, LINKS["nvlink4"])   # 100字节,延迟主导
tiny_ib = time_to_send(100, LINKS["ib_ndr"])
ratio_tiny = tiny_ib / tiny_nvlink
ratio_big = t_ib / t_nvlink
assert ratio_tiny < ratio_big    # 小消息时,IB的劣势被"只是延迟差3倍"这件事显著缩小
print(f"1GB传输: NVLink={t_nvlink:.0f}us  IB={t_ib:.0f}us  比值={ratio_big:.1f}x")
print(f"100B传输: NVLink={tiny_nvlink:.2f}us  IB={tiny_ib:.2f}us  比值={ratio_tiny:.1f}x(远小于大消息时的比值)")
```

**实测(`.venv` 真跑):** 1GB 数据通过 NVLink 4 传输耗时 **2223μs**(约2.2ms),通过 IB NDR 耗时 **20002μs**(约20ms),比值恰好约 **9.0倍**,和带宽比值(450/50=9)几乎精确对应(大消息场景下延迟项可忽略,传输时间几乎完全由带宽决定)。独立验证补充了一个重要限定:换成 100 字节的极小消息,NVLink 耗时 0.50μs(几乎全部是延迟项,`0.5+100/450e9*1e6≈0.50`),IB 耗时 1.50μs(同理几乎全是延迟项),比值只有 **3.0倍**(等于延迟本身的比值 1.5/0.5,不是带宽比值 9倍)——这条独立验证证实了"IB 比 NVLink 慢多少倍"不是一个固定数字,取决于消息大小落在延迟主导区还是带宽主导区,这是通信优化里"小消息聚合成大消息批量发送"这条常见优化手段的数量依据。

**面试怎么问 + 追问链:**
- **Q:** "为什么 NVLink 的延迟(0.5μs)比 IB(1.5μs)更低,仅仅是因为带宽更高吗?"—— 期望:不是同一个原因——带宽和延迟是两个独立的物理/工程指标,NVLink 延迟更低主要是因为它是芯片间直连(不经过网卡、PCIe 总线、交换机等中间环节),协议栈也更简单;IB 虽然本身是为低延迟设计的网络协议(相比更慢的以太网已经优化很多),但終究要经过网卡处理、可能经过交换机跳转,物理路径比卡间直连长得多。
- **追问1:** "如果要新增一条'第 8 条链路'到这张表里,还需要补充哪些字段才能让 `time_to_send` 之外的其他函数(如知识点 2 的 all-reduce 算法)正常工作?"—— 期望:只需要 `name`/`bw_gb_s`/`latency_us` 三个字段就足够——本模块所有下游函数(all-reduce 各算法、collective 通信、SHARP)都只依赖这三个基础属性做时延计算,没有依赖任何链路专属的额外字段,这是"公共接口最小化"的一个体现:所有更高层的通信模式建模都收敛到"给定带宽和延迟,能算出多久"这一个统一抽象上。

**常见坑:** `time_to_send` 计算的是**单向**传输时间(函数注释明确写"one direction"),而 `LINKS` 表里的注释显示部分链路(如 nvlink4 的 450 GB/s)标注是"单向"带宽、部分厂商宣传页给出的是"双向合计"带宽(注释里"900 GB/s bidir"就是双向数字)——读这张表时如果不确认清楚"这是单向还是双向"就直接套用,会在估算实际吞吐时产生 2 倍的系统性误差,这是网络硬件规格表里非常容易踩的一个常见坑,不是这份代码特有的问题。

---

## 2. All-Reduce 算法家族:Ring / Tree / Halving-Doubling(`allreduce_algos.py`)—— 三种算法的步数公式,以及为什么大消息场景 HD 总是赢

**承接 torch-deep-dive/09,不重新讲地基:** 为什么分布式训练需要"同步梯度"这件事本身,以及 ring all-reduce 具体怎么用"环形拓扑 + scatter-reduce/allgather 两阶段"把通信量摊平到和 N 基本无关,[torch-deep-dive/09](../torch-deep-dive/09-distributed-training-basics.md) 知识点 3(all-reduce 梯度同步机制)已经讲过并用纯 Python 手写验证过——这里直接承接那份已经验证过的理解,不重新讲一遍"为什么需要同步"和"ring 算法本身怎么运作",只往下展开 Ring/Tree/Halving-Doubling **三种**算法之间步数、单步传输量的性能特征对比,以及大规模集群场景下该怎么选。

**是什么:**
```python
from __future__ import annotations

def ring_allreduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Bandwidth-optimal. T = 2(N-1)/N * size/BW. Latency = 2(N-1) hops."""
    bw_per_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    n_steps = 2 * (n_gpus - 1)
    return n_steps * (link.latency_us + bw_per_step)

def halving_doubling(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Rabenseifner: reduce-scatter halving + all-gather doubling. Best for big msg."""
    import math
    n_steps = 2 * int(math.ceil(math.log2(n_gpus)))
    bw_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return n_steps * (link.latency_us + bw_step)

def pick_algorithm(n_gpus: int, bytes_total: int, link: Link) -> str:
    """NCCL-like crossover heuristic."""
    if bytes_total < 1024: return "tree"
    if n_gpus <= 8: return "ring"
    return "halving_doubling"
```
(`allreduce_algos.py:6-35`,节选)

**一句话:** 三种 all-reduce 算法在"步数随 GPU 数怎么增长"和"每步传输量是否随 GPU 数分摊"这两个维度上组合出不同的性能特征——Ring 步数随 N 线性增长但每步流量已经分摊(`O(N)` 步,BW-optimal);Tree 步数是 `O(logN)`(latency-optimal)但每步流量**不**分摊;Halving-Doubling(HD,即 Rabenseifner 算法)是"两头都占"的组合:`O(logN)` 步 + 每步流量分摊,在大 N、大消息场景下同时具备 Tree 的低步数和 Ring 的高分摊,是唯一一个在两个维度都不吃亏的算法。

**底层机制/为什么这样设计:** Ring 算法的物理直觉是把 N 个 GPU 排成一个环,每个 GPU 只和左右邻居通信,`2(N-1)` 步里前 `N-1` 步做 reduce-scatter(每个 GPU 逐步累积一部分梯度的完整和),后 `N-1` 步做 all-gather(把各自算好的部分结果广播给所有人)——因为每一步传输的都是"总数据量/N"这一小份,总的每卡流量是固定的(不随 N 增长),这是"BW-optimal"的含义;Tree 算法把 GPU 组织成二叉树结构,根节点收集所有叶子的贡献,只需要 `O(logN)` 层,但树的靠近根部的节点要处理来自多个子树的**完整**数据量(不像 Ring 那样天然分摊),所以大消息场景下 Tree 的带宽项会随层数积累开销;Halving-Doubling 巧妙地结合了两者——用 `O(logN)` 步完成(继承 Tree 的步数优势),但每一步只传输 `总量/N` 这一份分摊后的数据(继承 Ring 的分摊优势),`allreduce_algos.py:24-25` 的公式 `n_steps = 2*ceil(log2(N))` 和 `bw_step = (bytes_total/n_gpus)/...` 正是这两个特性的直接体现。

**AI 研究场景:** 这是任何真实分布式训练框架(PyTorch DDP、Horovod、Megatron-LM 的通信后端)选择 all-reduce 具体实现时的决策依据——`pick_algorithm` 的启发式规则(小消息用 Tree、GPU 数不多用 Ring、其余用 HD)和真实 NCCL 库的算法选择逻辑在定性方向上一致(知识点 6 的 `nccl_original_minimal.py` 会展开更贴近真实 NCCL 论文的选择模型)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from allreduce_algos import ring_allreduce, tree_allreduce, halving_doubling, pick_algorithm
from common import LINKS

nvl = LINKS["nvlink4"]
ib = LINKS["ib_ndr"]

# 大N+大消息场景: HD应该显著快于ring
n_gpus, size = 512, int(140e9)   # 70B模型bf16梯度
t_ring = ring_allreduce(n_gpus, size, ib)
t_hd = halving_doubling(n_gpus, size, ib)
assert t_hd < t_ring
speedup = t_ring / t_hd
print(f"512-GPU 140GB梯度: ring={t_ring/1e6:.2f}s  HD={t_hd/1e6:.2f}s  HD快{speedup:.1f}x")

# 小消息+大N场景: tree应该比ring快(延迟主导,tree步数少)
tiny_ring = ring_allreduce(64, 1024, ib)
tiny_tree = tree_allreduce(64, 1024, ib)
assert tiny_tree < tiny_ring

assert pick_algorithm(8, int(1e9), nvl) == "ring"          # 小规模GPU,NVLink,大消息 -> ring
assert pick_algorithm(64, int(1e9), ib) == "halving_doubling"  # 大规模GPU -> HD
assert pick_algorithm(64, 256, ib) == "tree"                 # 极小消息 -> tree
```

**实测(`.venv` 真跑):** 512 GPU、140GB 梯度(对应 70B 模型 BF16 参数量)、IB NDR 链路下,ring 耗时和 HD 耗时的比值验证 HD 显著更快(独立复算确认 HD 在这个大规模场景下步数`2*ceil(log2(512))=18`步 远少于 ring 的 `2*(512-1)=1022`步,即使 HD 每步的延迟项被放大了 18 次,总延迟开销仍然远小于 ring 1022 次延迟的累积)。64-GPU、1KB 极小消息场景下,tree(延迟主导,少步数)确认快于 ring(步数是 tree 的 `(N-1)/log2(N)≈10.5`倍,延迟主导场景下步数差异直接决定耗时差异)。`pick_algorithm` 的三条启发式分支全部按预期匹配。

**面试怎么问 + 追问链:**
- **Q:** "Tree 算法的步数是 `O(logN)`,比 Ring 的 `O(N)` 少得多,为什么大消息场景下 Ring/HD 反而更常被使用?"—— 期望:步数少不等于总耗时短——Tree 每一步传输的是**未分摊**的完整数据量(靠近根节点的边要承载来自多个子树的合并流量),大消息场景下这个"每步流量大"的代价远超"步数少"带来的收益;Ring/HD 的每步流量已经按 N 分摊,虽然步数是 `O(N)`(Ring)或 `O(logN)`(HD),但每步的绝对开销小得多,大消息场景下总耗时反而更低,这是"步数"和"每步开销"两个维度必须联合考虑,不能只看步数。
- **追问1:** "如果集群规模 N 从 8 增长到 8192(1024倍),HD 相对 Ring 的优势会怎么变化?"—— 期望:会持续扩大——Ring 的步数 `2(N-1)` 随 N 线性增长(8→8192,步数增长约1024倍),HD 的步数 `2*ceil(log2(N))`随 N 对数增长(8→8192,`log2` 从3到13,步数只增长约4.3倍);在大消息、大 N 的场景下,HD 相对 Ring 的耗时优势会随集群规模扩大而愈发明显,这正是超大规模集群(数千卡级别)几乎必然选择 HD 类算法而不是纯 Ring 的原因。

**常见坑:** `pick_algorithm` 的三条分支阈值(`bytes_total<1024`、`n_gpus<=8`)是教学演示用的简化启发式,不是真实 NCCL 库的实际决策逻辑——真实 NCCL 的算法选择(以及知识点 6 `nccl_original_minimal.py` 更贴近论文的模型)还要综合考虑拓扑结构、协议开销、channel 数量等更多因素,这里的简化版本只用来演示"存在这样一套决策思路",不能直接当作真实系统调优参数的参考值。

---

## 3. Fat-Tree 与 Dragonfly 拓扑(`fabric_topology.py`)—— Bisection 带宽公式,以及 2:1 Oversubscription 为什么精确减半

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class FatTree:
    n_nodes: int
    radix: int              # switch port count
    oversubscription: float = 1.0

    def bisection_gb_s(self, link_bw_gb_s: float) -> float:
        ideal = (self.n_nodes / 2) * link_bw_gb_s
        return ideal / self.oversubscription

    def n_switches_3tier(self) -> dict:
        leaf = self.n_nodes / (self.radix / 2)
        return {"leaf": int(leaf), "spine": int(leaf), "core": int(leaf / 2 + 1)}
```
(`fabric_topology.py:6-18`)

**一句话:** Bisection 带宽(把整个集群切成两半,两半之间能同时传输的总带宽)是衡量一个网络拓扑"够不够胖"的核心指标——理想情况下(oversubscription=1.0,即"full bisection")这个值是"一半节点数×单链路带宽",Fat-Tree 拓扑通过三层交换机(leaf/spine/core)组织起来,理论上能做到这个理想值,但真实部署为了省钱经常引入 oversubscription(用更少的上联链路服务更多下联节点),这时候 bisection 带宽会按 oversubscription 比例打折。

**画出来看:三层交换机长什么样(以下方例子 `FatTree(n_nodes=1024, radix=64)` 为例;`n_switches_3tier()` 用 `.venv` 实跑算出 `{'leaf': 32, 'spine': 32, 'core': 17}`——leaf=32 这一项和下方"实测"给出的"32 台 leaf 交换机"一致,spine/core 是同一次调用返回字典里的另外两项):**

```
core 层 —— 17 台交换机,只连 spine,是流量跨越不同"pod"(leaf+spine 构成的子树)时的最高层中转
┌────────┬────────┬────────┬───────────┬─────────┐
│ core0  │ core1  │ core2  │    ⋯⋯      │ core16  │
└───┬────┴───┬────┴───┬────┴───────────┴────┬────┘
    └────────┴────────┴─────────┬────────────┘
                                 │  (每台 spine 上联到全部/部分 core)
spine 层 —— 32 台交换机,上联全部 core、下联全部 leaf
┌────────┬────────┬────────┬───────────┬─────────┐
│ spine0 │ spine1 │ spine2 │    ⋯⋯      │ spine31 │
└───┬────┴───┬────┴───┬────┴───────────┴────┬────┘
    └────────┴────────┴─────────┬────────────┘
                                 │  (每台 leaf 上联到全部 spine)
leaf 层 —— 32 台交换机,radix=64 个端口里一半(32)朝下接服务器、一半(32)朝上接 spine
┌────────┬────────┬────────┬───────────┬─────────┐
│ leaf0  │ leaf1  │ leaf2  │    ⋯⋯      │ leaf31  │
└───┬────┴───┬────┴───┬────┴───────────┴────┬────┘
    │        │        │                     │
  32台服务器 32台服务器 32台服务器   ⋯⋯    32台服务器   ← 1024 节点 = 32 leaf × 32 服务器/leaf
```

从下往上看,"胖"的含义就是:leaf 层已经把 1024 台服务器聚合到 32 台交换机上,spine/core 层再逐级往上聚合——只要每一层的上联带宽总量不小于下联带宽总量(即 oversubscription=1.0),任意两台服务器之间,不管物理上隔着几层交换机,理论上都能以线速通信,这就是"full bisection"这个理想状态在拓扑图上的直观样子;知识点 3 讨论的 oversubscription>1.0,对应的正是"上联链路数比下联链路数更少"这个图上会体现出来的不对称结构。

**底层机制/为什么这样设计:** `bisection_gb_s` 的公式 `(n_nodes/2)*link_bw_gb_s / oversubscription` 背后的直觉是:把 N 个节点切成两半,每一半有 `N/2` 个节点,如果网络"足够胖"(full bisection),这 `N/2` 个节点可以同时以各自的全速链路带宽向另一半发送数据而互不阻塞,理想总带宽就是 `(N/2)*单链路带宽`;`oversubscription=2.0` 意味着"设计上允许 2:1 超订"——比如每个 leaf 交换机连接的下联(服务器)端口数是上联(去 spine 层)端口数的 2 倍,这意味着如果下面所有服务器同时想跑满带宽向外发送,上面的链路只能提供一半的带宽,必然有一半的流量被挤压,这就是为什么 oversubscription 精确对分带宽(不是任意比例)——超订比例本身就是"能承诺的带宽"和"理论峰值需求"之间的精确折算系数。`n_switches_3tier` 用 `n_nodes/(radix/2)` 算 leaf 层交换机数,这个公式假设每个 leaf 交换机一半端口朝下(接服务器)一半朝上(接 spine),`radix/2` 就是每个 leaf 能服务的服务器数量。

**AI 研究场景:** 集群网络设计里"要不要接受 oversubscription、接受多大的比例"是一个直接的成本-性能权衡决策——full bisection(oversubscription=1.0)网络能保证任意通信模式下都不会因为拓扑本身成为瓶颈,但需要的交换机和线缆数量(进而是成本)显著更高,真实工业界大规模集群(如训练 GPT 级别模型的集群)几乎都会在某个层级引入适度的 oversubscription 来控制成本,只要应用层的通信模式(如本文知识点 2 的 all-reduce)能被验证不会触发最坏情况的拥塞。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from fabric_topology import FatTree, Dragonfly

ft_full = FatTree(1024, 64, oversubscription=1.0)
bis_full = ft_full.bisection_gb_s(50.0)
assert bis_full == 512 * 50.0     # (1024/2)*50

ft_2to1 = FatTree(1024, 64, oversubscription=2.0)
bis_2to1 = ft_2to1.bisection_gb_s(50.0)
assert bis_2to1 == bis_full / 2    # 精确减半,不是近似

sw = ft_full.n_switches_3tier()
assert sw["leaf"] == 32              # 1024/(64/2) = 32

# 独立验证: 换不同的oversubscription比例(不只是2.0),确认精确按比例折算
for over in [1.5, 3.0, 4.0]:
    ft = FatTree(1024, 64, oversubscription=over)
    bis = ft.bisection_gb_s(50.0)
    assert abs(bis - bis_full / over) < 1e-9

df = Dragonfly(n_groups=8, nodes_per_group=64, intra_bw_gb_s=100, inter_bw_gb_s=25)
assert df.n_nodes() == 512
assert df.avg_hops() == 3.0          # 简化模型: 多group时固定3跳(2 intra + 1 inter)
print(f"1024节点fat-tree: full bisection={bis_full:.0f}GB/s, 2:1超订={bis_2to1:.0f}GB/s, leaf交换机={sw['leaf']}台")
```

**实测(`.venv` 真跑):** 1024 节点、64-port 交换机、50GB/s 链路下,full bisection(oversubscription=1.0)给出 **25600 GB/s**(=512×50),2:1 超订精确减半到 **12800 GB/s**——独立验证扩展到 1.5/3.0/4.0 三个额外的超订比例,全部精确验证 `bisection = full_bisection / oversubscription` 这条线性折算关系(不是仅在 2.0 这个特例下成立)。`n_switches_3tier` 给出 32 台 leaf 交换机,和手算 `1024/(64/2)=32` 精确一致。Dragonfly 拓扑的 `avg_hops()` 返回固定值 3.0(简化模型,不随具体拓扑规模变化,只要 `n_groups>1`)。

**面试怎么问 + 追问链:**
- **Q:** "Dragonfly 拓扑的平均跳数是固定的 3 跳,和节点规模无关,这和 Fat-Tree 的三层结构相比有什么本质区别?"—— 期望:Fat-Tree 的跳数也大致固定(三层结构下,任意两个叶子节点之间的跳数是常数,不随集群规模线性增长,这也是"胖树"设计的核心卖点之一),两者都属于"低直径"网络设计;但两者实现"低跳数"的物理拓扑完全不同——Fat-Tree 靠树状分层聚合,Dragonfly 靠"组内全连接+组间稀疏互联"的分组结构,Dragonfly 通常能用更少的长距离光纤线缆(组间连接)达到接近的性能,这是数据中心规模扩大时更受青睐的拓扑之一。
- **追问1:** "如果一个 all-reduce 算法(如知识点 2 的 Ring)的通信模式恰好总是相邻节点通信,oversubscription 的网络设计会不会成为瓶颈?"—— 期望:通常不会——Ring all-reduce 的通信模式是"每个节点只和逻辑环上的相邻节点通信",如果物理拓扑的节点编号/rank 分配和物理位置匹配得当(拓扑感知的 rank 分配),大部分流量可以被限制在同一个 leaf 交换机下的少数节点之间,不需要真的穿越整个 bisection——这是为什么"应用层通信模式"和"网络物理拓扑"需要协同设计,一个高 oversubscription 的网络对某些通信模式(拓扑感知)完全没问题,对另一些通信模式(如 all-to-all,知识点 4 会讨论)可能就是灾难。

**常见坑:** `n_switches_3tier` 返回的 `core` 层交换机数公式是 `leaf/2 + 1`,这是一个针对特定拓扑假设(每台 spine 连接到全部 leaf,每台 core 连接到全部 spine 的一个子集)简化推导出的经验公式,不是所有 Fat-Tree 变体通用的标准公式——真实网络架构设计中,leaf/spine/core 三层的具体交换机数量需要根据实际的端口密度、冗余需求、以及是否要支持后续扩容等工程约束重新推导,不能直接套用这个简化公式。

---

## 4. NCCL 五个核心 Collective(`nccl_collectives.py`)—— AllReduce = ReduceScatter + AllGather 这条恒等式怎么来的

**是什么:**
```python
from __future__ import annotations

def all_gather(n_gpus: int, bytes_per_gpu: int, link: Link) -> float:
    """T = (N-1)/N * total_size / BW. Bandwidth-optimal ring."""
    total = bytes_per_gpu * n_gpus
    bw_per_step = (total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return (n_gpus - 1) * (link.latency_us + bw_per_step)

def all_reduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """Ring all_reduce = reduce_scatter + all_gather."""
    return ring_allreduce(n_gpus, bytes_total, link)

def all_to_all(n_gpus: int, bytes_per_pair: int, link: Link) -> float:
    """Each GPU sends bytes_per_pair to N-1 others. Saturates per-GPU egress BW."""
    total_egress = (n_gpus - 1) * bytes_per_pair
    return link.latency_us + (total_egress / link.bw_gb_s) / 1e9 * 1e6
```
(`nccl_collectives.py:7-36`,节选)

**一句话:** NCCL(全称 NVIDIA Collective Communications Library,NVIDIA 官方提供的多 GPU/多节点集合通信库,本文其余知识点反复提到的"真实 NCCL"都是指这个库)对外暴露的核心能力,就是 AllReduce、ReduceScatter、AllGather、Broadcast、AllToAll 这分布式训练里最常用的 5 个通信原语,其中最重要的一条关系是 **AllReduce 的通信成本约等于 ReduceScatter + AllGather 两者之和**——这不是巧合,是 Ring AllReduce 算法本身的实现方式决定的(知识点 2 提到过,Ring AllReduce 内部就是先做 reduce-scatter 再做 all-gather 两阶段)。

**底层机制/为什么这样设计:** ReduceScatter 让每个 GPU 只保留"自己那一份"规约后的结果(不是完整的全局和,是全局和的 1/N 切片),AllGather 则相反——每个 GPU 已经各自持有不同的数据切片,把它们汇总广播给所有人拿到完整拼接结果;把这两个操作串起来(先 ReduceScatter 把梯度规约、切片分布到各卡,再 AllGather 把切片拼回完整梯度),数学效果和一次 AllReduce 完全等价,这也是为什么真实框架(如 DeepSpeed ZeRO,02号文件已经讨论过)会把"AllReduce"拆成"ReduceScatter+AllGather"两步分别处理——拆开后可以在 ReduceScatter 完成后就立即开始用那一份切片做优化器更新(不需要等 AllGather 完成),实现计算和通信的进一步重叠。`all_to_all` 的成本模型不同于其他四个——它是"每个 GPU 都要给其余 N-1 个 GPU 各发一份不同的数据"(不是所有卡协作产出同一份结果),这种模式下每张卡的**出口带宽**($(N-1)\times$每对数据量)成为瓶颈,和其他集合通信"总带宽在 N 张卡间分摊"的性质完全不同。

**AI 研究场景:** MoE(混合专家)模型训练里,token 路由到不同专家所在的 GPU 需要用 AllToAll(每个 token 去哪个专家、来自哪个 GPU 都不固定,是名副其实的"每对之间都可能有数据交换"),03 号文件知识点 10 讨论过 DeepSeek-V3 这类 MoE 架构——AllToAll 的通信开销模式和标准数据并行的 AllReduce 完全不同,是 MoE 训练相比稠密模型训练需要专门优化通信路径的核心原因之一。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from nccl_collectives import all_gather, reduce_scatter, all_reduce, broadcast, all_to_all
from common import LINKS
import math

nvl = LINKS["nvlink4"]
rs = reduce_scatter(8, int(1e8), nvl)
ag = all_gather(8, int(1e8) // 8, nvl)
ar = all_reduce(8, int(1e8), nvl)
assert abs(ar - (rs + ag)) / ar < 0.05     # AllReduce ≈ ReduceScatter + AllGather,5%内吻合

bc8 = broadcast(8, int(1e7), nvl)
bc64 = broadcast(64, int(1e7), nvl)
assert bc8 < bc64
assert bc64 / bc8 < math.log2(64) / math.log2(8) + 0.5   # 增长速率接近log(N),不是线性

# 独立验证: AllToAll的成本对"每对数据量"的敏感度,和AllReduce对"总数据量"的敏感度做对照
a2a_small = all_to_all(64, 1_000_000, nvl)      # 每对1MB
a2a_large = all_to_all(64, 1_000_000_000, nvl)  # 每对1GB,增大1000倍
ratio_a2a = a2a_large / a2a_small
assert 990 < ratio_a2a < 1000   # 两端都已深处带宽主导区,比值应接近(但略小于)字节比值1000
print(f"AllReduce({rs+ag:.0f}us理论) vs 实测({ar:.0f}us): 吻合")
print(f"AllToAll: 每对1MB={a2a_small:.1f}us, 每对1GB={a2a_large:.1f}us, 比值={ratio_a2a:.1f}")
```

**实测(`.venv` 真跑):** 8-GPU、1e8 字节负载下,`reduce_scatter+all_gather` 之和与直接 `all_reduce` 的相对误差在 5% 以内(两者理论上应该相等,5% 容差是不同函数各自延迟项累加方式的微小差异,不是系统性偏差)。`broadcast` 在 8→64 GPU 时增长速率确认接近对数(不是线性)增长。独立验证的 AllToAll 场景:每对数据量从 1MB 增大到 1GB(1000倍),耗时从 140.5μs 增长到 140000.5μs,比值 **996.4倍**——非常接近但不精确等于字节比值 1000倍,原因是即使在 1MB/对这个"已经算大"的起点,0.5μs 的延迟项仍占总耗时的 0.36%(140.5μs 里的 0.5μs),这一点点残留拉低了比值;这条对照最初我曾用 1KB→1MB(同样 1000 倍字节增长)测试,实测比值只有约 220 倍——因为 1KB/对时传输耗时仅 0.14μs,延迟项 0.5μs 反而占了总耗时的 78%,完全没有进入带宽主导区。这两组数字对照凸显了 AllToAll 的核心特征:它的开销主要由"每对之间要传多少"决定,但只有当每对数据量大到让延迟项可忽略时,耗时增长才会真正逼近线性——这和 AllReduce 由"总规约的数据量"决定、且步数本身自带对数/线性权衡的成本结构完全不同。

**面试怎么问 + 追问链:**
- **Q:** "ZeRO-3(02号文件讨论过)为什么倾向于把 AllReduce 拆分成 ReduceScatter 和 AllGather 分开处理,而不是直接调用一次性的 AllReduce?"—— 期望:ReduceScatter 完成后,每张卡已经拿到了"完整、正确的"那一份梯度切片(不是近似,是精确的最终结果),可以立即用这份切片更新对应的那部分优化器状态,不需要等待 AllGather 把完整梯度重新拼回来——这让"梯度规约"和"优化器更新"这两步能够重叠执行,而不是必须先等一次完整的 AllReduce 全部完成才能开始更新参数,是 ZeRO 系列显存优化之外额外获得的计算-通信重叠收益。
- **追问1:** "MoE 模型的 AllToAll 通信量和 token 数、专家数分别是什么关系?这对集群网络设计有什么启示?"—— 期望:AllToAll 的总通信量正比于"需要跨卡路由的 token 数×每 token 的 hidden 维度大小",且理论上"每对 GPU 之间"都可能需要传输数据(不像 AllReduce 那样每张卡的角色对称),这意味着 MoE 训练对**全连接**式的网络拓扑(每对节点之间都要有足够带宽,不能只优化"相邻节点通信快")要求更高,如果网络拓扑存在明显的 oversubscription(知识点 3),AllToAll 模式很容易触发最坏情况的拥塞,这是 MoE 训练集群网络设计时需要特别评估的场景。

**常见坑:** `reduce_scatter` 函数的文档字符串写"Same cost as all_gather"(和 all_gather 成本相同),这是因为两者的公式结构确实完全对称(`(N-1)*(latency+per_step_bw)`),但这个"成本相同"只在**同样的数据量、同样的链路**假设下成立——真实系统里 reduce_scatter 通常还涉及规约计算本身(加法运算),而 all_gather 只是纯数据搬运,如果规约计算本身的开销(哪怕很小)被考虑进来,两者的实际总耗时会有细微差异,这份简化模型出于教学目的忽略了计算开销,只建模通信部分。

---

## 5. SHARP 交换机内聚合(`sharp_inline.py`)—— 恒 2 步、与 N 无关,加速比精确收敛到 N-1

**是什么:**
```python
from __future__ import annotations

def sharp_allreduce(n_gpus: int, bytes_total: int, link: Link) -> float:
    """In-network reduction at the switch: 2-step constant cost.

    Each GPU sends size/N up to switch, switch aggregates, multicasts result.
    Total traffic per GPU = 2 * size/N (one up, one down).
    Latency O(1) instead of O(log N) or O(N).
    """
    bw_step = (bytes_total / n_gpus) / link.bw_gb_s / 1e9 * 1e6
    return 2 * (link.latency_us + bw_step)
```
(`sharp_inline.py:6-14`)

**一句话:** SHARP(Scalable Hierarchical Aggregation and Reduction Protocol,Mellanox/NVIDIA 的交换机内聚合技术)把"规约"这个计算动作从 GPU 端搬到了网络交换机内部——每个 GPU 只需要把自己那一份数据发给交换机,交换机做完加法后再把最终结果组播回所有 GPU,全程恒定 **2 步**(1 次上传+1 次下发),完全不随 GPU 数量增长,这是本模块所有算法里唯一一个时间复杂度和 N 无关的方案。

**底层机制/为什么这样设计:** Ring/HD 等纯软件算法之所以需要多步,是因为"规约"这个计算必须发生在某个 GPU 上,数据必须先物理搬运到那张卡才能做加法——SHARP 把加法这个计算能力下沉到网络交换机的 ASIC 硬件里,数据在"路过"交换机的过程中就顺便被累加了,不需要先集中到某一张卡再分发,这从根本上改变了通信-计算的先后关系(不再是"先搬运再计算",而是"搬运的同时就算完了")。步数恒为 2 且和 N 无关,是因为无论多少张 GPU 参与,每张卡都只需要做"发一次、收一次"这两个动作,交换机内部的聚合逻辑不需要 GPU 端感知或参与更多轮次。

**AI 研究场景:** SHARP 是超大规模训练集群(数百到数千卡)里性价比极高的硬件加速手段——它不需要修改任何训练代码或并行策略(对上层框架完全透明,NCCL 库检测到硬件支持 SHARP 时会自动启用),只需要交换机本身具备这个能力(通常是高端 InfiniBand 交换机的特性),这也是为什么知识点 6 的 `capstone_cluster_sim.py` 会看到"IB/RoCE 链路才能用 SHARP"这个限制条件(以太网普通交换机不具备这个 ASIC 加法能力)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from sharp_inline import sharp_allreduce, speedup_vs_ring
from common import LINKS

ib = LINKS["ib_ndr"]
sp_64 = speedup_vs_ring(64, int(1e8), ib)
sp_512 = speedup_vs_ring(512, int(1e8), ib)
assert sp_512 > sp_64      # N越大,SHARP的相对优势越明显

# 独立验证(比源码自测更精细): 加速比是否精确等于n_gpus-1,跨多个N和消息大小
for n in [16, 128, 256]:
    for size in [int(1e6), int(1e10)]:      # 1MB 和 10GB,量级差1万倍
        sp = speedup_vs_ring(n, size, ib)
        assert abs(sp - (n - 1)) < 0.01       # 精确等于n_gpus-1,不是近似
print(f"64-GPU speedup={sp_64:.1f}x(=63)  512-GPU speedup={sp_512:.1f}x(=511)")
print("独立验证: 16/128/256-GPU × 1MB/10GB(消息大小相差1万倍)下,加速比全部精确等于n_gpus-1")
```

**实测(`.venv` 真跑):** 64-GPU 场景加速比精确 **63.0倍**,512-GPU 场景精确 **511.0倍**,两者都精确等于 `n_gpus-1`。独立验证把这条规律推广到 16/128/256 三个额外的 GPU 数、以及 1MB 到 10GB(相差 1 万倍)两个消息量级,**全部 6 种组合都精确验证加速比等于 `n_gpus-1`**(误差小于 0.01,浮点精度极限)——这条规律的数学解释很直接:大消息场景下延迟项可忽略,Ring 耗时 `≈2(N-1)×bw_per_step`,SHARP 耗时 `≈2×bw_per_step`,比值精确等于 `N-1`,且这个比值**只取决于步数比、和消息大小完全无关**(因为 `bw_per_step` 这一项在比值计算中被约掉了)——这解释了为什么消息大小从 1MB 变到 10GB,加速比数字丝毫不变。

**面试怎么问 + 追问链:**
- **Q:** "SHARP 的加速比和消息大小完全无关,这在什么条件下会失效?"—— 期望:上面的推导依赖"大消息场景下延迟项可忽略"这个前提——如果消息小到延迟项(而不是带宽项)主导耗时,Ring 的延迟开销是 `2(N-1)×latency`,SHARP 是 `2×latency`,比值同样是 `N-1`(延迟主导下比值还是一样,因为两者的步数比是不变的),所以实际上这条"加速比=N-1"的规律对延迟主导和带宽主导两种场景**都成立**,只要 SHARP 的 2 步和 Ring 的 `2(N-1)` 步这个步数比例关系不变——真正会让这条规律"失效"的是引入了 SHARP 自身没有建模的额外开销(比如真实硬件的交换机聚合延迟不是严格恒定的,可能随负载变化)。
- **追问1:** "如果集群里只有部分交换机支持 SHARP(比如只有核心层支持,叶子层不支持),这个'恒2步'的模型还成立吗?"—— 期望:不完全成立——`sharp_allreduce` 假设整个规约过程能在单一层级的交换机内完成聚合,如果 SHARP 能力只存在于某一层(比如只有 core 层交换机支持),流量仍然需要先从 GPU 经过不支持 SHARP 的 leaf 层交换机才能到达 core 层,这部分路径的通信可能仍然需要额外的步骤或退化成软件规约,真实部署里"端到端全链路 SHARP"和"部分链路 SHARP"是完全不同的性能特征,这份简化模型假设的是理想的全链路场景。

**常见坑:** `sharp_allreduce` 的 docstring 明确写"O(1) instead of O(logN) or O(N)"，容易让人误以为 SHARP 在任何规模下都是"免费的",但真实交换机的聚合能力(ASIC 处理能力、缓冲区大小)本身是有限的物理资源,超大规模(数千卡)或超大消息场景下,交换机内部聚合本身可能成为新的瓶颈——这份模型的"恒定 2 步"是对交换机硬件能力足够、不构成瓶颈这一前提下的理想化描述,不代表 SHARP 在任意规模下都没有自己的扩展性上限。

---

## 6. 番外:NCCL Protocol/Channel/Algorithm 选择模型(`nccl_original_minimal.py`)—— 复合有效带宽,以及一个容易被漏看的乘法关系

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ProtocolProfile:
    name: str
    per_hop_latency_us: float
    bandwidth_fraction: float
    data_bytes_per_unit: int
    flag_bytes_per_unit: int

    @property
    def payload_fraction(self) -> float:
        total = self.data_bytes_per_unit + self.flag_bytes_per_unit
        return self.data_bytes_per_unit / total if total else 1.0

def protocol_transfer_time_us(bytes_total, link, protocol, hops=1) -> float:
    effective_bw = link.bw_gb_s * protocol.bandwidth_fraction * protocol.payload_fraction
    transfer_us = (bytes_total / 1e9) / effective_bw * 1e6 if effective_bw else 0.0
    return hops * protocol.per_hop_latency_us + transfer_us
```
(`nccl_original_minimal.py:15-78`,节选)

**一句话:** 这是本模块唯一忠实复现《Demystifying NCCL》论文本身分析框架的脚本(没有对应独立 lecture,只在论文导读里出现)——NCCL 真实有 Simple/LL/LL128 三种传输协议,各自在"延迟"和"有效带宽"之间做了不同取舍,`effective_bw = link_BW × bandwidth_fraction × payload_fraction` 这个**两个系数相乘**的复合公式,是很多人只看第一个系数就下结论、从而得出错误直觉的地方。

**底层机制/为什么这样设计:** `bandwidth_fraction` 衡量"这个协议能利用链路原始带宽的百分之多少"(比如 LL 协议只有 35%,远低于 Simple 的 98%),但这不是全部故事——`payload_fraction`(由 `data_bytes_per_unit/(data_bytes_per_unit+flag_bytes_per_unit)` 算出)衡量"每个传输单元里,有多少字节是真正的数据、多少字节是协议自身的标记开销"——LL 协议为了做到极低延迟(`per_hop_latency_us=1.0`,是三者中最低的),每传 4 字节数据要搭配 4 字节的 flag 标记(用于快速轮询检测数据是否到达,不依赖内存屏障),这意味着 `payload_fraction` 只有 50%(4/(4+4))。**两个系数相乘**才是真正的有效带宽:LL 的复合有效带宽是 `35%×50%=17.5%`,如果只看 `bandwidth_fraction=35%` 这一个数字,会显著高估 LL 协议实际能达到的吞吐。

**AI 研究场景:** 这是真实分布式训练 profiling 时"为什么小批量梯度同步的实测带宽利用率远低于理论峰值"的一个常见解释来源——训练框架为了追求低延迟(尤其是小消息、高频次同步场景)会自动切换到 LL 类协议,这类协议为了延迟牺牲的带宽效率(不止一个系数在打折,是两个系数相乘)如果不理解这个机制,容易误判成"网络配置有问题"而不是"协议选择本身就是低带宽高延迟权衡的代价"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from nccl_original_minimal import SIMPLE, LL, LL128, choose_protocol, choose_channels

# 三个协议的复合有效带宽(bandwidth_fraction x payload_fraction)
for p in [SIMPLE, LL, LL128]:
    effective = p.bandwidth_fraction * p.payload_fraction
    print(f"{p.name:>6}: bandwidth_fraction={p.bandwidth_fraction:.2f}  "
          f"payload_fraction={p.payload_fraction:.4f}  复合有效带宽={effective:.4f}")

assert abs(LL.payload_fraction - 0.5) < 1e-9          # 4/(4+4)
assert abs(LL128.payload_fraction - 120/128) < 1e-9     # 120/(120+8)
ll_effective = LL.bandwidth_fraction * LL.payload_fraction
assert ll_effective < LL.bandwidth_fraction              # 复合值一定比单看bandwidth_fraction更低(除非payload_fraction=1)

# choose_protocol: 小消息+节点内 -> LL128; 小消息+跨节点 -> LL; 大消息 -> Simple
assert choose_protocol(16*1024, intra_node=True).name == "LL128"
assert choose_protocol(16*1024, intra_node=False).name == "LL"
assert choose_protocol(512*1024*1024, intra_node=False).name == "Simple"

# choose_channels: 消息太小时收缩channel数,避免每个channel都吃不饱FIFO
assert choose_channels(128*1024, max_channels=16) == 1     # 128KB塞进1个channel都够呛(fifo=512KB)
assert choose_channels(64*1024*1024, max_channels=16) == 16  # 64MB充分利用全部16个channel
```

**实测(`.venv` 真跑):** 三协议复合有效带宽实测:Simple `98%×100%=98.0%`(几乎不打折,大消息首选),LL `35%×50%=17.5%`(延迟最低但有效带宽只有表面 `bandwidth_fraction` 数字的一半),LL128 `95%×93.75%=89.06%`(`120/128=0.9375`,在节点内、中等消息场景的"甜点"选择)。`choose_protocol` 的三条分支测试全部匹配(节点内小消息选 LL128、跨节点小消息选 LL、大消息统一选 Simple)。`choose_channels` 验证了"消息太小时主动减少并行 channel 数"这个反直觉的设计——不是 channel 越多越并行越快,如果每个 channel 分到的数据量都填不满 NIC 的 FIFO 缓冲区(512KB),硬件效率反而下降,所以小消息场景下宁可用更少的 channel 让每个 channel 吃饱。

**面试怎么问 + 追问链:**
- **Q:** "LL128 协议的 `payload_fraction` 是 93.75%,比 LL 的 50% 高得多,为什么它的名字里还是有'LL'(Low Latency)?"—— 期望:LL128 是 LL 协议的改进版,同样追求低延迟(`per_hop_latency_us=2.0`,虽然比纯 LL 的 1.0 略高,但远低于 Simple 的 6.0),但通过要求 128 字节对齐的原子写入(利用某些硬件平台的原子性保证),把 flag 开销从"每 4 字节配 4 字节"降低到"每 120 字节配 8 字节",大幅提升了 `payload_fraction`;它是 NCCL 在"低延迟"和"高带宽利用率"之间找到的一个更优折衷点,不是完全牺牲带宽换延迟,是三选一里"数据量适中、节点内"这个场景的专门优化路径。
- **追问1:** "`choose_channels` 的实现先算出理论需要的 channel 数,再检查是否会吃不饱 FIFO 后收缩——为什么不直接一步到位算出最终值?"—— 期望:因为"理论需要多少 channel"(基于总数据量/单 channel 目标负载)和"每个 channel 分到的数据是否够填满 FIFO"是两个独立的约束,前者是"并行度理论上限",后者是"单 channel 效率下限",`choose_channels` 的两步实现(先按理论算,再检查约束收缩)本质上是在这两个相互牵制的约束条件下做迭代求解,直接合并成一步公式在数学上是可行的,但分两步写更清楚地展示了决策逻辑本身的两个独立考量,可读性优先于代码行数。

**常见坑:** `ProtocolProfile.payload_fraction` 是一个 `@property`(每次访问都重新计算),而不是预先算好存储的字段——这意味着如果未来需要频繁访问这个值(比如在一个大循环里反复查询同一个协议的 payload_fraction),会有轻微的重复计算开销;在这份代码规模下这个开销完全可以忽略,但读代码时容易假设"这是一个存储好的字段直接读取",实际是每次都重新做一次除法运算,这类"看起来像字段实际是计算属性"的 Python 惯用法在阅读大型代码库时需要留意。

---

## 7. Capstone:4 Fabric × 4 集群规模综合选型(`capstone_cluster_sim.py`)—— 真的调用前 6 个知识点的函数,不是摆样子

**是什么:**
```python
from __future__ import annotations
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from common import LINKS
from allreduce_algos import ring_allreduce, halving_doubling
from sharp_inline import sharp_allreduce

SCENARIOS = [
    (8,   LINKS["nvlink4"]),
    (8,   LINKS["pcie5_x16"]),
    (64,  LINKS["ib_ndr"]),
    (512, LINKS["ib_xdr"]),
]
GRADIENT_BYTES = int(140e9)   # 70B模型BF16梯度

def run() -> list:
    rows = []
    for n_gpus, link in SCENARIOS:
        algos = {"ring": ring_allreduce(n_gpus, GRADIENT_BYTES, link),
                 "halving_doubling": halving_doubling(n_gpus, GRADIENT_BYTES, link)}
        if "IB" in link.name or "RoCE" in link.name:
            algos["sharp"] = sharp_allreduce(n_gpus, GRADIENT_BYTES, link)
        best = min(algos, key=algos.get)
        rows.append({
            "n_gpus": n_gpus, "link": link.name, "best_algo": best,
            "best_time_s": round(algos[best] / 1e6, 2),
            "ring_time_s": round(algos["ring"] / 1e6, 2),
        })
    return rows
```
(`capstone_cluster_sim.py:8-38`,节选)

**一句话:** 这是前 6 个知识点的真实串联(不是重新实现,直接 import `ring_allreduce`/`halving_doubling`/`sharp_allreduce` 这些已经验证过的函数),对 4 种典型集群配置(小规模 NVLink、小规模 PCIe-only、中规模 IB、大规模 IB)分别算出"70B 模型梯度 all-reduce 用哪种算法最快",用 `min()` 真实比较而不是硬编码答案。

**底层机制/为什么这样设计:** `if "IB" in link.name or "RoCE" in link.name` 这一行体现了知识点 5 提到的约束——SHARP 只在支持交换机内聚合的网络硬件(IB/RoCE)上可用,NVLink(卡间直连,没有"交换机"这一层可以做聚合)和 PCIe 场景根本不会尝试 SHARP 这个选项,`algos` 字典只在符合条件时才添加 `"sharp"` 这个键,`min(algos, key=algos.get)` 在剩下的候选算法里选真正耗时最小的那个——这个"先筛选可用选项、再在可用选项里比较"的两阶段逻辑,是真实系统做资源/算法选型时的标准模式。

**AI 研究场景:** 这类"给定硬件配置,自动推荐最优通信算法"的工具,是真实大规模训练项目在立项阶段做集群选型时会用到的决策辅助——本知识点虽然是简化的教学版本,但决策逻辑框架(枚举可行算法→逐一估算耗时→选最优)和真实工业界内部工具的设计思路是一致的。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/cluster-networking/src")
from capstone_cluster_sim import run, GRADIENT_BYTES

rows = run()
assert len(rows) == 4
assert GRADIENT_BYTES == 140_000_000_000    # 70B模型BF16梯度: 70e9参数 * 2字节 = 140GB

nvl_row = next(r for r in rows if "NVLink" in r["link"])
pcie_row = next(r for r in rows if "PCIe" in r["link"])
big_row = next(r for r in rows if r["n_gpus"] == 512)

assert "sharp" not in nvl_row.get("best_algo", "")   # NVLink没有交换机内聚合能力,不该选到sharp
assert big_row["best_algo"] == "sharp"                  # 512-GPU IB场景应该选SHARP(知识点5的加速比结论在这里复用)
assert pcie_row["ring_time_s"] > 5 * nvl_row["ring_time_s"]   # PCIe远慢于NVLink(知识点1的带宽差距在这里体现)

for r in rows:
    print(f"{r['n_gpus']:>4} GPU | {r['link']:<18} | best={r['best_algo']:<16} | {r['best_time_s']}s (ring={r['ring_time_s']}s)")
```

**实测(`.venv` 真跑):** 4 个场景全部真实算出(不是硬编码):8-GPU NVLink 选中 `halving_doubling`(0.23s,比 ring 的 0.54s 快)、8-GPU PCIe 同样选 `halving_doubling`(1.64s,比 ring 3.83s 快,且这两个 PCIe 数字比同规模 NVLink 慢 7 倍以上——直接体现知识点1揭示的链路带宽差距)、64-GPU IB NDR 选中 `sharp`(0.09s,远快于 ring 的 5.51s)、512-GPU IB XDR 同样选 `sharp`(0.01s,比 ring 2.8s 快 280 倍)。这条从"小规模 NVLink"到"大规模 IB"的完整对照表,把前 6 个知识点的公式和结论(链路带宽差距、HD 相对 ring 的优势随规模扩大、SHARP 只在 IB/RoCE 可用且加速比随 N 增长)全部串联进了一张真实计算出的表格里,不是分别验证互不relate的孤立结论。

**面试怎么问 + 追问链:**
- **Q:** "为什么 8-GPU 规模下,NVLink 和 PCIe 场景选中的都是 halving_doubling 而不是 ring,这不是和知识点 2 提到的'GPU 数不多时 ring 是常见选择'矛盾吗?"—— 期望:不矛盾——知识点 2 的 `pick_algorithm` 只是一个简化的教学启发式(`n_gpus<=8`时选 ring),不代表"8 GPU 时 ring 一定最快";本知识点是**真实计算**两种算法在这个具体消息大小(140GB,巨大)下的耗时后再比较,140GB 属于超大消息场景,HD 的"每步流量分摊"优势即使在 N=8 时也可能超过它"步数比 ring 略多几步"(`HD:2*ceil(log2(8))=6步` vs `ring:2*(8-1)=14步`,HD 步数其实更少)带来的开销,这里 HD 步数本来就比 ring 少,不存在矛盾,反而印证了"结论要靠真实计算验证,不能死记硬背简化启发式"这条本知识点想强调的原则。
- **追问1:** "如果要把这个 capstone 扩展成覆盖 8 种链路×8 种规模的完整矩阵,现有代码需要改动多少?"—— 期望:几乎不需要改动核心逻辑——只需要扩充 `SCENARIOS` 列表(加更多 `(n_gpus, link)` 元组),`run()` 函数的循环体和"筛选可用算法→选最优"的逻辑完全不变,这体现了这份设计的可扩展性:所有"重活"(具体算法的耗时计算)都封装在前 6 个知识点已经验证过的独立函数里,capstone 层只负责"组合调用+比较"这一层薄薄的编排逻辑。

**常见坑:** `run()` 函数在 `n_gpus=8` 且链路是 NVLink 时,由于 `"IB" in link.name or "RoCE" in link.name` 判断为 False,`algos` 字典里不会有 `"sharp"` 这个键——如果调用方代码不做防御性判断,直接假设每个场景的结果字典里都有 `sharp` 这个键去取值,会在 NVLink/PCIe 场景下触发 `KeyError`;本知识点"可运行例子"里用 `nvl_row.get("best_algo", "")` 这种带默认值的写法而不是直接索引,就是为了规避这个真实存在的边界情况,读这段代码或基于它扩展新逻辑时需要注意这个键的存在性是条件性的。

---

*上一篇:[05-cuda-essentials.md](05-cuda-essentials.md) | 下一篇:[07-storage-dataops.md](07-storage-dataops.md) —— 梯度在 GPU 间怎么搬讲完了,下一步看训练数据和 checkpoint 怎么在存储和 GPU 之间搬。*
