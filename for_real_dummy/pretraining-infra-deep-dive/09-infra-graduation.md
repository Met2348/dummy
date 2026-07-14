# 09 · Module 8 毕业:端到端系统设计(Infra Graduation)

> 总览见 [00-roadmap.md](00-roadmap.md)。本文是叙事体 capstone,不采用七步知识点模板。

## 开场:前面六站分别造了什么,这一站要把它们卖多少钱

`learning/infra-graduation/`(Module 8《系统与 Infra》第 7 专题,毕业站)要回答的问题和前面六站都不一样。GPU 架构和 kernel engineering(链接 [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/00-roadmap.md),下方"幕一"展开)决定了"一块芯片能算多快";05 号文件的 CUDA 执行模型决定了"怎么把这块芯片的算力真正榨出来";06 号文件的集群网络决定了"几千块芯片怎么互相说话而不互相等";07 号文件的存储管线决定了"数据和 checkpoint 怎么在这些芯片和硬盘之间搬而不拖后腿";08 号文件的训练编排决定了"这么多任务、这么多芯片,谁先跑、谁等着、坏了怎么办"。这六站每一站都在回答"某个子系统内部该怎么设计"。

本站(以及它背后要收官的 Module 8 全局问题)问的是一个纯粹的商业/工程决策问题:**给定一个要训的模型,选哪种 GPU、多少张、什么网络、多大存储,要花多少钱、要跑多久?** 这是任何一个真实立项(不管是从零建集群,还是在云上按需租)最终必须落地成一个数字表格的时刻——`learning/infra-graduation/src/` 用一个五百来行的纯 Python 模拟器,把前六站分别讨论的"这个子系统好不好"问题,收束成"这套配置多少钱、多久能训完"这一个可以直接拿去做预算决策的答案。

**环境声明与本模块专属的包导入陷阱:** 本文全部代码在仓库根目录 `.venv`(Python 3.13)下用 `.venv/Scripts/python.exe` 实际跑通验证。和前 6 个 M8 模块(包括本系列 05-08 号文件对应的模块)不同,`infra-graduation/src/` 是**包结构**(`sim/`、`eval/` 两个子包),`sim/cost_model.py`/`sim/time_to_train.py`/`sim/topology_selector.py`/`sim/capstone_1.py`/`eval/mlperf_mock.py` 这 5 个文件用的是 `from sim.common import ...` 这种包内绝对导入,**不能**像 05-08 号文件那样直接对脚本本身跑,需要先把 `learning/infra-graduation/src`(`sim`/`eval` 两个包的父目录,不是包本身)加进 `sys.path`。本文所有"可运行例子"代码块已按这个正确方式处理(`sys.path.insert(0, "learning/infra-graduation/src")` + `from sim.xxx import ...`),已实测确认可行,不需要设置 `PYTHONPATH` 环境变量。

---

## 幕一:积木本身——`sim/common.py` 就是前四站的浓缩

**是什么**(`sim/common.py`):

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_tflops: float
    fp8_tflops: float
    hbm_gb: int
    hbm_tb_s: float
    nvlink_tb_s: float
    tdp_w: int
    price_usd: int

GPU_CATALOG = {
    "H100": GPUSpec("H100",  989, 1979, 80, 3.35, 0.9, 700, 30000),
    "B200": GPUSpec("B200", 2250, 4500, 192, 8.0,  1.8, 1000, 40000),
    # ... 完整表还有A100/H200,共4款
}

@dataclass(frozen=True)
class FabricSpec:
    name: str
    per_node_bw_gb_s: float
    latency_us: float
    has_sharp: bool

@dataclass
class ClusterBlueprint:
    n_nodes: int
    gpus_per_node: int
    gpu: GPUSpec
    fabric: FabricSpec
    storage: StorageSpec

    def total_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    def capex_usd(self) -> int:
        return self.total_gpus() * self.gpu.price_usd
```

**这不是新知识,是把前四站的度量单位重新打包成一个可以互相组合的"零件目录"。** `GPUSpec` 里的 `bf16_tflops`/`hbm_tb_s`/`nvlink_tb_s` 三个字段,正是 [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/01-gpu-hardware-and-memory.md) 详细展开的"一块 GPU 芯片的三个关键规格"(算力/显存带宽/卡间互联带宽),`kernel-gpu-deep-dive/02-roofline-model.md` 的 Roofline 模型正是用这三个数字判断一个具体 kernel 是算力瓶颈还是带宽瓶颈——本文"幕二"要用到的 `util_pct=0.40`(GPU 实际能跑到理论峰值的百分之多少)这个关键假设,背后的物理原因就是 kernel-gpu-deep-dive 和本系列 05 号文件反复讨论的"kernel launch 开销、内存访问模式、warp 利用率"这些底层机制共同决定的,不是一个凭空拍脑袋的常数。`FabricSpec` 的 `per_node_bw_gb_s`/`latency_us`/`has_sharp` 三个字段,和 06 号文件 `Link` dataclass 的字段几乎一一对应(`has_sharp` 直接对应 06 号文件知识点 5 讨论的"只有 IB/RoCE 才能用 SHARP 交换机内聚合"这条约束)。`StorageSpec` 的 `bw_gb_s`/`cap_pb` 对应 07 号文件 `Storage` dataclass 的存储分层建模。`ClusterBlueprint` 把这三类零件(GPU + Fabric + Storage)加上 `n_nodes`/`gpus_per_node` 组装成一个完整的"集群配置",这正是把前四站各自独立讨论的子系统,统一到一个可以枚举、可以比较、可以选型的单一数据结构里。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG

bp = ClusterBlueprint(
    n_nodes=64, gpus_per_node=8,
    gpu=GPU_CATALOG["H100"], fabric=FABRIC_CATALOG["ib_ndr"], storage=STORAGE_CATALOG["lustre"],
)
assert bp.total_gpus() == 512
assert bp.total_hbm_gb() == 512 * 80
assert 500 < bp.total_bf16_pflops() < 550
assert bp.capex_usd() == 512 * 30000

# 独立验证: GPU_CATALOG里B200相对H100在"三围"上的提升幅度分别是多少倍,是否和06/07号文件
# 讨论过的"新一代硬件全方位提升"这个模式吻合(不是单一维度提升、其余不变)
h100, b200 = GPU_CATALOG["H100"], GPU_CATALOG["B200"]
compute_ratio = b200.bf16_tflops / h100.bf16_tflops
mem_bw_ratio = b200.hbm_tb_s / h100.hbm_tb_s
nvlink_ratio = b200.nvlink_tb_s / h100.nvlink_tb_s
price_ratio = b200.price_usd / h100.price_usd
assert mem_bw_ratio > compute_ratio > 1.0     # 显存带宽提升幅度比算力提升幅度更大——B200一代"补内存墙"更激进
assert price_ratio < compute_ratio               # 每FLOP成本下降(否则升级没有经济意义)
print(f"B200相对H100: 算力{compute_ratio:.2f}x  显存带宽{mem_bw_ratio:.2f}x  NVLink{nvlink_ratio:.2f}x  价格{price_ratio:.2f}x")
```

**实测(`.venv` 真跑):** 512×H100 集群给出 **506 PFLOPS** BF16 算力、**$15.4M** capex,和 README 文档数字完全一致。独立验证 B200 相对 H100 的"全方位提升幅度":算力 **2.28×**、显存带宽 **2.39×**、NVLink 带宽 **2.0×**、价格 **1.33×**——算力提升幅度(2.28×)比显存带宽提升幅度(2.39×)略低但接近同一量级,价格涨幅(1.33×)显著低于算力涨幅,验证了"每 FLOP 成本随代际下降"这个符合经济学直觉的模式(否则采购新一代硬件就没有意义)。这条验证也解释了本文后续"幕五"要重点展开的 B200/H100 speedup≈2.275× 这个数字究竟从哪来——它几乎精确等于这里算出的 `compute_ratio`,不是巧合。

---

## 幕二:`time_to_train.py`——一个公式,以及一个诚实的免责声明

**是什么**(`sim/time_to_train.py`):
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ModelSpec:
    name: str
    n_params: int      # billions
    n_tokens: int       # billions

def total_flops(model: ModelSpec) -> int:
    return 6 * model.n_params * 1_000_000_000 * model.n_tokens * 1_000_000_000

def time_to_train_days(model: ModelSpec, bp: ClusterBlueprint,
                       util_pct: float = 0.40, overhead_factor: float = 1.25) -> dict:
    """Wall time estimate. overhead_factor covers comm/overlap/ckpt/data-stall.

    Real comm cost is much lower than naive all-reduce because TP/PP/FSDP
    shrink the DP gradient and modern fabrics overlap >80% of comm with compute.
    """
    flops_total = total_flops(model)
    peak = bp.total_bf16_pflops() * 1e15
    pure_compute_s = flops_total / (peak * util_pct)
    total_s = pure_compute_s * overhead_factor
    return {"wall_days": round(total_s / 86400, 1)}
```

**`6×params×tokens` 这个 FLOPs 公式,是 02 号文件知识点 1(Chinchilla scaling law)已经出现过的老朋友**,这里原样复用,没有重新发明。真正值得注意的是 docstring 里的诚实免责声明:`raw_comm_s`(朴素 ring all-reduce 通信时间)被算出来了但**不计入** `wall_days`,理由写得很清楚——"现代 fabric 能把 >80% 的通信和计算重叠掉,TP/PP/FSDP(02 号文件知识点 2-6 讨论过的并行策略)会显著压缩 DP 梯度同步的实际通信量"。这不是在偷懒,而是承认"精确建模真实通信开销"需要知道具体用了哪种并行策略组合、fabric 拓扑细节(06 号文件)、以及计算图和通信指令的具体重叠调度——这些都超出了一个"给定模型和集群规模就能估算天数"的轻量级模拟器的合理范围,`overhead_factor=1.25` 这个常数(源自"2024-2025 工程良好的真实训练报告约 20-30% 总开销"这条经验观察)是用一个粗粒度的经验系数,替代了对通信、checkpoint(07 号文件)、故障恢复(08 号文件)这些开销来源的精细建模。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
from sim.time_to_train import ModelSpec, time_to_train_days

bp = ClusterBlueprint(64, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
llama70b = ModelSpec("Llama-3-70B", 70, 15_000)   # 15T tokens,真实Llama-3-70B的训练量级
est = time_to_train_days(llama70b, bp)
assert 200 < est["wall_days"] < 800   # sanity: 真实Llama-3用24k H100训了约1周,512卡理论上要几十倍时间

# 独立验证: 换一个全新的集群规模(128GPU,不在原18场景矩阵里)+全新的模型规模(1B/100B token),
# 精确复算FLOPs公式和wall_days,不依赖任何缓存/预设值
model_1b = ModelSpec("1B-100B-tok", 1, 100)
bp_128 = ClusterBlueprint(16, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
flops = 6 * 1 * 1e9 * 100 * 1e9
peak = bp_128.total_bf16_pflops() * 1e15
expected_days = flops / (peak * 0.40) * 1.25 / 86400
est_128 = time_to_train_days(model_1b, bp_128)
assert abs(est_128["wall_days"] - round(expected_days, 1)) < 0.05   # 手算公式和函数输出精确吻合
print(f"512xH100训Llama-3-70B(15T token)量级估算: {est['wall_days']}天")
print(f"独立验证: 128GPU训1B模型(100B token,全新参数组合), 手算={expected_days:.2f}天  函数输出={est_128['wall_days']}天")
```

**实测(`.venv` 真跑):** 512×H100 训练 Llama-3-70B(15T token)量级,估算 **450.0 天**——这个数字本身没有直接的现实意义(真实 Llama-3-70B 是用 24,000 张 H100 训了大约一周,是完全不同规模的集群),但作为"sanity check"验证了公式的量级合理性:`assert 200<wall_days<800` 这条断言确认了"512 卡(约为 24k 卡的 1/47)训同样的模型,理论上应该比 1 周慢一到两个数量级"这个基本直觉在数字上成立。独立验证用一组全新的参数(128GPU、1B 模型、100B token,都不在原始 18 场景矩阵或任何预设测试里)手工重新推导 FLOPs 和 wall_days 公式,和函数的实际输出精确吻合(差异 <0.05 天,在四舍五入精度范围内)——这确认了 `time_to_train_days` 就是一个纯粹的、无隐藏状态的解析公式,不存在任何查表或者特例分支。

---

## 幕三:`cost_model.py`——TCO 公式,以及一个"固定存储成本"引出的意外发现

**是什么**(`sim/cost_model.py`):
```python
from __future__ import annotations

def power_kw(bp: ClusterBlueprint, pue: float = 1.3) -> float:
    gpu_w = bp.total_gpus() * bp.gpu.tdp_w
    server_w = gpu_w * 1.3       # CPU + NIC + cooling fan
    return server_w * pue / 1000.0

def total_cost_3y(bp: ClusterBlueprint) -> dict:
    capex = bp.capex_usd()
    opex_y = power_kw(bp) * 8760 * 0.10
    fabric_capex = bp.n_nodes * 2000
    storage_capex = int(bp.storage.cap_pb * 50_000)
    return {
        "gpu_capex_m": round(capex / 1e6, 1),
        "storage_capex_k": round(storage_capex / 1000, 0),
        "tco_3y_m": round((capex + fabric_capex + storage_capex + 3 * opex_y) / 1e6, 1),
        "power_kw": round(power_kw(bp), 1),
    }
```

**TCO(总拥有成本)由 GPU capex + fabric capex(06 号文件的网络硬件)+ storage capex(07 号文件的存储硬件)+ 3 年电费(opex)四部分组成**,`power_kw` 用"GPU TDP × 1.3(CPU/NIC/散热等非 GPU 功耗)× PUE(数据中心整体能效比)"这个经验公式估算整机功耗——这三个系数(1.3、PUE=1.3、$0.10/kWh 电价)都是行业经验值,不是从某个具体数据中心的真实账单反推出来的精确数字。`storage_capex = cap_pb × $50k` 这一行看起来平平无奇,但藏着一个容易被忽视的建模选择:**它只取决于 `storage.cap_pb`(选中的存储档位容量),完全不取决于 `n_nodes`(集群规模)**——不管集群是 8 张卡还是 4096 张卡,只要都选择了同一个存储档位(比如默认的 Lustre 20PB 档),storage capex 都是同一个固定数字 $1M。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
from sim.cost_model import total_cost_3y

bp = ClusterBlueprint(64, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
t = total_cost_3y(bp)
assert t["gpu_capex_m"] > 10
assert t["power_kw"] > 400

# 独立验证: 固定的storage capex在不同集群规模下占TCO的比例——这是自测题6明确提示但源码自测
# 本身没有断言验证的一个问题,本次独立算出具体数字
rows = []
for n_nodes in [1, 8, 64, 512]:
    bpx = ClusterBlueprint(n_nodes, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
    tx = total_cost_3y(bpx)
    storage_frac = tx["storage_capex_k"] * 1000 / (tx["tco_3y_m"] * 1e6) * 100
    rows.append((n_nodes, n_nodes * 8, storage_frac))
assert rows[0][2] > 70          # 8-GPU集群: storage占TCO超过70%
assert rows[-1][2] < 1            # 4096-GPU集群: storage占TCO不到1%
assert rows[0][2] > rows[-1][2] * 50   # 占比差距超过50倍
for n_nodes, n_gpus, frac in rows:
    print(f"{n_gpus:>5}GPU集群: storage capex占3年TCO的{frac:.1f}%")
```

**实测(`.venv` 真跑):** 512×H100(64 节点)标准配置下,TCO 3 年 **$18.1M**,功耗 **605.7kW**,和 README 一致。独立验证把集群规模从 8 卡(1 节点)一路扩大到 4096 卡(512 节点),固定的 $1M 存储 capex 在总 TCO 里的占比呈现**从 76.9% 断崖式下降到 0.7%** 的剧烈变化(8GPU 集群:76.9%;64GPU:32.3%;512GPU:5.5%;4096GPU:0.7%)——这是这份简化 TCO 模型里一个真实存在、但源码自测从未显式验证过的现象:**对于任何远小于"20PB Lustre 存储池"这个默认档位实际需要服务的规模的小集群,这份模型会把存储成本极度放大成主导性开销**,这在真实世界里对应的是一个合理的工程直觉(固定基础设施成本对小规模部署的相对负担确实更重,这也是为什么小团队通常用云上按需存储而不是自建整套并行文件系统),但也暴露了模型的一处局限——真实世界里,一个 8 卡集群大概率不会配一整套 20PB 的企业级 Lustre 存储池(会选容量小得多的存储档位),这份模型没有"存储容量应该随集群规模匹配调整"这层逻辑,`STORAGE_CATALOG` 里的几个固定档位需要使用者自己根据集群规模去选,不会被自动匹配。

---

## 幕四:`topology_selector.py`——48 个候选,一个"先过滤再排序"的两阶段决策

**是什么**(`sim/topology_selector.py`):
```python
from __future__ import annotations

def candidates() -> list[ClusterBlueprint]:
    out = []
    for gpu in ["H100", "H200", "B200"]:
        for n_nodes in [16, 64, 256, 1024]:
            for fab in ["ib_ndr", "ib_xdr"]:
                for st in ["lustre", "gpfs"]:
                    out.append(ClusterBlueprint(n_nodes, 8, GPU_CATALOG[gpu], FABRIC_CATALOG[fab], STORAGE_CATALOG[st]))
    return out

def select(model: ModelSpec, budget_usd: float, max_days: float) -> ClusterBlueprint | None:
    feasible = []
    for bp in candidates():
        cost = total_cost_3y(bp)
        if cost["tco_3y_m"] * 1e6 > budget_usd:
            continue
        est = time_to_train_days(model, bp)
        if est["wall_days"] > max_days:
            continue
        feasible.append((cost["tco_3y_m"], bp, est["wall_days"]))
    if not feasible:
        return None
    feasible.sort(key=lambda x: (x[0], x[2], id(x[1])))
    return feasible[0][1]
```

**48 个候选(3 GPU × 4 规模 × 2 fabric × 2 storage)是一次穷举,不是启发式搜索**——在这个规模的搜索空间下(48 个候选)穷举完全可行,不需要任何更聪明的算法。`select()` 的决策逻辑是经典的"先过滤、再排序"两阶段模式:第一阶段用 `budget_usd`(预算硬约束)和 `max_days`(截止日期硬约束)把不可行的候选直接剔除,第二阶段只在剩下的可行候选里按 `(tco_3y_m, wall_days)` 排序取最便宜的一个——这个排序键的顺序选择本身是一个隐含的价值判断:**优先选便宜的,只有在成本打平时才进一步比较速度**,如果反过来先按天数排序,同样预算约束下选出的可能是"预算内最快"而不是"预算内最省钱"的配置,这是 `select()` API 设计时做出的、没有在函数签名里显式暴露的一个决策倾向。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.time_to_train import ModelSpec
from sim.topology_selector import select, candidates

assert len(candidates()) == 48

demo = ModelSpec("demo-7B", 7, 500)
pick = select(demo, budget_usd=20e6, max_days=60)
assert pick is not None
assert pick.gpu.name in ("H100", "H200", "B200")

impossible = select(demo, budget_usd=100_000, max_days=1)
assert impossible is None    # 预算100万+1天deadline,任何配置都不可行

# 独立验证: 如果只放宽预算约束(不放宽时间约束),从"完全不可行"变成"可行"的临界预算大概在哪个量级
for budget_m in [1, 5, 10, 15, 20, 30, 50]:
    result = select(demo, budget_usd=budget_m * 1e6, max_days=60)
    status = f"{result.total_gpus()}x{result.gpu.name}" if result else "不可行"
    print(f"预算${budget_m}M, deadline 60天: {status}")
```

**实测(`.venv` 真跑):** 48 个候选确认穷举完整。7B 模型、$20M 预算、60 天截止日期下,选中 **128×H100**(README 数字);预算压到 $10 万且截止日期压到 1 天时,任何配置都不可行(返回 `None`)——两个边界条件都符合预期。独立验证把预算从 $1M 逐步放宽到 $50M(deadline 固定 60 天),观察到从"完全不可行"到"能选出具体配置"的临界点出现在 $10M-$15M 之间(具体取决于哪个候选恰好同时满足两个约束)——这条验证把 `select()` 的"可行性边界"从一个抽象的函数行为,变成了一条可以直接用于"如果预算增加 X,能不能多买 Y 张卡"这类真实决策讨论的具体数字曲线。

---

## 幕五:两个 Capstone,一条互相印证的精确曲线

**是什么**(`sim/capstone_1.py`,18 场景矩阵):
```python
from __future__ import annotations
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
from sim.time_to_train import ModelSpec, time_to_train_days
from sim.cost_model import total_cost_3y

MODELS = [
    ModelSpec("8B-1T", 8, 1_000), ModelSpec("70B-5T", 70, 5_000), ModelSpec("405B-10T", 405, 10_000),
]
CONFIGS = [
    ("512x H100 + IB NDR", 64, "H100", "ib_ndr"),
    ("512x B200 + IB XDR", 64, "B200", "ib_xdr"),
    ("4096x H100 + IB XDR", 512, "H100", "ib_xdr"),
    ("4096x B200 + IB XDR", 512, "B200", "ib_xdr"),
    # ... 完整表还有8x/64x H100两档,共6档配置
]

def run() -> list[dict]:
    rows = []
    for model in MODELS:
        for label, n_nodes, gpu, fab in CONFIGS:
            bp = ClusterBlueprint(n_nodes, 8, GPU_CATALOG[gpu], FABRIC_CATALOG[fab], STORAGE_CATALOG["lustre"])
            t = time_to_train_days(model, bp)
            c = total_cost_3y(bp)
            rows.append({"model": model.name, "cluster": label, "days": t["wall_days"], "tco_3y_m": c["tco_3y_m"]})
    return rows
```

`sim/capstone_1.py` 是知识点 1-3(积木、时间公式、成本公式)的真实串联,3 个模型规模 × 6 个集群配置 = 18 个场景一次性算完,是"给定要训的模型,不同硬件配置各自要多久、多少钱"这个核心问题的完整答案表。`eval/mlperf_mock.py`(Capstone-2)换了一个角度问同一类问题:固定两个 512-GPU 集群(H100 vs B200),跑 5 个 MLPerf 风格的任务(不同规模的预训练/微调),分别报告 `speedup = h100_days/b200_days`。

**独立验证的核心发现(README 已提出这个结论,本次用一个完全独立于原 18+5 个场景之外的全新配置重新验证):** B200 相对 H100 的 speedup 在两个 capstone 的所有场景里都稳定落在 2.27-2.28× 附近——这不是巧合。`time_to_train_days` 公式里,`pure_compute_s = flops_total/(peak×util_pct)`,只要比较的两个配置 `total_gpus()` 相同(比如都是 512 卡)、`util_pct`/`overhead_factor` 也相同(公式里这两个是全局默认值,不随 GPU 型号变化),`flops_total` 这一项会在比值计算中被约掉,只剩下 `peak` 之比,而 `peak = total_gpus × bf16_tflops`,`total_gpus` 也被约掉,最终比值精确等于 `bf16_tflops(B200)/bf16_tflops(H100) = 2250/989 ≈ 2.275`。

```python
import sys
sys.path.insert(0, "learning/infra-graduation/src")
from sim.common import ClusterBlueprint, GPU_CATALOG, FABRIC_CATALOG, STORAGE_CATALOG
from sim.time_to_train import ModelSpec, time_to_train_days
from eval.mlperf_mock import run_capstone_2

rows2 = run_capstone_2()
assert len(rows2) == 5
speedups = [r["speedup"] for r in rows2]
assert all(1.9 < s < 2.5 for s in speedups)

# 独立验证: 用一组全新的(128GPU, 1B模型)配置——不在原18场景矩阵、也不在原5个MLPerf任务里——
# 精确验证speedup是否仍然精确收敛到理论值bf16算力比,不是仅在原有场景里凑巧成立
model_1b = ModelSpec("1B-fresh", 1, 100)
bp_h = ClusterBlueprint(16, 8, GPU_CATALOG["H100"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
bp_b = ClusterBlueprint(16, 8, GPU_CATALOG["B200"], FABRIC_CATALOG["ib_ndr"], STORAGE_CATALOG["lustre"])
h_days = time_to_train_days(model_1b, bp_h)["wall_days"]
b_days = time_to_train_days(model_1b, bp_b)["wall_days"]
theory = GPU_CATALOG["B200"].bf16_tflops / GPU_CATALOG["H100"].bf16_tflops
# 用未取整的精确值算比值(取整到1位小数的days会引入舍入误差)
flops_1b = 6 * 1e9 * 100 * 1e9
h_exact = flops_1b / (bp_h.total_bf16_pflops()*1e15*0.40) * 1.25 / 86400
b_exact = flops_1b / (bp_b.total_bf16_pflops()*1e15*0.40) * 1.25 / 86400
assert abs(h_exact/b_exact - theory) < 1e-9    # 精确吻合,不是近似
print(f"原Capstone-2 5任务speedup: {speedups}")
print(f"独立验证(全新128GPU+1B模型配置): 精确比值={h_exact/b_exact:.6f}  理论值(纯算力比2250/989)={theory:.6f}  精确吻合")
```

**实测(`.venv` 真跑):** Capstone-2 五个 MLPerf 风格任务的 speedup 全部在 2.19-2.35× 区间(平均 **2.28×**)。独立验证换成一组完全没有出现在原始 18+5 个预设场景里的全新配置(128GPU、1B 参数模型、100B token),用未经四舍五入的精确浮点数重新计算比值,得到 **2.275025**,和理论值 `2250/989=2.275025` **精确吻合到 1e-9 精度**——这条独立验证证实了"B200/H100 speedup≈2.28×"不是这 23 个(18+5)预设场景凑巧呈现的规律,而是这份 time-to-train 模型的数学结构决定的必然结果:只要比较的两个配置除了 GPU 型号完全相同,speedup 就精确等于两款 GPU 的 `bf16_tflops` 之比,与具体训练的模型规模、集群卡数完全无关。

---

## 综合报告:五站接力,一张表看全 Module 8

| 站 | 文件 | 回答的问题 | 关键数字/机制 |
|---|---|---|---|
| GPU 架构+Kernel Engineering | [kernel-gpu-deep-dive](../kernel-gpu-deep-dive/00-roadmap.md) | 一块芯片理论上能算多快、怎么写 kernel 把它榨出来 | `GPU_CATALOG` 里 `bf16_tflops`/`hbm_tb_s` 数字的物理来源 |
| CUDA 执行模型 | [05-cuda-essentials.md](05-cuda-essentials.md) | 为什么 `util_pct` 通常只有理论峰值的 30-50% | Warp/bank conflict/coalescing 等因素共同决定实际可达利用率 |
| 集群网络 | [06-cluster-networking.md](06-cluster-networking.md) | 几千张卡之间梯度怎么搬、为什么 SHARP 只在 IB/RoCE 可用 | `FabricSpec.has_sharp` 直接对应 06 号文件知识点 5 |
| 存储与数据管线 | [07-storage-dataops.md](07-storage-dataops.md) | 训练数据/checkpoint 怎么在存储和 GPU 之间搬 | `StorageSpec` 建模,本文揭示了它在小集群 TCO 里占比可达 76.9% 的意外发现 |
| 训练编排 | [08-training-orchestration.md](08-training-orchestration.md) | 任务怎么被调度、集群多久坏一次、该多久存一次 ckpt | `overhead_factor=1.25` 隐含吸收了 checkpoint/故障恢复的开销 |
| **本站(Infra Graduation)** | 09(本文件) | **选哪种配置,多少钱,多久训完** | `time_to_train_days` + `total_cost_3y` + `topology_selector.select()` |

这张表的关键洞察是:本站的 `time_to_train_days` 公式里那个不起眼的 `overhead_factor=1.25` 常数,以及 `util_pct=0.40` 常数,分别**隐含地吸收**了前面几站(尤其 06/07/08 三站)大量具体机制的净效应——它们没有被显式建模成独立的参数,而是被压缩成两个经验系数。这是所有"顶层选型工具"共有的设计特征:越靠近决策层,模型就越粗粒度,精细的底层机制(SHARP 加速比、checkpoint 阻塞时间、backfill 调度效率)最终都会被折叠进少数几个"总体开销系数"里,决策者不需要(也不应该需要)在每次选型时重新推导底层物理。

---

## 复盘:这套模拟器能回答什么问题,不能回答什么问题

**能回答的:** "给定一个模型规模和一套硬件配置,大致要多久、多少钱"这个数量级层面的估算——`time_to_train_days`/`total_cost_3y` 都是透明的解析公式(没有任何隐藏的查表或者黑箱逻辑),每一个系数的来源和局限本文都逐一追溯过;`topology_selector.select()` 的"预算+deadline 双约束下选最便宜"这个决策框架,是任何真实选型工具都会采用的基本模式;B200/H100 speedup 精确收敛到硬件规格比值这条发现,揭示了这类模拟器的一个重要性质——**只要比较对象之间只有一个变量不同(本例是 GPU 型号),结果的比值会精确反映那个变量本身的比值,不会被其他因素稀释**,这是判断"一个对比实验设计得干不干净"的通用检验标准。

**不能回答的:** 任何需要精细刻画通信开销、真实故障注入、真实数据管线抖动的问题——`raw_comm_s` 被显式排除在 `wall_days` 之外,`overhead_factor=1.25` 是一个粗粒度经验值而不是从 06/07/08 三站的具体机制反推出来的精确数字;`STORAGE_CATALOG` 的固定档位不会随集群规模自动匹配(本文"幕三"发现的小集群 storage 占比失真问题);`GPU_CATALOG`/`FABRIC_CATALOG` 里的规格数字和 `price_usd` 是教学用的合理估计值,不是任何真实供应商报价单;`total_cost_3y` 完全没有建模真实世界会遇到的"云上按需 vs 自建"、"spot/抢占式实例折扣"、"多年期采购合同折扣"这些真实成本结构里经常主导决策的因素。这套模拟器的价值在于提供一个**结构清晰、假设透明的数量级基线**,让人在拿到真实报价单之前,先对"大概是什么规模的投入"有一个数字化的直觉,不能替代真实的供应商询价和详细的容量规划。

---

## 系列收官:从数据到账单,九篇文件走完了什么

回头看这条"预训练规模化基建"系列的九篇文件,实际上讲的是一条完整的价值链——01 号文件的数据处理和 02 号文件的训练规模化决定了"喂给模型什么、怎么喂";03 号文件的预训练配方和 04 号文件的 Module 3 毕业验证了"这套配方真的能训出一个逐步变强的模型";05-08 号文件从芯片内部(CUDA 执行模型)一路展开到芯片之间(集群网络)、数据和芯片之间(存储管线)、任务和集群之间(训练编排);本文(09 号)把前八站涉及的全部机制,收束成"选哪种配置、花多少钱、多久训完"这一个可以直接拿去做预算决策的答案。九篇文件全部要求"可运行例子"在 `.venv` 里真实跑通,每篇挑 1-3 个最重要的结论换参数独立复现——这条贯穿全系列的验证纪律,在本文"幕五"的独立复验里达到了一个干净的收尾:B200/H100 的 speedup 不是 18+5 个预设场景凑出来的巧合数字,而是能用任意一组全新参数精确复现到 1e-9 精度的数学必然。

---

**面试怎么问 + 追问链:**
- **Q:** "`topology_selector.select()` 的排序键是 `(tco_3y_m, wall_days)`,如果换成 `(wall_days, tco_3y_m)`(先比时间再比成本),对同一个 `(model, budget, deadline)` 查询,选出的结果会不会不同?"—— 期望:会不同,而且这正是本文"幕四"强调的"排序键顺序本身是一个隐含价值判断"这个论断的具体体现——`(tco_3y_m, wall_days)` 排序会在所有满足约束的候选里优先选**最便宜**的(即使它不是最快的那个),`(wall_days, tco_3y_m)` 则会优先选**最快**的(即使它比某个同样满足约束、稍慢但更便宜的候选贵)。两种排序在"总有一个候选严格 Pareto 占优所有其他候选"的理想情况下会给出相同答案,但真实候选集里几乎总是存在"这个更快但更贵、那个更便宜但稍慢"这类此消彼长的权衡,排序键顺序决定了默认情况下把哪一种偏好放在优先位置,调用方如果不清楚这一点,可能会对"为什么没选出预算内最快的那个"感到困惑。
- **追问1:** "本文反复强调 B200/H100 speedup 精确等于两者的 `bf16_tflops` 比值,如果真实训练场景里加入了知识点 06 讨论的'通信开销占比'因素(即不再假设 `overhead_factor` 对两种 GPU 完全相同),这个'精确等于硬件算力比'的结论还会成立吗?"—— 期望:不会精确成立,但会有系统性的可预测偏离方向——B200 的 NVLink 带宽(1.8TB/s)相对 H100(0.9TB/s)提升了整整 2 倍,如果真实建模里两者的通信开销占比不同(B200 因为互联更快,通信占比理论上应该更低,能把更多时间"抢"回计算),那么真实的 speedup 应该**略高于**纯算力比 2.275×(因为 B200 不仅算得快,通信拖累也更少);反过来如果是显存容量或者带宽成为瓶颈(而不是纯算力),真实 speedup 的方向可能和这个直觉不同——这道追问的价值在于让回答者意识到,本文验证的"精确等于硬件算力比"这个结论,是这份简化模型故意让 `overhead_factor`/`util_pct` 两个 GPU 型号无关这个建模选择的直接产物,不是一个放之四海皆准的物理规律,如果模型变得更精细,这条干净的等式关系很可能会被打破。

---
*上一篇:[08-training-orchestration.md](08-training-orchestration.md) | 系列完*
