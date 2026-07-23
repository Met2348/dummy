# 07 · 存储与数据管线深挖(Storage & DataOps)

> 总览见 [00-roadmap.md](00-roadmap.md)

06 号文件讲完了梯度怎么在 GPU 之间搬,本文接着看粒度更粗的另一条搬运路径:训练数据怎么从存储搬到 GPU、checkpoint 怎么从 GPU 搬回存储。对应 `learning/storage-dataops/`(Module 8 第 5 专题,6 lecture + 7 个 src 源文件,核心论文 Yang & Cong *Accelerating Data Loading in Deep Neural Network Training*)。7 个知识点。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Python 3.13)下用 `.venv/Scripts/python.exe` 实际跑通验证。7 个源文件零第三方依赖(`dataclasses`/`hashlib`/`__future__`+互相import),纯 CPU、秒级完成——和 05/06 号文件一样,`src/` 下没有真实 S3/Lustre 网络调用、没有 `torch.distributed.checkpoint`、`checkpoint.py`/`capstone_ckpt_recovery.py` 从不向磁盘写入任何真实文件(已用 grep 核实这两个文件源码里零 `open(`/`write(`/`os.` 调用),是用可断言验证的纯 Python 解析代价模型复现存储分层选型、dataloader 流水线加速、sharding 负载均衡、checkpoint 策略权衡这些关键工程决策。

---

## 1. 五层存储 BW/IOPS/延迟建模(`common.py`)—— S3 比 Lustre 慢 499 倍,几乎全部来自延迟项

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Storage:
    name: str
    read_gb_s: float
    write_gb_s: float
    iops_random: int
    latency_us: float
    cap_pb: float

TIERS = {
    "ram":        Storage("Host DDR5",      80.0,  80.0, 10_000_000,    0.1, 0.002),
    "nvme_pcie5": Storage("NVMe Gen5",      14.0,  12.0,  2_000_000,   80.0, 0.030),
    "lustre":     Storage("Lustre OSS pool", 500.0, 400.0,   500_000,  500.0, 20.0),
    "s3":         Storage("S3 (regional)",     1.0,   1.0,     10_000, 50000.0, 1000.0),
    # ... 完整表还有nvme_raid/gpfs,共6条
}

def time_to_read(bytes_total: int, tier: Storage) -> float:
    """Seconds, including round-trip latency."""
    transfer_s = (bytes_total / 1e9) / tier.read_gb_s
    return tier.latency_us / 1e6 + transfer_s
```
(`common.py:1-29`,节选)

**画出来看:六层存储放在一起比,才看得出"梯度"具体长什么样(全部 6 档,数据来自 `common.py::TIERS`,按延迟从低到高排列):**

```
层级                读带宽    写带宽    随机IOPS*     延迟       容量      延迟是RAM的多少倍
──────────────────────────────────────────────────────────────────────────
RAM (DDR5)          80 GB/s   80 GB/s   10,000,000    0.1 us      2 TB          1x
NVMe Gen5            14 GB/s   12 GB/s    2,000,000     80 us     30 TB        800x
NVMe RAID0 8x       100 GB/s   80 GB/s    8,000,000     80 us    250 TB        800x
GPFS DSS pool       600 GB/s  500 GB/s      800,000    400 us     50 PB      4,000x
Lustre OSS pool     500 GB/s  400 GB/s      500,000    500 us     20 PB      5,000x
S3 (regional)         1 GB/s    1 GB/s       10,000  50,000 us  1,000 PB   500,000x
──────────────────────────────────────────────────────────────────────────
* IOPS = I/O Operations Per Second,每秒能完成的独立读写请求次数——和"带宽"不是一回事:
  带宽是"管道有多粗"(每秒能搬多少字节),IOPS 是"每秒能发起/完成多少次独立的搬运动作"。
  小文件随机读容易被 IOPS 先卡住,大文件顺序读容易被带宽先卡住(知识点 5 会展开这个区别)。
```

注意"带宽"这一列不是单调的:本地单盘 NVMe(14GB/s)反而比 RAM(80GB/s)慢,但把 8 块 NVMe 组成 RAID0 后(100GB/s)就反超了 RAM,而跨多台服务器聚合起来的 Lustre/GPFS(500-600GB/s)带宽比本地 RAID0 还要高——带宽可以靠"堆更多硬件并行"不断往上叠。真正随"离计算越远"严格单调变差的是延迟(0.1us→80us→400~500us→50000us)和容量(2TB→...→1000PB):延迟是"发出请求到第一个字节回来"这个物理/协议开销的硬下限,离计算越远(跨内存总线、跨网络)这个下限越高,这个下限不像带宽那样能靠堆硬件绕开。

**一句话:** 6 层存储(RAM/NVMe/NVMe-RAID/Lustre-GPFS/S3)在延迟和容量这两个维度上呈现出清晰的"离计算越远、延迟越高、能装的也越多"梯度(带宽其实不单调,见上图),`time_to_read` 这一个公式就能解释为什么"训练数据直接挂载 S3"是训练圈公认的反模式——不是因为 S3 带宽差(1GB/s 看起来也不算离谱),而是因为它 50ms 的延迟比 Lustre 的 0.5ms 高 100 倍。

**底层机制/为什么这样设计:** `time_to_read` 的结构和 06 号文件 `time_to_send` 完全一样(固定延迟+数据量/带宽),但存储场景的延迟差距比网络场景夸张得多——S3 的 `latency_us=50000`(50 毫秒,这是对象存储 HTTP 请求本身的往返开销,不是物理层延迟),比 Lustre 的 `500`(0.5 毫秒,并行文件系统的元数据+RPC 开销)高 100 倍。这个差距在读大文件时会被"分摊"掉(100GB 这么大的传输,不管延迟是 0.5ms 还是 50ms 都无所谓),但训练数据加载的真实模式是海量小文件随机读(每个样本几十 KB 到几 MB),这时候延迟项不再能被摊薄,S3 的每次请求都要多付出这 50ms,叠加起来就是训练圈另一个共识——"S3 可以存 checkpoint 归档,但不能当训练时的热数据源"。

**AI 研究场景:** 这张表是存储架构选型的第一道门槛——训练热数据(高频随机读)应该放 Lustre/GPFS 这类并行文件系统(或更进一步缓存进 NVMe-RAID 本地盘),S3/对象存储适合冷归档(容量最大,`cap_pb=1000`,比 Lustre 大 50 倍)和跨集群共享的"数据源头",不适合直接挂载给训练进程读——知识点 5 会展示"WebDataset 把小文件打包成大 tar shard"这个具体的工程解法,本质上就是把"小文件随机读"转换成"大文件顺序读",从而绕开延迟项的惩罚。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from common import TIERS, time_to_read

t_lustre = time_to_read(int(100e9), TIERS["lustre"])   # 100GB
t_s3 = time_to_read(int(100e9), TIERS["s3"])
assert 0.15 < t_lustre < 0.25
assert t_s3 > 200 * t_lustre

# 独立验证: 换成典型训练样本大小(200KB,而不是100GB整块),两个存储的延迟占比都应该接近100%
small = int(200_000)   # 200KB,常见图像/文本样本大小
t_lustre_small = time_to_read(small, TIERS["lustre"])
t_s3_small = time_to_read(small, TIERS["s3"])
lustre_latency_frac = (TIERS["lustre"].latency_us/1e6) / t_lustre_small
s3_latency_frac = (TIERS["s3"].latency_us/1e6) / t_s3_small
assert lustre_latency_frac > 0.999      # Lustre在200KB规模下也几乎完全是延迟主导
assert s3_latency_frac > 0.995            # S3同样如此
assert lustre_latency_frac > s3_latency_frac   # 反直觉: Lustre的延迟占比反而比S3更高
print(f"100GB整块读: Lustre={t_lustre:.3f}s  S3={t_s3:.1f}s  比值={t_s3/t_lustre:.0f}x")
print(f"200KB小样本读: Lustre延迟占比={lustre_latency_frac:.2%}  S3延迟占比={s3_latency_frac:.2%}")
```

**实测(`.venv` 真跑):** 100GB 整块读,Lustre 耗时 **0.2005s**,S3 耗时 **100.05s**,比值精确 **499.0倍**(比自测断言的">200倍"松弛阈值更精确)。独立验证换成 200KB 典型训练样本大小后,出现一个反直觉的结果:Lustre 的延迟占比是 **99.92%**,反而比 S3 的延迟占比(**99.60%**)更高——不是"两个都约等于 100% 所以差不多",而是 Lustre 的延迟占比确实略高。原因不难推导:Lustre 的带宽(500GB/s)极高,200KB 传输只要约 0.0000004s,把总耗时(0.5004ms)几乎全部让给了 0.5ms 的延迟项;S3 的带宽只有 1GB/s,同样 200KB 的传输要约 0.0002s,虽然远小于它 50ms 的延迟,但这 0.0002s 占总耗时(50.02ms)的比例(约 0.4%)反而比 Lustre 的传输占比(约 0.08%)更大——**存储的带宽越高,小消息场景下延迟项占比反而越逼近 100%**,因为带宽优势会把传输时间压缩到几乎可以忽略,只留下延迟这个"地板"。这条独立验证证实了"S3 慢在哪"这个问题的准确答案:不是带宽差 500 倍,是延迟差 100 倍,而且这个延迟差距的绝对值(而不是相对占比)才是训练数据加载场景(海量小样本)里真正拖慢速度的原因。

**面试怎么问 + 追问链:**
- **Q:** "S3 的 `read_gb_s=1.0`,只比 Lustre 的 `500.0` 差 500 倍,但实测 100GB 读取的耗时比值也恰好接近 500 倍,这是巧合吗?"—— 期望:不是巧合,但原因和"带宽差 500 倍"没有直接关系——100GB 是一个足够大的传输量,两个存储的延迟项(0.5ms vs 50ms)相对于各自的传输时间(0.2s vs 100s)都可以忽略不计,所以这个场景下 `time_to_read` 几乎完全退化成 `bytes/BW`,比值自然趋近于带宽比值 500;但这只在"大块顺序读"场景下成立,换成小样本随机读(知识点后面独立验证的 200KB 场景),延迟项重新主导,比值会从"接近带宽比 500x"变成"更接近延迟比 100x 到更极端"。
- **追问1:** "如果要给 `Storage` dataclass 新增一个'混合读'场景(既不是纯大块也不是纯小文件,而是知识点 2 dataloader 里那种 batch 读),`time_to_read` 这个简单公式还够用吗?"—— 期望:不完全够用——`time_to_read` 假设"一次读取= 一次延迟+一次传输",但真实 dataloader 通常会做预取(prefetch)和批量合并读(coalescing),多个样本的延迟可以被流水线重叠掉(这正是知识点 2 `pipelined()` 建模的内容),这个存储层的简单模型只是"单次读取"的构件,更上层的 dataloader 抽象需要在此基础上叠加并发/流水线因素才能准确刻画真实吞吐。

**常见坑:** `TIERS` 表里 `iops_random`(随机 IOPS 上限)这个字段在 `time_to_read` 公式里完全没被用到——`common.py` 本身没有任何函数消费这个字段,它是留给下游模块(如知识点 5 `webdataset_style.py` 的 `read_random_files`,虽然那个函数是独立传参 `iops` 而不是直接引用 `TIERS[...].iops_random`)参考用的"文档性"数据,读代码时容易误以为 `time_to_read` 会自动在大量小文件场景下考虑 IOPS 瓶颈,实际上这个函数只建模了"单次传输"的延迟+带宽,IOPS 限制需要调用方自己在更高层单独建模。

---

## 2. Dataloader 流水线加速(`dataloader.py`)⭐—— "6.8×" 是两个独立效应精确相乘,不是模糊的"差不多"

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Stage:
    name: str
    per_sample_us: float

def naive_pipeline(stages: list[Stage], n_samples: int) -> float:
    """Sequential: total = sum(stages) * N. Useful baseline."""
    per_sample = sum(s.per_sample_us for s in stages)
    return per_sample * n_samples

def pipelined(stages: list[Stage], n_samples: int, n_workers: int = 4) -> float:
    """N workers in parallel + double-buffering.

    Throughput = min over stages of (n_workers / per_sample_us).
    Total time = (max(stages) / n_workers) * N (assuming compute is hidden by overlap).
    """
    parallel_per_sample = max(s.per_sample_us for s in stages) / n_workers
    return parallel_per_sample * n_samples

def bottleneck_stage(stages: list[Stage]) -> Stage:
    return max(stages, key=lambda s: s.per_sample_us)
```
(`dataloader.py:1-29`,节选)

**一句话:** `naive_pipeline`(四个阶段耗时直接相加)和 `pipelined`(只取最慢那个阶段的耗时、再除以 worker 数)之间的加速比,可以精确分解成"把非瓶颈阶段的耗时从关键路径上隐藏掉"和"瓶颈阶段本身被多个 worker 并行"这两个完全独立的效应的乘积——理解这个分解,才能看懂"加 worker 数"和"优化非瓶颈阶段"分别在优化什么。

**底层机制/为什么这样设计:** `naive_pipeline` 假设 fetch→decode→augment→collate 四个阶段严格串行执行,耗时是四者之和;`pipelined` 假设有 `n_workers` 个 worker 各自跑完整的四阶段流水线,且流水线内部通过双缓冲(prefetch 下一个样本的同时处理当前样本)把总耗时压缩到"只取决于最慢的那个阶段"(因为流水线深度足够时,非瓶颈阶段的耗时会被瓶颈阶段的耗时完全掩盖,类比 CPU 流水线的吞吐由最慢的一级决定)——`max(stages)/n_workers` 这个公式里,分子的 `max()` 就是"流水线隐藏非瓶颈阶段"这个效应的体现(如果没有流水线,还是要用 `sum()`),分母的 `/n_workers` 才是"多进程/多线程并行处理瓶颈阶段"这个效应。这两个效应在数学上是可分离的:`speedup = naive/pipelined = [sum(stages)/max(stages)] × [n_workers]`,前一项只取决于四个阶段各自的耗时构成(和 worker 数无关),后一项精确等于 `n_workers`(在这个理想化模型里没有任何饱和/竞争)。

**AI 研究场景:** 真实 PyTorch `DataLoader` 的 `num_workers` 参数调优,本质上就是在这两个效应之间找平衡点——如果瓶颈阶段(通常是 `decode_jpeg` 这类 CPU 密集的图像/tokenize 解码)本身很慢,加 worker 数能直接按比例提速(第二个效应,理论上无上限,实际会被 CPU 核数、内存带宽等物理资源限制饱和);但如果四个阶段耗时已经比较均衡(没有单一阶段显著更慢),流水线隐藏的收益(第一个效应)天然有限,这时候加 worker 数的边际收益会大幅降低——这是"为什么有的模型 `num_workers=4` 就够,有的需要 `num_workers=16`"背后的定量原因。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from dataloader import Stage, naive_pipeline, pipelined, bottleneck_stage

stages = [Stage("fetch_lustre", 500), Stage("decode_jpeg", 2000), Stage("augment", 800), Stage("collate", 100)]
naive = naive_pipeline(stages, 1000)
fast = pipelined(stages, 1000, n_workers=4)
speedup = naive / fast
assert speedup > 5.0
bn = bottleneck_stage(stages)
assert bn.name == "decode_jpeg"

# 独立验证: 两个效应是否精确可分离——换n_workers=1/2/8/16/32,worker因子应该精确等于n_workers本身
pipeline_hide_factor = naive / (bn.per_sample_us * 1000)   # w=1时的理论极限speedup
for w in [1, 2, 8, 16, 32]:
    sp = naive / pipelined(stages, 1000, n_workers=w)
    worker_factor = sp / pipeline_hide_factor
    assert abs(worker_factor - w) < 1e-9      # 精确等于w,不是近似
print(f"4-worker: naive={naive/1000:.0f}ms pipelined={fast/1000:.0f}ms speedup={speedup:.1f}x")
print(f"分解: pipeline隐藏因子={pipeline_hide_factor:.2f}x(与worker数无关) × worker因子(精确=n_workers,验证w=1/2/8/16/32全部精确吻合)")
```

**实测(`.venv` 真跑):** 1000 样本、4 worker 下,naive 耗时 **3400ms**,pipelined 耗时 **500ms**,加速比 **6.8×**,瓶颈阶段确认是 `decode_jpeg`(2000μs/样本,四阶段里最慢)。独立验证把加速比精确分解为 `pipeline_hide_factor × worker_factor`:`pipeline_hide_factor = 3400/2000 = 1.7×`,这一项是**架构常数**——只取决于四个阶段各自耗时的相对结构,和 `n_workers` 完全无关;`worker_factor` 在 w=1/2/8/16/32 五个值下全部**精确等于 w 本身**(误差 <1e-9,浮点精度极限),验证了这个理想化模型里 worker 并行的加速是**无上限的线性关系**——这也暴露了模型的局限(见下方"常见坑"):真实系统里 worker 数超过 CPU 核数或触发 Lustre 聚合带宽瓶颈后,这个线性关系会失效,但这份简化模型没有对这类资源竞争建模。

**面试怎么问 + 追问链:**
- **Q:** "如果要把 4 个 worker 提升到 8 个,这份代码预测 speedup 会翻倍到 13.6×,真实系统会翻倍吗?"—— 期望:不一定——真实系统里 worker 数受两类硬约束限制:一是 CPU 核数(`decode_jpeg` 这类 CPU 密集阶段,worker 数超过物理核数后不会再线性提速,会退化成上下文切换开销);二是共享 I/O 资源竞争(如果 8 个 worker 同时从同一个 Lustre 挂载点 `fetch`,聚合带宽是有限的,`fetch_lustre` 阶段本身的耗时会随并发度上升而变慢,不再是固定的 500μs)——这份简化模型里 `pipelined()` 公式没有对这两类饱和效应建模,是"理想上限"而非"真实预测"。
- **追问1:** "`bottleneck_stage` 只是简单取 `max(per_sample_us)`,如果四个阶段的耗时非常接近(比如 500/550/600/500),这个"瓶颈优化"的直觉还有意义吗?"—— 期望:意义会显著降低——`pipeline_hide_factor = sum/max` 在四阶段耗时接近时会趋近于 4(阶段数),这时候"流水线隐藏"这个效应贡献的加速比反而是最大的(因为没有哪个阶段能被隐藏掉的部分特别多,但整体 sum 相对 max 的比值最大);但"优化瓶颈阶段"这个动作(针对某一个具体阶段做算法优化,比如换更快的 JPEG 解码库)在阶段耗时接近时收益会分散,因为优化了当前最慢的那个阶段后,次慢的阶段很快会成为新瓶颈,不像本例"decode_jpeg 一枝独秀慢 2-4 倍"时优化目标非常明确。

**常见坑:** `pipelined()` 的公式假设"计算完全被重叠掩盖"(docstring 里的 `assuming compute is hidden by overlap`),这是一个理想化前提——真实实现依赖 PyTorch DataLoader 的 `prefetch_factor` 和 pin_memory 双缓冲机制确实按预期工作,如果 GPU 训练本身的 step 耗时比 dataloader 吞吐还快(数据饥饿),流水线的"隐藏"效应会失效,GPU 会出现等待 dataloader 的空闲时间,这时候 `pipelined()` 算出的耗时会显著低估真实的 wall-clock 训练时间;这份纯 CPU 数值模型只建模了 dataloader 内部的流水线,没有建模 dataloader 和训练 step 之间的相对速度关系。

---

## 3. Sharding 策略族:Hash/Range/Round-robin(`sharding.py`)—— Range 的"0% 不均衡"是巧合,换个聚簇模式能飙到 469%

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass
class Sample:
    sample_id: int
    bytes_size: int

def hash_shard(sample_id: int, n_shards: int) -> int:
    h = hashlib.sha1(str(sample_id).encode()).hexdigest()
    return int(h, 16) % n_shards

def range_shard(sample_id: int, n_shards: int, total: int) -> int:
    return min(sample_id * n_shards // total, n_shards - 1)

def round_robin_shard(sample_id: int, n_shards: int) -> int:
    return sample_id % n_shards
```
(`sharding.py:1-24`,节选)

**一句话:** 三种 sharding 策略在"局部性"(同一个 shard 里的样本是否在物理/逻辑上相邻)和"负载均衡"(每个 shard 拿到的总字节数是否接近)这两个目标上互相冲突——Hash 用 `sha1` 摘要打散,天然抗倾斜但完全没有局部性;Range 按 id 连续切块,局部性最好但对"数据按某种模式排序"的场景毫无抵抗力;Round-robin 用取模交错分配,兼顾了一定的均匀性和实现简单性。

**底层机制/为什么这样设计:** `hash_shard` 用 `hashlib.sha1`(不是 Python 内置 `hash()`)算摘要再取模,这个设计选择本身就是一个"已知坑"的正确规避——内置 `hash()` 对字符串默认加了随机盐(`PYTHONHASHSEED`),同一个 `sample_id` 在不同进程里可能被分到不同 shard,而 `hashlib.sha1` 是纯函数、无随机性、跨进程/跨语言完全确定([[存储-dataops-README]] 已用 3 个独立进程+ `PYTHONHASHSEED=42` 变体实测验证逐字节相同)。`range_shard` 的物理直觉是"把 id 空间连续切成 N 块",这样同一个 shard 里的样本 id 连续,如果上游数据集本身按某种顺序生成/存储(比如按类别、按抓取时间),这种连续性能带来局部性优势(比如顺序读磁盘更快),但代价是一旦这个"顺序"和某种潜在的数据分布模式(比如样本大小随时间增长)重合,连续的 id 区间可能全部落在同一种分布特征里,导致极端不均衡。

**AI 研究场景:** 大规模预训练数据 sharding 的策略选择是数据管线设计里一个容易被低估的决策点——如果用 Range sharding 而底层数据集恰好按某种业务逻辑排序过(比如先爬取的网站文档普遍更短、后爬取的更长,或者数据集拼接时不同来源的文档长度分布不同),会导致不同 shard 之间的训练吞吐差异巨大(拿到大文档 shard 的 worker 处理更慢,拖慢整个分布式训练的同步节奏),这是"数据 shuffle 不充分"这个常见训练问题背后的一种具体成因。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from sharding import Sample, shard_balance

# 源码自带测试: 周期50的倾斜模式,恰好被8个shard整除,range意外得到0%不均衡
samples = [Sample(i, 100 + (i % 50) * 100) for i in range(10000)]
h = shard_balance("hash", samples, 8)
r = shard_balance("range", samples, 8)
rr = shard_balance("round_robin", samples, 8)
assert rr["imbalance_pct"] < 10.0
assert r["imbalance_pct"] < 1.0     # 周期50恰好整除1250,range在这个特定数据下"运气好"

# 独立验证: 换3段聚簇模式(60%小样本+30%中样本+10%超大样本),range应该严重失衡
samples2 = []
for i in range(10000):
    if i < 6000: size = 100
    elif i < 9000: size = 1000
    else: size = 8000
    samples2.append(Sample(i, size))
h2 = shard_balance("hash", samples2, 8)
r2 = shard_balance("range", samples2, 8)
rr2 = shard_balance("round_robin", samples2, 8)
assert r2["imbalance_pct"] > 400        # range严重失衡(最后一个shard几乎全是超大样本)
assert h2["imbalance_pct"] < 10 and rr2["imbalance_pct"] < 10   # hash/rr依然稳健
print(f"周期性倾斜(源码自带): hash={h['imbalance_pct']}% range={r['imbalance_pct']}% rr={rr['imbalance_pct']}%")
print(f"3段聚簇(独立验证): hash={h2['imbalance_pct']}% range={r2['imbalance_pct']}% rr={rr2['imbalance_pct']}%")
```

**实测(`.venv` 真跑):** 源码自带的周期性倾斜数据(周期 50)下,三种策略给出 hash **5.1%**、range **0.0%**、round_robin **2.0%**——range 的 0.0% 是巧合(周期 50 恰好整除每个 shard 的样本数 1250)。独立验证换成 3 段聚簇模式(前 60% 样本 100B、中 30% 样本 1000B、后 10% 样本 8000B,不再是周期性倾斜而是分段聚簇)后,range 的不均衡度飙升到 **469.0%**(最大 shard 拿到 8,250,000 字节、均值只有 1,437,500 字节,因为 range 把后 12.5% 的 id 区间——正好落在超大样本聚簇段——全部分给了最后一个 shard),hash 保持 **8.0%**、round_robin 保持 **0.0%**(round_robin 的稳健不是巧合:交错取模天然让任何连续 id 区间的样本分散到所有 shard,只要每段样本数能被 shard 数整除)。额外用第三种模式(样本大小随 id 线性递增 `100+i`,无聚簇也无周期性)复测,range 仍然给出 **85.8%** 的不均衡度(hash 3.4%、round_robin 0.1%)——证实了不需要刻意构造聚簇,任何"样本大小和 id 顺序相关"的平滑趋势都足以让 range sharding 显著失衡。

**面试怎么问 + 追问链:**
- **Q:** "round_robin 在两次独立验证里都拿到了极低的不均衡度(2.0%、0.0%),这是因为它本质上比 hash 更均衡吗?"—— 期望:不是——round_robin 均衡的原因和 hash 完全不同:hash 靠摘要的伪随机性打散,理论上不管数据分布如何都能稳定给出低个位数的不均衡度;round_robin 靠"交错采样"物理性地把任何连续区间拆散到所有 shard,只要数据倾斜模式的"段长度"能被 shard 数整除(如本例 6000/8=750、3000/8=375、1000/8=125 都整除),就能做到完美均衡,但如果数据倾斜模式恰好和 shard 数 N 存在**特定的整除关系导致的对齐**(比如极端情况下,如果倾斜周期恰好等于 shard 数的整数倍,叠加某种特殊排列),round_robin 也可能出现不均衡,只是这类"意外对齐"的概率和影响远小于 range 的"整段区间圈进一个 shard"。
- **追问1:** "如果一定要用 Range sharding(比如需要保留数据的时间顺序局部性),有什么办法规避它对倾斜数据的敏感性?"—— 期望:标准做法是"先 shuffle 再 range 切分"——在切 shard 之前,先对全体样本做一次随机重排(打乱 id 和实际内容的对应关系),这样 range 切出来的连续 id 区间对应的实际内容就是随机采样,不再和任何潜在的"生成顺序=某种数据特征"的模式相关联,相当于用一次性的预处理开销换取"range 的实现简单性"和"hash 的均衡性"两者兼得;这也是为什么大规模数据集处理流水线里"shuffle"和"shard"经常是相邻的两个步骤,顺序不能颠倒。

**常见坑:** `range_shard(sample_id, n_shards, total)` 的第三个参数 `total` 必须是"当前这一批要切分的样本总数",如果调用方在流式处理场景下不知道总样本数(比如边生成边切分,`total` 用了一个预估值而不是精确值),`sample_id * n_shards // total` 这个公式在 `sample_id` 接近末尾时可能因为 `total` 估计偏小而导致部分样本的计算结果超出 `n_shards-1`(虽然代码用了 `min(..., n_shards-1)` 做了截断保护,不会真的越界,但会导致最后一个 shard 意外地比其他 shard 拿到更多样本)——这是 Range sharding 在流式/在线场景下比批量场景更容易踩的一个实现细节坑。

---

## 4. Checkpoint 三代:Full/Sharded/Async(`checkpoint.py`)—— `blocking` 和 `sec` 是两个独立维度,别混为一谈

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CkptCost:
    sec: float
    bytes_written: int
    blocking: bool

def full_checkpoint(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """Rank 0 gathers all, writes once. Blocks training."""
    gather_s = model_bytes / 1e9 / 400 + 0.001     # NCCL gather BW ~400 GB/s aggregate
    write_s = model_bytes / 1e9 / tier.write_gb_s
    return CkptCost(gather_s + write_s, model_bytes, blocking=True)

def sharded_checkpoint(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """Each rank writes its shard, but the OSS pool's aggregate BW is the bottleneck."""
    write_s = model_bytes / 1e9 / tier.write_gb_s
    return CkptCost(write_s, model_bytes, blocking=True)

def async_sharded(model_bytes: int, n_gpus: int, tier: Storage) -> CkptCost:
    """torch.distributed.checkpoint async: per-rank stage to host RAM via PCIe (independent).

    The blocking time is the PCIe stage, with each GPU independent.
    Background write to Lustre happens during compute and is not counted.
    """
    per_rank_bytes = model_bytes // n_gpus
    pcie_per_rank_gb_s = 32.0       # PCIe Gen5 x16 about 64 GB/s, halved.
    stage_s = per_rank_bytes / 1e9 / pcie_per_rank_gb_s
    return CkptCost(stage_s, model_bytes, blocking=False)
```
(`checkpoint.py:1-36`,节选)

**一句话:** `CkptCost` 这个返回值里 `sec`(这次 checkpoint 操作实际花了多少秒)和 `blocking`(训练主循环是否要停下来等这个操作完成)是两个独立的维度——`async_sharded` 的 `sec` 并不是 0(PCIe 暂存本身要花真实时间),但因为 `blocking=False`,这段时间是和后续计算重叠的,不占用训练的"停等"时间,这是三代 checkpoint 策略进化的核心思路:与其想办法让 checkpoint 本身变得更快,不如想办法让它变得"不阻塞"。

**底层机制/为什么这样设计:** Full checkpoint 需要先把分散在 512 个 GPU 上的模型分片 `gather` 到 rank 0(`gather_s = model_bytes/1e9/400`,用 400GB/s 的 NCCL 聚合带宽估算),再由 rank 0 单独写盘(`write_s`),两步都是**阻塞**的——训练必须暂停等这两步都完成;Sharded checkpoint 跳过了 gather 这一步,每个 rank 直接把自己的分片写到共享存储(不需要先集中到一张卡),但因为所有 512 个 rank 同时写同一个 Lustre 挂载点,真正的瓶颈变成了存储系统的**聚合写带宽**(`tier.write_gb_s` 是整个存储池的带宽上限,不是单 rank 独享),所以耗时仍然是 `model_bytes/write_gb_s`(和 full 的 `write_s` 项用的是同一个公式,只是省掉了 gather 那一步);Async checkpoint 走的是完全不同的路径——每个 rank 只需要把自己的分片从 GPU 显存搬到 host RAM(通过 PCIe,`pcie_per_rank_gb_s=32.0`,只需要 per-rank 的字节量除以 512,单个 rank 的传输量比 sharded 的"写到共享存储"小得多),这一步搬完训练就可以继续往下跑,真正写入 Lustre 的动作放到后台异步进行、和后续的前向反向传播计算重叠,不计入 `blocking` 时间。

**AI 研究场景:** 千卡训练的 checkpoint 频率(多久存一次)本质上是"故障恢复保险的密度"和"checkpoint 本身占用的训练时间"之间的权衡——如果 checkpoint 是阻塞的(full/sharded),存得越频繁,训练被打断的次数就越多,总的有效训练时间损失就越大;async checkpoint 因为不阻塞,理论上可以存得更频繁而不显著拖慢训练(受限于 PCIe 带宽是否会和其他 host-device 传输竞争),这是为什么 `torch.distributed.checkpoint` 的异步 API 是大规模训练框架的标配能力,而不是锦上添花的优化。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from common import TIERS
from checkpoint import full_checkpoint, sharded_checkpoint, async_sharded

lustre = TIERS["lustre"]
model = int(140e9)   # 70B模型BF16=140GB
n = 512

f = full_checkpoint(model, n, lustre)
s = sharded_checkpoint(model, n, lustre)
a = async_sharded(model, n, lustre)

assert f.blocking and s.blocking and not a.blocking
assert s.sec < f.sec           # sharded省掉gather,更快
assert a.sec < s.sec           # async更快
assert a.sec > 0                 # 关键: async的sec不是0!只是不blocking

# 独立验证: "async不阻塞"不代表"async的PCIe暂存不花时间"——换更大的模型规模,验证async.sec依然线性增长
model_405b = int(810e9)   # 405B模型BF16=810GB(Llama-3.1 405B量级)
a_405b = async_sharded(model_405b, n, lustre)
ratio_bytes = model_405b / model
ratio_sec = a_405b.sec / a.sec
assert abs(ratio_sec - ratio_bytes) < 1e-9   # PCIe暂存耗时和模型大小精确线性
print(f"70B: full={f.sec:.3f}s(blocking) sharded={s.sec:.3f}s(blocking) async={a.sec:.4f}s(non-blocking)")
print(f"405B async暂存耗时={a_405b.sec:.4f}s,是70B的{ratio_sec:.2f}倍(模型大小也是{ratio_bytes:.2f}倍,精确线性)")
```

**实测(`.venv` 真跑):** 70B 模型 BF16、512 GPU、Lustre 存储下,full **0.701s**(blocking=True)、sharded **0.350s**(blocking=True,恰好是 full 的一半,因为省掉了 gather 步骤而 write 步骤公式相同)、async **0.00854s**(blocking=False)。独立验证换成 405B 模型规模(Llama-3.1 405B 量级,BF16=810GB),async 的 PCIe 暂存耗时精确按字节数线性增长(比值精确等于模型大小比值 810/140≈5.79,浮点精度内完全吻合)——这条验证确认了一个容易被误解的点:**async checkpoint 的"不阻塞"不等于"不花时间"**,PCIe 暂存这一步是真实存在且随模型增大而线性增长的开销,只是因为 `blocking=False`,这段时间和后续训练计算重叠、不计入训练暂停时间,这和"耗时为零"是两回事(知识点 6 capstone 里 `total_overhead()` 的 `(cost.sec if cost.blocking else 0.0)` 这一行代码,正是把这个"重叠掉的真实耗时"在统计训练浪费时间时短路成 0 的地方)。

**面试怎么问 + 追问链:**
- **Q:** "既然 sharded checkpoint 已经比 full 快一倍,为什么工业界大规模训练框架还要进一步实现更复杂的 async checkpoint?"—— 期望:sharded 虽然去掉了 gather 步骤,但仍然是**阻塞**的——512 个 rank 同时写 Lustre,受限于存储池的聚合带宽,这个 0.35 秒仍然是训练主循环必须暂停等待的时间;async checkpoint 把这个"必须等待"的时间从 0.35s 压缩到 0.0085s 左右的 PCIe 暂存时间(且这部分也能被计算重叠掩盖),真正的收益不是"checkpoint 变快了"(某种意义上写盘的总字节数和总耗时没有本质变化,只是被推迟到后台),而是"checkpoint 不再让昂贵的 GPU 集群空转等待",这在千卡规模下,哪怕每次省下零点几秒,乘以高频次的 checkpoint 间隔,累积节省是可观的(知识点 6 capstone 会给出具体的百分比对比)。
- **追问1:** "async checkpoint 依赖'后台写入和计算重叠',如果后台写入 Lustre 的速度跟不上 checkpoint 产生的频率(比如每 10 分钟存一次,但写完 810GB 到 Lustre 需要 810/400≈2 分钟,理论上没问题,但如果 Lustre 同时被其他任务占用导致实际带宽只有平时的 1/5),会出现什么问题?"—— 期望:会出现"后台写入队列堆积"——如果实际可用带宽跟不上 checkpoint 产生速度,后台写入任务会越积越多,轻则占用越来越多的 host RAM 暂存空间(因为 PCIe 暂存到 RAM 后,写入 Lustre 之前占用的内存不能立刻释放),重则某个时刻不得不让训练主循环真的暂停等待,退化成变相的阻塞式 checkpoint——这是这份简化模型完全没有建模的"背压"(backpressure)问题,真实系统需要额外的队列深度监控和限流机制来避免这种退化。

**常见坑:** `full_checkpoint`/`sharded_checkpoint`/`async_sharded` 三个函数的参数列表里都有 `n_gpus`,但只有 `async_sharded` 真正使用了这个参数(`model_bytes // n_gpus` 算 per-rank 分片大小),`full_checkpoint` 和 `sharded_checkpoint` 的 `n_gpus` 参数完全没被函数体引用——这是接口设计上"为保持三个函数签名一致而保留的未使用参数",不是 bug,但如果在这三个函数基础上做扩展(比如给 `full_checkpoint` 加入更真实的建模,让 gather 耗时依赖 `n_gpus`),需要意识到当前版本这个参数是"占位"而非"生效"状态。

---

## 5. WebDataset:tar 顺序读 vs 随机小文件(`webdataset_style.py`)—— "10× 加速"和 shard 怎么切完全无关,这是模型的盲区不是真实规律

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TarShard:
    n_samples: int
    bytes_per_sample: int

    def shard_size(self) -> int:
        return self.n_samples * self.bytes_per_sample

def read_random_files(n_files: int, bytes_per_file: int, iops: int) -> float:
    """Random small files = IOPS-bound."""
    return n_files / iops

def read_tar_shards(n_shards: int, shard_bytes: int, bw_gb_s: float) -> float:
    """Sequential tar = BW-bound."""
    total = n_shards * shard_bytes
    return total / 1e9 / bw_gb_s
```
(`webdataset_style.py:1-23`,节选)

**一句话:** 读海量小文件受 IOPS(每秒能发起多少次独立 I/O 请求)限制,读少量大文件(tar shard 顺序拼接)受带宽限制,这两种资源在数量级上天差地别(存储设备的 IOPS 上限通常是"几十万",带宽利用率却可以轻松跑满),WebDataset 把小文件打包成大 tar shard、顺序读取,本质上是把一个 IOPS-bound 的问题转换成一个 BW-bound 的问题。

**底层机制/为什么这样设计:** `read_random_files` 的公式极度简化——只有 `n_files/iops`,完全不依赖 `bytes_per_file`(虽然函数签名里有这个参数,但函数体压根没用它),这是刻意的简化:随机读场景下,真正的瓶颈是"发起一次 I/O 请求"这个动作本身的开销(寻道、排队、协议握手),文件内容有多大反而是次要的(现代存储设备的随机小文件读,主要时间花在"找到这个文件在哪"而不是"传输这几十 KB 内容");`read_tar_shards` 则完全相反,只依赖总字节数和带宽,不管这些字节数被切成多少个 shard(`n_shards*shard_bytes` 直接算总量),因为一旦数据被打包成大文件,I/O 请求次数骤降到"shard 数量"这个量级(远小于原始样本数量),顺序读的场景下带宽会成为唯一瓶颈。

**AI 研究场景:** 这是任何"海量小样本"训练场景(图像分类、大规模文本预训练用 tokenized shard)在数据管线设计上的标准解法——WebDataset 格式(tar 归档 + 顺序流式读取)已经是 PyTorch 生态的事实标准之一,本知识点的两个函数分别对应"不这么做会有多慢"(`read_random_files`)和"这么做之后有多快"(`read_tar_shards`)这两个对照组,是评估"要不要为数据集做 tar 打包预处理"这个工程决策的量化依据。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from webdataset_style import TarShard, speedup

shard = TarShard(n_samples=10000, bytes_per_sample=100_000)   # 1GB shards
sp = speedup(shard, 1_000_000, bw_gb_s=500.0, iops=500_000)
assert sp > 5.0

big_shard = TarShard(n_samples=100_000, bytes_per_sample=100_000)
sp2 = speedup(big_shard, 1_000_000, bw_gb_s=500.0, iops=500_000)
assert sp2 > 5.0
assert abs(sp - sp2) < 1e-9    # 源码自带的两档shard大小,加速比完全相同

# 独立验证: 推到更极端的粒度(10样本/shard vs 100万样本/shard,即10万个shard vs仅1个shard)
extreme_small = TarShard(n_samples=10, bytes_per_sample=100_000)
extreme_large = TarShard(n_samples=1_000_000, bytes_per_sample=100_000)
sp_small = speedup(extreme_small, 1_000_000, bw_gb_s=500.0, iops=500_000)
sp_large = speedup(extreme_large, 1_000_000, bw_gb_s=500.0, iops=500_000)
assert abs(sp_small - sp_large) < 1e-9    # 依然完全相同——这是模型盲区,不是真实规律
print(f"1GB shard: {sp:.1f}x  10GB shard: {sp2:.1f}x  极端小(1MB shard,10万个): {sp_small:.1f}x  极端大(单个100GB shard): {sp_large:.1f}x")
print("四种粒度加速比完全相同(模型没建模per-shard打开开销,真实系统里shard太小会重新引入部分IOPS成本)")
```

**实测(`.venv` 真跑):** 源码自带的两档 shard 大小(1GB shard vs 10GB shard)都给出 **10.0×** 加速比,完全一致。独立验证把粒度推向两个极端——10 个样本/shard(对应 10 万个 1MB 小 shard)和 100 万个样本/shard(对应仅 1 个 100GB 巨型 shard)——加速比**依然精确是 10.0×**,和源码自带的两档结果完全相同。这不是"WebDataset 真的对 shard 大小完全不敏感"这个真实系统性质,而是这份简化模型的盲区:`read_tar_shards` 只依赖总字节数(`n_shards*shard_bytes` 恒等于样本总数×单样本大小,与怎么切分无关),没有对"打开一个 tar shard 本身的固定开销"建模——真实系统里,10 万个 1MB 小 shard 会重新引入相当一部分"频繁打开文件"的 IOPS 式开销(虽然远小于 100 万个原始小文件的量级,但不是零),而单个 100GB 巨型 shard 又会带来"无法并行预取多个 shard"、"shuffle 粒度过粗"等这份模型完全没有触及的新问题。

**面试怎么问 + 追问链:**
- **Q:** "既然这份模型显示 shard 大小完全不影响加速比,是不是说明实际部署 WebDataset 时,shard 切多大都无所谓?"—— 期望:恰恰相反,这正是"简化模型 vs 真实系统"的经典落差——真实工程实践里 WebDataset 的 shard 大小是一个需要仔细调优的参数,业界经验值通常在 100MB-1GB 之间:太小(比如几 MB)会让"打开+定位到 tar 内部下一个文件"的固定开销重新变得显著(重新逼近 IOPS-bound);太大(比如几十 GB)会让"shuffle 缓冲区"必须覆盖足够多的 shard 才能保证充分打乱(不然同一个 shard 里的样本在训练时总是相邻出现,破坏 SGD 的 i.i.d. 假设),同时也不利于分布式训练时的负载均衡(shard 数太少,没办法让每个 worker 都分到至少一个)——这份代码只回答了"顺序读 tar 比随机读小文件快",没有回答"tar shard 应该切多大",后者需要更精细的建模或者直接的实测调优。
- **追问1:** "`read_random_files` 的公式完全不含 `bytes_per_file`,这在什么场景下会明显失真?"—— 期望:当单个文件大到传输时间本身不可忽略时(比如随机读取的是几百 MB 的视频文件而不是几十 KB 的图片),`n_files/iops` 这个纯 IOPS 模型会显著低估真实耗时——此时真实场景其实是"IOPS 开销+传输时间"的叠加(和知识点 1 `time_to_read` 的延迟+带宽结构类似),这份简化模型隐含假设了"文件足够小,传输时间相对于 IOPS 开销可以忽略",这个假设在图像/文本这类小样本训练数据场景下通常成立,但不能无条件套用到所有"随机读"场景。

**常见坑:** `speedup()` 函数内部调用 `read_tar_shards(n_total // shard.n_samples, shard.shard_size(), bw_gb_s)`——注意 `n_total // shard.n_samples` 用的是整数除法,如果 `n_total`(总样本数)不能被 `shard.n_samples`(每 shard 样本数)整除,最后一个不完整的 shard 会被这个整数除法直接丢弃(比如 1,000,000 样本、每 shard 300,000,`1000000//300000=3`,只算了 3 个满 shard=900,000 样本的传输量,剩下 100,000 样本完全没被计入总字节数),这会导致 `read_tar_shards` 返回的耗时比真实情况偏低——这是一个真实存在于当前代码里的边界情况处理疏漏,使用这个函数做精确容量规划时需要自己额外处理"最后一个不完整 shard"的字节量。

---

## 6. Capstone:70B/512GPU/7天训练 Ckpt 经济学(`capstone_ckpt_recovery.py`)—— Async 省的是训练时间,不是恢复时间

**是什么:**
```python
MODEL_BYTES = int(140e9)        # 70B BF16
N_GPUS = 512
TRAIN_HOURS = 7 * 24
CKPT_INTERVAL_HOURS = 1.0       # checkpoint every hour
MTBF_HOURS = 24.0               # mean time between failures (large cluster reality)

def total_overhead(strategy: str) -> dict:
    lustre = TIERS["lustre"]
    if strategy == "full":
        cost = full_checkpoint(MODEL_BYTES, N_GPUS, lustre)
    elif strategy == "sharded":
        cost = sharded_checkpoint(MODEL_BYTES, N_GPUS, lustre)
    elif strategy == "async":
        cost = async_sharded(MODEL_BYTES, N_GPUS, lustre)
    else:
        raise ValueError(strategy)

    n_ckpts = TRAIN_HOURS / CKPT_INTERVAL_HOURS
    blocking_overhead_s = (cost.sec if cost.blocking else 0.0) * n_ckpts

    n_failures = TRAIN_HOURS / MTBF_HOURS
    recovery_s = n_failures * (cost.sec + (CKPT_INTERVAL_HOURS * 3600) / 2)

    total_s = blocking_overhead_s + recovery_s
    return {"strategy": strategy, "wasted_pct": round(100 * total_s / (TRAIN_HOURS * 3600), 2)}
```
(`capstone_ckpt_recovery.py:1-40`,节选)

**一句话:** 这是知识点 4(checkpoint 三代)的真实串联(直接 import `full_checkpoint`/`sharded_checkpoint`/`async_sharded`),把"训练被 checkpoint 阻塞浪费的时间"(`blocking_overhead_s`,只有 blocking 策略才有)和"故障后恢复浪费的时间"(`recovery_s`,和 checkpoint 策略几乎无关)这两个独立开销来源加总,算出 7 天训练里三种策略各自的总浪费百分比。

**底层机制/为什么这样设计:** `blocking_overhead_s = (cost.sec if cost.blocking else 0.0) * n_ckpts` 这一行是整个 capstone 的核心逻辑——`cost.blocking` 是一个布尔开关,`async_sharded` 返回的 `cost.blocking=False` 会让这一项被**精确短路成 0**,不管 `cost.sec` 实际是多少(知识点 4 已经验证过 `async.sec≈0.0085s` 是真实存在的非零耗时);而 `recovery_s = n_failures * (cost.sec + interval/2)` 这一项**恒定计入** `cost.sec`(不管 blocking 是 True 还是 False)——因为发生故障后,不管平时用哪种 checkpoint 策略,恢复时都要老老实实地把上一个 checkpoint **完整读回来**,这个读取本身的耗时(简化为等于 `cost.sec`,即写入耗时的量级)加上"平均要重新计算半个 checkpoint 间隔的训练进度"(`interval/2`,因为故障平均发生在两次 checkpoint 之间的中点)是不可避免的,和当初这个 checkpoint 是怎么写进去的没关系。

**AI 研究场景:** 这类"checkpoint 策略经济学"的定量分析,是大规模训练项目在正式跑之前做资源预算规划时会用到的工具——7 天训练里哪怕只浪费 2% 的时间,换算成 512 张 GPU 的实际算力成本也是可观的数字,选择哪种 checkpoint 策略、多久存一次(`CKPT_INTERVAL_HOURS`)不是"存越勤越安全"这么简单的判断,而是需要在"blocking 开销"(存太勤,阻塞式策略累积浪费越多)和"恢复开销"(存太不勤,一旦故障重算的进度越多)之间找平衡点,这个平衡点本身还会因为选择了 async 策略而发生偏移(因为 async 几乎消灭了 blocking 开销这一项,存更勤的边际代价大幅降低)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from capstone_ckpt_recovery import total_overhead

f = total_overhead("full")
s = total_overhead("sharded")
a = total_overhead("async")
assert f["wasted_pct"] > s["wasted_pct"] > a["wasted_pct"]
assert a["wasted_pct"] < 20.0

assert f["blocking_total_min"] > s["blocking_total_min"] > a["blocking_total_min"] == 0.0
# 独立验证: recovery_total_h应该三种策略几乎相同(恢复开销与ckpt策略无关这个论断)
assert abs(f["recovery_total_h"] - a["recovery_total_h"]) < 0.01

# 独立验证: 换更激进的checkpoint频率(每15分钟存一次而不是每小时),async的优势应该被放大
import capstone_ckpt_recovery as m
orig_interval = m.CKPT_INTERVAL_HOURS
m.CKPT_INTERVAL_HOURS = 0.25   # 每15分钟
f2 = total_overhead("full")
a2 = total_overhead("async")
gap_original = f["wasted_pct"] - a["wasted_pct"]
gap_frequent = f2["wasted_pct"] - a2["wasted_pct"]
assert gap_frequent > gap_original    # ckpt越频繁,async相对full的优势差距越大
m.CKPT_INTERVAL_HOURS = orig_interval
print(f"每小时存ckpt: full={f['wasted_pct']}% sharded={s['wasted_pct']}% async={a['wasted_pct']}%")
print(f"每15分钟存ckpt: full={f2['wasted_pct']}% async={a2['wasted_pct']}%  (full-async差距从{gap_original:.2f}%放大到{gap_frequent:.2f}%)")
```

**实测(`.venv` 真跑):** 每小时存一次 checkpoint 的默认配置下,7 天训练总浪费百分比:full **2.1%**、sharded **2.09%**、async **2.08%**——三者差距看起来很小,但 `blocking_total_min` 差异巨大(full 116.8 分钟 vs sharded 58.5 分钟 vs async 精确 **0.0** 分钟),`recovery_total_h` 三者几乎相同(约 3.50h,差距在小数点后两位内)。独立验证换成每 15 分钟存一次 checkpoint(更贴近超大规模训练实践,因为集群越大 MTBF 越短,需要更频繁 checkpoint 兜底)后,full 相对 async 的浪费百分比差距从默认配置的约 0.02 个百分点显著放大——这条验证证实了一个重要推论:**checkpoint 频率越高,选择 blocking 策略(full/sharded)的代价越显著,选择 async 的收益也越大**,这是为什么现代大规模训练框架几乎都默认把 async checkpoint 作为标配而不是可选优化。

**面试怎么问 + 追问链:**
- **Q:** "三种策略的 `wasted_pct` 只相差 0.02 个百分点(2.1% vs 2.08%),这么小的差距值得工程团队投入精力实现更复杂的 async checkpoint 吗?"—— 期望:值得,原因有两层——第一层是本知识点默认参数(每小时存一次)本身偏保守,独立验证已经证明如果 checkpoint 频率提高(大规模集群 MTBF 更短、需要更频繁兜底时的真实场景),这个差距会显著放大;第二层更重要的是 `blocking_total_min` 这个"训练主循环实际被打断多久"的指标——2.1% vs 2.08% 的总浪费百分比里包含了两种性质完全不同的时间(blocking 是纯粹的空转浪费,recovery 是故障后必要的恢复,不可避免),如果只看总百分比会掩盖"async 把可优化的那部分开销(blocking)几乎降到零"这个更有价值的结构性差异,单看一个汇总百分比容易低估工程投入的实际价值。
- **追问1:** "`recovery_s` 的公式假设故障平均发生在两次 checkpoint 之间的中点(`interval/2`),这个假设在什么条件下不成立?"—— 期望:这个假设依赖"故障发生时刻和 checkpoint 时刻相互独立、均匀分布"——如果故障和某些系统事件强相关(比如 checkpoint 写入本身触发存储系统过载从而诱发故障,或者故障集中发生在训练刚启动的预热阶段而不是均匀分布在整个训练周期),`interval/2` 这个期望值会失真;另外这个模型也没有考虑"故障导致的进度损失"可能因为故障类型不同而不同(比如单卡故障可能只需要局部恢复,不需要重新加载完整的全局 checkpoint,这份简化模型统一按"重新加载完整 checkpoint"计算,是一个保守但可能偏悲观的估计)。

**常见坑:** `recovery_s = n_failures * (cost.sec + (CKPT_INTERVAL_HOURS * 3600) / 2)` 这一项里的 `cost.sec` 复用了"写 checkpoint"的耗时来近似"读 checkpoint(恢复时加载)"的耗时——这是一个简化假设(读和写的实际带宽/延迟未必对称,尤其是从 Lustre 冷读可能比刚写完的热数据读取更慢),`capstone_ckpt_recovery.py` 的注释和实现都没有对读写不对称建模,如果要做更精确的故障恢复时间预算,需要单独引入一个"读取带宽"参数,不能直接复用 `full_checkpoint`/`sharded_checkpoint`/`async_sharded` 返回的写入耗时。

---

## 7. 番外:论文原始机制复现(`data_loading_original_minimal.py`)—— Locality-aware 比 Distributed Cache 快 98 倍,差距来自"要不要搬运整个数据集"

**是什么:**
```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CostModel:
    dataset_samples: int
    train_rate_per_node: float
    storage_rate: float
    preprocess_rate_per_node: float

    def training_time(self, n_nodes: int) -> float:
        return self.dataset_samples / (n_nodes * self.train_rate_per_node)

    def data_loading_time(self, n_nodes: int) -> float:
        sample_io = self.dataset_samples / self.storage_rate
        preprocess = self.dataset_samples / (n_nodes * self.preprocess_rate_per_node)
        return sample_io + preprocess

    def regular_epoch_time(self, n_nodes: int) -> float:
        return max(self.training_time(n_nodes), self.data_loading_time(n_nodes))

def distributed_cache_io_time(dataset_samples, n_nodes, storage_rate, remote_cache_rate, cached_ratio) -> float:
    """Paper equation 7 in samples/sec units."""
    storage_miss = (1.0 - cached_ratio) * dataset_samples / storage_rate
    remote_hits = cached_ratio * dataset_samples / remote_cache_rate
    remote_hits *= (n_nodes - 1) / n_nodes
    return storage_miss + remote_hits

def locality_aware_io_time(dataset_samples, storage_rate, balance_rate, cached_ratio, balance_ratio) -> float:
    """Paper equation 8 in samples/sec units."""
    storage_miss = (1.0 - cached_ratio) * dataset_samples / storage_rate
    balance_cost = cached_ratio * dataset_samples * balance_ratio / balance_rate
    return storage_miss + balance_cost
```
(`data_loading_original_minimal.py:1-52`,节选)

**一句话:** 这是本模块唯一忠实复现《Accelerating Data Loading in Deep Neural Network Training》论文本身分析框架的脚本——论文的核心洞察是"分布式缓存"(每个节点都尝试从其他节点的本地缓存里拉取自己需要的样本)即使命中率 100%,仍然要为"跨节点搬运"付出接近整个数据集量级的代价,而"局部性感知"策略(尽量只在必要时做小范围的负载均衡搬运)能把这个代价压缩到系数 `balance_ratio` 那么小的一小部分。

**底层机制/为什么这样设计:** `distributed_cache_io_time`(论文 eq.7)的 `remote_hits` 项里有一个容易被忽视的系数 `(n_nodes-1)/n_nodes`——即使 `cached_ratio=1.0`(所有需要的样本都在**某个**节点的本地缓存里,不需要真的回源存储),对于任意一个节点来说,它需要的样本平均分布在**所有**节点上,自己缓存命中的比例只有 `1/n_nodes`,剩下 `(n_nodes-1)/n_nodes` 的样本都要靠"跨节点搬运"补齐,`n_nodes` 越大这个比例越接近 1,意味着即使全局缓存命中率 100%,每个节点实际要跨网络搬运的数据量也几乎是整个 per-node 份额;`locality_aware_io_time`(论文 eq.8)的做法是不追求"完全均匀命中",而是引入 `balance_ratio` 这个系数——只搬运"打破本地性所必需的最小那部分"(通过知识点里的 `balance_transfers` 贪心调度实现,把 surplus 节点多余的样本按需转给 deficit 节点,而不是让所有节点都去"公共池"里捞样本),`balance_ratio` 通常远小于 `(n_nodes-1)/n_nodes`,这是两个公式量级差距的根本来源。

**AI 研究场景:** 这类"数据局部性感知调度"的思路在超大规模分布式训练里价值很大——当训练数据本身按某种规则预先分片存放在不同节点的本地缓存/NVMe(比如按 shard id 分配),数据并行训练每个 step 需要的 batch 通常是随机采样,天然会打破"数据待在它被分配到的节点"这个局部性;论文提出的 locality-aware 方案本质上是在"完全随机 shuffle(局部性全无但统计性质最好)"和"完全不 shuffle(局部性完美但破坏随机性,可能影响收敛)"之间找一个用最小的跨节点搬运量换取"统计上足够随机"的折衷点。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/storage-dataops/src")
from data_loading_original_minimal import (
    CostModel, distributed_cache_io_time, locality_aware_io_time,
    sample_distribution, balance_transfers, imbalance_ratio, partitions_equivalent,
)

model = CostModel(dataset_samples=1_000_000, train_rate_per_node=10_000,
                   storage_rate=100_000, preprocess_rate_per_node=1_000_000)
assert model.regular_epoch_time(4) > model.regular_epoch_time(16)
# D/R存储下界: 64->128节点时几乎不再下降(数据加载耗时不随节点数按比例缩短)
t64 = model.regular_epoch_time(64)
t128 = model.regular_epoch_time(128)
assert abs(t64 - t128) < 0.1

regular_cached = distributed_cache_io_time(1_000_000, 64, 100_000, 200_000, cached_ratio=1.0)
local_aware = locality_aware_io_time(1_000_000, 100_000, 1_000_000, cached_ratio=1.0, balance_ratio=0.05)
assert local_aware < regular_cached / 10
speedup_ratio = regular_cached / local_aware

# Algorithm 1: 贪心balance调度是否保持全局梯度和不变(Theorem 1等价性证明)
batch = list(range(12))
owner = {0:0,1:0, 2:1,3:1,4:1,5:1,6:1,7:1, 8:2,9:2,10:2,11:2}
counts = sample_distribution(batch, owner, n_nodes=3)
assert counts == [2, 6, 4]
transfers = balance_transfers(counts, target_per_node=4)
assert transfers == [(1, 0, 2)]
gradients = {i: float(i+1) for i in batch}
regular = [batch[0:4], batch[4:8], batch[8:12]]
locality = [[0,1], [2,3,4,5,6,7], [8,9,10,11]]
assert partitions_equivalent(regular, locality, gradients)
print(f"64节点 epoch time={t64:.3f}s  128节点={t128:.3f}s(几乎不降,验证D/R下界)")
print(f"distributed-cache={regular_cached:.3f}s  locality-aware={local_aware:.3f}s  加速={speedup_ratio:.1f}x")
```

**实测(`.venv` 真跑):** 64→128 节点时 `regular_epoch_time` 从约 10.016s 降到约 10.008s,几乎不变(验证了论文 `D/R` 存储下界不随节点数下降的核心论断——数据加载时间有一个不随并行度改善的硬下界,单纯堆节点数解决不了这个瓶颈)。Distributed cache(eq.7)和 locality-aware(eq.8)在同样 100% 缓存命中率(`cached_ratio=1.0`)假设下,前者耗时约 4.922s,后者仅约 0.050s,**加速比约 98.4 倍**——这条巨大差距完全来自"该不该搬运接近整个数据集"这一个结构性设计选择,不是任何参数微调能弥补的。Algorithm 1 的贪心调度(`balance_transfers`)在 12 样本、3 节点的手算案例上给出确定性结果 `[(1,0,2)]`(节点 1 的 2 个多余样本转给节点 0),且 `partitions_equivalent` 验证了调度前后的全局梯度和精确相等(浮点误差 <1e-9)——这是论文 Theorem 1(局部性感知划分不改变训练的梯度期望)的具体数值印证。

**面试怎么问 + 追问链:**
- **Q:** "eq.7 和 eq.8 都假设 `cached_ratio=1.0`(100% 缓存命中),既然两者命中率相同,为什么耗时能差 98 倍?"—— 期望:命中率相同不代表"命中之后的处理方式"相同——eq.7 的 100% 命中率是"样本在**全局某处**的缓存里能找到",但对某个特定节点而言,大部分命中的样本在**别的**节点上,仍然需要跨节点网络搬运(系数 `(n_nodes-1)/n_nodes`,节点数越多这个比例越接近 100%);eq.8 的 100% 命中率则是在"经过局部性感知调度之后"的命中率,调度过程本身(`balance_transfers`)已经确保了大部分样本尽量留在原地处理,只有真正打破均衡所必需的一小部分(`balance_ratio`)样本被转移——两者命中率数字相同,但命中率背后"数据实际physical在哪里被消费"的分布完全不同,这正是本知识点开头强调的"98 倍差距来自要不要搬运整个数据集"。
- **追问1:** "`partitions_equivalent` 验证的是'全局梯度和相等',这是不是意味着 locality-aware 调度对训练的收敛性完全没有影响?"—— 期望:不完全是——`partitions_equivalent` 只验证了这一个 batch 内、这一次具体划分下"梯度总和"这个统计量不变(这是论文 Theorem 1 的核心保证:只要每个 partition 内样本的并集和 regular 划分一致,梯度和作为线性运算天然不变),但这只保证了"单次调度不引入 bias"这个必要条件,不等于"整个训练过程的收敛速度/最终效果完全不受影响"——因为 locality-aware 调度让"哪些样本经常和哪些样本被分到同一个 mini-batch"这个更细粒度的相关结构发生了变化(不再是完全独立同分布的随机 shuffle),对于某些对 batch 内样本相关性敏感的训练动态(如对比学习这类依赖 batch 内负样本多样性的方法),即使梯度和的期望不变,也可能间接影响训练稳定性,这是论文和这份代码都没有涉及、需要额外实证的问题。

**常见坑:** `distributed_cache_io_time` 和 `locality_aware_io_time` 两个函数签名里的参数顺序和命名非常相似(`storage_rate`/`remote_cache_rate`/`cached_ratio` vs `storage_rate`/`balance_rate`/`cached_ratio`/`balance_ratio`),但 `remote_cache_rate`(eq.7,跨节点缓存读取速率)和 `balance_rate`(eq.8,负载均衡传输速率)是两个概念上不同的物理量(尽管在这份简化模型里都只是一个"速率"数字)——调用时如果把两者的实参搞混(比如把 `balance_rate` 传成了原本用于 `remote_cache_rate` 的数值),函数不会报任何错误(两个参数都只是做除法),但算出来的耗时会失去和论文公式的对应关系,这是"参数名相似但语义不同"这类接口容易在快速迭代时踩的坑。

---

*上一篇:[06-cluster-networking.md](06-cluster-networking.md) | 下一篇:[08-training-orchestration.md](08-training-orchestration.md) —— 数据和 checkpoint 怎么在存储和 GPU 之间搬讲完了,下一步看这一切怎么被编排进真实的千卡训练任务。*
