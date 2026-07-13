# 01 · Inference Engine Core 深挖(vLLM 骨架复刻)

> 总览见 [00-roadmap.md](00-roadmap.md)

LLM 推理服务和训练是两个特性完全不同的问题:训练看的是几千 token 一起算的吞吐,推理(尤其是 decode 阶段)看的是一个 token 一个 token 吐、且大部分时间在等显存带宽而不是在算力上打满。本文是 `inference-serving-deep-dive` 系列第 1 篇,对应 `learning/inference-engine-core/`(Module 5《用大模型》第 1 专题,13 lectures + 11 个 src 源文件),从"一个最朴素的推理服务长什么样、哪里会烂"讲起,一路推到"vLLM 到底是靠哪几件东西撑起 24 倍吞吐提升"。11 个知识点大致是:1(全图/prefill-decode 两阶段特性)→2(朴素 KV cache 为什么碎片化)→3(PagedAttention 怎么用 OS 虚拟内存的思路解决碎片)→4(分页在 kernel 层怎么落地,这个仓库这一层是个诚实标注的 stub)→5(Continuous Batching 怎么消灭"静态批"的空转)→6(Chunked Prefill 怎么消灭"长 prompt 阻塞 decode")→7(Prefix Caching,给 02 号文件 RadixAttention 埋伏笔)→8(调度策略:FCFS/SJF/priority)→9(CUDA Graphs + Attention Backends,概念性)→10(Sampling 引擎)→11(Capstone:~175 行拼出 mini-vLLM)。

**和 00-roadmap.md 差异化声明的关系:** 00-roadmap.md 已经交代过,`learning/` 侧 7 个源模块合计约 60 篇 lecture、7000+ 行代码,本系列不是逐 lecture 罗列,而是按"一个可独立验证的机制"为颗粒度合并——具体到本文,L09(CUDA Graphs)和 L10(Attention Backends)被合并成知识点 9(两者共享"教学 stub、这台机器上到底能不能跑出真实数字"这同一条诚实标注,分开写会有大量重复);其余 9 个 lecture 大致一 lecture 对一知识点,加上 L12+L13 合并成 capstone 知识点 11,总共 11 个知识点,和 roadmap 给出的"10-11"量级吻合。

**和 02 号文件(sglang-radixattention)的关系:** 知识点 7(Prefix Caching)讲的是**平面 hash** 版本的前缀复用——`prefix_cache.py` 把 prompt 切成定长块,每块算一个 hash,精确匹配才能命中;02 号文件会讲 SGLang 的 RadixAttention,用 **radix tree** 取代平面 hash,连"部分共享前缀"也能利用(比如两个请求只有前几个 token 完全一样、后面分叉)。这是本文知识点 7 提前埋下的伏笔,阅读顺序上不要求先读 02 号文件,但本文会在知识点 7 里明确指出这条未来要展开的分界线。

**和 kernel-gpu-deep-dive/03 的关系:** 知识点 4(PagedAttention 的 kernel 层)只讲"分页寻址在 kernel 里怎么落地"这一件事,不重复讲解 Triton 语言基础(`@triton.jit`/`tl.load`/`tl.program_id` 这些)本身——那部分已经在 [kernel-gpu-deep-dive/03-kernel-design-triton-cutlass.md](../kernel-gpu-deep-dive/03-kernel-design-triton-cutlass.md) 知识点 1-2(Triton autotune 的 SMEM 预算与打分公式)讲过,对 Triton 完全陌生的读者建议先读那一篇。

**一个重要的诚实标注:** 本文涉及的 11 个源文件,`requirements.txt` 里列出的 vllm/flash-attn/triton 在 `src/*.py` 里几乎不被真实 import——这不是本文的疏漏,是源材料自己的设计:11 个源文件全部是纯 Python/torch 手写复现,真正需要重型依赖的部分(`vllm_compare.py`)会在缺依赖时**显式 fail-fast**(`SystemExit`),不是静默跳过冒充成功(本文知识点 11 会给出实测的真实报错和退出码,并指出它和自己 docstring 承诺的退出码并不一致)。知识点 4 和知识点 9 在撰写过程中各自发现了一处"源码自身文档/常识预设"和"真实运行结果"不一致的地方(分别是"Triton kernel 从未被真正调用"和"CUDA Graph 在这台 Windows 机器上其实能跑、还能跑出真实两位数倍速比"),下文会用真实运行结果说清楚,不是凭印象转述。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13.9,torch 2.11.0+cu128,CUDA 可用,`torch.cuda.get_device_name(0)` 实测为 `NVIDIA GeForce RTX 3080 Ti Laptop GPU`)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出,不是手算或转述。这台机器上 `triton` 实际已安装(`3.7.1`),但 `vllm`/`flash_attn`/`flashinfer` 均未安装(import 立即 `ModuleNotFoundError`,已实测确认)——和 00-roadmap.md 的环境声明一致。11 个知识点的"可运行例子"全部纯 CPU 或纯 GPU-tensor-op(不需要真实模型权重),只有知识点 9 的 CUDA Graph 演示需要 CUDA 设备,这台机器满足。知识点 11 capstone 的"真 vllm 对照"选做部分本次不做(WSL2+vllm 环境尚未就绪),只做 mini-vLLM 复刻本身。

---

## 1. LLM 推理服务全图(`common.py` + L01)—— prefill 计算密集、decode 访存密集,三个浪费对三个现代解法

**是什么:**
```python
@dataclass
class Request:
    """One inference request, identified by integer id."""
    rid: int
    prompt_ids: List[int]
    max_new_tokens: int
    output_ids: List[int] = field(default_factory=list)
    finished: bool = False
    cache_ref: object = None
    t_admit: float = 0.0
    t_first_token: float = 0.0
    t_finish: float = 0.0

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)
```
(`common.py:15-31`)

```python
@dataclass
class EngineMetrics:
    n_requests: int = 0
    n_tokens_in: int = 0
    n_tokens_out: int = 0
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def throughput_out(self) -> float:
        return self.n_tokens_out / self.elapsed
```
(`common.py:35-48`,节选)

**一句话:** LLM 推理服务的核心矛盾是 prefill(一次性算完 prompt 全部 K/V,compute-bound)和 decode(每步只算 1 个新 token、但要读全部历史 KV,memory-bound)这两个阶段特性完全相反,`Request`/`EngineMetrics` 是全系列 11 个源文件共享的"通用语言"——`t_admit`/`t_first_token`/`t_finish` 三个时间戳直接对应 TTFT(time-to-first-token)和 ITL(inter-token latency)两个延迟指标,`total_len` 是后面 PagedAttention/调度算法反复要用到的"这个请求现在占多少 KV"的单一事实来源。

**底层机制/为什么这样设计:** 从最笨的想法讲起——如果推理服务只是"调用一次模型的 forward",prefill 和 decode 的区别根本不需要被建模:每次都是一次完整的 forward,输入变长就是了。问题是真实场景里一次 prefill 之后往往跟着几十到几百次 decode,如果每次 decode 都把整个 prompt+已生成 token 重新过一遍模型,复杂度是 O(生成长度²) 而不是 O(生成长度)——KV cache 的意义就是把"已经算过的 K、V"缓存下来,decode 时只对**新来的这一个 token** 算 Q、K、V,再拿这个新 Q 去和缓存的全部历史 K、V 做 attention。这解释了 L01 说的"decode 阶段 1 token/step,memory-bound":decode 每一步计算量很小(只处理 1 个新 token),但要把可能几千甚至上万个历史 token 的 K、V 从显存搬到计算单元,搬运数据的时间远大于计算时间,是访存瓶颈而不是算力瓶颈;prefill 反过来,一次性算完 prompt 全部几十到几千个 token 的 K、V,矩阵乘法维度大、并行度高,是算力瓶颈。`Request.total_len` 把"prompt 已有多长 + 已生成多少"合并成一个数字,这正是知识点 3 PagedAttention 计算"这个请求需要多少物理块"的输入。`EngineMetrics` 的吞吐相关字段对应 L01 强调的"三大指标"里的吞吐(throughput);延迟(TTFT+ITL)是 `Request` 自己三个时间戳的差值,`t_first_token - t_admit` 就是 TTFT,后续每个 token 之间的间隔就是 ITL——第三个指标 goodput(满足 SLO 的吞吐)这套源码没有单独实现,是在吞吐和延迟两个已有指标之上叠加一个"是否满足业务约束"的过滤条件,概念上不需要新的数据结构。L01 提出的三个浪费(静态 batch 空转、KV 碎片、prefill 阻塞 decode)分别对应三个现代解法(Continuous Batching、PagedAttention、Chunked Prefill),是知识点 5、3、6 的主题,这里只给全景地图,不展开实现。

**AI 研究场景:** 这套 prefill/decode 两阶段划分不是纯学术区分——它直接决定了推理服务的硬件选型和成本模型:prefill 密集的场景(比如长文档摘要、RAG 长 context)更吃算力,decode 密集的场景(比如长对话、长链推理)更吃显存带宽和显存容量。知识点 5(distributed-inference 系列会展开)的 Disaggregated Prefill/Decode 架构,本质就是发现"两阶段资源需求形状不同",干脆把它们物理拆到不同机器上分别优化,而不是绑在同一张卡上互相拖累——这是本文一开始就该建立的直觉,后面所有优化(分页、连续批处理、chunked prefill)都是在"两阶段特性不同"这个前提下才成立的。

**可运行例子:**
```python
import sys
import time
sys.path.insert(0, "learning/inference-engine-core/src")
from common import Request, EngineMetrics, now, stub_token_step
from continuous_batching import Engine, make_request

# --- KV cache size formula (L02 explicit config: n_kv_heads=8, head_dim=128, fp16) ---
def kv_bytes_per_token_per_layer(n_kv_heads, head_dim, dtype_bytes):
    return 2 * n_kv_heads * head_dim * dtype_bytes  # 2 = K and V

per_layer = kv_bytes_per_token_per_layer(n_kv_heads=8, head_dim=128, dtype_bytes=2)
assert per_layer == 4096  # 4 KB, matches L02 "1 token KV / layer = 2*8*128*2 = 4KB"
n_layers = 32
per_token_all_layers = per_layer * n_layers
assert per_token_all_layers == 131_072  # 128 KB / token, matches L02
max_len = 2048
per_request_reserved = per_token_all_layers * max_len
assert per_request_reserved == 268_435_456  # 256 MB/request, matches L02
gpu_24gb = 24 * 1024**3
max_concurrent = gpu_24gb // per_request_reserved
assert max_concurrent == 96  # L02 says "~80"，量级一致（L02 的 ~80 大概率多算了其他显存开销）

# --- prefill vs decode: structural asymmetry using the real Request/Engine primitives ---
prompt_len = 100
max_new = 20
req = Request(rid=0, prompt_ids=list(range(prompt_len)), max_new_tokens=max_new)
assert req.total_len == prompt_len  # before any decode step, total_len == prompt only

n_decode_iters = 0
for _ in range(max_new):
    tok = stub_token_step(req)
    req.output_ids.append(tok)
    n_decode_iters += 1
assert n_decode_iters == max_new == 20
assert req.total_len == prompt_len + max_new == 120
prefill_tokens_per_iter = prompt_len / 1
decode_tokens_per_iter = max_new / n_decode_iters
assert prefill_tokens_per_iter == 100.0
assert decode_tokens_per_iter == 1.0
assert prefill_tokens_per_iter / decode_tokens_per_iter == 100.0  # 100x 单次迭代并行度差距

# --- three metrics: throughput / TTFT(latency) / goodput, from EngineMetrics ---
m = EngineMetrics(n_requests=1, n_tokens_in=prompt_len, n_tokens_out=max_new)
m.t_start = 0.0
m.t_end = 2.0
assert m.elapsed == 2.0
assert m.throughput_out == 10.0  # 20 output tokens / 2s
assert m.throughput_total == 60.0  # (100 in + 20 out) / 2s

# --- TTFT 是 Request 上真实被写入的字段，不只是 lecture 概念 ---
eng = Engine(max_running=1, kv_budget=1000, eos_id=-1)
r = make_request(0, prompt_len=5, max_new=3)
eng.add_request(r)
assert r.t_admit > 0.0 and r.t_first_token == 0.0  # 已入队，但还没产生第一个 token
time.sleep(0.02)
eng.step()  # admission + 第一次 decode 迭代在同一个 step() 里发生
assert r.t_first_token > r.t_admit
ttft = r.t_first_token - r.t_admit
assert ttft >= 0.02
while r.output_ids and len(r.output_ids) < 3:
    eng.step()
assert r.t_finish >= r.t_first_token
```

**实测(`.venv` 真跑):** KV cache 公式核实为真:`per_layer=4096` 字节(4KB)、`per_token_all_layers=131072` 字节(128KB)、单请求按 `max_len=2048` 预留 `256MB`,24GB 显存最多容纳 `96` 个并发请求(L02 原文写的是"~80",量级一致,差异大概率来自 L02 没有把权重/激活值等其他显存开销刨除干净)。prefill/decode 的"单次迭代并行度"差距被精确验证为 `100` 倍(100 token/次 vs 1 token/次)。`EngineMetrics.throughput_out=10.0`、`throughput_total=60.0`,和手算吻合。真实用 `Engine.step()` 跑出的 TTFT(人为插入 20ms 延迟做对照)实测为 `20.2~20.3ms`(多次运行有 0.1-0.3ms 抖动,量级稳定),证明 `t_admit`/`t_first_token`/`t_finish` 三个时间戳在真实调度循环里确实会被正确写入,不是摆设字段。

**面试怎么问 + 追问链:**
- **Q:** "为什么说 LLM 推理的 decode 阶段是 memory-bound,prefill 阶段是 compute-bound?" —— 期望说出"decode 每步只新算 1 个 token 的 Q/K/V,但要从显存搬运全部历史 KV cache 做 attention,搬运数据时间远超计算时间;prefill 一次性对整个 prompt 做大矩阵乘法,计算密度高,GPU 算力能被打满"。
- **追问 1:** "如果一个请求的 prompt 特别长(比如 32k token),decode 阶段还会是 memory-bound 吗?" —— 期望能辨析:decode 每一步待搬运的 KV 数据量确实会随 prompt 变长而增大,单步耗时也会变长,但"每步只产出 1 个 token、访存量远超计算量"这个结构性质不变,量变不会改变"memory-bound"这个定性判断,只是访存量的绝对值更大。
- **追问 2(深挖三大指标):** "吞吐、延迟、goodput 三个指标里,为什么不能只优化吞吐?" —— 期望举出反例:一个引擎可以靠疯狂攒大 batch 把吞吐刷到很高,但单个请求的 TTFT/ITL 可能因此被拖得很长,超出业务能接受的 SLO——goodput 就是"吞吐"和"延迟约束"两者的合取,只看吞吐会诱导出对用户体验很差的调度策略。
- **追问 3:** "`Request.total_len` 这个字段在后面知识点里具体被谁使用?" —— 期望能连回知识点 3/5(PagedAttention 的 block 分配、Engine 的 KV 预算判断)都是直接读这个字段做容量决策,是贯穿全系列的一个基础量。

**常见坑:** 把"prefill compute-bound、decode memory-bound"错误地理解成"prefill 慢、decode 快"——实际上单次 decode 迭代通常比单次 prefill 快得多(处理的 token 少),这里说的"bound"是"瓶颈资源类型"而不是"绝对耗时";另一个坑是把 goodput 和 throughput 混为一谈,以为二者只是叫法不同,实际 goodput 是"吞吐"叠加了"是否满足延迟 SLO"的过滤条件,一个高吞吐、高延迟的引擎 goodput 可能反而很低。

---

## 2. Naive KV Cache 及碎片问题(`naive_kv.py`,L02)—— 静态预留 vs 实际使用的错配

**是什么:**
```python
@dataclass
class NaiveKvPool:
    """Static `[B, max_len, n_kv_heads, d_h]` KV cache."""
    batch: int
    max_len: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    dtype: torch.dtype = torch.float16
    device: str = "cpu"

    def __post_init__(self) -> None:
        shape = (self.n_layers, self.batch, self.max_len, self.n_kv_heads, self.head_dim)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.actual_lens: List[int] = [0] * self.batch

    def utilization(self) -> float:
        used = sum(self.actual_lens)
        total = self.batch * self.max_len
        return used / max(total, 1)
```
(`naive_kv.py:13-30`、`:44-48`,节选)

```python
def demo_fragmentation(B: int = 8, max_len: int = 2048, avg_len: int = 512) -> dict:
    """Compute static-frag waste for a representative workload."""
    ...
```
(`naive_kv.py:54`,签名)

**一句话:** `NaiveKvPool` 一上来就按 `[n_layers, batch, max_len, n_kv_heads, head_dim]` 把整块张量分配到位——这是"按 slot 预留"(预留空间在 `max_len` 那一刻就定死),但 KV cache 实际是"按 token 写入"的(每个请求真正用掉的长度千差万别),两者的错配就是碎片的根源,`utilization()` 直接把这个错配量化成一个百分比。

**底层机制/为什么这样设计:** 从最笨的想法讲起——要给一个 batch 里的每个请求存 KV,最直接的做法就是"先按最坏情况(`max_len`)把张量分配好,写的时候按位置填进去"。这样做的好处是实现极其简单(`self.k[layer, b, pos] = k`,一行索引赋值,连内存管理都不用操心),坏处是显存分配量只取决于 `batch × max_len`,和请求的**实际**长度完全无关。`demo_fragmentation()` 把这个错配具体量化:如果 `max_len=2048` 而请求实际长度平均只有 `512` 左右,那么每个请求预留的 2048 个 slot 里大约 1500 个永远不会被写入——这叫 **internal fragmentation**(块内碎片,预留了但没用上);即便两个请求长度不同,它们的空闲 slot 也不能互相借用给对方(每个请求的 slot 范围在分配那一刻就被物理锁定),这叫 **external fragmentation**。`NaiveKvPool.reset(b)` 只是把 `actual_lens[b]` 清零,并不释放或收缩底层张量的物理大小(`memory_bytes()` 在 `reset()` 前后完全不变)——这进一步说明"逻辑上腾出空间"和"物理上真的少占显存"在这套朴素实现里是两件不同的事,朴素方案只能做前者,做不到后者(除非重新分配整个张量)。这正是知识点 3 PagedAttention 要解决的问题:把"预留"和"实际使用"这两件事在更细的粒度(block,而不是整个请求)上解耦。

**AI 研究场景:** 这不是一个只存在于教学演示里的问题——早期(2022 年前)几乎所有开源 LLM 推理服务代码(包括很多论文的 baseline 实现)都是这种"按 max_len 静态分配"的写法,因为实现简单、调试容易。vLLM 论文(SOSP'23)正是拿这个 naive 方案当基线,测出真实生产工作负载下 KV cache 利用率普遍只有 20.4%-38.2%,这也是为什么"PagedAttention"这篇论文的第一个卖点不是"更快的 attention 计算",而是"用更少的显存装下更多并发请求"——本知识点的 `demo_fragmentation()` 就是在复现这类基线测量的方法论(不是复现论文的具体数值,种子和参数都不同,但浪费量级一致)。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from naive_kv import NaiveKvPool, demo_fragmentation

# --- Part A: 直接操作真实的 NaiveKvPool 类（不只是看纯统计模拟函数） ---
pool = NaiveKvPool(batch=4, max_len=64, n_kv_heads=2, head_dim=8, n_layers=2)
assert pool.k.shape == (2, 4, 64, 2, 8)  # [n_layers, batch, max_len, n_kv_heads, head_dim]
assert pool.utilization() == 0.0  # 还没写入任何 token

lengths = [10, 64, 1, 30]  # 一个请求恰好写满 max_len，一个只写 1 个 token
for b, length in enumerate(lengths):
    for pos in range(length):
        for layer in range(2):
            k = torch.randn(2, 8, dtype=torch.float16)
            v = torch.randn(2, 8, dtype=torch.float16)
            pool.write(layer, b, pos, k, v)

assert pool.actual_lens == lengths
used = sum(lengths)
reserved = 4 * 64
assert pool.utilization() == used / reserved
util_before_reset = pool.utilization()
assert abs(util_before_reset - 0.41016) < 1e-3  # (10+64+1+30)/256

# fetch 只返回 actual_lens[b] 行，不是 max_len 行——“浪费”体现在预留张量里，不体现在 fetch 结果里
k_fetched, v_fetched = pool.fetch(layer=0, b=1)
assert k_fetched.shape == (64, 2, 8)
k_fetched0, _ = pool.fetch(layer=0, b=3)
assert k_fetched0.shape == (30, 2, 8)  # 请求 3 只用掉预留的 64 个 slot 里的 30 个

expected_bytes = 2 * pool.k.numel() * pool.k.element_size()
assert pool.memory_bytes() == expected_bytes
assert pool.memory_bytes() == 2 * (2 * 4 * 64 * 2 * 8) * 2  # K+V, fp16=2 bytes/elem

# reset() 只清空逻辑长度，物理预留（以及 memory_bytes）完全不变
pool.reset(b=3)
assert pool.actual_lens[3] == 0
assert pool.memory_bytes() == expected_bytes  # 预留本身从不收缩

# --- Part B: demo_fragmentation() 换一组和文件自带 __main__ 不同的配置，独立复核“浪费依旧很高” ---
info = demo_fragmentation(B=16, max_len=1024, avg_len=256)
assert info["reserved"] == 16 * 1024 == 16384
assert info["used_tokens"] == sum(info["lengths"])
assert info["wasted_tokens"] == info["reserved"] - info["used_tokens"]
assert info["wasted_pct"] > 0.5  # max_len 缩到 __main__ 默认值的 1/4，浪费依然过半
assert abs(info["utilization"] + info["wasted_pct"] - 1.0) < 1e-9
```

**实测(`.venv` 真跑):** 手工构造的 `NaiveKvPool(batch=4, max_len=64, ...)`,写入 `lengths=[10,64,1,30]` 后 `utilization()` 精确为 `0.41016`(105/256);`reset(b=3)` 后 `utilization()` 变为 `0.2930`,但 `memory_bytes()` 前后都是 `32768` 字节,一字节没少——验证了"reset 只清逻辑、不收缩物理"这条结论。文件自带 `__main__`(`B=8, max_len=2048, avg_len=512`,固定种子)实测 `wasted_pct=70.2%`(`utilization≈29.8%`);本知识点独立换一组配置(`B=16, max_len=1024, avg_len=256`)重新测,`wasted_pct=73.8%`——两组不同配置下浪费都在 70% 以上,和 L02 给出的"naive 平均利用率 25-40%"结论方向一致(甚至更极端),说明这不是某一组参数凑巧导致的现象。

**面试怎么问 + 追问链:**
- **Q:** "朴素 KV cache 的显存浪费具体来自哪两类碎片?" —— 期望区分 internal fragmentation(单个请求预留但没用满)和 external fragmentation(不同请求之间的空闲 slot 不能互相借用)。
- **追问 1:** "如果把 `max_len` 设得刚好等于这一批请求的平均长度,是不是就能避免浪费?" —— 期望指出:调小 `max_len` 只能缓解 internal fragmentation,但会导致长请求直接被截断(over-reservation 问题的反面),而且 external fragmentation(不同请求 slot 不能互相复用)完全没解决,真正的修法是从"按请求预留连续大块"换成"按更小的、可复用的物理块分配"。
- **追问 2(考察是否真读过代码):** "`NaiveKvPool.reset(b)` 调用之后,这个请求占的显存真的被释放了吗?" —— 期望明确说"没有,`reset()` 只把 `actual_lens[b]` 清零,底层 `self.k`/`self.v` 张量的物理大小从头到尾不变,`memory_bytes()` 前后完全相同",这是本知识点验证过的真实结论,不是靠猜。
- **追问 3:** "如果这套朴素方案要支持长度大到无法预知上限的请求(比如流式生成、agent 多轮),会遇到什么新问题?" —— 期望说出"必须在建表时就预估一个 `max_len` 上限,一旦实际长度超过这个上限,要么截断要么整块重新分配拷贝——这正是"over-reservation"这个第三类浪费的来源,不只是"用不满"的问题,还有"根本不够用"的风险"。

**常见坑:** 把 `demo_fragmentation()` 的模拟结果当成"这就是真实 `NaiveKvPool` 类跑出来的浪费率"——两者是分开的:`demo_fragmentation()` 是一个独立的纯统计函数(用随机长度模拟一批请求,不实际创建任何张量),`NaiveKvPool` 才是真正分配显存的类,本知识点特意把两者都跑了一遍并交叉核对结论方向一致,但如果只跑其中一个就以为覆盖了另一个,是不准确的。另一个坑是以为 `utilization()` 低就等于"这个实现有 bug"——`NaiveKvPool` 的行为完全符合它自己的设计(静态预留),利用率低是这种设计范式的必然结果,不是实现缺陷,这也是为什么解法是换一种数据结构(PagedAttention),而不是给 `NaiveKvPool` 打补丁。

---

## 3. PagedAttention 原理(`paged_kv.py`,L03)—— OS 虚拟内存分页类比,block table 间接寻址

**是什么:**
```python
@dataclass
class PagedKvPool:
    n_blocks: int
    block_size: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    dtype: torch.dtype = torch.float16
    device: str = "cpu"

    def alloc_block(self) -> int:
        if not self.free_ids:
            raise RuntimeError("PagedKvPool OOM: no free blocks")
        blk = self.free_ids.popleft()
        self.refcount[blk] = 1
        return blk
```
(`paged_kv.py:23-47`,节选)

```python
@dataclass
class BlockTable:
    """Per-request logical-to-physical mapping."""
    pool: PagedKvPool
    block_ids: List[int] = field(default_factory=list)
    n_tokens: int = 0

    def append_token(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.ensure_capacity(self.n_tokens + 1)
        blk = self.block_ids[self.n_tokens // self.block_size]
        slot = self.n_tokens % self.block_size
        self.pool.write_token(layer, blk, slot, k, v)
        if layer == 0:
            self.n_tokens += 1

    def fork(self) -> "BlockTable":
        """Copy-on-write fork: share existing blocks read-only."""
        child = BlockTable(pool=self.pool, n_tokens=self.n_tokens)
        for blk in self.block_ids:
            self.pool.share_block(blk)
            child.block_ids.append(blk)
        return child
```
(`paged_kv.py:66-102`,节选)

**一句话:** `PagedKvPool` 是全局共享的物理块池,`BlockTable` 是每个请求自己的"逻辑位置 → 物理块 id"映射表——这一对结构和操作系统"虚拟地址空间 + 页表 + 物理页"的关系完全同构,请求不再要求自己的 KV 连续存放在显存里,只要 block table 记得每一段逻辑位置对应哪个物理块,kernel 内部做一次"间接寻址"(gather)就行。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 2 的浪费根源是"一个请求的 KV 必须占一段连续、且大小在分配时就固定的显存"。PagedAttention 的做法是照抄操作系统解决同一类问题(进程虚拟地址空间碎片化)的方案:不再要求逻辑上连续的数据在物理上也连续存放,而是把逻辑空间切成固定大小的"页"(这里叫 block,默认 `block_size=16` 个 token),物理显存也切成同样大小的块池(`PagedKvPool`),每个请求维护一张"页表"(`BlockTable.block_ids`),记录它的第 `i` 个逻辑 block 对应哪个物理 block id。`append_token()` 写入第 `n_tokens` 个 token 时,先算出这个 token 属于第几个逻辑 block(`n_tokens // block_size`),通过 `block_ids` 查到对应物理 block,再定位块内偏移(`n_tokens % block_size`)——这一步就是"页表查找 + 页内偏移"的直接翻版。`ensure_capacity()` 只在当前逻辑 block 数不够时才向 `PagedKvPool` 申请新的物理块(`alloc_block()`),这是"按需分配"而不是知识点 2 那种"一次性按上限预留",碎片被压缩到最多只有"最后一个未写满的 block"这一点点(最坏情况浪费 `block_size - 1` 个 token,不是整个 `max_len`)。`fork()` 对应操作系统的**写时复制(COW)**:两个 `BlockTable` 可以共享同一批物理 block(`share_block()` 只是把 `refcount` 加一,不拷贝任何数据),这是 beam search/parallel sampling 场景下"多个候选序列共享同一段历史 KV,只在各自新增的部分才需要新物理块"这个优化的直接实现;`free_block()` 只有在 `refcount` 降到 0 时才真正把物理块还给 `free_ids`,避免一个还在被别的请求引用的 block 被提前释放。

**AI 研究场景:** "共享 block、只在分叉点才复制"这个能力不只是 beam search 的优化技巧——它是知识点 7 Prefix Caching 的底层机制(多个请求的 system prompt 部分直接 mount 同一批物理 block,不需要真的复制 KV 数据),也是 02 号文件 RadixAttention 把这个思路从"整条链共享"推广到"树状部分共享"的起点。理解 `BlockTable`/`refcount`/`share_block`/`free_block` 这一套引用计数机制,是理解 vLLM 生态里几乎所有"多请求复用 KV"类优化(prefix caching、beam search、parallel sampling、speculative decoding 的草稿树)的公共前置知识。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from paged_kv import PagedKvPool, BlockTable, utilization

pool = PagedKvPool(n_blocks=20, block_size=8, n_kv_heads=2, head_dim=4, n_layers=1)
assert pool.n_free() == 20
lengths = [13, 8, 41, 3]  # 一个恰好整除 block_size，一个远小于，一个横跨多个 block
tables = [BlockTable(pool) for _ in lengths]
for t, length in zip(tables, lengths):
    for _ in range(length):
        k = torch.randn(2, 4, dtype=torch.float16)
        v = torch.randn(2, 4, dtype=torch.float16)
        t.append_token(0, k, v)

# ceil(len/block_size) 个 block：13->2, 8->1, 41->6, 3->1（共占 20 块里的 10 块）
blocks_used = [len(t.block_ids) for t in tables]
assert blocks_used == [2, 1, 6, 1]
assert pool.n_free() == 20 - sum(blocks_used) == 10

util = utilization(tables)
assert abs(util - sum(lengths) / (sum(blocks_used) * 8)) < 1e-9
assert util > 0.8  # 远高于 naive 的 25-40%，即便长度组合刻意选得不规整

# --- fork() 是真正的写时复制：共享 block id、refcount+1，不消耗任何新 block ---
free_before_fork = pool.n_free()
child = tables[2].fork()  # fork 41-token / 6-block 的那个请求
assert pool.n_free() == free_before_fork  # fork 不应分配任何新块
assert child.block_ids == tables[2].block_ids  # 相同物理 id，不是拷贝
for blk in tables[2].block_ids:
    assert pool.refcount[blk] == 2  # parent + child 都引用它

child.free()
assert pool.n_free() == free_before_fork  # 仍未释放——parent 的 refcount==1 还在用
for blk in tables[2].block_ids:
    assert pool.refcount[blk] == 1
tables[2].free()
assert pool.n_free() == free_before_fork + 6  # 现在 6 个块才真正归还

# --- OOM：alloc_block() 耗尽时显式抛错，不会静默包装/复用 ---
drain = BlockTable(pool)
n_before_drain = pool.n_free()
for _ in range(n_before_drain * 8):
    drain.append_token(0, torch.zeros(2, 4, dtype=torch.float16), torch.zeros(2, 4, dtype=torch.float16))
assert pool.n_free() == 0
try:
    drain.append_token(0, torch.zeros(2, 4, dtype=torch.float16), torch.zeros(2, 4, dtype=torch.float16))
    assert False, "expected RuntimeError"
except RuntimeError as e:
    assert "OOM" in str(e)
```

**实测(`.venv` 真跑):** `n_blocks=20, block_size=8` 下,`lengths=[13,8,41,3]` 精确分配 `blocks_used=[2,1,6,1]`(共 10 块),剩余 `pool.n_free()=10`;整体 `utilization=0.8125`,远高于知识点 2 naive 方案的 25-40%。`fork()` 验证为真正的 COW:`child.block_ids == tables[2].block_ids`(同一组物理 id),fork 前后 `pool.n_free()` 完全不变,共享期间 `refcount` 精确为 `2`,`child.free()` 后降回 `1`(6 个块仍未归还,因为父表还在用),父表也 `free()` 后 `pool.n_free()` 才真正加回 `6`。OOM 场景:耗尽 20 个块后,第 21 次分配精确抛出 `RuntimeError("PagedKvPool OOM: no free blocks")`,不是静默截断或复用别的请求的块。

**面试怎么问 + 追问链:**
- **Q:** "PagedAttention 和操作系统虚拟内存管理的类比具体怎么对应?" —— 期望完整说出"逻辑 KV 序列↔虚拟地址空间,KV block↔物理页,block table↔页表,block 分配↔缺页中断,block 共享↔共享页/COW"这一整套对应关系,不是只说出"都叫分页"这种表面相似。
- **追问 1:** "block table 本身要占多少额外开销,`BLOCK_SIZE` 选得越小是不是共享粒度越细、就一定越好?" —— 期望能双向权衡:`BLOCK_SIZE` 越小,共享前缀的粒度越细(命中率上限更高),但 block table 本身的条目数会变多(每个请求要记录更多"逻辑块→物理块"映射),kernel 内部要 gather 的次数也变多;vLLM 默认选 16 是这个权衡下的经验值,SGLang 甚至激进到 `BLOCK_SIZE=1`(配合 radix tree 结构专门优化这个开销)。
- **追问 2(深挖 COW 正确性):** "如果两个请求 fork 共享了同一批 block,其中一个请求要继续往后写新 token,会不会污染另一个请求已经共享的历史?" —— 期望说出"不会,因为 `append_token` 只会在当前逻辑长度对应的位置写,新 token 一定落在 `ensure_capacity` 新分配出来的、只属于自己的新 block 里,已经共享的旧 block 是只读的,不会被就地修改——这也是为什么 PagedAttention 的共享被称为'read-only sharing'"。
- **追问 3:** "OOM 的时候,`alloc_block()` 选择直接抛异常,而不是尝试驱逐(evict)某个不活跃请求的 block 来腾地方,这是不是设计缺陷?" —— 期望能辨析:这是这个教学实现故意保持简单的选择,真实 vLLM 确实有更复杂的处理(swap to CPU、抢占重算),`PagedKvPool` 本身只负责"页表/物理块池"这一层机制,"OOM 时怎么办"是调度器(知识点 5/8)的职责,两者是分层设计,不应该把驱逐逻辑塞进最底层的块池类里。

**常见坑:** 把"PagedAttention 消除了 external fragmentation"理解成"完全没有任何浪费"——精确的结论是"最坏情况只浪费最后一个未写满 block 的空间(最多 `block_size-1` 个 token)",不是零浪费,只是浪费量级从"整个 `max_len`"降到了"一个 block"。另一个坑是把 `refcount` 机制想象成需要请求方手动维护——实际上 `share_block()`/`free_block()` 都是 `PagedKvPool` 自己管理的内部计数,调用方只需要正确地在 fork 时调 `fork()`、在请求结束时调 `free()`,不需要也不应该直接操作 `pool.refcount` 字典。

---

## 4. PagedAttention 的 kernel 层(`paged_attention_triton.py`,L04)—— 诚实标注的 stub,triton 装了也没被用到

**是什么:**
```python
def paged_attention_torch(
    q: torch.Tensor,                      # [n_heads, d_h]
    block_table: List[int],
    n_tokens: int,
    k_pool: torch.Tensor,                 # [n_blocks, block_size, n_kv_heads, d_h]
    v_pool: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation: collect K/V from blocks and run softmax."""
    ...
    K = K.repeat_interleave(rep, dim=1)  # GQA: repeat KV heads across query head groups
    V = V.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(d_h)
    scores = torch.einsum("hd,thd->th", q, K) * scale
    attn = torch.softmax(scores, dim=0)
    out = torch.einsum("th,thd->hd", attn, V)
    return out


def paged_attention_triton(*args, **kwargs):
    """Triton kernel placeholder.

    A real port lives in vLLM `csrc/attention/`; for the workshop we ship the
    torch reference and exercise correctness against it.  Plug in the Triton
    implementation by replacing this body.
    """
    return paged_attention_torch(*args, **kwargs)
```
(`paged_attention_triton.py:22-57`、`:60-67`)

**一句话:** `paged_attention_torch` 是"按 block table 逐块 gather K/V、拼起来再做标准 attention"的纯 torch 参考实现(含 GQA 的 `repeat_interleave`);名字叫 `paged_attention_triton` 的函数体只有一行 `return paged_attention_torch(*args, **kwargs)`——它不是"triton 装了就走 triton 路径、没装就回退 torch"这种条件分支,而是**无条件**委派给 torch 版,文件里从头到尾没有一处 `import triton`,这是 docstring 已经写明的占位符,不是隐藏的 bug。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么 PagedAttention 光有 `paged_kv.py` 那套"block table + 物理池"数据结构还不够,非要单独一个 kernel 层?因为知识点 3 描述的"按 block_table 遍历、逐块取出 K/V、拼成连续张量再做 attention"这个过程,如果真的按 Python `for` 循环 + `torch.cat` 的方式执行,本知识点验证过对一个只有十几个 token、几个 block 的极小例子都要经过多次 Python 级别的张量拼接——放到真实场景(几千 token、上百 block、每步都要对一整个 batch 的每个请求都做一遍)会慢到不可用,这也是 L04 说的"朴素 Python loop 慢 100×"的来源。真实 vLLM 的做法是写一个 CUDA/Triton kernel,在 kernel 内部直接按 block id 做地址计算和 gather,把"block 一次性 load 到 SRAM""避免重复从全局显存读 K"这些优化下沉到 kernel 层,不经过 Python 解释器和张量拼接的开销。这个教学仓库的做法是诚实分成两条路径:`paged_attention_torch` 老老实实用 `torch.cat`+`einsum` 把计算过程摆出来,方便学习和做数值对照;`paged_attention_triton` 顶着"triton kernel"的名字,但函数体明确写着"占位符,真正的 Triton 实现留给你自己接入",直接返回 torch 版的结果。这里有一个容易被想当然的细节:这台机器上 `triton` 实际是装着的(`3.7.1`,可以正常 `import`),如果只看"环境里有没有 triton"就推断"这个函数应该会用 triton 加速",会得出错误结论——真正决定的是**函数体本身有没有 `import triton`/`@triton.jit`/`tl.load` 这些代码**,而这个文件从头到尾一处都没有,triton 装没装、装的是哪个版本,对这个函数的行为**没有任何影响**,它永远走 torch 路径。这和 kernel-gpu-deep-dive/03 知识点 5(`swizzle_32b` stub)是同一类"文档承诺 vs 函数体实际实现"的落差,但这里的落差更彻底:那边好歹是"用错的公式返回了一个恒等布局",这里是"函数名叫 triton,但连 triton 这个词在可执行代码里出现的次数都是零"。

**AI 研究场景:** 这套"先用纯 Python/torch 把算法逻辑摆清楚、验证数值正确,再单独换一层 kernel 实现去追求性能"的分层方法论,是几乎所有生产级推理引擎(vLLM、SGLang、TensorRT-LLM)共享的开发范式——先有一个"慢但一定对"的参考实现当作正确性基准(ground truth),后续任何 kernel 层面的优化(重排计算顺序、换用不同精度、改内存访问模式)都要先过这个基准的数值对照关卡,才敢合入。knowledge point 里 `test_paged_attention_matches_dense` 这条测试,以及本文下面独立验证的"paged 路径与稠密 attention 参考实现最大绝对误差为 0"结论,正是这套方法论落地的证据:一个正确性可以做到 bit-exact 对齐的参考实现,才有资格作为后续任何 kernel 加速版本的验收标准。

**可运行例子:**
```python
import sys, inspect, math
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from paged_attention_triton import paged_attention_torch, paged_attention_triton

# --- 事实 1：paged_attention_triton 是纯委派，函数体里没有任何 triton 相关代码 ---
src = inspect.getsource(sys.modules["paged_attention_triton"])
assert "import triton" not in src
assert "@triton.jit" not in src and "tl.load" not in src and "tl.store" not in src
triton_body = inspect.getsource(paged_attention_triton)
assert "return paged_attention_torch(*args, **kwargs)" in triton_body

# --- 事实 2：这台机器其实装了 triton（和这个模块自己 runbook.yaml 里的陈旧说明相反）---
import triton
assert triton.__version__  # 真实可 import，包确实存在
# ……但下面 paged_attention_triton() 依然和 torch 参考实现逐 bit 相同，因为它压根不分发到 triton。

# --- 事实 3：独立换一组和 src/tests 不同的配置，重新验证 bit-exact 对齐 ---
torch.manual_seed(123)
block_size, n_blocks, n_kv_heads, d_h = 4, 6, 3, 24
n_heads = 6   # GQA: rep = n_heads // n_kv_heads = 2
n_tokens = 11
k_pool = torch.zeros(n_blocks, block_size, n_kv_heads, d_h)
v_pool = torch.zeros(n_blocks, block_size, n_kv_heads, d_h)
K_dense = torch.randn(n_tokens, n_kv_heads, d_h)
V_dense = torch.randn(n_tokens, n_kv_heads, d_h)
block_table = [5, 0, 2]   # 故意用乱序的物理 block id，不同于已有测试的 [i+2]
for i in range(n_tokens):
    bid = i // block_size
    phys = block_table[bid]
    slot = i % block_size
    k_pool[phys, slot] = K_dense[i]
    v_pool[phys, slot] = V_dense[i]

q = torch.randn(n_heads, d_h)
rep = n_heads // n_kv_heads
K_rep = K_dense.repeat_interleave(rep, dim=1)
V_rep = V_dense.repeat_interleave(rep, dim=1)
scale = 1.0 / math.sqrt(d_h)
scores = torch.einsum("hd,thd->th", q, K_rep) * scale
attn = torch.softmax(scores, dim=0)
ref = torch.einsum("th,thd->hd", attn, V_rep)

out_torch = paged_attention_torch(q, block_table, n_tokens, k_pool, v_pool)
out_triton_name = paged_attention_triton(q, block_table, n_tokens, k_pool, v_pool)

assert (out_torch - ref).abs().max().item() == 0.0
assert (out_triton_name - out_torch).abs().max().item() == 0.0
assert torch.equal(out_triton_name, out_torch)  # 逐 bit 相同，不是数值接近
```

**实测(`.venv` 真跑):** 静态检查确认 `paged_attention_triton.py` 全文没有 `import triton`/`@triton.jit`/`tl.load`/`tl.store`,`paged_attention_triton()` 函数体精确只有一行委派语句。`import triton` 在这台机器上真实成功,`triton.__version__` 为 `3.7.1`——但这不影响任何计算路径。独立换一组配置(`block_size=4, n_blocks=6, n_kv_heads=3, n_heads=6` 的 GQA,`rep=2`,乱序 `block_table=[5,0,2]`)重新验证:`paged_attention_torch` 输出与手动展开的稠密 attention 参考实现最大绝对误差为 `0.0`,`paged_attention_triton` 的输出与 `paged_attention_torch` `torch.equal`(逐 bit 相同)。

**面试怎么问 + 追问链:**
- **Q:** "`paged_attention_triton()` 这个函数真的用 Triton 实现了 kernel 吗?" —— 期望明确说"没有,它是一个占位符,函数体直接委派给纯 torch 的 `paged_attention_torch`,docstring 里已经写清楚这一点"。
- **追问 1(核心陷阱):** "如果我在这台机器上 `pip install triton` 装好了,再跑这个函数,它会不会自动切换到用 triton 加速?" —— 期望明确说"不会,决定路径的是函数体里有没有 triton 相关的调用代码,不是环境里有没有装这个包;这个函数从代码层面就没有任何'检测 triton 是否可用、可用就切换'的分支逻辑,triton 装不装、装哪个版本对它的行为没有任何影响"。这道题专门筛"只看包管理器输出就下结论"和"真读函数体"两类候选人。
- **追问 2:** "为什么真实 kernel 要把 attention 计算搬到 SRAM 上做,而不是直接在朴素 Python/torch 层面对着 block_table 做 gather?" —— 期望说出"频繁访问全局显存(HBM)延迟高,真实 kernel 会把当前需要的 K/V block 一次性搬进片上共享内存(SRAM)反复复用,配合 online softmax 避免物化整个 attention 矩阵;而这里的 torch 参考实现是为了教学/正确性验证图省事直接用 `torch.cat`+`einsum`,性能不是它的设计目标"。
- **追问 3:** "既然这个 kernel 是假的,那知识点 3 里 `paged_kv.py` 那套数据结构还有意义吗?" —— 期望能区分"数据结构设计"和"kernel 实现"是两层独立的事:block table、物理池、COW 引用计数这些是 PagedAttention 算法层面的核心贡献,`paged_attention_triton.py` 只是"这些数据结构喂给 attention 计算"这最后一步的性能实现,后者是不是真 Triton kernel 完全不影响前者的算法正确性和教学价值。

**常见坑:** 看到文件名叫 `paged_attention_triton.py`、函数名叫 `paged_attention_triton`,就直接假设"这里面一定用了真 Triton kernel"——这是本知识点要纠正的最大陷阱,必须读函数体或者至少搜索 `import triton` 才能确认。另一个坑是把"triton 装了但没被用到"当成这个仓库的 bug 去报告——docstring 已经诚实说明这是占位符、把真实 Triton 实现留作后续练习,这是有意为之的教学设计,不是遗漏;真正需要警惕的"名不副实"是那种连 docstring 都不说明差距的情况,这里不属于此类。

---

## 5. Continuous Batching(`continuous_batching.py`,L05)—— Orca 式迭代级调度,消灭"静态批"空转

**是什么:**
```python
@dataclass
class Engine:
    max_running: int = 8
    kv_budget: int = 1024
    forward_fn: Callable[[List[Request]], List[int]] = stub_token_step
    eos_id: int = -1

    def _can_admit(self, req: Request) -> bool:
        need = len(req.prompt_ids) + req.max_new_tokens
        return (
            len(self.running) < self.max_running
            and self._kv_used() + need <= self.kv_budget
        )

    def step(self) -> bool:
        """Run one scheduling iteration.  Returns True if anything happened."""
        admitted = 0
        while self.pending and self._can_admit(self.pending[0]):
            self.running.append(self.pending.popleft())
            admitted += 1
        if not self.running:
            return admitted > 0
        ...
```
(`continuous_batching.py:21-56`,节选)

**一句话:** `Engine.step()` 把"准入(admission)"和"forward 一次"绑在同一次调用里,每调一次 `step()` 就是 Orca 论文说的一个 **iteration**——完成的请求立刻从 `running` 移出、新请求立刻从 `pending` 补进来,不需要等一整个 batch 里最慢的那个请求彻底结束才能开始下一轮,这就是"迭代级调度"消灭静态批"空转"的直接实现。

**底层机制/为什么这样设计:** 从最笨的想法讲起——最直观的批处理方式是"一次性收集一批请求,大家一起 prefill、一起 decode,直到这一批全部完成才收下一批"。问题是同一批里请求的输出长度天差地别:一个请求 5 个 token 就结束、另一个要生成 500 个 token,静态批处理必须等最长的那个走完,已经完成的请求所占的那部分 batch 位置在这期间只能空转(要么用 padding token 填充继续陪跑、要么整个 GPU 等着),L05 lecture 用"3 个请求剩余 token 数不同、短请求要等长请求"的图示直观地展示了这个浪费。`Engine.step()` 的解法是把"决定这一轮谁在跑"和"跑一次模型"拆成每次迭代都重新评估的两件事:`step()` 一开始先做 admission(`while self.pending and self._can_admit(...)`),尽量把等待队列里排得上的请求塞进本轮;然后对 `running` 里现在这批请求做一次 forward,采样出新 token;最后检查每个请求是否已经完成(`len(r.output_ids) >= r.max_new_tokens` 或采样到 eos),完成的立刻移到 `finished`、留下 `still_running` 继续参与下一轮。这样一来,一个请求完成的那一刻空出来的"槽位"在下一次 `step()` 调用里立刻可能被别的等待中的请求接管,不需要等其他还在跑的请求也全部结束。`_can_admit()` 的准入条件同时检查 `max_running`(并发数上限)和 `kv_budget`(显存/KV 容量上限,`_kv_used()` 精确统计当前 `running` 里所有请求 `total_len` 之和)——这是知识点 1 的 `Request.total_len` 在这里的第一处实际使用:调度器不能只看"槽位够不够",还要看"KV 显存装不装得下",两个约束都满足才真正入队。

**AI 研究场景:** Orca(OSDI'22)论文测出"continuous batching + PagedAttention 一起用能拿到 24 倍吞吐提升,单独用任一个只有 2-3 倍"——这个"1+1 远大于 2"的结果不是偶然:continuous batching 解决的是"时间维度"的碎片(GPU 在等最慢请求时空转),PagedAttention 解决的是"空间维度"的碎片(显存预留和实际使用错配);两者独立时,continuous batching 想要塞进更多并发请求会被静态 KV 分配的显存浪费卡住,PagedAttention 省出来的显存如果还是按静态批处理调度、批次之间互相等待,也发挥不出应有的并发潜力。理解这种"多个优化互相解锁对方潜力"的关系,是理解现代推理引擎为什么要同时具备一整套机制(而不是随便挑一两个"最重要的"去实现)的关键。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/inference-engine-core/src")
from common import Request
from continuous_batching import Engine, make_request

# --- Part A: 迭代级准入——一个“迟到”的请求会在中途某一次 step() 里被补进来，
# 不需要等到某个固定的“批次边界” ---
eng = Engine(max_running=2, kv_budget=1000, eos_id=-1)
eng.add_request(make_request(0, prompt_len=4, max_new=3))
eng.add_request(make_request(1, prompt_len=4, max_new=3))
eng.add_request(make_request(2, prompt_len=4, max_new=5))  # 第 3 个请求：max_running=2 先卡住它

eng.metrics.t_start = 0.0
assert len(eng.running) == 0 and len(eng.pending) == 3
eng.step()  # 准入：只有 2 个能进（max_running=2），1 个继续 pending
assert len(eng.running) == 2 and len(eng.pending) == 1
assert {r.rid for r in eng.running} == {0, 1}

for _ in range(2):
    eng.step()
assert all(len(r.output_ids) == 3 for r in eng.running)
eng.step()  # 第 3 个 token -> 请求0/1 完成退场；请求2 在同一个 step() 里被补入
assert {r.rid for r in eng.finished} == {0, 1}
assert len(eng.running) == 1 and eng.running[0].rid == 2
assert len(eng.pending) == 0

while eng.running:
    eng.step()
assert {r.rid for r in eng.finished} == {0, 1, 2}

# --- Part B: kv_budget 太紧不会死锁，只会退化成串行准入，最终仍能全部跑完 ---
tight = Engine(max_running=8, kv_budget=12, eos_id=-1)
for i in range(4):
    tight.add_request(make_request(i, prompt_len=3, max_new=2))  # 每个需要 5 个 token 的预算
m = tight.run()
assert len(tight.finished) == 4
assert m.n_tokens_out == 8
assert m.n_tokens_in == 12

# --- Part C: forward_fn 是可插拔的——用一个确定性 callable 替换默认 stub ---
def constant_token_fn(running):
    return [7 for _ in running]  # 每个请求永远采样到 token id 7

custom = Engine(max_running=4, kv_budget=100, forward_fn=constant_token_fn, eos_id=7)
custom.add_request(make_request(0, prompt_len=2, max_new=10))
cm = custom.run()
assert cm.n_tokens_out == 1  # eos_id=7 且每次都采到 7 -> 恰好 1 步就结束
assert custom.finished[0].output_ids == [7]
```

**实测(`.venv` 真跑):** 3 请求场景下,`max_running=2` 精确限制第一轮只有 `{0,1}` 被准入;第 3 次 `step()` 里,请求 0/1 完成退场和请求 2 被补入**发生在同一次调用**内,验证了"迭代级"而不是"批次级"调度。紧 `kv_budget=12`(4 个请求各需 5 个预算)场景下,`Engine.run()` 没有死锁,`finished` 数量精确为 `4`,`n_tokens_out=8`、`n_tokens_in=12`,全部吻合预期。自定义 `forward_fn`(恒定返回 token 7、`eos_id=7`)场景下,请求恰好在第 1 步就因为采样到 eos 而结束,`output_ids == [7]`,验证了 `forward_fn` 参数真的是可插拔的,不是只能用默认 stub。

**面试怎么问 + 追问链:**
- **Q:** "Continuous Batching 具体解决了什么问题,和静态批处理的核心区别是什么?" —— 期望说出"静态批处理要等一批里最长的请求结束才能开始下一批,短请求完成后的槽位在这期间空转;continuous batching 在每个 iteration 边界重新决策谁在跑,完成的请求立刻退场、等待的请求立刻补入"。
- **追问 1:** "`_can_admit()` 只检查 `max_running` 和 `kv_budget` 两个条件就放行,如果放进来的请求后面在某一步突然需要比预期更多的 KV 空间(比如没有提前预留 `max_new_tokens` 对应的空间)怎么办?" —— 期望能观察到 `_can_admit` 用的 `need = prompt + max_new_tokens` 是"预估这个请求一辈子最多要用多少",提前把上限算进准入判断,所以只要请求没有超过自己声明的 `max_new_tokens`,后续步骤不会突然超预算;如果结合 PagedAttention 的按需分配,极端情况下仍可能出现"提前准入判断过于乐观"的问题,这也是为什么真实引擎还需要抢占/swap 机制兜底。
- **追问 2(深挖 Orca+PagedAttention 协同):** "论文说 continuous batching 和 PagedAttention 各自单独用只有 2-3 倍提升,一起用有 24 倍,这是为什么不是简单相加?" —— 期望说出"两者分别解决时间碎片和空间碎片,单独用其中一个会被另一个维度的浪费卡住上限——比如光有 continuous batching 但 KV 还是静态预留,能塞进的并发请求数依然受限于显存碎片;两者一起用才能同时把'调度灵活性'和'显存利用率'都打满,互相解锁对方的潜力"。
- **追问 3:** "如果一个请求的优先级很低,一直排在 `pending` 队列后面进不去,会不会永远得不到执行?" —— 期望能提到 starvation(饥饿)问题,以及 L05 lecture 提到的应对方式(max wait time 强制 promote、guaranteed slot fairness ratio),并能指出这套教学版 `Engine` 目前是纯 FCFS 准入顺序(`while self.pending and ...` 是顺序检查 `pending[0]`),没有实现任何反饥饿机制,这属于知识点 8 调度策略要补的部分。

**常见坑:** 把 `step()` 理解成"只做准入决策,不涉及实际 forward"——实际上 `step()` 是"准入 + 一次完整的 forward/采样/退场判断"打包在一起的原子操作,如果只调用 `step()` 一次就去检查 `running` 列表,可能会看到"刚准入的请求已经因为 `max_new_tokens=1` 立刻完成退场"这种反直觉的结果,不能想当然认为准入和执行是分开的两个步骤。另一个坑是把 `kv_budget` 理解成物理显存字节数——在这个教学实现里它的单位是"token 数"(`_kv_used()` 直接累加 `total_len`),不是字节,和知识点 3 `PagedKvPool` 那种真正按物理 block 计量的显存管理是两个不同粒度的抽象,不要混用。

---

## 6. Chunked Prefill(`chunked_prefill.py`,L06)—— 长 prompt 切块与 decode 混批,消灭 head-of-line blocking

**是什么:**
```python
@dataclass
class PrefillState:
    req: Request
    prefill_pos: int = 0

    @property
    def done(self) -> bool:
        return self.prefill_pos >= len(self.req.prompt_ids)


@dataclass
class ChunkedPrefillEngine:
    max_tokens_per_iter: int = 512
    decoding: List[Request] = field(default_factory=list)
    prefilling: List[PrefillState] = field(default_factory=list)

    def step(self) -> dict:
        budget = self.max_tokens_per_iter
        decode_n = min(len(self.decoding), budget)
        budget -= decode_n
        for ps in list(self.prefilling):
            if budget <= 0:
                break
            remaining = len(ps.req.prompt_ids) - ps.prefill_pos
            take = min(remaining, budget)
            ps.prefill_pos += take
            budget -= take
            if ps.done:
                self.prefilling.remove(ps)
                self.decoding.append(ps.req)
        ...
```
(`chunked_prefill.py:14-56`,节选)

**一句话:** `ChunkedPrefillEngine.step()` 每一轮先给 `decoding` 里每个请求预留 1 个 token 的预算(decode 优先),剩下的预算才拿去推进 `prefilling` 队列里某个请求的 prefill 进度(`prefill_pos` 每轮往前挪 `take` 个 token,凑不满一个 `max_tokens_per_iter` 就先停在半路,下一轮接着切),这样一个几千 token 的长 prompt 不会一次性占满整个 iteration,decode 请求不用眼巴巴等它。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 5 的 continuous batching 解决了"批次之间互相等待"的问题,但如果同一个 iteration 里混进了一个几千 token 的长 prompt 需要 prefill,这一整个 iteration 的耗时会被这个长 prefill 拖得很长(prefill 一个 8k token 的 prompt 可能占用 100ms),而同一批里的 decode 请求本来每步只需要几毫秒——所有 decode 请求都要陪着这个长 prefill 等 100ms,这就是 **head-of-line blocking**(队首阻塞):一个"大活"卡住了所有"小活"。Chunked Prefill 的解法是不再要求"一个请求的 prefill 必须在一个 iteration 里做完",而是把长 prompt 切成固定大小的 `chunk_size`(经验值 512)分批处理,每个 iteration 只往前推进一个 chunk,`PrefillState.prefill_pos` 就是"这个请求的 prefill 目前进行到第几个 token"的进度指针,`done` 属性判断是不是已经推进到了 prompt 末尾。`step()` 里预算分配的顺序很关键:**先满足 decode、再用剩余预算做 prefill**(`decode_n = min(len(self.decoding), budget)` 排在 `for ps in ...prefilling` 循环之前)——这保证了不管长 prompt 切得完不完,已经在 decode 阶段的请求每一轮都至少能往前挪 1 个 token,不会被一个正在切块的长 prefill 饿死。`take = min(remaining, budget)` 处理了"prompt 剩余长度不是 chunk_size 整数倍"的收尾情况:最后一个 chunk 天然是一个比 `chunk_size` 小的"零头",本知识点验证过这种"参差不齐的尾巴"被正确处理(不会多切或者漏切)。一旦某个请求的 `ps.done` 为真,它立刻从 `prefilling` 移到 `decoding`,从下一轮开始就能参与"decode 优先"的那部分预算分配——这也是为什么"prefill 阶段结束"和"decode 阶段开始"在这个引擎里是无缝衔接的,不需要额外的"阶段切换"开销。

**AI 研究场景:** L06 给出的实测数字很有代表性:不做 chunk 时,短 prompt 的 TTFT 是 0.05s,但如果这个短请求恰好排在一个长 prompt 后面,TPOT(每个输出 token 的耗时)会被拖到 50ms(被长 prefill 阻塞);做了 `chunk=512` 之后,长 prompt 自己的 TTFT 变差了(从 2.0s 变成需要多等一点,因为它自己被拆成多轮),但 TPOT 从 50ms 降到 8ms——这是一个刻意的权衡:牺牲长请求自己一部分的首字延迟,换取所有其他并发请求的尾延迟大幅改善。这正是生产环境里"公平性"和"单请求最优"之间的经典取舍,vLLM 的 `--enable-chunked-prefill --max-num-batched-tokens 512` 参数就是把这个权衡暴露给运维人员去调,本知识点的 `max_tokens_per_iter` 就是同一个旋钮的教学版本。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/inference-engine-core/src")
from common import Request
from chunked_prefill import ChunkedPrefillEngine, PrefillState

# 45 个 token 的 prompt，预算 20——45 不是 20 的整数倍，最后一个 chunk 天然是零头
eng = ChunkedPrefillEngine(max_tokens_per_iter=20)
long_req = Request(rid=0, prompt_ids=list(range(45)), max_new_tokens=4)
eng.add_request(long_req)
log = eng.run_until_idle()

prefill_total = sum(c[1] for step in log for c in step["prefill_chunks"] if c[0] == 0)
assert prefill_total == 45  # 每个 prompt token 恰好被 prefill 一次，零头也不例外
chunk_sizes = [c[1] for step in log for c in step["prefill_chunks"] if c[0] == 0]
assert chunk_sizes == [20, 20, 5]  # 45 = 20+20+5
assert long_req in eng.finished
assert long_req.output_ids == [0, 0, 0, 0]  # prefill 完成后 4 步 decode（stub token 恒为 0）

# --- head-of-line blocking 正是这里要验证的：短请求排在长 prefill 后面，
# 不需要等长 prefill 彻底做完，只要还有预算剩余就能被混进同一个 iteration ---
eng2 = ChunkedPrefillEngine(max_tokens_per_iter=16)
eng2.decoding.append(Request(rid=99, prompt_ids=[0] * 5, max_new_tokens=30))  # 预热的 decode 请求
eng2.add_request(Request(rid=1, prompt_ids=list(range(50)), max_new_tokens=2))
log2 = eng2.run_until_idle()
mixed_iters = [s for s in log2 if s["decode"] > 0 and s["prefill_chunks"]]
assert len(mixed_iters) >= 1
first_mixed = mixed_iters[0]
assert first_mixed["decode"] == 1  # 此刻恰好有 1 个 decoding 请求（rid=99）
prefill_take = first_mixed["prefill_chunks"][0][1]
assert prefill_take == 16 - 1  # 预算(16) - decode(1) 的剩余全部给 prefill chunk

# --- is_idle() / PrefillState.done 是两个终止判定原语 ---
fresh = ChunkedPrefillEngine(max_tokens_per_iter=8)
assert fresh.is_idle() is True
ps = PrefillState(Request(rid=5, prompt_ids=list(range(3)), max_new_tokens=1))
assert ps.done is False
ps.prefill_pos = 3
assert ps.done is True  # 只要 prefill_pos 追上 prompt 长度就算完成，和预算无关
```

**实测(`.venv` 真跑):** 45 token 的 prompt、预算 20,`chunk_sizes` 精确为 `[20, 20, 5]`(5-token 的零头被正确处理);4 步 decode 后请求完成,`output_ids == [0,0,0,0]`。混批场景下,首个"decode+prefill 同时出现"的 iteration 里 `decode=1`(预热的请求 99),`prefill_take=15`(预算 16 减去 decode 占的 1),验证了"decode 预算优先扣、剩余才给 prefill"这条顺序。

**面试怎么问 + 追问链:**
- **Q:** "Chunked Prefill 具体解决了什么问题?" —— 期望说出"long prompt 的 prefill 会独占一整个 iteration、阻塞同批次里所有 decode 请求(head-of-line blocking);Chunked Prefill 把长 prefill 切成固定大小的小块,每个 iteration 只推进一块,和 decode 请求混在同一批里"。
- **追问 1:** "`step()` 里为什么先给 decode 预留预算,再把剩下的给 prefill,而不是反过来?" —— 期望说出"decode 请求通常是已经在流式吐字的、对延迟敏感的请求,如果 prefill 优先占预算,decode 请求可能被完全饿死(每次预算都被新来的长 prompt 抢光);先满足 decode 保证了在场的每个流式请求每一轮都至少有进展,prefill 只是在'不影响 decode'的前提下见缝插针地推进"。
- **追问 2(深挖收益权衡):** "L06 给出的数字是不加 chunk 时短 prompt TTFT 0.05s、chunk=512 时变成 0.1s——TTFT 反而变差了,这是不是说明 chunked prefill 得不偿失?" —— 期望能辨析"这里的 TTFT 变差指的是长 prompt 自己的 TTFT(因为它自己被切成多轮,首 token 出来得比一次性做完晚),但换来的是所有其他并发请求的 TPOT 从 50ms 降到 8ms——这是牺牲一个请求的指标去换全局大多数请求指标的经典权衡,不能孤立看单一指标是变好还是变差,要看它优化的是谁的体验"。
- **追问 3:** "`chunk_size` 应该怎么选,是不是越小越好?" —— 期望结合 L06 给出的公式方向作答:`chunk_size` 太小,prefill 完成一个长 prompt 需要的 iteration 数暴增,调度开销和 Python 层面的循环次数增加;`chunk_size` 太大,又会退化回接近"prefill 独占一个 iteration"的问题,512 是经验取值,真实系统里通常会结合 `decode` 请求的每步耗时反推一个"既不明显拖慢 decode、又不至于让 prefill 完成得过慢"的折中值。

**常见坑:** 把 `chunk_size` 理解成"这个引擎每次一定切固定大小的块"——实际上最后一个 chunk 大概率是"零头"(`take = min(remaining, budget)`),不是固定大小,如果假设所有 chunk 都一样大去做后续计算(比如估算总迭代次数),会在边界情况算错。另一个坑是把 Chunked Prefill 和知识点 5 Continuous Batching 当成互斥的两选一方案——实际上两者是叠加关系:continuous batching 解决"批次之间"的调度粒度,chunked prefill 解决"单个长请求内部"要不要一次性做完的粒度,真实 vLLM 是两者一起开的,`README` 里"与 continuous batching 共存"这句话就是在强调这一点。

---

## 7. Prefix Caching(`prefix_cache.py`,L07)—— block-hash + LRU,给 02 号文件 RadixAttention 埋伏笔

**是什么:**
```python
def block_hash(tokens: Tuple[int, ...]) -> str:
    return hashlib.sha1(bytes(repr(tokens), "utf-8")).hexdigest()[:16]


@dataclass
class PrefixCache:
    block_size: int = 16
    cap: int = 256
    table: Dict[str, int] = field(default_factory=dict)     # hash -> phys_id
    hits: int = 0
    misses: int = 0
    lru: List[str] = field(default_factory=list)

    def mount(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """Return (phys_block_ids, hit_token_count) for a prompt prefix."""
        ...
        for i in range(0, len(token_ids), self.block_size):
            blk = tuple(token_ids[i : i + self.block_size])
            if len(blk) < self.block_size:
                break       # partial trailing block never cached
            h = block_hash(blk)
            if h in self.table:
                ... self.hits += 1 ...
            else:
                if len(self.table) >= self.cap:
                    evict = self.lru.pop(0)
                    self.table.pop(evict, None)
                ... self.misses += 1 ...
```
(`prefix_cache.py:17-57`,节选)

**一句话:** `PrefixCache` 把 prompt 按固定 `block_size` 切块、每块算一个 SHA1 hash,全局维护一张"hash → 物理块 id"的表——精确匹配(整块 token 完全相同)才能命中,命中就跳过这一块的 prefill、直接复用已有物理块;这是"整条前缀完全相同才能共享"的**平面**版本,和 02 号文件 RadixAttention 用 radix tree 做"部分前缀也能共享"的树状版本,是同一个问题的两种粒度不同的解法。

**底层机制/为什么这样设计:** 从最笨的想法讲起——真实 LLM 服务里,同一个 system prompt 会被成千上万个不同用户请求共享(每个请求的 system message 一字不差),如果每个请求都从头 prefill 这部分,是纯粹的重复计算。`PrefixCache.mount()` 的做法是把输入切成 `block_size` 大小的块,对每一块的 token 序列算 hash;如果这个 hash 之前出现过(说明有别的请求写过完全相同的这一块 token),直接复用之前分配的物理块 id,不需要重新计算这块的 K/V;如果没出现过,才真正分配新物理块、记录到 hash 表里。这里有个容易忽视但很重要的边界条件:`if len(blk) < self.block_size: break`——一个 prompt 末尾凑不满一整块的"零头"永远不参与缓存(直接跳出循环,不再处理后面哪怕是完整的块也不管,因为末尾不满一块通常意味着这是这个请求独有的、不太可能被复用的部分,比如每个用户各自的问题文本)。`cap` 限制了 hash 表最多缓存多少个不同的块,超过上限时按 LRU(最近最少使用)策略淘汰最久没被访问的块(`self.lru.pop(0)` 弹出列表最前面、也就是最久没被 touch 过的那个 hash)。这套机制能够工作的前提,恰恰是知识点 3 `PagedAttention` 已经把 KV 拆成了大小统一、可以独立寻址的物理块——如果没有分页,"复用某一块"这个操作没有意义(因为 KV 是按请求整块连续存放的,没法只共享其中一段)。`block_hash()` 用 `repr(tokens)` 再做 SHA1,保证只有 token id **序列完全相同、顺序也相同**才会撞出同一个 hash——这也解释了为什么这是"平面"匹配:两个 prompt 只要在某一个 block 边界之前有一个 token 不一样,从那个 block 起两边的 hash 就完全不同,不会有任何部分匹配的余地,哪怕后面 99% 的 token 其实是一样的。

**AI 研究场景:** L07 给出的命中率数字很说明问题:单一 system prompt 服务命中率能到 95%,多 system prompt(比如一个平台同时服务多个不同产品线,各自有不同的系统提示)降到 60%,agent 多轮对话(每一轮都在前几轮的基础上继续)大概 70%,通用聊天(几乎每次输入都不一样)只有 10-20%——这说明 Prefix Caching 的收益高度依赖"请求之间到底有多少共享前缀"这个业务特征,不是无脑开了就一定有大收益。这正是本知识点要埋的伏笔:平面 hash 版本要求"从头开始完全一致的一段"才能命中,如果两个请求的 system prompt 几乎一样、只有中间插了一个可变的用户名占位符,平面版本会因为这一个 token 的差异导致后面所有 block 全部退化成 miss;02 号文件的 RadixAttention 用 radix tree 记录"公共前缀到哪里分叉",能在分叉点之前的部分依然享受共享,分叉点之后才重新计算——这是从"整块匹配"进化到"逐 token 级别的最长公共前缀匹配",本知识点先把"为什么需要前缀共享""共享的物理机制是什么"讲清楚,分叉粒度的优化留给 02 号文件展开。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/inference-engine-core/src")
from prefix_cache import PrefixCache, block_hash

cache = PrefixCache(block_size=8, cap=16)
sys_prompt = list(range(32))  # 恰好 4 个 block 的共享系统提示
for q in range(15):
    ids, hit_tokens = cache.mount(sys_prompt + [1000 + q, 1001 + q])  # 末尾 2 个各自不同的 token
    if q == 0:
        assert hit_tokens == 0  # 第一个请求：还没有任何缓存，4 个 block 全 miss
    else:
        assert hit_tokens == 32  # 后续每个请求都命中全部 4 个共享的 system-prompt block

assert cache.hits == 4 * 14        # 14 个重复请求 * 每个命中 4 个共享 block
assert cache.misses == 4           # 只有第一个请求的 4 个 block 是真正的新写入
assert cache.hit_rate > 0.7        # 方向上匹配 L07 "单一 system prompt 服务 ~95%" 的结论
assert abs(cache.hit_rate - 4 * 14 / (4 * 14 + 4)) < 1e-9

# 末尾 2 个 token 永远凑不满一个 block（block_size=8）——“零头永不缓存”，表大小恒为 4
assert len(cache.table) == 4

assert block_hash((1, 2, 3)) == block_hash((1, 2, 3))
assert block_hash((1, 2, 3)) != block_hash((3, 2, 1))  # 顺序也参与哈希，不只是集合成员

# --- LRU 淘汰：cap=3，5 个各自独立的单 block 前缀 -> 只有最近 3 个能存活 ---
lru_cache = PrefixCache(block_size=4, cap=3)
for i in range(5):
    lru_cache.mount(list(range(i * 10, i * 10 + 4)))
assert len(lru_cache.table) == 3
assert lru_cache.misses == 5 and lru_cache.hits == 0

# 重新挂载最早被淘汰的那个前缀（block 0）：必须是新 miss，不是 hit
_, hit_tokens_after_evict = lru_cache.mount(list(range(0, 4)))
assert hit_tokens_after_evict == 0
assert lru_cache.misses == 6

# 重新挂载最近还在使用的前缀（block 4，仍然驻留）：必须是 hit
_, hit_tokens_recent = lru_cache.mount(list(range(40, 44)))
assert hit_tokens_recent == 4
```

**实测(`.venv` 真跑):** 15 次挂载(4-block 共享前缀 + 各自不同的 2-token 尾巴)后,`cache.hit_rate=0.9333`,`hits=56`(`4×14`)、`misses=4`;哈希表大小恒为 `4`,验证了"凑不满一块的尾巴永不缓存"。LRU 场景(`cap=3`,5 个各自独立的单 block 前缀)里,重新挂载最早被淘汰的前缀得到 `hit_tokens=0`(新 miss),重新挂载仍驻留的最近前缀得到 `hit_tokens=4`(命中)——两者行为精确相反,验证了 LRU 淘汰策略确实按访问时间生效,不是随机或者 FIFO。

**面试怎么问 + 追问链:**
- **Q:** "Prefix Caching 的平面 hash 版本,命中的前提条件是什么?" —— 期望明确说"从 prompt 开头起,按 block_size 切出的某一个 block,必须和之前已经缓存过的某个 block 的 token 序列**逐个完全相同**(包括顺序),才能命中;哪怕只差一个 token,从那个 block 起后面全部退化成 miss"。
- **追问 1:** "为什么末尾凑不满一个 block 的部分永远不缓存?这是不是一种浪费?" —— 期望说出"这部分通常是用户各自输入的、大概率不会被复用的内容(比如具体问题文本),给它分配缓存 slot 且几乎不可能命中,不如把这部分的 cap 配额留给真正高频复用的完整 block;这是一个基于'尾部大概率是独有内容'的工程假设,不是所有场景都成立,但对'共享 system prompt + 各自问题'这个最常见的模式是合理的"。
- **追问 2(埋伏笔,考察是否理解局限性):** "如果两个请求的 system prompt 几乎一样,只是中间插入了一个用户名变量(比如'你好,张三'和'你好,李四'),平面 hash 版本还能命中多少?" —— 期望说出"从插入用户名的那个 block 开始,后面所有 block 的 hash 全部不同,即使 95% 的内容完全一样,也只能命中'用户名之前'那几个 block,插入点之后全部退化成 miss——这正是平面 hash 版本的结构性局限,需要 radix tree 这种能表达'公共前缀到哪里分叉'的结构才能更精细地处理"。
- **追问 3:** "LRU 淘汰策略在这里具体是怎么实现的,有没有更好的替代策略?" —— 期望说出"这里用一个 Python list 模拟 LRU:命中时把对应 hash 移到列表末尾,淘汰时弹出列表最前面的;真实系统可能会用带时间戳的更高效结构(比如双向链表+哈希表实现 O(1) 的 LRU),也可能结合 refcount(正在被使用的 block 不能被淘汰,这是知识点 3 `PagedKvPool.refcount` 机制要解决的问题,这里的教学版 `PrefixCache` 没有和 `refcount` 打通)"。

**常见坑:** 把"hit_rate 高"直接等同于"这个引擎一定跑得快"——命中率只反映了"省了多少次重复 prefill 计算",如果业务场景本身请求之间高度独立(通用聊天场景,10-20% 命中率),开 Prefix Caching 的额外 hash 计算和表维护开销可能得不偿失,是否值得开取决于业务的前缀共享特征,不是无脑收益。另一个坑是把这里的 `PrefixCache` 和知识点 3 的 `PagedKvPool.refcount` 机制想象成已经打通的——本知识点的教学实现是分开的两个独立类,真实生产系统(vLLM `--enable-prefix-caching`)会把两者整合到一起(缓存的 block 也纳入 refcount 管理,避免被正在使用的请求依赖的 block 被 LRU 误淘汰),但这里为了教学清晰把两者拆开演示,不能想当然认为这两个类在这份代码里已经互相感知对方。

---

## 8. 调度策略(`scheduling_policies.py`,L08)—— FCFS/SJF/priority 三种 picker,抢占是 lecture-only

**是什么:**
```python
def fcfs_picker(pending: Deque[Request]) -> Request:
    return pending.popleft()


def sjf_picker(pending: Deque[Request]) -> Request:
    """Shortest-job-first by total expected length."""
    best_i = min(range(len(pending)), key=lambda i: len(pending[i].prompt_ids) + pending[i].max_new_tokens)
    best = pending[best_i]
    del pending[best_i]
    return best


def priority_picker(pending: Deque[Request]) -> Request:
    """Picks highest priority (stored as attribute `priority`, default 0)."""
    best_i = max(range(len(pending)), key=lambda i: getattr(pending[i], "priority", 0))
    best = pending[best_i]
    del pending[best_i]
    return best


PICKERS = {"fcfs": fcfs_picker, "sjf": sjf_picker, "priority": priority_picker}
```
(`scheduling_policies.py:9-33`,节选)

**一句话:** 三个 picker 函数从同一个 `pending` 双端队列里"挑下一个该处理谁",区别只在挑选规则(先到先得 / 预期总长度最短优先 / `priority` 属性最高优先),`schedule_order()` 是把某个 policy 反复应用到底、导出一个完整调度顺序的测试工具;真正的"抢占"(preemption,已经在跑的请求被中途打断)在这份源码里完全没有实现,只存在于 L08 的 lecture 叙述里。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 5 的 `Engine._can_admit` 只回答"这个请求能不能被放进来",没有回答"如果 `pending` 里同时排着好几个请求,该先放哪个进来"——这正是调度策略要解决的问题。`fcfs_picker` 最简单,谁先来谁先处理,不看任何请求自身的特征;`sjf_picker`(shortest-job-first)反过来,主动挑"预期总长度(`prompt_ids` 长度 + `max_new_tokens`)最短"的请求优先处理,这是排队论里经典的"最短作业优先能最小化平均等待时间"结论的直接应用——代价是长请求可能被无限期地往后排(如果短请求源源不断地来)。`priority_picker` 引入外部业务信号(`getattr(pending[i], "priority", 0)`,默认 0,意味着没有显式设置优先级的请求被当作最低优先级处理),适合多租户场景(比如付费用户优先于免费用户)。三者共享同一个函数签名(`Deque[Request] -> Request`),这是一个典型的策略模式(strategy pattern):`PICKERS` 字典把策略名字映射到具体函数,调用方只需要知道"策略名",不需要关心具体挑选逻辑怎么实现,想加一种新策略只需要再写一个同签名的函数、注册进字典。这里有一个容易被忽视的实现细节:`fcfs_picker`/`sjf_picker`/`priority_picker` 三者都会**原地修改**传入的 `pending`(要么 `popleft()`,要么 `del pending[best_i]`)——这意味着 `schedule_order()` 必须自己在内部用 `deque(reqs)` 拷贝一份全新的队列,不能直接对调用方传入的原始列表反复调用 picker,否则每调一次 `schedule_order` 原始数据就会被吃掉一部分,第二次调用就会得到错误结果。真正的"抢占"(vLLM 支持 recompute 或 swap-to-CPU 两种)是一个完全不同维度的能力——它处理的是"一个已经在 `running` 里跑着的请求,中途因为显存不够被迫暂停、让位给别的请求"，而这份源码的三个 picker 只处理"`pending` 里还没开始跑的请求,先挑谁"，从未涉及"打断一个正在运行的请求"这件事，`scheduling_policies.py` 全文搜索不到任何和"preempt"相关的代码。

**AI 研究场景:** 调度策略的选择本质上是在业务目标之间做取舍:FCFS 公平但不管效率,SJF 能降低平均等待时间但对长请求不友好(极端情况下可能饥饿),priority 能满足多租户 SLA 但需要额外的治理机制防止低优先级请求被无限期饿死。vLLM 真实的 `_schedule()` 三阶段(`_schedule_running` → `_schedule_swapped` → `_schedule_prefills`,见 L12)本质上是把这几种朴素策略和显存约束、抢占机制组合成一套更完整的调度器——理解这三个最基础的 picker,是理解任何生产调度器复杂实现的地基,面试里被问"设计一个推理服务的调度策略"时,这三种是最基本的备选方案,再往上叠加的都是复杂度和约束的组合。

**可运行例子:**
```python
import sys
from collections import deque
sys.path.insert(0, "learning/inference-engine-core/src")
from common import Request
from scheduling_policies import fcfs_picker, sjf_picker, priority_picker, schedule_order, PICKERS


def mk(rid, plen, mlen, prio=0):
    r = Request(rid=rid, prompt_ids=list(range(plen)), max_new_tokens=mlen)
    r.priority = prio
    return r


# 和 src/tests/test_policies_sampling.py 不同的一组实例化（不同 rid/长度/顺序）
reqs = [
    mk(10, plen=50, mlen=50, prio=2),   # 总长 100，中等优先级
    mk(11, plen=3,  mlen=3,  prio=0),   # 总长 6，最短，默认优先级
    mk(12, plen=200, mlen=1, prio=9),   # 总长 201，最长，最高优先级
    mk(13, plen=1,  mlen=1,  prio=0),   # 总长 2，比 11 更短
]

assert schedule_order(reqs, "fcfs") == [10, 11, 12, 13]          # 纯插入顺序，无视长度/优先级
assert schedule_order(reqs, "sjf") == [13, 11, 10, 12]            # 2 < 6 < 100 < 201
assert schedule_order(reqs, "priority") == [12, 10, 11, 13]       # 9 > 2 > 0==0（同分按先出现者）

assert set(PICKERS.keys()) == {"fcfs", "sjf", "priority"}
assert PICKERS["fcfs"] is fcfs_picker

# schedule_order() 必须在内部拷贝一份全新的 deque，不能直接吃掉调用方传入的 reqs
pending = deque(reqs)
first_call = schedule_order(reqs, "sjf")
assert list(pending) == reqs          # 原始 reqs 未被 schedule_order 消耗
second_call = schedule_order(reqs, "sjf")
assert first_call == second_call      # 可重复调用，结果一致

# 直接使用 picker：fcfs_picker/sjf_picker 确实会原地修改传入的 deque（真实队列语义）
q = deque(reqs)
picked = fcfs_picker(q)
assert picked.rid == 10
assert len(q) == 3 and 10 not in [r.rid for r in q]

# 抢占（preemption）不存在于这份源码里——L08 lecture 讨论的是 vLLM 的
# recompute/swap，这个模块只实现"pending 队列里先挑谁"的准入顺序
import inspect
src_text = inspect.getsource(sys.modules["scheduling_policies"])
assert "preempt" not in src_text.lower()
```

**实测(`.venv` 真跑):** 4 请求(总长分别为 100/6/201/2,优先级 2/0/9/0)在三种策略下的调度顺序精确为:`fcfs=[10,11,12,13]`(纯插入序)、`sjf=[13,11,10,12]`(2→6→100→201)、`priority=[12,10,11,13]`(9→2→0(先出现的 11)→0(13))。验证了 `schedule_order()` 内部对 `pending` 做了独立拷贝,两次调用互不影响、结果一致;而直接调用 `fcfs_picker(q)` 则会真实修改传入的 `q`(`len(q)` 从 4 变 3)。全文搜索确认 `scheduling_policies.py` 不含任何 `preempt` 相关代码。

**面试怎么问 + 追问链:**
- **Q:** "FCFS、SJF、priority 三种调度策略各自的优缺点是什么?" —— 期望说出"FCFS 最公平但不管效率;SJF 能降低平均等待时间,但长请求可能被持续饿死;priority 能满足业务优先级需求(比如付费用户优先),但需要额外机制防止低优先级请求无限期等待"。
- **追问 1(容易考出真假理解):** "这个模块支持抢占吗,比如一个正在 decode 的请求能不能被临时换出去让给更高优先级的新请求?" —— 期望明确说"不支持,这三个 picker 只处理'pending 队列里还没开始跑的请求先挑谁进来',不涉及'打断一个已经在 running 里跑着的请求',这是完全不同的两件事;真实的抢占(recompute 或 swap-to-CPU)是 L08 lecture 讨论的 vLLM 特性,这份 `scheduling_policies.py` 源码里没有对应实现"。
- **追问 2:** "SJF 策略需要提前知道请求的'预期总长度',但用户没有明确说要生成多少个 token,这个长度从哪来?" —— 期望观察到 `sjf_picker` 用的是 `len(prompt_ids) + max_new_tokens`,`max_new_tokens` 是调用方(客户端)自己声明的生成上限,不是真实会用到的确切长度(真实生成经常提前遇到 eos 就结束,用不满这个上限)——所以 SJF 排序依据的其实是一个"悲观估计的上界",不是真实长度,这个估计不准的时候,排序结果的最优性也会打折扣。
- **追问 3(考察工程直觉):** "如果我想要'高优先级请求优先,但同优先级内部按 FCFS'这种复合策略,现有的 `priority_picker` 够用吗?" —— 期望观察到 `priority_picker` 用 `max(..., key=lambda i: getattr(...))` 在多个最高优先级候选之间"选第一个遇到的"(Python `max` 平局取先出现的),这恰好天然实现了"同优先级内 FCFS"这个语义(因为 `pending` 本身是按到达顺序排列的双端队列),但这是 `max()` 平局行为的副产品,不是显式设计的复合策略,如果 `pending` 的顺序被其他逻辑打乱,这个"同优先级内 FCFS"的性质就不再成立。

**常见坑:** 把 roadmap/lecture 提到的"抢占"当成这份 `scheduling_policies.py` 已经实现的功能去引用——这是本知识点特意澄清的一点,三个 picker 只是"准入顺序"层面的策略,抢占是完全独立的、这份源码没有涉及的能力。另一个坑是直接对调用方传入的原始列表反复调用某个 picker 函数做实验,却忘了 `fcfs_picker`/`sjf_picker`/`priority_picker` 都会原地消耗传入的 deque——如果不清楚这一点,写测试代码时容易出现"第一次调用结果正确,第二次调用结果对不上"的诡异 bug,根源就是数据已经在第一次调用时被吃掉了。

---

## 9. CUDA Graphs + Attention Backends(L09+L10,`attention_naive.py`)—— capture/replay 在这台 Windows 机器上真的能跑,还能跑出真实两位数倍速比

**是什么:**
```python
# L09 lecture 给出的机制骨架（源码本身留空,教学 stub,不依赖 cuda）
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = model(static_input)

static_input.copy_(real_input)
graph.replay()
out_real = out.clone()
```
(`lectures/09-cuda-graphs.md`,第 2 节机制骨架,`src/` 无对应可运行文件——README 明确写"教学留 stub,不依赖 cuda")

```python
def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """q,k,v shape [B, H, S, D].  Returns [B, H, S, D]."""
    ...

def has_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False
```
(`attention_naive.py:12-22`、`:25-30`)

**一句话:** CUDA Graph 是"录制一段 GPU 操作序列(capture),之后反复原样重放(replay),省掉每次都要经过 Python/CPU 重新下发每个 kernel 的开销"这一个机制,`attention_naive.py` 则是"没有 FlashAttention/FlashInfer 这些专用库时退回到的、教学用的纯 torch attention 基线",两者被 roadmap 合并成一个知识点是因为都属于"这台机器上装不了/没有专用重型依赖时的现实约束"这同一条主线——但本知识点最值得记住的是一个反直觉的发现:CUDA Graph 的 capture/replay 机制本身在这台 Windows 机器上是**可以真实跑通、而且能跑出真实两位数倍速比**的,此前"Windows 上无法真正验证"的预设只对"完整的 vLLM 引擎 + FlashAttention/FlashInfer"这个组合成立,不对"裸的 `torch.cuda.CUDAGraph` API"成立。

**底层机制/为什么这样设计:** 从最笨的想法讲起——decode 阶段每一步都要发起几十甚至上百次 kernel launch(每层 attention、每层 FFN 各自拆成好几个 kernel),而每次 kernel launch 即便计算本身很快,CPU 侧准备参数、下发指令给 GPU 驱动也有几微秒到十几微秒的固定开销;当单个 kernel 的实际计算时间(尤其是 decode 阶段 batch 很小的情况下)已经逼近这个固定开销的量级时,大部分时间实际花在"等 CPU 一个个下发指令"而不是"GPU 真正在算"。CUDA Graph 的思路是:如果一段 kernel 序列每次执行的操作完全一样、只是输入数据不同,那就没必要每次都重新走一遍"CPU 准备指令→下发→GPU 执行"的完整流程——先完整录制一遍这个序列(`torch.cuda.graph(g)` 上下文管理器),之后只要把新数据拷贝进录制时用的那块固定内存(`static_input.copy_(real_input)`),一次 `g.replay()` 调用就能让 GPU 按录制好的顺序重新跑一遍全部 kernel,CPU 侧几乎不需要重新准备任何东西。这里必须澄清一个容易和"编译"混淆的概念:裸 `torch.cuda.CUDAGraph` 录制的是**已经编译好的 kernel 的调用序列**(cuBLAS/cuDNN/ATen 内置算子本来就是预编译的二进制),不涉及现场生成/编译任何新代码——这和 Triton JIT 编译一个新 kernel、或者 TensorRT-LLM 需要 `nvcc` 现场构建插件是完全不同的两件事。这台机器(Windows 11,RTX 3080 Ti Laptop,torch 2.11.0+cu128)上,本知识点独立验证了这一点:一个 32 层的小型全连接网络(模拟 decode 阶段"很多层、每层都是小 kernel"的密集 launch 场景),`batch=1` 时用 `torch.cuda.Event` 精确计时,eager 模式(每次都重新走 Python 调度)每次前向约 3.7-4.0ms,capture 之后 `replay()` 每次约 0.17ms,真实倍速比在 21-23 倍区间(多次独立重跑,数字有波动但量级稳定);一个更浅但 batch 更大的网络(3 层、`batch=8`)也测出 16-18 倍的真实加速。这两个数字都远超roadmap原始预期"这里讲清楚机制原理,不能现场跑出真实加速比数字"这条预设——本知识点在撰写过程中先后用两种计时方法测过:**第一次**用 Python 标准库 `time.perf_counter()` 分别给 eager 循环和 replay 循环计时,得到的结果是"replay 反而比 eager 慢"(约 0.84 倍),这是一个测量方法的陷阱——CUDA kernel 是异步下发的,`perf_counter()` 包裹的 eager 循环里,CPU 可能早就把一堆 kernel 提交进队列然后继续往下跑了,只有循环结束时的 `synchronize()` 才真正等待 GPU 完成,这种计时方式会让 GPU 的执行时间和 CPU 的调度时间产生难以区分的重叠,给出误导性结论;**第二次**换成 `torch.cuda.Event(enable_timing=True)` 做 GPU 侧精确计时(在 GPU 的执行流里插入时间戳,而不是从 CPU 侧用墙钟去猜),才测出上面这组真实、稳定、方向正确的加速比。这个"先用错误的计时方法得出反直觉结论、再用正确方法纠正"的过程本身,是一堂关于"怎么给异步 GPU 代码计时"的现场教训,比直接给出正确数字更有参考价值。真正跑不出来的,是 L09/L10 lecture 里引用的"vLLM 生产级 7B 模型 decode iter 从 8ms 降到 3ms"这类数字——那需要真实的 vLLM 引擎 + 真实 7B 权重 + 通常还需要 FlashAttention/FlashInfer 这类专用 attention kernel,这台机器上 `vllm`/`flash_attn`/`flashinfer` 三个包全部 `ModuleNotFoundError`(已实测确认),所以能验证的是"CUDA Graph 机制本身在这台机器上真实生效、真实有巨大加速"这个更底层的事实,不能验证"某个具体生产系统在这台机器上到底能跑多快"这个更上层的事实——两者是不同粒度的问题,不能因为后者验证不了就连带认为前者也验证不了。`attention_naive.py` 这边,`has_flash_attn()`/`has_flashinfer()` 就是两个纯粹的 `try/except ImportError` 探测函数,`naive_attention()` 是标准的"手写 QKᵀ→mask→softmax→乘V"实现,存在的意义是给"没有专用库时如何保证代码依然能跑、依然数值正确"提供一个诚实的兜底路径,不是追求性能。

**AI 研究场景:** CUDA Graph 对"batch 小、层数多、单层计算量小"的场景收益最大——这恰好是 LLM decode 阶段(尤其是低并发在线服务场景)的典型特征,这也是为什么 vLLM/SGLang/TensorRT-LLM 都默认在 decode 阶段开启 CUDA Graph(vLLM 用 bucketing,给不同 batch size 各录一份 graph)。反过来 prefill 阶段因为序列长度经常变化(不同请求 prompt 长度不同),形状不固定,天然不适合用一份固定的录制序列去 replay,这也是为什么 L09 明确写"只能 decode 阶段用,prefill 不能用"。Attention Backend 的选型(FlashAttention v2/v3、FlashInfer、naive)则是另一个独立的性能杠杆:FlashInfer 对 paged KV 是原生支持的(不需要先 gather 成连续张量),这和知识点 4 讨论的"kernel 层要不要感知分页结构"是同一个话题在不同 backend 上的具体体现——这台机器因为没装 FlashAttention/FlashInfer,knowledge point 4 的 `paged_attention_torch` 和这里的 `naive_attention` 是同一类"教学期替代实现",两者共享"没有专用库时先保证正确性"这条设计哲学。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from attention_naive import naive_attention, has_flash_attn, has_flashinfer

assert torch.cuda.is_available()
device = "cuda"

# --- Part A: CUDA Graph capture + replay 在这台 Windows 机器上真的能跑（反直觉发现）---
torch.manual_seed(0)


class ManyTinyLayers(torch.nn.Module):
    def __init__(self, d=128, n_layers=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, x):
        for l in self.layers:
            x = torch.relu(l(x))
        return x


model = ManyTinyLayers().to(device).eval()
static_input = torch.randn(1, 128, device=device)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(5):
        model(static_input)
torch.cuda.current_stream().wait_stream(s)

graph = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(graph):
    static_out = model(static_input)

# 正确性：replay 后必须和对同样数据重新做一次 eager forward 完全一致
real_input = torch.randn(1, 128, device=device)
static_input.copy_(real_input)
graph.replay()
torch.cuda.synchronize()
with torch.no_grad():
    eager_out = model(real_input)
assert torch.allclose(static_out, eager_out, atol=1e-5)


def time_ms(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


with torch.no_grad():
    t_eager_deep = time_ms(lambda: model(static_input))
t_graph_deep = time_ms(lambda: graph.replay())
assert t_graph_deep < t_eager_deep * 0.5   # 真实、大幅度的加速，不是误差范围内的抖动
speedup_deep = t_eager_deep / t_graph_deep

# --- Part A2: 换一个更浅但 batch 更大的网络，收益方向依然成立 ---
shallow = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 256)).to(device).eval()
wide_input = torch.randn(8, 256, device=device)
s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2), torch.no_grad():
    for _ in range(5):
        shallow(wide_input)
torch.cuda.current_stream().wait_stream(s2)
graph2 = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(graph2):
    static_out2 = shallow(wide_input)
with torch.no_grad():
    t_eager_shallow = time_ms(lambda: shallow(wide_input))
t_graph_shallow = time_ms(lambda: graph2.replay())
assert t_graph_shallow < t_eager_shallow  # 依然是加速，只是幅度和 batch/层数配置有关

# --- Part B: bucketing（L09 第 4 节）——把一个请求的 batch 向上取整到最近的“已录制”桶 ---
BUCKETS = [1, 2, 4, 8, 16, 32]


def round_to_bucket(batch: int, buckets=BUCKETS) -> int:
    for b in buckets:
        if batch <= b:
            return b
    raise ValueError(f"batch {batch} exceeds max captured bucket {buckets[-1]}")


assert round_to_bucket(1) == 1
assert round_to_bucket(3) == 4
assert round_to_bucket(9) == 16
try:
    round_to_bucket(33)
    assert False, "should raise"
except ValueError as e:
    assert "exceeds max captured bucket" in str(e)

# --- Part C: attention backend 探测——这台机器的真实状态 ---
assert has_flash_attn() is False
assert has_flashinfer() is False

torch.manual_seed(0)
q = torch.randn(1, 4, 16, 32, device=device)
k = torch.randn(1, 4, 16, 32, device=device)
v = torch.randn(1, 4, 16, 32, device=device)
out_causal = naive_attention(q, k, v, causal=True)
out_full = naive_attention(q, k, v, causal=False)
assert out_causal.shape == (1, 4, 16, 32)
assert not torch.allclose(out_causal[:, :, 0], out_full[:, :, 0], atol=1e-3)  # causal 遮住未来，结果应不同
```

**实测(`.venv` 真跑):** Part A(32 层、`batch=1`,用 `torch.cuda.Event` 精确计时):`eager≈3.7-4.0ms`、`graph replay≈0.17ms`,真实倍速比 `21-23x`(3 次独立重跑分别是 23.11x/23.12x/21.55x);Part A2(3 层、`batch=8`)加速比 `16-18x` 区间。`static_out` 与对同一份数据重新做 eager forward 的结果 `torch.allclose`(容差 `1e-5`),证明 replay 的正确性不只是"跑得动",数值也对得上。**方法论教训**:同一段代码,第一次尝试用 `time.perf_counter()` 分别包裹 eager 循环和 replay 循环计时,由于 CUDA kernel 异步下发、CPU 侧循环本身很快就能把指令排进队列继续往下走,给出的结果是 replay **反而比 eager 慢**(约 0.84 倍)——这是一个错误但值得记录的中间结果,根源是 wall-clock 计时被异步执行掩盖了真实的 GPU 侧耗时;换成 `torch.cuda.Event` 后测出的才是真实、稳定、方向正确的加速比。`has_flash_attn()`/`has_flashinfer()` 均为 `False`(真实 `ModuleNotFoundError`),`naive_attention` 的 causal/non-causal 结果在第 0 行(只能看到自己)确实不同,验证了因果遮罩生效。

**面试怎么问 + 追问链:**
- **Q:** "CUDA Graph 为什么能加速 LLM 的 decode 阶段,机制是什么?" —— 期望说出"decode 阶段每步会发起大量小 kernel,CPU 侧下发指令的固定开销在 kernel 本身很快时占比很高;CUDA Graph 先录制一遍完整的 kernel 调用序列,之后只需要更新输入数据、一次 replay 调用就能重放整个序列,不需要 CPU 重新逐个下发,从而消除大部分 launch overhead"。
- **追问 1(核心陷阱,直接考察是否只是背了一个'Windows 不能用'的结论):** "有没有可能在 Windows 机器上真正验证 CUDA Graph 的加速效果?" —— 期望能区分"裸 `torch.cuda.CUDAGraph` API 只是录制/重放已编译好的 kernel 调用序列,不涉及现场编译,在 Windows 上完全可以工作"和"完整的 vLLM 引擎 + FlashAttention/FlashInfer 这些专用 kernel 库在 Windows 上大概率装不上"是两个不同粒度的问题,后者装不上不代表前者的机制本身验证不了——这道题专门筛"人云亦云复述预设"和"自己动手验证边界在哪"两类候选人。
- **追问 2(计时方法论,极容易被问倒):** "用 `time.time()` 或 `time.perf_counter()` 包裹一段 GPU 代码计时,可能会有什么问题?" —— 期望说出"CUDA 调用默认是异步的,Python 侧的 wall-clock 计时可能只测到了'CPU 把指令扔进队列'的时间,而不是'GPU 真正执行完'的时间,除非显式 `torch.cuda.synchronize()` 才能保证测的是真实完成时间;即便加了 synchronize,循环里每次调用之间的异步重叠也可能让最终数字产生偏差,更可靠的做法是用 `torch.cuda.Event` 在 GPU 执行流内部打时间戳",最好能提到本知识点自己就因为用错方法得到过一次误导性结论(replay 反而变慢),后来换 `torch.cuda.Event` 才纠正过来。
- **追问 3:** "CUDA Graph 为什么只能用在 decode 阶段,不能用在 prefill 阶段?" —— 期望说出"CUDA Graph 录制的是一段固定 shape、固定控制流的 kernel 序列,replay 时只能替换输入数据,不能改变执行的 kernel 序列本身;prefill 阶段不同请求的 prompt 长度经常不同,形状不固定,没法用同一份录制去覆盖所有可能的长度,vLLM 的应对方式是只在 decode 阶段用 CUDA Graph,并且用 bucketing(给若干个离散的 batch size 各录一份 graph,实际请求向上取整到最近的桶)去部分缓解 decode 阶段 batch size 也会变化的问题"。

**常见坑:** 不加验证就全盘接受"某个平台/操作系统上做不到某件事"这类预设——本知识点是一个具体的反例:实际测试表明裸 CUDA Graph capture/replay 在这台 Windows 机器上完全可用,且能给出真实、大幅度的加速比,此前的预设混淆了"底层机制能不能跑"和"某个具体重型依赖包能不能装上"这两件事。另一个坑是用不恰当的方式给异步 GPU 代码计时就直接下结论——第一次用 `perf_counter()` 得到"replay 更慢"的错误结论就是一个真实案例,任何看到"GPU 优化技巧实测没有效果"这类结论时,都应该先检查计时方法本身是否可靠,而不是急于反过来否定这个优化技巧本身。

---

## 10. Sampling 引擎(`sampling.py`,L11)—— temperature/top-k/top-p/min-p/repetition penalty

**是什么:**
```python
def top_k_top_p_mask(logits: torch.Tensor, top_k: int, top_p: float, min_p: float) -> torch.Tensor:
    """Return a boolean mask of tokens KEPT (True = candidate)."""
    keep = torch.ones_like(logits, dtype=torch.bool)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values.min()
        keep &= logits >= kth
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprob = torch.softmax(sorted_logits, dim=0).cumsum(0)
        cutoff = (cumprob - cumprob[0]) < top_p   # always keep top-1
        sorted_keep = cutoff | (torch.arange(len(logits)) == 0)
        nucleus_mask = torch.zeros_like(keep)
        nucleus_mask[sorted_idx] = sorted_keep
        keep &= nucleus_mask
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=0)
        keep &= probs >= min_p * probs.max()
    return keep
```
(`sampling.py:33-50`)

```python
def apply_repetition_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    """Decay logits of tokens already produced (presence-style)."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 0, prev_tokens)
    score = torch.where(score > 0, score / penalty, score * penalty)
    out = logits.clone()
    out.scatter_(0, prev_tokens, score)
    return out
```
(`sampling.py:22-30`)

**一句话:** `top_k_top_p_mask` 把 top-k(硬截断到固定数量候选)、top-p/nucleus(动态截断到累积概率刚好覆盖 `p` 的候选集)、min-p(相对最高概率的比例阈值)三种独立的过滤条件用逻辑与(`keep &= ...`)依次收窄同一个候选集合,`apply_repetition_penalty` 则是对已经出现过的 token 做"正的往下压、负的往下拉得更负"的双分支惩罚,两者共同构成从 logits 到最终采样 token 的完整流水线。

**底层机制/为什么这样设计:** 从最笨的想法讲起——如果每次都直接对全部词表(几万个 token)的 softmax 概率分布做多项式采样(`torch.multinomial`),会经常采到概率虽然非零但极小、根本不合理的低质量 token(纯 greedy 又太保守、容易重复),所以几乎所有生产采样器都会先用某种规则"砍掉"候选集合里明显不该考虑的部分,再在剩下的候选里按概率采样。`top_k_top_p_mask` 的三个过滤条件是三种不同的"砍法":top-k 直接说"只保留概率最高的 k 个,不管它们的概率具体是多少"——`torch.topk(logits, top_k).values.min()` 取出第 k 大的 logit 值作为门槛,`keep &= logits >= kth` 保留所有不低于这个门槛的 token(如果有并列,可能保留多于 k 个);top-p 反过来,不固定候选数量,而是把 token 按概率从高到低排序,累加概率直到刚好达到 `p`,这个候选集合的大小会随分布的"陡峭程度"自适应变化(分布很集中时可能几个 token 就够了,分布很平时可能需要几十个);min-p 是 2024 年才出现的新方法,用"候选 token 的概率必须至少是当前最高概率的 `min_p` 倍"这个相对阈值做截断,好处是不需要像 top-p 那样排序整个词表,计算更便宜,行为上和 top-p 类似但对分布形状的适应方式不同。三个条件用 `keep &= ...` 依次收窄,意味着它们是**合取**关系:一个 token 必须同时满足开启的所有条件才能保留,不是"满足任意一个就行"的析取关系——这一点在同时开启 top_k 和 top_p 时很关键,最终候选集合是两者各自候选集合的交集,不是并集。`apply_repetition_penalty` 的双分支设计(`score > 0` 走除法、否则走乘法)是为了保证"惩罚"这个操作对正负 logit 都朝着"更不容易被选中"的方向起作用:如果统一用除法,一个本来是负数的 logit 除以 `penalty>1` 会变得**更接近 0**(反而更容易被选中,方向搞反了);统一用乘法,正数 logit 乘以 `penalty>1` 会变得更大(同样方向搞反)。用 `torch.where` 按符号分流,正数除、负数乘,才能保证不管原始 logit 是正是负,惩罚后都朝着"更不吸引"的方向移动。`sample_one` 里 `temperature<=0` 直接短路成 `argmax()`,完全跳过后面的 mask/softmax/multinomial 流程——这是因为温度趋近于 0 时 softmax 分布会退化成一个尖峰(概率质量全部集中在最大 logit 上),直接取 argmax 在数学上是这个极限的精确结果,没必要真的构造一个几乎全零的概率分布再去采样,徒增计算量和数值不稳定的风险。

**AI 研究场景:** 这套采样参数组合(temperature + top-k + top-p + repetition_penalty)是几乎所有 LLM API(OpenAI、Anthropic、开源推理引擎)共同暴露给用户的标准接口,本知识点强调"这里的实现在数学层面对齐 vLLM `Sampler` 的行为"——意味着理解这几个参数的精确语义,不只是理解这一个教学模块,而是理解整个行业的采样参数标准。min_p 是这几个参数里最新的一个(2024 年提出),它的动机是解决 top-p 在"分布本身很平"的场景下容易保留太多低质量候选、而在"分布本身很尖"的场景下又可能因为固定的 `p` 阈值保留过少候选这两类边界情况——用相对当前最高概率的比例做门槛,是一种更"感知分布形状"的截断方式。理解这几种方法的取舍,在实际调参时能帮助快速判断"生成结果过于随机/过于重复,应该调哪个参数"这类问题,是推理服务对外接口设计和内部实现两头都要用到的知识。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from sampling import SamplerConfig, sample_one, sample_batch, top_k_top_p_mask, apply_repetition_penalty

# --- greedy（temperature<=0）是纯 argmax，完全绕开 mask/softmax/multinomial 整套流程 ---
torch.manual_seed(7)
logits = torch.tensor([0.1, 5.0, -3.0, 2.0, 4.9])
cfg_greedy = SamplerConfig(temperature=0)
tok = sample_one(logits, torch.tensor([], dtype=torch.long), cfg_greedy)
assert tok == 1 == int(logits.argmax().item())

# --- top_k 精确保留 k 个 token（无并列时） ---
lg = torch.tensor([2.0, 8.0, 1.0, 6.0, 4.0, 7.0])
mask_k3 = top_k_top_p_mask(lg, top_k=3, top_p=1.0, min_p=0.0)
assert mask_k3.sum().item() == 3
assert mask_k3.tolist() == [False, True, False, True, False, True]  # 下标 1,3,5 对应 8,6,7

# --- top_p（nucleus）：即使 p 极小，也始终保留 top-1 ---
mask_p_tiny = top_k_top_p_mask(lg, top_k=0, top_p=0.01, min_p=0.0)
assert mask_p_tiny.sum().item() == 1
assert mask_p_tiny.tolist()[1] is True

# --- min_p：相对最高概率的比例阈值，不是绝对概率阈值 ---
uniform_ish = torch.tensor([0.0, 0.0, 0.0, 10.0])  # 一个远大于其余的离群 logit
mask_minp = top_k_top_p_mask(uniform_ish, top_k=0, top_p=1.0, min_p=0.5)
probs = torch.softmax(uniform_ish, dim=0)
assert mask_minp.tolist() == (probs >= 0.5 * probs.max()).tolist()
assert mask_minp.sum().item() == 1  # 离群值主导 softmax，其余 3 个全部被砍掉

# --- top_k 和 top_p 是合取（AND），不是析取（OR） ---
combo = top_k_top_p_mask(lg, top_k=4, top_p=0.3, min_p=0.0)
assert combo.sum().item() <= 4
assert combo[1].item() is True  # 下标 1（值 8.0，唯一最高）在两个条件下都存活

# --- repetition penalty：验证 torch.where(score>0, /penalty, *penalty) 的两条分支 ---
rp_logits = torch.tensor([-2.0, 3.0, 0.5])
out = apply_repetition_penalty(rp_logits, torch.tensor([0, 1]), penalty=2.0)
assert out[0].item() == -4.0   # 负分支：score<=0 -> score*penalty（-2*2=-4，更负，更不吸引）
assert out[1].item() == 1.5    # 正分支：score>0  -> score/penalty（3/2=1.5，更小，更不吸引）
assert out[2].item() == 0.5    # 未出现过的 token 不受影响
assert apply_repetition_penalty(rp_logits, torch.tensor([0, 1]), penalty=1.0) is rp_logits  # 快速路径

# --- sample_batch：每个请求独立的 config，互不干扰 ---
torch.manual_seed(3)
batch_logits = torch.randn(3, 20)
prev = [torch.tensor([0], dtype=torch.long) for _ in range(3)]
cfgs = [SamplerConfig(temperature=0), SamplerConfig(temperature=0), SamplerConfig(temperature=0.8, top_k=5)]
toks = sample_batch(batch_logits, prev, cfgs)
assert len(toks) == 3
assert toks[0] == int(batch_logits[0].argmax().item())
assert toks[1] == int(batch_logits[1].argmax().item())
```

**实测(`.venv` 真跑):** greedy 采样精确等于 `argmax`(token `1`)。`top_k=3` 精确保留 3 个 token(下标 `1,3,5`);`top_p=0.01` 无论多小都保留 top-1;`min_p=0.5` 在一个离群值主导的分布上只保留 `1` 个 token。`top_k=4` 和 `top_p=0.3` 同时开启时结果集合大小 `≤4`,且唯一最高 token 在两个条件下都存活,验证了合取关系。`repetition_penalty=2.0` 下,负 logit `-2.0` 变成 `-4.0`(乘法分支),正 logit `3.0` 变成 `1.5`(除法分支),未出现过的 `token 2` 保持 `0.5` 不变;`penalty=1.0` 时函数直接返回同一个对象(`is` 恒等,验证了快速路径)。`sample_batch` 对 3 个独立配置的请求给出独立结果,两个 greedy 请求精确等于各自的 `argmax`。

**面试怎么问 + 追问链:**
- **Q:** "top-k、top-p、min-p 三种采样截断方式的核心区别是什么?" —— 期望说出"top-k 固定保留数量最高的 k 个 token,不管概率具体分布;top-p 按累积概率动态决定候选数量,分布越集中候选越少;min-p 用相对最高概率的比例做阈值,不需要排序整个词表,计算更便宜"。
- **追问 1(容易被绕进去的细节):** "如果同时设置 `top_k=10` 和 `top_p=0.9`,最终候选集合是两者的并集还是交集?" —— 期望明确说"交集(合取),`top_k_top_p_mask` 内部是用 `keep &= ...` 逐个收窄同一个 mask,一个 token 必须同时满足所有开启的条件才会被保留"。
- **追问 2(重复惩罚的符号陷阱):** "`apply_repetition_penalty` 为什么要用 `torch.where` 区分正负 logit,统一用除法或者统一用乘法不行吗?" —— 期望说出"除法对负数会让它更接近 0(变得更容易被选中,方向和'惩罚'的意图相反);乘法对正数会让它变得更大(同样方向搞反);只有'正数除、负数乘'才能保证不管原始符号如何,惩罚后都朝着'logit 更小、更不容易被选中'的方向移动"，能现场举一个具体反例(比如负数除法的例子)验证方向搞反的候选人更加分。
- **追问 3:** "为什么 `temperature<=0` 要单独处理成 argmax,而不是让 `temperature` 趋近于一个很小的正数,走正常的 softmax+multinomial 流程?" —— 期望说出"temperature 趋近于 0 时 softmax 之前的 `logits/temperature` 会趋于无穷大,数值上容易溢出或者产生 NaN/Inf,而且这个极限的数学结果就是 argmax,直接短路成 argmax 既避免了数值稳定性问题,又省了一次不必要的 softmax+multinomial 计算"。

**常见坑:** 把 min_p 和 top_p 记混,以为两者是同一种机制的不同名字——两者的阈值基准完全不同:top_p 基于**累积概率**(排序后从高到低累加到 `p`),min_p 基于**相对最高概率的比例**(不需要排序,只需要知道最大概率是多少);在分布非常尖锐(一个 token 概率接近 1)的场景下,两者的候选集合可能都退化到只有 1 个 token,容易让人误以为它们在所有场景下都等价,但在分布较平的场景下两者的候选集合规模可以有明显差异。另一个坑是以为 `apply_repetition_penalty` 会对所有出现过的 token 一视同仁地"降低概率"——精确的机制是对 logit 做符号相关的缩放,如果不理解双分支的方向,容易在自己实现类似功能时对负数 logit 用错运算符,导致惩罚方向反而变成了鼓励。

---

## 11. Capstone:mini-vLLM 复刻(`mini_vllm.py`,L12+L13)—— ~175 行拼出 vLLM 五大件骨架

**是什么:**
```python
@dataclass
class MiniEngine:
    pool: PagedKvPool
    sampler_cfg: SamplerConfig = field(default_factory=SamplerConfig)
    forward_fn: Optional[Callable[[List[Request]], torch.Tensor]] = None
    max_running: int = 8
    ...
    prefix_cache: Optional[PrefixCache] = None
    vocab_size: int = 50257

    def _can_admit(self, req: Request) -> bool:
        needed = (len(req.prompt_ids) + req.max_new_tokens + self.pool.block_size - 1) // self.pool.block_size
        return (
            len(self.running) < self.max_running
            and self.pool.n_free() >= needed
        )

    def step(self) -> bool:
        # admission
        while self.pending and self._can_admit(self.pending[0]):
            r = self.pending.popleft()
            self.block_tables[r.rid] = self._make_table(r)
            self.running.append(r)
        ...
```
(`mini_vllm.py:30-56`、`:66-70`,节选)

**一句话:** `MiniEngine` 把知识点 3 的 `PagedKvPool`/`BlockTable`、知识点 5 的准入+迭代循环、知识点 10 的 `sample_one` 拼到同一个类里,`CASES`/`run_case`/`run_all` 提供 5 个覆盖不同瓶颈场景(短进短出、长进短出、短进长出、大 batch、单流)的对照用例——这是 L12 vLLM 源码导读(engine + scheduler + block_manager + attention backend + sampler 五大件)在教学规模下的真实落地,不是又一份和前面知识点无关的新内容。

**底层机制/为什么这样设计:** 从最笨的想法讲起——L12 拆解真实 vLLM 源码时列出五个核心模块(`engine/llm_engine.py`、`core/scheduler.py`、`core/block_manager.py`、`attention/backends/`、`model_executor/layers/sampler.py`),`mini_vllm.py` 的核心论点是"这五大件本质上没有那么复杂,可以用不到 200 行拼出一个骨架版本"——`MiniEngine.step()` 里能看到这五大件的教学对应:`self.pool`/`self.block_tables` 对应 `block_manager`(知识点 3),`_can_admit`+准入循环对应 `scheduler`(知识点 5),`self.forward_fn` 是可插拔的"attention backend + 模型本体"占位符(默认用随机 logits 模拟,真实场景换成 HF 模型的 forward),`sample_one` 对应 `sampler`(知识点 10),整个 `MiniEngine` 类本身对应最外层的 `engine`。`_can_admit` 的写法比知识点 5 的 `Engine._can_admit` 更贴近真实 vLLM:它不是简单统计"已用 token 数",而是按 `PagedKvPool` 的 `block_size` 精确算出这个请求全生命周期(`prompt + max_new_tokens`)需要多少个物理块(`(需要的 token 数 + block_size - 1) // block_size`,向上取整的经典写法),再检查 `pool.n_free()` 是否够——这是知识点 1 的"needed"概念、知识点 3 的分页粒度、知识点 5 的准入循环三者的第一次真正融合。`CASES` 里 5 个场景(`short-in-short-out`/`long-in-short-out`/`short-in-long-out`/`big-batch`/`streaming-1`)刻意覆盖了 prefill 主导(case 2,`p=1024,m=8`)和 decode 主导(case 3,`p=16,m=256`)两种极端,以及大并发(case 4,`n=32`)和单流(case 5,`n=1`)两种规模极端,这样"哪个环节是瓶颈"会随 case 不同而系统性地变化,不是只测一种负载模式就下结论。

关于 capstone 本身,本知识点独立验证了两处和文档/直觉不完全一致的地方:第一,`MiniEngine` 的字段列表里声明了 `prefix_cache: Optional[PrefixCache] = None`,模块开头的 docstring 也把"Prefix cache (prefix_cache.py)"列为五个"组装到一起"的部件之一,但全文搜索 `mini_vllm.py`,"prefix_cache"只出现在 docstring、`import` 语句、字段声明这三处,`add`/`_can_admit`/`_make_table`/`step`/`run` 五个方法体内一次都没有引用这个字段——也就是说,前缀缓存被"接了线"(import + 字段)但没有真正"通电"(没有任何读写逻辑),capstone 实际运行时完全不会做前缀复用,这是一处文档承诺和真实实现之间的落差,和知识点 4 `paged_attention_triton` 的落差是同一类问题(声明了能力、但函数体没有兑现)。第二,`vllm_compare.py`(真 vLLM 对照工具)的模块 docstring 写"缺 vllm 时这个脚本 fail-fast(`SystemExit(2)`)",但实际代码是 `raise SystemExit("vllm not installed — ...")`——传给 `SystemExit` 的是一个字符串,不是整数 `2`;Python 处理"`SystemExit` 携带非整数参数"的规则是把这个值打印到 stderr、并以退出码 `1` 结束进程,本知识点在这台机器上实测退出码精确是 `1`,不是 docstring 里写的 `2`。这不影响"fail-fast 而不是静默假成功"这个设计意图本身(exit code 1 依然是非零、依然会被 CI/shell 判定为失败),但如果有自动化脚本真的去校验"退出码必须精确等于 2"这种细节,会因为这个文档和实现的不一致而出错。

**AI 研究场景:** "用一两百行代码复刻一个几万行生产系统的核心骨架"是学习复杂系统源码非常有效的方法论——不是要替代真实系统,而是通过亲手拼接"骨架版"的过程,建立起对真实系统里"这个模块大概在解决什么问题"的心智地图,后续再去读真实 vLLM 源码时,能快速把具体实现(带着大量工程优化、边界处理、多硬件适配代码)映射回这套骨架里的对应位置,而不会在几万行代码里迷失方向。这也是为什么 L12(源码导读)和 L13(capstone)被安排背靠背:先建立骨架,再动手拼一个能跑的骨架版本,两者互相印证。

**可运行例子:**
```python
import sys, inspect
import torch
sys.path.insert(0, "learning/inference-engine-core/src")
from common import Request
from paged_kv import PagedKvPool
from mini_vllm import MiniEngine, CASES, run_case, run_all, make_engine_for

# --- Part A: 5 个 capstone case，tokens_out 必须精确等于 n*m（case 5 是单流，用 m）---
results = run_all()
assert len(results) == 5
for cid, r in zip(range(1, 6), results):
    case = CASES[cid]
    expected = case["m"] if case["n"] == 1 else case["n"] * case["m"]
    assert r["tokens_out"] == expected, (cid, r["tokens_out"], expected)
    assert r["throughput"] > 0
by_case = {r["case"]: r for r in results}
# case 2（long-in-short-out，p=1024）是 prefill 主导：吞吐远低于其余 4 个 case，
# 因为 _make_table() 是逐个 token 调用 append_token() 的 Python 级循环（不是批量算子）
assert by_case[2]["throughput"] < by_case[1]["throughput"] / 5
assert by_case[2]["throughput"] < by_case[3]["throughput"] / 5

# --- Part B: 手工搭建一个不经过 CASES/make_engine_for 的小规模 MiniEngine，
# 直接检验 step() 的准入/退场机制 ---
torch.manual_seed(0)
pool = PagedKvPool(n_blocks=4, block_size=4, n_kv_heads=1, head_dim=2, n_layers=1)
eng = MiniEngine(pool=pool, max_running=2, vocab_size=16, eos_id=-1)
eng.add(Request(rid=0, prompt_ids=[1, 2], max_new_tokens=3))
eng.add(Request(rid=1, prompt_ids=[3, 4, 5], max_new_tokens=3))
eng.add(Request(rid=2, prompt_ids=[6], max_new_tokens=3))  # 第 3 个请求：先被 max_running=2 卡住

assert len(eng.pending) == 3 and len(eng.running) == 0
eng.step()  # 准入 req0+req1，并顺带跑 1 次 decode 迭代（step() 不是"只准入"）
assert {r.rid for r in eng.running} == {0, 1}
assert 0 in eng.block_tables and 1 in eng.block_tables
assert len(eng.pending) == 1
assert all(len(r.output_ids) == 1 for r in eng.running)

m = eng.run()
assert {r.rid for r in eng.finished} == {0, 1, 2}
assert eng.block_tables == {}       # 每个完成的请求的 block table 都被释放并弹出，没有泄漏
assert pool.n_free() == 4           # 全部物理块归还

# --- Part C: 验证过的真实发现——prefix_cache 字段被声明/import，但从未被读写 ---
src = inspect.getsource(sys.modules["mini_vllm"])
mentions = [ln for ln in src.splitlines() if "prefix_cache" in ln.lower()]
assert len(mentions) == 3  # docstring 一处 + import 一处 + 字段声明一处，仅此而已
for method in (MiniEngine.step, MiniEngine.add, MiniEngine._make_table, MiniEngine._can_admit):
    assert "prefix_cache" not in inspect.getsource(method)

# --- Part D: make_engine_for 的池容量公式，逐 case 交叉核对 ---
for cid, case in CASES.items():
    e = make_engine_for(case)
    max_tokens_per_req = case["p"] + case["m"]
    expected_blocks = max(64, case["n"] * (max_tokens_per_req // 16 + 2))
    assert e.pool.n_blocks == expected_blocks
    assert e.pool.block_size == 16
```

**实测(`.venv` 真跑):** 5 case 全部 `tokens_out` 精确匹配 `n*m` 公式。三次独立重跑(每次全新 Python 进程)测出的吞吐量级稳定但绝对值有波动,例如某次为 `case1=2395.1, case2=112.2, case3=2597.1, case4≈2400-2800, case5≈2100-2500`(单位 tok/s)——**这组数字和 `README.md` 里记录的参考值(`case1≈17k, case2≈830, case3≈22k, case4≈22k, case5≈21k`)相差 5-10 倍**,但两者内部的相对大小关系完全一致(case2 远低于其余 4 个,其余 4 个量级相近)。这是意料之中的:这套 benchmark 的"模型"只是 `torch.randn(vocab_size)` 生成随机 logits,真正耗时的是 `MiniEngine` 自己的 Python 级调度/KV 写入/采样开销,吞吐绝对值直接取决于运行当下这台机器的 CPU 单核速度和系统负载,不代表任何真实模型的推理速度,换一台机器测出成倍差异的绝对数字是正常现象,只有相对趋势(哪个 case 慢、慢多少倍)才是这套 benchmark 真正想说明的问题。追查 case 2 明显更慢的原因:`_make_table()`(`mini_vllm.py:58-64`)对 `prompt_ids` 里的每一个 token 都单独调用一次 `table.append_token()`,`prompt_len=1024` 意味着光是"prefill"这一步就要执行 1024 次 Python 级函数调用,这也是 L13 lecture 自己标注 case 2"prefill 主导"的真实原因所在——不是模型计算慢,是这套教学实现故意用逐 token 循环模拟 prefill,让"prefill 需要处理的 token 数"这个变量的影响被放大到清晰可见。`prefix_cache` 相关的搜索确认全文只有 `3` 处提及(docstring/import/字段声明),`step`/`add`/`_make_table`/`_can_admit` 四个方法体内一次都没有出现,验证了"接了线但没通电"的结论。`make_engine_for` 的池容量公式在全部 5 个 case 上精确吻合。另外单独验证了 `vllm_compare.py` 的 fail-fast 行为:本机缺 `vllm` 时运行该脚本,退出码精确为 `1`,stderr 打印"vllm not installed — cannot run the reference benchmark. ..."——退出码与其自身模块 docstring 声称的 `SystemExit(2)` 不一致(docstring 写的整数 2,实际传参是字符串,Python 因此以退出码 1 结束)。

**面试怎么问 + 追问链:**
- **Q:** "mini-vLLM 这个 capstone 具体复刻了真实 vLLM 的哪几个核心模块?" —— 期望对应到 L12 提到的五大件:`engine`(`MiniEngine` 本身)、`scheduler`(准入循环)、`block_manager`(`PagedKvPool`/`BlockTable`)、`attention backend`(可插拔的 `forward_fn`)、`sampler`(`sample_one`),并能说出这是"把前面几个知识点的成果拼装起来"而不是全新内容。
- **追问 1(核心陷阱,考察是否真读过完整代码):** "capstone 的 `MiniEngine` 支持 prefix caching 吗?模块文档里不是把它列为五个组成部分之一吗?" —— 期望明确说"docstring 和字段声明确实提到了 `prefix_cache`,但检查 `step()`/`add()`/`_make_table()`/`_can_admit()` 四个方法体,没有任何一处真正读写这个字段——这是一处'接了线但没通电'的未完成集成,实际运行时不会做任何前缀复用,不能只看 docstring 或者类字段列表就断定某个能力已经生效"。
- **追问 2:** "5 个 case 里,case 2(long-in-short-out)吞吐远低于其他 4 个,根本原因是什么?" —— 期望说出"`prompt_len=1024` 且 `_make_table()` 对每个 prompt token 都单独调用一次 `append_token()`(Python 级循环,不是批量算子),这个'逐 token 模拟 prefill'的实现方式让长 prompt 场景的调度开销被线性放大,这不是真实模型计算慢,而是这套 CPU stub 引擎本身的实现方式决定的",能进一步指出"这也说明拿这套 benchmark 的绝对吞吐数字去类比真实模型性能是不合适的,只能看相对趋势"。
- **追问 3:** "如果这台机器和 README 记录数据的那台机器测出的 5 个 case 绝对吞吐差 5-10 倍,是不是说明这个引擎或者这次测量有问题?" —— 期望说出"不一定,这套 benchmark 的瓶颈是 `MiniEngine` 自身的 Python 调度开销,不是任何真实模型计算,吞吐绝对值天然对运行机器的 CPU 速度、系统负载高度敏感;真正应该关注、也应该保持稳定的是不同 case 之间的相对关系(哪个 case 慢、量级差多少),这些关系在两次不同机器的测量里应该是一致的,本知识点已经验证过这一点"。

**常见坑:** 只看 `MiniEngine` 的字段列表或者模块 docstring 就断定某个能力(比如 prefix caching)已经集成生效——必须实际读方法体确认字段真的被读写过,这是本知识点特意验证并证伪的一个例子。另一个坑是把 mini-vLLM 5 case 的吞吐数字当作有跨机器可比性的"性能基准"去引用或者对比——这套数字的绝对值高度依赖运行当下的机器状态(这台机器和 README 记录数据的机器差了 5-10 倍),只有相对趋势(哪个 case 更慢、慢多少倍)才是稳定、可复现、值得记住的结论,拿绝对 tok/s 数字去跟别的机器或者别的引擎比较是没有意义的。第三个坑是假设 `vllm_compare.py` 缺依赖时的退出码是文档里写的 `2`——实测是 `1`,这类"docstring 写的具体数值和真实运行结果不一致"的情况提醒我们,涉及具体数值的文档声明也需要独立验证,不能因为写在 docstring 里就默认它是准确的。

---

*本文覆盖 11 个知识点,对应 `learning/inference-engine-core/` 全部 13 lectures + 11 个 src 源文件(不含 `tests/`)。下一篇:[02-sglang-radixattention.md](02-sglang-radixattention.md)(待撰写)。*




