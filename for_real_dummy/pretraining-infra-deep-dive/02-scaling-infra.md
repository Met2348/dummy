# 02 · 训练规模化深挖(Scaling Infra)

> 总览见 [00-roadmap.md](00-roadmap.md)

数据处理好了(01号文件),下一个问题是"怎么把一个几十上百 B 参数的模型塞进有限显存、并让几十上百张卡都不闲着"。本文对应 `learning/scaling-infra/`(Module 3《造大模型》第 6 专题,14 lecture + 10 个 src 源文件),从 Scaling Laws 的"给定算力预算怎么分配模型大小和数据量"讲起,一路讲到并行策略选型(DP/ZeRO/TP/PP)、混合精度、训练监控,最后用一个训练估算器串联全部结论。10 个知识点(源 14 个 lecture 里 L08-L11 四个推理优化主题被合并成 1 个交叉引用点,理由见知识点 7)。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13,torch 2.11.0+cu128,CUDA 可用)下用 `.venv/Scripts/python.exe` 实际跑通验证。14 个 demo 脚本全部无 argparse(纯 `if __name__ == "__main__"` 直跑),全部 CPU 秒级(不需要真实多卡,`fsdp_demo.py`/`megatron_tp_demo.py` 用单进程建真 torch 模块跑 forward 但不需要 `torchrun`)。

**和 `for_real_dummy/inference-serving-deep-dive/` 的关系(必读):** 知识点 7 覆盖的 4 个推理优化脚本(`paged_attention_demo.py`/`vllm_demo.py`/`sglang_demo.py`/`speculative_decoding.py`/`quantization_demo.py`)在同仓库 [inference-serving-deep-dive](../inference-serving-deep-dive/00-roadmap.md) 系列已有深度覆盖(01-04 号文件,71 个知识点里的大部分)。本文知识点 7 不重复展开七步模板,只从"一个训练 infra 工程师为什么也要懂推理侧这几个技术"的全局视角简要串联并给出交叉引用链接。

---

## 1. Scaling Laws:Chinchilla 与 Llama-3 风格 Over-train(`scaling_laws.py`)—— 给定算力预算,模型和数据怎么分配

**是什么:**
```python
def chinchilla_loss(N: int, D: int,
                    A: float = 406.4, B: float = 410.7, E: float = 1.69,
                    alpha: float = 0.34, beta: float = 0.28) -> float:
    """Hoffmann 2022 公式."""
    return E + A * (N ** -alpha) + B * (D ** -beta)

def chinchilla_optimal_split(C: float) -> tuple:
    """给 budget C, 求最优 (N, D). 论文得: D / N ≈ 20."""
    N_opt_ratio = 20.0
    N = (C / (6 * N_opt_ratio)) ** 0.5
    D = C / (6 * N)
    return int(N), int(D)
```
(`scaling_laws.py:5-32`,节选)

**一句话:** 给定固定的算力预算(FLOPs),Chinchilla scaling law 告诉你"模型参数量 N 和训练 token 数 D 应该按什么比例分配才能让 loss 最小"——原始论文给出的最优比例是 D/N≈20,但工业界(Llama-3 等)故意偏离这个"算力最优点",转而选择远超 20 倍的 D/N(over-train),因为推理成本要考虑进总拥有成本。

**底层机制/为什么这样设计:** `chinchilla_loss` 的公式形式 `E + A·N^-α + B·D^-β` 本身是从上百次不同 (N, D) 组合的实际训练实验里拟合出的经验公式,不是理论推导——`E` 是不可约损失(哪怕参数/数据无限大也存在的熵下界),`A·N^-α` 是模型容量不足带来的损失,`B·D^-β` 是数据量不足带来的损失,两项都随各自的变量增大而趋于零但永远不为零。`chinchilla_optimal_split` 反过来用这个经验公式求偏导数=0 的驻点,得到"给定总 FLOPs=6ND 时,N 和 D 怎么分配使 loss 最小"的闭式解,算出来的最优比例约为 1:20。

**AI 研究场景:** 这是任何"从零训一个模型"项目最先要回答的问题——决定模型架构参数(层数/宽度)之前,必须先基于目标算力预算算出大致的目标参数量,Chinchilla 论文(2022)之前业界普遍训得"参数量过大、数据量相对不足"(如原始 GPT-3),该论文发布后 Llama-1 等模型明确采用了更接近 1:20 的配比。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from scaling_laws import chinchilla_loss, chinchilla_optimal_split, over_train_split

C = 1e23  # 固定算力预算
N_opt, D_opt = chinchilla_optimal_split(C)
L_opt = chinchilla_loss(N_opt, D_opt)

N_over, D_over = over_train_split(C, ratio=200)
L_over = chinchilla_loss(N_over, D_over)

assert abs(D_opt / N_opt - 20) < 1e-6   # chinchilla_optimal_split 目标 1:20 比例(N/D 各自 int() 截断,不是精确等于20)
assert N_over < N_opt        # over-train: 同算力下模型更小
assert D_over > D_opt        # 但数据量(=推理时的"性价比")远大于最优点
assert L_over < L_opt         # 反直觉: over-train 的 loss 反而比"最优点"更低
print(f"Chinchilla 1:20  N={N_opt/1e9:.1f}B D={D_opt/1e9:.1f}B loss={L_opt:.4f}")
print(f"Over-train 1:200 N={N_over/1e9:.1f}B D={D_over/1e9:.1f}B loss={L_over:.4f}")
```

**实测(`.venv` 真跑):** C=1e23 FLOPs 下,Chinchilla 1:20 配比给出 N=28.9B/D=577.4B/loss=2.0119;1:200 over-train 给出 N=9.1B/D=1825.7B/loss=**2.0084**——**over-train 版本的 loss 比"算力最优点"还要略低**,这不是计算错误:源码 `__main__` 自带的 6 组离散 ratio(10/20/50/100/200/500)里,ratio=100(N=12.91B/D=1291.0B)loss 最低,为 2.0052。**独立复现时进一步把扫描步长从"6个离散点"加密到 `range(10,301,1)`(每整数 ratio 都算一遍)**,发现真实连续最小值其实出现在 **ratio=78**(N=14.62B,D=1140.2B,loss=**2.0050**),比源码 demo 展示的 ratio=100 那个点还要再低一点点(2.0050 vs 2.0052,差异很小但确实更低)——这说明源码自己打印的 6 个点只是"恰好离最优区间较近的几个样本",不是精确定位过的最优值,ratio≈100 是"6 点粗扫描下的局部最优代表",真实最优点在更精细的分辨率下会进一步左移到 ratio≈80 附近。额外验证:把 C 换成 1e22/1e24 两个不同数量级重新做精细扫描,最优 ratio 分别落在 60 和 100 附近——**最优 ratio 本身随算力预算 C 的量级而漂移,不是一个对所有规模都固定不变的常数**,这比"最优比例是某个固定数字"这个结论本身更重要。这揭示了一个重要事实:`chinchilla_optimal_split` 名字里的"optimal"是**論文原始推导下的理论最优**(在 D/N=20 处 loss 对 N、D 的偏导数同时为零,这是解析解,不依赖具体 C),但用这份具体的拟合系数代入 `chinchilla_loss` 做数值扫描,呈现出的实际最小 loss 点系统性偏离 20、且随 C 变化——这提示"最优比例"这个数字本身依赖于具体使用的经验公式拟合系数和算力规模,不能把 1:20 当成放之四海皆准的常数。

**面试怎么问 + 追问链:**
- **Q:** "Llama-3 用远超 Chinchilla 最优比例的数据量训练,这是不是意味着 Chinchilla scaling law 错了?"—— 期望:不是——Chinchilla law 优化的目标是"给定训练算力预算,loss 最小",没有考虑**推理成本**;工业界模型一旦训完要服务海量请求,推理阶段的总算力消耗(边际用户数×每次推理成本)在模型全生命周期里往往远超训练算力,一个更小但训得更"过饱和"的模型能在几乎不损失质量的前提下大幅降低推理成本,这是训练算力最优和总拥有成本最优之间的权衡,不是学术结论被推翻。
- **追问1:** "为什么本知识点独立扫描发现真实最优点在 ratio≈78-100 区间且随 C 漂移,而不是论文声称的固定 20?这说明什么方法论问题?"—— 期望:这正说明"验证一个结论不能只信源码自带的几个打印点、要做独立的高分辨率扫描"——本知识点是这条纪律的直接示范:源码自己的 6 点粗扫描已经能看出"ratio=100 比 ratio=20 更优"这个定性结论,但如果止步于这 6 个点,会把 ratio=100 误当成精确最优值写进文档;换成每 1 个整数一个采样点的精细扫描后才发现真实最优在 78 附近,而且换算力预算重扫一遍还发现这个"最优点"本身会随 C 漂移——独立验证的价值不只是"确认原文没错",也包括"发现原文的精度不够、需要补充更细致的结论"。

**常见坑:** 不要把 `chinchilla_optimal_split` 返回的 D/N=20 和"实测 loss 最小点"混为一谈——前者是原始论文声称的理论最优比例(这个函数按定义强制实现 1:20),后者是这份具体拟合系数代入具体算力预算后,数值扫描算出的真实最小点,两者在这份代码的具体参数设置下并不重合,差异本身就是一条值得记录的发现而不是要"修正"的 bug。

---

## 2. 并行训练总览:DP/ZeRO/TP/PP 显存账本(`parallelism_demo.py`)—— 同一个 70B 模型,不同并行策略下每卡显存差 64 倍

**是什么:**
```python
def dp_memory(n_params: int, n_gpu: int) -> dict:
    """DP: 每卡全模型."""
    per_gpu = (model_size_bytes(n_params, "bf16")      # weights
               + model_size_bytes(n_params, "bf16")      # grad
               + optimizer_state_bytes(n_params, "adamw"))  # optimizer (不分片!)
    return {"per_gpu_gb": per_gpu / 1e9, "total_gb": per_gpu * n_gpu / 1e9}

def zero3_memory(n_params: int, n_gpu: int) -> dict:
    """ZeRO-3 / FSDP: shard W + G + O."""
    w = model_size_bytes(n_params, "bf16") / n_gpu
    g = model_size_bytes(n_params, "bf16") / n_gpu
    o = optimizer_state_bytes(n_params, "adamw") / n_gpu
    return {"per_gpu_gb": (w + g + o) / 1e9}
```
(`parallelism_demo.py:7-37`,节选)

**一句话:** DP(数据并行)每张卡都保有完整的模型权重+梯度+优化器状态,只是分数据;ZeRO 系列(1/2/3)依次把优化器状态、梯度、权重三样东西逐步切片分摊到各卡;TP/PP 则是从模型本身下手,直接把权重矩阵或网络层切开分布到不同卡——这是"数据侧分片 vs 模型侧分片"两条完全不同的并行化路线。

**底层机制/为什么这样设计:** Adam 优化器状态(一阶矩+二阶矩,均为 fp32)是训练显存里最大的单项开销——`optimizer_state_bytes` 里 `n_params*4*2` 意味着仅优化器状态就是权重本身(bf16,2 bytes/param)显存占用的 4 倍;ZeRO-1 只分片这一项就能显著降低单卡显存,是"改动最小、收益最大"的第一步;ZeRO-3 把权重本身也分片,代价是前向/反向传播时需要动态 all-gather 回完整权重(带来额外通信开销),这是"显存最省但通信最贵"的极端点。TP/PP 不分片"状态"而是分片"计算图本身",适用场景和 ZeRO 不同:模型大到单卡连一个 layer 的权重矩阵都放不下时(如 175B+ 模型),必须用 TP 把矩阵乘法本身切开算,ZeRO 无法解决这种情况。

**AI 研究场景:** 这张显存账本是任何"设计训练集群配置"任务的起点——面试/实际工作中常见问题"70B 模型用 64 张 A100(80GB)训练,选哪种并行策略",答案必须结合这张账本(哪些策略能把显存压到 80GB 以内)和通信带宽(哪些策略的通信开销集群能承受)两方面权衡。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from parallelism_demo import dp_memory, zero1_memory, zero2_memory, zero3_memory, tp_memory, pp_memory

N, G = 70_000_000_000, 64
dp = dp_memory(N, G)["per_gpu_gb"]
z1 = zero1_memory(N, G)["per_gpu_gb"]
z2 = zero2_memory(N, G)["per_gpu_gb"]
z3 = zero3_memory(N, G)["per_gpu_gb"]
tp8 = tp_memory(N, 8)["per_gpu_gb"]

assert dp > z1 > z2 > z3   # 分片越彻底，显存越省，严格单调
assert z3 < 16              # ZeRO-3 在 64 卡下应显著低于消费级 GPU 显存量级
ratio = dp / z3
print(f"DP={dp:.1f}GB  ZeRO-1={z1:.1f}GB  ZeRO-2={z2:.1f}GB  ZeRO-3={z3:.1f}GB  TP=8:{tp8:.1f}GB")
print(f"DP/ZeRO-3 显存比 = {ratio:.1f}x")
```

**实测(`.venv` 真跑):** Llama-2-70B(不含激活值)在 64 卡下:DP **840.0GB**(单卡装不下,DP 在这个规模下**根本不可行**,这个数字本身就说明纯数据并行对 70B+ 模型是无意义的选项)→ ZeRO-1 288.8GB → ZeRO-2 150.9GB → ZeRO-3 **13.1GB**——DP 到 ZeRO-3 之间显存相差 **64.1 倍**,恰好约等于 GPU 数量(64),这不是巧合:ZeRO-3 把权重+梯度+优化器状态三项全部按卡数均分,理想情况下显存占用应严格反比于卡数,和 DP(每卡显存与卡数无关)相比,差距自然趋近于卡数本身。TP=8 和 PP=8 给出相同的 105.0GB(源码里两者用完全相同的公式计算,这是本模块的简化:真实 TP 和 PP 在通信模式、激活值分布上有本质差异,但静态权重显存分摊公式碰巧一致)。

**面试怎么问 + 追问链:**
- **Q:** "ZeRO-3 显存最省,为什么实际训练不总是无脑用 ZeRO-3?"—— 期望:ZeRO-3 需要在每次 forward/backward 前动态 all-gather 出完整的层权重,这意味着大量额外的通信流量(尤其是层数很多、每层都要来回聚合/释放时),在带宽不够高的集群(如 PCIe 互联而非 NVLink)上,通信开销可能超过省下的显存收益带来的计算效率提升,ZeRO-2(只分片梯度和优化器状态,权重仍完整保留在每卡)在通信带宽受限场景下往往是更实际的选择。
- **追问1:** "如果 70B 模型即使用 ZeRO-3 仍然装不下(比如换成 500B 模型),下一步该怎么办?"—— 期望:需要引入模型并行(TP/PP)进一步切分——ZeRO 系列本质上仍然要求"每卡至少能装下与卡数成反比的那一份状态并完成对应计算",当模型大到连均分后的份额都装不下单卡时(或单层权重矩阵本身就超过单卡显存),必须用 TP 把矩阵运算切开分布式完成,这是 ZeRO 无法替代的场景,工业界超大模型训练通常是 ZeRO+TP+PP 的"3D 并行"组合,不是单一策略。

**常见坑:** 这张显存账本**没有包含激活值(activation)显存**——`parallelism_demo.py` 文件头注释明确写"no activation",实际训练时激活值(尤其是长序列、大 batch 场景下)可能占到总显存的相当比例,单独用这份"权重+梯度+优化器"账本估算就下结论"某个并行策略一定装得下",在真实场景中可能因为激活值爆表而失败——这也是为什么下方知识点 10 的训练估算器专门把 activation 作为独立一项计入。

---

## 3. FSDP 显存分片实测(`fsdp_demo.py`)—— 真建 PyTorch 模块,验证 ZeRO-3 公式不是纸上谈兵

**是什么:**
```python
def estimate_fsdp_memory(n_param: int, n_gpu: int = 4) -> dict:
    """估算 FSDP 各阶段显存."""
    bf16 = 2
    fp32 = 4
    return {
        "weights_shard_gb": n_param * bf16 / n_gpu / 1e9,
        "grad_shard_gb": n_param * bf16 / n_gpu / 1e9,
        "optimizer_shard_gb": n_param * fp32 * 2 / n_gpu / 1e9,
        "all_gather_peak_gb": n_param * bf16 / 1e9,   # 注意:不除以 n_gpu
    }
```
(`fsdp_demo.py:42-51`)

**一句话:** 和知识点 2 的纯公式版本不同,本文件先用真实 PyTorch `nn.Module`(`TinyModel`,含 `MultiheadAttention`+`LayerNorm`+MLP 的完整 Transformer block)构建一个可以真实 forward 的模型确认参数计数逻辑无误,再把同样的分片公式套用到 Llama-8B 规模上做估算,并额外暴露了一个知识点 2 没有的关键指标:`all_gather_peak_gb`(瞬时峰值显存,不随卡数分摊)。

**底层机制/为什么这样设计:** `all_gather_peak_gb = n_param * bf16 / 1e9`(**没有** `/ n_gpu`)是这个知识点最容易被忽视但最重要的一行——FSDP 平时每卡只保有 1/n_gpu 的权重分片,但在具体某一层做 forward/backward 计算之前,必须临时把**那一层**的完整权重 all-gather 回来才能计算(计算完立刻释放),这个瞬时峰值和总卡数无关,只取决于模型总大小——这正是"FSDP 降低了*平均*显存占用,但*瞬时峰值*显存不会无限降低"这条容易被误解的性质的具体体现。

**AI 研究场景:** 显存 OOM 排查时,如果开发者只看"稳态显存占用正常,为什么偶尔还是 OOM",这个 `all_gather_peak_gb` 概念就是答案——瞬时峰值出现在具体某层的 all-gather 窗口期,监控工具如果只采样低频率的显存快照,容易完全错过这个峰值窗口,需要专门理解这个机制才能定位问题。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/scaling-infra/src")
from fsdp_demo import TinyModel, estimate_fsdp_memory

m = TinyModel(n_layer=4, d=128, vocab=1000)
n_param = sum(p.numel() for p in m.parameters())
x = torch.randint(0, 1000, (2, 16))
out = m(x)
assert out.shape == (2, 16, 1000)     # 真实 forward 验证模型可跑通

for ngpu in [1, 4, 8, 16]:
    r = estimate_fsdp_memory(n_param=8_000_000_000, n_gpu=ngpu)
    assert r["all_gather_peak_gb"] == 16.0          # 恒定，不随卡数变化
    assert r["weights_shard_gb"] == 16.0 / ngpu      # 严格反比

print(f"TinyModel real params: {n_param/1e6:.2f}M, real forward output shape: {tuple(out.shape)}")
print(f"Llama-8B FSDP: weights_shard(16卡)={estimate_fsdp_memory(8e9,16)['weights_shard_gb']:.2f}GB "
      f"vs all_gather_peak(恒定)={estimate_fsdp_memory(8e9,16)['all_gather_peak_gb']:.2f}GB")
```

**实测(`.venv` 真跑):** `TinyModel(n_layer=4, d=128, vocab=1000)` 真实建模+forward,参数量 1.05M,输出 shape 验证正确。Llama-8B 规模估算下,`all_gather_peak_gb` 在 1/4/8/16 卡全部恒定为 **16.0GB**,而 `weights_shard_gb` 从 16.00→4.00→2.00→1.00GB 严格反比于卡数——16 卡时,稳态权重分片只占 1GB,但瞬时峰值仍然是 16GB,两者相差 **16 倍**,如果显存预算只按"稳态 1GB"规划,完全没有为这个瞬时峰值预留空间,训练会在第一次 forward 就直接 OOM。

**面试怎么问 + 追问链:**
- **Q:** "FSDP 分片数越多(卡数越多),是不是稳态显存降得越低就一定越好?"—— 期望:不完全是——稳态显存确实严格反比于卡数,但 `all_gather_peak_gb` 不随卡数变化,这意味着**卡数增加到一定程度后,显存瓶颈会从"稳态占用"转移到"瞬时峰值"**,继续增加卡数对解决 OOM 问题的边际收益会迅速衰减;真正缓解峰值问题需要用"限制同时 all-gather 的层数"(`limit_all_gathers=True`,源码 `print_fsdp_setup()` 里已经带了这个参数)或结合 activation checkpointing。
- **追问1:** "这个瞬时峰值可以被完全消除吗?"—— 期望:不能被消除但可以被控制——本质上"某一层要计算,就必须先有这一层的完整权重"是矩阵乘法的硬性前提,唯一能做的是控制"同时处于 all-gather 状态的层数"(prefetch 窗口大小)来在"显存峰值"和"通信/计算重叠效率"之间找平衡点,不存在既不需要临时聚合完整权重、又能做矩阵乘法的办法。

**常见坑:** `TinyModel` 的真实参数量(1.05M)和后面 Llama-8B 估算(8e9)之间没有任何直接联系——`estimate_fsdp_memory` 的第二部分调用是**手动传入** `n_param=8e9` 这个硬编码数字,不是从 `TinyModel` 推算出来的;`TinyModel` 存在的目的只是验证"FSDP wrap policy 面对的是一个真实的、带 attention/norm/mlp 结构的 PyTorch 模块",不是用于验证具体的参数量数字,阅读源码时容易误以为两部分数字有换算关系。

---

## 4. DeepSpeed ZeRO 配置生成(`deepspeed_config.py`)—— 从"策略选择"到"可直接喂给 deepspeed 命令行的 JSON"

**是什么:**
```python
def make_ds_config_zero3(micro_batch: int = 1, grad_accum: int = 16,
                          cpu_offload: bool = False) -> dict:
    cfg = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": 5e8, "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        "gradient_clipping": 1.0,
    }
    if cpu_offload:
        cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        cfg["zero_optimization"]["offload_param"] = {"device": "cpu"}
    return cfg
```
(`deepspeed_config.py:34-53`)

**一句话:** 知识点 2/3 讲的是 ZeRO 各 stage 的显存分片"原理",本知识点展示这些原理在真实工具(Microsoft DeepSpeed)里最终落地成什么样的配置文件——从"我选 ZeRO-3"这个决策,到一份可以直接 `deepspeed --config ds_config.json` 启动训练的 JSON,中间要填哪些字段。

**底层机制/为什么这样设计:** `cpu_offload` 分支展示了一个 ZeRO-3 的进阶变体:把优化器状态和参数进一步卸载到 CPU 内存(而不只是分片到各 GPU),这是"当 GPU 显存分片后仍然不够用"时的下一级手段,代价是 CPU-GPU 之间的 PCIe 数据搬运会成为新的性能瓶颈,通常只在显存实在紧张(如消费级显卡训练大模型)时才启用。`make_ds_config_megatron` 展示了 DeepSpeed 和 Megatron-LM 风格张量/流水并行结合使用时的配置组合(`tensor_parallel`+`pipeline_parallel`+`zero_optimization` stage 1)——这是"3D 并行"(数据+张量+流水)在真实配置文件层面的样子,ZeRO 通常只在这种组合里用较低的 stage(1 或 2),因为 TP/PP 已经承担了大部分显存压力。

**AI 研究场景:** 任何真实使用 DeepSpeed 训练框架的项目,第一步都是写这样一份 JSON 配置——理解这些字段(`overlap_comm` 通信与计算重叠、`reduce_bucket_size` 梯度规约的分桶粒度、`stage3_prefetch_bucket_size` 预取窗口)的含义,是能不能针对具体硬件/模型规模调优训练吞吐的关键。

**可运行例子:**
```python
import sys, json
sys.path.insert(0, "learning/scaling-infra/src")
from deepspeed_config import make_ds_config_zero1, make_ds_config_zero2, make_ds_config_zero3, make_ds_config_megatron

z3_offload = make_ds_config_zero3(cpu_offload=True)
assert z3_offload["zero_optimization"]["stage"] == 3
assert "offload_optimizer" in z3_offload["zero_optimization"]
assert "offload_param" in z3_offload["zero_optimization"]

z3_no_offload = make_ds_config_zero3(cpu_offload=False)
assert "offload_optimizer" not in z3_no_offload["zero_optimization"]

megatron_cfg = make_ds_config_megatron(tp=8, pp=8)
assert megatron_cfg["tensor_parallel"]["tp_size"] == 8
assert megatron_cfg["pipeline_parallel"]["stages"] == 8
assert megatron_cfg["zero_optimization"]["stage"] == 1   # 3D并行组合下ZeRO通常只需stage 1

# 确认合法JSON，能被真实deepspeed命令行工具解析
json_str = json.dumps(z3_offload)
assert json.loads(json_str) == z3_offload
print(f"ZeRO-3+CPU offload config keys: {list(z3_offload['zero_optimization'].keys())}")
```

**实测(`.venv` 真跑):** 四个配置生成函数全部产出合法 JSON,`stage3_prefetch_bucket_size` 等字段的浮点数(如 `5e8`)在 JSON 序列化后正确显示为 `500000000.0`(Python `json.dumps` 默认行为,不是 bug);`make_ds_config_megatron(tp=8, pp=8)` 确认 `zero_optimization.stage=1`——验证了"3D 并行下 ZeRO 只需要最轻量的 stage 1"这条工程经验在配置生成逻辑里确实是硬编码体现的(不是知识点 2 讨论时的推测,是这份配置生成器自己的实际默认值)。

**面试怎么问 + 追问链:**
- **Q:** "`cpu_offload=True` 之后训练会变慢吗,慢多少数量级?"—— 期望:会显著变慢,因为 PCIe 带宽(通常几十 GB/s)比 GPU 显存带宽(A100/H100 上通常 2-3 TB/s)低两个数量级以上,每次优化器 step 都要把状态在 CPU/GPU 之间搬运,这个开销可能是训练吞吐的主要瓶颈——CPU offload 是"用速度换空间"的最后手段,只应该在其他并行策略都无法把显存压到预算内时才考虑。
- **追问1:** "`overlap_comm: True` 具体重叠的是什么和什么?"—— 期望:重叠的是"梯度的 all-reduce/reduce-scatter 通信"和"反向传播的计算"——理论上一旦某一层的梯度计算完成,就可以立即开始这层梯度的通信,同时 GPU 继续算前一层的梯度,让通信时间被计算时间"掩盖"而不是串行等待,这是几乎所有分布式训练框架的标配优化,不重叠的话总训练时间是"计算时间+通信时间"简单相加。

**常见坑:** 这些函数只是生成配置**字典**,不真的调用 DeepSpeed 库(`import deepspeed` 完全不出现在源码里)——本知识点验证的是"配置的结构和字段是否合理",不是"配置能不能让真实训练跑起来",后者需要真实安装 DeepSpeed(Linux/WSL2 环境,Windows 原生不支持)并实际启动训练进程才能验证,这是本文全系列"纯 CPU 数值模拟"定位的自然边界,不是遗漏。

---

## 5. Megatron 张量并行:ColumnLinear/RowLinear(`megatron_tp_demo.py`)—— 一个 MLP 怎么被切成 4 份还能拼回正确结果

**是什么:**
```python
import torch.nn as nn

class MockColumnLinear(nn.Module):
    """col-split: weight [d_in, d_out / tp]."""
    def forward(self, x):
        return x @ self.W          # 每卡独立算，输出天然是"部分"结果，无需通信

class MockRowLinear(nn.Module):
    """row-split: weight [d_in / tp, d_out]."""
    def forward(self, x_local):
        out = x_local @ self.W
        return out                  # 每卡输出需要 all-reduce 求和才是最终结果

class TpMlp(nn.Module):
    """TP MLP: col → activation → row → all-reduce."""
    def forward(self, x):
        h = self.fc1(x).relu()
        out = self.fc2(h)
        # 实际多卡: dist.all_reduce(out)
        return out
```
(`megatron_tp_demo.py:11-47`,节选)

**一句话:** Megatron-LM 张量并行的核心技巧是"第一个线性层按列切、第二个线性层按行切",这样中间的 activation(激活函数输出)天然就是"每卡各算各的、互不需要通信",只有最后 MLP 输出时才需要一次 all-reduce 求和——把原本"切一刀就要通信一次"降低到"两层只需要通信一次"。

**底层机制/为什么这样设计:** 关键在于列切+行切的**配合**:`MockColumnLinear` 把 `d_out` 维度切成 `tp` 份,每张卡算出的是完整 `d_in` 维度上、部分 `d_out` 维度的结果——这部分结果不需要跨卡通信就能直接送入激活函数(逐元素运算,互不依赖);`MockRowLinear` 反过来把 `d_in` 维度切开,每张卡用自己那一份(刚好是上一层对应卡产出的那部分激活值)算出一个"部分和",这些部分和加总(all-reduce sum)才是最终的完整输出。如果只用列切或只用行切,中间就需要额外的通信,Megatron 论文这套"列-行配对"的设计正是为了把整个 MLP block 的通信量压缩到最少(一次 forward 只需 1 次 all-reduce,而不是 2 次)。

**AI 研究场景:** 这是训练 100B+ 参数模型(单层权重矩阵本身可能就有几十 GB,单卡装不下)时唯一的解法——ZeRO 系列(知识点 2/3)分片的是"每卡的状态副本",TP 分片的是"单次矩阵乘法本身",两者不冲突、可以叠加使用,大模型训练的标准 3D 并行(DP+TP+PP)里 TP 通常只在同一物理节点内的卡之间使用(因为 all-reduce 频率高、对带宽要求最苛刻,节点内 NVLink 带宽远高于跨节点)。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/scaling-infra/src")
from megatron_tp_demo import TpMlp, gather_tp_outputs

d, d_ff, tp = 64, 256, 4
mlps = [TpMlp(d, d_ff, tp_size=tp, tp_rank=i) for i in range(tp)]
x = torch.randn(2, 10, d)

local_outs = [mlps[i](x) for i in range(tp)]
out = gather_tp_outputs(local_outs)          # 模拟 all-reduce sum
assert out.shape == (2, 10, d)                # 切了又拼回来，最终形状和不切一样

fc1_params_per_rank = sum(p.numel() for p in mlps[0].fc1.parameters())
assert fc1_params_per_rank == d * (d_ff // tp)   # 每卡只有 1/tp 的 fc1 参数
assert fc1_params_per_rank * tp == d * d_ff       # 但 tp 卡的参数总和等于完整 vanilla fc1

print(f"vanilla fc1: {d*d_ff} params, 每卡 TP fc1: {fc1_params_per_rank} params ({tp}卡合计={fc1_params_per_rank*tp})")
```

**实测(`.venv` 真跑):** `d=64, d_ff=256, tp=4` 下,vanilla `fc1` 权重形状 `[64,256]`(16384 参数),TP 切分后每卡 `[64,64]`(4096 参数),4 卡合计 16384——精确等于未切分时的参数总量,验证了"张量并行不改变模型总参数量,只是把同一份参数拆到不同卡上"这个基本性质。4 卡分别独立 forward 后用 `gather_tp_outputs`(源码里就是简单的 `sum(local_outs)`,模拟真实多卡场景下的 `all_reduce`)合并,最终输出 shape `[2,10,64]` 和不做任何切分时完全一致,证明了"列切+行切+一次 all-reduce"这套设计确实能在数学上精确还原未切分时的计算结果,不是近似。

**面试怎么问 + 追问链:**
- **Q:** "如果我把 ColumnLinear 和 RowLinear 的顺序反过来(先行切再列切),还能保持这种'只需一次通信'的性质吗?"—— 期望:不能——先行切意味着每卡的输入是"部分维度",算出来的输出必须先做一次 all-reduce 才能得到正确的完整中间结果,再送入下一层列切的线性层,这样至少需要 2 次通信(中间一次+最后一次);"列切→激活→行切→通信"这个顺序的精妙之处正是利用了"逐元素激活函数不需要跨卡信息"这一点,把通信次数压到最少,顺序颠倒就失去了这个性质。
- **追问1:** "TP 的通信量和 batch size、序列长度有什么关系?这对选择 TP size 有什么影响?"—— 期望:TP 的 all-reduce 通信量正比于 `batch × seq_len × hidden_dim`(激活值张量的大小),和模型参数量本身无关——这意味着长序列/大 batch 场景下 TP 的通信开销会显著增长,这也是为什么 TP 通常控制在较小规模(如 8,受限于单节点 GPU 数和 NVLink 拓扑)而不会像 DP 那样扩展到成百上千卡,通信量会主导总训练时间。

**常见坑:** `gather_tp_outputs` 在这份教学代码里就是普通的 Python `sum()`,不涉及任何真实的跨进程通信——真实多卡场景下这一步是 `torch.distributed.all_reduce`,需要 NCCL 后端和真实的多进程/多机环境,源码注释 `# 实际多卡: dist.all_reduce(out)` 已经明确标注这是简化;误以为这份单进程代码"已经验证了 TP 的通信效率"是错误的,它只验证了 TP 切分方案在数学上的正确性,不涉及任何真实通信性能。

---

## 6. Pipeline Parallel:Bubble 与 Micro-batch 数的权衡(`pipeline_parallel_demo.py`)—— 独立验证 GPipe 与 Interleaved 1F1B 的气泡公式

**是什么:**
```python
def gpipe_bubble(n_stage: int, n_micro: int) -> float:
    """GPipe / 1F1B bubble 占比."""
    return (n_stage - 1) / (n_stage + n_micro - 1)

def interleaved_bubble(n_stage: int, n_micro: int, n_chunk: int = 4) -> float:
    """interleaved 1F1B."""
    return (n_stage - 1) / (n_chunk * (n_stage + n_micro - 1))
```
(`pipeline_parallel_demo.py:8-15`)

**一句话:** 流水线并行把模型切成 n_stage 段分布到不同卡,但任何一张卡在整个训练 step 里都有"等待上下游"的空闲时间(bubble)——bubble 占比随微批数(micro-batch,即把一个大 batch 切成多少个小块流水线送入)增加而下降,Megatron-LM 的 interleaved 1F1B 通过让每张卡负责不连续的多段模型(而非连续一段)进一步把 bubble 压低 `n_chunk` 倍。

**底层机制/为什么这样设计:** 直观理解 `(n_stage-1)/(n_stage+n_micro-1)` 这个公式:一条 n_stage 级流水线处理 n_micro 个微批总共需要 `n_stage+n_micro-1` 个时间片(前 n_stage-1 个时间片是"流水线灌注"阶段,只有部分卡有活干;之后才是稳态,所有卡都在忙碌;最后 n_stage-1 个时间片是"流水线排空"阶段)——分子的 `n_stage-1` 正是这段"灌注+排空"总长度里必然浪费的部分(简化到两段合并计算),分母是总时间片数,比值就是空闲时间占比。当 `n_micro` 远大于 `n_stage` 时,稳态运行时间远超灌注/排空开销,bubble 占比趋近于零;`n_micro` 太小(比如等于 1)时,几乎全程都在灌注/排空,bubble 占比逼近 100%。

**AI 研究场景:** 这是设计训练配置时"micro-batch 数至少要设多大"的理论依据——训练吞吐监控里如果发现 GPU 利用率长期偏低,check 的第一件事往往就是"当前 pipeline 的 micro-batch 数是不是设太小了,bubble 占比是不是过高"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from pipeline_parallel_demo import gpipe_bubble, interleaved_bubble

# 独立复现：手动按公式定义验证 n_stage=8 时几个关键点
assert abs(gpipe_bubble(8, 1) - 7/8) < 1e-9        # n_micro=1: 几乎全程都是灌注+排空
assert abs(gpipe_bubble(8, 8) - 7/15) < 1e-9        # 对照 README 给出的 46.7%
assert gpipe_bubble(8, 128) < 0.06                  # micro-batch 足够多，bubble 趋近于 0

# interleaved 应该在同样(n_stage, n_micro)下 bubble 更低
for M in [1, 8, 32]:
    b_gpipe = gpipe_bubble(8, M)
    b_inter = interleaved_bubble(8, M, n_chunk=4)
    assert b_inter < b_gpipe
    assert abs(b_inter - b_gpipe / 4) < 1e-9        # 恰好是 gpipe 的 1/n_chunk（本模块的简化关系）
    print(f"M={M:>3}: GPipe={b_gpipe:.1%}  Interleaved(chunk=4)={b_inter:.1%}  比值={b_gpipe/b_inter:.1f}x")
```

**实测(`.venv` 真跑):** 8-stage 流水线下,`n_micro=1` 时 bubble 高达 **87.5%**(几乎整个 step 都在空转),`n_micro=128` 时降到 **5.2%**;Interleaved 1F1B(`n_chunk=4`)在完全相同的 (n_stage, n_micro) 下,bubble 精确等于 GPipe/1F1B 版本的 **1/4**(如 M=8 时 GPipe 46.7% vs Interleaved 11.7%,比值恰好 4.0)——独立验证确认这个"恰好 1/n_chunk"的关系在本模块的具体公式定义下是精确成立的(不是近似),因为两个函数的公式定义本身就只相差一个 `n_chunk` 分母因子。`split_model_into_stages` 额外验证:24 层模型切成 8 段,每段精确 3 层,平均切分无余数残留。

**面试怎么问 + 追问链:**
- **Q:** "既然 Interleaved 1F1B 的 bubble 更低,为什么不是所有场景都默认用它?"—— 期望:Interleaved 让每张卡负责模型里不连续的多段(比如卡 0 负责第 1 层和第 9 层),这意味着每处理一个 micro-batch,数据需要在同一张卡上"进出"多次、且卡间通信次数也相应增加(通信频率变成原来的 `n_chunk` 倍)——这是用更高的通信频率换取更低的 bubble,在通信带宽充裕(节点内高速互联)时收益明显,带宽受限场景下增加的通信开销可能抵消甚至超过 bubble 减少带来的收益。
- **追问1:** "如果 n_stage 固定,能不能通过无限增大 n_micro 把 bubble 压到接近 0?这样做有什么代价?"—— 期望:理论上可以,但 micro-batch 越小(为了在固定 batch size 下切出更多个 micro-batch),每次前向/反向的矩阵运算规模越小,GPU 计算效率(尤其是 Tensor Core 利用率)会下降,过小的 micro-batch 会让"计算本身不饱和"取代"流水线 bubble"成为新的性能瓶颈,micro-batch 数量存在一个实际有效的甜点区间,不是越多越好。

**常见坑:** 这两个 bubble 公式本身只衡量"流水线气泡"这一种效率损失,不包含"层与层之间切分是否均衡"(如果某一段恰好包含计算量特别大的层,即使 bubble 占比数字很低,那一段仍然会拖慢整条流水线的实际吞吐)、也不包含通信开销——`gpipe_bubble`/`interleaved_bubble` 返回的百分比是一个理想化上界分析,实际训练吞吐还要综合考虑这些本模块未建模的因素。

---

## 7. 推理优化技术的全局定位(交叉引用点,源:`paged_attention_demo.py`/`vllm_demo.py`/`sglang_demo.py`/`speculative_decoding.py`/`quantization_demo.py`)

**是什么:** 这 5 个脚本覆盖原 lecture L08-L11(PagedAttention/vLLM、SGLang、投机解码、量化推理),深度内容已在同仓库 [inference-serving-deep-dive](../inference-serving-deep-dive/00-roadmap.md) 系列讲透,本点只做定位性串联,不重复展开七步模板:

| 主题 | 本模块脚本 | 本模块深度 | 完整深度见 |
|---|---|---|---|
| PagedAttention | `paged_attention_demo.py` | 真实 `BlockManager` 模拟(唯一真代码) | [inference-serving-deep-dive/01](../inference-serving-deep-dive/01-inference-engine-core.md) 知识点 2-4 |
| vLLM 离线/在线推理 | `vllm_demo.py` | 仅打印 setup 模板(不 import 真 vllm) | 同上,含真实 WSL2 部署 bonus |
| SGLang DSL | `sglang_demo.py` | 仅打印模板 | [inference-serving-deep-dive/02](../inference-serving-deep-dive/02-sglang-radixattention.md) |
| 投机解码 | `speculative_decoding.py` | 用 token 概率分布模拟 accept/reject,不跑真模型 | [inference-serving-deep-dive/03](../inference-serving-deep-dive/03-speculative-decoding.md) |
| 量化推理 | `quantization_demo.py` | 显存账本 + GPTQ/AWQ/FP8 调用模板 | [inference-serving-deep-dive/04](../inference-serving-deep-dive/04-quantization-deploy.md) |

**一句话:** 一个只做训练规模化的 infra 工程师,仍然需要对推理侧这几项技术有基本认知——训练集群和推理集群经常共享同一批硬件资源池,评估"值不值得把模型做得更大"这类决策时,必须同时权衡训练成本和部署后的推理成本(呼应知识点 1 里 over-train 的讨论)。

**底层机制/为什么这样设计:** 本模块里唯一有真实实现(而非纯打印模板)的是 `paged_attention_demo.py` 的 `BlockManager`——真实维护 block 分配、引用计数、fork 场景下的 CoW(copy-on-write)共享逻辑,这和 `inference-serving-deep-dive` 01 号文件深度覆盖的 PagedAttention 是同一机制在不同精细度下的两次独立实现,两边的 `BlockManager` 概念可以互相印证。其余 4 个脚本(vLLM/SGLang/投机解码/量化)在本模块只是"知道这项技术存在、大致怎么调用"级别的定位性内容,不承担把机制讲透的责任。

**AI 研究场景:** 系统设计面试里常见的"设计一个 LLM 训练+服务一体化平台"类问题,候选人如果只懂训练侧或只懂推理侧,通常答不出"为什么要把这两类工作负载分开调度"这类关键设计决策——这正是本知识点存在的意义:标注清楚"训练规模化"专题里推理技术的定位,而不是完全不提及。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from paged_attention_demo import BlockManager, Sequence, fork_for_beam_search

mgr = BlockManager(total_blocks=1024)
seq = Sequence(seq_id=0)
for _ in range(50):
    seq.append_token(mgr)
util_before = mgr.utilization()

fork = fork_for_beam_search(seq, mgr, new_seq_id=1)   # beam search / 多候选场景下的常见操作
util_after_fork = mgr.utilization()
assert util_after_fork == util_before   # fork 共享 block，不应立即多占显存(CoW)

print(f"50 token 后 util={util_before:.1%}, fork 后 util={util_after_fork:.1%}(应相等,验证CoW共享)")
```

**实测(`.venv` 真跑):** `BlockManager(total_blocks=1024)` 真实运行:50 个 token 追加后占用 4 个 block(util=6.2%),`fork()` 操作后 util **保持 6.2% 不变**(共享 block,没有立即复制),confirmed 这是真实的 copy-on-write 语义,不是打印模板;fork 后再追加 20 个 token,util 才涨到 7.8%(此时才真正触发新 block 分配)。这个行为和 `inference-serving-deep-dive` 01 号文件描述的 PagedAttention CoW 机制完全一致。

**面试怎么问 + 追问链:**
- **Q:** "为什么这个'训练规模化'专题里会出现 PagedAttention 这个明显是推理侧的技术?"—— 期望:因为源课程设计上,Module 3 覆盖"造大模型"全过程,scaling-infra 专题定位是"训练+推理的规模化基建总览",PagedAttention/vLLM/SGLang 等作为"推理侧同样需要规模化技术"的示例被并入,深度内容留给专门的推理服务系列(inference-serving-deep-dive)展开,这是课程结构设计上的合理分工,不是内容重复。

**常见坑:** 不要因为这 5 个脚本在本模块只是"简单演示",就误以为这些技术本身很简单——`vllm_demo.py`/`sglang_demo.py` 只打印模板正是因为它们背后的真实实现极其复杂(数万行 C++/CUDA/Python 混合代码),用几行 Python 无法真实复现,这也是为什么 `inference-serving-deep-dive` 需要专门一个系列(71 个知识点)才能讲透同样这几项技术。

---

## 8. 混合精度与训练稳定性(`mixed_precision_demo.py`)—— bf16 为什么是训练首选,以及两道自动求救机制

**是什么:**
```python
def loss_spike_guard(loss: float, ema: float, threshold: float = 5.0) -> bool:
    """True if 应跳过这一步."""
    return loss > threshold * ema

def grad_norm_warn(grad_norm: float) -> str:
    if grad_norm > 100: return "DANGER: grad explosion"
    if grad_norm > 10: return "warn: grad high"
    if grad_norm < 1e-4: return "warn: grad vanishing"
    return "ok"
```
(`mixed_precision_demo.py:40-52`)

**一句话:** bf16 和 fp16 都是 16-bit 浮点数,但指数位/尾数位分配完全不同——bf16 用更多位数表示指数(和 fp32 相同的指数范围)、更少位数表示尾数,fp16 相反;这个差异直接决定了 bf16 训练时几乎不会遇到"数值溢出变成 inf"的问题,而 fp16 需要额外的 loss scaling 机制配合才能安全训练。

**底层机制/为什么这样设计:** 浮点数溢出(超出可表示范围变成 `inf`/`-inf`)在深度学习训练中是致命的(一旦出现,后续所有计算都会被污染成 NaN);bf16 的指数位数和 fp32 完全相同,这意味着它能表示和 fp32 同样宽的数值范围(只是精度更低),训练过程中梯度/激活值即使数值较大也几乎不会溢出;fp16 的指数位更少,可表示范围远小于 fp32,深度学习里常见的小梯度值容易下溢成 0、大梯度值容易上溢成 inf,必须配合"loss scaling"(训练时把 loss 乘一个大数再反向传播,梯度也相应放大避免下溢,更新前再除回来)才能安全使用。`loss_spike_guard` 和 `grad_norm_warn` 则是训练稳定性的另一道防线——即使精度格式选对了,训练过程中仍可能因为数据异常、学习率过大等原因出现 loss 突然飙升或梯度爆炸,这两个函数是自动检测并响应这类异常的标准手段。

**AI 研究场景:** "为什么现代 LLM 训练几乎全部选择 bf16 而不是 fp16"是训练工程的高频面试题;`loss_spike_guard`(遇到异常大的 loss 直接跳过这一步,不更新参数)和 EMA loss tracking 是任何大规模训练 run 的标配监控组件,防止某个异常 batch(如脏数据、数值不稳定的中间状态)污染整个训练进程。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/scaling-infra/src")
from mixed_precision_demo import loss_spike_guard, grad_norm_warn

bf16_info = torch.finfo(torch.bfloat16)
fp16_info = torch.finfo(torch.float16)
fp32_info = torch.finfo(torch.float32)
assert abs(bf16_info.max - fp32_info.max) / fp32_info.max < 0.01   # 同指数范围下,max几乎相同(差异只来自尾数位舍入,不是精确相等)
assert fp16_info.max < bf16_info.max     # fp16 可表示范围远小于 bf16
assert bf16_info.eps > fp16_info.eps     # bf16 精度(eps)反而更粗，是"宽范围换低精度"的权衡

ema = 2.0
assert loss_spike_guard(15.0, ema) is True     # loss=15 是 ema 的 7.5倍，超过默认阈值5.0，应跳过
assert loss_spike_guard(8.0, ema) is False      # loss=8 是 ema 的 4倍，未超阈值，正常更新
assert grad_norm_warn(250) == "DANGER: grad explosion"
assert grad_norm_warn(0.00005) == "warn: grad vanishing"
print(f"bf16 max={bf16_info.max:.3e}  fp32 max={fp32_info.max:.3e}  (同量级,验证同指数范围)")
print(f"fp16 max={fp16_info.max:.3e}  (远小于bf16,验证下溢/上溢风险差异)")
```

**实测(`.venv` 真跑):** `torch.finfo` 真实查询确认 bf16 的 `max=3.390e+38` 和 fp32 的 `max=3.403e+38` 几乎完全相同(差异只来自 bf16 尾数位不足导致的舍入,量级完全一致),而 fp16 的 `max=6.550e+04` 比 bf16 小了 **33 个数量级**——这不是夸张,是浮点数指数位分配差异导致的真实硬件行为。`loss_spike_guard` 对 4 个真实测试值(2.1/3.5/15.0/8.0,ema=2.0)的判定:只有 loss=15.0(达到 ema 的 7.5 倍)触发跳过,其余(包括 loss=8.0,达到 ema 的 4 倍)都判定为正常更新,精确对应默认阈值 5.0 的边界。

**面试怎么问 + 追问链:**
- **Q:** "fp16 的精度(尾数位)比 bf16 更高,为什么训练界还是更偏爱精度更低的 bf16?"—— 期望:训练稳定性优先于单步数值精度——深度学习模型对"数值范围能覆盖多大"比"小数点后精确到第几位"更敏感,一旦某个中间值溢出成 inf/NaN,整个训练直接崩溃且往往无法恢复,而精度不足只会导致"训练收敛得稍微慢一点/最终 loss 稍微差一点"这种更温和的影响;bf16 用可承受的精度损失换取了几乎不需要额外工程手段(loss scaling)就能稳定训练的巨大简化。
- **追问1:** "如果推理阶段对精度要求更高(比如需要 fp16 而不是 bf16),训练用 bf16、推理用 fp16 会有问题吗?"—— 期望:确实存在权衡——很多部署场景选择 fp16 做推理是因为老一代 GPU(如 V100)对 fp16 的 Tensor Core 支持比 bf16 更成熟、速度更快,但如果训练全程用 bf16、推理强行转 fp16,某些训练时依赖 bf16 宽范围而产生的大数值可能在转换时溢出,这是精度格式切换时需要专门验证的兼容性问题,不能假设无缝迁移。

**常见坑:** `grad_norm_warn` 的三个阈值(100/10/1e-4)是教学示范用的经验数字,不是理论推导出的普适常数——真实训练里"多大的梯度范数算异常"高度依赖具体模型架构、学习率、batch size 等超参数组合,直接照搬这几个数字用于不同规模/架构的模型监控,可能得到大量误报或漏报,生产环境这类阈值通常需要基于该模型训练初期的实际梯度范数分布统计后再校准。

---

## 9. 训练监控:MFU 计算与 GPU 健康检查(`monitoring_demo.py`)—— 从"GPU 利用率 95%"到"真实计算效率只有 14.5%"的落差

**是什么:**
```python
def compute_mfu(tokens_per_sec: float, n_params: int,
                n_gpu: int, gpu_tflops: float) -> float:
    """6N D / time = throughput (loss FLOPs)."""
    actual_flops = 6 * n_params * tokens_per_sec
    theoretical = n_gpu * gpu_tflops * 1e12
    return actual_flops / theoretical
```
(`monitoring_demo.py:5-10`)

**一句话:** MFU(Model FLOPs Utilization,模型算力利用率)= 实际用于模型计算的 FLOPs / 硬件理论峰值 FLOPs,是衡量"这次训练 run 有没有把硬件性能真正压榨出来"的核心指标——和更容易被误读的"GPU 利用率(nvidia-smi 里的百分比)"是完全不同的两个数字。

**底层机制/为什么这样设计:** `6 * n_params * tokens_per_sec` 复用了知识点 1 提到的"6ND"经验公式(每个 token 的一次前向+反向传播大约需要 `6×参数量` FLOPs,这是 Transformer 架构下的粗略但被广泛使用的估计),除以硬件理论峰值算力(`n_gpu × gpu_tflops × 1e12`)就得到 MFU。这个指标之所以重要,是因为"GPU 利用率 95%"这类监控数字只反映"GPU 的计算单元有没有闲着",不反映"GPU 在忙的时候是不是在做真正推进训练的有效计算"——比如 GPU 忙于等待数据搬运、忙于低效的小矩阵运算(利用率高但吞吐低),`nvidia-smi` 的利用率数字仍然会显示很高,但这些"忙碌"对模型训练进度的实际贡献很低。

**AI 研究场景:** MFU 是训练大模型时最重要的单一效率指标——PaLM/Llama 等技术报告都会公布训练时的 MFU 数字(通常在 30%-55% 之间被认为是良好水平),这是评估"这次训练的并行策略/kernel 效率/数据管线有没有系统性问题"的第一手依据,MFU 明显偏低(如低于 20%)通常意味着存在某个环节(通信/数据加载/kernel 效率)成为了瓶颈。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from monitoring_demo import compute_mfu, EmaLossTracker, gpu_health

mfu = compute_mfu(tokens_per_sec=3000, n_params=8e9, n_gpu=1, gpu_tflops=990)
assert 0 < mfu < 1                          # MFU 应该是一个 0-1 的比例
assert mfu < 0.5                             # 这个具体场景下的实测应远低于"理想"水平

tracker = EmaLossTracker(alpha=0.99)
losses = [2.5, 2.4, 2.3, 2.3, 9.0, 2.3]
spikes = []
for L in losses:
    is_spike = tracker.is_spike(L)
    spikes.append(is_spike)
    tracker.update(L if not is_spike else tracker.ema)
assert spikes == [False, False, False, False, True, False]   # 只有9.0那次应被识别为spike

healthy = gpu_health(util=95, temp_c=70, mem_used_gb=20, mem_total_gb=24)
unhealthy = gpu_health(util=40, temp_c=92, mem_used_gb=23, mem_total_gb=24)
assert healthy == "OK"
assert "high" in unhealthy and "low" in unhealthy   # 同时命中温度过高和利用率过低两项
print(f"Llama-3 8B @ 3000 tok/s on 1×H100(990 TFLOPS): MFU={mfu:.1%}")
```

**实测(`.venv` 真跑):** `compute_mfu(tokens_per_sec=3000, n_params=8e9, n_gpu=1, gpu_tflops=990)` 算出 MFU=**14.5%**——这是一个刻意选取的"偏低"演示场景(3000 tok/s 对 8B 模型单卡 H100 而言确实是较低的吞吐,真实良好配置通常能到 40%+),用来说明即使 GPU 利用率监控显示很高,MFU 揭示的"真正有效算力利用率"可能远低于直觉。`gpu_health` 的两组测试:健康场景(util=95%,温度 70°C,显存 20/24GB)返回 `OK`,异常场景(util=40%,温度 92°C,显存 23/24GB)**同时命中三条**告警(利用率低+温度高+接近 OOM),真实体现了这类健康检查函数在生产监控里"一次调用捕获多个独立问题维度"的设计。

**面试怎么问 + 追问链:**
- **Q:** "MFU 只有 14.5%,可能的原因有哪些?"—— 期望:常见原因包括:数据加载跟不上计算速度(GPU 等数据)、并行策略引入的通信开销未被有效重叠、batch size/序列长度太小导致 kernel 效率低(小矩阵乘法无法充分利用 Tensor Core)、频繁的显存分配/释放、activation checkpointing 引入的重计算开销过高等——诊断 MFU 偏低需要结合 profiling 工具(如 PyTorch Profiler)具体定位是哪个环节在拖后腿,不能只看 MFU 这一个汇总数字直接下结论。
- **追问1:** "EmaLossTracker 的 `is_spike` 判断用的是当前 EMA 的倍数(3倍),这个阈值设计有什么潜在问题?"—— 期望:训练早期(EMA 还在收敛、样本数很少)EMA 本身可能不稳定,用一个不稳定的基准去判断"是否 spike"容易在训练最初的几十步产生误判;另外 EMA 是单调平滑的滞后指标,如果 loss 出现持续多步的缓慢上升(而不是单步剧烈跳变),EMA 会跟着慢慢上移,threshold 判断可能永远追不上这种"温水煮青蛙"式的异常,只对"单步剧烈跳变"这类 spike 敏感。

**常见坑:** MFU 公式里的 `6*n_params` 是"整个前向+反向传播每 token 的 FLOPs"的**近似**估计(来自知识点 1 讨论过的 6ND 经验公式),对于使用了 Flash Attention 等非标准计算路径、或者激活了 activation checkpointing(会引入额外的重计算 FLOPs,不是"有效"计算但确实消耗了算力)的真实训练场景,实际 FLOPs 和这个近似值可能有系统性偏差,MFU 数字在跨不同工程配置比较时需要注意这一点,不是绝对精确的物理量。

---

## 10. Capstone:训练估算器(`capstone_train_estimator.py`)—— 从 (模型规模, GPU 配置) 一键推出策略/显存/时长/成本

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

def estimate(spec: TrainSpec, gpu_hour_usd: float = 1.5) -> TrainPlan:
    # 前半段(算 w/g/opt/activ 四项显存分量)节选省略,完整版见 capstone_train_estimator.py:58-66
    strategies = [
        ("DP", w + g + opt + activ),
        ("ZeRO-1", w + g + opt / spec.n_gpu + activ),
        ("ZeRO-2", w + g / spec.n_gpu + opt / spec.n_gpu + activ),
        ("ZeRO-3/FSDP", (w + g + opt) / spec.n_gpu + activ),
        ("FSDP + grad ckpt", (w + g + opt) / spec.n_gpu + activ * 0.2),
        ("TP=8 + FSDP + grad ckpt", (w + g + opt) / (spec.n_gpu * 8) + activ * 0.2 / 8),
    ]
    selected = None
    for name, mem in strategies:
        if mem <= spec.gpu_vram_gb * 0.9:      # 留 10% 显存余量
            selected = (name, mem)
            break
    if selected is None:
        return TrainPlan(strategy="INFEASIBLE", mem_per_gpu_gb=strategies[-1][1],
                         feasible=False, tok_per_s=0, hours=0, notes="Need more GPU or PP")
```
(`capstone_train_estimator.py:58-90`,节选)

**一句话:** 这是本文全部 9 个知识点的落地整合——把 Scaling Laws(算力预算分配)、显存账本(DP/ZeRO/TP)、MFU(吞吐估算)这几套独立的公式串成一个函数:输入"多大模型、多少卡、什么 GPU",自动按"显存需求从大到小"依次尝试 6 种并行策略,选出第一个能装进显存预算的方案,再据此估算训练时长和云端成本。

**底层机制/为什么这样设计:** `strategies` 列表按显存需求**从高到低排列**、用 `for...break` 找第一个满足 `mem <= gpu_vram_gb*0.9` 的方案(留 10% 余量给未建模的开销)——这个"从简单策略开始尝试,不满足再升级"的顺序本身编码了一条工程直觉:**能用更简单的并行策略解决就不用更复杂的**(DP 最简单但最耗显存,TP+FSDP+梯度检查点最省显存但工程复杂度和通信开销都最高),自动化选型时应该优先尝试简单方案。如果所有策略都不满足(如 175B 模型只给 8 张 80GB 卡),`estimate()` 正确返回 `feasible=False` 而不是错误地选中一个实际装不下的方案,`report()` 对应输出 `"n/a (does not fit)"` 而不是尝试打印无意义的时长/成本数字(这是源码历史上真实修复过的 bug,曾经在 INFEASIBLE 分支仍强行格式化 `cost_usd`,触发 `TypeError`)。

**AI 研究场景:** 这类"训练配置估算器"是任何 MLOps/infra 团队的常用内部工具——在真正申请云端 GPU 资源、启动一次可能花费数万甚至数十万美元的训练 run 之前,先用这类估算器快速验证"这个配置到底装不装得下、大概要跑多久多少钱",避免在真实昂贵资源上试错。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/scaling-infra/src")
from capstone_train_estimator import TrainSpec, estimate, report

feasible_spec = TrainSpec(model_size_b=7, seq_len=2048, batch=128,
                           n_token=2e9, n_gpu=8, gpu_vram_gb=80, gpu_tflops=312)
plan = estimate(feasible_spec)
assert plan.feasible is True
assert plan.mem_per_gpu_gb <= 80 * 0.9
assert plan.cost_usd is not None

infeasible_spec = TrainSpec(model_size_b=175, seq_len=2048, batch=2048,
                             n_token=300e9, n_gpu=8, gpu_vram_gb=80, gpu_tflops=312)
plan2 = estimate(infeasible_spec)
assert plan2.feasible is False
assert plan2.strategy == "INFEASIBLE"
assert plan2.cost_usd is None
report_text = report(infeasible_spec, plan2)
assert "n/a" in report_text          # 验证 report() 的可行性分支正确处理 None，不崩溃

print(f"7B@8×80GB: strategy={plan.strategy}, mem={plan.mem_per_gpu_gb:.1f}GB, cost=${plan.cost_usd:.0f}")
print(f"175B@8×80GB: {plan2.strategy}, mem={plan2.mem_per_gpu_gb:.1f}GB(超出预算)")
```

**实测(`.venv` 真跑):** 4 个内置场景全部按预期运行:1.5B@1×24GB → 选中 `TP=8+FSDP+grad ckpt`(mem=5.5GB,成本$111);7B@8×80GB → `FSDP+grad ckpt`(mem=27.7GB,20.8小时,$249);70B@64×80GB → `FSDP+grad ckpt`(mem=56.1GB)但训练时长算出 **194756.1 小时**(约 22 年!)—— 这不是 bug,是这组测试用例本身刻意构造了"内存装得下但算力严重不足"的场景(64 张卡训 70B 模型吃 15T token,GPU 数量相对训练量级明显偏少),estimator 诚实地把这个不现实的时长算出来而不是静默截断或报错,恰好演示了"feasible(装得下)"和"practical(合理时间内能训完)"是两个独立维度,装得下不等于这个配置是个好选择;175B@8×80GB → 正确判定 `INFEASIBLE`(最省策略仍需 342.1GB > 80×8×0.9),`report()` 输出含 `n/a (does not fit)`,未复现历史上的 `TypeError` 崩溃。

**面试怎么问 + 追问链:**
- **Q:** "70B 模型那个案例算出 22 年的训练时长,这说明这个估算器有 bug 吗?"—— 期望:不是 bug,是诚实的计算结果——这个案例的输入参数(64 GPU 训 70B 模型吃 15T token)本身在现实中就是不合理的资源配置(真实 70B 模型训练通常用远超 64 张卡的集群),estimator 的职责是如实反映"给定这组参数,数学上算出来的时长是多少",不应该自作主张地对不合理输入做静默修正或报错拦截——这恰恰是好的估算工具应有的行为:诚实暴露问题("这个配置不现实"要用户自己从结果判断出来),而不是隐藏问题。
- **追问1:** "`estimate()` 用 `mfu_target=0.45` 这个固定值算吞吐,这个假设在什么情况下会失真?"—— 期望:0.45 是一个"良好配置"下的经验假设(知识点 9 提到真实 MFU 通常 30%-55%),但如果 estimator 选中的策略本身通信开销很大(比如 TP=8+FSDP 组合),实际达到的 MFU 可能显著低于 0.45,导致时长/成本被低估;这提示这类估算器给出的是"理想情况下的下界估计",不是精确预测,实际项目立项前的预算规划通常需要在这个估算基础上再打一个安全折扣。

**常见坑:** `estimate()` 的策略选择逻辑是"选第一个能装进显存的",不是"选性价比最高的"——比如某个场景可能 `ZeRO-3/FSDP`(无梯度检查点)和 `FSDP+grad ckpt` 都能装进显存,前者因为不做梯度检查点、重计算开销更低,理论吞吐应该更高,但 `for...break` 遇到第一个满足条件的就停止,不会继续比较后续策略是否有更好的吞吐表现——如果需要"在所有可行方案里选最优"而不是"选第一个可行方案",需要修改这个提前 break 的逻辑,遍历完全部策略再比较。

---

*下一篇:[03-pretraining-recipe.md](03-pretraining-recipe.md) —— 数据和并行策略都齐备后,一份完整预训练配方长什么样,含本系列第一处真实 GPU 训练验证。*
