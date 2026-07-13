# 04 · Quantization Deploy 深挖(推理期量化部署全谱)

> 总览见 [00-roadmap.md](00-roadmap.md)

[peft-deep-dive/02-quantized-lora.md](../peft-deep-dive/02-quantized-lora.md) 讲过 NF4 怎么在**训练**阶段把冻结的底座权重压小、腾出显存做 LoRA 微调——那是"量化服务于训练"。本文讲的是同一批数学工具在**推理部署**阶段的另一种用法:模型已经训好了,现在要塞进更小的显存、跑得更快,精度不能掉太多,该怎么选量化方案。本文是 `inference-serving-deep-dive` 系列第 4 篇,对应 `learning/quantization-deploy/`(Module 5《用大模型》第 4 专题,13 lectures + 10 个 src 源文件),从 int8 的三种粒度讲起,走过 GPTQ(二阶 Hessian 补偿)→ AWQ(激活感知缩放)→ SmoothQuant(离群值搬家)→ LLM.int8()(离群通道拆分,概念性)→ FP8 格式,再到 KV cache 量化和"量化动物园"capstone(6 种量化器真跑在同一个 toy 权重层上)收尾。11 个知识点:1(量化全图/4 维度)→2(int8 三粒度)→3(GPTQ)→4(AWQ)→5(SmoothQuant)→6(LLM.int8(),概念性)→7(FP8 格式)→8(FP8 训练,概念性)→9(W4A16/W4A8,NF4)→10(KV Cache 量化)→11(评测方法论+Capstone)。

**和 00-roadmap.md 差异化声明的关系:** 严格遵守 roadmap 给出的知识点划分,W4A16 和 W4A8(L09+L10)因为在这个仓库里共享同一份 `bnb_int4.py` 实现(L10 自己的 lecture 原话是"教学版略,参考 awq_minimal 的扩展"——即没有独立代码),合并成知识点 9。

**一个重要的诚实标注(和 [peft-deep-dive/02](../peft-deep-dive/02-quantized-lora.md) 的关系):** `bnb_int4.py`(本文知识点 9)和 peft-deep-dive 知识点 2-4(`nf4_quant.py`/`qlora_minimal.py`/`qlora_peft.py`)都实现了 NF4(NormalFloat 4-bit)量化,但**是两份完全独立的重新实现,互不引用**——已经过独立核实,两者除了共享"NF4 16 个分位码"这个数学思想之外没有代码层面的重复。核心区别是使用场景:peft-deep-dive 的 NF4 服务于"冻结底座+LoRA 微调"这个**训练时**流程,本文的 NF4(以及 W4A16/W4A8)服务于"模型训好后怎么部署"这个**推理时**流程,两者是模型生命周期里两个不同阶段用到的同一种数学工具,不是重复内容。

**另一个重要诚实标注(源材料自身的文档滞后):** `learning/quantization-deploy/README.md` 已经明确记录 `capstone_quant_zoo.py` 被重写过——"旧版本是一张硬编码 paper 数字表,未跑任何量化——已重写"——但 `lectures/13-capstone-quant-zoo.md` 这篇 lecture 原文**没有同步更新**,第 5 节仍然写着"与真实数字相符(取自 paper)"这种描述旧版本行为的话。本文以 README 和实际源码(已逐行核实)为准:knowledge point 11 会展示 capstone **真的**调用 6 个量化器、真的算输出重建 MSE,不是查表。这个"模块 README 已修正,但同模块的某篇 lecture 没跟着改"的现象本身,也是精读代码仓库时值得留心的一类坑——不能假设同一个模块内部所有文档永远互相同步。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13.9,torch 2.11.0+cu128)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出,纯 CPU 张量运算,不需要 GPU、不需要 `bitsandbytes`/`auto-gptq`/`autoawq` 等重型依赖(这些库在源模块 `requirements.txt` 里列出,但 10 个 `src/*.py` 文件全部是独立手写实现,零真实 import,已逐文件 grep 确认)。

---

## 1. 量化全图与 4 个选择维度(L01)—— 用精度换显存,用 kernel 换速度

**是什么:** 本知识点没有对应的 `src/*.py` 文件(README 总览表格 L01 那一行"代码"列是 `—`),内容是纯概念性的框架搭建,"可运行例子"部分改用简单算术复核 lecture 给出的显存数字是否站得住脚。

**一句话:** 量化要做的选择可以拆成 4 个互相独立的维度——**对象**(量 weight 还是 activation 还是 KV cache)、**精度**(int8/int4/fp8/fp4 这几档具体编码)、**粒度**(per-tensor/per-channel/per-group,精度和速度的权衡)、**时机**(训练后量化 PTQ,还是训练时就考虑量化 QAT)——后面 10 个知识点里每一种具体方法,本质上都是在这 4 个维度上做出的一组具体选择。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么要在意模型占多少显存?因为显存是推理服务能不能跑起来、能跑多大 batch 的硬约束(01 号文件知识点 2 已经讨论过 KV cache 的显存问题,本文关注的是**weight** 本身的显存)。fp16 存一个参数要 2 字节,int4 只要 0.5 字节,直接是 4 倍压缩——lecture 给出的例子是 Llama-7B fp16 权重 14GB,int4 只要 3.5GB;Llama-70B fp16 需要 140GB(单卡完全装不下),int4 只要 35GB(单张 A100 80GB 或者两张 24GB 卡就能考虑)。但压缩比不是唯一目标,4 个维度互相牵制:量 weight 最简单(weight 训完就固定了,可以慢慢做 PTQ 校准),量 activation 更难(每次推理输入都不同,需要动态量化或者提前搬走离群值,这是知识点 5 SmoothQuant 要解决的问题);粒度越细(per-group 比 per-channel 比 per-tensor)精度越高但要存的 scale 越多、kernel 也越复杂;PTQ 省事但有精度上限,QAT(训练时模拟量化)精度更高但要重新过一遍训练数据,成本高得多——本专题全部 10 个方法都是 PTQ,QAT 不在范围内。

**AI 研究场景:** lecture 给出的"量化的精度损失"参考表(int8 PPL 只涨约 1%,GPTQ/AWQ 4bit 涨 2-3%,W4A8 涨约 5%)说明一个重要事实:2024-2026 年的量化技术已经把"几乎无损"和"4 倍压缩"这两件事同时做到了(相比 2022 年之前"量化=明显掉点"的普遍印象),这也是为什么 05/06 号文件会讨论的分布式推理、生产部署里,量化已经从"精度妥协的最后手段"变成"几乎默认打开的选项"。

**可运行例子:**
```python
def model_mem_gb(n_params, bytes_per_param):
    return n_params * bytes_per_param / 1e9

llama7b = 7_000_000_000
llama70b = 70_000_000_000

fp16_7b = model_mem_gb(llama7b, 2.0)   # fp16 = 2 bytes/param
int4_7b = model_mem_gb(llama7b, 0.5)   # int4 = 0.5 byte/param (4 bit)
fp16_70b = model_mem_gb(llama70b, 2.0)
int4_70b = model_mem_gb(llama70b, 0.5)

assert abs(fp16_7b - 14.0) < 0.01 and abs(int4_7b - 3.5) < 0.01
assert abs(fp16_70b - 140.0) < 0.01 and abs(int4_70b - 35.0) < 0.01
assert fp16_7b / int4_7b == 4.0   # int4 相对 fp16 精确 4 倍压缩,不是约数
```

**实测(`.venv` 真跑):** `7e9 参数 × 2 字节 = 14.0 GB`(fp16)、`× 0.5 字节 = 3.5 GB`(int4),和 lecture L01 给出的数字精确一致;`70e9` 参数同理精确得到 `140.0 GB`/`35.0 GB`。压缩比精确为 `4.0` 倍——这不是量化算法本身的功劳(不同量化方法在同样 4bit 下压缩比都一样),纯粹是"用几分之一的比特数存一个数"这个信息论意义上的算术结果,量化方法之间真正的差异在于"用相同的比特预算,谁的精度损失更小"(这是知识点 3-9 要回答的问题)。

**面试怎么问 + 追问链:**
- **Q:** "为什么量化能同时省显存又提速?" —— 期望说出"省显存是直接的信息论结果(更少比特存每个数);提速主要来自 01 号文件讨论过的'decode 阶段是 memory-bound'——weight-only 量化减少了每步要从显存搬运的字节数,这是主要提速来源;如果连 activation 也量化(W+A 量化),matmul 本身能用更快的低精度 kernel 计算,这是第二重提速"。
- **追问 1:** "量化的 4 个维度里,哪个维度对精度影响最大?" —— 期望能说出"精度"(int8 vs int4 vs更低)和"粒度"(per-tensor vs per-group)通常影响最直接,但更准确的答案是"取决于具体张量的数值分布"——如果某个 tensor 有显著的离群值,粒度不够细(比如 per-tensor)会让精度断崖式下降,这是知识点 4/5/6 分别用不同思路解决的同一个核心痛点。
- **追问 2:** "PTQ 和 QAT 除了成本不同,还有什么本质区别?" —— 期望说出"QAT 在训练时就让模型'感知'到量化会引入的误差(前向用量化后的模拟值,反向仍用全精度梯度更新),模型的权重分布会主动适应量化网格;PTQ 是训练完全不知道以后会被量化,量化算法只能在训练好的固定权重上想办法把误差降到最低,这也是为什么 GPTQ/AWQ 这类 PTQ 方法要在'怎么补偿误差'上做精细设计"。
- **追问 3:** "如果显存完全不是瓶颈(比如单卡就能装下 fp16),还有必要量化吗?" —— 期望能辩证回答:如果纯粹看单请求延迟,01 号文件讨论过的"weight-only 量化让 decode 加速 1.5-3x"这个收益在 memory-bound 场景下依然成立,和显存是否紧张是两件独立的事;但如果服务的是大 batch/高吞吐场景,量化对吞吐的边际收益会变小(01 号文件知识点 5 讨论过的"batch 越大越接近算力打满"同一个道理),需要具体测。

**常见坑:** 把"量化压缩比"和"精度损失"看成同一个维度上的东西、认为"压得越狠精度损失一定越大"——这在同一种量化算法内部大致成立(比如同一个 GPTQ,4bit 比 8bit 损失更大),但跨算法比较时不成立:AWQ 4bit 的精度损失(lecture 数字 PPL+2%)可以比一个设计粗糙的 int8 方案更小,压缩比和精度损失是两个可以被算法设计独立优化的维度,不能只看比特数下结论。

---

## 2. int8 基础:per-tensor / per-channel / per-group(`int8_basics.py`,L02)—— 粒度越细,scale 数越多,精度越高

**是什么:**
```python
def quantize_per_tensor(x: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, float]:
    """Symmetric int8 quantization with a single scale."""
    qmax = (1 << (n_bits - 1)) - 1
    scale = x.abs().max().item() / max(qmax, 1)
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale
```
(`int8_basics.py:7-12`)

```python
def quantize_per_group(x: torch.Tensor, group_size: int = 128, n_bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric quantization along last dim."""
    qmax = (1 << (n_bits - 1)) - 1
    *prefix, K = x.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    g = K // group_size
    x_grouped = x.reshape(*prefix, g, group_size)
    scale = x_grouped.abs().amax(dim=-1, keepdim=True) / max(qmax, 1)
    scale = scale.clamp(min=1e-9)
    q = (x_grouped / scale).round().clamp(-qmax, qmax)
    return q.reshape(*prefix, K).to(torch.int8), scale.squeeze(-1)
```
(`int8_basics.py:35-45`)

**一句话:** 三个量化函数的核心区别只在于"一个 scale 覆盖多大范围的数":`quantize_per_tensor` 整个张量共用 1 个 scale(最粗、最快、精度最差),`quantize_per_channel` 每个输出通道一个 scale,`quantize_per_group` 把每一行再切成若干个 `group_size` 大小的小段、每段一个 scale(最细、scale 数量最多、精度最高但存储/计算开销也最大)。

**底层机制/为什么这样设计:** 从最笨的想法讲起——对称量化的核心公式很简单:找到这批数里绝对值最大的那个 `max(|x|)`,除以量化范围能表示的最大整数 `qmax`(int8 是 127),得到 `scale`,量化就是把每个数除以 `scale` 再四舍五入。这个公式的精度完全取决于 `scale` 覆盖的这批数里"最大值和大多数值差多远"——如果一批数里绝大多数集中在 `[-1,1]`,但有一个孤立的 `100`,那 `scale` 就要被这个 `100` 撑大(`scale=100/127≈0.79`),这时候原本在 `[-1,1]` 范围内的数经过 `round(x/0.79)` 大概率全部量化成 `0` 或 `±1`,信息几乎全丢了——这就是"离群值撑大 scale、拖累其余大多数正常值精度"这个 02 号文件(sglang 系列)以来反复出现的"一粒老鼠屎坏一锅粥"式问题在量化领域的具体表现。`quantize_per_channel` 把"一批数"的范围从"整个矩阵"缩小到"一行/一列",如果离群值只集中在某几个通道,不会拖累其他通道;`quantize_per_group` 再把"一行"继续切细,进一步缩小"一个 scale 要覆盖的数值范围"。粒度越细,越能把离群值的破坏范围限制在越小的区域内,代价是要存的 `scale` 数量线性增长(per-tensor 存 1 个数,per-channel 存 `out_features` 个数,per-group 存 `out_features × in_features/group_size` 个数),而且 kernel 实现也更复杂(要按 group 分别处理,不能整块一次性算完)。

**AI 研究场景:** lecture L02 明确指出 LLM weight 通常用 symmetric(权重分布大致关于 0 对称,不需要额外的 zero-point),activation 通常用 asymmetric(激活值可能整体偏正,比如 ReLU 之后全是非负数,加一个 zero-point 能显著提升精度)——这不是任意选择,是根据两类张量各自的真实数值分布特征做出的针对性设计。粒度选择上,vLLM/SGLang 生产系统里 GPTQ/AWQ 常用 `group_size=128` 作为精度和效率的折中点(04 号文件知识点 3/4 会具体展开),不是越细越好——粒度太细会让 scale 存储本身占用的显存和 kernel 里频繁切换 scale 的开销超过精度提升带来的收益。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from int8_basics import (quantize_per_tensor, dequantize_per_tensor,
                          quantize_per_channel, dequantize_per_channel,
                          quantize_per_group, dequantize_per_group, mse)

torch.manual_seed(123)
# 32 个输出通道，幅度从 1x 依次涨到约 10.3x(制造真实的跨通道幅度差异)
W = torch.randn(32, 256)
for row in range(32):
    W[row] *= (1 + row * 0.3)

qt, st = quantize_per_tensor(W)
dqt = dequantize_per_tensor(qt, st)
qc, sc = quantize_per_channel(W, axis=0)
dqc = dequantize_per_channel(qc, sc)
qg, sg = quantize_per_group(W, group_size=64, n_bits=8)   # 256/64 = 4 组/行
dqg = dequantize_per_group(qg, sg, group_size=64)

mse_t, mse_c, mse_g = mse(W, dqt), mse(W, dqc), mse(W, dqg)
assert mse_t > mse_c   # per-channel 精度必须优于 per-tensor
assert mse_c >= mse_g * 0.99   # per-group 不应该比 per-channel 更差
```

**实测(`.venv` 真跑):** 32 行、每行幅度从 1x 到约 10.3x 递增(专门制造跨通道幅度差异)的张量上,`per-tensor` MSE 是 `0.00707`,`per-channel` 降到 `0.00195`(约 3.6 倍改善),`per-group(64,8bit)` 进一步降到 `0.00137`——三者精确按"粒度越细、精度越高"的顺序排列,和 lecture L02 给出的"per-tensor 差、per-channel 中、per-group 高"定性结论完全吻合。

**面试怎么问 + 追问链:**
- **Q:** "per-tensor、per-channel、per-group 量化,scale 的数量分别是多少?" —— 期望说出"per-tensor 是 1 个(整个张量共用);per-channel 是 `out_features` 个(每个输出通道一个);per-group 是 `out_features × in_features/group_size` 个(每行再按 group_size 切段,每段一个)"。
- **追问 1(考察是否理解离群值的破坏机制):** "如果一个张量只有一个元素是离群值,其余全是正常范围,per-channel 量化一定能解决问题吗?" —— 期望说出"不一定——如果这个离群值和大多数正常值处在同一个输出通道里,per-channel 量化的 scale 依然会被它撑大,同通道内其他值照样遭殃;per-channel 只能解决'离群值集中在少数几个通道'这种情况,如果离群值和正常值混在同一行内,需要更细的 per-group,或者像知识点 4/5/6 那样专门处理离群值"。
- **追问 2:** "per-group 量化比 per-channel 精度更高,那为什么不是所有场景都用尽可能细的 group_size(比如 group_size=1)?" —— 期望说出"group_size=1 等价于给每个权重单独存一个 scale,存储开销会超过权重本身节省下来的显存(4bit 权重 + fp32 scale 反而比 fp16 权重还占地方),而且 kernel 每次都要按 group 切换 scale,访存模式变得零碎,速度会明显下降——`group_size=128` 是工业界摸索出的精度/开销平衡点,不是越小越好"。
- **追问 3:** "为什么 LLM 权重通常用 symmetric 量化,而不是 asymmetric?" —— 期望说出"symmetric 不需要额外存一个 zero-point、kernel 实现更简单更快;权重经过训练后的分布通常大致关于 0 对称(没有 activation 那种因为 ReLU/GELU 之类激活函数导致的系统性偏移),用 symmetric 损失的精度相对可以接受,这是'精度换简单性'的一个合理权衡,不是唯一正确答案"。

**常见坑:** 把"per-group 量化"的 `group_size` 理解成"沿着任意维度切"——本知识点的实现明确是沿着**最后一个维度**(`x.reshape(*prefix, g, group_size)`)切分,如果权重矩阵的维度顺序和预期不一致(比如以为是沿 `out_features` 切,实际代码是沿 `in_features` 切),会直接导致 `K % group_size == 0` 断言失败或者切分方式和预期完全不同。另一个坑是认为三种粒度的量化，**推理时的计算方式**也不同——实际上三者的核心公式(`round(x/scale).clamp(...)`)完全一样,唯一区别是"scale 的形状"和"reshape 的方式",这也是为什么这三个函数的实现高度相似、可以放在同一个文件里对照着看。

---

## 3. GPTQ(`gptq_minimal.py` + `gptq_original_minimal.py`,L03)—— 二阶 Hessian 误差补偿,以及一个关于 `damp` 正则化强度的真实数值稳定性发现

**是什么:**
```python
def gptq_columnwise(
    W: torch.Tensor, X: torch.Tensor, n_bits: int = 4, damp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize W with the core GPTQ column-by-column compensation rule."""
    W_work = W.float().clone()
    out_dim, in_dim = W_work.shape
    grid = make_symmetric_row_grid(W_work, n_bits=n_bits)
    H = calibration_hessian(X, damp=damp)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    Q = torch.zeros_like(W_work)
    E = torch.zeros_like(W_work)
    for j in range(in_dim):
        q_col = quantize_column(W_work[:, j], grid)
        err = (W_work[:, j] - q_col) / H_inv[j, j]
        Q[:, j] = q_col
        E[:, j] = err
        if j + 1 < in_dim:
            W_work[:, j + 1 :] -= torch.outer(err, H_inv[j, j + 1 :])
    return Q, E
```
(`gptq_original_minimal.py:66-99`)

**一句话:** GPTQ 逐列量化权重矩阵,每量完一列就把这一列的量化误差(除以 `H_inv[j,j]` 归一化后)反向摊给**尚未量化的**后续列,用校准数据估计出的二阶信息(Hessian `H=X^T X`)指导"该往哪个方向、摊多少"这个误差补偿过程;`damp` 是加在 Hessian 对角线上的正则化强度,本知识点独立验证发现:**默认 `damp=0.01` 在某些校准数据的相关性/维度组合下不够,会导致 GPTQ 的补偿机制不仅没有改善精度,反而比完全不做补偿的朴素四舍五入(RTN,round-to-nearest)更差**。

**底层机制/为什么这样设计:** 从最笨的想法讲起——如果只是把权重矩阵 `W` 的每个元素独立四舍五入到最近的量化格点(朴素 RTN),每个元素的量化误差是独立、无偏的随机噪声,但这些误差在下游"这一层的真实输出 `Wx`"上会累积成一个系统性的偏差。GPTQ(Frantar 2023)的核心创新是:与其独立量化每一列,不如按顺序**逐列**量化,每量完一列,立刻计算这一列引入的误差,然后用这批**校准数据的二阶统计信息**(Hessian `H=X^T X`,直觉上刻画了"权重矩阵不同列之间,在真实输入分布下的相关性")算出应该怎么调整**剩余未量化列**来抵消这一列刚引入的误差——这是"贪心 + 事后补偿"的组合。这个补偿公式的数学基础是最优脑外科手术(OBS,Optimal Brain Surgeon)理论,本质是在解一个约束优化问题:每量化一个权重,如何微调其余权重让"层输出的变化量"最小。这套机制的正确运作**依赖 `H_inv`(Hessian 逆矩阵)数值稳定**——`damp` 参数的作用是在求逆前给 `H` 的对角线加一点正则化(`H += damp * H.diag().mean() * I`),防止 `H` 接近奇异导致求逆时数值爆炸。本知识点独立复现时发现:用不同于内置 `_self_test()`/`toy_comparison()` 默认配置(`seed=0`,`W: 24×48`,`mixing_strength=0.15`)的另一组参数(`seed=55`,`W: 40×80`,`mixing_strength=0.2`,即校准激活的列间相关性更强、维度更高),在**默认** `damp=0.01` 下,GPTQ 的重建 MSE(`4.4525`)反而**高于**朴素 RTN(`4.1378`)——这不是这一个种子的偶然,20 个不同随机种子的批量测试里有 18 个都复现了"GPTQ 更差"这个现象。深挖发现根因和 `damp` 直接相关:同样的权重矩阵和校准数据,只把 `damp` 从 `0.01` 调到 `0.1`,GPTQ 的重建 MSE 立刻降到 `2.7294`,稳定优于朴素 RTN(独立测试 10 个种子,10/10 都在 `damp=0.1` 下反超朴素 RTN)——说明问题根源是"默认 `damp=0.01` 这个正则化强度,相对这组更高维、列间相关性更强的校准数据,不足以让 `H_inv` 保持数值稳定,导致误差补偿在传播过程中被放大而不是被抵消"。这是这份教学代码"忠实复刻了 GPTQ 论文机制"这件事的一体两面:它精确复现了真实 GPTQ 实现里"`damp` 是一个需要根据具体层/校准数据调整的敏感超参数,不是一个可以无脑固定的常数"这个工程现实,而不是一个只在理想条件下工作的简化演示。

**AI 研究场景:** 这个发现直接对应真实 GPTQ 部署里的一条工程经验:量化大模型的不同层时,`percdamp`(真实 auto-gptq 库里对应参数的名字)往往需要按层甚至按模型调整,不能全模型用同一个固定值——层的宽度、深度、校准数据分布的差异都会影响 Hessian 的条件数。lecture L03 提到"GPTQ 量化慢、calibration 数据敏感"这两条已知缺点,本知识点的发现是"calibration 数据敏感"这条缺点的一个具体、可复现、有诊断方法(检查 `damp` 是否需要调大)的实例,不只是一句抽象的警告。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from gptq_original_minimal import naive_round, gptq_columnwise, reconstruction_mse

# 独立于文件自带 toy_comparison()(seed=0, W: 24x48, mixing=0.15)的另一组配置
torch.manual_seed(55)
W = torch.randn(40, 80)
Z = torch.randn(320, 80)
mixing = torch.eye(80) + 0.2 * torch.randn(80, 80)   # 更强的列间相关性
X = Z @ mixing

naive = naive_round(W, n_bits=4)
mse_naive = reconstruction_mse(W, naive, X)

gptq_default_damp, _ = gptq_columnwise(W, X, n_bits=4, damp=0.01)   # 默认 damp
mse_gptq_default = reconstruction_mse(W, gptq_default_damp, X)
assert mse_gptq_default > mse_naive   # 真实发现:默认 damp 下 GPTQ 反而更差

gptq_higher_damp, _ = gptq_columnwise(W, X, n_bits=4, damp=0.1)   # 调大 damp
mse_gptq_higher = reconstruction_mse(W, gptq_higher_damp, X)
assert mse_gptq_higher < mse_naive   # 调大 damp 后稳定反超朴素 RTN
```

**实测(`.venv` 真跑):** `seed=55, W: 40×80, mixing_strength=0.2` 配置下,朴素 RTN 重建 MSE 为 `4.1378`,默认 `damp=0.01` 的 GPTQ 反而是 `4.4525`(比朴素 RTN 差 `7.6%`)——20 个随机种子批量复测,18/20 复现"GPTQ 更差"。把 `damp` 调到 `0.1`(其余参数不变),GPTQ MSE 降到 `2.7294`,反超朴素 RTN(改善 `34%`)——10 个种子批量复测,10/10 稳定反超。这个发现进一步在知识点 11 的真实 capstone 上得到印证:capstone 自己的 toy 层构造(`seed=0` 默认配置)下 GPTQ 稳定优于 NF4(10/10 个种子 0-9 都成立,符合 README 的宣称),但换成 `seed=99` 时,同样用默认 `damp=0.01` 的 capstone `_q_gptq_4bit` 也会反过来输给 NF4——同一个数值稳定性问题,在两个独立的实验设置里都被复现。

**面试怎么问 + 追问链:**
- **Q:** "GPTQ 相比朴素逐元素量化,核心改进是什么?" —— 期望说出"逐列量化,每量完一列用校准数据估计出的二阶 Hessian 信息,把这一列的量化误差补偿到剩余未量化的列上,而不是让每一列的误差独立累积"。
- **追问 1(核心陷阱,考察是否真的独立验证过):** "GPTQ 的误差补偿机制是不是在任何校准数据上都能稳定改善精度?" —— 期望明确说"不是——本知识点独立发现,在校准数据列间相关性较强、维度较高的一组配置下,默认的 `damp=0.01` 正则化强度不足以保证 Hessian 求逆的数值稳定性,GPTQ 反而会比完全不做补偿的朴素四舍五入更差(20 个种子里 18 个复现);把 `damp` 调大到 0.1 能稳定修复这个问题"。这道题专门筛"只会背'GPTQ 用 Hessian 补偿所以更准'这句话、没有意识到这依赖数值稳定性前提"的候选人。
- **追问 2:** "`damp` 这个正则化参数,调得越大是不是越安全?" —— 期望能辩证回答:`damp` 越大,`H` 越接近纯对角矩阵(即"完全不用二阶信息、退化成朴素量化"的极限情况),数值稳定性确实越好,但也意味着放弃了 GPTQ 真正想利用的"列间相关性"这个信息,`damp` 太大会让 GPTQ 的补偿效果趋近于朴素 RTN、损失掉它本该有的优势——`damp` 本质上是"数值稳定性"和"补偿精度上限"之间的权衡,不是越大越好。
- **追问 3:** "如果你在生产环境里量化一个新模型,发现某一层用 GPTQ 效果反而变差了,你会怎么排查?" —— 期望能提出结构化排查思路:先检查 `damp`/正则化强度是否对这一层的校准数据条件数合适(本知识点已经验证过的思路);检查校准数据本身是否有代表性(样本量是否够、是否覆盖了这一层真实推理时会遇到的输入分布);检查 Hessian 计算本身有没有数值问题(比如某些维度上校准数据方差过小导致病态)。

**常见坑:** 把 GPTQ 论文/lecture 给出的"几乎无损"这类精度数字当成放之四海而皆准的保证——那些数字是在特定模型(Llama-7B)、特定校准数据集、特定超参数下测出来的,本知识点已经证明同一个算法在不同校准数据条件下可以表现出方向相反的结果(比朴素方法更好 vs 更差),量化任何新模型/新层之前都需要验证,不能假设论文数字自动适用。另一个坑是遇到"GPTQ 效果不好"就直接归咎于算法本身有缺陷、转而放弃使用——本知识点的发现恰恰说明问题往往出在一个具体、可调的超参数(`damp`)上,值得先排查这一层,而不是急着换算法。

---

## 4. AWQ(`awq_minimal.py`,L04)—— 不量化激活,靠给权重"整容"来保护显著通道

**是什么:**
```python
def search_scales(
    W: torch.Tensor, X: torch.Tensor, n_bits: int = 4, grid: int = 20,
) -> torch.Tensor:
    """Grid-search per-channel scale s ∈ (0, 1]; minimise ||(W s) X(1/s) - W X||²."""
    out_dim, in_dim = W.shape
    x_mean = X.abs().mean(dim=0).clamp(min=1e-9)
    best_s = torch.ones(in_dim, device=W.device)
    best_err = float("inf")
    ref = W @ X.t()
    for ratio in [1.0 - i / grid for i in range(grid)]:
        s = x_mean.pow(ratio).clamp(min=1e-3)
        W_s = W * s
        Wq_s, scale = quantize_per_channel(W_s, axis=0, n_bits=n_bits)
        W_dq = dequantize_per_channel(Wq_s, scale) / s
        approx = W_dq @ X.t()
        err = float(((approx - ref) ** 2).mean().item())
        if err < best_err:
            best_err = err
            best_s = s.clone()
    return best_s
```
(`awq_minimal.py:8-31`)

**一句话:** AWQ 的关键洞察是"重要的不是权重本身多准,而是层的输出 `Wx` 多准"——`search_scales` 用网格搜索找一个per-channel 缩放向量 `s`,让"权重乘 `s` 再量化、激活除以 `s`"这套变换后的结果,尽量逼近未量化时的原始输出 `W@X`,`s` 的搜索范围由激活幅度 `x_mean` 决定,天然会给"激活幅度大的通道"分配更大的缩放,变相保护这些通道对应的权重精度。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 2 已经建立了"离群值撑大 scale、拖累其他值精度"这个直觉,LLM.int8()(知识点 6)的解法是"把离群通道单独摘出来用 fp16 算";AWQ 换了个思路:**不改动数值本身该有的精度分布,而是让权重在被量化之前先"变形"**。数学恒等式很直接:`y = W·x = (W·diag(s))·(diag(1/s)·x) = W_s · x_s`——这是一个纯粹的重参数化,只要 `s` 处处非零,这个变换在**不量化**的情况下是精确等价的(本知识点在知识点 5 SmoothQuant 里会验证同类型变换的精确不变性)。AWQ 的技巧在于:量化只发生在 `W_s` 上(`x_s` 保持 fp16 不量化,这是 AWQ 和知识点 5 SmoothQuant 的本质区别——AWQ 不动激活),如果给"激活幅度大的通道"分配一个**更大**的 `s`,`W_s` 里对应那一列会被放大,量化时这一列的相对量化误差(误差占该列数值大小的比例)会变小;`x_s` 除以同样的 `s` 变小,但 `x_s` 是 fp16、不损失精度。等价于"用一个几乎无损的操作,把量化预算集中倾斜给激活值大、对最终输出影响大的那些权重列"。`search_scales` 用网格搜索(`s = x_mean^ratio`,`ratio` 从 1 降到 0,共 `grid` 个候选)而不是解析求解,是因为这个优化目标(量化后重建误差最小)本身涉及不可导的取整操作,网格搜索是这份教学实现选择的简化处理方式,真实 AWQ 实现会做更精细的搜索(比如对 `ratio` 做更细网格,或者按 group 分别搜索)。

**AI 研究场景:** lecture L04 强调 AWQ"不需要 activation 量化"这一点直接决定了它的部署简易性——因为最终参与 matmul 的两个操作数里,`x_s` 依然是标准 fp16,推理 kernel 只需要处理"int4 权重 dequant 后和 fp16 激活相乘"这一件事,不需要像 SmoothQuant/W4A8 那样额外处理"两个低精度操作数相乘"的复杂 kernel,这也是为什么 lecture 给出 AWQ 精度(MMLU 44.9)比 GPTQ(44.5)略高、部署 kernel 评价是"极佳"——04 号文件(本文自己所在系列)的决策树也把 AWQ-4bit 列为"最小显存"的推荐选项之一。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from int8_basics import quantize_per_channel, dequantize_per_channel
from awq_minimal import search_scales, awq_quantize

torch.manual_seed(7)
W = torch.randn(20, 64)
X = torch.randn(200, 64)
salient = [3, 40, 55]      # 3 个显著通道(和 capstone 用的 (7,33,91) 不同)
X[:, salient] *= 15.0      # 幅度放大 15 倍

s = search_scales(W, X, n_bits=4)
non_salient = [i for i in range(64) if i not in salient][:3]
assert (s[salient].mean() - s[torch.tensor(non_salient)].mean()).abs() > 0.01   # 显著通道确实拿到不同的缩放

Wq, s_final = awq_quantize(W, X, n_bits=4)
ref = W @ X.t()
approx_awq = (dequantize_per_channel(Wq, quantize_per_channel(W * s_final, axis=0, n_bits=4)[1]) / s_final) @ X.t()
err_awq = float(((approx_awq - ref) ** 2).mean().item())

naive_q, naive_s = quantize_per_channel(W, axis=0, n_bits=4)   # 不做 AWQ 缩放的朴素 per-channel int4
naive_dq = dequantize_per_channel(naive_q, naive_s)
err_naive = float(((naive_dq @ X.t() - ref) ** 2).mean().item())

assert err_awq < err_naive   # AWQ 的激活感知缩放确实降低了输出重建误差
```

**实测(`.venv` 真跑):** 3 个显著通道(幅度放大 15 倍)、其余通道正常幅度的构造下,AWQ 搜索出的显著通道缩放均值(约 `2.11-2.16`)明显不同于非显著通道的缩放中位数(约 `0.93`)——确认 AWQ 确实"认出"了哪些通道更重要并给了不同处理。输出重建 MSE 对比:AWQ 是 `3.5559`,不做激活感知缩放的朴素 per-channel int4 是 `10.6694`——AWQ 的误差只有朴素方案的约 `1/3`,验证了"给重要通道让路"这个策略的实际效果,不只是理论上说得通。

**面试怎么问 + 追问链:**
- **Q:** "AWQ 和 GPTQ 都是给权重做 4bit 量化,核心思路有什么不同?" —— 期望说出"GPTQ 用二阶 Hessian 信息,量化一列后把误差补偿到后续列;AWQ 不做误差补偿,而是提前对权重做一次和激活幅度相关的缩放,让重要通道(激活幅度大的)在量化前被'放大',相对量化误差变小,量化本身还是朴素的 round-to-nearest"。
- **追问 1(核心陷阱):** "AWQ 的 per-channel 缩放 `s`,是只根据权重本身的分布算出来的吗?" —— 期望明确说"不是,`s` 依赖 `X`(校准激活),`x_mean = X.abs().mean(dim=0)` 是缩放搜索的输入之一——这正是'activation-aware'这个名字的来源,如果只看权重分布、不看激活分布,就无法知道哪些通道在真实推理时的输入幅度更大、更值得保护"。
- **追问 2:** "为什么 AWQ 要用网格搜索选 `s`,而不是直接解析算出最优值?" —— 期望说出"因为优化目标里包含 `round()` 这个不可导操作(量化本身就是取整),没有解析解,网格搜索是在有限候选里找一个近似最优,这份教学实现只在 `ratio ∈ [0,1]` 的一维网格上搜(每次搜出一个整体缩放的'形状参数'),真实 AWQ 实现会做更细粒度、按 group 分别搜索的版本"。
- **追问 3:** "AWQ 声称'不需要量化 activation',这是不是意味着推理时完全不用管激活精度?" —— 期望说出"不是——`x_s = x/s` 这一步在推理时依然要真实执行(除法本身在 fp16 下进行,不损失精度),只是不需要把 `x_s` 进一步量化成低精度整数;'不量化 activation'指的是激活值本身保持 fp16,不是'完全不处理激活'"。

**常见坑:** 把 AWQ 的"per-channel 缩放"和知识点 2 的"per-channel 量化"混为一谈——两者都是"per-channel"操作,但目的完全不同:知识点 2 的 per-channel 量化是给**每个通道单独一个量化 scale**;AWQ 的 `s` 是在量化**之前**对数值做的一次可逆变形(重参数化),之后依然要走(可以是 per-tensor 或 per-channel 的)标准量化流程,`s` 本身不是量化 scale。另一个坑是认为 AWQ 一定比 GPTQ 精度更高——lecture 给出的数字(AWQ MMLU 44.9 vs GPTQ 44.5)是在 Llama-7B 这个特定模型上的结果,本知识点自己的独立实验(以及知识点 11 capstone 的多种子测试)已经证明"哪种方法更准"高度依赖具体权重/校准数据的分布特征,不存在放之四海而皆准的排名。

---

## 5. SmoothQuant(`smooth_quant.py`,L05)—— 把激活离群值搬到权重上,验证一次精确的数学恒等式

**是什么:**
```python
def find_smooth_scale(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """s_k = max(|X_:k|)^α / max(|W_k:|)^(1-α)"""
    x_amax = X.abs().amax(dim=0).clamp(min=1e-9)
    w_amax = W.abs().amax(dim=1).clamp(min=1e-9)
    s = x_amax.pow(alpha) / w_amax.pow(1.0 - alpha)
    return s.clamp(min=1e-3)


def apply_smoothing(X: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Y = X·W = (X / s) · (diag(s)·W)"""
    return X / s, s.unsqueeze(-1) * W
```
(`smooth_quant.py:7-27`)

**一句话:** SmoothQuant 和 AWQ 用的是同一条数学恒等式(`X·W = (X/s)·(s·W)`),但目的相反——AWQ 保持激活不量化、只用这条恒等式保护权重量化精度;SmoothQuant **两边都要量化**(int8 weight + int8 activation),`s` 的作用是把激活里"大到难以量化"的离群值成比例地"搬"到权重上(权重反正对每个通道都有独立的量化余地),让激活和权重的数值范围都变得更"温和"、都更容易被量化。

**底层机制/为什么这样设计:** 从最笨的想法讲起——W8A8(权重和激活都是 int8)在理论上比 W4A16 更快(matmul 两个操作数都是低精度整数,能用最快的 int8 kernel),但实践中卡在激活量化这一步:LLM 的激活值(尤其是经过 LayerNorm 之后送进下一层的那些)天然容易出现某些通道数值特别大的离群模式(知识点 6 LLM.int8() 要解决的同一个现象),直接对激活做 per-tensor int8 量化精度损失明显。SmoothQuant 的解法不是像 AWQ 那样放弃量化激活,而是**先让激活变得更容易量化**:`find_smooth_scale` 给每个通道算一个 `s_k = max(|X_:k|)^α / max(|W_k:|)^(1-α)`——如果某个通道的激活远大于权重(`x_amax` 远大于 `w_amax`),这个通道的 `s_k` 会偏大;`apply_smoothing` 用这个 `s` 把激活除小、权重乘大,相当于把"这个通道数值太大导致难以量化"的负担从激活转移到权重上——权重是静态的,可以做更精细的逐通道量化处理来消化这个负担,而激活的离群不再像原来那样直接冲击运行时的 per-tensor int8 scale。`alpha` 是这个"转移力度"的调节旋钮:`alpha=0.5` 是两边各让一半(lecture 给出的典型选择),`alpha` 越接近 1 越偏向"把负担全转给权重"、越接近 0 越偏向"激活自己扛"。这套变换在**不量化**的情况下是数学上精确等价的:`(X/s)·(s·W) = X·(1/s · s)·W = X·W`,本知识点独立验证了这一点(见下方可运行例子,最大误差在 `1e-5` 量级,属于浮点运算本身的舍入噪声,不是变换引入的系统性偏差)——量化只发生在这个恒等重参数化**之后**,这也是为什么"搬家"这个操作本身不会凭空产生任何精度损失,损失只来自后续真正的量化步骤,而且这一步的损失因为两边数值范围都变"温和"了而被显著压低。

**AI 研究场景:** lecture L05 给出 SmoothQuant W8A8 相比朴素 W8A8 的 PPL 从 6.50 降到 5.75(几乎追平 fp16 的 5.68),同时 matmul 用 int8 CUDA kernel 能拿到约 2 倍加速——这是"精度几乎无损 + 速度实打实提升"的组合,和 AWQ"精度更高但激活不量化、速度提升相对有限(W4A16 约 1.8x)"形成一组直接的工程权衡对比,lecture L05 自己的对照表也点明"两者可以组合 → W4A8"(知识点 9 会展开),这正是量化领域"不同方法不是互斥选项、可以叠加使用各自解决不同环节的问题"这个思路的典型例子。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from int8_basics import quantize_per_tensor, dequantize_per_tensor, mse
from smooth_quant import find_smooth_scale, apply_smoothing

torch.manual_seed(3)
X = torch.randn(50, 30)
W = torch.randn(30, 20)
outlier_cols = [2, 17]
X[:, outlier_cols] *= 20.0   # 两个离群激活通道

s = find_smooth_scale(X, W, alpha=0.5)
X_s, W_s = apply_smoothing(X, W, s)

# --- Part A: 变换前后精确等价(不涉及量化,纯重参数化) ---
Y_direct = X @ W
Y_smoothed = X_s @ W_s
assert torch.allclose(Y_direct, Y_smoothed, atol=1e-4)

# --- Part B: 平滑后的激活确实更容易做 int8 量化 ---
def int8_quant_error(mat):
    q, sc = quantize_per_tensor(mat)
    dq = dequantize_per_tensor(q, sc)
    return mse(mat, dq)

err_raw = int8_quant_error(X)
err_smoothed = int8_quant_error(X_s)
assert err_smoothed < err_raw   # 平滑后量化误差应该更小
```

**实测(`.venv` 真跑):** 两个离群通道(幅度放大 20 倍)的构造下,`X@W` 直接计算和"平滑后 `X_s@W_s`"的最大绝对误差只有 `1.5e-5`(浮点舍入量级,视为精确相等)——确认这一步变换在数学上是精确的重参数化,不引入任何系统性偏差。对激活做 per-tensor int8 量化,原始 `X` 的 MSE 是 `0.03138`,平滑后 `X_s` 的 MSE 只有 `0.000723`——误差降到约 `1/43`,验证了"把离群值搬去权重"这个操作确实大幅改善了激活的可量化性。

**面试怎么问 + 追问链:**
- **Q:** "SmoothQuant 和 AWQ 都用了 `X·W=(X/s)(sW)` 这条恒等式,两者的核心区别是什么?" —— 期望说出"AWQ 只量化权重,激活保持 fp16 不量化,`s` 的作用是让量化预算向重要(激活幅度大的)通道倾斜;SmoothQuant 权重和激活都要量化,`s` 的作用是把激活里难以量化的离群值转移到权重上,让两边都变得更容易被量化"。
- **追问 1(考察是否真的验证过精确性):** "`X/s` 和 `s·W` 这个变换,会不会本身就引入精度损失?" —— 期望明确说"不会——这是一个纯数学上的重参数化,`(X/s)·(s·W)` 在 fp32/fp64 精度下和 `X·W` 精确相等,本知识点实测最大误差只有 1e-5 量级、是浮点舍入噪声不是系统性偏差;真正的精度损失来自变换之后对 `X_s`/`W_s` 做的量化,不是变换本身"。
- **追问 2:** "`alpha` 参数如果设成 0 或 1,分别对应什么极端情况?" —— 期望说出"`alpha=0` 时 `s_k = 1/w_amax^1`,负担完全压向权重那一侧的公式极限(激活几乎不被缩放);`alpha=1` 时 `s_k = x_amax^1`,负担完全压向激活那一侧;`alpha=0.5` 是 lecture 给出的、经验上两边各让一半的折中选择,极端值通常都不是最优的,需要针对具体模型调"。
- **追问 3:** "如果一个模型的激活分布完全没有离群值(所有通道幅度都差不多),SmoothQuant 还有意义吗?" —— 期望能辩证回答:如果 `x_amax` 在各通道之间差异很小,`find_smooth_scale` 算出来的 `s` 会接近某个常数(各通道缩放程度相近),平滑操作对量化精度的改善会很有限——SmoothQuant 解决的具体是"激活离群值集中在少数通道"这一类问题,如果这个前提不成立,它的收益会明显下降,不是任何模型上都能带来同样幅度的改善。

**常见坑:** 把 SmoothQuant 的"离群值转移"理解成"离群值被消除了"——实际上离群值只是换了个载体(从激活变成权重的对应通道),数值本身依然存在,只是权重是**静态**的、可以用比激活更精细的量化处理(比如 per-channel)去消化这部分负担,而不是真的把这些大数值"抹掉"。另一个坑是把这个知识点和知识点 4 AWQ 的"是不是同一个东西"搞混——两者数学公式相似度很高,容易在面试里被追问细节时说混,区分的关键记忆点是"AWQ 不量化 activation、SmoothQuant 两边都量化"这一句话。

---

## 6. LLM.int8()(L06,概念性;离群通道问题用 `int8_basics.py` 独立复现)—— 90% 通道 int8 + 10% 离群通道 fp16 的混合精度方案

**是什么:** L06 自己的"实现"条目写的是"见 `bnb_int4.py`(lib 轨;minimal 部分略)"——但 `bnb_int4.py` 实际实现的是 NF4(4-bit,knowledge point 9 的主题),不是 LLM.int8() 论文本身"int8 混合精度分解"的机制,这是源材料自己承认"minimal 部分略"的一个空缺。本知识点如实标注这一点,用已经读过的 `int8_basics.py` 独立构造一个"离群通道"场景,复现 LLM.int8() 论文动机(离群值破坏 per-tensor 量化精度)所描述的**问题**本身,但不假装实现了它的**解法**(fp16/int8 混合矩阵乘法)。

**一句话:** Dettmers et al. 2022 发现 6.7B+ 规模的模型出现"少数几个通道数值远大于其余通道"这种系统性离群现象,直接 int8 量化整个矩阵精度会崩;LLM.int8() 的解法是把这些离群通道**摘出来单独用 fp16 计算**,剩下 90% 的正常通道正常做 int8,矩阵乘法拆成 `y = X_int8·W_int8 + X_outlier_fp16·W_outlier_fp16` 两部分相加。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 2 已经讲过"离群值撑大 scale、拖累其余通道精度"这个机制,LLM.int8() 面对的是这个问题在大模型上一个特别极端的具体表现:Dettmers 发现当模型规模超过 6.7B 左右,某些**固定**的少数通道(不是随机出现,是训练过程中系统性形成的)会稳定地比其余通道大出一到两个数量级,如果对整个矩阵做统一的 per-tensor(甚至 per-channel)量化,这些通道的存在会让 scale 被迫设得很大,其余 90%+ 通道的有效量化精度被严重压缩。LLM.int8() 不像 AWQ/SmoothQuant 那样试图"转移"或"缩放"来缓解离群值的影响,而是**彻底不量化**这一小撮离群通道——运行时把输入激活按"是否是离群通道"拆成两部分,离群部分和对应权重都保留 fp16 精度做矩阵乘法,非离群部分正常做 int8 矩阵乘法,两部分结果相加得到最终输出。这个方案的精度收益很直接(lecture 给出 MMLU 掉分 < 1pp,几乎无损),但代价也很直接:一次矩阵乘法拆成了两次(一次 int8、一次 fp16)串行执行,lecture 明确指出"速度:慢(mixed matmul kernel 不优化)"——这是本专题里少数一个"精度优先、不追求速度"的方案,后续 SmoothQuant/AWQ 某种意义上都是在尝试"既要精度、又要速度"这个更难的目标,而 LLM.int8() 的历史地位更多在于**验证了"离群通道是大模型量化掉点的根本原因"这个诊断本身**,而不是它自己的混合精度方案成为了长期的主流部署选择。

**AI 研究场景:** lecture L06 把这个方案定位成一个承前启后的节点:"LLM.int8() → 验证 outlier 假说 → 之后 SmoothQuant/GPTQ 用不同思路解决 outlier"——这条时间线在本文里能完整对上:知识点 3(GPTQ,用 Hessian 补偿间接处理)、知识点 4(AWQ,用缩放保护权重)、知识点 5(SmoothQuant,用缩放转移激活离群值)都是在 LLM.int8() 之后,针对同一个"离群值"根本问题给出的、比"混合精度分离"更快的不同解法。理解这条演化脉络,是回答"为什么量化领域有这么多看似相似的方法"这类面试问题的关键框架。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from int8_basics import quantize_per_tensor, dequantize_per_tensor, quantize_per_channel, dequantize_per_channel, mse

torch.manual_seed(9)
W = torch.randn(16, 64)
W[:, 5] *= 50.0   # 第 5 个通道是极端离群通道(模拟 LLM.int8() 论文描述的现象)

qt, st = quantize_per_tensor(W)
dqt = dequantize_per_tensor(qt, st)
qc, sc = quantize_per_channel(W, axis=0)
dqc = dequantize_per_channel(qc, sc)

other_cols = [i for i in range(64) if i != 5]
mse_tensor_others = mse(W[:, other_cols], dqt[:, other_cols])
mse_channel_others = mse(W[:, other_cols], dqc[:, other_cols])
# 验证 LLM.int8() 论文的核心动机：一个离群通道就能明显拖累其余 63 个正常通道的 per-tensor 量化精度
assert mse_tensor_others > mse_channel_others * 3
```

**实测(`.venv` 真跑):** 64 个通道里只有 1 个是离群通道(幅度放大 50 倍),per-tensor 量化在**其余 63 个正常通道**上的 MSE 是 `0.07505`,per-channel 量化在同样 63 个通道上的 MSE 只有 `0.01663`——差了约 `4.5` 倍。这验证了 LLM.int8() 论文的核心诊断:哪怕只有 1/64 的通道是离群值,per-tensor 量化依然会明显拖累其余全部正常通道的精度,这正是"必须把离群通道特殊处理"这个设计决策的数值证据。需要再次强调:这个例子验证的是 LLM.int8() 要解决的**问题**,per-channel 量化本身不是 LLM.int8() 的解法(它的解法是"离群通道 fp16 + 其余 int8 的混合矩阵乘法",本仓库没有对应实现)。

**面试怎么问 + 追问链:**
- **Q:** "LLM.int8() 具体怎么处理大模型里的离群通道?" —— 期望说出"把矩阵乘法拆成两部分:少数几个离群通道(数值远大于其余通道)保留 fp16 精度单独计算,其余大多数正常通道做 int8 量化计算,两部分结果相加"。
- **追问 1(诚实性检验):** "这个仓库有没有 LLM.int8() 的混合精度矩阵乘法实现?" —— 期望明确说"没有——L06 自己的 lecture 写的是'minimal 部分略',指向的 `bnb_int4.py` 实际实现的是 NF4(4-bit,一个不同的、更晚的 bitsandbytes 技术),不是 LLM.int8() 本身的 int8 混合精度分解;本知识点只用 `int8_basics.py` 独立验证了'离群通道会拖累 per-tensor 量化精度'这个 LLM.int8() 要解决的问题本身"。
- **追问 2:** "为什么 LLM.int8() 的混合精度矩阵乘法'速度慢',知识点 4/5 的方案却能做到接近无损又提速?" —— 期望说出"LLM.int8() 是把一次 matmul 拆成两次串行执行(int8 部分 + fp16 部分),两次矩阵乘法的开销叠加;AWQ/SmoothQuant 用缩放变换把离群值的影响'吸收'掉之后,最终依然只做**一次**低精度 matmul(AWQ 是 int4 权重×fp16 激活,SmoothQuant 是 int8×int8),没有'拆两次算'这个额外开销"。
- **追问 3:** "如果一个模型规模远小于 6.7B(比如 100M 参数),还会出现 LLM.int8() 描述的离群通道现象吗?" —— 期望能说出:lecture 提到的"6.7B+"是 Dettmers 论文观察到离群现象**变得普遍且显著**的经验阈值,不是一个精确的数学边界,更小的模型可能也存在类似但不那么极端的离群通道、也可能几乎不存在——离群现象和具体模型的训练方式、规模、架构都有关,不是所有模型都会在同一个规模阈值上表现出同样的模式。

**常见坑:** 把知识点 6 里用 `int8_basics.py` 做的"离群通道对比实验"误当成"这就是 LLM.int8() 的实现"——上面的追问链已经反复强调,这只是在复现 LLM.int8() 要解决的**问题**的数值证据,真正的解法(fp16/int8 混合矩阵乘法)本仓库没有实现。另一个坑是把 LLM.int8() 和知识点 9 会讲的 `bnb_int4.py`(NF4)当成同一个方法的两种叫法——两者都出自 bitsandbytes/Dettmers 团队的工作,但是先后两个不同的技术(LLM.int8() 是 8bit 混合精度,NF4 是更晚提出的、专门给 4bit 权重设计的分位数编码),lecture L06 自己也在第 7 节把"bitsandbytes 4bit(nf4/fp4)进一步压"列为 LLM.int8() 之后的"后续方向",明确是两件不同的事。

---

## 7. FP8 格式:E4M3/E5M2(`fp8_demo.py`,L07)—— 8 位浮点的指数/尾数取舍,以及一处真实的编码表偏差

**是什么:**
```python
def _build_e4m3_table() -> torch.Tensor:
    """Enumerate all 256 E4M3 representations and return their fp32 values."""
    vals = set()
    for sign in (0, 1):
        for exp in range(16):
            for mant in range(8):
                if exp == 0:
                    val = 2 ** (1 - 7) * (mant / 8)
                else:
                    val = 2 ** (exp - 7) * (1 + mant / 8)
                if sign == 1:
                    val = -val
                vals.add(val)
    return torch.tensor(sorted(vals))
```
(`fp8_demo.py:13-26`)

**一句话:** E4M3(4 位指数+3 位尾数)相比 int8 的关键区别是"精度不是均匀分布的"——用浮点格式意味着数值越小、能表示的相对精度越高,数值越大、格点越稀疏,`_build_e4m3_table` 用标准 IEEE 754 式的公式(`exp=0` 走非规格化数分支,`exp>0` 走 `2^(exp-bias)·(1+mant/8)` 规格化分支)枚举出全部 256 种编码对应的真实数值;本知识点独立核实发现,这份教学实现枚举出的**最大值是 480,不是 lecture 给出的 448**。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么不直接用 int8 存激活/权重,非要发明一种"8 位浮点"格式?因为浮点格式(指数+尾数)天然能同时覆盖"很小的数需要精细区分"和"偶尔出现的大数值也能表示"这两个需求——int8 定点数的精度在整个表示范围内是均匀的,如果一批数里既有 `0.001` 这种小值又有 `100` 这种大值,要么牺牲小值精度(scale 定大了)要么大值直接溢出(scale 定小了);FP8 因为有指数位,`0.001` 附近和 `100` 附近各自有自己的一套尾数格点,不需要在两者之间做全局取舍,这也是为什么 lecture L07 强调 FP8"接近无损"(PPL 仅涨 0.5% vs int8 的约 1%)。E4M3(4 位指数、3 位尾数)和 E5M2(5 位指数、2 位尾数)是两种不同的权衡:E4M3 指数位少、尾数位多,数值范围小(`±448`)但精度更高,适合 forward 阶段的 weight/activation;E5M2 指数位多、尾数位少,数值范围大(`±57344`)但精度更粗,适合 backward 阶段数值范围波动更大的梯度——知识点 8(FP8 训练)会展开这个"为什么梯度要用不同格式"的问题。本知识点独立核实 `_build_e4m3_table()` 的具体输出时发现一处真实偏差:这份实现枚举全部 `2×16×8=256` 种(符号×指数×尾数)组合,取值集合去重后剩 255 个(正负 0 合并),其中**最大值是 480.0**(对应 `exp=15, mant=7`:`2^(15-7)×(1+7/8)=256×1.875=480`),但真实的 OCP/NVIDIA Hopper E4M3 硬件规范会把 `exp=15, mant=7` 这个编码**保留给 NaN**,真正的最大有限值是 `exp=15, mant=6`:`2^8×(1+6/8)=448`——恰好是 lecture 给出的数字。这份教学实现的枚举公式没有实现"最大指数下的最大尾数编码保留给 NaN"这个特殊情况,把它当成了一个普通的有限值,导致代码实测的表示范围比真实硬件规范多出一档。

**AI 研究场景:** 这个发现提醒一个在读带有硬件相关性的代码时容易被忽视的地方:很多数值格式的"标准定义"里,并不是所有比特组合都对应有效数值——除了 NaN(通常保留给特殊值,不是可用的数据编码)之外,某些格式还会保留特定编码给 Inf 或者其他特殊状态,教学实现如果只按最朴素的"指数+尾数"公式枚举、不考虑这些特殊编码保留规则,枚举出来的"最大/最小可表示值"就会和真实硬件的规范存在细微但真实的偏差。这类偏差在这份教学代码的场景下影响很小(只影响"极端边界值"这一个点,不影响 `fp8_round`/`fp8_matmul_mock` 对绝大多数正常数值的量化行为),但如果这份逻辑被直接照搬进对接真实硬件 kernel 的代码,480 和 448 这个差异就会导致真实 GPU 上的行为不一致(硬件会把"这个数值"解读成 NaN,软件却以为它是一个合法的 480)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from fp8_demo import _build_e4m3_table, fp8_round, fp8_matmul_mock, relative_error

table = _build_e4m3_table()
assert len(table) == 255   # 256 种编码，正负 0 合并成 1 个
assert abs(table.max().item() - 480.0) < 1e-6   # 真实发现：不是 lecture 说的 448

sorted_vals = sorted(table.tolist())
assert abs(sorted_vals[-2] - 448.0) < 1e-6   # 448 确实在表里，只是不是最大值——480 (exp=15,mant=7,
                                              # 真实硬件规范里保留给 NaN) 被这份实现当成了合法有限值

x = torch.tensor([0.001, 1.0, 100.0, 500.0, -300.0])
rounded = fp8_round(x)
assert rounded[3].item() <= 480.0 + 1e-6   # 500 饱和到这份实现的实际最大值 480，不是 448

torch.manual_seed(1)
W = torch.randn(10, 20)
X = torch.randn(15, 20)
ref = W @ X.t()
approx = fp8_matmul_mock(W, X, w_scale=W.abs().max().item() / 400, x_scale=X.abs().max().item() / 400)
rel_err = relative_error(approx, ref)
assert rel_err < 0.05
```

**实测(`.venv` 真跑):** `_build_e4m3_table()` 精确返回 `255` 个不同数值,最大值精确为 `480.0`(不是 lecture 给出的 `448.0`),排序后第二大的值精确是 `448.0`(确认它在表里,只是不是最大值)。`fp8_round([0.001, 1.0, 100.0, 500.0, -300.0])` 把 `500.0` 饱和舍入到 `480.0`(这份实现的真实上限),`-300.0` 舍入到 `-288.0`(E4M3 在这个数量级上的格点间距本身就有 `16` 左右,不是精确复现原值)。用 `fp8_matmul_mock` 对一个随机 `10×20` 权重和 `15×20` 激活做一次 fp8 模拟矩阵乘法,相对误差 `0.0424`(低于 5%)。

**面试怎么问 + 追问链:**
- **Q:** "E4M3 和 int8 相比,表示同样 8 个比特,为什么精度分布不一样?" —— 期望说出"int8 是定点数,整个范围内精度均匀(相邻两个可表示值之间的间距处处相同);E4M3 有指数位,数值越接近 0 格点越密(相对精度越高),数值越大格点越稀疏,这种'非均匀'的精度分布更贴近真实权重/激活的数值分布特征(大多数值集中在较小范围,偶尔有较大值)"。
- **追问 1(核心陷阱,考察是否真的核实过边界情况):** "这份代码算出来的 E4M3 最大可表示值是多少?和 lecture 给出的数字一致吗?" —— 期望明确说"实测是 480.0,不是 lecture 给出的 448.0——这份实现的枚举公式没有像真实 OCP/Hopper 硬件规范那样,把最大指数下的最大尾数编码保留给 NaN,把这个本该是 NaN 的编码当成了合法的有限值 480,真实硬件上的最大有限值是 448(对应稍小一档的尾数编码)",这道题专门筛"只信 lecture 文字描述、没有真的跑代码核对边界值"的候选人。
- **追问 2:** "E4M3 和 E5M2 分别适合什么场景,为什么?" —— 期望说出"E4M3 指数位少(4 位)尾数位多(3 位),数值范围小但精度更高,适合数值范围相对可控的 forward 权重/激活;E5M2 指数位多(5 位)尾数位少(2 位),数值范围大(±57344)但精度更粗,适合训练时反向传播梯度这种数值范围可能剧烈波动(尤其是训练初期或者某些层)的场景——用一个范围更宽的格式兜底,避免梯度下溢或溢出"。
- **追问 3:** "如果一份代码里发现类似这样的'教学实现和硬件真实规范不完全一致'的偏差,应该怎么处理?" —— 期望能给出合理判断:如果这份代码明确定位是"教学/CPU 模拟",不直接对接真实硬件 kernel,这类边界值偏差影响有限(不影响绝大多数正常数值的量化行为演示),如实标注清楚就足够;但如果这份逻辑将被复用到真实对接硬件的场景,就必须修正,否则会在边界值上产生软件行为和硬件实际行为不一致的真实 bug。

**常见坑:** 假设"教学仓库里的数值格式实现,细节上一定和硬件规范精确一致"——本知识点已经证明不是这样,教学实现为了保持代码简单直接,可能会漏掉"某些编码保留给特殊值"这类不那么显眼但真实存在的规范细节。另一个坑是把这个 448/480 的偏差理解成"这份代码有 bug、算错了"——严格来说这份代码没有算术错误(公式本身套用正确,`2^(exp-bias)×(1+mant/8)` 对每个 `exp`/`mant` 组合都算得没错),偏差来自"没有实现一条格式规范里的特殊编码保留规则",这是"规范覆盖不全"而不是"算术出错",两者在诊断和修复思路上是不同的问题。

---

## 8. FP8 训练(L08,概念性,回顾 Module 3 scaling-infra)—— 训推一致的代价:动态 loss scaling 防止梯度下溢

**是什么:** 本知识点没有对应的 `src/*.py` 文件——L08 自己的 lecture 明确写"本课为概念课,回顾 Module 3 scaling-infra 的 FP8 training 章节",本仓库对应 Module 3(`learning/scaling-infra/` 或类似模块)的具体训练循环实现不在本系列范围内。本知识点如实标注这一点,用纯 Python/torch 构造一个通用的"粗糙量化网格下,小数值容易被舍入成 0"的最小示例,来说明**为什么** FP8 训练需要动态 loss scaling,不假装复现了真实的 FP8 训练循环。

**一句话:** FP8 训练(以 DeepSeek-V3 为代表)在 forward 用 E4M3、backward 梯度用 E5M2、accumulate/参数更新仍用 fp32 主权重,这套"训推都用 FP8"的方案能省下 30% 左右训练时间、并且推理时直接复用同一份 FP8 权重、不需要额外量化步骤——但代价是梯度的数值范围天然比激活/权重更不稳定,必须靠"动态 loss scaling"人为放大梯度数值,防止小梯度在粗糙的 FP8 网格上被舍入成 0(下溢),训练完全失去这部分信号。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么 fp16/bf16 训练已经很成熟了,还要冒险换成范围更窄、精度更粗的 FP8?lecture L08 给出的动机是"fp16/bf16 matmul 在 H100/Blackwell 上没用到全部 TFLOPs"——现代 GPU 的 Tensor Core 对更低精度格式有专门优化的算力通道,FP8 matmul 能达到 fp32 的 3-4 倍加速(lecture 数字),这个诱惑大到即便要多花工程成本处理数值稳定性问题也值得。核心工程难点是梯度:训练前中期梯度数值可能很大(参数还没收敛),训练后期梯度可能变得很小(接近收敛,更新量本该精细),这个动态范围如果直接映射到一个固定量化网格上,要么早期梯度溢出要么后期梯度全部下溢成 0。"动态 loss scaling"的做法是:训练时给 loss 乘一个较大的缩放因子(比如 1024),这样反向传播算出来的梯度也被同比例放大,采用 FP8 表示前先放大、量化后再等比例缩小回来——这样即便梯度本身很小,放大后落在 FP8 网格能有效区分的范围内,不会被舍入成 0;这个缩放因子本身还需要"动态"调整(如果训练过程中出现梯度上溢/NaN,自动调小缩放因子;长时间没有溢出,可以尝试调大以获得更好精度),这是一整套需要专门工程实现的机制,不是一次性设定就能一劳永逸。DeepSeek-V3 的实际方案是全程 FP8 weight+activation,配合这套动态 scaling,训练时间相对 bf16 节省约 30%,而且因为训练全程就是 FP8,推理时直接复用这份 FP8 checkpoint,不需要再走一遍知识点 3-7 讨论的那些"训练后量化"流程——这是 lecture 强调的"训推一致"这个本质区别:PTQ(GPTQ/AWQ/SmoothQuant)是"fp16/bf16 训好之后再量化",天然有量化误差;FP8 训练是"从头到尾都是 FP8",没有额外引入的量化步骤,理论上精度上限更高。

**AI 研究场景:** lecture L08 提到"2025-2026 中小开源模型大量用 FP8 训练(DeepSeek-V3 引爆)"——这是本专题里少数直接和"当前时间点行业趋势"挂钩的知识点,反映出量化技术正在从"训练后补救测量"往"训练时就纳入设计"这个方向演进,呼应知识点 1 提到的 PTQ vs QAT 这组维度选择:FP8 训练某种意义上是"QAT 的一种极端形式"(不是模拟量化误差去适应,而是训练全程真的就在量化精度下运行)。TransformerEngine(NVIDIA 官方库)把这套复杂的动态 scaling 机制封装成"一行代码切换 fp8/bf16",是这类基础设施级别优化"底层足够复杂、但要让上层用户几乎无感"的典型设计目标。

**可运行例子:**
```python
import torch

torch.manual_seed(2)
grad_like = torch.randn(5000) * 0.01   # 模拟训练后期数值偏小的梯度

def coarse_round(x: torch.Tensor, step: float) -> torch.Tensor:
    """Round to a coarse grid with fixed spacing `step` -- a stand-in for a low-precision format's granularity."""
    return (x / step).round() * step

# 不做 loss scaling：直接在粗糙网格(step=1.0)上量化，绝大多数小梯度被舍成 0
unscaled = coarse_round(grad_like, step=1.0)
frac_zero_unscaled = (unscaled == 0).float().mean().item()

# 动态 loss scaling：先放大 1024 倍再量化，再等比例缩小回来
loss_scale = 1024.0
scaled = coarse_round(grad_like * loss_scale, step=1.0) / loss_scale
frac_zero_scaled = (scaled == 0).float().mean().item()

assert frac_zero_unscaled > frac_zero_scaled   # loss scaling 显著减少"梯度被舍入成 0"的比例
assert frac_zero_unscaled > 0.9   # 不做 scaling 时，几乎所有小梯度都会下溢成 0
```

**实测(`.venv` 真跑):** 5000 个模拟"训练后期偏小梯度"(标准差 `0.01`)的随机数,在一个格点间距为 `1.0` 的粗糙量化网格上(用来类比低精度格式有限的表示粒度),不做 loss scaling 时 `100%` 的值被舍入成精确的 `0`(信息完全丢失);用 `loss_scale=1024` 放大后再量化再缩小,只有约 `3.84%` 的值变成 `0`——loss scaling 把"下溢成 0"的比例从全部降到了个位数百分比,直观展示了这套机制要解决的具体数值问题。

**面试怎么问 + 追问链:**
- **Q:** "FP8 训练为什么需要动态 loss scaling,fp16/bf16 训练时这个问题不明显吗?" —— 期望说出"FP8(尤其 backward 用的 E5M2)可表示的有效精度范围比 fp16/bf16 更粗,梯度数值本身在训练不同阶段动态范围很大,不做处理容易在精度粗糙的格式上被直接舍入成 0(下溢);fp16 本身精度范围更宽松,这个问题相对不那么尖锐,但 fp16 训练同样有历史上著名的'loss scaling'技术用于处理类似问题(不是 FP8 独有,只是 FP8 上更突出)"。
- **追问 1(诚实性检验):** "这个仓库有没有真实的 FP8 训练循环实现,能让我看看具体怎么写吗?" —— 期望明确说"没有——L08 自己的 lecture 说明这是概念课,指向 Module 3 scaling-infra,不在本系列范围内;本知识点只用一个通用的粗糙量化网格模拟了'为什么需要 loss scaling'这个问题本身,不是 FP8 训练循环的复现"。
- **追问 2:** "为什么不能简单地把 loss scaling 设成一个足够大的固定值,一劳永逸?" —— 期望说出"缩放因子太小,精度改善有限(小梯度还是可能下溢);缩放因子太大,反而可能让原本正常范围的梯度经过放大后超出格式能表示的最大值(上溢,变成 Inf/NaN),训练直接崩掉——这是为什么需要'动态'调整:根据训练过程中是否出现溢出,自动增大或减小缩放因子,没有一个对所有阶段都最优的固定值"。
- **追问 3:** "FP8 训练相比传统'先 bf16 训练、再 PTQ 量化部署'这条路线,最大的优势是什么?" —— 期望说出"'训推一致'——PTQ 路线在训练完成后才引入量化误差,精度上限受限于'一个从未见过量化的模型,量化后还能保留多少精度';FP8 训练全程就在 FP8 精度下运行,模型的参数分布是在 FP8 约束下训出来的,推理时直接复用同一份 checkpoint,没有额外的'训练后突然换精度'这个断层,理论上精度天花板更高,同时还省了训练算力(3-4x)"。

**常见坑:** 把这个知识点的"可运行例子"误当成"FP8 训练的真实数值行为模拟"——它只是用一个抽象的、和具体 FP8 格式细节无关的"粗糙网格量化"模型来说明"下溢"这个现象的普遍机制,不是对 E5M2 格式本身编码细节的复现(知识点 7 的 `_build_e4m3_table` 才是对具体 FP8 编码格式的真实建模,而且那也只做了 E4M3、没有做 E5M2)。另一个坑是认为"训推一致"意味着"完全没有精度损失"——lecture 给出的"推理无任何量化损失"指的是"部署时不需要再额外做一次量化"(没有第二次精度下降),但 FP8 训练本身相比全程 fp32/bf16 训练依然存在 FP8 精度带来的表示误差,只是这个误差被模型在训练过程中"学着适应"了,不是零误差。

---

## 9. W4A16 / W4A8(`bnb_int4.py`,L09+L10)—— NF4 量化码本,以及和 peft-deep-dive 训练期 NF4 的独立对照

**是什么:**
```python
# NF4 quantile-derived codes from the bitsandbytes paper (16 levels)
NF4_CODES = torch.tensor([
    -1.0,    -0.6962,  -0.5251,  -0.3949,
    -0.2844, -0.1848,  -0.0911,   0.0,
     0.0796, 0.1609,    0.2461,    0.3379,
     0.4407, 0.5626,    0.7230,    1.0,
])


def quantize_nf4(W: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block nf4 quantization: each `block_size` elements share a scale."""
    flat = W.flatten()
    pad = (-len(flat)) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
    normalised = blocks / scales
    idx = (normalised.unsqueeze(-1) - NF4_CODES).abs().argmin(dim=-1)
    return idx.to(torch.uint8), scales.squeeze(-1)
```
(`bnb_int4.py:7-27`)

**一句话:** W4A16(weight int4 + activation fp16)是工业界目前最常见的部署配置,`bnb_int4.py` 用 16 个**非均匀分布**的分位数编码(`NF4_CODES`,越接近 0 编码越密集)代替均匀网格量化 weight——这个非均匀性专门针对"神经网络权重训练完之后大致呈正态分布、接近 0 的值远比接近 ±1 的值多"这个经验事实设计,`quantize_nf4` 按 `block_size=64` 分块、每块独立算 scale,是比知识点 2 per-channel 更细的粒度。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 2 的均匀量化网格(比如 int4 的 16 个格点均匀分布在 `[-1,1]`)隐含假设"每个格点被用到的概率差不多",但真实权重分布通常是"接近 0 的值出现概率高、远离 0 的值出现概率低"(近似正态分布的形状)。如果用均匀网格量化正态分布的数据,大量集中在 0 附近的权重会挤在很少几个格点上(精度浪费在"格点稀疏、用不上"的两端);NF4(NormalFloat 4-bit)反过来设计:先假设数据服从标准正态分布,推导出"让每个格点覆盖的概率质量相等"(quantile-based,分位数)这个 16 个非均匀分布的编码点——`NF4_CODES` 从 `-1.0` 到 `1.0`,可以看到越接近 `0.0` 的地方间距越小(比如 `-0.0911` 到 `0.0` 到 `0.0796` 这几个中间格点间距明显小于两端 `0.7230` 到 `1.0` 的间距),这正是"让常见的值区分得更细、罕见的值区分得更粗"这个思路的直接体现。`quantize_nf4` 的具体做法是:每 `block_size=64` 个元素分一组,组内先除以这组的最大绝对值归一化到 `[-1,1]`(这样可以直接套用固定的 `NF4_CODES`,不需要为每个不同数值范围重新设计编码表),再对归一化后的每个值找最接近的 `NF4_CODES` 索引存下来(`argmin` 找最近邻)。`block_size=64` 比知识点 2 常见的 `group_size=128` 更细,这是 NF4 在"scale 数量"和"精度"这组权衡上做出的选择,呼应它面向的场景(端侧部署、显存最紧张)对精度的更高要求。

**AI 研究场景:** 本知识点必须和 [peft-deep-dive/02-quantized-lora.md](../peft-deep-dive/02-quantized-lora.md) 划清关系:那边的知识点 1-3(`nf4_quant.py`/`qlora_minimal.py`/`qlora_peft.py`)也实现了 NF4,服务于 **QLoRA**——把预训练底座权重压成 NF4 腾出显存,在这份压缩过的底座上叠加可训练的 LoRA 适配器,这是**训练阶段**的用法;本知识点的 `bnb_int4.py` 是同一套 NF4 数学思想在**推理部署阶段**的独立实现,已经过核实,两份代码互不引用,是完全独立的两次重写,不是同一份代码被复制到两个系列。lecture L09 强调 W4A16 是"工业最常见配置"——weight 部分用 bitsandbytes(NF4)或 autoawq(AWQ)或 auto-gptq(GPTQ)三选一压缩,activation 保留 fp16,vLLM 对三者都有支持;L10 W4A8 是在此基础上激活也量化到 int8(通常配合知识点 5 SmoothQuant 先把激活离群值处理掉),lecture 给出的收益是牺牲 1-2pp 精度换 30% 速度提升(decode tok/s 从 W4A16 的 220 到 W4A8 的 280,Marlin kernel)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from int8_basics import quantize_per_group, dequantize_per_group, mse
from bnb_int4 import NF4_CODES, quantize_nf4, dequantize_nf4

assert len(NF4_CODES) == 16
# NF4 编码点在 0 附近确实比在 ±1 附近更密集(非均匀设计的直接验证)
gap_near_zero = (NF4_CODES[8] - NF4_CODES[7]).item()    # 0 到 0.0796
gap_near_one = (NF4_CODES[15] - NF4_CODES[14]).item()   # 0.7230 到 1.0
assert gap_near_zero < gap_near_one

torch.manual_seed(21)
W = torch.randn(8, 256)   # 大致正态分布的权重(和 peft-deep-dive 的例子用不同形状)
idx, scales = quantize_nf4(W, block_size=64)
recon = dequantize_nf4(idx, scales, block_size=64, orig_shape=W.shape)
nf4_mse = mse(W, recon)

# 对照：同样 4bit 预算，但用均匀网格(per-group)量化同一批数据
uniform_q, uniform_s = quantize_per_group(W, group_size=64, n_bits=4)
uniform_recon = dequantize_per_group(uniform_q, uniform_s, group_size=64)
uniform_mse = mse(W, uniform_recon)

assert nf4_mse < uniform_mse   # 对正态分布数据，非均匀 NF4 码本应该比均匀网格精度更高
```

**实测(`.venv` 真跑):** `NF4_CODES` 里 `0` 附近的格点间距(`0` 到 `0.0796`,即 `0.0796`)确实小于 `±1` 附近的间距(`0.7230` 到 `1.0`,即 `0.2770`)——直接验证了"非均匀、中间密两端疏"这个设计。对一个 `8×256` 的标准正态分布权重矩阵,NF4(`block_size=64`)量化重建 MSE 是 `0.008147`,同样 4bit 预算下的均匀 per-group(`group_size=64`)量化 MSE 是 `0.010730`——NF4 精度确实更高(改善约 `24%`),验证了"码本形状匹配数据分布"这个设计带来的实际收益。

**面试怎么问 + 追问链:**
- **Q:** "NF4 相比均匀 int4 量化,核心改进是什么?" —— 期望说出"NF4 用 16 个非均匀分布的分位数编码代替均匀网格,编码点在 0 附近更密集、在 ±1 附近更稀疏,专门匹配'神经网络权重训练后大致呈正态分布、多数值集中在 0 附近'这个经验规律,让常见的值获得更精细的量化,不常见的值可以接受更粗的量化"。
- **追问 1(考察是否理解这个系列和 peft-deep-dive 的关系):** "这个仓库里是不是已经在别的地方讲过 NF4 了?两处是同一份代码吗?" —— 期望说出"peft-deep-dive 系列的知识点 1-3 也讲了 NF4,但那边是服务于 QLoRA(训练时压缩冻结底座、腾显存做 LoRA);这里的 `bnb_int4.py` 是服务于推理部署的独立实现,两份代码已经核实互不引用、是分别独立编写的两次实现,共享的只是'NF4 16 个分位码'这个数学思想,不是同一份代码的复制"。
- **追问 2:** "W4A16 为什么比 W4A8 更常见,尽管 W4A8 理论上更快?" —— 期望说出"activation 是运行时动态产生的,量化难度天然高于静态的 weight(知识点 5 讨论过的离群值问题),W4A16 只需要处理 weight 这一边的量化,工程实现和精度保证都更简单;W4A8 要额外处理动态 activation 量化(通常还要配合 SmoothQuant 类技术先处理离群值),复杂度和风险都更高,只有在真正需要极致吞吐(比如高 QPS 的 agent 推理场景,lecture L10 提到的使用场景)时才值得承担这份额外复杂度换来的 30% 速度提升"。
- **追问 3:** "`block_size` 从 128 缩小到 64,对精度和存储各有什么影响?" —— 期望说出"block_size 越小,每个 scale 覆盖的数值范围越小,精度通常越高(呼应知识点 2 的'粒度越细精度越高'结论),但需要存储的 scale 数量翻倍,存储/带宽开销增加——64 这个选择是 NF4 在这组权衡上比常见的 128 更偏向精度一侧的具体体现"。

**常见坑:** 把 NF4 的"非均匀编码"和知识点 4 AWQ 的"per-channel 缩放"混为一谈——两者都是应对"数据分布不均匀"这个大问题的手段,但作用层面不同:AWQ 的缩放是在量化**之前**改变数值的相对大小(重参数化);NF4 的非均匀编码是量化**本身**用什么样的格点去逼近数值,两者甚至可以叠加使用(先用类似 AWQ 的手段调整数值分布,再用 NF4 这种非均匀码本量化)。另一个坑是假设 NF4 对任何数据分布都比均匀量化好——NF4 的编码点是针对**标准正态分布**推导出来的,如果被量化的数据分布明显偏离正态(比如本文知识点 4/6 讨论的那种带有极端离群值的分布),NF4 的量化质量优势会打折扣,这也是为什么 AWQ/GPTQ 这类"考虑具体校准数据分布"的方法在处理离群值场景时通常比固定码本的 NF4 更稳健。

---

## 10. KV Cache 量化(`kv_quant.py`,L11)—— 按 token 定 scale,量化误差在 attention 里会怎么传播

**是什么:**
```python
def quantize_kv_per_token(kv: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize KV `[seq, n_heads, head_dim]` with one scale per token."""
    qmax = (1 << (n_bits - 1)) - 1
    seq = kv.shape[0]
    flat = kv.reshape(seq, -1)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / qmax
    q = (flat / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q.reshape(kv.shape), scale.reshape(seq, 1, 1)


def attention_with_quant_kv(
    Q: torch.Tensor, Kq: torch.Tensor, Vq: torch.Tensor, k_scale: torch.Tensor, v_scale: torch.Tensor,
) -> torch.Tensor:
    K = dequantize_kv_per_token(Kq, k_scale)
    V = dequantize_kv_per_token(Vq, v_scale)
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.einsum("qhd,khd->qkh", Q, K) * scale
    attn = torch.softmax(scores, dim=1)
    return torch.einsum("qkh,khd->qhd", attn, V)
```
(`kv_quant.py:7-34`)

**一句话:** KV cache 量化和前面知识点 2-9 量化的对象(weight)不同——`quantize_kv_per_token` 给**每个 token**(而不是每个通道/每个 group)单独算一个量化 scale,`attention_with_quant_kv` 演示了量化后的 K/V 怎么先反量化回浮点、再走标准 attention 计算(不是在量化状态下直接做 attention 运算)。

**底层机制/为什么这样设计:** 从最笨的想法讲起——01 号文件知识点 2 已经建立过"KV cache 随请求数/序列长度线性增长、常常比 weight 本身还占显存"这个痛点(lecture L11 给出的具体数字:7B 模型 fp16 weight 只要 14GB,但 32k 上下文的 KV cache 就要 32GB,单卡装不下)。量化 KV cache 是缓解这个痛点最直接的手段,但和量化 weight 有一个关键差异:**weight 是静态的**(量化一次、之后一直复用),KV cache 是**动态增长的**(每生成一个新 token,就要给这个 token 的 K/V 分配存储、决定 scale)——这就是为什么 `quantize_kv_per_token` 选择"per-token"这个粒度而不是知识点 2 讨论的 per-tensor/per-channel/per-group:一个 token 的 K/V 向量一旦写入 cache 就不会再变,给它单独定一个 scale 是自然的选择,后续新 token 各自定各自的 scale,不需要像 weight 量化那样纠结"整批数据的 scale 怎么定"这个问题。`attention_with_quant_kv` 的实现选择"先反量化、再做标准 attention"而不是"直接对量化后的整数做特殊的量化 attention kernel"——这是教学实现追求代码清晰度、放弃性能的典型简化(lecture L11 提到真实系统会用"fused dequant + attention"的 kernel、不需要真的先展开成完整的 fp32/fp16 张量再计算,那是工程实现层面的优化,和这里要讲清楚的数值行为是分开的两件事)。量化误差在这里会经过 softmax 传播:`scores` 计算用的是反量化后的 `K`(已经带有量化误差),这个误差会体现在 softmax 之前的 attention score 上,再经过 softmax(一个非线性、对输入敏感的操作)传播到最终的注意力权重分布上——这是 lecture L11 强调"KV 比 weight 更敏感(attention softmax 放大误差)"这句话的具体机制:不是量化本身在 KV 上误差更大,而是这个误差要经过 softmax 这道非线性放大的关卡,和 weight 量化误差"线性地"体现在最终输出上不完全是同一回事。

**AI 研究场景:** lecture L11 的精度对比表显示 int8 KV(PPL 5.75)和 fp8 KV(PPL 5.70)都比 int4 KV(PPL 6.20)温和得多,fp16 baseline 是 5.68——这解释了为什么"fp8 KV 接近无损,是最优选项(Hopper+ 硬件支持)"这个 lecture 结论:相比 weight 可以相对激进地压到 4bit(GPTQ/AWQ 都只涨 2-3% PPL),KV cache 更适合停留在 8bit 这个精度档位,再往下(int4 KV)精度损失开始明显放大。这也是 vLLM(`--kv-cache-dtype fp8`)、SGLang(`--kv-cache-dtype int8`)这些真实系统默认给 KV cache 单独一套量化精度配置(通常比 weight 量化更保守)的原因,和 01 号文件 PagedAttention 讨论的"KV cache 用专门的显存管理策略、不和 weight 混为一谈"是同一种"KV cache 的特殊性值得单独设计"的思路在不同维度上的体现。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
import torch
from kv_quant import quantize_kv_per_token, dequantize_kv_per_token, attention_with_quant_kv

torch.manual_seed(31)
seq, n_heads, d_h = 12, 4, 16
K = torch.randn(seq, n_heads, d_h)
V = torch.randn(seq, n_heads, d_h)
Q = torch.randn(3, n_heads, d_h)   # 3 个 query token，attend 到 12 个已缓存的 K/V

Kq, k_scale = quantize_kv_per_token(K)
Vq, v_scale = quantize_kv_per_token(V)
K_dq = dequantize_kv_per_token(Kq, k_scale)
assert (K - K_dq).abs().max().item() < 0.05   # int8 per-token 重建误差应该很小

def full_precision_attn(Q, K, V):
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.einsum("qhd,khd->qkh", Q, K) * scale
    attn = torch.softmax(scores, dim=1)
    return torch.einsum("qkh,khd->qhd", attn, V)

out_full = full_precision_attn(Q, K, V)
out_quant = attention_with_quant_kv(Q, Kq, Vq, k_scale, v_scale)
diff = (out_full - out_quant).abs().max().item()
assert diff < 0.1   # 量化 KV 的 attention 输出应该接近全精度，但不是逐 bit 相同
assert diff > 0.0   # 也确实存在非零误差，不是巧合地精确相等
```

**实测(`.venv` 真跑):** `12` 个已缓存 token、`4` 个 head、`head_dim=16` 的构造下,int8 per-token 量化后反量化的 K,和原始 K 的最大绝对误差是 `0.0138`(在 `0.05` 的宽松容差内,量级很小)。用量化后的 K/V 走完整 attention 流程,和全精度 attention 的最终输出相比,最大绝对误差在 `0.014` 左右——确认了"接近但不精确相等"这个预期:量化确实引入了非零误差,但对这组规模不大的随机测试数据而言,误差幅度远小于注意力输出本身的数值量级。

**面试怎么问 + 追问链:**
- **Q:** "KV cache 量化为什么用 per-token 粒度,而不是像 weight 那样用 per-channel/per-group?" —— 期望说出"KV cache 是随生成过程动态增长的,每个 token 的 K/V 向量一旦写入就不再改变,给它单独定一个 scale 是自然、高效的选择;weight 是训练完就固定的静态数据,可以在离线阶段花更多时间琢磨'整批数据用什么粒度切分 scale 最优',两者的更新模式不同,导致量化粒度的自然选择也不同"。
- **追问 1:** "为什么 lecture 说'KV 量化的误差比 weight 量化更敏感,会被 attention softmax 放大'?" —— 期望说出"量化后的 K 参与计算 attention score,这个 score 要经过 softmax 才变成最终的注意力权重——softmax 是非线性、对输入变化敏感的函数(尤其在 score 差距不大时,小的绝对误差可能导致相对权重发生更明显的变化),量化误差不是'线性地'体现在最终输出里,而是先经过这道非线性放大关卡再传播下去,这是它比一般的线性变换(比如直接的 weight matmul)更敏感的原因"。
- **追问 2:** "这份教学实现的 `attention_with_quant_kv` 是'先反量化再计算',这和真实生产系统的做法一样吗?" —— 期望说出"数值结果的逻辑是一样的(量化引入的误差表现应该相同),但真实系统(比如 FlashInfer)会用'fused dequant + attention'的 kernel,在计算过程中动态反量化,不需要真的先把完整的 K/V 张量展开成 fp32/fp16 再存一份,这样能省掉一次完整的显存读写,是工程实现层面的性能优化,不影响这里要讲清楚的'量化会引入多大误差'这个数值行为本身"。
- **追问 3:** "如果要进一步压缩 KV cache 显存,除了降低量化精度(int8→int4),还有什么思路?" —— 期望能连回 01 号文件已经讲过的机制:PagedAttention(01 号知识点 3)通过消除内部/外部碎片间接省显存(不改变精度,改变存储布局);01 号文件知识点 7/02 号文件 RadixAttention 通过共享公共前缀的 KV 间接省显存(多个请求复用同一份物理存储);KV 量化(本知识点)是从"每个数值占多少 bit"这个维度直接压缩——三条路径互相独立、可以同时叠加使用,不是互斥选项。

**常见坑:** 把"KV cache 量化"和"weight 量化"用同一套直觉去理解精度损失的可接受范围——lecture 数据显示同样是 int4,KV 量化的 PPL 涨幅(6.20)比 weight 量化(GPTQ/AWQ 4bit 约 5.81-5.85)更明显,如果照搬"weight 4bit 几乎无损"的经验去推断"KV 4bit 应该也差不多",会低估实际的精度风险。另一个坑是认为 KV 量化和 PagedAttention(知识点 4 分页存储)是互斥的两种优化——lecture L11 第 6 节明确提到两者可以集成("KV 量化时存进 paged block,读取时 fused dequant + attention 算"),量化决定"每个数值占多少 bit",分页决定"这些数值物理上怎么组织存放",是两个正交的设计维度。

---

## 11. 评测方法论 + Capstone(`quant_eval.py` + `capstone_quant_zoo.py`,L12+L13)—— 6 种量化器真跑在同一个 toy 权重层上,以及 GPTQ/NF4 排位的种子敏感性

**是什么:**
```python
def _q_awq_4bit(W, X):
    s = search_scales(W, X, n_bits=4)
    q, scale = quantize_per_channel(W * s, axis=0, n_bits=4)
    return dequantize_per_channel(q, scale) / s

_SPECS = [
    ("fp16",             16, None),
    ("int8 (pc)",         8, _q_int8_pc),
    ("GPTQ-4bit",         4, _q_gptq_4bit),
    ("AWQ-4bit",          4, _q_awq_4bit),
    ("NF4 (bnb)",         4, _q_nf4),
    ("FP8 (E4M3)",        8, _q_fp8_e4m3),
    ("SmoothQuant-int8",  8, _q_smoothquant_int8),
]

def run_all(seed: int = 0) -> List[Dict]:
    """Quantize one toy layer every way and return per-variant real metrics."""
    W, X = build_layer(seed=seed)
    rows: List[Dict] = []
    for name, bits, quantizer in _SPECS:
        err = 0.0 if quantizer is None else _output_mse(quantizer(W, X), W, X)
        rows.append(dict(variant=name, bits=bits, error=round(err, 4),
                          compression=round(16.0 / bits, 2),
                          mem_mib=round(BASE_PARAMS * bits / 8 / (1024 ** 2), 1)))
    return rows
```
(`capstone_quant_zoo.py:91-94`、`126-156`,节选)

**一句话:** `capstone_quant_zoo.py` 在同一个 `build_layer()` 构造出的 toy 权重层(带 3 个人为放大的"显著通道",专门让 AWQ/SmoothQuant 的激活感知机制"有活可干")上,真实调用知识点 2-9 讲过的 6 种量化器(`int8_basics`/`gptq_minimal`/`awq_minimal`/`bnb_int4`/`fp8_demo`/`smooth_quant`),`error` 列是真算出来的输出重建 MSE,`compression`/`mem_mib` 由每种方法真实的 bit 宽度(`16/bits`、`BASE_PARAMS×bits/8`)算出,不是任何硬编码的数字。

**底层机制/为什么这样设计:** 从最笨的想法讲起——这个 capstone 文件的 docstring 开头就交代了一段真实的自我修正历史:"这曾经是一张硬编码的 Llama-7B 论文数字表,看起来齐整,但什么都没证明——这个目录里的 int8/GPTQ/AWQ/NF4/FP8/SmoothQuant 模块没有一个被真正调用过,'压缩比'和'显存'那两列是手打的字面量"。现在的版本把每一种量化器**真的**跑一遍,在同一个 `W`(`64×128` 权重)、同一批 `X`(`512` 条校准激活,其中第 `7/33/91` 三个通道被放大 `12` 倍模拟真实存在的显著通道)上算出真实的输出重建 MSE(`mean((W_q·x - W·x)²)`,这正是 GPTQ/AWQ 论文本身优化的目标函数),`compression`/`mem_mib` 两列直接由 `bits` 这个每种方法真实存储的比特宽度算出来,不再是脱离量化器本身、单独手打的数字。本知识点在这个基础上做了两组独立验证:第一组是 `run_all(seed=0)`(和文件自带 `__main__` 使用相同的默认种子)确认 README 宣称的"GPTQ 的 Hessian 补偿 < NF4 静态码本"这个排位关系成立(`GPTQ error=12.124` < `NF4 error=17.2706`);第二组是把 seed 从 0 扫到 9(全部 10 个种子)确认这个排位关系稳定成立(10/10),再额外测一个种子 99——**这个排位关系在 seed=99 上反过来了**(`GPTQ error=18.3488` > `NF4 error=17.4719`)。这不是一个孤立的意外:知识点 3 已经独立发现并系统验证过,`gptq_quantize`/`gptq_columnwise` 默认的 `damp=0.01` 在某些校准数据条件(相关性更强、维度更高)下会让 GPTQ 的 Hessian 补偿机制数值不稳定、反而伤害精度——`capstone_quant_zoo._q_gptq_4bit` 调用 `gptq_quantize(W, H, n_bits=4)` 时用的正是这个默认 `damp`,同一个数值稳定性问题,在"知识点 3 我自己独立构造的 40×80 测试矩阵"和"capstone 自己 64×128、带显著通道的 toy 层"这两个完全不同的具体设置里都被复现,说明这不是某一次构造凑巧撞上的边界情况,而是这份 GPTQ 实现在默认超参数下真实存在的一类风险。

**AI 研究场景:** 这个 capstone 的自我修正历史(硬编码表 → 真跑 6 个量化器)本身是一个值得记住的工程教训:一张"看起来权威、逐列对得整整齐齐"的对比表,如果背后的每一列数字都不是真正跑代码算出来的,读者完全没有办法分辨它是真实结果还是排版好看的假象——这正是 huggingface-deep-dive 系列反复强调"显存测量必须交叉核对 `torch.cuda.max_memory_allocated()` 和 `nvidia-smi`"这条纪律,以及本系列自己"诚实标注"这条主线的又一个具体案例。种子敏感性这个发现进一步说明:即便是"真跑代码"得到的数字,也不能只信一次运行的结果就下结论(不管是这份仓库自己的 capstone,还是本文知识点 3/11 做的独立验证),排位关系需要在多组独立随机种子/构造下重复验证,才有资格被当成一条可以信赖的经验规律。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/quantization-deploy/src")
from quant_eval import eval_ppl_mock, memory_table
from capstone_quant_zoo import run_all, best_for, best_quantized_for, BASE_PARAMS

# --- Part A: 复现 README 在默认种子下的排位关系 ---
rows0 = run_all(seed=0)
by_v0 = {r["variant"]: r for r in rows0}
assert by_v0["fp16"]["error"] == 0.0
assert by_v0["GPTQ-4bit"]["compression"] == 4.0
assert by_v0["GPTQ-4bit"]["mem_mib"] == round(BASE_PARAMS * 4 / 8 / 1024**2, 1)
assert by_v0["GPTQ-4bit"]["error"] < by_v0["NF4 (bnb)"]["error"]   # README: GPTQ 补偿 < NF4 静态码本

# --- Part B: 10 个种子批量验证这条排位关系是否稳定 ---
gptq_wins = sum(
    run_all(seed=s) and
    (lambda r: {x["variant"]: x for x in r}["GPTQ-4bit"]["error"] < {x["variant"]: x for x in r}["NF4 (bnb)"]["error"])(run_all(seed=s))
    for s in range(10)
)
assert gptq_wins == 10

# --- Part C: seed=99 是一个真实的反例，呼应知识点 3 的 damp 数值稳定性发现 ---
rows99 = run_all(seed=99)
by_v99 = {r["variant"]: r for r in rows99}
assert by_v99["GPTQ-4bit"]["error"] > by_v99["NF4 (bnb)"]["error"]   # 排位反过来了

def toy_predictor(prefix):
    import torch
    torch.manual_seed(sum(prefix) if prefix else 0)
    return torch.randn(50)
ppl = eval_ppl_mock(toy_predictor, [[1, 2, 3, 4], [5, 6, 7]])
assert ppl >= 1.0
```

**实测(`.venv` 真跑):** `seed=0`(文件自带 `__main__` 的默认种子)下,`fp16` 重建误差精确为 `0`,`GPTQ-4bit` 压缩比精确 `4.0` 倍、显存精确 `3337.9 MiB`(`BASE_PARAMS=70亿 × 4bit / 8 / 1024²` 算出来的真实数字,不是手填的),`GPTQ-4bit` 误差(`12.124`)确实低于 `NF4`(`17.2706`),符合 README 宣称。种子 `0-9` 批量扫描,这个排位关系 `10/10` 次成立。种子 `99` 是一个真实反例:`GPTQ-4bit` 误差(`18.3488`)反而**高于** `NF4`(`17.4719`)——和知识点 3 独立发现的"默认 `damp=0.01` 在某些校准数据条件下数值不稳定"是同一个根因在 capstone 自己的具体构造上的复现,不是两个不相关的巧合。`eval_ppl_mock` 用一个纯随机 logits 的 toy predictor 算出的 PPL 是 `123.2`(数值本身没有实际意义,只是确认这个评测函数在合法输入下能正常跑出一个 `≥1` 的困惑度)。

**面试怎么问 + 追问链:**
- **Q:** "怎么系统对比多种量化方案的效果?" —— 期望至少说出"精度(重建误差/PPL/下游任务 acc)、显存、速度(TTFT/ITL/tok-s)、稳定性(极端输入下会不会出问题)"这几个独立维度,并能指出"只看单一维度容易得出片面结论"——比如本知识点的 capstone 只算了精度和理论压缩比/显存,没有测真实速度,lecture L12 提到的"速度测量"部分本身也需要另外的真实 benchmark。
- **追问 1(核心陷阱,考察是否读过 capstone 自己的修正历史):** "这个仓库的量化对比 capstone,数字是真跑出来的还是查表得到的?" —— 期望明确说"现在是真跑的——文件自己的 docstring 记录了一段自我修正历史,旧版本是硬编码的 paper 数字表('看起来齐整但什么都没证明'),现在真的调用 6 个量化器、在同一个 toy 层上算出真实重建 MSE;`compression`/`mem_mib` 两列由每种方法真实的 bit 宽度推算,不是手填常数",能进一步指出"这个模块自己的 L13 lecture 原文其实还没跟上这次重写、仍然写着旧版本的描述"是加分项。
- **追问 2(考察是否理解种子敏感性的意义):** "这个 capstone 显示 GPTQ 比 NF4 精度更高,这个结论可靠吗?" —— 期望说出"在默认种子(以及 0-9 共 10 个种子)下稳定成立,但独立测试发现 seed=99 这个排位会反过来,根因是知识点 3 已经诊断过的 GPTQ 默认 `damp` 参数在某些校准数据条件下数值不稳定——'GPTQ 通常比 NF4 精度更高'是一条有较强证据支持、但不是任何条件下都成立的经验规律,不能当成绝对真理"。
- **追问 3:** "如果你要往这个 capstone 里再加一种量化方法(比如知识点 6 讨论的 LLM.int8() 混合精度),需要改哪些地方?" —— 期望能说出:需要先补全 LLM.int8() 本身的实现(本仓库目前没有,知识点 6 已经讨论过这个空缺),然后在 `_SPECS` 列表里加一行 `(name, bits, quantizer_fn)`,`quantizer_fn` 需要返回反量化后的权重(和 `_q_int8_pc`/`_q_gptq_4bit` 等函数同样的接口约定),`run_all()`/`_zoo_table()` 等其余逻辑不需要改动——这套注册表模式(registry pattern)是这份代码故意设计成"容易横向扩展新方法"的地方。

**常见坑:** 只跑一次(默认种子)capstone 就把结果里的排位关系当成普适结论直接引用——本知识点已经证明,即便是"真跑代码得到的真实数字",也可能在特定随机种子/构造下出现和"通常情况"相反的结果,任何要作为决策依据的排位关系都应该在多组独立设置下重复验证,而不是信一次运行。另一个坑是把 `mem_mib` 这一列理解成"这个 toy 层实际占用的显存"——`BASE_PARAMS=70亿` 是代码里显式声明的"假设这些 bit 宽度用在一个 7B 参数量的真实模型上会占多少显存"这个换算基准,和眼前这个几千参数的 toy `W`/`X` 本身占用的实际内存完全是两回事,`mem_mib` 列是一个"按比例换算到真实规模"的展示数字,不是这次实验实际测量到的进程内存占用。
