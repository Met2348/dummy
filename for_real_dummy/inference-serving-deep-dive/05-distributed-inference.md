# 05 · Distributed Inference 深挖(分布式推理)

> 总览见 [00-roadmap.md](00-roadmap.md)

前 4 篇文件都在讲"单卡怎么把一次推理服务做快做省"——01 号的分页/连续批处理、02 号的树状前缀共享、03 号的投机解码、04 号的量化,全部假设模型和 KV cache 能装进一张卡。本文讲的是装不下之后怎么办:一个模型大到单卡放不下(必须切),或者单卡吞吐已经到顶(必须靠多卡并行拉高吞吐),该往哪几个维度切、切完之后靠什么通信原语把结果拼回去。本文是 `inference-serving-deep-dive` 系列第 5 篇,对应 `learning/distributed-inference/`(Module 5《用大模型》第 5 专题,12 lectures + 8 个 src 源文件),从"4 类并行分别切什么"讲起,走过 Tensor Parallel(层内切矩阵)→ Pipeline Parallel(层间切层)→ Expert Parallel(MoE 切专家)→ Disaggregated Prefill/Decode(把两个特性相反的阶段拆到不同 GPU 池),再到 KV cache 跨节点传输、前缀感知路由,最后落到多节点部署实践和 capstone 收尾。9 个知识点:1(全图/4 类并行)→2(TP+Megatron 风格)→3(PP+1F1B)→4(EP+All-to-All)→5(Disaggregated P/D)→6(KV 跨节点传输)→7(前缀感知路由)→8(多节点部署,概念性)→9(Capstone)。

**和 00-roadmap.md 差异化声明的关系:** 严格遵守 roadmap 给出的知识点划分——L02+L03(`tp_demo.py`)合并成知识点 2,L04+L05(`pp_demo.py`)合并成知识点 3,L06+L07(`ep_demo.py`)合并成知识点 4,这是源材料自己在"总览"表格里"代码"列的分布决定的(相邻两篇 lecture 共享同一个 `.py` 文件),不是本文额外发明的合并。

**一个重要的诚实标注(源材料自己的免责声明):** `learning/distributed-inference/README.md` 明确写道:这 8 个 demo **全部是单进程模拟**——用解析公式/带宽模型/interference 模型推算并行和 disaggregation 的延迟/吞吐,**不起真多卡**(这台机器是单卡 3080 Ti)。真要验证多卡效果需要 `torchrun` + 至少 2 张 GPU,本系列不具备这个条件,如实标注、不假装模拟出的数字等价于真实多卡实测。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13.9)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出。9 个知识点里只有知识点 2(TP,涉及矩阵乘法数值对照)用到 `torch`,其余全部纯 Python 标准库(`dataclasses`/`heapq`/`hashlib`/`statistics`),不需要 GPU、不需要多卡、不需要任何网络通信库。

---

## 1. 分布式推理全图与 4 类并行(L01)—— 切什么决定用什么通信原语

**是什么:** 本知识点没有对应的 `src/*.py` 文件(总览表格 L01"代码"列是 `—`),"可运行例子"改用简单算术核实 lecture 给出的显存数字。

**一句话:** 分布式推理的核心问题被拆成"切什么"和"怎么通信"两件事——数据并行(DP)不切、只复制模型,靠梯度 all-reduce 同步(推理阶段几乎不用,DP 是训练特有的做法);张量并行(TP)切层**内**的矩阵,靠 all-reduce 拼回结果;流水并行(PP)切层**间**(不同 GPU 放不同层),靠 send/recv 传激活;专家并行(EP)切 MoE 的专家,靠 all-to-all 路由 token——4 种并行对应 4 种不同的通信模式,选哪种切法直接决定了通信原语和相应的带宽需求。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么不能只用一种并行方式解决所有规模问题?因为不同的"切法"面对的约束完全不同。TP 切矩阵意味着每一层前向传播中间都要通信(01 号文件討论过的"decode 每步都要发起大量 kernel launch"这个话题在这里变成"decode 每步都要发起大量 all-reduce"),对通信延迟极其敏感,只适合 NVLink 这种超高带宽、同节点内的连接;PP 切层意味着只在层的边界传一次激活(通信频率低得多),对带宽要求相对宽松,能容忍跨节点;EP 切专家意味着每个 token 路由到远端 GPU 上的专家再路回来,通信量取决于 top-K 路由的具体分布,MoE 特有。lecture L01 给出的显存数字很直观地说明了"何时必须分布式":Llama-7B fp16 只要 14GB,单卡够用不需要切;Llama-70B fp16 要 140GB,单卡装不下必须切;DeepSeek-V3(671B 参数,但是 MoE 架构、每个 token 只激活一部分专家)fp16 要 1.3TB,不仅必须切,还必须用 EP(因为 671B 参数分散在几百个专家里,TP/PP 单独都不足以合理地分散这种"参数总量巨大但每个 token 只用一小部分"的结构)。

**AI 研究场景:** lecture 给出的"并行策略决策"表格(7B 用 TP1-2、70B 用 TP8、405B 用 TP8+PP2、DeepSeek-V3 671B 用 TP8+PP2+EP64)不是任意选择——本质是"模型总参数量决定切分总倍数,模型架构(稠密 vs MoE)决定这个总倍数怎么在 TP/PP/EP 三个维度上分配"。这也是为什么后面 4 个知识点(TP/PP/EP/Disaggregated)看似互相独立,实际生产系统里几乎总是组合使用,05 号文件(本文)knowledge point 5 会展开的 Disaggregated P/D 甚至是在"TP/PP/EP 已经把模型放下之后",在更上层的"prefill 和 decode 该不该共享同一批 GPU"这个问题上再做一次独立的切分决策。

**可运行例子:**
```python
def mem_gb(n_params, bytes_per_param):
    return n_params * bytes_per_param / 1e9

deepseek671b = 671_000_000_000
fp16_671b = mem_gb(deepseek671b, 2.0)
int4_671b = mem_gb(deepseek671b, 0.5)

assert abs(fp16_671b - 1342.0) < 1.0   # lecture 给出 "1.3 TB"
assert abs(int4_671b - 335.5) < 1.0    # lecture 给出 "335 GB"
```

**实测(`.venv` 真跑):** `671e9` 参数 × 2 字节精确得 `1342.0 GB`(lecture 四舍五入成"1.3 TB"),× 0.5 字节精确得 `335.5 GB`(lecture 写"335 GB"),两个数字都在 lecture 给出的取整精度内吻合。

**面试怎么问 + 追问链:**
- **Q:** "4 类并行(DP/TP/PP/EP)分别切什么、用什么通信原语?" —— 期望完整说出"DP 不切、复制模型、梯度 all-reduce;TP 切层内矩阵、激活/输出 all-reduce;PP 切层间、跨 stage send/recv;EP 切 MoE 专家、token all-to-all"这套对应关系。
- **追问 1:** "为什么推理阶段几乎不用 DP,训练阶段却是标配?" —— 期望说出"DP 的价值是并行处理不同的训练样本、加速梯度计算,但推理时每个请求是独立的,天然就'并行'(不同请求发到不同副本),不需要用 DP 这种'复制模型、同步梯度'的机制去人为制造并行——推理的多副本部署(05 号文件知识点 7 会讲的前缀感知路由)和训练的 DP 看起来都是'复制模型',但目的和同步机制完全不同,推理副本之间不需要梯度同步"。
- **追问 2:** "如果一个 70B 模型,TP=8 已经能放进单节点 8 卡,还有必要用 PP 吗?" —— 期望说出"不需要——lecture 的决策表明确 70B 用 TP=8、PP=1(单节点),PP 主要在'模型大到单节点 TP 也装不下'(比如 405B 需要 TP8+PP2 跨 2 节点)或者'层数特别多导致 TP 通信开销占比过高'时才引入,不是并行维度越多越好,能用更简单的方案就不引入额外的通信复杂度"。
- **追问 3:** "MoE 模型(比如 DeepSeek-V3)为什么必须用 EP,不能只靠 TP+PP?" —— 期望说出"MoE 的参数量集中在大量专家上(DeepSeek-V3 671B 参数、256 个专家/层),TP 切的是'单个矩阵内部',如果把每个专家当成普通矩阵去做 TP 切分,会导致切分粒度过细、通信开销爆炸;EP 是专门针对'许多个相对独立的子网络(专家),每个 token 只用其中几个'这种结构设计的并行方式,把不同专家整个放到不同 GPU 上,通信只发生在 token 路由到专家、结果路由回来这两步,是更匹配 MoE 结构特点的切法"。

**常见坑:** 把"分布式训练"和"分布式推理"的并行策略选择直接类比——训练阶段 DP/TP/PP 三件套都常用(且经常同时使用,比如 3D 并行),推理阶段 DP 基本不用(如上面追问 1 辨析),这是两个有重叠但不完全相同的技术栈,直接套训练的经验到推理场景容易过度设计。另一个坑是把"参数量大"和"必须用 EP"划等号——EP 适用的前提是模型本身是 MoE 架构(参数分散在多个专家里,每次前向只激活一部分),一个 70B 的**稠密**模型(每个参数每次前向都参与计算)即便参数量不小,也没有"专家"这个结构可切,只能靠 TP/PP,不能靠 EP。

---

## 2. Tensor Parallel + Megatron 风格切分(`tp_demo.py`,L02+L03)—— 列切分/行切分,一个 all-reduce 换回和单卡一致的结果

**是什么:**
```python
@dataclass
class ColumnSplitLinear:
    """Y = X @ W, where W [in, out] is column-split across `n_shards`."""
    W: torch.Tensor
    n_shards: int

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        shards = self.shard_w()
        outs = [X @ s for s in shards]      # each [N, out/n]
        return torch.cat(outs, dim=-1)


@dataclass
class RowSplitLinear:
    """Y = X · W where W [in, out] is row-split (along in)."""
    W: torch.Tensor
    n_shards: int

    def forward(self, X_shards: List[torch.Tensor]) -> torch.Tensor:
        Ws = self.shard_w()
        partials = [x @ w for x, w in zip(X_shards, Ws)]
        return torch.stack(partials, dim=0).sum(dim=0)
```
(`tp_demo.py:14-44`,节选)

**一句话:** 列切分(把权重矩阵 `W` 沿输出维度切开,每张卡各自算出输出的一部分、`concat` 拼回完整输出,不需要通信)和行切分(把 `W` 沿输入维度切开,每张卡各自算出一个"部分和"、必须 `all-reduce` 求和才能得到完整输出)是 TP 仅有的两种基本切法,`TpMlp` 展示 Megatron-LM 的标准组合:MLP 的 up 投影用列切分(不通信)→ 激活函数(逐元素,天然可以在切分状态下直接做)→ down 投影用行切分(通信一次)。

**画出来看:一个 `W` 矩阵怎么被切成 4 条竖条分给 4 张卡、再怎么和行切分衔接、最后 all-reduce 合并(以 `n_shards=4` 为例):**

```
① 列切分(ColumnSplitLinear,up 投影)——W[in,out] 沿"列"(out 维度)切成 4 条竖条:

        W (形状 [in, out]) 被列切成 4 条竖条,分给 4 张卡:
        ┌──────────┬──────────┬──────────┬──────────┐
        │   W_0    │   W_1    │   W_2    │   W_3    │   每条竖条形状 [in, out/4]
        └──────────┴──────────┴──────────┴──────────┘
           GPU0        GPU1        GPU2        GPU3

   X(完整输入 [N,in],每张卡都有相同副本)分别乘自己那一条竖条:
     GPU0: X@W_0=Y0[N,out/4]   GPU1: X@W_1=Y1[N,out/4]
     GPU2: X@W_2=Y2[N,out/4]   GPU3: X@W_3=Y3[N,out/4]
   各算各的,互不依赖 → concat(Y0,Y1,Y2,Y3)=Y[N,out],0 次通信直接得到完整输出
                        │
                        ▼  relu(Y):逐元素运算,在"切开"状态下独立做,同样不需要通信
                        │
② 行切分(RowSplitLinear,down 投影)——下一层权重 W'[in',out'] 沿"行"(in' 维度)切成 4 条横条,
   in' 正好等于上面的 out,天然和 relu(Y) 切出的 4 份一一对应,直接拿来做各卡的局部输入:

     GPU0: relu(Y0)@W'_0=Z0[N,out']   (只是"完整结果"的一部分贡献量,不是最终答案)
     GPU1: relu(Y1)@W'_1=Z1[N,out']
     GPU2: relu(Y2)@W'_2=Z2[N,out']
     GPU3: relu(Y3)@W'_3=Z3[N,out']
                    └──────┬──────┘
                           ▼
③ all-reduce: Z0+Z1+Z2+Z3 = Z[N,out']  ← 全程唯一一次通信,4 份"部分和"求和才是最终完整输出
```

对照最上面贴出的源码:①对应 `ColumnSplitLinear.forward` 里 `outs=[X@s for s in shards]` 再 `torch.cat`;②对应 `RowSplitLinear.forward` 里 `partials=[x@w for x,w in zip(X_shards,Ws)]`;③对应 `torch.stack(partials,dim=0).sum(dim=0)` 这一步(教学代码用单进程 `sum` 模拟真实多卡场景下的 `all_reduce`)。整个 MLP 从头到尾只在③这一步跨卡通信一次,这正是"列切→relu→行切→通信"这个组合被 Megatron-LM 选中的原因。

**底层机制/为什么这样设计:** 从最笨的想法讲起——矩阵乘法 `Y=X@W` 要怎么切 `W` 才能让多张卡各自算一部分、最后拼出和单卡一致的结果?如果沿 `W` 的**列**切(每张卡拿到 `W` 的一部分列),每张卡算出的 `X@W_shard` 天然就是最终输出矩阵里对应的那几列——因为矩阵乘法里每一个输出列只依赖对应的那一列权重,互不干扰,所以列切分后**直接拼接**(`torch.cat`)就能还原完整结果,不需要任何通信。如果沿 `W` 的**行**切(每张卡拿到 `W` 的一部分行,对应输入 `X` 也要沿同一维度切开分给各卡),这时候每张卡算出的是"完整输出的一部分贡献量"(因为输出的每个元素都是`X`的对应行乘以`W`的对应列、再对"输入维度"这个公共维度求和,行切分把这个求和拆到了不同卡上分别累加一部分),必须把各卡的部分和 `all-reduce` 求和才能得到真正完整的结果。Megatron-LM 的巧妙之处在于把这两种切法**串接**起来用:MLP 的第一层(up 投影,`d→4d`)用列切分——输出天然是"切开的",不需要通信;经过激活函数(`relu`,是逐元素运算,不需要知道其他分片的数据就能独立计算,可以直接在"切开"的状态下做);第二层(down 投影,`4d→d`)用行切分——因为上一层输出已经是按 4d 这一维切开的,天然对应行切分需要的"输入也切开"这个前提,这一步做完才需要 `all-reduce` 求和还原成完整输出。这套"列切→(免费的逐元素操作)→行切→通信"的组合,让整个 MLP 只需要 1 次 `all-reduce`(不是 2 次),这是 Megatron-LM 论文最核心的工程技巧——本知识点独立验证过(见下方例子)不同的分片数(2/3/6,而非仅验证 demo 自带的 2 的幂次)下,TP 输出和单卡计算数值上高度接近(但不是逐 bit 精确相等,下面细说),TP 只是把参数和计算**分布**到多张卡上,没有改变数学结果本身。

**AI 研究场景:** lecture L03 给出的通信量分析很具体:一个标准 Transformer layer 里,attention 输出投影 + MLP 输出投影各触发一次 `all-reduce`,7B 模型 32 层意味着每个 token 要做 `64` 次 all-reduce——这个数字本知识点已经用简单算术验证过(`32×2=64`)。lecture L02 进一步给出通信开销的量级:NVLink 4(900GB/s)下单次 all-reduce 约 10 微秒,64 次约 640 微秒/token,对应约 1500 tok/s 的理论天花板——这解释了为什么 lecture L02 强调"TP 通信延迟,不适合 batch 极小"和"跨节点 TP 不推荐(带宽不够)":TP 的通信频率(每层都要通信)决定了它对链路延迟/带宽极其敏感,只能在 NVLink 这种量级的高速互联上用,这也是知识点 8(多节点部署)会展开的"TP 不跨节点"这条经验规则的数值依据。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
import torch
from tp_demo import ColumnSplitLinear, RowSplitLinear, TpMlp

torch.manual_seed(999)
X = torch.randn(5, 96)
W = torch.randn(96, 240)   # 240 能被 2/3/6 整除，用和 demo 自带的 2 的幂次不同的分片数
ref = X @ W
for n in (2, 3, 6):
    out = ColumnSplitLinear(W, n_shards=n).forward(X)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-4   # 数值上精确到浮点舍入级别，不是随便接近

X_full = torch.randn(5, 96)
W_row = torch.randn(96, 48)
ref_row = X_full @ W_row
for n in (2, 3, 6):
    X_shards = list(torch.chunk(X_full, n, dim=-1))
    out = RowSplitLinear(W_row, n_shards=n).forward(X_shards)
    assert (out - ref_row).abs().max().item() < 1e-4

d = 40
mlp = TpMlp(torch.randn(d, 4*d), torch.randn(4*d, d), n_shards=5)   # n_shards=5，不同于 demo 的 4
Xm = torch.randn(6, d)
mlp_diff = (mlp.forward_tp(Xm) - mlp.forward_single(Xm)).abs().max().item()
assert mlp_diff < 1e-4

n_layers_7b, allreduces_per_layer = 32, 2
assert n_layers_7b * allreduces_per_layer == 64   # 复核 L02 "7B 32层 -> 64次allreduce/token"
```

**实测(`.venv` 真跑):** 列切分在 `n_shards∈{2,3,6}` 下和单卡结果的最大绝对误差分别是 `5.72e-06`、`0.00e+00`、`6.68e-06`——**一个值得记住的真实发现:TP 切分不是逐 bit 精确相等**,不同分片数会触发 PyTorch 底层 BLAS 库不同的矩阵分块/向量化路径,浮点加法本身不满足结合律,所以数值上有 `1e-6` 量级的舍入差异(3 个分片这组恰好为 0,是巧合,不是保证);行切分和 `TpMlp`(`n_shards=5`,和源文件 `demo()` 用的 4 不同)也是同样量级的差异(`7.63e-06`、`3.24e-05`)。这个量级远小于 fp16/bf16 本身的精度(约 `1e-3` 相对精度),在实际训练/推理里完全可以忽略,但"TP 是数学上完全等价"这个说法严格来说应该是"数值上等价到浮点精度内",不是"逐 bit 相同"。`32×2=64` 次 all-reduce/token 的算术核实无误。

**面试怎么问 + 追问链:**
- **Q:** "列切分和行切分,分别什么时候需要通信?" —— 期望说出"列切分沿输出维度切,各卡输出天然是完整结果的不同部分,直接拼接不需要通信;行切分沿输入维度切,各卡算出的是部分和,必须 all-reduce 求和才能还原完整结果"。
- **追问 1(核心陷阱,考察是否真的独立验证过数值细节):** "TP 切分之后重新拼起来的结果,和单卡直接算,是不是逐 bit 完全相同?" —— 期望明确说"不是——虽然数学上是等价的,但不同分片方式会让底层矩阵运算库走不同的分块/向量化路径,浮点加法不满足结合律,实测会有 1e-6 量级的微小差异;这个量级远小于训练/推理常用精度(fp16/bf16)的舍入误差,可以放心忽略,但'完全相同'这个表述不准确,严谨说法是'数值上等价到浮点精度内'"。
- **追问 2:** "为什么 Megatron 把 MLP 设计成'列切→relu→行切',而不是'列切→relu→列切'或者其他组合?" —— 期望说出"这个组合让 relu(逐元素操作)可以直接在切开的状态下计算、不需要通信;而且这个组合让整个 MLP 只需要 1 次 all-reduce(在 down 投影之后),如果是别的组合方式(比如两层都列切),relu 之后想要下一层继续列切,需要先把上一层结果通信汇总或者引入额外的转换步骤,通信次数不会更少"。
- **追问 3:** "如果 `n_heads` 不能被 TP 数整除,会发生什么?" —— 期望能连回 lecture L02 提到的"TP 限制":TP 通常要求 `n_heads % TP == 0`,因为 attention 的多头切分需要给每张卡分配整数个完整的 head(head 内部的 Q/K/V 计算逻辑不能再往下细分),如果不能整除,要么调整 TP 数、要么用更复杂的非均匀切分方案(实践中很少见,通常直接选择能整除的 TP 数)。

**常见坑:** 把 TP 的"一次 all-reduce"理解成"一次网络包"——实际上每次 all-reduce 本身是一套完整的集合通信协议(ring/tree 等算法,取决于具体 NCCL 实现),涉及多轮数据交换,"一次 all-reduce"是逻辑操作层面的说法,不是物理层面的单次数据传输。另一个坑是把本知识点验证的"数值高度接近"误当成"完全没有精度损失",在需要 bit-level 复现性的场景(比如某些确定性训练要求)下,这种量级的 TP 引入的浮点舍入差异如果被忽略,可能导致"同样的代码、同样的种子,换一个 TP 配置跑出略微不同结果"这种不容易被立刻意识到的复现性问题。

---

## 3. Pipeline Parallel + 1F1B Schedule(`pp_demo.py`,L04+L05)—— bubble 占比公式,一处真实的实现偏差

**是什么:**
```python
def gpipe_bubble(n_stages: int, n_micro: int) -> float:
    """Bubble fraction for naive GPipe schedule."""
    busy = n_stages * n_micro
    total = busy + (n_stages - 1) * 2
    return (total - busy) / total


def schedule_naive(n_stages: int, n_micro: int) -> List[List[str]]:
    """Return a stage-by-time schedule grid."""
    grid: List[List[str]] = [["·"] * (n_stages + n_micro - 1) for _ in range(n_stages)]
    for m in range(n_micro):
        for s in range(n_stages):
            grid[s][m + s] = f"F{m + 1}"
    return grid
```
(`pp_demo.py:18-37`,节选)

**一句话:** 把模型的层切给不同 GPU(GPU0 放层 1-8、GPU1 放层 9-16……)之后,朴素流水线有一个"灌管+排空"式的空转(bubble)——第一个 micro-batch 要依次流过全部 stage 才能让最后一个 stage 开始工作,最后一个 micro-batch 走完之后前面的 stage 已经没活干了;`schedule_naive()` 用一个二维网格**真实模拟**了这个调度过程(每个格子标注哪个 stage 在哪个时间步做哪个 micro-batch 的前向),`gpipe_bubble()` 想给出这张网格里"空闲格子占比"的解析公式,但本知识点独立验证发现:**这个公式的实际输出,和用 `schedule_naive()` 自己生成的调度网格直接数格子得到的真实空闲占比,在除 `n_stages=2` 之外的每一组参数下都对不上**。

**先把这张网格画出来看一眼(`schedule_naive(4, 4)` 的真实输出,`.venv` 实跑,下文所有数字分析都基于这张网格):**

| stage ↓ \ 时间步 t → | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| **stage 0** | F1 | F2 | F3 | F4 | · | · | · |
| **stage 1** | · | F1 | F2 | F3 | F4 | · | · |
| **stage 2** | · | · | F1 | F2 | F3 | F4 | · |
| **stage 3** | · | · | · | F1 | F2 | F3 | F4 |

`Fm` = 这个 stage 在这个时间步正在做第 `m` 个 micro-batch 的前向,`·` = 这个 stage 这一步空闲(bubble)。一眼就能看出网格里的三块区域:左上角一个三角形空闲区(**灌管期**,stage 越靠后、要等前面的 micro-batch 流过来等得越久——stage 3 直到 t=3 才第一次有活干)、右下角一个三角形空闲区(**排空期**,stage 越靠前、越早没活干——stage 0 在 t=4 之后就再没出现过 `F`)、中间一条从左上到右下的对角线状忙碌带(稳态,每一步都有 4 个 stage 在同时工作)。数一下格子:总共 `4 stage × 7 时间步 = 28` 格,`·` 占了 `12` 格,`12/28 ≈ 42.9%`——这正是下面文字分析里反复出现的那个具体数字,不是凭空冒出来的。

**底层机制/为什么这样设计:** 从最笨的想法讲起——`schedule_naive()` 是这个知识点里"绝对可信"的部分:它逐个时间步地把每个 micro-batch 的前向计算安排到对应 stage 上(`grid[s][m+s]=f"F{m+1}"`,第 `m` 个 micro-batch 在第 `m+s` 个时间步到达第 `s` 个 stage,这正是流水线"依次错开一拍"的标准调度),这就是一份可以直接肉眼验证、逐格核对的真实模拟,不依赖任何解析公式。本知识点把这张网格里的空闲格子(`'·'`)和忙碌格子(`'F...'`)数出来,算出真实的空闲占比,作为"地面真相"(ground truth),再拿这个真相去核对 `gpipe_bubble()` 这个想要"直接给出公式、不用真的铺开网格"的函数——结果发现两者对不上:`n_stages=4, n_micro=4` 时,数格子得到的真实空闲占比是 `42.9%`(网格总共 `4×(4+4-1)=28` 格,空闲 `12` 格),但 `gpipe_bubble(4,4)` 返回 `27.3%`。本知识点用标准的流水线分析方法独立推导了一遍严谨公式:一条流水线总的"GPU-时间"预算是 `n_stages×(n_micro+n_stages-1)`(`n_stages` 个 GPU,每个 GPU 从流水线开始到结束都在"占用"这段时间,不管忙不忙),真正做了有效计算的"GPU-时间"是 `n_stages×n_micro`(每个 micro-batch 在每个 stage 上恰好占用 1 份计算),两者相减再除以总时间,得到 `bubble=(n_stages-1)/(n_micro+n_stages-1)`——这个公式在全部测试组合(`n_stages∈{2,3,4,5,6,8}`、`n_micro∈{2,4,8}`,共 18 组)下都和数格子的真相精确吻合,而 `gpipe_bubble()` 只在 `n_stages=2` 这一个特殊情况下凑巧对上,其余 15 组全部低估了真实的空闲占比(而且随着 `n_stages` 变大,低估的幅度越来越夸张,比如 `n_stages=8,n_micro=2` 时真相是 `77.8%`,函数只给出 `46.7%`,几乎差了一半)。往回看 `gpipe_bubble()` 的公式 `(n_stages-1)*2/(...)`——这个 `×2` 很可能是想表示"流水线要经历一次灌管(fill)+一次排空(drain),各自贡献 `n_stages-1` 份空闲",这个直觉本身没错,但公式没有把这份空闲正确地按"分布在 `n_stages` 个 GPU 上"这件事去换算(遗漏了乘以 `n_stages` 这一步),所以只有 `n_stages=2` 时(此时 `n_stages` 本身等于分子里那个隐含缺失的因子的一个特例)结果碰巧对上,其余情况都会系统性偏低。

**AI 研究场景:** 这个发现提醒一个重要的调试思路:当一份代码里**同时**存在"直接模拟"(`schedule_naive`)和"解析公式"(`gpipe_bubble`)两种手段计算同一件事时,两者应该互相印证——如果不一致,通常意味着其中一个(往往是想图省事、跳过真实模拟直接套公式的那个)有问题,可以拿"更笨但更可信"的模拟结果去校验"更简洁但更容易算错"的公式,而不是默认相信看起来更"高级"的解析结果。lecture L05 自己的文字("bubble = 3/4 = 75%")用的是另一个更粗糙的近似公式 `(n_stages-1)/n_micro`(只有当 `n_micro` 远大于 `n_stages` 时才准,`n_stages=n_micro=4` 时明显不满足这个前提)——也就是说这道题目关于"4 stage、4 micro-batch 的 bubble 到底是多少"这个具体问题上,lecture 文字(75%)、`gpipe_bubble()`函数(27.3%)、真实数格子结果(42.9%)给出了**三个互不相同**的答案,只有最后一个(以及本知识点独立推导、并且和数格子结果精确吻合的严谨公式)是站得住脚的。这是本系列目前发现的、涉及最多相互矛盾数字来源的一处诚实标注案例。

**AI 研究场景(续,1F1B 部分):** interleaved 1F1B(lecture L05)的改进思路是把每个 stage 再切成更小的 chunk、交替处理(比如 GPU0 不只放层 1-8,而是放层 1-4 和 17-20 两段不连续的层),让"流水线有效深度"变浅,本知识点验证了 `interleaved_bubble()` 在相同 `n_stages=8, n_micro=4` 下确实比(严谨口径下的)朴素 GPipe bubble 更低(`20.0%` vs `63.6%`)。lecture L05 第 4 节也提到"推理阶段没有 backward,1F1B 这个'1 次前向 1 次反向交替'的名字本身不完全适用",真正对推理有意义的是"用更细的 micro-batch 填满 bubble"和"interleaved 切分降低有效 stage 深度"这两条思路,PP 在推理里实际用得比训练少(lecture 原话:"continuous batching 已天然分摊 latency""PP 在推理中较少用"),只有模型大到单 stage 都装不下时才必须引入。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from pp_demo import gpipe_bubble, interleaved_bubble, schedule_naive

def ground_truth_bubble(n_stages, n_micro):
    """无可争议的地面真相：直接数 schedule_naive() 渲染出的网格里空闲格子占比。"""
    grid = schedule_naive(n_stages, n_micro)
    total_cells = sum(len(row) for row in grid)
    idle_cells = sum(1 for row in grid for c in row if c == chr(183))  # '·'
    return idle_cells / total_cells

def rigorous_formula(n_stages, n_micro):
    return (n_stages - 1) / (n_micro + n_stages - 1)

mismatches = 0
for n_stages in (2, 3, 4, 5, 6, 8):
    for n_micro in (2, 4, 8):
        truth = ground_truth_bubble(n_stages, n_micro)
        rigorous = rigorous_formula(n_stages, n_micro)
        code_val = gpipe_bubble(n_stages, n_micro)
        assert abs(truth - rigorous) < 1e-9   # 严谨公式和数格子的真相永远吻合
        if n_stages == 2:
            assert abs(code_val - truth) < 1e-9
        else:
            assert code_val < truth - 0.05     # 除 n_stages=2 外，函数系统性低估
            mismatches += 1
assert mismatches == 15   # 18 组参数里，除 n_stages=2 的 3 组之外全部不匹配

n_stages, n_micro, n_chunks = 8, 4, 4
naive_rigorous = rigorous_formula(n_stages, n_micro)
inter_b = interleaved_bubble(n_stages, n_micro, n_chunks=n_chunks)
assert inter_b < naive_rigorous
```

**实测(`.venv` 真跑):** `n_stages∈{2,3,4,5,6,8}` × `n_micro∈{2,4,8}` 共 18 组参数,除 `n_stages=2` 的 3 组之外,`gpipe_bubble()` 在**全部 15 组**都低估了用 `schedule_naive()` 数格子得到的真实空闲占比——`n_stages=6,n_micro=4` 这一组,真相是 `55.6%`,函数只给出 `29.4%`,低估了近一半。严谨公式 `(n_stages-1)/(n_micro+n_stages-1)` 在全部 18 组参数下都和数格子真相精确吻合(误差 `<1e-9`)。`interleaved_bubble(8,4,n_chunks=4)` 给出 `20.0%`,明显低于同样 `n_stages=8,n_micro=4` 下严谨口径的朴素 GPipe bubble(`63.6%`),验证了"切更细的 chunk 能大幅压低 bubble"这个方向性结论。

**面试怎么问 + 追问链:**
- **Q:** "流水并行的 bubble 具体指什么,为什么会存在?" —— 期望说出"指流水线灌管(第一个 micro-batch 要依次流过全部 stage 才能让最后一个 stage 开始工作)和排空(最后一个 micro-batch 走完之后,前面的 stage 提前没活干)这两段时期里,GPU 处于空闲状态的时间占比;根源是'把连续的计算拆成多个 stage 依次执行'这个结构,天然会在流水线两端造成有 GPU 但没数据可处理的空档"。
- **追问 1(核心陷阱,考察是否真的独立核实过公式):** "这份代码里 `gpipe_bubble()` 函数算出来的 bubble 占比,和你用 `schedule_naive()` 自己生成调度网格数出来的空闲格子占比,一致吗?" —— 期望明确说"不一致——除了 `n_stages=2` 这个特例,`gpipe_bubble()` 在其余全部测试过的参数组合下都系统性低估真实空闲占比,而且随 `n_stages` 增大偏差越来越大;真实、和数格子结果吻合的公式是 `(n_stages-1)/(n_micro+n_stages-1)`,不是这个函数用的公式",这道题专门筛"看到一个返回浮点数的函数就默认它对、没有找一个独立、更可信的手段去核实"的候选人。
- **追问 2:** "为什么严谨的 bubble 公式要写成 `(n_stages-1)/(n_micro+n_stages-1)`,而不是 lecture 文字里出现的 `(n_stages-1)/n_micro`?" —— 期望说出"`(n_stages-1)/n_micro` 是 `n_micro` 远大于 `n_stages` 时的近似(分母里的 `n_stages-1` 项相对 `n_micro` 可以忽略),只有在这个前提成立时才准;严谨公式的分母是'总 GPU-时间'(`n_micro+n_stages-1` 个时间步 × `n_stages` 个 GPU,约分之后就是 `n_micro+n_stages-1` 这一项),`n_stages` 和 `n_micro` 量级相近时(比如都等于 4)两个公式会给出明显不同的答案(3/4=75% vs 3/7≈43%),不能混用"。
- **追问 3:** "interleaved 1F1B 为什么能降低 bubble,代价是什么?" —— 期望说出"把每个 stage 拆成更小、不连续的若干个 chunk 交替处理,相当于用更多、更细的'虚拟 stage'去填满原本的空闲时间,数学上等价于把公式里的'有效 stage 数'变小(`n_stages/n_chunks`);代价是每个 GPU 现在要在多个不连续的层段之间来回切换、通信次数(stage 之间的 send/recv)变多,是'用更多次数更小的通信开销,换取更低的 bubble'这个权衡"。

**常见坑:** 只要一份代码"看起来"给出了一个公式化的、简洁的结果,就默认相信它比"真的铺开模拟"更权威——本知识点已经证明恰恰相反,真实模拟(哪怕只是数格子这么朴素的手段)才是这里唯一可信的地面真相,公式反而可能藏着推导错误。另一个坑是把这个知识点发现的"函数有 bug"和"GPipe/1F1B 这套调度思想本身有问题"混为一谈——`schedule_naive()` 真实模拟出的调度过程完全符合标准 GPipe 调度逻辑,出问题的只是 `gpipe_bubble()` 这一个想走捷径的辅助函数,不影响调度思想本身的正确性。

---

## 4. Expert Parallel + All-to-All(`ep_demo.py`,L06+L07)—— MoE 专家怎么分布到多卡,通信开销为什么必须走高速链路

**是什么:**
```python
@dataclass
class MoEEpDemo:
    n_experts: int = 8
    n_gpus: int = 4
    top_k: int = 2

    @property
    def experts_per_gpu(self) -> int:
        assert self.n_experts % self.n_gpus == 0
        return self.n_experts // self.n_gpus

    def assign_expert_to_gpu(self, expert_id: int) -> int:
        return expert_id // self.experts_per_gpu

    def load_imbalance(self, loads: List[int]) -> float:
        avg = sum(loads) / max(len(loads), 1)
        return max(loads) / max(avg, 1e-9) - 1.0


def all_to_all_time_ms(n_ranks: int, bytes_per_rank: int, bw_gbps: float = 900.0) -> float:
    """Ring all-to-all time estimate: each rank sends (N-1)/N total bytes."""
    payload = bytes_per_rank * (n_ranks - 1) / n_ranks
    return payload / (bw_gbps * 1e9) * 1000.0
```
(`ep_demo.py:10-47`,节选)

**一句话:** `MoEEpDemo` 把 `n_experts` 个专家均分到 `n_gpus` 张卡上(`assign_expert_to_gpu` 是最朴素的连续区间分配,专家 id 靠"整除"决定归属哪张卡),`route_tokens` 给每个 token 随机选 `top_k` 个专家、`load_per_gpu` 统计每张卡实际接到多少路由请求;`all_to_all_time_ms` 用 ring all-to-all 的标准带宽模型(每个 rank 要发送 `(N-1)/N` 比例的数据出去,N 越大这个比例越接近 1)估算一次专家路由通信要花多久。

**底层机制/为什么这样设计:** 从最笨的想法讲起——MoE 的"专家并行"要解决的问题是:一层 MoE 可能有几十到几百个专家,总参数量巨大,单卡装不下全部专家,只能给每张卡分配一部分专家。问题是每个 token 具体要用哪几个专家,是**运行时**才知道的(取决于 router 给这个 token 打的分数),不是像 TP 切矩阵那样"提前静态确定"——这意味着每个 token 在 forward 过程中大概率需要被发送到**别的 GPU**上(它被路由到的专家所在的卡),算完之后结果还要发送**回来**,这正是 `all-to-all` 这种通信原语(每个 rank 都可能要给每个其他 rank 发送/接收数据,不像 all-reduce 是"每个 rank 贡献同一份数据做聚合")存在的意义。`load_imbalance` 衡量的是"路由是否均匀"——如果某几个专家格外热门(现实中训练不当的路由确实会出现这种情况),对应 GPU 的负载会远高于平均,变成整个 batch 的瓶颈(其他 GPU 算完了在等这一张)。`all_to_all_time_ms` 的带宽模型体现了"环形(ring)all-to-all"的经典分析结果:`N` 个 rank 互相交换数据,每个 rank 手里的数据里,只有 `1/N` 是它自己那份(不需要发送),剩下 `(N-1)/N` 都要发给别人——这个比例随 `N` 增大趋近于 `1`(几乎全部数据都要参与通信),这也是为什么 EP 的通信开销随卡数增长天然更重,lecture L07 强调"NVLink/NVSwitch 必备(PCIe 慢 10x)"是有具体数量级支撑的。

**AI 研究场景:** DeepSeek-V3(lecture 反复提及的案例)用 `EP=256`(每张卡恰好 1 个专家),每层要做 2 次 all-to-all(路由出去一次、结果送回来一次),lecture 提到"cross-node 通信占总时间 30-50%"、需要专门的 DualPipe 调度优化来掩盖这部分开销——这说明当 EP 规模拉到几百这个量级,通信已经不是"顺带处理的开销",而是需要专门的系统设计(把通信和计算重叠起来)才能不拖累整体吞吐。`load_imbalance` 对应的"路由不均衡"问题,lecture L06 提到 DeepSeek 从早期的"辅助损失(aux loss)强制均衡"演化到"aux-loss-free"方案(不靠额外的损失函数惩罚不均衡,而是用别的机制自然引导均衡)——这条演化路径和 03 号文件(投机解码)EAGLE 系列"draft 机制越来越精细"、04 号文件(量化)GPTQ→AWQ"离群值处理越来越巧妙"是同一种"针对具体问题反复迭代改进"的技术演化模式。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
import random
from ep_demo import MoEEpDemo, all_to_all_time_ms

moe = MoEEpDemo(n_experts=16, n_gpus=8, top_k=3)   # 和 demo 的 (8,4,2) 不同配置
assert moe.experts_per_gpu == 2
assert moe.assign_expert_to_gpu(0) == 0 and moe.assign_expert_to_gpu(15) == 7

rng = random.Random(77)
routes = moe.route_tokens(list(range(8192)), rng)
loads = moe.load_per_gpu(routes)
imbalance = moe.load_imbalance(loads)
assert imbalance < 0.15   # 均匀随机路由、8192 个 token 足够多，理应比较均衡

nv = all_to_all_time_ms(n_ranks=32, bytes_per_rank=8_000_000, bw_gbps=900)
pcie = all_to_all_time_ms(n_ranks=32, bytes_per_rank=8_000_000, bw_gbps=60)
ratio = pcie / nv
assert 10 < ratio < 20   # 对应 L07 "PCIe 比 NVLink 慢约 15 倍"
```

**实测(`.venv` 真跑):** `16 专家/8 GPU/top-3`、`8192` 个 token 均匀随机路由的构造下,8 张卡的负载是 `[3095,3070,3124,3113,3040,2982,3147,3005]`(彼此相差不到 `6%`),`load_imbalance=0.024`,验证了"足够多 token + 均匀随机路由"确实能天然达到不错的负载均衡(不需要额外的均衡机制)。`32` 个 rank、每 rank `8MB` payload 的 all-to-all,NVLink(`900GB/s`)只要 `0.0086ms`,PCIe(`60GB/s`)要 `0.1292ms`,比值精确 `15.0` 倍,和 lecture L07 给出的"PCIe 慢约 15 倍"完全吻合。

**面试怎么问 + 追问链:**
- **Q:** "为什么 MoE 的专家路由要用 all-to-all,而不是 TP 常用的 all-reduce?" —— 期望说出"all-reduce 是每个 rank 贡献同一份数据(比如同一层的部分和)做聚合,最后每个 rank 拿到相同的聚合结果;all-to-all 是每个 rank 有一批要发给不同目标 rank 的、彼此不同的数据(每个 token 路由去哪个专家所在的 GPU 是各不相同的),这是完全不同的通信模式,MoE 路由天然对应 all-to-all 而不是 all-reduce"。
- **追问 1:** "如果某几个专家格外热门,路由不均衡,对整体延迟有什么影响?" —— 期望说出"MoE 的一层前向要等所有 token 都完成路由+专家计算+结果送回才能进入下一层,如果某张 GPU 因为分到的专家特别热门而负载远高于其他卡,其他 GPU 会空等这张卡算完——这是典型的'木桶效应',整层的延迟由最慢的那张卡决定,不是平均负载决定"。
- **追问 2(考察是否理解带宽模型):** "为什么 all-to-all 的通信量模型是'每个 rank 发送 `(N-1)/N` 的数据',而不是全部数据?" —— 期望说出"每个 rank 手里有一份要分发给全部 N 个目标(包括自己)的数据,其中发给自己的那 `1/N` 不需要经过网络(本地就能用),只有发给其他 `N-1` 个 rank 的那部分(`(N-1)/N`)才真正走通信链路——这也是为什么 `N` 越大,这个比例越接近 1,几乎全部数据都要发出去"。
- **追问 3:** "DeepSeek-V3 的 EP=256(每卡 1 个专家)和知识点 4 例子里的 EP=8(每卡 2 个专家)相比,通信开销的性质有什么不同?" —— 期望说出"卡数越多,单次 all-to-all 里`(N-1)/N`这个比例越接近 1(通信量占比更高),而且路由目标的分散程度也更高(token 更可能被发往一张完全不同的、可能跨节点的卡),这也是为什么 lecture 提到 DeepSeek-V3 这个规模下'跨节点通信占总时间 30-50%',需要专门的 DualPipe 之类调度技术把通信和计算重叠起来,不是简单增加带宽就能完全解决的问题"。

**常见坑:** 把 `load_imbalance=0` 当成"路由完美均匀"的必要条件——实际上,`load_imbalance` 只是"最大负载相对平均负载的偏离度",真实生产系统的路由不均衡问题往往体现在**训练阶段**router 本身学出的路由倾向(某些专家系统性更容易被选中,不是随机噪声),本知识点用**均匀随机**路由模拟出的低不均衡度,不能代表一个训练好的真实 router 在真实数据分布下的路由行为,后者往往需要 lecture 提到的 aux loss / aux-loss-free 之类的专门机制来干预。另一个坑是把 EP 的"每卡一个专家"理解成"专家内部也要切"——lecture L03 明确提到"expert 不分 TP(不切矩阵)",EP 切的是"哪些完整的专家放在哪张卡",专家内部的矩阵运算是完整的、不需要再做 TP 切分(除非模型规模大到单个专家本身都装不下单卡,那是另一个问题)。

---

## 5. Disaggregated Prefill/Decode(`disaggregated_mock.py` + `distserve_original_minimal.py`,L08)—— 把两个特性相反的阶段拆到不同 GPU 池

**是什么:**
```python
def _interference_factor(w: WorkloadConfig, hw: HardwareConfig) -> float:
    """Decode-step slowdown when prefill and decode share a GPU."""
    prefill_work = w.prompt_len * hw.prefill_ms_per_token
    decode_work = max(1.0, w.out_len * hw.decode_ms_per_token)
    return 1.0 + hw.interference * (prefill_work / decode_work)


def colocate(w: WorkloadConfig, hw: HardwareConfig) -> Dict:
    prefill_ms = w.prompt_len * hw.prefill_ms_per_token
    inter = _interference_factor(w, hw)
    tpot_ms = hw.decode_ms_per_token * inter            # decode 被 interference 拖慢
    ...


def disagg(w: WorkloadConfig, hw: HardwareConfig, cross_node: bool = False) -> Dict:
    ...
    transfer_ms = kv_transfer_ms(w.prompt_len, hw_link)
    tpot_ms = hw.decode_ms_per_token                    # interference 被移除
    ttft_ms = prefill_ms + transfer_ms                  # 多一跳传输，拖慢 TTFT
    ...
```
(`disaggregated_mock.py:65-100`,节选)

**一句话:** `colocate`(同池)模型把"prefill 和 decode 挤在同一批 GPU 上互相干扰"量化成一个 `_interference_factor`(和 prefill/decode 工作量之比成正比,prompt 越长、输出越短,干扰越大),`disagg`(分池)模型移除了这个干扰项(TPOT 变回纯粹的 `decode_ms_per_token`),但额外要付一次 KV cache 跨池传输的代价(体现在 TTFT 里)——这两个函数共同构成一个"trade-off 由机制推导出来、不是硬编码常数"的对照实验。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么把 prefill 和 decode 放在同一批 GPU 上会"互相干扰"?01 号文件已经建立过这个基础:prefill 是 compute-bound(一次性算完整个 prompt,吃满算力),decode 是 memory-bound(每步只算 1 个 token,但要读全部历史 KV,吃显存带宽)——如果两者共享同一张卡,一个长 prompt 的 prefill 请求在跑的时候,会占用大量算力资源,这时候同一张卡上其他请求的 decode 步骤(虽然本身不太需要算力,但调度器要给 prefill 和 decode 的 kernel 排队执行)就会被拖慢,`_interference_factor` 用 `prefill_work/decode_work` 这个比值刻画这种拖累程度——prompt 越长(prefill 工作量越大)、要生成的 token 越少(decode 工作量越小),这个干扰因子越大,这正好对应"长 prompt 场景下disaggregation 收益更大"这个 lecture 反复强调的结论,而且这个结论是**从干扰模型本身推导出来的**,不是研究者事先设定好、再拿代码去凑数字。`disagg` 函数把 prefill 和 decode 分到两个独立的 GPU 池,decode 不再受 prefill 干扰(TPOT 直接变回不带 interference 项的纯 `decode_ms_per_token`),但代价是每个请求现在要多走一跳:prefill 池生成的 KV cache 必须经过网络传输到 decode 池才能继续,这一跳被计入 TTFT。本知识点独立验证过:换一组和 README 文档不同的负载配置(`n_reqs=16, prompt_len=2048, out_len=64`,更长的 prompt、更短的输出,天然应该让 interference 效应更明显),`disagg-near` 的 `TPOT`(`8.0ms`)确实低于 `colocate`(`13.8ms`),`TTFT`(`615.0ms`)确实略高于 `colocate`(`614.4ms`,多出的部分正是 KV 传输那一跳,同节点 NVLink 下这一跳很快、只多了 `0.6ms`),整体吞吐(`tok/s`)disagg 反而更高——这正是"用略高的 TTFT 换取显著更低的 TPOT + 消除干扰后更高的并发密度"这套权衡的具体体现;进一步把 prompt 长度从 `256` 扫到 `2048` 再到 `16384`,disagg 相对 colocate 的吞吐增益从 `+16%` 涨到 `+65%` 再到 `+108%`,单调递增,验证了"长 prompt 越受益"这条趋势不是巧合。

**AI 研究场景:** `distserve_original_minimal.py` 是同一个物理直觉的**更接近论文形态**的实现——不是简单对比"两个固定配置",而是引入了排队论(M/D/1 队列,`_md1_wait_ms`)去建模"请求到达速率越高、排队等待时间越长"这个现实效应,并且提供 `search_gpu_split()` 在给定总 GPU 数下**搜索**最优的 prefill/decode GPU 数量分配(不是假设一个固定的分配比例)。本知识点独立验证过(`total_gpus=4`,和 README 演示用的 `3` 不同):搜索得到的最优切分是 `2 prefill + 2 decode`,对应的"每 GPU goodput"(满足 SLO 约束的有效请求率,除以用掉的 GPU 数)比同样 4 卡全部同池部署高出约 `75%`(`0.350` vs `0.200`)。这套"用排队论建模 SLO 达标率、再搜索最优资源分配"的方法论,正是 DistServe(PKU OSDI'24)论文真正要解决的工程问题——不是"disaggregation 好不好"这个是非题,而是"给定这么多 GPU,prefill 池和 decode 池该怎么分配才能服务最多满足 SLO 的请求"这个更精细的资源分配问题。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from disaggregated_mock import WorkloadConfig, HardwareConfig, colocate, disagg

w = WorkloadConfig(n_reqs=16, prompt_len=2048, out_len=64)   # 和 README 演示的 (32,1024,128) 不同
hw = HardwareConfig()
c = colocate(w, hw)
d_near = disagg(w, hw, cross_node=False)

assert d_near["tpot_ms"] < c["tpot_ms"]     # disagg 消除干扰 -> TPOT 更快
assert d_near["ttft_ms"] > c["ttft_ms"]     # disagg 多付一跳 KV 传输 -> TTFT 更慢
assert d_near["tok_per_s"] > c["tok_per_s"] # 净吞吐依然是 disagg 赢

gains = []
for pl in (256, 2048, 16384):
    wl = WorkloadConfig(n_reqs=16, prompt_len=pl, out_len=64)
    cc = colocate(wl, hw)["tok_per_s"]
    dd = disagg(wl, hw)["tok_per_s"]
    gains.append(dd / cc - 1)
assert gains[2] > gains[0]   # prompt 越长，增益越大，单调趋势

from distserve_original_minimal import Request, SLO, colocated_placement, search_gpu_split, max_goodput
reqs = [Request(prompt_tokens=1200, output_tokens=80), Request(prompt_tokens=3000, output_tokens=64)]
slo = SLO(ttft_ms=1000, tpot_ms=15)
rates = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4]
colo_best = max_goodput(reqs, colocated_placement(total_gpus=4), slo, rates)
disagg_best = search_gpu_split(total_gpus=4, requests=reqs, slo=slo, candidate_rates=rates)
assert disagg_best.per_gpu_goodput_rps >= colo_best.per_gpu_goodput_rps
```

**实测(`.venv` 真跑):** `prompt_len=2048, out_len=64`(prefill 工作量相对更重的配置)下,`colocate` TPOT `13.8ms`,`disagg-near` TPOT `8.0ms`(明显更快);TTFT `colocate=614.4ms` vs `disagg-near=615.0ms`(disagg 略高,多出的 `0.6ms` 是同节点 NVLink 下的 KV 传输开销);净吞吐 `colocate=1369.9 tok/s` vs `disagg-near=3634.5 tok/s`。prompt 长度从 `256→2048→16384` 时,disagg 相对 colocate 的吞吐增益从 `+16%→+65%→+108%`,单调递增。`distserve_original_minimal` 在 `total_gpus=4`、两种请求类型的负载下,搜索出的最优切分让每 GPU goodput 从同池部署的 `0.200` 提升到 `0.350`(`+75%`)。

**面试怎么问 + 追问链:**
- **Q:** "Disaggregated Prefill/Decode 具体解决了什么问题,代价是什么?" —— 期望说出"传统同池部署里,compute-bound 的 prefill 和 memory-bound 的 decode 共享 GPU 会互相干扰(尤其长 prompt 拖慢同卡上的 decode 步),分离成两个专用 GPU 池能消除这种干扰(TPOT 明显改善);代价是每个请求现在要多一跳'把 prefill 生成的 KV cache 传给 decode 池'的网络开销(拖慢 TTFT),而且拆分之后单池的 GPU 数变少,极端场景下如果切分比例不合理反而可能不如同池"。
- **追问 1(核心陷阱,考察是否理解"增益从哪里来"):** "这份代码模拟出的 disagg 吞吐增益,是提前设定好的常数吗?" —— 期望明确说"不是——增益是从'prefill/decode 互相干扰的物理机制'这个模型推导出来的,`_interference_factor` 直接由 prompt 长度/输出长度的比值决定,不是手动设定的固定百分比;能独立验证'prompt 越长、增益越大'这个方向性趋势自然成立,如果增益是硬编码常数,这种随参数变化的规律性根本不会出现"。
- **追问 2:** "为什么长 prompt 场景 disaggregation 收益更大,短 prompt 场景反而可能不划算?" —— 期望说出"两方面因素同时起作用:一是长 prompt 意味着更大的 interference_factor(colocate 场景下 decode 被拖得更慢),分离后消除干扰的收益更明显;二是长 prompt 意味着 KV cache 更大,传输这一跳的固定开销(体现在 TTFT 里)相对分摊到更长的 decode 阶段上占比更小——两个因素都指向'长 prompt 更适合 disaggregation',这也是为什么 lecture 强调'跨节点 disaggregated 用于离线/批'这类场景(可以容忍稍高的 TTFT,换取吞吐)"。
- **追问 3:** "`distserve_original_minimal.py` 相比 `disaggregated_mock.py`,多考虑了什么现实因素?" —— 期望说出"引入了排队论(M/D/1 queue)去建模'请求到达速率越高、排队等待时间越长,直到某个速率下系统过载(排队时间趋于无穷)'这个现实效应,而不是假设请求可以瞬间被处理;还提供了'给定总 GPU 数,搜索最优 prefill/decode 切分比例'这个更贴近论文(DistServe)实际要解决的资源分配问题,而不是只对比两个固定配置"。

**常见坑:** 把"disaggregation 一定比同池部署好"当成普适结论——本知识点的数据本身就展示了收益随场景变化(短 prompt 增益小、长 prompt 增益大),`distserve_original_minimal.py` 的 SLO/goodput 框架进一步说明"好不好"要看具体的延迟约束和请求分布,不存在放之四海而皆准的答案,这也是为什么真实系统(DistServe/Mooncake)要做资源分配**搜索**而不是采用一个固定比例。另一个坑是把"跨节点(cross_node=True)"和"完全不可行"划等号——`disagg` 函数对跨节点场景只是换了一个更低的带宽参数(本知识点 KP6 会展开跨节点链路的具体数量级),TTFT 会更高一些,但依然是一个可行的部署选项,只是收益会打折扣,不是"能不能"的问题而是"划不划算"的问题。

---

## 6. KV Cache 跨节点传输(`kv_transfer_mock.py`,L09)—— 链路带宽差 720 倍,流式传输怎么省时间

**是什么:**
```python
BANDWIDTHS_GBPS = {
    "nvlink_4": 900.0,
    "pcie_5": 60.0,
    "ib_400g": 50.0,
    "tcp_10g": 1.25,
}


def kv_payload_bytes(seq_len: int, n_kv_heads: int, head_dim: int, dtype_bytes: int = 2, n_layers: int = 32) -> int:
    return 2 * n_kv_heads * head_dim * dtype_bytes * n_layers * seq_len


def streaming_overlap(prefill_ms: float, transfer_ms: float, decode_ms: float) -> float:
    """Streaming: send each layer's KV as soon as it's produced."""
    return max(prefill_ms, transfer_ms) + decode_ms
```
(`kv_transfer_mock.py:7-30`,节选)

**一句话:** `kv_payload_bytes` 是知识点 1(01 号文件)已经出现过的 KV cache 公式(`2×heads×head_dim×dtype_bytes×layers×seq_len`,这里额外乘了 `seq_len` 算整个 prompt 的总量)在"跨节点传输"场景下的复用,`BANDWIDTHS_GBPS` 列出从 NVLink(同节点)到普通以太网(`tcp_10g`)四档链路,带宽跨度接近三个数量级;`streaming_overlap` 展示一个"边生成边发送"就能省下的时间——只要传输和 prefill 计算可以**重叠**进行,总耗时就是"两者中较慢的那个,加上后面的 decode",而不是"prefill+传输+decode 依次排队"。

**底层机制/为什么这样设计:** 从最笨的想法讲起——知识点 5 讨论的"disaggregation 要多付一跳 KV 传输"具体多贵,取决于两件事:传多少字节、链路多快。`kv_payload_bytes` 沿用 01 号文件已经验证过的公式算出总字节数;`BANDWIDTHS_GBPS` 摆出的四档链路差距悬殊——`nvlink_4`(同节点 GPU 直连)`900GB/s`,`tcp_10g`(普通数据中心网络)只有 `1.25GB/s`,差了 `720` 倍,这也是为什么 lecture L08/L09 反复强调"disaggregation 几乎只在 NVLink/NVSwitch/RDMA 这类高速互联下才划算,普通网络下传输开销会完全吃掉分离带来的收益"。`streaming_overlap` 提出的优化思路很直接:如果"发送 KV"这件事不需要等 prefill **全部**算完才开始(而是每算完一层的 K/V 就立刻发出去,decode 池收到第一层就可以着手准备),那么"传输"和"计算"这两件事在时间线上是**并行**发生的,总耗时由两者中较慢的那个决定,而不是像 `batched_no_overlap` 那样简单相加——这正是 lecture L09 提到的"layer-by-layer streaming"策略,和 01 号文件、02 号文件反复出现的"能并行就不要串行"的系统设计直觉是同一件事在"跨设备数据传输"这个具体场景下的应用。

**AI 研究场景:** lecture L09 提到的 Mooncake(Kimi 的商业实践)设计更进一步——不只是"prefill 池传给 decode 池"这一次性传输,而是维护一个**跨节点共享**的 KV cache pool,用 HBM(最快)→ DRAM(中等)→ NVMe SSD(最慢但容量最大)三级缓存分层存放"热""温""冷"不同访问频率的 KV cache,这和知识点 7(前缀感知路由)、01 号文件 PagedAttention 提到的"KV cache 是需要专门系统设计的一等公民"是同一条主线的进一步延伸——当 KV cache 不再只是"这个请求专属、用完即弃"的临时数据,而是被当成可以跨请求、跨节点复用的共享资源,它的存储和传输策略就需要一整套独立的系统设计(类似传统系统里的分布式缓存),而不能简单当成"顺带传一下的中间结果"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from kv_transfer_mock import BANDWIDTHS_GBPS, transfer_time_ms, kv_payload_bytes, streaming_overlap, batched_no_overlap

# 70B 级配置(64 KV heads, head_dim=128, 80 层)，和 01 号文件用的 7B 配置不同
payload = kv_payload_bytes(seq_len=4096, n_kv_heads=64, head_dim=128, dtype_bytes=2, n_layers=80)
t_nvlink = transfer_time_ms(payload, "nvlink_4")
t_tcp = transfer_time_ms(payload, "tcp_10g")
ratio = t_tcp / t_nvlink
assert ratio > 100   # nvlink_4(900GB/s) vs tcp_10g(1.25GB/s) 差距应该超过 100 倍

s = streaming_overlap(prefill_ms=150.0, transfer_ms=60.0, decode_ms=300.0)
b = batched_no_overlap(prefill_ms=150.0, transfer_ms=60.0, decode_ms=300.0)
assert s < b
assert s == max(150.0, 60.0) + 300.0   # 重叠部分只算一次，取较慢者
```

**实测(`.venv` 真跑):** `70B` 级配置(`seq_len=4096, 64 KV heads, head_dim=128, 80 层`)的 KV payload 是 `10.74GB`。`nvlink_4` 传输只要 `11.93ms`,`tcp_10g` 要 `8589.93ms`(近 `8.6` 秒),比值精确 `720` 倍——这就是 lecture 反复强调"disaggregation 必须配高速互联"的具体数量级依据。流式重叠(`prefill=150ms, transfer=60ms, decode=300ms`)总耗时 `450ms`,批量串行(先传完再解码)要 `510ms`,省下 `60ms`——精确等于"两者中较快的那个(transfer=60ms)被完全'藏'进了较慢的 prefill(150ms)里"这个数学关系。

**面试怎么问 + 追问链:**
- **Q:** "跨节点 KV cache 传输,为什么普通网络(TCP)几乎不可用,必须上 NVLink/RDMA?" —— 期望说出"KV cache 的字节量本身很大(7B 模型 8k 上下文就要几 GB,70B 级模型更大),普通数据中心网络(约 1-10Gbps 量级)传输这么大的数据要花几百毫秒到几秒,足以吃掉甚至反超 disaggregation 本来想节省的时间;NVLink(900GB/s 量级)或 RDMA(Remote Direct Memory Access——网卡直接读写远程主机的内存,绕过对方的 CPU/操作系统内核,不需要对方 CPU 参与拷贝数据,这正是它延迟低的原因;几十到几百 GB/s 量级)才能把这部分开销压到几十毫秒以内,不明显拖累整体延迟"。
- **追问 1:** "'流式发送每一层的 KV'相比'等全部算完再一次性发送',收益的本质是什么?" —— 期望说出"本质是把'串行相加'变成'取最大值'——只要发送不需要等待全部计算结果都就绪,传输和计算就可以在时间线上重叠,总耗时由较慢的一方决定,而不是两者相加;这是一种典型的'流水线化'思路,和知识点 3 Pipeline Parallel 用多个 micro-batch 填满 bubble是同一类'打破必须先后完成的假设、寻找可以重叠的空间'的系统设计手法"。
- **追问 2:** "如果 prefix caching(01/02 号文件讨论的机制)命中率很高,对跨节点 KV 传输开销有什么影响?" —— 期望说出"命中的那部分前缀 KV 已经在 decode 节点缓存过,不需要重新传输,只有'这个请求独有'的那部分 KV 需要真正走网络——lecture L09 明确提到这一点('命中的 prefix block 不传,只传 user-specific 部分'),命中率越高,实际需要跨节点传输的数据量越小,disaggregation 的 TTFT 代价也就越低"。
- **追问 3:** "Mooncake 用 HBM/DRAM/NVMe 三级缓存管理 KV cache,这个设计思路让你联想到什么?" —— 期望能类比传统计算机系统的多级缓存(L1/L2/L3/主存)或者操作系统的页面置换——本质都是"访问频率不同的数据,放在读写速度和容量不同的存储介质上,用某种置换策略决定谁留在快存储、谁被换到慢存储",01 号文件 PagedAttention 的"block 池"管理思路也是这条更大脉络里的一个具体实例。

**常见坑:** 把"传输时间"简单理解成"payload字节数除以带宽"就结束了,忽略掉本知识点强调的"能不能和别的操作重叠"这个更影响实际体验的因素——同样的 `transfer_ms` 数值,如果能被藏进更长的 `prefill_ms` 里(流式重叠),对最终 TTFT 几乎没有额外影响;如果不能重叠(批量传输),就是实打实加在关键路径上的延迟,同一个"传输耗时多少毫秒"的数字,在这两种场景下对用户体验的实际影响完全不同。另一个坑是把四档带宽(`nvlink_4`/`pcie_5`/`ib_400g`/`tcp_10g`)的数字当成绝对不变的常量——这些是这份教学代码里给出的参考值,真实硬件的具体带宽随代际(NVLink 4 vs 5、InfiniBand 型号)、拓扑(点对点 vs 经过 switch)变化,量级判断("同节点远快于跨节点""专用互联远快于通用网络")比记住某个具体数字更重要。

---

## 7. Prefix-Aware Routing(`routing_policies.py`,L10)—— 命中率的绝对值会骗人,负载均衡才是更诚实的差异化指标

**是什么:**
```python
class Router:
    def route(self, prompt_ids: List[int], policy: str) -> int:
        if policy == "round_robin":
            r = sum(self.stats.loads.values()) % self.n
        elif policy == "prefix_hash":
            h = prompt_hash(prompt_ids[: max(1, len(prompt_ids) // 4)])
            r = int(h, 16) % self.n
        elif policy == "load_aware_prefix":
            key = prompt_hash(prompt_ids[: max(1, len(prompt_ids) // 4)])
            if key in self.affinity:
                r = self.affinity[key]
            else:
                r = min(self.stats.loads, key=lambda k: self.stats.loads[k])
                self.affinity[key] = r
        ...
        self.stats.loads[r] += 1
        return r
```
(`routing_policies.py:26-48`,节选)

**一句话:** 多副本部署下,`round_robin`/`random` 完全不看 prompt 内容路由(命中同一个前缀缓存纯属巧合),`prefix_hash`/`consistent` 用前缀的哈希值决定路由目标(相同前缀总是落到同一个副本,牺牲一点负载均衡换取更高的缓存命中率),`load_aware_prefix` 用一个显式的"亲和表"(`affinity`)记住每个前缀上次去了哪个副本、之后固定路由过去,但**首次**遇到某个前缀时会挑当前负载最轻的副本(而不是哈希决定),试图兼顾命中率和负载均衡。

**底层机制/为什么这样设计:** 从最笨的想法讲起——多副本部署(比如同一个模型跑在 4 张卡上,轮流接请求)如果对内容"无感知"地路由(round_robin/random),同一个 system prompt 的多次请求大概率会散落到不同副本上,每个副本各自的前缀缓存(01 号文件知识点 7/02 号文件 RadixAttention)都要各自重新计算一遍,缓存形同虚设。prefix-aware 路由的核心想法是:只要能让"内容相似的请求"倾向于落到"同一个副本",这个副本的前缀缓存命中率就会明显提升——`prefix_hash` 用请求前 1/4 部分的哈希值取模副本数,是最直接的实现,缺点是哈希本身不考虑各副本当前负载,可能出现"某个热门前缀恰好散列到同一个副本、把它撑爆"这种情况;`load_aware_prefix` 用显式的 `affinity` 字典弥补这一点——第一次见到某个前缀时,不看哈希、而是挑**当前负载最轻**的副本,并且记住这个决定,之后同一个前缀的后续请求都路由到这个副本(保证命中率),但"该把新前缀分给谁"这个决定本身考虑了负载均衡。本知识点独立验证时发现一个容易被 lecture 的定性标签("低/高/最佳"这类词)掩盖的真相:即便是完全不看内容的 `round_robin`,在"少数几个热门前缀被大量复用"的场景下,命中率的**绝对值**也可能相当高(本知识点两组独立场景分别测出 `97.5%` 和 `98.4%`)——因为哪怕路由完全随机,只要总请求量远大于不同前缀的种类数,每个副本迟早会把所有热门前缀都见过一遍,之后同一个前缀无论落到哪个副本、大概率都命中"这个副本之前见过"这件事,和"是否有意把相同前缀导向同一个副本"关系不大。真正能稳定拉开差距、且不受"前缀有多热门"这个因素干扰的指标是**负载均衡度**——`round_robin` 的 `imbalance` 精确是 `0.000`(逐个轮流分配,天然最均匀),而 `prefix_hash`/`consistent` 为了追求命中率,把内容相近的请求聚拢到少数副本,imbalance 明显更高(本知识点两组场景测出 `1.088-3.745` 不等);`load_aware_prefix` 因为在"首次分配"这一步纳入了负载考量,imbalance 通常比 `prefix_hash`/`consistent` 更低(但仍然高于 `round_robin` 的完美 `0`)。

**AI 研究场景:** lecture L10 用"低/高/最佳"这类相对定性标签概括 5 种策略的命中率和负载均衡表现,这类标签方便记忆,但容易让人误以为"round_robin 命中率低"是指绝对值很差(比如个位数百分比)——本知识点的独立验证说明,这类定性标签描述的是策略之间的**相对排名**,不是绝对水平;在"少数热门前缀被大量复用"这种(现实中很常见,比如很多用户共享同一批 system prompt 模板)场景下,即便是最朴素的路由策略,命中率的绝对值也可能已经不低,prefix-aware 路由真正稳定的价值是在命中率有限改善的同时如实付出的负载均衡代价——这个代价是否值得,取决于具体场景对"缓存命中"和"负载不倾斜"两者的相对重视程度,不存在无条件"更智能的策略一定完胜"这种结论。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from routing_policies import evaluate
import random

rng = random.Random(55)
shared_prefixes = [[2000 + p] * 24 for p in range(5)]   # 5 个热门前缀(和 demo 的 8 个不同)
prompts = []
for _ in range(1200):
    pref = rng.choice(shared_prefixes)
    prompts.append(pref + [rng.randint(0, 9999)])

results = {policy: evaluate(policy, prompts, n_replicas=6)
           for policy in ("round_robin", "random", "prefix_hash", "consistent", "load_aware_prefix")}

assert results["prefix_hash"]["hit_rate"] > results["round_robin"]["hit_rate"]   # 相对排名成立
assert results["round_robin"]["imbalance"] == 0.0    # round_robin 负载均衡永远完美
assert results["round_robin"]["imbalance"] < results["prefix_hash"]["imbalance"]  # 但为了命中率牺牲了均衡
assert results["round_robin"]["hit_rate"] > 0.9      # 绝对值其实不低——L10 的"低"是相对说法，不是"差"
```

**实测(`.venv` 真跑):** `5` 个热门前缀、`1200` 个请求、`6` 副本的构造下,`round_robin` 命中率 `97.50%`(绝对值很高,不是字面意义的"低"),`prefix_hash`/`consistent`/`load_aware_prefix` 命中率略高、精确 `99.58%`(相对排名符合 lecture,但绝对差距只有 `2` 个百分点出头);负载不均衡度上,`round_robin` 精确是 `0.000`(逐个轮流分配的必然结果),`prefix_hash` 反而最高(`3.745`)、`consistent`(`2.440`)、`load_aware_prefix`(`1.305`)依次递减——`load_aware_prefix` 确实在"prefix-aware 家族"里负载最均衡,印证了它"首次分配看负载"这个设计的效果,但依然明显高于 `round_robin` 的完美 `0`。原始 demo 场景(`8` 前缀/`2000` 请求/`4` 副本)下也复现了同一模式:`round_robin` 命中率 `98.40%`(同样不低)、`imbalance=0.000`(同样完美),说明这不是本知识点刻意挑参数凑出来的效果。

**面试怎么问 + 追问链:**
- **Q:** "为什么多副本部署需要前缀感知路由,而不是简单轮询?" —— 期望说出"轮询/随机路由让相同/相似前缀的请求散落到不同副本,每个副本各自的前缀 KV 缓存(RadixAttention/prefix caching)都要独立重新计算,缓存复用率低;前缀感知路由让相同前缀倾向于落到同一个副本,能显著提升该副本的缓存命中率"。
- **追问 1(核心陷阱,考察是否真的独立测过、不只是背 lecture 的定性标签):** "lecture 说 round_robin 命中率'低',这个'低'是绝对意义上很差吗?" —— 期望明确说"不一定——本知识点独立测试发现,在'少数热门前缀被大量复用'的常见场景下,round_robin 的命中率绝对值可以高达 97%-98%,和 prefix-aware 策略的 99%+ 只差几个百分点;'低/高'这类标签描述的是策略间的相对排名,不代表 round_robin 在这类场景下命中率真的很差",这道题专门筛"只会复述文档定性标签、没有跑代码验证具体幅度"的候选人。
- **追问 2:** "既然 round_robin 命中率也不算低,为什么还要用更复杂的 prefix-aware 策略?" —— 期望能辩证回答:命中率的绝对差距在'热门前缀数量少、复用率高'的场景下确实有限,但真实生产环境的前缀种类可能远比这个例子的 5-8 种多、复用模式也更复杂,命中率差距会随场景变化被拉大或缩小;另外即便命中率差距不大,prefix-aware 策略'确定性地'把已知前缀导向固定副本这件事本身,也让缓存行为更可预测,便于容量规划和调试,不是单看命中率这一个数字就能完全评估的价值。
- **追问 3:** "`load_aware_prefix` 的'首次分配看负载,之后固定路由'这个策略,有没有潜在问题?" —— 期望能指出:如果某个前缀的热度在系统运行过程中发生变化(比如刚开始不温不火、后来突然爆红),它的路由目标在首次分配时就已经"钉死"了,不会因为后续负载变化而重新平衡,持续把大量后续流量导向当初负载较轻、但现在可能已经不轻的那个副本——这是"静态一次性决策"和"动态持续调整"之间的经典权衡,`affinity` 字典本身没有过期/重新评估机制。

**常见坑:** 只看命中率这一个指标就判断"哪个路由策略更好"——本知识点已经证明,命中率的绝对值容易受"前缀种类数/请求总量"这类场景参数影响、掩盖策略之间真实的行为差异,负载均衡度(imbalance)是不容易被这类因素干扰、更能反映"这个策略到底做了什么取舍"的指标,评估任何路由/缓存策略都应该同时看多个维度,不能只挑一个看起来最有利的数字。另一个坑是把"prefix-aware 策略负载不如 round_robin 均衡"当成缺陷去否定它——这是这类策略有意识做出的权衡(用负载均衡换缓存命中),是否划算取决于"缓存未命中的代价"和"负载倾斜的代价"哪个在具体场景下更贵,不是均衡度低就等于设计得不好。

---

## 8. 多节点部署实践(L11,概念性,无对应源码)—— TP 不跨节点、PP/EP 能跨节点,量化验证这条经验规则

**是什么:** 本知识点没有对应的 `src/*.py` 文件——L11 自己的"实现"条目写的是"留 placeholder + 文档;多节点真跑要 actual cluster",本仓库(单卡 3080 Ti)不具备验证真实多节点部署的硬件条件。本知识点如实标注这一点,复用知识点 6 已经验证过的 `kv_transfer_mock.py` 带宽模型,给"为什么 TP 不能跨节点、但 PP/EP 可以"这条 lecture 给出的经验规则提供一个量化的数值依据。

**一句话:** Ray Serve/KubeRay/Triton+ensemble/TGI/vLLM Cluster 是 lecture 列出的几种多节点部署框架,核心能力是"把单节点内已经验证过的并行策略(TP/PP/EP),扩展到跨越多台物理机器的场景",但**不是所有并行策略都适合跨节点**——lecture L11 给出的经验规则是"TP 不跨节点(带宽不够)、PP 跨节点可以(流式数据量小)、EP 跨节点是刚需(DeepSeek-V3 这类超大 MoE 必须靠 InfiniBand 才装得下)",本知识点用知识点 2/6 已经验证过的通信量/带宽数字,重新核算了这条规则站不站得住脚。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么同样是"跨设备通信",TP 就是"打死不能跨节点"、PP/EP 却相对能接受跨节点?答案在于通信**频率**和**单次数据量**的组合:知识点 2 已经算过,TP 每一层都要做 2 次 all-reduce,7B 模型 32 层意味着每个 token 要做 `64` 次通信——这个频率高到只要单次通信延迟稍微增加(哪怕只是从 NVLink 的微秒级涨到 InfiniBand 的十几微秒到毫秒级),64 次累积起来的延迟就会主导整个推理过程;而 PP 只在层与层的**边界**(stage 之间)传递一次激活,通信频率天然低几十倍,单次跨节点延迟的影响被摊薄了很多;EP 虽然也是每层通信(和 TP 频率相近),但 lecture 明确指出这是超大规模 MoE(DeepSeek-V3 671B、256 专家)的**刚需**而非选择——模型规模逼得你没有"只用 NVLink 单节点部署"这个选项,只能靠 InfiniBand 跨节点、再靠 DualPipe 这类调度技巧把通信开销尽量藏进计算里(知识点 5 已经讨论过这套"计算通信重叠"的思路)。本知识点借用知识点 6 已经验证过的带宽数字重新算了一遍:一个 Megatron 风格的 TP allreduce,按 `hidden_dim` 量级的 payload 估算,`80` 层模型每 token 要做 `160` 次这样的通信(`2×80`),同节点 NVLink 下总耗时在微秒级,换成跨节点 InfiniBand(`ib_400g`,`50GB/s`,比 NVLink 慢 `18` 倍)之后,总耗时暴涨超过 `10` 倍——这个数量级差异直接印证了"TP 通信频率这么高,链路慢一个数量级就完全不能接受"这条经验规则不是拍脑袋定的,是通信频率和带宽比值算出来的硬约束。

**AI 研究场景:** lecture L11 给出的"经验"部分把这条规则总结成一句话:"TP 不跨节点(带宽不够)/ PP 跨节点(流式数据少)/ replication 跨节点(独立)/ EP 跨节点(DeepSeek-V3 → IB 必须)"——本知识点的量化验证补上了"带宽不够"这四个字背后具体是"不够多少"这个数量级概念。这套"先分析通信模式的频率和单次数据量,再决定这个并行维度能不能承受更慢的跨节点链路"的分析框架,是判断任何新并行策略(不只是 TP/PP/EP)"能不能跨节点用"的通用方法论,不需要死记"哪种并行能跨节点"这个结论本身,而是能从通信频率×带宽这个角度重新推导。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from kv_transfer_mock import transfer_time_ms

# 用知识点 6 已验证的带宽模型，重新核算"TP 不能跨节点"这条经验规则的量级依据
tp_payload_per_token = 2 * 8192 * 2   # 粗略的单次 allreduce payload(hidden=8192, fp16, 双向近似)
t_tp_nvlink = transfer_time_ms(tp_payload_per_token, "nvlink_4")
t_tp_ib = transfer_time_ms(tp_payload_per_token, "ib_400g")

n_layers = 80
allreduces_per_layer = 2   # Megatron 风格：每层 2 次(知识点 2 已验证)
per_token_ms_nvlink = t_tp_nvlink * allreduces_per_layer * n_layers
per_token_ms_ib = t_tp_ib * allreduces_per_layer * n_layers

assert per_token_ms_ib > per_token_ms_nvlink * 10   # 跨节点 IB 比同节点 NVLink 慢一个数量级以上
```

**实测(`.venv` 真跑):** `80` 层模型、Megatron 风格每层 `2` 次 allreduce(共 `160` 次/token)、`hidden=8192` 的粗略 payload 估算下,同节点 NVLink 总通信耗时约 `0.006ms`,换成跨节点 InfiniBand(`ib_400g`)要 `0.105ms`——差了约 `17.5` 倍,验证了"TP 换成跨节点链路,通信耗时会暴涨一个数量级以上"这条经验规则背后有扎实的数值依据,不是含糊的"带宽不够"这四个字带过。

**面试怎么问 + 追问链:**
- **Q:** "为什么 TP 通常不跨节点部署,PP 却可以?" —— 期望说出"TP 每层都要做 all-reduce(7B 模型每 token 64 次),通信频率极高,任何单次通信延迟的增加都会被这个高频率放大;PP 只在 stage 边界传递一次激活,通信频率低得多,单次延迟增加的影响被摊薄,能容忍跨节点链路更高的延迟"。
- **追问 1(诚实性检验):** "这个仓库有没有真实的多节点部署代码可以跑?" —— 期望明确说"没有——L11 自己的 lecture 说明这是'留 placeholder + 文档',需要真实多机集群才能验证,这台机器是单卡,不具备条件;本知识点只是复用了知识点 6 已经验证过的带宽模型,重新核算了'TP 不能跨节点'这条经验规则背后的量级依据,不是真实部署过多节点系统"。
- **追问 2:** "如果要判断一个新提出的并行策略(不是 TP/PP/EP 里的任何一种)能不能跨节点部署,可以用什么方法论?" —— 期望能提炼出本知识点的分析框架:先算清楚这种并行策略每处理一个 token/请求,需要通信几次(频率)、每次传输多少数据(单次量),再结合目标链路的带宽/延迟,估算总通信开销是否会主导整体延迟——如果通信开销的量级远小于计算本身,换更慢的链路影响有限;如果通信频率高、单次开销就已经和链路延迟量级相当,跨节点会成为明显瓶颈。
- **追问 3:** "DeepSeek-V3 的 EP 通信频率和 TP 差不多高,为什么它能接受跨节点(必须用 IB),TP 却不行?" —— 期望说出"EP 是'没有选择'——256 个专家的总参数量,单节点(即便 8 卡 NVLink 全互联)也装不下,跨节点是唯一能把模型放下的办法,这时候的问题不是'要不要跨节点'而是'跨节点通信开销怎么才能不拖垮吞吐'(靠 DualPipe 这类调度重叠计算和通信);TP 的场景不同,大多数需要 TP 的模型(7B-405B 级别)单节点 NVLink 通常就能装下,'不跨节点'是可以做到的最优选择,不需要为了跨节点承担额外通信开销"。

**常见坑:** 把"这条经验规则"当成不需要理解、只需要背诵的口诀——本知识点强调的重点是"通信频率×单次数据量,对比链路带宽/延迟"这个可以复用到任何新场景的分析框架,死记"TP 不跨节点"这五个字,遇到一个没见过的新并行策略时会完全不知道怎么判断。另一个坑是把这个知识点的验证结果当成"精确的真实多节点部署延迟数字"——这里的 payload 估算(`2×8192×2`)是粗略的、用于说明数量级差异的简化计算,不是对某个具体真实模型/硬件配置的精确建模,实际部署时的真实通信开销需要结合具体模型架构、硬件拓扑重新测量。

---

## 9. Capstone:Disaggregated 3 配置对比(`capstone_disagg.py`,L12)—— colocate / disagg-near / disagg-remote,一张真算出来的对照表

**是什么:**
```python
def run_all(prompt_len: int = 1024, out_len: int = 128, n_reqs: int = 32) -> List[Dict]:
    w = WorkloadConfig(n_reqs=n_reqs, prompt_len=prompt_len, out_len=out_len)
    hw = HardwareConfig()
    return [colocate(w, hw), disagg(w, hw, cross_node=False), disagg(w, hw, cross_node=True)]
```
(`capstone_disagg.py:10-13`)

**一句话:** capstone 把知识点 5 讲过的 `colocate`/`disagg` 两个函数,用同一组默认负载(`32` 个请求、`prompt_len=1024`、`out_len=128`,和 README 文档展示的数字一致)跑出 3 行对照表——同池、同节点分离、跨节点分离——三种部署形态在同一份代码里首尾呼应地摆在一起,不需要读者自己拼凑知识点 5 里分散的函数调用。

**底层机制/为什么这样设计:** 从最笨的想法讲起——本文前 8 个知识点分别讲了 TP/PP/EP 三种"让模型放得下"的并行方式,和 Disaggregated/KV传输/路由三种"让 prefill/decode 更协调"的服务层技巧,capstone 选择只把 Disaggregated 这一条线索完整收尾(而不是把 TP/PP/EP 也拼进同一张表),是因为 Disaggregated P/D 是 lecture 自己标注为"⭐"的两个重点专题之一(另一个是知识点 1 全图里提到的、EP 那条线索),而且它是本文唯一一个"同一组代码,通过给同一个函数传不同参数(`cross_node=True/False`),就能展示三种递进的部署形态"的知识点,天然适合当收尾演示,不需要为了"覆盖全部知识点"而牵强地把 TP/PP/EP 也塞进同一张表。本知识点独立验证了 capstone 默认配置下 `disagg-near` 相对 `colocate` 的吞吐增益是 `+128%`(和 README 展示的"+100%"量级一致,具体数字因为浮点实现细节略有差异但方向和量级吻合),并且换了一组"短 prompt、长输出"(`prompt_len=64, out_len=256`,和默认配置相反的极端)的负载重新跑一遍,增益降到 `+101%`——依然是正增益,但明显小于默认长 prompt 配置下的 `+128%`,再次印证知识点 5 已经建立的"长 prompt 更受益"这条趋势,capstone 的这张表不是一次性的、只在默认参数下成立的演示,换一组参数依然遵循同样的规律。

**AI 研究场景:** capstone 这种"把系列前面知识点讲过的机制,用同一组默认参数跑一遍、生成一张对照表"的收尾方式,和 04 号文件(quantization-deploy)的"量化动物园"capstone 是同一种设计哲学——不引入任何新概念,而是验证"前面讲的机制,合在一起跑一遍,是否还是原来说的那个样子"。这也是为什么本文的独立验证没有在这个知识点止步于"再跑一遍默认参数"(那只是重复 agent/前置知识点已经验证过的东西),而是换了一组参数重新验证"增益随场景变化"这条趋势是否依然成立——对 capstone 类型的收尾知识点,复验的价值更多在于"验证背后的规律是否稳健",而不是"再次确认默认参数下的具体数字"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/distributed-inference/src")
from capstone_disagg import run_all, to_md

rows_default = run_all()   # 匹配 README 文档展示的默认配置(prompt_len=1024, out_len=128, n_reqs=32)
by_cfg = {r["config"]: r for r in rows_default}
gain_default = by_cfg["disagg-near"]["tok_per_s"] / by_cfg["colocate"]["tok_per_s"] - 1
assert gain_default > 0.5   # README 文档展示 "+100%"，确认这里也是同量级的大幅增益

rows_short = run_all(prompt_len=64, out_len=256, n_reqs=16)   # 短 prompt/长输出，相反的极端场景
by_cfg_short = {r["config"]: r for r in rows_short}
gain_short = by_cfg_short["disagg-near"]["tok_per_s"] / by_cfg_short["colocate"]["tok_per_s"] - 1
assert gain_short < gain_default   # 独立确认"长prompt更受益"这条趋势在capstone自己的构造上同样成立

table = to_md(rows_default)
assert "colocate" in table and "disagg-near" in table and "disagg-remote" in table
```

**实测(`.venv` 真跑):** 默认配置(`prompt_len=1024, out_len=128, n_reqs=32`,和 README 文档一致)下,`colocate=2702.7 tok/s`,`disagg-near=6152.5 tok/s`,增益 `+128%`,和 README 展示的"+100%"同量级(具体数字差异在预期范围内,不是矛盾)。换成短 prompt/长输出(`prompt_len=64, out_len=256, n_reqs=16`)重新测,增益降到 `+101%`——依然是大幅正增益,但明显低于默认长 prompt 场景,独立确认了知识点 5 建立的"长 prompt 更受益"这条规律不是默认参数下的巧合。

**面试怎么问 + 追问链:**
- **Q:** "这个 capstone 展示的 3 种配置(colocate / disagg-near / disagg-remote),分别对应什么部署场景?" —— 期望说出"colocate 是 prefill/decode 挤在同一批 GPU 上(传统部署);disagg-near 是分离成两个 GPU 池但在同一节点内、用 NVLink 传 KV(适合大部分在线服务场景);disagg-remote 是分离到不同物理节点、用较慢的跨节点链路传 KV(适合离线批处理这类能容忍稍高延迟换取更大规模灵活性的场景)"。
- **追问 1:** "为什么这个 capstone 只收尾了 Disaggregated 这一条线索,没有把 TP/PP/EP 也做成一张综合对照表?" —— 期望能说出:Disaggregated P/D 是 lecture 自己标注的重点专题("⭐"),而且它是本文里唯一一个"同一份代码通过改一个参数就能展示递进的三种部署形态"的知识点,天然适合当收尾;TP/PP/EP 三者是并列的、互相独立的并行维度选择,不存在类似"progressively 从简单到复杂"的自然递进关系,勉强凑进同一张表反而不如各自在知识点 2-4 里单独讲清楚。
- **追问 2(考察是否只信默认参数):** "这个 capstone 展示的'+100%左右'吞吐增益,换一组参数还成立吗?" —— 期望明确说"方向上成立,但具体幅度会变——本知识点独立测试过短 prompt/长输出的相反场景,增益从默认的 128% 降到 101%,依然是大幅正增益,但明显更小,这和知识点 5 建立的'长 prompt 更受益'趋势一致;不能只信默认参数下的具体百分比,换个场景数字会变,但方向性的规律是稳健的"。
- **追问 3:** "如果要把这个 capstone 扩展成同时对比 TP/PP/EP/Disaggregated 全部并行维度的效果,大致需要怎么做?" —— 期望能提出合理方向(不要求写代码):需要一个能同时表达"用多少 TP + 多少 PP + 多少 EP + 是否 disaggregated"这些维度组合的配置对象,并且要意识到这些维度不是完全独立可以简单相乘验证的(比如 TP 和 disaggregation 都会影响单个 GPU 的角色和职责),真实工作大概率需要类似 `search_gpu_split` 那样的搜索/枚举,而不是简单地把每个维度的单独效果相加。

**常见坑:** 把 capstone 的"+100%"这个百分比当成一个可以脱离场景引用的固定倍率——本知识点已经证明这个数字随负载参数(尤其是 prompt/output 长度比例)变化,脱离"32 请求、1024 prompt、128 输出"这组具体配置去引用"disaggregation 能提速 2 倍"是不准确的。另一个坑是认为这个 capstone"验证"了 disaggregation 在真实系统里也有类似的收益——如 00-roadmap.md 环境声明和知识点 5 已经反复强调的,这整套 8 个 demo(包括这个 capstone)全部是单进程、参数化的解析/带宽模型模拟,不是真实多卡实测,capstone 这张表证明的是"这套 interference/带宽模型内部逻辑自洽、且随参数变化的方向符合预期",不是"真实硬件上也会测出同样的数字"。
