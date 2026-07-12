# 02 · 长上下文 Attention 架构深挖(Long-Context Attention Architectures)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 上一批([01-rope-scaling-family.md](01-rope-scaling-family.md))解决的是"怎么告诉模型每个 token 排第几号"(位置编码怎么外推到训练时没见过的长度);这一批解决的是完全正交的另一个问题——就算位置编码能处理 100 万 token 的坐标,**算 attention 本身这一步**(`O(L²)` 的算力和显存)会不会先把 GPU 撑爆。这是面试里"你打算怎么训练/部署长上下文模型"这类问题的核心考点,也是最容易只会背"Ring Attention 就是切成块"这种口号、答不出"切完块之后数值凭什么还完全不变"的一块。

**本文定位:** 本篇对应 `learning/long-context/src/{ring_attention_naive,infini_attention}.py` 和 `lectures/{06,07,08}-*.md`,讲的是三条互相竞争的"长上下文怎么不爆显存"技术路线——不是同一个方案的三个版本,而是三种不同的设计哲学(精确并行 / 负载均衡工程优化 / 有损压缩)。第 4 个知识点会把它们放进同一张表对比选型。

本文所有可运行例子已在仓库根目录 `.venv`(Windows 原生,Python 3.13,纯 CPU 路径,和 [00-roadmap.md](00-roadmap.md) 的环境声明一致)下实际跑通验证,凡是写出来的 diff/shape 数值,全部是现场跑出来的,不是转述文档或凭经验断言。

**本篇统一结构(与 00-roadmap.md 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制 / 为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究场景
5. 可运行例子(带 `assert` 验证,真的在仓库 `.venv` 里跑过)
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

---

## 1. Ring Attention(`ring_attention_naive.py`)—— 切块 + online softmax 递推,单卡模拟多卡环形通信

**是什么:**
```python
def vanilla_attn(q, k, v):
    """标准 scaled-dot-product attention:一次性看到完整序列,算出的结果是后面所有对照的"标准答案"."""
    d = q.shape[-1]
    return ((q @ k.transpose(-2, -1)) / math.sqrt(d)).softmax(dim=-1) @ v


def ring_attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          n_rank: int = 4) -> torch.Tensor:
    """单 GPU 模拟 ring attention:把 q,k,v 沿 seq 维度切成 n_rank 块,
    用 online softmax 递推,模拟"n_rank 张卡"协作拼出和 vanilla_attn 完全一致的结果."""
    ...
```
路径:`learning/long-context/src/ring_attention_naive.py`。

**一句话:** Ring Attention 把 Q/K/V 沿序列维切成 `n_rank` 块分给"n_rank 张卡"(这里用一个 for 循环在单进程里模拟),每张卡永远只装自己的 Q 块,K/V 块像接力棒一样绕一圈("ring"传递),配合 online softmax 递推,最终拼出来的结果和一次性算完整个序列的标准 attention **在数值上完全一致**——它是并行策略,不是近似算法。

**底层机制/为什么这样设计:**

最笨的想法是:既然每张卡都要用到全部的 K/V 才能算出自己那部分 Q 的 attention,那就让每张卡先把完整的 K/V 全部拷贝一份(all-gather)。但这样一来,"每张卡该存多少 K/V"根本没有减少——本来是"一张卡存下整个序列的 K/V",现在变成"每张卡各自都存下整个序列的 K/V",序列并行的意义就没了。

Ring Attention 的想法是反过来:**任何时刻,任何一张卡都只需要临时持有 `1/n_rank` 的 K/V**。具体做法(对应代码里两层循环):外层循环 `rank` 代表"当前是哪张卡在算",这张卡的 Q 永远固定是自己那一块 `q[..., rank*chunk:(rank+1)*chunk, :]`;内层循环 `r` 模拟"环形传递了几步",每一步只处理编号为 `j = (rank + r) % n_rank` 的那一块 K/V,处理完就(模拟意义上)转手给下一张卡,自己也接收上一张卡传来的下一块——转完 `n_rank` 圈,每张卡都和全部 K/V 打过一次交道,期间没有哪张卡在任意时刻持有超过 `1/n_rank` 的 K/V。

问题是:K/V 分批到达,怎么保证最后算出来的 softmax 和"一次性看到全部 K/V"完全一样?这就是 `m_run`/`l_run`/`O_run` 三个变量在做的事(online softmax,和 FlashAttention 单卡内分块用的是同一套数学):

```python
m_new = torch.maximum(m_run, scores.amax(-1, keepdim=True))   # 目前见过的所有块里,分数的最大值
p = torch.exp(scores - m_new)                                  # 用新 max 归一化这一块的分数
scale_old = torch.exp(m_run - m_new)                            # 旧 max 变成新 max 需要的修正系数
l_run = scale_old * l_run + p.sum(-1, keepdim=True)              # 修正旧的分母,再加上新块的贡献
O_run = scale_old * O_run + p @ v_j                               # 修正旧的分子,再加上新块的贡献
m_run = m_new
```

softmax 要求分母是"所有候选项的 `exp` 之和",但分块处理时,来了新的一块,可能把全局最大值刷新——而 `exp(x - m_old) * exp(m_old - m_new) = exp(x - m_new)` 这个恒等式保证了:只要把"旧的累积值"乘上修正系数 `exp(m_old - m_new)`,就完全等价于"从一开始就用 `m_new` 做归一化"。这不是近似技巧,是精确的代数恒等变形,这也是为什么 `assert diff < 1e-4` 这种断言里的 diff 只是浮点误差(见下面可运行例子的真实数值),而不是"数量级上"的近似误差。

需要额外强调一点,避免误解:这份 `ring_attention_naive.py` 是**单进程/单设备**代码,q/k/v 全程完整地待在同一块内存里,`j = (rank + r) % n_rank` 只是用下标切片模拟"这张卡此刻拿到了哪一块",并没有真的发生跨设备通信。真正的多 GPU 实现需要用 NCCL 之类的库做 `send`/`recv`(仓库里 `ring_attention_lib.py` 就是对 `ring_flash_attention` 这个真实分布式库的一个存在性检测 wrapper,只有 Linux 多卡环境能真正跑起来)。

**AI 研究场景:**
- 训练超长上下文模型(128k~百万 token 级别)绕不开显存墙,Ring Attention(以及 flash-attention 系的变体)是 Llama-3 等模型训练 128k~1M 上下文的标准手段之一。
- 生产环境不会写这种教学版 for 循环模拟,而是用 `ring_flash_attention` 这类库在真实多卡 NCCL 环境跑;但教学版的数值逻辑和真实库完全一致,理解这份代码就理解了底层机制,这也是本系列"讲同一份代码,但补齐为什么"的定位。
- 面试里常考"给定显存和卡数,你怎么估算能训多长的上下文",Ring Attention 把"总长度"和"卡数"解耦成线性关系,而不是平方关系——这是回答这类问题的核心论点。

**可运行例子:**

仓库自带验证,从仓库根目录直接跑该文件(`python learning/long-context/src/ring_attention_naive.py`):
```python
if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    out_vanilla = vanilla_attn(q, k, v)
    out_ring = ring_attention_naive(q, k, v, n_rank=4)
    diff = (out_vanilla - out_ring).abs().max().item()
    print(f"vanilla vs ring naive max diff: {diff:.2e}")
    assert diff < 1e-4
    print("✓ Ring attention 数值等价")
```
实测输出(仓库 `.venv`):
```
vanilla vs ring naive max diff: 2.38e-07
✓ Ring attention 数值等价
```
diff 在 `1e-7` 量级,是 float32 矩阵乘法顺序不同带来的浮点误差,不是算法近似误差。

补充验证 `tests/test_ring_correctness.py::test_ring_different_n_rank` 的论据("切几块不影响结果,证明这是并行策略不是近似算法"),额外现场打印 diff(仓库测试本身只 assert,不打印数值):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("learning/long-context/src").resolve()))

import torch
from ring_attention_naive import ring_attention_naive

torch.manual_seed(0)
q2 = torch.randn(1, 1, 24, 4)
k2 = torch.randn(1, 1, 24, 4)
v2 = torch.randn(1, 1, 24, 4)
out_a = ring_attention_naive(q2, k2, v2, n_rank=2)
out_b = ring_attention_naive(q2, k2, v2, n_rank=4)
diff_rank = (out_a - out_b).abs().max().item()
print(f"n_rank=2 vs n_rank=4 max diff: {diff_rank:.2e}")
assert diff_rank < 1e-4
```
实测输出:`n_rank=2 vs n_rank=4 max diff: 8.94e-08`。另外用 `pytest learning/long-context/src/tests/test_ring_correctness.py -v` 跑了仓库自带的两个用例,实测 `2 passed`。切成 2 块还是 4 块,最终结果只差 `8.94e-08`(同样是浮点误差量级)——这是"Ring Attention 是并行策略,不是近似算法"这句话最直接的数值证据。

**面试怎么问 + 追问链:**
- **Q:** "序列长度要到 1M,你打算怎么把 attention 塞进有限显存里训练?"—— 期望能提出序列并行(sequence parallelism)/ Ring Attention 这个方向,而不是只会说"调小 batch size"或"上梯度检查点"这类不对症的答案。
- **追问 1:** "Ring Attention 切完块之后,数值上和不切块的标准 attention 还一样吗,为什么?"—— 期望答"完全一样(只有浮点误差),因为用了 online softmax 递推,数学上是精确的增量式累积,不是近似"。
- **追问 2(区分度很高):** "既然只是换个顺序累积同一个 softmax,为什么一定要设计成'环形'传递 K/V,而不是让每张卡直接找自己缺的那部分数据要一次(all-gather)?"—— 期望答到通信效率:环形传递保证每一轮通信量恒定、只发生在相邻 rank 之间,总通信量是 `O(L·d)`,不会有某张卡瞬时带宽压力过大的问题,这是分布式训练里常见的通信拓扑设计(和 ring all-reduce 思想相通,但目的不同——见下面常见坑)。
- **追问 3(工程向):** "这份 `ring_attention_naive.py` 是真的在做多卡通信吗?"—— 期望能看穿这是单设备模拟(q/k/v 全程在同一块内存里,只是用下标切片模拟"某张卡此刻拿到哪一块"),说出这一点说明读代码读到了细节,不是脑补。

**常见坑:**
- 把"Ring Attention"和"Ring All-Reduce"搞混——两者都用"环形"通信拓扑,但 Ring Attention 传的是 K/V 块用来做 attention 计算,Ring All-Reduce 传的是梯度做汇总平均,只是通信模式相通,目的完全不同。
- 以为 `n_rank` 必须能整除 batch 或 head 数——代码里实际 assert 的是 `t % n_rank == 0`(序列长度可以整除即可),和 batch/head 无关。
- 把 Ring Attention 当成一种"近似加速技巧"(类似 sparse attention 牺牲精度换速度)——它是精确等价的并行策略,不省 FLOPs(总计算量不变,见知识点 4),只是把显存需求分摊到多卡,这是很多初学者的第一反应错误。

---

## 2. Striped Attention(仅概念,无独立实现)—— Ring Attention 的负载均衡改进

**是什么:** 用 `grep -rn` 核实过,仓库里没有单独的 `striped_attention` 之类的源码文件:
```bash
grep -rn "striped" learning/long-context/src/ --include="*.py" -i
# (无输出 —— 零命中)
```
只在 `learning/long-context/lectures/07-striped-attention.md`(16 张 slide)里有概念讲解。它描述的是 Ring Attention 在 **causal(自回归)场景**下的一个调度层优化:不改变"切几块""转几圈"这两件事,只改变"每一圈里,各个 rank 实际按什么顺序处理 K 块":
```
# Ring:     K_order = [K_0, K_1, K_2, K_3]      —— 顺序处理
# Striped:  K_order = [K_0, K_2, K_1, K_3]      —— 按下标奇偶重新分组(偶数下标块排前面,奇数下标块排后面)
```

**一句话:** Striped Attention 通过打乱 K 块被处理的顺序(而不是顺序 0,1,2,3),让"因果 mask 导致某些块整体是 `-inf`、可以整块跳过计算"这件事在各个 rank/round 之间分布得更均匀,减少"有的卡先算完在空等,有的卡还在苦算"的同步浪费。

**底层机制/为什么这样设计:**

先讲清楚"为什么 causal 场景下朴素 Ring 会负载不均衡"这个前提,这是 Striped 存在的全部理由。在 causal attention 里,持有"序列早期"Q 块的那个 rank(比如 rank 0,拿到 `Q_0`),它的 Q 只需要跟同样早期的 K 块做非平凡计算,后面轮到的 `K_1`、`K_2`、`K_3`(相对 `Q_0` 全是"未来")对应的 score 整块是 `-inf`,可以直接跳过、不用真算(lecture 提到"某些 round 全 -inf 可 skip");但持有"序列末尾"Q 块的 rank(比如 rank 3,拿到 `Q_3`),几乎需要跟所有 K 块做真实计算,因为末尾的 token 能看到几乎全部历史,几乎不能跳过任何一轮。朴素 Ring 顺序下,"谁先没活干、谁一直有活干"和 rank 编号强相关,而每一轮通信都要等最慢的那个 rank(同步屏障),于是先没活干的 rank 只能空等——lecture 07 给的实测数字是 8 GPU 场景下某 GPU 闲置能达到 30%。

Striped 的解法是重新安排"每一轮里各 rank 实际消费 K 块的顺序"(K/V 本身的切分方式不变,只是被消费的先后次序变了),让"可以整块跳过的计算"和"必须真算的计算"更均匀地分布到每个 rank 头上,没有谁能提前收工、也没有谁每轮都满载,整体等待时间变短(lecture 数字:8 GPU 闲置率从 30% 降到 10% 以内)。这个重排为什么不会改变最终结果——直接沿用知识点 1 已经证明过的性质:online softmax 递推对 K/V 块被喂入的顺序不敏感(`n_rank=2` 和 `n_rank=4` 结果一致,本质上就是"用几步凑出同一个 softmax"不影响结果这条性质的一个体现),所以 Striped 纯粹是调度层/工程层优化,不引入任何近似,这也是 lecture 里"Striped 在数学上与 Ring 等价"这句话的依据。

**AI 研究场景:**
- 训练超长上下文的 causal 语言模型(GPT 系列都是 causal)用多卡 Ring Attention 时,GPU 利用率是真金白银的训练成本问题——8 卡训练如果有一张卡闲置 30% 时间,相当于间接浪费了接近 30% 的算力预算,预训练动辄数百万美元的场景下这不是小数目。Striped 这类优化的价值就在这里:不改变精度,单纯把已经买好的 GPU 用得更满。
- 如实标注局限:lecture 明确指出这类优化目前收益是"5-10%",且"主流框架(vLLM / Megatron)多用 Ring + 内部优化",不是把 Striped 作为独立方案实现——这提示一个更普遍的工程规律:很多论文提出的调度层改进,最终会被大框架"吸收"成一个内部优化选项,而不是长期作为独立库存在,这也是为什么本仓库没有为它单独建一个 `striped_attention.py`。

**可运行例子:** 本知识点无独立代码,见知识点 1 的 Ring Attention 代码(`ring_attention_naive.py`)作为对照理解基础——Striped 做的事情,等价于把 `ring_attention_naive` 内层循环里 `j = (rank + r) % n_rank` 这一行的取值顺序,按某种负载均衡策略重新排列(比如奇偶分组),而不是简单地顺序自增。这里不重新实现,只指出改动点在哪一行:数学上结果不变(仍然是同一组 `k_j`,只是被访问的先后顺序变了,而知识点 1 已经用 `n_rank=2` vs `n_rank=4` 数值一致验证过"顺序/分块方式不影响 online softmax 的最终结果")。

**面试怎么问 + 追问链:**
- **Q:** "Ring Attention 在训练 causal(自回归)模型时,是不是每张卡的负载都一样?"—— 期望答"不一定,因果 mask 会让持有'序列末尾'的 rank 需要做几乎全部真实计算,持有'序列开头'的 rank 很早就没活干,朴素 Ring 顺序下负载不均衡"。这题能筛出"只知道 Ring Attention 大致原理"和"理解 causal + Ring 结合后实际工程问题"的候选人。
- **追问 1:** "怎么解决这个负载不均衡问题?"—— 期望提到"重新排列 K 块被处理的顺序(Striped Attention),让能跳过的计算和不能跳过的计算在各卡间分布更均匀",哪怕答不出具体排列方式,能说出"打乱顺序做负载均衡"这个方向就算合格。
- **追问 2(容易被问倒):** "这种重排顺序的优化,会不会改变最终 attention 的计算结果?"—— 期望能推理出"不会,online softmax 对 K/V 块被喂入的顺序不敏感(知识点 1 里 `n_rank` 改变都不影响结果就是证据),Striped 只是换个顺序访问同一组 K/V,是纯调度层优化,不是近似算法"。

**常见坑:**
- 以为 Striped Attention 是一种"新的 attention 计算公式"(类似 sparse attention 那样改变了算什么)——实际上它和 Ring Attention 算的是完全相同的数学函数,只是"用哪个顺序把 K 块喂给 online softmax 递推"这一层调度不同。
- 以为负载均衡问题在单卡场景也存在——单卡的 FlashAttention 是在一个 SRAM 里做分块,没有"多卡同步屏障互相等待"这个问题,负载不均衡本质上是"多卡 + 因果 mask + 同步通信"三者叠加才会出现的问题,不能脱离这个场景泛化理解。
- 把"仓库没有独立实现"等同于"这个技术不重要/过时"——本节 AI 研究场景已经提到,一个技术没有被单独抽出来实现,很多时候只是因为它已经被更大的框架内部吸收,不代表没有价值。

---

## 3. Infini-Attention(`infini_attention.py::InfiniBlock`)—— 局部精确 attention + 压缩记忆的混合方案

**是什么:**
```python
class InfiniBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int = 4):
        ...

    def forward(self, x: torch.Tensor, M_prev: torch.Tensor | None = None,
                Z_prev: torch.Tensor | None = None):
        """返回 (out, M_new, Z_new):out 是这一段的 attention 输出,
        M_new/Z_new 是更新后的压缩记忆,留给下一段调用时当 M_prev/Z_prev 传进来."""
        ...
```
路径:`learning/long-context/src/infini_attention.py`。

**一句话:** Infini-Attention 在标准局部因果 attention 之外,额外维护一份 **shape 固定** 的"压缩记忆"(一个矩阵 `M` 和一个归一化向量 `Z`)——不管喂进去多长的历史,记忆的 shape 永远不变,用可学习的门控把"当前 chunk 的精确 local attention 结果"和"从压缩记忆里检索出来的历史信息"混合起来,换来理论上无限长的上下文,代价是压缩记忆不是精确存储,存在信息损失。

**底层机制/为什么这样设计:**

最笨的想法是"无限上下文"就是把每个历史 token 的 K/V 都存下来(标准 KV cache),但这样显存随 token 数线性增长,到百万级依然会爆。Infini-Attention 的想法是不存"每个 token 自己的"K/V,而是把它们提前"压缩累加"进一个固定大小的矩阵,用的时候直接查这个压缩后的矩阵,不用回头看任何一个原始 token。具体拆成 4 个机制:

1. **局部因果 softmax attention** —— 和标准 transformer block 完全一样,只在当前 chunk 内部做,用一个上三角(不含对角线)`-inf` mask 保证因果性(`torch.full(...).triu_(1)`)。这部分保证"近距离信息"精确无损。
2. **线性 attention 风格的记忆检索** —— 不用 softmax(因为 softmax 需要看到所有候选项统一归一化,没法做增量更新),改用对 `q`、`k` 各自独立(elementwise)算的 `sigmoid`。这样检索可以写成 `sigmoid(q) @ M / (sigmoid(q) @ Z)` 这种满足结合律的矩阵乘法形式,不需要重新回看所有历史 K/V,只需要一份聚合过的 `M`、`Z`。这是 linear attention 技巧的直接应用:标准 softmax 的分母依赖所有 token 联合归一化,没法拆成"先处理 A 部分、再处理 B 部分、最后合并"的增量形式;换成两个独立的非负映射(这里是 `sigmoid`,不是 `exp`)之后,`K^T V` 这一步可以提前算好、缓存起来,新来的 `Q` 直接乘上去查表——序列长度维度被"求和掉"了,聚合结果的 shape 只取决于 `d_h`,和 token 数量无关。
3. **关联记忆更新公式** —— `M_new = M_prev + sigmoid(k)ᵀ @ v`,`Z_new = Z_prev + sigmoid(k).sum()`,是一个不断累加的 running sum:新到的 chunk 的"贡献"用外积形式加进已有的 `M` 里,不保留任何一个历史 token 的 K/V 本身。一旦累加进 `M`,原始的某条 K/V 具体是多少就再也拿不回来了——这正是"有损压缩"的数学根源:不同的 `(K, V)` 组合可能累加出同一个 `M`,信息论意义上不可逆。
4. **每个 head 一个可学习的 sigmoid 门控** —— `gate = sigmoid(self.gate)`,初始化 `nn.Parameter(torch.zeros(n_head))`,所以刚开始训练时 `sigmoid(0) = 0.5`,是 local 和 retrieval 各占一半的中立起点(下面例子里现场验证),训练过程中模型自己学要多依赖精确的局部信息、还是压缩过的长程记忆。

**AI 研究场景:**
- 这是 Google 2024 年提出的方案(lecture 08),核心卖点是用固定大小的 memory state 实现理论上无限长的上下文,不需要像 Ring Attention 那样堆更多卡,训练也更友好("不需要 ring")——这是它和 Ring/Striped 最大的定位差异:一个是"用更多硬件精确地算",一个是"用同样的硬件,牺牲一部分精度换无限长度"。
- 不能只讲优点:lecture 08 明确提到 Google 自己报告的结果是"1M 上下文任务 NIAH 准确率 > 90%,但 RULER(更综合的评测)表现明显更弱","局限:memory state 有信息损失(类 SSM),retrieval 不如 full attention 精确"——这是"压缩记忆不是精确存储"这句话的具体量化证据,不是空泛地说"有损失"。
- 工程上,`forward(x, M_prev, Z_prev)` 返回 `(out, M_new, Z_new)` 这种"分段处理 + 跨调用传递 state"的接口设计,和 RNN/SSM(比如 Mamba)的 hidden state 接口是同一类模式——本质上是在做"attention 和 RNN 的杂交":局部窗口内还是 attention,跨窗口退化成 RNN 式的状态传递,这也是为什么它被归为"显存不随 token 数增长"的方案,而不是 Ring 那种"显存仍随 L 增长、但可以线性分摊到多卡"的方案。

**可运行例子:**

仓库自带的最小验证,从仓库根目录直接跑该文件(`python learning/long-context/src/infini_attention.py`):
```python
if __name__ == "__main__":
    m = InfiniBlock(d_model=32, n_head=4)
    x = torch.randn(1, 8, 32)
    y, M, Z = m(x)
    print(f"Infini out {y.shape}  M {M.shape}  Z {Z.shape}")
```
实测输出(仓库 `.venv`):
```
Infini out torch.Size([1, 8, 32])  M torch.Size([1, 4, 8, 8])  Z torch.Size([1, 4, 8])
```

在此基础上补充验证"压缩记忆 shape 恒定""更新公式和代码精确对应""记忆为空时检索项为 0""局部分支因果性"这几条核心性质(全部在 `.venv` 里现场跑过,assert 全部通过):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("learning/long-context/src").resolve()))

import torch
from infini_attention import InfiniBlock

torch.manual_seed(0)
m = InfiniBlock(d_model=32, n_head=4)
assert m.d_h == 8                                      # 32 // 4

# 门控初始值:nn.Parameter(torch.zeros(n_head)) -> sigmoid(0) = 0.5,local/retrieval 五五开的中立起点
assert torch.allclose(torch.sigmoid(m.gate), torch.full((4,), 0.5))

x1 = torch.randn(1, 8, 32)
y1, M1, Z1 = m(x1)                                      # 第一段:不传 M_prev/Z_prev,内部用全 0 初始化
assert y1.shape == (1, 8, 32)
assert M1.shape == (1, 4, 8, 8)                          # (b, h, d_h, d_h),d_h=8,和 seq_len 无关
assert Z1.shape == (1, 4, 8)                              # (b, h, d_h)

# 核心性质:换一段完全不同长度的输入(20 而不是 8),记忆 shape 依然不变
x2 = torch.randn(1, 20, 32)
y2, M2, Z2 = m(x2, M1, Z1)                                # 接着第一段的记忆继续处理第二段
assert M2.shape == M1.shape                                # 序列变长了,压缩记忆的 shape 岿然不动
assert Z2.shape == Z1.shape                                # 对比:标准 KV cache 会随 token 数线性增长

# 手动复现更新公式,验证代码和公式精确对应(不是"大概长这样")
k2 = m.k(x2).view(1, 20, 4, 8).transpose(1, 2)
v2 = m.v(x2).view(1, 20, 4, 8).transpose(1, 2)
sig_k2 = torch.sigmoid(k2)
Z_manual = Z1 + sig_k2.sum(dim=-2)
M_manual = M1 + torch.einsum("bhtd,bhte->bhde", sig_k2, v2)
assert torch.allclose(Z2, Z_manual, atol=1e-6)
assert torch.allclose(M2, M_manual, atol=1e-6)

# 记忆为空时(M_prev 全 0),检索项精确为 0 —— 第一段没有历史可检索,自然退化成只看 local
sig_q1 = torch.sigmoid(m.q(x1).view(1, 8, 4, 8).transpose(1, 2))
retrieval_zero = torch.einsum("bhtd,bhde->bhte", sig_q1, torch.zeros(1, 4, 8, 8))
assert torch.equal(retrieval_zero, torch.zeros(1, 4, 8, 8))

# 局部 attention 分支的因果性:改动后面 token 不影响前面位置的输出
x1b = x1.clone()
x1b[:, 5:, :] = torch.randn(1, 3, 32)
y1b, _, _ = m(x1b)
assert torch.equal(y1[:, :5, :], y1b[:, :5, :])           # 前 5 个位置完全不受后面 token 影响
assert not torch.equal(y1[:, 5:, :], y1b[:, 5:, :])        # 但后面位置确实变了(排除"根本没用到 x"的假阳性)
```
以上代码块全部 assert 通过,没有一条失败。

**面试怎么问 + 追问链:**
- **Q:** "Infini-Attention 号称'无限上下文',它是怎么做到显存不随序列长度增长的?"—— 期望答"用固定大小的 M/Z 矩阵累积压缩记忆,不保留原始 K/V,序列多长都不影响 M/Z 的 shape"(可以现场引用上面 `M2.shape == M1.shape` 这条断言作为证据)。
- **追问 1:** "这种压缩记忆和标准的 KV cache 比,牺牲了什么?"—— 期望答"精度/可逆性——KV cache 能精确还原每个历史 token 的贡献,压缩记忆是把很多 token 的信息加总进同一个固定大小的矩阵,没法从 `M` 反推出某个具体历史 token 的 K/V 是多少,是有损压缩"。
- **追问 2(区分度很高):** "检索这一步为什么用 sigmoid 而不是 softmax?直接把公式里的 sigmoid 换成 softmax 行不行?"—— 期望答到"softmax 需要看到所有候选项之后统一归一化,这个'先收集所有历史再归一化'的操作没法写成增量累积的形式;sigmoid 对每个元素独立计算,配合结合律可以先把 `K^T V` 算好存起来,来一个新的 `q` 直接'查表',这是 linear attention 能做到 O(1) 显存/token 的根本原因,换成 softmax 这套增量更新的数学就不成立了"。
- **追问 3(工程向):** "如果你训练时发现模型几乎完全不用 retrieval 分支(gate 始终接近 0),可能是什么原因?"—— 没有唯一答案,考察对"门控是训练出来的,不是设计时定死的"这件事的理解,可以提任务本身不需要长程记忆、初始化/学习率导致门控没学动、压缩记忆检索质量太差模型学会了忽略它等等,能提出诊断思路比给出唯一正确答案更重要。

**常见坑:**
- 以为 `M_prev=None` 时是"没有记忆所以要走一条特殊的降级代码路径"——实际代码里 `M_prev is None` 只是触发"用全 0 矩阵初始化"这一个分支,后续计算路径完全一致,检索项自然算出 0(上面例子已验证),不是走了一条特殊分支,理解这一点能避免过度紧张地找"隐藏 if"。
- 把 `Z_new = Z_prev + sigmoid(k).sum(dim=-2)` 里的 `sum` 理解成"取平均/归一化"——它是纯粹的累加(running sum),会无上限增长,不是滑动平均,这也是压缩记忆的"信息饱和"隐患之一:chunk 数很多之后,新信息在 `M` 里占的相对权重会被历史积累稀释。
- 只看到"理论无限上下文"这个卖点,忽略 lecture 08 明确报告的局限(NIAH > 90% 但 RULER 综合表现更弱)——"能处理无限长度"和"处理得好"是两件事,压缩记忆类方案通常在需要精确检索具体某个历史事实的任务上明显弱于精确方法。

---

## 4. 三种长上下文 Attention 架构对比选型

**是什么:** 不是新代码,而是把前 3 个知识点(Ring / Striped / Infini-Attention)放进同一个决策框架里,回答一个具体问题:"给定序列长度、硬件预算、精度要求,该选哪一个?"

**一句话:** 单卡序列长度还没触及显存墙,直接用标准 attention;序列长到必须靠多卡才能装下、且要求数值精确(比如预训练阶段),用 Ring/Striped 这类序列并行;需要"理论无限"长度、能接受压缩带来的精度损失、想省掉多卡通信开销的场景,才考虑 Infini-Attention 这类压缩记忆方案。

**底层机制/为什么这样设计:**

这不是"哪个更好"的排序关系,而是三个设计变量的取舍(trade-off),可以按几个维度拆开看:

- **精确 vs 近似:** 标准 attention / Ring / Striped 在数学上完全等价(知识点 1、2 已用 assert 验证过,diff 在 `1e-7` 量级的浮点误差),是精确算法;Infini-Attention 引入了有损压缩,是近似算法。这是最本质的分野。
- **显存怎么被消化掉:** 标准 attention 用一张卡但显存随 `L²` 增长;Ring/Striped 用多张卡把显存摊薄成线性(`O(L²/n_rank)`);Infini-Attention 不需要更多硬件,靠牺牲精度把显存变成和 `L` 基本无关的常数。
- **FLOPs 有没有真的变少(容易被忽略的一条):** Ring/Striped 只解决"装不装得下"的显存问题,不解决"算不算得动"的计算量问题——总 FLOPs 仍然是 `O(L²)`,只是被 `n_rank` 张卡分摊,序列长度翻倍,不管用多少张卡,总计算量还是翻 4 倍。Infini-Attention 因为把跨 chunk 的历史压缩进固定大小的 `M/Z`,只在 chunk 内部(固定窗口)做平方复杂度的精确 attention,chunk 数随 `L` 线性增长、每个 chunk 代价是常数,总 FLOPs 近似 `O(L)`——这是用精度换来的复杂度类别的改变,不只是常数因子优化,是 Ring/Striped 做不到的。
- **训练 vs 推理场景的适配:** Ring/Striped 对训练更友好(反向传播需要精确梯度);Infini-Attention 在需要长程"印象式"记忆而非精确检索的场景(比如推理时的超长对话记忆)更有性价比。

**AI 研究场景(对比表格):**

| 维度 | 标准 Attention | Ring / Striped Attention | Infini-Attention |
|---|---|---|---|
| 精确性 | 精确 | 精确(与标准 attention 数值等价,实测 diff ≈ 1e-7 量级) | 近似,压缩记忆有损 |
| 总 FLOPs | `O(L²)` | `O(L²)`(不变,只是分摊到多卡) | 近似 `O(L)`(chunk 内 `O(chunk²)` + 记忆更新线性于 L) |
| 单卡显存 | `O(L²)` | `O(L²/n_rank)`,随卡数线性下降 | `O(1)`,和 L 基本无关(压缩记忆 shape 固定) |
| 需要的硬件 | 单卡够用(L 不太长时) | 必须多卡(n_rank 张) | 单卡也可以 |
| 典型场景 | L 没触及显存墙 | 预训练超长上下文(128k~1M),要求梯度精确 | 需要"无限"长度、能接受精度损失的推理/长程记忆场景 |
| 主要局限 | L 变长直接 OOM | 只解决显存不省 FLOPs;causal 场景需要 Striped 之类的负载均衡优化 | NIAH 高但 RULER 综合表现弱(Google 自测数据),无法精确检索具体历史事实 |

**可运行例子:**

三份源码互相没有 import,可以在同一个脚本里分别导入使用(从仓库根目录跑,用 `sys.path.insert` 指向 `src` 目录,现场跑通,assert 全部通过):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("learning/long-context/src").resolve()))

import math
import torch
import torch.nn.functional as F

from ring_attention_naive import vanilla_attn, ring_attention_naive
from infini_attention import InfiniBlock

torch.manual_seed(0)

# 场景 A:标准 attention 和 Ring Attention —— 精确等价,只是"怎么分摊到多卡"的区别
q = torch.randn(1, 2, 16, 8)
k = torch.randn(1, 2, 16, 8)
v = torch.randn(1, 2, 16, 8)
out_vanilla = vanilla_attn(q, k, v)
out_ring = ring_attention_naive(q, k, v, n_rank=4)
assert out_vanilla.shape == out_ring.shape == (1, 2, 16, 8)
assert (out_vanilla - out_ring).abs().max().item() < 1e-4

# 场景 B:Infini-Attention 接口形状完全不同 —— 吃的是没投影过的 x,不是现成的 q/k/v,
# 且记忆 M 的 shape 只取决于 d_model/n_head,和输入序列长度无关(对比场景 A 的 q/k/v 直接带 seq 维)
m = InfiniBlock(d_model=16, n_head=2)
x = torch.randn(1, 16, 16)
out_infini, M, Z = m(x)
assert out_infini.shape == (1, 16, 16)
assert M.shape == (1, 2, 8, 8)

# 场景 C:把 Infini 的门控关到几乎 0(sigmoid(-1e4)≈0),应该精确退化成标准因果 local attention
# —— 证明 Infini 不是凭空发明的黑箱公式,而是"标准 attention 之上叠加了一条可关闭的记忆检索通路"
with torch.no_grad():
    m.gate.fill_(-1e4)
out_gated, _, _ = m(x)

qh = m.q(x).view(1, 16, 2, 8).transpose(1, 2)
kh = m.k(x).view(1, 16, 2, 8).transpose(1, 2)
vh = m.v(x).view(1, 16, 2, 8).transpose(1, 2)
causal_mask = torch.full((16, 16), float("-inf")).triu_(1)
local_manual = F.softmax((qh @ kh.transpose(-2, -1)) / math.sqrt(8) + causal_mask, dim=-1) @ vh
local_manual = local_manual.transpose(1, 2).reshape(1, 16, 16)
out_manual = m.o(local_manual)

diff_gate = (out_gated - out_manual).abs().max().item()
assert diff_gate < 1e-6
```
实测:场景 A 的 diff 和知识点 1 同量级(< 1e-4,浮点误差);场景 B 的 shape 断言全部通过;场景 C 的 `diff_gate` 实测精确为 `0.0`(float32 下 `sigmoid(-1e4)` 精确下溢成 0,`(1-gate)` 精确等于 1,所以退化是精确的,不是近似的)。这组例子直接把"三种架构接口/行为的真实差异"跑给自己看,而不是停留在文字描述。

**面试怎么问 + 追问链:**
- **Q:** "如果要训一个支持 1M 上下文的大模型,预训练阶段和推理阶段,你会分别推荐用什么方案?"—— 期望能区分场景:预训练要求梯度精确、通常硬件预算也更充足,倾向 Ring/Striped 这类精确并行;推理阶段如果是单卡部署、且能接受一定精度损失换取超长记忆(比如长对话历史的"印象式"记忆),Infini-Attention 这类压缩方案更有性价比。
- **追问 1:** "Ring Attention 用了 8 张卡,相比单卡跑标准 attention,总的计算量(FLOPs)变小了吗?"—— 期望答"没有变小,总 FLOPs 还是 `O(L²)`,只是分摊到 8 张卡并行算,显存和时间墙被打开了,但硬件成本(要 8 张卡)是实打实的"。这题专门筛选"以为多卡 = 更快 = 更省"这种模糊理解的候选人。
- **追问 2(区分度很高):** "Infini-Attention 为什么能把总 FLOPs 做到接近线性,而不是像 Ring Attention 一样还是平方复杂度?"—— 期望答"因为它把跨 chunk 的历史压缩进固定大小的 M/Z,不再对全部历史 token 做两两打分,只在 chunk 内部(固定大小窗口)做平方复杂度的精确 attention,chunk 数随 L 线性增长、每个 chunk 代价是常数,总代价就是线性的;这是用精度换来的复杂度类别的改变,不只是常数因子的优化"。
- **追问 3(开放题):** "这三种方案能不能叠加使用,比如 Ring Attention + Infini-Attention 一起用?"—— 没有标准答案,考察是否理解这是两个独立的设计轴("怎么把已经决定要做的精确计算分摊到多卡" vs "要不要一开始就用压缩记忆减少要做的计算量"),原则上可以叠加(每张卡内部的 local attention 窗口继续用 FlashAttention 的分块技巧,压缩记忆状态本身也可以在卡间同步)。这类"能否组合"的追问考察的是知识的迁移能力,不是背答案。

**常见坑:**
- 把"支持长上下文"简化成一句话推荐,不考虑训练/推理场景差异——这是本节反复强调的重点,三种方案不是排位赛,是针对不同约束条件的不同答案。
- 误以为 Ring/Striped Attention 能像 Infini-Attention 一样把复杂度降到线性——它们只解决"装不装得下"的显存问题,不解决"算得动算不动"的 FLOPs 问题(见上面"底层机制"第 3 条)。
- 反过来误以为 Infini-Attention"更先进所以应该取代 Ring/Striped"——精度要求高的场景(尤其预训练)现在主流仍然是精确的 Ring + FlashAttention 组合(lecture 06/07 反复提到的 vLLM/Megatron 现状),压缩记忆类方案目前更多是特定场景(如推理时的超长记忆)的补充方案,不是全面替代关系。
