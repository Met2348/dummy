"""推理部署与服务化深水区追问链（12 个 DeepPoint）。

覆盖 PagedAttention 具体怎么算显存碎片率、投机采样的接受率估计与 draft model "
选择权衡、量化推理（INT8/AWQ/GPTQ）掉精度后怎么debug定位到具体层、continuous "
batching 对延迟和吞吐的权衡、KV cache 在真实服务里的具体显存占用计算。

边界：不涉及训练阶段的显存/通信（那是 distributed_training 类目）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, drill, grade_chain  # noqa: E402

CAT = "推理部署与服务化深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-ai-inf-01", cat=CAT,
        trigger="你提到用了 PagedAttention 来管理 KV cache。",
        chain=(
            ("PagedAttention 具体怎么解决显存碎片问题？",
             "传统实现给每个请求预先分配一段连续显存来存 KV cache（按最大可能生成长度预留），但实际生成"
             "长度往往远小于预留上限，造成大量内部碎片（internal fragmentation）；不同请求的连续段之间"
             "长度不一还会造成外部碎片（external fragmentation），无法把空闲区拼成足够大的连续块给新请求；"
             "PagedAttention 借鉴操作系统虚拟内存的分页思想，把 KV cache 切成固定大小的 block，每个请求"
             "按需逐块申请，用一张 block table 记录逻辑序列位置到物理 block 的映射，物理 block 可以不连续"
             "存放，从而把外部碎片基本消除，只留下每个请求最后一个未填满 block 内的内部碎片。",
             ("内部碎片", "外部碎片", "block table")),

            ("这个显存碎片率具体怎么算出来？block size 怎么选会有什么权衡？",
             "碎片率可以粗略估计为：所有请求在最后一个未填满 block 里浪费的 slot 数之和，除以总共分配的 "
             "block 数乘以 block size；如果 block size 设为 B，平均每个请求最后一块的浪费期望约为 B/2 "
             "个 slot，所以碎片率约等于 (B/2) / 平均序列长度，block size 越大，理论最大浪费越多，但 "
             "block size 太小又会增加 block table 的元数据开销和每次注意力计算时需要处理的 block 数量"
             "（更多次非连续访存和 kernel launch 开销），实践中（如 vLLM）常取 16 左右做折中。",
             ("B/2", "元数据开销", "折中")),

            ("PagedAttention 把显存浪费降到接近于零了吗？它还有哪些残余的碎片/开销没解决？",
             "没有降到零：每个请求最后一个 block 依然有内部碎片（期望约半个 block 大小），这是分页机制"
             "结构性决定的，除非把 block size 缩到 1（这样会让访存和调度开销急剧上升，得不偿失）；此外"
             "block table 本身、以及支持 prefix sharing/copy-on-write 所需的引用计数元数据也占用额外"
             "显存和计算开销，这部分开销随并发请求数增长；所以更准确的说法是 PagedAttention 把外部碎片"
             "基本消除、把内部碎片从'整段预留的巨大浪费'降到'每请求最多半个block'的量级，而不是完全消除"
             "浪费。",
             ("内部碎片", "结构性决定", "引用计数元数据")),

            ("如果你要进一步压榨这最后一点碎片，理论上还能怎么做？现实中为什么这些方案很少被采用？",
             "理论上可以让不同请求共享同一个未填满的 block（把多个请求的尾部拼到一个 block 里），或者"
             "动态调整 block size（对长请求用大block减少元数据、对短请求用小block减少浪费）；现实中这类"
             "方案很少被采用，因为共享未填满block会让不同请求的KV数据混在同一物理块里，大幅增加"
             "attention kernel实现的复杂度（需要处理块内的请求边界和mask），动态block size又会破坏"
             "kernel对固定block size做的向量化/编译期优化，工程复杂度的增加相对于'省下半个block'这点"
             "收益往往不划算，这是一个已知但尚未有普遍认可的更优解的开放优化空间。",
             ("共享未填满block", "kernel实现的复杂度", "开放优化空间")),
        ),
        pitfall="第2层大多数人只会说'block size取16'，说不出这个碎片率和block size之间B/2的量级关系；"
                "第4层几乎没人能提出具体的进一步优化思路，容易直接说'已经很好了'搪塞过去。",
        real_world_link="learning/inference-engine-core/src/paged_kv.py",
    ),

    DeepPoint(
        id="dp-ai-inf-02", cat=CAT,
        trigger="PagedAttention 这种非连续存储的 KV cache，attention kernel 怎么处理，会不会比标准连续显存慢？",
        chain=(
            ("PagedAttention 的 attention kernel 具体要多做什么才能处理非连续的 KV block？",
             "标准 FlashAttention 假设 K/V 是一段连续显存，可以直接按固定 stride 分块读取；PagedAttention "
             "的 kernel 需要先查 block table 得到当前逻辑位置对应的物理 block 地址，再从物理地址读取，"
             "相当于在原本的分块读取循环外面多加了一层地址间接寻址（indirection），对每个 block 都要做"
             "一次 block table 查表再发起显存访问，而不是简单的地址+偏移量计算。",
             ("block table", "间接寻址", "查表")),

            ("这层间接寻址在实测中会带来多大的性能损耗？主要靠什么工程手段抵消？",
             "间接寻址本身的查表开销相对于整个 attention 计算的 FLOPs 很小，但非连续访存会破坏显存合并"
             "访问（coalesced access）的模式，理论上会降低有效显存带宽利用率；工程上通过专门为 "
             "PagedAttention 写定制 CUDA/Triton kernel（如 vLLM 的 paged attention kernel）来缓解——把 "
             "block table 的查表提前在 kernel 里批量做、按 block 粒度组织线程块，让同一线程块内的访存"
             "尽量保持局部连续，实测下来 PagedAttention 相比理论上限的连续显存实现只有个位数百分比的"
             "额外开销，远小于它换来的显存利用率提升。",
             ("显存合并访问", "定制 CUDA/Triton kernel", "个位数百分比")),

            ("既然还是有开销，为什么几乎所有主流推理引擎最后都收敛到用 PagedAttention（或类似机制）而不是连续显存？",
             "因为连续显存方案的显存浪费（尤其是要为最大生成长度预留）在真实流量下造成的有效吞吐损失，"
             "远大于分页机制这几个百分点的 kernel 开销——显存装不下更多并发请求直接限制了 continuous "
             "batching 能同时跑的请求数，是影响吞吐的主要矛盾；相比之下 kernel 层面几个百分点的效率"
             "损失可以通过后续持续的 kernel 优化（更好的 tiling、FlashAttention 式的分块）逐步抹平，这"
             "是一个'显存利用率是一等公民，kernel效率是可以后续优化的二等约束'的工程判断。",
             ("有效吞吐损失", "主要矛盾", "一等公民")),
        ),
        pitfall="很多人只停留在'PagedAttention会有点慢因为不连续'的直觉，说不出具体是显存合并访问被破坏；"
                "第3层容易答不出'为什么大家还是都用它'背后的吞吐优先级判断。",
        real_world_link="learning/inference-engine-core/src/paged_attention_triton.py",
    ),

    DeepPoint(
        id="dp-ai-inf-03", cat=CAT,
        trigger="你提到用了投机采样（speculative decoding）加速推理。",
        chain=(
            ("投机采样的接受率（acceptance rate）在数学上是怎么定义、怎么估计的？",
             "投机采样用 draft model 先生成若干候选 token，target model 一次前向并行验证这些候选；对每"
             "个候选 token，接受概率是 min(1, p_target(token)/p_draft(token))（rejection sampling"
             "准则），保证最终采样分布严格等于只用 target model 采样的分布；接受率 alpha 通常在一个held"
             "-out验证集上离线估计：让 draft 和 target 对同一段前缀分别打分，逐 token 统计满足接受条件"
             "的比例，得到一个经验平均接受率。",
             ("min(1, p_target(token)/p_draft(token))", "rejection sampling", "经验平均接受率")),

            ("知道了 alpha，怎么估算投机采样带来的期望加速比？draft model 越小越好吗？",
             "在每轮猜 k 个候选 token 的情形下，期望每轮被接受的 token 数约为 (1-alpha^(k+1))/(1-alpha)，"
             "而每轮的时间成本约为 k 次 draft 前向加 1 次 target 前向验证；把'期望产出token数'除以'期望"
             "耗时'得到有效吞吐，draft model 越小推理越快、但通常和 target 的分布差异越大导致 alpha 越"
             "低，两者需要联合权衡——draft 太大会让'k次draft前向'本身变成新瓶颈，draft 太小会让 alpha 掉"
             "得太低、大部分候选被拒绝，白白付出验证成本却没换来加速，最优点是让 draft 前向成本和它带来"
             "的期望接受 token 数收益打平。",
             ("(1-alpha^(k+1))/(1-alpha)", "k次draft前向", "联合权衡")),

            ("draft model 的选择上，'独立小模型' vs 'EAGLE 这类自蒸馏特征级 draft' 各自的权衡是什么？",
             "独立小模型（比如用同系列的更小版本）部署简单、可以复用现成模型，但它是在token embedding"
             "层面独立训练的，和 target model 的隐藏状态分布没有直接对齐，天花板受限于两个独立模型的"
             "分布差异；EAGLE 这类方法让 draft 直接消费 target model 的顶层特征（feature-level）做自"
             "回归外推，相当于是在 target 自己的表示空间里做低成本外推而不是另起一个模型重新建模分布，"
             "实测接受率显著更高，代价是训练更复杂、draft 和 target 强耦合（target 换版本 draft 通常要"
             "重新训练），部署上不如独立小模型灵活可插拔。",
             ("token embedding层面独立训练", "feature-level", "强耦合")),

            ("如果线上流量的分布和你估计 alpha 时用的验证集不一样（比如从聊天场景切到代码生成），接受率会怎样，你有多大把握提前预判？",
             "接受率大概率会下降，因为 alpha 本质上衡量的是 draft 和 target 在具体输入分布下的一致程度，"
             "领域切换（比如代码 token 的局部可预测性和自然语言不同）会改变这个一致程度，且方向不一定是"
             "线性可预测的——代码这类高度结构化、重复模式多的文本有时反而更容易被小 draft 命中（接受率"
             "可能更高），但涉及生僻API或者复杂逻辑推理时命中率可能骤降；诚实地说，没有离线测过目标场景"
             "就没有把握，这也是为什么生产系统通常会做在线接受率监控、按场景分桶评估，而不是只信一个"
             "全局离线跑出来的 alpha 数字。",
             ("一致程度", "不一定是线性可预测的", "在线接受率监控")),
        ),
        pitfall="第2层很多人只会说'加速几倍'但给不出接受token数期望的公式结构；第4层几乎没人诚实回答"
                "'不确定'，容易编一个自信但没有依据的预判方向。",
        real_world_link="learning/speculative-decoding/src/classic_spec_decode.py",
    ),

    DeepPoint(
        id="dp-ai-inf-04", cat=CAT,
        trigger="EAGLE / Medusa 这类基于树结构的投机采样，和经典的单链 speculative decoding 有什么本质区别？",
        chain=(
            ("树结构验证（tree attention）具体是怎么在一次前向里验证多条候选路径的？",
             "经典投机采样每轮只生成一条长度为 k 的候选链，target 一次前向验证这一条链；树结构方法（如 "
             "Medusa 的多头并行猜测、EAGLE 的多分支展开）同时展开多条候选路径构成一棵树，用一个特制的 "
             "attention mask（树形因果 mask，tree attention）让 target model 一次前向就能对树上所有节点"
             "同时算出对应位置的概率，之后按照树的每条根到叶路径分别做逐 token 的 rejection sampling 验"
             "证，选出被接受的最长路径。",
             ("tree attention", "树形因果 mask", "逐 token 的 rejection sampling")),

            ("树越大（候选路径越多）是不是验证收益就越大？这里的权衡是什么？",
             "不是线性收益：树越大，被接受的期望路径长度确实会提高（因为覆盖了更多可能性），但同一次"
             "前向需要处理的 token 数（树的节点总数）也线性增加，验证阶段的计算量和显存（尤其是树形 "
             "attention mask 的存储和计算）随树规模增长，超过一定规模后单次前向本身变慢，抵消了多候选带"
             "来的接受率收益；此外候选路径之间如果高度相似（大量早期分叉后又合流的情况），会造成计算"
             "冗余但收益很小，所以树的宽度和深度通常要针对具体 target model 和硬件做实测调优，而不是越"
             "大越好。",
             ("期望路径长度", "单次前向本身变慢", "实测调优")),

            ("树形验证在实现上对 KV cache 管理提出了什么额外要求？和 PagedAttention 怎么配合？",
             "标准自回归 KV cache 假设是一条线性序列，树形候选意味着同一个前缀可能分叉出多条不同的后续"
             "token，KV cache 需要支持'共享公共前缀、分叉部分各自独立存储'的结构，这正好是 PagedAttention "
             "分页管理天然适合处理的场景——公共前缀对应的 block 可以被多条分支共享（引用计数），分叉之后"
             "各自申请新的 block；如果 KV cache 管理不支持这种分支共享，树形投机采样要么退化成对每条路径"
             "重复存一份 KV（显存浪费严重），要么根本无法实现，这也是为什么树形投机采样通常和 "
             "PagedAttention 类的分页 KV 管理绑定实现。",
             ("分叉部分各自独立存储", "引用计数", "绑定实现")),
        ),
        pitfall="第2层很多人默认'树越大越好'，说不出验证阶段计算量也在增长这个反向压力；第3层很少有人"
                "主动联系到KV cache需要支持分支共享，容易只谈attention mask不谈存储结构。",
        real_world_link="learning/speculative-decoding/src/tree_attention.py",
    ),

    DeepPoint(
        id="dp-ai-inf-05", cat=CAT,
        trigger="你说做了 INT8/AWQ/GPTQ 量化后模型掉了不少精度，你是怎么定位到具体哪层出问题的？",
        chain=(
            ("你会用什么方法定位是模型的哪一层对量化最敏感？",
             "标准做法是层级敏感度分析（layer-wise sensitivity analysis）：保持其他所有层原精度（fp16/"
             "bf16），每次只把其中一层换成量化版本，在一个固定的校准/评测集上测整体 perplexity 或下游"
             "任务指标的变化量，对每一层重复这个过程，把 perplexity 涨幅最大的层标记为量化敏感层；这个"
             "过程可以自动化脚本跑一遍所有层，产出一张'层号 vs 精度损失'的曲线，直接定位到具体第几层的"
             "哪个子模块（attention还是MLP）。",
             ("层级敏感度分析", "perplexity", "对每一层重复这个过程")),

            ("实践中最常见的量化敏感层是哪些？为什么恰好是这些层？",
             "最常见的是 MLP 里的 down_proj（降维投影）和 attention 的输出投影（o_proj），根源是这些层"
             "的输入激活里存在少数幅值远大于其他维度的离群通道（outlier channel），这些离群值决定了量化"
             "的动态范围，如果直接对整个张量做统一的per-tensor量化，大多数正常范围的数值会被极少数离群"
             "值挤压到只剩很粗的分辨率；相比之下 QKV 投影和第一层MLP升维层的激活分布通常更均匀，量化"
             "损失较小；这也是为什么 embedding 层和 lm_head 输出层通常被建议保留更高精度或单独处理——"
             "它们直接决定输出token分布，误差会被softmax放大。",
             ("离群通道", "per-tensor量化", "embedding 层和 lm_head")),

            ("AWQ 具体是怎么针对这个离群通道问题设计的？它比朴素 per-tensor 量化好在哪？",
             "AWQ（Activation-aware Weight Quantization）的核心洞察是：与其在量化时被动忍受激活里的离群"
             "通道，不如在量化前主动做一次逐通道的等价缩放（scale）——把激活里数值大的通道对应的权重"
             "列除以一个缩放系数、同时把该通道的激活乘以同样系数抵消，这样量化目标从'权重列尺度悬殊'的"
             "张量变成尺度更均匀的张量，量化误差显著降低；缩放系数的选择基于激活的统计幅度（用少量"
             "校准数据估计每个通道的重要性），是一种'把量化难度从激活转移到影响更小的权重上'的重参数化"
             "技巧，而不是简单地降低量化精度或跳过某些层。",
             ("等价缩放", "尺度更均匀", "重参数化技巧")),

            ("即便用了 AWQ 这类方法，你有多大把握量化后的精度损失是可以完全预测/避免的？还有没有已知搞不定的情况？",
             "没有完全的把握——AWQ 和 GPTQ 这类方法本质上都是基于校准数据的统计特性做局部最优的重参数化"
             "或误差补偿，如果线上真实分布和校准数据分布差异较大（比如校准用了通用语料但线上主要是罕见"
             "领域文本），离群通道的位置和幅度可能和校准阶段观察到的不一致，量化效果会打折扣；另外某些"
             "对数值精度本身就敏感的能力（比如需要精确计数/复杂多步算术推理的任务）即使把敏感层都保留"
             "高精度，量化后的累积误差仍可能在长链推理里被放大，这类问题目前没有一个通用的'保证不掉点'"
             "的方法，只能针对具体任务做量化后评测，发现问题就针对性地把该层退回高精度（混合精度量化）。",
             ("校准数据分布差异", "累积误差", "混合精度量化")),
        ),
        pitfall="第2层很多人只会说'量化整体掉点'说不出具体哪类层、哪种outlier机制最脆弱；第4层几乎没人"
                "诚实说'没有完全把握'，容易过度自信地说AWQ/GPTQ肯定能解决所有量化精度问题。",
        real_world_link="learning/quantization-deploy/src/awq_minimal.py",
    ),

    DeepPoint(
        id="dp-ai-inf-06", cat=CAT,
        trigger="AWQ、GPTQ、SmoothQuant 都是量化方法，你觉得它们本质上的区别是什么？",
        chain=(
            ("这三种方法各自的核心思路分别是什么？",
             "SmoothQuant 的思路是把激活里的量化难度通过逐通道缩放转移一部分到权重上，让激活和权重都"
             "变得更容易量化（本质是一种平滑激活分布的预处理）；AWQ 沿用类似的逐通道缩放思想，但更聚焦"
             "于保护对输出影响最大的'显著权重通道'，通过校准数据识别哪些通道重要再针对性缩放；GPTQ 则"
             "是完全不同的路线——它逐层、逐权重列地做二阶误差补偿（基于 Optimal Brain Quantization 的"
             "思想），量化一列权重后，用海森矩阵（Hessian）信息把这一列引入的量化误差分摊补偿到还未量化"
             "的其余权重上，是一种迭代式的贪心误差最小化过程，而不是像 AWQ/SmoothQuant 那样做前置的分布"
             "变换。",
             ("逐通道缩放", "显著权重通道", "海森矩阵")),

            ("这两条路线（分布变换 vs 逐层误差补偿）对校准数据的依赖程度一样吗？各自的风险点是什么？",
             "都依赖校准数据，但依赖的信息不同：AWQ/SmoothQuant 依赖校准数据来估计激活的通道级统计幅度"
             "（哪些通道数值大），如果校准数据的领域分布和真实部署场景差异大，估计出的缩放系数可能保护"
             "错了通道；GPTQ 依赖校准数据来估计权重的海森矩阵（近似为激活的二阶统计量），对校准数据的"
             "数量和多样性更敏感，校准集太小或者分布单一时，海森矩阵估计不准，逐层贪心补偿的误差可能在"
             "深层累积放大（因为前面层的量化误差会通过激活影响后面层的海森矩阵估计），这是 GPTQ 相比 "
             "AWQ 在极低比特（如2-3bit）时更容易出现累积误差爆炸的原因之一。",
             ("通道级统计幅度", "海森矩阵估计", "累积误差爆炸")),

            ("如果要在同一个模型上二选一，你会用什么标准来决定选 AWQ 还是 GPTQ？",
             "一个实用的判断标准是权衡'量化后推理速度实现的成熟度'和'比特数目标'：AWQ 的缩放操作在推理"
             "时可以完全提前融合进权重（fused），几乎不引入额外推理开销，且社区内针对 4-bit AWQ 的"
             "kernel优化已经很成熟，适合追求部署简单和稳定吞吐的场景；GPTQ 在极低比特（如3-bit甚至"
             "更低）时因为逐层误差补偿通常能拿到更好的困惑度指标，适合对精度要求更极致、愿意为此多花"
             "校准和调试成本的场景；但这只是经验性的选型参考，具体到某个模型架构和目标任务，最终结论"
             "往往需要都跑一遍量化后评测集才能确定，没有理论上绝对更优的一方。",
             ("提前融合进权重", "极低比特", "都跑一遍量化后评测集")),
        ),
        pitfall="很多人会笼统地说'AWQ和GPTQ都是权重量化方法'，说不出前置分布变换和逐层误差补偿这个本质"
                "路线区别；第3层容易给出'GPTQ永远比AWQ精度高'这类绝对化结论，而不是诚实说需要实测对比。",
        real_world_link="learning/quantization-deploy/src/smooth_quant.py",
    ),

    DeepPoint(
        id="dp-ai-inf-07", cat=CAT,
        trigger="你提到服务端用了 continuous batching。",
        chain=(
            ("continuous batching 和传统的静态 batching 具体有什么区别？",
             "静态 batching（static batching）要求一批请求同时到达、一起跑完所有生成步骤才能返回，批内"
             "任何一个请求生成得比其他请求长，其余已经生成完的请求也必须等它，造成严重的计算浪费和延迟"
             "膨胀；continuous batching（也叫 iteration-level scheduling）以单个解码步（而不是整个"
             "请求）为调度粒度，每一步结束后立刻检查有没有请求已完成可以退出、有没有新请求可以插入补上"
             "空出来的槽位，批的组成在迭代过程中动态变化，不需要等一批里最长的请求结束才能处理下一批。",
             ("static batching", "iteration-level scheduling", "动态变化")),

            ("既然是动态调度，最大 batch size 或 token 预算怎么设，这对延迟和吞吐分别有什么影响？",
             "batch 越大，GPU 利用率和整体吞吐（tokens/s）通常越高，因为矩阵乘的批量维度更大、更接近"
             "compute-bound；但对单个请求而言，batch 越大意味着它要和更多请求共享每一步的计算资源，"
             "每步延迟（inter-token latency / TPOT）会上升，首字延迟（TTFT）也可能因为要等更多并发"
             "prefill 请求处理完而变长；所以调度器通常设置一个 token 预算或最大并发数上限，在吞吐和"
             "延迟 SLO 之间找一个工作点，而不是无限制地塞请求进batch。",
             ("TPOT", "TTFT", "延迟 SLO")),

            ("如果一个很长的 prompt 和很多短请求同时到达，continuous batching 会不会产生新的公平性问题？chunked prefill 是怎么缓解的？",
             "会：长 prompt 的 prefill 阶段本身是一次计算量很大的前向，如果调度器把它整个一次性塞进某一"
             "步，会让同批里其他短请求的这一步解码延迟显著变长（相当于长 prefill 挡在了队头，即 "
             "head-of-line blocking）；chunked prefill 把长 prompt 的 prefill 拆成多个小块，分散插入到"
             "多个连续的解码步里和其他请求的解码 token 混合处理，避免单步计算量被一个大 prefill 请求"
             "独占，代价是长 prompt 本身的首字延迟（TTFT）会因为拆成多步而略微增加，但换来了整体调度的"
             "公平性和更平滑的 tail latency。",
             ("head-of-line blocking", "chunked prefill", "tail latency")),

            ("如果 batch 无限增长（调度器完全不设上限，来多少请求塞多少），系统最终会发生什么？为什么这不是'吞吐越高越好'的简单单调关系？",
             "batch 持续增长到显存放不下 KV cache 时会直接 OOM，即便显存足够，计算也会从访存密集"
             "（memory-bound，小 batch 时矩阵乘规模小，主要瓶颈是读取权重/KV的显存带宽）逐渐过渡到计算"
             "密集（compute-bound），越过这个临界点后继续增大 batch，单步计算耗时开始线性增长，而不再"
             "像之前那样'因为原本没占满GPU所以几乎不增加延迟'，此时继续加大 batch 只是让延迟直接摊到了"
             "吞吐上，边际吞吐收益迅速递减，所以吞吐和 batch size 的关系是先陡峭上升后趋于饱和甚至因为"
             "调度开销增加而略微下降的曲线，不是简单的单调递增。",
             ("memory-bound", "compute-bound", "边际吞吐收益迅速递减")),
        ),
        pitfall="第2层很多人只说'batch越大吞吐越高'不谈延迟代价；第4层经常被问到饱和点在哪答不上来，"
                "容易含糊说'batch越大越好'而不了解compute-bound转折点的存在。",
        real_world_link="learning/inference-engine-core/src/continuous_batching.py",
    ),

    DeepPoint(
        id="dp-ai-inf-08", cat=CAT,
        trigger="服务化的时候你怎么计算 KV cache 实际占用多少显存？",
        chain=(
            ("KV cache 显存占用的计算公式是什么？每一项怎么来的？",
             "标准公式是：2（K 和 V 各一份）* num_layers * num_kv_heads * head_dim * seq_len * batch * "
             "每元素字节数；其中 2 对应同时要存 K 和 V，num_kv_heads 而不是 num_query_heads 是因为 KV "
             "cache 存的是 key/value 投影后的张量，如果用了 GQA/MQA，num_kv_heads 远小于 query 头数，"
             "这也是这些方法能大幅压缩 KV cache 的直接原因；seq_len 是当前累计的上下文长度而不是最大长度"
             "（只要显存分配是按需增长的分页机制）。",
             ("num_kv_heads", "GQA/MQA", "seq_len")),

            ("GQA 相比 MHA 具体能省多少 KV cache？这个比例和它带来的质量损失怎么权衡？",
             "如果 MHA 有 H 个 query 头也就有 H 个 KV 头，GQA 把 KV 头分成 G 组（G<H），KV cache 占用"
             "直接从正比于 H 降到正比于 G，比如 32 个 query 头分 4 组，KV cache 缩小到原来的 1/8；这个"
             "压缩比是确定性的、和模型质量无关的纯几何关系，但 G 选得越小，多个 query 头共享同一份 K/V "
             "损失的表达多样性越大，质量损失风险越高，实践中这个 trade-off 在预训练阶段就已经通过消融"
             "定好了（比如 Llama系模型选定的组数），推理服务化阶段无法再调整这个比例，只能选择用哪个"
             "checkpoint。",
             ("正比于 H 降到正比于 G", "1/8", "预训练阶段就已经通过消融定好")),

            ("在真实服务里，为什么按这个公式算出来的理论值和实际能支撑的并发数经常对不上？",
             "几个偏差来源：1）PagedAttention 按 block 分配，每个请求的最后一个 block 有内部碎片，理论"
             "公式没有考虑这部分浪费；2）真实请求的 seq_len 是变化的、不是所有请求都长到 max_len，如果"
             "按 max_len 静态预留（没用分页机制）会大幅高估可用并发数的下界、实际反而会因为提前预留浪费"
             "而更早耗尽显存；3）多模态或使用了 sliding window attention/attention sink 的模型，有效"
             "参与 attention 计算的 KV 长度和'已生成的总 token 数'不是线性对应关系，需要按具体注意力"
             "模式重新推导公式；4）显存里还有模型权重、CUDA context、临时激活缓冲区和 PagedAttention 的"
             "block table 元数据，这些不在 KV cache 公式里但会挤占同一块显存预算。",
             ("内部碎片", "sliding window attention", "block table 元数据")),

            ("给定一块 GPU 的显存预算，你怎么决定最大并发请求数？这个决策过程有多大把握是准的？",
             "流程通常是：先用权重大小+固定开销估算出'留给 KV cache 的显存预算'，再用 KV cache 公式除以"
             "预算得到理论最大 token 总容量，再除以平均预期序列长度得到理论并发数上限；但这只是一个"
             "静态上界，把握程度有限——真实流量的序列长度分布、请求到达的突发性（burst）、以及"
             "PagedAttention碎片开销都会让实际能安全支撑的并发数比理论值低一截，生产系统通常会再打一个"
             "经验安全系数（比如理论值的 70-85%）作为实际配置的并发上限，并配合运行时显存水位监控做动态"
             "限流，而不是完全信任一次性算出来的静态公式。",
             ("静态上界", "突发性", "经验安全系数")),
        ),
        pitfall="第2层很多人知道GQA省显存但说不出具体的H/G比例关系；第4层几乎没人提到要打安全系数和"
                "运行时动态限流,容易以为算出理论值就能直接拿来配置生产环境。",
        real_world_link="learning/inference-engine-core/src/naive_kv.py",
    ),

    DeepPoint(
        id="dp-ai-inf-09", cat=CAT,
        trigger="你提到线上服务对 KV cache 也做了量化（比如 FP8/INT8 KV cache）。",
        chain=(
            ("为什么要单独对 KV cache 做量化，而不是只量化权重？",
             "权重量化省的是模型静态存储的显存，但推理时占用显存增长最快、和并发/序列长度强相关的其实是 "
             "KV cache（尤其是长上下文、大并发场景下 KV cache 总量可能远超权重本身）；单独把 KV cache "
             "从标准的 fp16/bf16 降到 int8 或 fp8，可以在不改变模型权重精度、几乎不影响权重侧计算路径"
             "的情况下，直接把能支撑的最大并发数或最大上下文长度翻倍甚至更多，这是服务化场景里性价比"
             "很高的一个独立优化点。",
             ("权重量化", "并发/序列长度强相关", "性价比很高")),

            ("KV cache 量化和权重量化相比，对精度的敏感度有什么不同？为什么要格外小心 attention 的 softmax？",
             "KV cache 量化影响的是每一步 attention 计算里 Q 和 K 的点积、以及 attention 权重和 V 的"
             "加权求和，量化误差会直接影响 softmax 前的 logits 分布——如果 K 的量化误差在数值上偏向某个"
             "方向，会系统性地扭曲某些 token 的注意力得分排序，而 softmax 对输入的微小扰动在极端情况下"
             "（比如温度很低、注意力本来就很尖锐时）可能被放大成完全不同的注意力分布，这和权重量化误差"
             "更多是'整体幅度缩放偏差'的性质不同，KV 量化误差是逐 token、逐位置累积并直接影响'该关注谁'"
             "这个决策，理论上更容易导致长距离依赖任务的精度下降。",
             ("softmax 前的 logits 分布", "长距离依赖任务", "逐 token、逐位置累积")),

            ("实践中怎么判断 KV cache 量化是否导致了精度问题，是哪个位置（K 还是 V，哪一层）出的问题？",
             "常见做法是分离评测：先只量化 V 不量化 K（V 的量化误差只影响加权求和的数值精度，相对温和），"
             "看指标掉多少；再单独只量化 K（K 的误差影响 attention 权重分布本身，通常更敏感），对比两者"
             "掉点幅度就能判断哪个更敏感；逐层的话可以用类似权重量化的逐层敏感度扫描——只在某一层量化 "
             "KV cache 其余层保持全精度，跑一遍评测集看 perplexity 或下游任务指标的变化，通常会发现浅层"
             "（负责局部/语法特征）对 KV 量化更鲁棒，某些负责长距离检索型 attention 的深层（如 induction "
             "head 所在层）对 KV 精度更敏感。",
             ("分离评测", "K 的误差影响 attention 权重分布本身", "induction head 所在层")),
        ),
        pitfall="很多人知道'KV cache也能量化'但说不出它和权重量化在误差性质上的区别（幅度缩放 vs "
                "扭曲注意力排序）；第3层很少有人想到分离K和V单独评测这个具体的debug方法。",
        real_world_link="learning/quantization-deploy/src/kv_quant.py",
    ),

    DeepPoint(
        id="dp-ai-inf-10", cat=CAT,
        trigger="服务化系统的请求调度策略，除了 continuous batching，还考虑过哪些调度维度？",
        chain=(
            ("最基础的 FCFS（先来先服务）调度在推理服务里有什么明显缺陷？",
             "FCFS 按到达顺序处理请求，如果一个超长 prompt 排在队首，它的 prefill 计算会长时间占用计算"
             "资源，后面即使是很短、本可以秒回的请求也必须排队等待，造成明显的 head-of-line blocking，"
             "尾部延迟（tail latency）恶化；FCFS 也没有考虑请求的优先级或 SLO 差异（比如付费用户和免费"
             "用户应该有不同的延迟保证），把所有请求一视同仁地排队处理。",
             ("head-of-line blocking", "尾部延迟", "SLO 差异")),

            ("如果引入优先级调度（比如短请求优先或按 SLO 分级），会带来什么新的风险？",
             "优先级调度容易导致低优先级请求的饥饿（starvation）——如果高优先级/短请求流量持续不断，"
             "长请求或低优先级请求可能被无限期推迟；工程上通常引入老化机制（aging，等待时间越长优先级"
             "被逐步提升）或者预留最小资源配额给低优先级队列，来保证一个最坏情况下的延迟上界，纯粹的"
             "静态优先级排序在生产系统里几乎不会单独使用。",
             ("饥饿", "老化机制", "最坏情况下的延迟上界")),

            ("调度策略的选择最终要在哪些指标之间做权衡？有没有一个'全局最优'的调度算法？",
             "核心要在吞吐（tokens/s）、平均延迟、尾部延迟（p99/p999）、以及公平性/SLO 达成率之间做多目标"
             "权衡，这几个目标往往互相冲突（比如最大化吞吐通常意味着让 batch 尽量大、尽量少空闲，这会"
             "牺牲部分请求的延迟）；诚实地说不存在一个在所有流量模式下都最优的通用调度算法，这类问题在"
             "排队论里对应的是随机到达、多目标、抢占式调度的组合优化，理论上有一些启发式（如最短剩余"
             "时间优先的变体）在特定假设下证明较优，但真实流量的到达模式往往不满足这些理论假设的前提，"
             "生产系统普遍是针对自己的流量特征做定制调优+持续监控迭代，而不是套用某个理论最优算法。",
             ("多目标权衡", "互相冲突", "理论假设的前提")),
        ),
        pitfall="第2层很多人只想到'加优先级就行'，忽略饥饿问题；第3层容易说出一个听起来自信的'最优调度"
                "算法'，而不是诚实承认这是一个没有普遍最优解的多目标权衡问题。",
        real_world_link="learning/inference-engine-core/src/scheduling_policies.py",
    ),

    DeepPoint(
        id="dp-ai-inf-11", cat=CAT,
        trigger="draft model 用一段时间后目标模型升级了新版本，你的投机采样方案要怎么应对？",
        chain=(
            ("target model 换了新 checkpoint（哪怕架构不变，只是继续训练过），draft model 还能直接复用吗？",
             "如果 draft 是独立训练的小模型，理论上可以直接复用（因为 rejection sampling 的正确性只依赖"
             "target 分布本身、draft 只影响效率不影响正确性），但接受率 alpha 大概率会下降，因为新"
             "checkpoint 的输出分布和训练 draft 时用来对齐的旧分布产生了偏移，尤其如果新版本做了额外的"
             "对齐/风格调整（比如新一轮 RLHF），分布偏移可能更明显；如果 draft 是 EAGLE 这类基于 target "
             "隐藏特征训练的自蒸馏 draft，情况更糟——它是针对旧 checkpoint 的特征分布专门训练的，target "
             "换版本后特征空间可能整体偏移，draft 的外推能力可能显著失效，通常需要重新蒸馏训练。",
             ("正确性只依赖target 分布本身", "分布偏移", "重新蒸馏训练")),

            ("如果要判断'要不要重新训练draft'，你会怎么低成本地做这个决策，而不是每次都重新训练一遍？",
             "低成本做法是先离线跑一次接受率评估：用旧 draft 对新 target checkpoint 在一个代表性验证集"
             "上估计 alpha，如果 alpha 相比旧版本的基线只是小幅下降（比如从 0.75 降到 0.65 左右），可能"
             "仍然值得先凑合用，因为投机采样带来的净加速在alpha没有跌破某个阈值之前依然为正；如果 alpha "
             "大幅跌落（比如跌到 0.3 以下，验证成本可能已经超过省下的解码步数），就必须重新训练或至少"
             "微调 draft；这个阈值本身取决于 draft 和 target 的相对前向成本比，需要结合具体硬件和模型"
             "尺寸重新计算，没有一个通用固定的阈值数字。",
             ("离线跑一次接受率评估", "净加速在alpha没有跌破某个阈值之前依然为正", "没有一个通用固定的阈值数字")),

            ("从长期工程角度，怎么设计投机采样系统能让它对 target model 的持续迭代更鲁棒，而不是每次升级都被动救火？",
             "一种思路是把 draft 的训练变成 target model 训练/微调流水线里的一个常规后处理步骤（类似"
             "每次发布新 checkpoint 时自动跑一次蒸馏 pipeline 产出配套的新 draft），把'draft 会过时'这个"
             "事实产品化而不是当作意外；另一种思路是优先选择对分布偏移更鲁棒的 draft 设计（比如 EAGLE-2/"
             "EAGLE-3 这类利用多层特征、动态调整树结构的方案，据报告对 target 的持续训练/微调有更好的"
             "适应性），但要诚实承认这仍然是一个经验性的工程权衡而不是有理论保证的鲁棒性，具体多鲁棒"
             "需要针对自己的迭代频率和 target 变化幅度实测验证，业界目前没有形成一个公认的最佳实践"
             "标准。",
             ("产品化", "对分布偏移更鲁棒", "没有形成一个公认的最佳实践标准")),
        ),
        pitfall="第1层很多人不区分独立draft和EAGLE式自蒸馏draft对target升级的敏感度差异；第3层容易说"
                "一个'肯定没问题'的方案，而不诚实承认这是经验性权衡、缺乏理论鲁棒性保证。",
        real_world_link="learning/speculative-decoding/src/eagle2.py",
    ),

    DeepPoint(
        id="dp-ai-inf-12", cat=CAT,
        trigger="给定一块 GPU 的显存，你怎么规划权重、KV cache、激活缓冲之间的显存预算？",
        chain=(
            ("一块 GPU 的显存大致要在哪几个部分之间分配？",
             "大致分四块：1）模型权重（固定占用，量化与否直接决定这部分大小）；2）KV cache（随并发请求数"
             "和序列长度动态增长，是服务化场景里最大的可变量）；3）激活值/临时计算缓冲区（forward 过程中"
             "attention、MLP 等算子需要的中间张量，包括 PagedAttention kernel 的临时 buffer）；4）框架"
             "和驱动的固定开销（CUDA context、NCCL 通信 buffer、显存分配器本身的碎片和预留）。",
             ("模型权重", "KV cache", "固定开销")),

            ("这四块里，服务化和训练场景相比，哪一块的显存占比结构发生了明显变化？为什么？",
             "训练场景里优化器状态和梯度通常占大头（尤其是 Adam，一个参数背负好几份额外状态），推理服务"
             "完全不需要优化器状态和梯度，这两块直接归零；但 KV cache 在推理服务里从训练时几乎可以忽略"
             "的部分，变成了和并发规模、上下文长度强相关、经常成为显存瓶颈的主导项——一个长上下文、高"
             "并发的服务场景里，KV cache 占用甚至可能超过模型权重本身好几倍，这是从训练思维切换到服务化"
             "思维时最容易被低估的一点。",
             ("优化器状态和梯度", "归零", "超过模型权重本身好几倍")),

            ("如果你只根据'权重占用+理论KV cache公式'算出的预算配置服务，实际上线后经常在什么场景下会 OOM？",
             "几类典型场景：1）流量里偶发出现远超平均值的超长上下文请求（比如平时都是几百token，突然来"
             "几个几万token的长文档请求），理论预算按平均序列长度估计时没有为这类长尾请求预留余量；"
             "2）并发请求数的突发峰值（流量抖动）超过了按稳态估计的并发上限；3）某些请求触发了非常规"
             "路径（比如超长的 tool call 结果被拼回 context，或者多轮对话历史不断累积超出预期），这些"
             "都是'理论公式没建模到的长尾'，而不是公式本身算错了；这也是为什么生产系统必须叠加运行时"
             "显存水位监控和请求级别的准入控制（admission control，比如超过 token 预算的请求排队或拒绝）"
             "作为纸面预算规划的补充，而不是完全依赖一次性算好的静态配置。",
             ("长尾请求", "流量抖动", "准入控制")),

            ("如果要你对'这套显存预算规划'给一个诚实的置信度评估，你会怎么表达，而不是简单说'应该没问题'？",
             "诚实的表达应该是分层次的：权重占用和固定开销这两块几乎是确定性的，置信度很高；KV cache 的"
             "稳态估计（基于历史流量的平均/p90序列长度和并发数）置信度中等，取决于历史数据对未来流量的"
             "代表性；真正不确定的是长尾和突发场景，这部分本质上是一个概率性风险，无法通过更精确的公式"
             "消除，只能通过监控+动态限流+适当的安全边际（比如只用理论显存上限的70-85%做稳态配置）把"
             "OOM发生的概率压到可接受水平，而不是压到零；把这种不确定性讲清楚，比给出一个'肯定够用'的"
             "过度自信答案更接近真实工程状态。",
             ("分层次", "概率性风险", "压到可接受水平")),
        ),
        pitfall="第2层很多人还带着训练场景的直觉，忘了优化器状态归零、KV cache成为新主导项这个结构性"
                "变化；第4层几乎没人会给出分层次的置信度表达，大多数人要么过度自信要么完全说不出所以然。",
        real_world_link="learning/production-serving/src/cost_calc.py",
    ),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("dp-ai-inf-") for i in ids), "id 前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的点"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的点"
    total, hit = 0, 0
    for dp in BANK:
        answers = [ref for (_q, ref, _keys) in dp.chain]
        scores = grade_chain(dp, answers)
        for s in scores:
            total += 1
            if s == 1.0:
                hit += 1
    ratio = hit / total
    assert ratio >= 0.95, f"参考答案自洽性过低: {ratio:.0%} ({hit}/{total})"
    assert drill(BANK, cat=CAT, n=3) == BANK[:3]
    print(f"[PASS] dp_inference_serving: {len(BANK)}点 + 追问链自洽性 {ratio:.0%}")


if __name__ == "__main__":
    _self_test()
