"""Transformer与注意力深水区追问链（12个DeepPoint）。

覆盖：self-attention/MHA/MQA/GQA的精度-显存权衡细节、FlashAttention的IO感知重排原理
(纠正"它是近似算法"这个常见误解)、RoPE/ALiBi等位置编码在长上下文外推时的失败模式、
causal mask的实现细节与边界情况、attention复杂度优化的历史脉络、注意力数值稳定性。

边界：不涉及KV cache具体显存计算/PagedAttention(那是inference_serving类目的范围)，
不涉及MoE路由(那是moe_systems类目)。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, grade_chain  # noqa: E402

CAT = "Transformer与注意力深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-ai-trf-01",
        cat=CAT,
        trigger="你的模型用了32个attention head，每个head维度只有64，这个头数和head_dim的分配是怎么定的，有什么权衡？",
        chain=(
            ("头数和head_dim一般怎么分配？",
             "d_model=num_heads*head_dim是固定预算，头数和head_dim互相制约；常见选择是head_dim取64或128，"
             "保证每个head有足够的子空间维度做表达，同时保留多头带来的多样性。",
             ("d_model", "head_dim", "多样性")),
            ("如果头数远多于head_dim会带来什么代价？",
             "头数增多而head_dim变小，每个头能编码的子空间维度不够，单头表达能力被稀释，注意力分数的softmax"
             "在低维空间上更容易饱和、退化成近似均匀分布，实验上头数过多且head_dim过小时会出现质量下降。",
             ("子空间", "稀释", "softmax")),
            ("这个'头数太多质量下降'的结论有多可靠？是不是只在特定模型规模下成立？",
             "这主要来自经验消融，不是严格理论证明，跟模型规模、任务复杂度也有关，大模型能支撑更多头是因为"
             "d_model本身更大，把头数和模型规模脱钩讨论容易得出错误结论，这是经验性trade-off不是理论上的硬约束。",
             ("经验消融", "模型规模", "trade-off")),
            ("如果让你重新设计一个头数配置，你会怎么系统化地选，而不是照抄现成配置？",
             "理想是做小规模消融扫描头数-head_dim比例，但受限于算力，实践中通常直接沿用已验证的比例"
             "(如d_model/head_dim=64或128)，这是行业里普遍存在的'照抄配方'现象的诚实反映。",
             ("消融", "照抄配方", "算力")),
        ),
        pitfall="很多人第2层只会说'头多适合多头信息'，答不出稀释的具体机制；第3层容易把经验规律说成'理论上证明如此'，被追问'证明在哪'就露馅。",
        real_world_link="learning/transformer-deep/src/mha.py",
    ),
    DeepPoint(
        id="dp-ai-trf-02",
        cat=CAT,
        trigger="你提到推理阶段用了MQA节省显存，具体K/V是怎么共享的，这个共享对训练难度有影响吗？",
        chain=(
            ("MQA具体是怎么共享K/V的？",
             "MQA把所有query头的K/V投影统一成一份，仍然有H个独立的query头各自做投影和注意力计算，但K/V的线性"
             "投影矩阵只有一份，所有头复用同一组K/V。",
             ("query头", "K/V", "复用")),
            ("从头训练MQA是不是比从MHA转换过来更容易？",
             "从头训练MQA更难收敛，因为共享K/V在训练早期限制了每个头能学到的键值组合；更常见做法是先沿用MHA"
             "checkpoint，用mean-pooling把K/V头合并成一份，再做少量步数的uptraining，这样比直接从头训练MQA"
             "收敛更快、质量损失更小。",
             ("mean-pooling", "uptraining", "收敛")),
            ("uptraining能挽回多少质量？如果不做uptraining直接推理会怎样？",
             "论文报告只需要原始训练预算的5%左右做uptraining就能把质量拉回接近MHA水平；如果不做uptraining、"
             "直接暴力平均K/V头就使用，会出现明显的质量断崖式下降，因为attention pattern已经依赖了每个头独立"
             "的K/V，强行合并相当于分布外扰动。",
             ("5%", "断崖式下降", "分布外")),
            ("这个'5%预算能挽回'的结论对多大规模模型成立，你验证过没有？",
             "这是论文里针对特定规模模型报告的数值，不同架构和规模是否线性外推没有普适证明，自己没有在超大"
             "规模上验证过，只能类比小规模实验的趋势。",
             ("普适证明", "类比", "小规模")),
        ),
        pitfall="大多数人只知道MQA'共享KV省显存'，答不出uptraining这一具体缓解方案，被追问'如果不做uptraining会怎样'时开始编。",
        real_world_link="learning/transformer-deep/src/mqa.py",
    ),
    DeepPoint(
        id="dp-ai-trf-03",
        cat=CAT,
        trigger="Llama2用GQA分了8组，为什么是8组而不是4组或16组，这个数字怎么定的？",
        chain=(
            ("GQA分组机制是怎样的？",
             "GQA把H个query头分成G组，同一分组内的query头共享一份K/V投影，不同组各自独立；G=H退化为MHA，"
             "G=1退化为MQA，通过G在两者间连续权衡。",
             ("分组", "G=H", "G=1")),
            ("组数8是怎么定的，这个数字有什么工程考量？",
             "组数通常要和张量并行(TP)的GPU数对齐，比如8卡TP就选8组，这样每张卡刚好负责1组的K/V，避免跨卡"
             "读取KV cache；组数越多越接近MHA质量、但显存带宽收益越小。",
             ("张量并行", "8卡", "显存带宽")),
            ("如果GQA组数要考虑TP并行度，这是不是说明这个超参数本质上被系统工程约束反向决定，而不是纯建模最优？",
             "确实如此，很多所谓的建模超参数实际上是被服务/训练时的并行拓扑倒逼出来的选择，组数选8很大程度是"
             "服务时8卡TP不用跨卡通信的工程决定，先满足这个约束再在此范围内选质量最好的组数。",
             ("并行拓扑", "工程决定", "质量最好")),
            ("这种'工程约束优先于建模最优'的现象你怎么看？",
             "这是实际大规模训练里的常态，理想化的先做消融再定超参在超大规模场景下往往不现实，只能说这是一种"
             "合理但不完美的次优妥协。",
             ("常态", "次优妥协", "不现实")),
        ),
        pitfall="只会说'8组是经验值'，答不出和张量并行度对齐这个具体工程原因，第3层容易被问懵。",
        real_world_link="learning/transformer-deep/src/gqa.py",
    ),
    DeepPoint(
        id="dp-ai-trf-04",
        cat=CAT,
        trigger="你提到用了FlashAttention加速训练，它是怎么做到又快又省显存的，是不是牺牲了精度做近似？",
        chain=(
            ("FlashAttention是不是近似算法？",
             "FlashAttention是IO感知的精确注意力算法，不是近似算法，计算出的结果和标准attention数学上完全"
             "一致，没有牺牲任何精度。",
             ("IO感知", "精确", "数学上完全一致")),
            ("它到底解决的是什么问题？",
             "标准实现要把T×T的完整注意力矩阵读写进GPU的HBM，而现代GPU算力远大于显存带宽，attention在长序列"
             "下是memory-bound而非计算瓶颈；FlashAttention通过分块(tiling)让小块常驻SRAM完成计算，避免把T×T"
             "矩阵整体落盘到HBM。",
             ("HBM", "memory-bound", "tiling")),
            ("如果它完全精确，FLOPs总量应该和标准attention一样，那到底降低的是什么？",
             "确实FLOPs总量不变(仍是O(T²d))，降低的是HBM读写字节数，从O(T²)降到接近线性的访存量；在"
             "memory-bound场景下wall-clock time主要由显存搬运决定，所以加速比在长序列上更大、短序列上不明显。",
             ("FLOPs总量不变", "显存搬运", "长序列")),
            ("有没有场景FlashAttention反而不划算或者收益很小？",
             "序列很短、batch很小时kernel launch开销和tiling调度成本可能让收益不明显甚至持平；某些非标准"
             "attention变体如果没有对应的flash实现，需要retreat到naive实现。",
             ("kernel launch", "调度成本", "retreat")),
        ),
        pitfall="几乎所有人第一层就说错——把FlashAttention当成近似/稀疏方法；能纠正这个误解的人里，很多在第3层答不出'降低的是访存量不是FLOPs'这个关键区分。",
        real_world_link="learning/transformer-deep/src/flash_attn_naive.py",
    ),
    DeepPoint(
        id="dp-ai-trf-05",
        cat=CAT,
        trigger="FlashAttention具体怎么在分块的情况下算出和整行softmax完全一样的结果，反向传播不保存注意力矩阵怎么算梯度？",
        chain=(
            ("分块情况下怎么保证softmax结果精确？",
             "用在线softmax(online softmax)技巧，每处理完一个新块就用running max和running sum增量更新之前"
             "块的归一化统计量，数学上保证最终结果和一次性算完整行softmax完全等价。",
             ("在线softmax", "running max", "等价")),
            ("反向传播为什么选择重算而不是缓存完整注意力矩阵？",
             "反向传播只保存每行的logsumexp统计量，需要梯度时用recomputation在SRAM里重新计算对应的注意力"
             "权重，用少量额外FLOPs换取大幅减少显存读写，因为保存完整T×T矩阵在长序列下显存开销远大于重算。",
             ("logsumexp", "recomputation", "显存读写")),
            ("这个'recompute比cache更划算'的判断在什么条件下会反过来？",
             "这个判断依赖硬件的计算/带宽比，如果换到计算相对稀缺、显存带宽相对充裕的硬件，recompute的换算比"
             "就会变得不划算；说明FlashAttention是对A100/H100这类计算远快于HBM带宽的具体硬件特征做的算法工程。",
             ("计算/带宽比", "A100/H100", "算法工程")),
            ("所以FlashAttention是不是有过拟合到当前硬件的风险？",
             "某种程度上是的，FlashAttention-3针对H100的异步执行和FP8支持做了专门优化，说明这类算法需要跟着"
             "硬件代际重新设计，是当前的诚实局限而非一劳永逸的通用解。",
             ("FlashAttention-3", "H100", "诚实局限")),
        ),
        pitfall="很多人只答得出'用了在线softmax'，答不出反向为什么选择recompute而不是cache这个具体权衡，更答不出这个权衡是硬件相关的。",
        real_world_link="learning/transformer-deep/src/flash_attn_lib.py",
    ),
    DeepPoint(
        id="dp-ai-trf-06",
        cat=CAT,
        trigger="FlashAttention要分块计算，causal mask在分块情况下怎么处理，是每块都算完再mask掉吗，那不是浪费算力？",
        chain=(
            ("causal mask在分块下具体怎么处理？",
             "对角线上完全在下三角内的block正常算全部，完全在上三角(未来)的block直接跳过不计算，只有对角线上"
             "的block需要在块内部再应用逐元素mask。",
             ("下三角", "跳过", "逐元素mask")),
            ("这种block-level skip能带来多大收益？",
             "理论上causal mask下只需要算T²/2的block对而不是T²，能省掉大约一半的计算量，但naive实现如果每个"
             "block都算完再mask就完全浪费了这个理论收益，需要在调度层面提前判断block是否可跳过。",
             ("T²/2", "一半", "调度")),
            ("如果load balance因为causal skip变得不均衡，这对GPU并行效率有什么影响？",
             "不同query block需要处理的key block数量不同，越靠后位置需要处理的block越多，会造成负载不均衡、"
             "前面的block提前完成、GPU利用率下降；FlashAttention-2/3通过调整并行调度策略缓解这种不均衡。",
             ("负载不均衡", "GPU利用率", "并行调度")),
            ("这个负载均衡问题解决得完美吗？",
             "不完美，超长序列+causal场景下这仍是持续优化的方向，FlashAttention-3通过warp专化和异步流水线"
             "进一步缓解但没有完全消除这个不均衡。",
             ("不完美", "warp专化", "异步流水线")),
        ),
        pitfall="大多数人以为causal mask只是'算完再置-inf'，答不出block跳过省算力这一层，更不会想到这带来负载不均衡的问题。",
        real_world_link="learning/transformer-deep/src/flash_attn_naive.py",
    ),
    DeepPoint(
        id="dp-ai-trf-07",
        cat=CAT,
        trigger="训练时batch里有变长序列做了padding，causal mask和padding mask要怎么一起用，推理decode阶段只有一个新token时mask又是什么样的？",
        chain=(
            ("causal mask和padding mask怎么组合？",
             "两个mask是逻辑与的关系，causal mask保证看不到未来，padding mask保证看不到pad位置，实现上把两个"
             "mask取交集后一起在softmax前置-inf。",
             ("逻辑与", "交集", "-inf")),
            ("如果某一行mask后所有key位置都是-inf会怎样？",
             "如果某个位置的causal+padding交集为空，softmax会产生NaN，工程实现要专门处理这种全屏蔽行的边界"
             "情况，比如保底让自身位置可见。",
             ("NaN", "全屏蔽行", "保底")),
            ("推理decode阶段为什么不需要显式的下三角mask矩阵了？",
             "decode阶段每步只有1个新query token，要attend到KV cache里所有已经生成的历史token，这些历史"
             "token在因果关系上天然都在过去，不存在需要屏蔽的未来位置，所以mask退化成全1，这是prefill阶段和"
             "decode阶段代码路径完全不同的原因。",
             ("KV cache", "退化成全1", "prefill")),
            ("如果用left padding而不是right padding，对RoPE这类依赖绝对位置的编码会有什么坑？",
             "如果用left padding，每个样本的有效起始位置不一样，如果直接按物理下标算RoPE的旋转角度，不同样本"
             "里同一个逻辑位置会被赋予不同角度，必须用position_ids显式跟踪每个token的逻辑位置来计算RoPE。",
             ("left padding", "position_ids", "逻辑位置")),
        ),
        pitfall="很多人答得出causal+padding要一起mask，但答不出全屏蔽行导致NaN的边界情况，也答不出decode阶段为什么不需要mask矩阵；left padding+RoPE位置错位这个坑基本没人第一次就能答对。",
        real_world_link="learning/transformer-deep/src/gpt_mini.py",
    ),
    DeepPoint(
        id="dp-ai-trf-08",
        cat=CAT,
        trigger="RoPE号称外推性比绝对位置编码好，但你训练时上下文只有4K，直接推理到32K效果为什么还是崩了？",
        chain=(
            ("RoPE的数学形式是什么？",
             "RoPE把Q、K向量每两个维度看成一个二维子空间，对位置m的向量乘以旋转角度为m·θ_i的二维旋转矩阵，"
             "θ_i按维度几何衰减，低频维度编码长距离关系、高频维度编码近距离关系。",
             ("旋转矩阵", "θ_i", "几何衰减")),
            ("外推失败的具体机制是什么？",
             "高频维度在训练时从未见过超出训练长度对应的旋转角度组合，这些角度在超出训练分布后本质上是OOD"
             "输入，导致attention score模式发生分布外错乱，实践中直接外推会看到attention entropy突然升高、"
             "perplexity爆炸。",
             ("OOD", "attention entropy", "perplexity爆炸")),
            ("是所有维度都会失败，还是只有部分维度失败？这个失败有没有可观测信号？",
             "主要是高频(短周期)维度先失效，低频维度受外推影响相对小；可观测信号包括困惑度急剧升高、"
             "needle-in-a-haystack这类长上下文检索评测通过率骤降、以及可视化不同维度的旋转角度分布定位"
             "哪些维度先转飞了。",
             ("高频", "needle-in-a-haystack", "旋转角度分布")),
            ("如果不能重新训练，只能推理时补救，这些补救方法完全解决问题了吗？",
             "可以用position interpolation或NTK-aware/YaRN这类方法缓解，但都只是缓解不是根治，超出某个长度后"
             "即使用了这些技巧质量仍会有不同程度下降，只是把断崖式失败变成渐进式衰减。",
             ("position interpolation", "缓解不是根治", "渐进式衰减")),
        ),
        pitfall="很多人只会说'RoPE外推性好'当作结论背，追问'为什么会崩'时说不出高频维度OOD这个具体机制，更给不出可观测的诊断信号。",
        real_world_link="learning/long-context/src/rope_yarn.py",
    ),
    DeepPoint(
        id="dp-ai-trf-09",
        cat=CAT,
        trigger="ALiBi不用位置embedding，理论上能外推到任意长度，那它是不是就没有长度限制了？",
        chain=(
            ("ALiBi具体是怎么做位置编码的？",
             "ALiBi不使用位置embedding，而是在注意力分数softmax之前直接加一个线性偏置项m·(j-i)，斜率m是每个"
             "注意力头固定的超参数，只依赖相对距离。",
             ("线性偏置", "斜率m", "相对距离")),
            ("既然公式对任意距离都有定义，为什么还会有长度限制？",
             "虽然公式对任意距离都有定义，但如果实际使用长度远超训练长度，距离penalty会变得非常大，导致远处"
             "的注意力权重被压到几乎为0，模型实际上退化成一个有效上下文窗口远小于名义无限长度的局部注意力。",
             ("penalty", "压到几乎为0", "局部注意力")),
            ("ALiBi的'无限外推'是不是名不副实？它和RoPE比外推能力谁更强？",
             "ALiBi不会数值崩溃和ALiBi能真正利用超长距离信息是两件不同的事，ALiBi更多是优雅退化成短距离窗口，"
             "RoPE配合YaRN这类专门技巧后在检索类任务上反而可能保留更多远距离信息，这个比较高度依赖'外推'"
             "具体指的是稳定性还是可用性。",
             ("优雅退化", "YaRN", "稳定性还是可用性")),
            ("现在主流大模型选RoPE的多、选ALiBi的少，这说明了什么？",
             "这更多是工程生态和实证效果的综合考量，RoPE和FlashAttention等kernel生态结合更成熟，这个选择更多"
             "是生态锁定而非纯技术最优。",
             ("工程生态", "kernel生态", "生态锁定")),
        ),
        pitfall="很多人只记住'ALiBi外推好'这个标签，答不出它其实是优雅退化到局部窗口而不是真正利用长距离信息，容易在'外推'这个词的定义上被面试官带偏。",
        real_world_link="learning/transformer-deep/src/pe_alibi.py",
    ),
    DeepPoint(
        id="dp-ai-trf-10",
        cat=CAT,
        trigger="你提到用position interpolation把4K模型扩展到16K，这个线性插值具体怎么做，为什么效果不如YaRN？",
        chain=(
            ("position interpolation具体怎么做？",
             "把位置下标乘以一个缩放因子(训练长度/目标长度)压缩进原本训练时见过的位置范围内，不改变频率base，"
             "只是让绝对位置数值变小从而落回训练分布。",
             ("缩放因子", "频率base", "落回训练分布")),
            ("这个线性压缩有什么代价？",
             "对所有频率维度做同样比例的线性压缩，代价是压缩了相邻token之间的有效角度差，高频维度被压缩后"
             "相邻很近的token之间角度差过小，模型分辨近距离精细差异的能力下降，需要少量继续训练来恢复。",
             ("有效角度差", "高频维度", "继续训练")),
            ("YaRN具体怎么避免这个压缩了近距离分辨率的问题？",
             "YaRN不对所有维度做同样的线性插值，而是按频率分段处理，高频维度基本不插值以保留近距离分辨率，"
             "低频维度做更大程度的插值，中间频率做平滑过渡(NTK-by-parts)，同时引入attention scaling因子"
             "补偿softmax分布变化。",
             ("按频率分段", "NTK-by-parts", "attention scaling")),
            ("YaRN是不是就终结了长上下文外推这个问题？",
             "没有，YaRN仍然依赖少量的continued pretraining才能达到好效果，纯zero-shot的YaRN虽然比PI好但仍有"
             "质量损失，这仍是一个持续演进的开放问题。",
             ("continued pretraining", "zero-shot", "开放问题")),
        ),
        pitfall="很多人只知道PI/YaRN是'扩展上下文的方法'，答不出PI具体牺牲了什么(近距离分辨率)，更答不出YaRN'按频率分段处理'这个关键区别，容易把两者混为一谈。",
        real_world_link="learning/long-context/src/rope_yarn.py",
    ),
    DeepPoint(
        id="dp-ai-trf-11",
        cat=CAT,
        trigger="除了FlashAttention，你还知道哪些降低attention复杂度的方法，它们最后为什么大多没有成为主流？",
        chain=(
            ("历史上有哪些降低attention复杂度的思路？",
             "稀疏注意力(固定稀疏模式)、线性注意力(用低秩或核函数近似softmax把复杂度降到线性)、滑动窗口/局部"
             "注意力，以及FlashAttention这种不改变数学定义只优化IO的精确算法。",
             ("稀疏注意力", "线性注意力", "滑动窗口")),
            ("这些方法和FlashAttention的本质区别是什么？",
             "稀疏/线性注意力通过改变attention的数学定义把理论FLOPs复杂度降到O(T)或O(T log T)，代价是牺牲了"
             "建模能力上限；FlashAttention不改变数学定义只优化硬件层面IO，没有建模能力损失，这是它能被无损"
             "大规模采用的根本原因。",
             ("数学定义", "建模能力上限", "无损")),
            ("如果线性注意力理论复杂度更低，为什么FlashAttention反而成为主流？",
             "实际训练/推理在中等长度下瓶颈往往不是FLOPs而是访存，FlashAttention把O(T²)问题在这个长度区间内"
             "解决得足够好，而线性/稀疏注意力牺牲的建模精度在需要精确检索的任务上造成的效果损失往往得不偿失，"
             "只有序列长度扩展到百万级别时次二次方案才重新有吸引力。",
             ("访存", "得不偿失", "百万级别")),
            ("现在业界对'to be or not to be O(T²)'这个问题有共识了吗？",
             "没有完全的共识，超长上下文场景下Ring Attention、SSM/Mamba、线性注意力的复兴都还在竞争，谁能"
             "最终胜出取决于长上下文任务对精确长距离检索能力的实际需求有多刚性，这是仍在演化中的开放问题。",
             ("没有完全的共识", "Ring Attention", "SSM/Mamba")),
        ),
        pitfall="很多人只会列举'有稀疏注意力、线性注意力、FlashAttention'这几个名字，答不出为什么FlashAttention(理论复杂度没变)反而比理论复杂度更优的线性注意力更成功这一反直觉的关键点。",
        real_world_link="learning/long-context/src/ring_attention_lib.py",
    ),
    DeepPoint(
        id="dp-ai-trf-12",
        cat=CAT,
        trigger="你用bf16混合精度训练一个大模型，attention内部的softmax要不要转成fp32算，为什么？",
        chain=(
            ("softmax一般在什么精度下计算？",
             "softmax通常在fp32(或至少fp32累加)里计算，即使Q/K/V的matmul本身是在bf16/fp16里做的，因为softmax"
             "对数值范围敏感。",
             ("fp32", "matmul", "数值范围敏感")),
            ("为什么softmax对精度这么敏感？",
             "QK^T点积在d_k较大时数值范围会很大，如果不做sqrt(d_k)缩放，softmax输入方差会随d_k线性增长，在"
             "fp16下容易上溢出变成inf，或者多个大值经过softmax后除了最大值其他全部下溢为0导致梯度消失，"
             "注意力退化成one-hot。",
             ("sqrt(d_k)", "上溢出", "梯度消失")),
            ("如果不做fp32 softmax，实际训练会出现什么可观测现象？怎么定位？",
             "会观察到loss出现突然的spike或者变成NaN/Inf，定位方法包括对attention模块单独开启fp32做对照实验、"
             "监控QK^T点积的最大绝对值是否有异常尖峰，以及用梯度裁剪作为兜底但不能替代根因修复的手段。",
             ("spike", "对照实验", "梯度裁剪")),
            ("缩放因子1/sqrt(d_k)在超大d_k或非标准head维度设计下还完全适用吗？",
             "1/sqrt(d_k)是在假设QK各分量独立同分布的简化推导下得到的经验缩放，实际里如果d_k设计得不标准，"
             "这个缩放因子是否仍是最优选择并没有被严格重新证明过，工程上更多是延续用、必要时做小范围消融微调。",
             ("独立同分布", "简化推导", "延续用")),
        ),
        pitfall="很多人知道'除以sqrt(d_k)防止梯度消失'这个背诵版说法，但说不清楚fp16/bf16下溢出/上溢的具体数值机制，更不会想到非标准维度设计下缩放因子是否还适用这个更深的问题。",
        real_world_link="learning/transformer-deep/src/mla.py",
    ),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("dp-ai-trf-") for i in ids), "id前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的条目"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的条目"
    assert all(dp.trigger for dp in BANK), "存在缺失trigger的条目"
    for dp in BANK:
        answers = [ref for (_q, ref, _k) in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 采分关键词未能在参考答案里全部命中: {scores}"
    print(f"[PASS] dp_transformer_attention: {len(BANK)}个DeepPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
