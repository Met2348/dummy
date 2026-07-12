"""Transformer核心架构八股问答库（约 14 题）。

与 interview-prep/src/mlqa/qbank.py 里已有的 6 道 Transformer 原题
(自注意力复杂度/为什么除√d/多头意义/位置编码种类/KV cache为什么能用/encoder-decoder区别)
不重复——本库覆盖更深、更细分的问题：MHA的局限与MQA/GQA的具体机制区别、
FlashAttention的原理(IO感知的精确计算重排，不是近似算法)、位置编码的数学形式
(RoPE/ALiBi)、Transformer整体结构怎么拼起来、Q/K/V怎么来、为什么需要causal mask。

边界：只写KV cache"机制层面"(为什么MHA的KV cache会成为解码瓶颈)，不涉及
KV cache具体占多少显存、PagedAttention怎么管理显存碎片这类部署工程细节——
那些属于"推理部署与服务化"类别。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "Transformer核心架构"

BANK: list[QA] = [
    QA(id="ai-trf-01", cat=CAT,
       q="一个标准 Transformer(以 encoder-decoder 为例)在结构上是怎么拼起来的？",
       a="输入先经过token embedding加位置编码，然后堆叠N个相同结构的block；每个encoder block = "
         "多头自注意力(Multi-Head Self-Attention) + 残差连接(residual) + LayerNorm，再接一个"
         "position-wise前馈网络(FFN，两层MLP中间过GELU/ReLU，先升维再降维) + 残差连接 + LayerNorm；"
         "decoder block在此基础上多一层masked(causal)自注意力和一层对encoder输出做cross-attention"
         "的模块，同样每个子层都包一层残差+LayerNorm；最后接输出线性层+softmax得到词表分布。整体是"
         "'注意力做信息交互、FFN做逐位置特征变换、残差+LayerNorm保证深层可训练'的三件套反复堆叠。",
       keys=("残差", "LayerNorm", "FFN", "多头自注意力"),
       follow_ups=("为什么每个子层都要配残差连接和LayerNorm？", "FFN的参数量占整个模型的比例大概是多少？")),

    QA(id="ai-trf-02", cat=CAT,
       q="self-attention 里的 Q/K/V 分别是怎么来的？直觉上各自代表什么？",
       a="Q(query)、K(key)、V(value)都是同一个输入序列的隐藏状态分别乘以三个独立可学习的投影矩阵"
         "W_Q/W_K/W_V得到的，本身没有预先规定的语义，是训练中学出来的。直觉上Q代表'当前位置在找什么"
         "信息'，K代表'每个位置能提供什么信息的索引'，V代表'每个位置实际携带的内容'；注意力分数由Q和K"
         "的点积算出'谁该关注谁'的权重，再用这组权重对V做加权求和，取出真正被使用的信息。self-attention"
         "里Q/K/V来自同一序列，cross-attention里Q来自一侧(如decoder)而K/V来自另一侧(如encoder输出)。",
       keys=("W_Q", "W_K", "W_V", "加权求和"),
       follow_ups=("self-attention和cross-attention在QKV来源上有什么区别？", "为什么Q和K要用不同的投影矩阵而不能共用一个？")),

    QA(id="ai-trf-03", cat=CAT,
       q="为什么 Transformer 的自回归解码需要 causal mask？",
       a="自回归语言模型训练时用teacher forcing、一次性把整段目标序列喂进去并行计算loss，如果不加"
         "掩码，某个位置的self-attention会直接看到它自己右边(未来)的token，等于训练时作弊看到了要"
         "预测的答案，推理时却根本不可能有未来token可看，会导致训练/推理分布不一致、模型学到无法在"
         "推理时复现的捷径。causal mask(下三角掩码)把每个位置到未来位置的注意力分数在softmax前置为"
         "-inf，强制每个位置只能聚合它自己和之前的信息，保证并行训练时的计算过程和自回归推理时'只能"
         "看历史'的约束完全一致。",
       keys=("teacher forcing", "下三角", "-inf", "自回归"),
       follow_ups=("causal mask和padding mask是什么关系，能不能一起用？", "encoder为什么不需要causal mask？")),

    QA(id="ai-trf-04", cat=CAT,
       q="标准多头注意力(MHA)在实际部署里的局限是什么？",
       a="标准多头注意力(MHA)在训练阶段没有明显短板，但在自回归推理阶段暴露出瓶颈：每个头都有自己"
         "独立的K/V投影，解码时要为每个头都缓存一份KV cache，头数H越多，KV cache占用的显存和每步"
         "读取KV cache的显存带宽开销就越大；由于自回归解码每步都是访存密集(memory-bound)而非计算"
         "密集，KV cache的读取带宽往往是真正的推理速度瓶颈，这正是MQA/GQA被提出来专门解决的问题。",
       keys=("KV cache", "显存带宽", "memory-bound", "头数"),
       follow_ups=("MQA具体怎么改造MHA来缓解这个问题？", "这个瓶颈在训练阶段也存在吗？")),

    QA(id="ai-trf-05", cat=CAT,
       q="MQA(Multi-Query Attention)具体是怎么改造 MHA 的？",
       a="MQA把K和V的投影从'每个头一份'改成'全部query头共享同一份K头和V头'——也就是说仍然有H个"
         "独立的query头各自做投影和注意力计算，但K、V的线性投影矩阵只有一份，所有头在计算注意力时"
         "复用同一组K/V。这样KV cache的大小和读取带宽直接从与头数H成正比降到与1个头等价，大幅缓解"
         "解码时的访存瓶颈，代价是K/V的表达能力被压缩、每个头不能再有各自定制的键值子空间，实践中"
         "会带来一定的质量下降。",
       keys=("共享", "K/V", "头数H", "质量下降"),
       follow_ups=("MQA大幅降低KV cache后为什么还会有质量损失？", "有哪些模型实际用了MQA？")),

    QA(id="ai-trf-06", cat=CAT,
       q="GQA(Grouped-Query Attention)是怎么在 MHA 和 MQA 之间做折中的？",
       a="GQA(Grouped-Query Attention)是MHA和MQA之间的插值方案，核心做法是对query头分组(grouping)："
         "把H个query头分成G组(1<=G<=H)，同一分组内的所有query头共享一份K头和V头，不同组各自拥有"
         "独立的K/V投影。G=H时每个query头都有自己的K/V，退化为标准MHA；G=1时所有query头共享同一份"
         "K/V，退化为MQA。通过调节组数G，可以在'KV cache省多少显存带宽'和'保留多少K/V表达多样性'之间"
         "连续权衡，Llama2、Mistral等主流模型都采用GQA作为MHA和MQA的折中。",
       keys=("分组", "G=1", "G=H", "折中"),
       follow_ups=("实际工程里组数G一般怎么选？", "GQA相比MQA能在多大程度上挽回质量损失？")),

    QA(id="ai-trf-07", cat=CAT,
       q="MQA/GQA 省的到底是计算量还是显存？为什么会有质量损失、怎么缓解？",
       a="MQA/GQA主要降低的是KV cache的显存带宽开销，而不是注意力本身的计算量(FLOPs)——因为每个"
         "query头依然要对(共享的)K/V完整计算一遍注意力分数，只是K/V的'制造成本'被摊薄了。质量下降"
         "的根源在于多个query头被迫共用同一份键值子空间，丧失了MHA里'每个头各自学一种关系模式'的"
         "多样性；GQA通过分组保留部分多样性来缓解，另外从头训练时给GQA配合适的分组数、或者用MHA "
         "checkpoint通过mean-pooling方式初始化再少量继续训练(uptraining)，都能进一步挽回质量损失。",
       keys=("显存带宽", "FLOPs", "多样性", "uptraining"),
       follow_ups=("uptraining具体怎么把一个MHA模型转换成GQA模型？", "除了GQA还有哪些方法能压缩KV cache？")),

    QA(id="ai-trf-08", cat=CAT,
       q="FlashAttention 解决的是什么问题？它是近似算法吗？",
       a="FlashAttention是一种IO感知(IO-aware)的精确注意力算法，而不是近似算法——它计算出的结果和"
         "标准注意力(softmax(QK^T/√d)V)数学上完全一致，没有牺牲任何精度。它解决的问题是：标准实现"
         "要把T×T的完整注意力矩阵读写进GPU的HBM(显存)，而现代GPU的算力(FLOPS)远大于显存带宽，导致"
         "attention在长序列下是访存瓶颈(memory-bound)而不是计算瓶颈；FlashAttention通过分块(tiling)"
         "让Q/K/V的小块常驻在更快的片上SRAM里完成计算，避免把T×T矩阵整体落盘到HBM，从而把显存读写"
         "量从O(T²)降到接近线性，实测能显著加速训练和推理并降低显存占用。这是一个常见误区——很多人"
         "把它和线性注意力/稀疏注意力这类近似方法混为一谈。",
       keys=("IO感知", "精确", "HBM", "tiling"),
       follow_ups=("FlashAttention具体怎么在不materialize完整注意力矩阵的情况下算出精确的softmax？", "FlashAttention-2/3相比第一版做了什么改进？")),

    QA(id="ai-trf-09", cat=CAT,
       q="FlashAttention 具体是怎么做到不 materialize 完整注意力矩阵、又保证结果精确的？",
       a="FlashAttention把Q按行分块、K/V按列分块，依次把每一小块从HBM搬到SRAM，在片上完成这一小块"
         "的QK^T、mask、局部softmax统计量(最大值和指数和)和与V的加权累加；因为完整的softmax需要看到"
         "一整行所有元素之后才能归一化，而分块只能看到局部，所以用'在线softmax(online softmax)'"
         "技巧——每处理完一个新块就用running max和running sum增量更新之前块的归一化统计量，数学上"
         "保证最终结果和一次性算完整行softmax完全等价。反向传播时它不保存完整的注意力矩阵，而是只"
         "保存每行的logsumexp统计量，需要梯度时用它们在SRAM里重新计算(recomputation)对应的注意力"
         "权重，用少量额外FLOPs换取大量减少的显存读写。",
       keys=("在线softmax", "分块", "logsumexp", "recomputation"),
       follow_ups=("为什么反向传播宁愿重算也不缓存完整注意力矩阵？", "这种分块方式对因果mask(causal)的实现有什么额外要求？")),

    QA(id="ai-trf-10", cat=CAT,
       q="RoPE(旋转位置编码)的数学形式是什么？为什么能编码相对位置？",
       a="RoPE(Rotary Position Embedding，旋转位置编码)不是把位置信息加到embedding上，而是把Q、K"
         "向量的每两个维度看成一个二维子空间，对位置为m的向量在第i个子空间上乘以一个旋转角度为m·θ_i"
         "的二维旋转矩阵，其中θ_i按维度呈几何衰减的频率——低频维度转得慢、编码长距离关系，高频维度"
         "转得快、编码近距离关系。这样构造的关键性质是：旋转后的q_m和k_n做点积，结果只依赖于相对"
         "位置(m-n)而与绝对位置m、n无关，等价于用绝对位置的旋转操作隐式实现了相对位置编码，这也是"
         "RoPE被LLaMA、GPT-NeoX等主流模型采用、并且长度外推性优于可学习绝对位置编码的数学基础。",
       keys=("旋转矩阵", "θ_i", "相对位置", "LLaMA"),
       follow_ups=("为什么说RoPE用绝对位置的操作实现了相对位置的效果？", "RoPE在超出训练长度的外推(extrapolation)上有什么已知问题？")),

    QA(id="ai-trf-11", cat=CAT,
       q="ALiBi 是怎么做位置编码的？和加性位置embedding有什么本质不同？",
       a="ALiBi(Attention with Linear Biases)完全不使用位置embedding，而是在注意力分数softmax之前"
         "直接加一个静态的、不可学习的线性偏置项(linear bias)：对query位置i和key位置j，加上m·(j-i)"
         "(距离越远惩罚越负)，其中斜率m是每个注意力头固定的超参数(不同头用不同的m，通常取一个几何"
         "数列)。因为这个偏置只依赖相对距离|i-j|，不需要在输入端编码任何位置信息，训练长度和推理长度"
         "的差异不会破坏它的语义，所以ALiBi在长度外推(在比训练序列更长的输入上推理)上表现明显优于"
         "正弦/可学习式绝对位置编码。",
       keys=("线性偏置", "斜率m", "相对距离", "外推"),
       follow_ups=("ALiBi的斜率m是怎么给不同头分配的？", "ALiBi和RoPE在外推能力上谁更好，为什么？")),

    QA(id="ai-trf-12", cat=CAT,
       q="绝对位置编码和相对位置编码的本质区别是什么？这如何决定了外推能力？",
       a="绝对位置编码(如原始Transformer的正弦编码、BERT的可学习embedding)直接给每个绝对位置分配"
         "一个固定或可学习的向量加到token embedding上，模型学到的是'位置0/1/2...各自长什么样'的具体"
         "模式；相对位置编码(RoPE、ALiBi、经典的相对位置bias)不关心token在序列里的绝对下标，只编码"
         "'两个token之间差几步'。这个区别决定了外推能力：绝对编码在超过训练时见过的最大位置后就没有"
         "对应的向量可用(或行为未定义)，而相对编码天然是关于相对距离的函数，距离更远只是同一套公式"
         "代入更大的数字，不需要见过训练时没见过的'位置'，因此RoPE、ALiBi等相对方案的长度外推性普遍"
         "优于绝对位置编码。",
       keys=("绝对位置", "相对位置", "外推", "下标"),
       follow_ups=("RoPE严格来说算绝对编码还是相对编码，为什么两种说法都有？", "除了外推性，相对位置编码还有什么其他优势？")),

    QA(id="ai-trf-13", cat=CAT,
       q="FFN(前馈网络)在 Transformer 里起什么作用，和 attention 分工有什么不同？",
       a="Transformer每个block里的position-wise feed-forward network(FFN)对序列每个位置进行逐位置"
         "(position-wise)的两层MLP变换：先升维(通常到4倍隐藏维度)，过一个非线性激活(ReLU/GELU/"
         "SwiGLU等)，再降维回原维度。它和self-attention分工不同——attention负责在不同位置之间做信息"
         "交互和路由，FFN不做位置间交互，只对已经聚合好信息的每个位置单独做非线性特征变换，相当于给"
         "每个token的表示'再加工提炼'一次；在参数量上FFN通常占了整个Transformer总参数的三分之二左右，"
         "是模型容量的主要载体。",
       keys=("升维", "非线性", "逐位置", "参数量"),
       follow_ups=("为什么FFN不像attention那样做位置间的交互？", "SwiGLU相比标准ReLU FFN做了什么改动？")),

    QA(id="ai-trf-14", cat=CAT,
       q="残差连接(residual connection)在 Transformer 里的作用是什么？",
       a="残差连接(residual connection)把每个子层(self-attention或FFN)的输入直接加到其输出上，即"
         "output = x + Sublayer(x)。它的核心作用是给梯度提供一条不经过任何非线性变换的'恒等捷径'，"
         "反向传播时梯度可以直接沿着这条捷径回传到浅层，避免深层网络出现梯度消失问题，使得几十甚至"
         "上百层的Transformer可以被稳定训练；同时它也让每个子层只需要学习'相对于输入的增量修正'而"
         "不是从头拟合整个变换，优化难度更低。",
       keys=("恒等捷径", "梯度消失", "Sublayer", "增量修正"),
       follow_ups=("残差连接和LayerNorm的相对顺序(pre-norm/post-norm)为什么会影响训练稳定性？", "如果去掉残差连接，深层Transformer会出现什么问题？")),
]


def _self_test() -> None:
    assert 9 <= len(BANK) <= 16, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-trf-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    assert quiz(BANK, cat=CAT) == BANK
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_transformer: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
