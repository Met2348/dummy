"""分布式训练 八股问答库（约 14 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "分布式训练"

BANK: list[QA] = [
    QA(
        id="ai-dist-01", cat=CAT,
        q="数据并行(DP)和模型并行(张量并行TP/流水并行PP)有什么区别？各自适用什么场景？",
        a=(
            "数据并行是把同一份完整模型复制到每张GPU上，不同GPU吃不同的数据分片（mini-batch的子集），"
            "前向/反向各自独立计算梯度，然后通过all-reduce同步梯度再统一更新参数——它解决的是"
            "\"数据吃得快\"的问题，前提是单卡能放下完整模型。模型并行则是把同一个模型本身切开放到不同"
            "GPU上，用于单卡放不下模型的场景，又分两种：张量并行(TP)把单层内部的矩阵乘法按行或列切分"
            "到多张卡，每次前向/反向都需要卡间通信（如all-reduce），通信频繁但显存节省直接，通常用于"
            "同一节点内高带宽NVLink互联的少数几张卡；流水并行(PP)把模型按层切成几段(stage)，不同stage"
            "放在不同GPU上，数据像流水线一样按micro-batch流过，通信量小（只传递激活值），适合跨节点"
            "扩展，但存在\"气泡\"(bubble)导致的空闲等待。实践中常把DP+TP+PP组合成3D并行：节点内TP、"
            "跨节点PP、最外层DP。"
        ),
        keys=("数据并行", "张量并行", "流水并行", "all-reduce", "气泡", "3D并行"),
        follow_ups=(
            "TP和PP的通信量谁更大？为什么TP要求节点内高带宽互联？",
            "3D并行里三者的切分顺序/优先级一般怎么定？",
        ),
    ),
    QA(
        id="ai-dist-02", cat=CAT,
        q="DeepSpeed 的 ZeRO 分为 stage 1/2/3，分别切分训练状态里的哪些部分？",
        a=(
            "ZeRO(Zero Redundancy Optimizer)的核心思想是训练状态里的优化器状态、梯度、参数在数据并行"
            "的每张卡上其实都是完全冗余的一份拷贝，ZeRO把这些状态按数据并行度切分，每张卡只保留自己的"
            "一份分片而不是完整拷贝。Stage 1(Pos)只切分优化器状态（Adam的一阶矩、二阶矩以及fp32主权重），"
            "这部分在混合精度训练下往往是显存占用最大头，切分后能大幅省显存又几乎不增加通信量。"
            "Stage 2(Pos+g)在stage 1基础上再把梯度也切分，每张卡只保留和自己负责的优化器状态对应的那部分"
            "梯度，通信量和标准数据并行的all-reduce基本相当，显存进一步下降。Stage 3(Pos+g+p)连模型参数"
            "本身也切分，每张卡平时只存参数的一个分片，前向/反向需要用到某层参数时再临时all-gather回完整"
            "参数、用完就释放，显存随并行度近似线性下降，代价是通信量明显增加（参数需要反复收集），对"
            "互联带宽要求更高。三个stage是层层递进的关系，stage数字越大显存越省但通信开销越大。"
        ),
        keys=("优化器状态", "梯度", "参数", "Pos", "线性下降", "冗余"),
        follow_ups=(
            "ZeRO-3和张量并行(TP)有什么区别？都是切分参数为什么效果不一样？",
            "ZeRO-Offload/ZeRO-Infinity是在stage基础上做了什么？",
        ),
    ),
    QA(
        id="ai-dist-03", cat=CAT,
        q="DeepSpeed 和 Megatron-LM 有什么区别？各自的定位是什么？",
        a=(
            "两者不是竞争关系，而是经常搭配使用、各自解决不同维度的问题。DeepSpeed是微软的训练加速库，"
            "核心卖点是ZeRO系列显存优化（把优化器状态/梯度/参数切分到数据并行的各张卡上）以及配套的"
            "CPU/NVMe offload（ZeRO-Offload、ZeRO-Infinity），目标是让更大的模型能在有限显卡数量下训练"
            "起来，同时也提供流水并行、MoE训练等能力，使用门槛相对低，接入已有PyTorch模型改动较小。"
            "Megatron-LM是NVIDIA的框架，强项是张量并行和序列并行的高效实现，针对Transformer结构做了"
            "专门的算子和通信优化，在节点内张量并行的效率上通常被认为是业界最优实现，但要求按Megatron"
            "的方式重写模型层（如把Attention/MLP换成它的并行层），接入成本更高。实践中大规模预训练"
            "（如Megatron-Turing NLG 530B）常把二者结合：Megatron-LM负责节点内的张量并行+跨节点流水并行"
            "把模型立起来，DeepSpeed在最外层数据并行维度上叠加ZeRO做显存优化，即\"Megatron-DeepSpeed\""
            "的3D并行组合方式。"
        ),
        keys=("ZeRO", "张量并行", "显存优化", "3D并行", "Megatron-Turing"),
        follow_ups=(
            "如果只训一个能塞进单卡的模型，还需要用这两个框架吗？",
            "Megatron-Core和早期Megatron-LM有什么演进关系？",
        ),
    ),
    QA(
        id="ai-dist-04", cat=CAT,
        q="训练一个模型时显存占用具体由哪几部分组成？用Adam优化器时大致是参数量的多少倍？",
        a=(
            "训练显存主要由四块组成：①模型参数本身；②梯度，和参数同精度同大小；③优化器状态，Adam"
            "需要额外存一阶矩(momentum)和二阶矩(variance)，通常用fp32保存以保证数值稳定，即便模型本身"
            "用fp16/bf16训练，优化器状态这两份fp32数据也各是参数量的4倍字节数；④激活值(activation)，"
            "训练时反向传播要用到前向过程中每一层的中间输出，层数越多、batch/序列长度越大，激活值显存"
            "越大，且不像前三项是固定量，是随batch size和seq_len线性增长的。粗略估算：若参数量为P，"
            "混合精度训练下，参数fp16占2P字节，梯度fp16占2P字节，优化器状态（fp32一阶矩+fp32二阶矩+"
            "一份fp32主权重）约12P字节，三项加起来大约16P字节，也就是差不多\"参数量的16倍字节数\"这个"
            "数量级（这也是为什么\"训练显存远大于推理显存\"、优化器状态往往是显存里最大的一块，几倍于"
            "参数本身），激活值另算，量级取决于具体配置，可以用gradient checkpointing去换。"
        ),
        keys=("梯度", "优化器状态", "一阶矩", "二阶矩", "激活值", "16"),
        follow_ups=(
            "为什么优化器状态要用fp32存而不是直接用fp16？",
            "激活值显存和batch size/序列长度是什么关系？",
        ),
    ),
    QA(
        id="ai-dist-05", cat=CAT,
        q="gradient checkpointing（梯度检查点/激活重计算）是怎么用计算换显存的？",
        a=(
            "标准反向传播需要保留前向过程中每一层的激活值，用于计算该层的梯度，层数越多、这部分显存"
            "越大。gradient checkpointing的做法是：前向时只保留少数几个\"检查点\"位置的激活值（比如"
            "每隔几层存一次），中间层的激活值算完就直接丢弃不保留；到反向传播需要用某段中间激活值时，"
            "从最近的检查点重新做一次前向计算，现算现用，用完再丢。这样显存占用从\"和层数成正比\"降到"
            "大致\"和检查点数量成正比\"，代价是反向传播阶段多做了一次局部前向计算，通常额外增加约30%"
            "左右的计算时间，但换来的显存节省往往是数倍甚至能让原本OOM的大模型/大batch训练跑起来，是"
            "训练显存紧张时最常用、性价比很高的手段之一。"
        ),
        keys=("激活值", "前向计算", "检查点", "显存", "计算时间"),
        follow_ups=(
            "检查点应该放在哪些位置比较划算？",
            "gradient checkpointing和ZeRO是否可以同时叠加使用？",
        ),
    ),
    QA(
        id="ai-dist-06", cat=CAT,
        q="MoE 模型的专家并行(Expert Parallelism)是怎么把专家分布到不同GPU上的？为什么会有"
           "all-to-all通信开销？",
        a=(
            "MoE模型里每层有多个专家(expert)，但每个token只会被路由到其中的top-k个专家，专家并行的"
            "做法是把不同专家分别放到不同GPU上（比如8个专家分到8张卡，每卡一个专家），而不是每张卡都"
            "放全部专家。前向时的流程是：先在每张卡上算出该卡上token的路由决策（去哪些专家），然后"
            "需要把每个token的隐藏状态发送到它被路由到的那张专家所在的卡上——这一步因为每个token去"
            "哪张卡是不确定的、且各卡收发的数据量不对齐，只能用all-to-all通信原语（每张卡都要给其它"
            "所有卡发数据、也从其它所有卡收数据）来实现这次\"洗牌\"；专家在本地对收到的token做计算后，"
            "还要再做一次all-to-all把结果送回token原来所在的卡，才能继续走后面的层。这两次all-to-all"
            "(dispatch和combine)是MoE专家并行的主要通信开销来源，实测里all-to-all能占到整个MoE层耗时的"
            "30%-70%不等，是MoE训练/推理里除了计算本身外最主要的性能瓶颈，常见优化手段包括限制专家路由"
            "的跨节点范围（如DeepSeek限制专家只分布在少数几个节点内、优先用节点内高带宽NVLink）、通信"
            "与计算overlap等。"
        ),
        keys=("专家", "路由", "all-to-all", "token", "dispatch", "瓶颈"),
        follow_ups=(
            "专家并行和张量并行/数据并行是什么关系，能不能叠加？",
            "如果某个专家被路由到特别多token（负载不均衡）会发生什么？怎么缓解？",
        ),
    ),
    QA(
        id="ai-dist-07", cat=CAT,
        q="分布式训练里常见的通信原语all-reduce、all-gather、reduce-scatter分别是什么？",
        a=(
            "这三个是集合通信(collective communication)里最常用的原语。all-reduce：每张卡都有一份数据"
            "（比如各自算出的梯度），all-reduce把所有卡的数据做一次归约操作（通常是求和，比如梯度求和"
            "再除以卡数取平均），然后把归约后的结果同步广播回每一张卡，结果是操作完之后每张卡都拥有"
            "完整且相同的归约结果——数据并行梯度同步用的就是它。all-gather：每张卡各自只有整体数据的"
            "一部分（比如ZeRO-3里每卡只有参数的一个分片），all-gather把所有卡的分片收集拼接起来，让"
            "每张卡都得到完整的数据，但不做归约计算，只是\"收集拼全\"。reduce-scatter：是all-reduce的"
            "\"一半\"：先做归约（比如求和），但归约后的完整结果不会广播给所有卡，而是按分片切开，每张"
            "卡只留自己该负责的那一部分——ZeRO-2/3的梯度同步就是用reduce-scatter替代完整的all-reduce，"
            "配合每卡只保留自己那份梯度分片，省下保存完整梯度的显存。工程上有个常见等式：all-reduce的"
            "通信量约等于reduce-scatter+all-gather两步的通信量之和，很多框架就是把all-reduce拆成这两步"
            "分别优化实现的。"
        ),
        keys=("all-reduce", "all-gather", "reduce-scatter", "归约", "广播", "分片"),
        follow_ups=(
            "为什么说all-reduce约等于reduce-scatter加all-gather？",
            "ring all-reduce的通信复杂度和卡数是什么关系？",
        ),
    ),
    QA(
        id="ai-dist-08", cat=CAT,
        q="张量并行(Tensor Parallelism)具体是怎么把一层内部的矩阵乘法切开的？",
        a=(
            "以Transformer的MLP层为例（先升维再降维的两个线性层），Megatron-LM的经典切法是\"列切+行切\""
            "配合：第一个线性层（升维，X乘W1）按列切分W1，每张卡只算列的一部分，各卡独立算完不需要通信"
            "（输出是按列拆开的）；紧接着的激活函数（如GELU）也可以逐列独立做，不需要通信；第二个线性层"
            "（降维，乘W2）按行切分W2，正好接上第一层按列拆开的输出，每张卡用自己那部分做局部矩阵乘，"
            "得到的是\"部分和\"，最后需要一次all-reduce把各卡的部分和加起来，才能得到正确的完整输出。"
            "Attention层的切法类似：按注意力头(head)切，每张卡负责一部分头，天然独立计算，最后同样在"
            "输出投影处需要一次all-reduce。所以一个Transformer block里通常前向要做2次all-reduce（MLP"
            "一次、Attention一次），反向再各做一次，通信频繁，因此张量并行强烈依赖同节点内NVLink这种"
            "高带宽低延迟互联，跨节点做TP效率会明显下降。"
        ),
        keys=("列切", "行切", "all-reduce", "注意力头", "局部矩阵乘"),
        follow_ups=(
            "为什么要列切配合行切，而不是两层都按同一种方式切？",
            "张量并行度一般不会设得很大（比如常见到8），是因为什么限制？",
        ),
    ),
    QA(
        id="ai-dist-09", cat=CAT,
        q="流水并行(Pipeline Parallelism)的\"气泡\"(bubble)问题是什么？怎么缓解？",
        a=(
            "流水并行把模型按层切成几个stage分别放在不同GPU上，一个batch的数据要依次流过stage 1->2->"
            "...->N才能完成一次前向，再反向流回来。如果把整个batch作为一个整体送进流水线，那么在第一个"
            "micro-step，只有stage 1在工作，其它stage都在空等；要等所有stage都被数据填满、又要在最后"
            "阶段依次排空，这段\"填充\"和\"排空\"时间里大部分GPU是闲置的，这就是气泡，气泡占比大致和"
            "stage数量成正比、和流水线深度成反比。缓解办法是把一个batch切成更多、更小的micro-batch，让"
            "多个micro-batch连续送入流水线，这样在稳定阶段各stage可以持续保持忙碌，气泡占比随micro-batch"
            "数量增多而下降（经典的GPipe调度）；进一步的优化是1F1B(one-forward-one-backward)调度，让每个"
            "stage交替做一次前向、一次反向而不是等所有micro-batch都做完前向才开始反向，这样能显著减少"
            "每个stage需要同时缓存的中间激活值数量，降低显存占用又不牺牲吞吐。"
        ),
        keys=("气泡", "micro-batch", "1F1B", "流水线", "闲置"),
        follow_ups=(
            "micro-batch切得越多越好吗？有什么代价？",
            "1F1B相比GPipe在显存上具体省在哪里？",
        ),
    ),
    QA(
        id="ai-dist-10", cat=CAT,
        q="3D并行（数据并行+张量并行+流水并行）在实践中一般怎么组合、怎么分配并行度？",
        a=(
            "3D并行的基本原则是\"让通信量大的并行方式用在通信带宽最高的范围内\"：张量并行(TP)通信最"
            "频繁（每层都要all-reduce），所以TP的并行度通常设置为和单节点内GPU数一致（比如单节点8卡就"
            "设TP=8），充分利用节点内的NVLink高带宽；流水并行(PP)通信量小（只传每个stage边界的激活值），"
            "适合跨节点扩展，PP的切分对应模型按层数分成的stage数；数据并行(DP)在最外层，通信是梯度的"
            "all-reduce（或ZeRO的reduce-scatter+all-gather），对带宽要求相对最低，可以跨越很多节点组。"
            "总GPU数=TP并行度 × PP并行度 × DP并行度。调参时的经验取舍是：先把TP设到节点内卡数上限"
            "（受限于单层参数/激活是否能切分均匀以及NVLink能承受的通信量），再用PP把模型撑到显存能放下，"
            "剩下的GPU资源全部划给DP去提升数据吞吐，DP维度上通常还会叠加ZeRO（一般是stage 1）进一步省"
            "优化器状态显存。"
        ),
        keys=("TP", "PP", "DP", "NVLink", "节点内", "跨节点"),
        follow_ups=(
            "为什么TP并行度一般不会超过单节点GPU数？",
            "如果显存已经够用，是不是就不需要PP，只用TP+DP就好？",
        ),
    ),
    QA(
        id="ai-dist-11", cat=CAT,
        q="大模型训练里为什么普遍用混合精度(fp16/bf16)而不是直接fp32训练？fp16和bf16有什么区别？"
           "各自要注意什么？",
        a=(
            "混合精度训练的目的是用更少字节的浮点格式做大部分前向/反向计算，从而省显存、提速度（现代"
            "GPU的Tensor Core对fp16/bf16有专门加速），同时保留一份fp32的主权重和用fp32做优化器状态更新，"
            "以避免精度损失累积影响收敛。fp16（半精度）只有5位指数、10位尾数，表示范围窄，大模型训练中"
            "梯度/激活值容易出现数值下溢或上溢（变成0或inf），所以fp16训练通常需要配合loss scaling"
            "（把损失乘一个较大的系数放大梯度、更新前再除回来）来规避下溢问题。bf16(bfloat16)牺牲了尾数"
            "精度换取和fp32一样宽的指数位（8位），表示范围和fp32相当，基本不会出现fp16那种上溢/下溢"
            "问题，因此现代大模型预训练（如Llama系列）大多直接用bf16而不需要loss scaling，代价是bf16的"
            "尾数精度比fp16更粗糙，单次运算的相对误差略大，但对大模型训练的收敛影响总体可控。"
        ),
        keys=("fp16", "bf16", "loss scaling", "指数位", "下溢", "Tensor Core"),
        follow_ups=(
            "为什么优化器状态还要单独存一份fp32主权重，而不是直接用fp16/bf16的权重去更新？",
            "bf16和fp8比起来又是怎样的取舍？",
        ),
    ),
    QA(
        id="ai-dist-12", cat=CAT,
        q="序列并行(Sequence Parallelism)解决的是什么问题？和张量并行是什么关系？",
        a=(
            "张量并行只切分了Transformer里Attention和MLP这些矩阵乘法密集的部分，但LayerNorm、Dropout"
            "这类逐元素/逐token的操作在Megatron-LM的标准TP实现里其实是在每张卡上都重复计算完整序列的，"
            "这部分激活值显存并没有被TP切分掉，当序列长度(seq_len)变得很长时，这部分未被切分的激活值会"
            "成为显存瓶颈。序列并行的做法是把这些TP没有切分的逐token操作按序列维度切分到各张卡上，每张"
            "卡只处理序列的一部分，配合TP在Attention/MLP前后做必要的all-gather/reduce-scatter完成数据"
            "交接，这样整个Transformer block的激活值显存都能随TP并行度切分下去，而不只是Attention/MLP"
            "部分。这项技术在训练长上下文模型时尤其重要，通常和张量并行搭配使用、共享同一组通信组，不"
            "额外增加并行维度的GPU划分。"
        ),
        keys=("LayerNorm", "逐token", "激活值", "长上下文", "张量并行"),
        follow_ups=(
            "序列并行和Ring Attention/长上下文推理的序列切分是一回事吗？",
            "为什么普通TP切不掉LayerNorm的显存？",
        ),
    ),
    QA(
        id="ai-dist-13", cat=CAT,
        q="ZeRO-Offload和ZeRO-Infinity是做什么的？和ZeRO stage 1/2/3是什么关系？",
        a=(
            "ZeRO-Offload和ZeRO-Infinity是在ZeRO切分的基础上进一步\"往哪存\"的扩展，而不是新的切分维度。"
            "ZeRO-Offload在ZeRO-2的基础上，把原本要放在GPU显存里的优化器状态和梯度转移到CPU内存里存放，"
            "优化器的更新计算（Adam的逐元素运算）也搬到CPU上做，GPU只负责前向反向的矩阵运算，这样单卡"
            "能训练的模型规模显著变大，代价是CPU-GPU之间的数据搬运和CPU算力都会拖慢速度，适合\"卡不够"
            "多、但CPU内存够大\"的场景。ZeRO-Infinity是对ZeRO-3的进一步扩展，除了CPU内存，还能把参数/"
            "梯度/优化器状态按需offload到NVMe固态硬盘上，理论上把可训练模型规模的上限从\"GPU显存总和\""
            "和\"CPU内存总和\"进一步扩展到\"加上NVMe容量\"，可以在数量有限的GPU上训练万亿参数级别的"
            "模型，代价是NVMe的带宽/延迟比CPU内存更差，需要仔细的预取(prefetch)和通信-计算-IO三者的"
            "overlap才能不让IO成为瓶颈。"
        ),
        keys=("CPU", "NVMe", "offload", "ZeRO-2", "ZeRO-3", "预取"),
        follow_ups=(
            "offload到CPU/NVMe之后，训练速度会受到什么限制？",
            "什么场景下应该用ZeRO-Offload而不是直接加更多GPU？",
        ),
    ),
    QA(
        id="ai-dist-14", cat=CAT,
        q="训练大模型时通信和计算的overlap为什么重要？NVLink和InfiniBand在其中分别扮演什么角色？",
        a=(
            "分布式训练里每一步都需要通信（梯度同步的all-reduce、TP的all-gather/reduce-scatter、PP的"
            "激活值传递、MoE的all-to-all等），如果通信和计算严格串行执行（算完再等通信、通信完再算下"
            "一步），通信耗时会直接加到总训练时间上；overlap的做法是让通信在后台异步进行的同时，GPU继续"
            "做下一部分不依赖该通信结果的计算（比如反向传播时，先算出的层的梯度可以立刻开始all-reduce，"
            "同时继续反向计算更前面层的梯度），只要计算时间能盖住通信时间，通信开销就能被\"隐藏\"掉，"
            "不增加总耗时。互联硬件决定了能隐藏多少：NVLink是同一节点内GPU之间的高带宽低延迟直连（带宽"
            "通常是几百GB/s量级），适合张量并行这种通信频繁、单次数据量相对小但延迟敏感的场景；"
            "InfiniBand(IB)是跨节点之间的高速网络互联，带宽和延迟都不如NVLink，但覆盖范围能跨越整个"
            "集群，适合流水并行（通信量小）和数据并行/ZeRO（通信可以更容易和计算overlap、对延迟没那么"
            "敏感）这种跨节点场景。这也是3D并行里\"TP放节点内、PP和DP放跨节点\"这一设计原则背后的硬件"
            "依据。"
        ),
        keys=("overlap", "NVLink", "InfiniBand", "隐藏", "跨节点", "延迟"),
        follow_ups=(
            "如果通信时间比计算时间还长，overlap还能完全隐藏开销吗？",
            "为什么TP对通信延迟特别敏感，而DP相对没那么敏感？",
        ),
    ),
]


def _self_test() -> None:
    assert 10 <= len(BANK) <= 16, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-dist-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_distributed_training: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
