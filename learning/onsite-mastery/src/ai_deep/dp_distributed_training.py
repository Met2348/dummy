"""分布式训练深水区追问链（12 个 DeepPoint）。

覆盖数据并行/张量并行/流水线并行各自的通信瓶颈怎么profile、ZeRO-3的
all-gather/reduce-scatter具体时机、梯度累积和BatchNorm统计量的冲突、
显存估算公式在真实训练里为什么经常对不上、通信与计算重叠的具体工程手段，
以及FSDP/DeepSpeed实现差异、gradient checkpointing权衡、弹性容错、3D并行组合。

边界：不涉及MoE专家并行的路由/负载均衡（那是 moe_systems 类目的范围）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, drill, grade_chain  # noqa: E402

CAT = "分布式训练深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-ai-dist-01", cat=CAT,
        trigger="你提到用 DDP 做了数据并行训练。",
        chain=(
            ("DDP 的通信瓶颈具体发生在哪一步，通信量有多大？",
             "DDP 在反向传播过程中，每个 rank 算完梯度后要做一次 all-reduce（ring all-reduce）把所有 "
             "rank 的梯度做平均同步；每个 rank 的通信量约为 2*(N-1)/N*参数量*每元素字节数，这是 ring "
             "all-reduce 的经典通信量公式，总线带宽利用率与 rank 数 N 基本无关（带宽最优），但每一步的 "
             "延迟（latency）项会随 N 增大而增大。",
             ("ring all-reduce", "2*(N-1)/N", "带宽最优")),

            ("这个通信瓶颈实际部署中怎么 profile 出来，怎么确认它就是瓶颈？",
             "用 PyTorch Profiler 或 Nsight Systems 抓一个 training step 的 timeline，看 NCCL "
             "all-reduce kernel 在时间线上是不是和反向传播计算 kernel 有大段不重叠的空白；如果 "
             "all-reduce 耗时占比高且这段时间 GPU 计算利用率（SM occupancy）接近 0，说明通信没被计算掩盖，"
             "是真瓶颈；也可以对比关掉 all-reduce 的 dry run（纯计算时间）和实际 step 时间的差值，定量估计"
             "通信开销占比。",
             ("timeline", "SM occupancy", "dry run")),

            ("为什么 DDP 要用 bucketing（分桶）而不是等全部梯度算完再一次性 all-reduce？这样做的收益上限在哪？",
             "DDP 把参数按注册顺序切成若干 bucket，一个 bucket 的梯度全部就绪就立刻发起异步 all-reduce，"
             "不必等整个反向传播结束——这样通信可以和还没算完的更早层的反向计算重叠（overlap），理论收益"
             "上限是把通信完全隐藏在反向传播的计算时间之内，总 step 时间趋近于 max(计算时间, 通信时间) "
             "而不是两者之和；但收益受限于最后一个 bucket（通常是第一层的梯度）之后没有更多计算可以重叠，"
             "这部分通信是硬性暴露的。",
             ("bucket", "overlap", "max(计算时间, 通信时间)")),

            ("如果集群里某台机器网络抖动或算力稍弱（straggler），DDP 的同步 all-reduce 会怎样，你怎么定位是谁拖后腿？",
             "同步 all-reduce 要求所有 rank 都到达通信点才能完成，一个 straggler（网络抖动/温度墙降频/"
             "磁盘 IO 抢占 CPU）会让所有其他 rank 在通信点空等，整体 step 时间被拖到最慢 rank 的水平；"
             "定位方法是给每个 rank 加不进 all-reduce 的本地计时打点（纯前向+反向耗时）和进程级 system "
             "metrics（nvidia-smi 温度/功耗、网卡吞吐）按 rank 对比，如果通信本身的 all-reduce 耗时在所有 "
             "rank 上均匀增大，大概率不是某个 rank 慢而是网络拥塞；这是训练规模变大后经常被低估的稳定性"
             "问题，没有放之四海而皆准的定位公式，通常要结合基础设施监控逐步排除。",
             ("straggler", "本地计时打点", "网络拥塞")),
        ),
        pitfall="大多数人第2层开始编——只会说'用了profiler看了一下'，说不出具体看timeline上的什么信号"
                "（NCCL kernel和计算kernel是否重叠）；到第4层（straggler定位）基本没人有真实经验，容易"
                "编一个'肯定是网络问题'的过度自信答案。",
        real_world_link="learning/cluster-networking/src/allreduce_algos.py",
    ),

    DeepPoint(
        id="dp-ai-dist-02", cat=CAT,
        trigger="你提到用张量并行（TP）切了模型。",
        chain=(
            ("TP 具体在哪几个位置插入了 all-reduce/all-gather？通信量怎么算？",
             "以 Megatron 风格 TP 为例，一层 Transformer block 里插两次同步点：MLP 部分先列并行"
             "（column-parallel）切第一层线性层不需要通信，过完激活函数后第二层线性层做行并行"
             "（row-parallel），行并行的输出必须做一次 all-reduce 把各 TP rank 的部分和加起来；attention "
             "部分同理，QKV 投影列并行、输出投影行并行后再 all-reduce 一次。所以每个 TP group 每个 block "
             "前向要做 2 次 all-reduce、反向再 2 次，通信量与激活张量大小（batch*seq*hidden）成正比，而"
             "不是参数量。",
             ("列并行", "行并行", "激活张量")),

            ("为什么 TP 对通信带宽的要求比 DP 高得多，工程上一般怎么应对？",
             "DP 的 all-reduce 发生在反向传播结束、每个训练 step 只有一次（可以被 overlap 掉），而 TP 的 "
             "all-reduce 在每个 block 的前向和反向都要发生、频率高得多，且必须同步阻塞在原地等结果才能"
             "继续下一步计算，通信延迟直接串行叠加到 step 时间里；因此工程上 TP 几乎总是限制在单机内的 "
             "NVLink/NVSwitch 高带宽域内（几百 GB/s），很少跨机做 TP，跨机场景更依赖 PP 或 DP 来扩展。",
             ("阻塞", "串行叠加", "NVLink")),

            ("既然 TP 这么依赖高带宽域，为什么 Megatron 还要引入序列并行（sequence parallel）？它解决的是通信还是显存问题？",
             "序列并行解决的主要是显存问题而不是通信量本身：标准 TP 中 LayerNorm、Dropout 这些非矩阵乘"
             "操作没有被切分，每个 TP rank 都要完整保存这部分的激活，造成激活显存在 TP rank 之间被冗余"
             "复制；序列并行把这些操作按序列维度切分到不同 TP rank，配合把原本的 all-reduce 拆成 "
             "all-gather+reduce-scatter 的组合，通信总量基本不变，但每个 rank 只需要保存 1/TP 大小的"
             "LayerNorm/Dropout 激活，从而降低峰值显存，这也是它能和 TP 无缝拼接使用的原因。",
             ("冗余复制", "reduce-scatter", "峰值显存")),

            ("TP 的度数是不是切得越大越好？有没有一个明显的收益递减点，为什么？",
             "不是，TP 度数增大到超过单机 GPU 数（通常 8）之后就要跨机通信，带宽从 NVLink 的几百 GB/s "
             "骤降到跨机网络的几十 GB/s，通信延迟不再能被有效隐藏，收益会陡降甚至转负；即便在单机内，"
             "TP 度数越大，每个 block 里的 all-reduce 次数不变但每次通信涉及的 rank 数越多、单卡分到的"
             "矩阵乘越小，计算/通信比（compute-to-communication ratio）下降，GPU 利用率降低，所以实践中 "
             "TP 度数一般只设到与单机 GPU 数匹配，更大规模扩展交给 PP 和 DP，这个'节点边界'是经验规律，"
             "并不是有严格闭式解的最优值，不同硬件拓扑（NVSwitch 全互联 vs PCIe）会移动这个临界点。",
             ("跨机通信", "compute-to-communication", "经验规律")),
        ),
        pitfall="第1层经常记不全两次all-reduce的具体位置（漏掉attention那一次或搞混列/行并行谁需要通信）；"
                "第3层容易把序列并行说成'减少通信量'，其实主收益是显存。",
        real_world_link="learning/scaling-infra/src/megatron_tp_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-03", cat=CAT,
        trigger="你提到用流水线并行（PP）切了模型。",
        chain=(
            ("PP 的气泡（bubble）率怎么算，受哪些参数影响？",
             "标准 GPipe 式 PP 的气泡率公式约为 (p-1)/m，其中 p 是流水线 stage 数、m 是一个 batch 切成的 "
             "micro-batch 数；直觉是每个 stage 要等前一个 stage 先产出、最后一个 micro-batch 算完后还要"
             "等所有 stage 排空（drain），stage 数越多、micro-batch 切得越少，气泡占比越高，因此要让 m "
             "远大于 p 才能把气泡摊薄到可接受水平。",
             ("(p-1)/m", "micro-batch", "drain")),

            ("1F1B 调度相比 GPipe 具体改进了什么？它降低的是气泡率还是别的？",
             "1F1B（one-forward-one-backward）调度并不降低理论气泡率（仍是 (p-1)/m 这个量级），它改进的"
             "是显存：GPipe 要求先做完所有 micro-batch 的前向、缓存所有中间激活才能开始反向，峰值激活"
             "显存随 micro-batch 数 m 线性增长；1F1B 让每个 stage 一算完一个 micro-batch 的前向就尽快"
             "安排它的反向，交替执行前向和反向，使得同一时刻需要缓存的未完成 micro-batch 数量被限制在 "
             "stage 数 p 左右，峰值激活显存不再随 m 增长，这才是 1F1B 真正的收益。",
             ("1F1B", "峰值激活显存", "不再随 m 增长")),

            ("如果各 stage 之间计算量不均衡（load imbalance），气泡和整体吞吐会受什么影响？为什么 interleaved schedule 能缓解？",
             "气泡公式 (p-1)/m 假设所有 stage 计算耗时相等；如果切分不均衡（load imbalance），整体吞吐由"
             "最慢的那个 stage 决定，其余更快的 stage 会有额外的空闲等待，实际气泡比理论值更差；"
             "interleaved（交错）1F1B 调度让每个物理 GPU 负责多个不连续的虚拟 stage，将一次大的、可能"
             "不均衡的切分打散成更多更细粒度的小块，细粒度切分更容易通过调整每个虚拟 stage 的层数来找到"
             "均衡点，代价是每个虚拟 stage 之间多了额外的激活传递通信，通信次数增加换取气泡进一步降低。",
             ("load imbalance", "最慢的那个 stage", "interleaved")),

            ("PP 的 stage 之间传递的是什么？这部分通信为什么通常不是瓶颈，但在什么条件下会变成瓶颈？",
             "stage 之间只传递切分点处的激活张量（activation，形状通常是 batch*seq*hidden）和对应梯度，"
             "数据量远小于 TP 需要 all-reduce 的中间结果、更远小于 DP 需要同步的全部参数梯度，而且 PP 的"
             "stage 间通信天然只需要点对点（P2P send/recv），可以用不同硬件链路与 TP 的高带宽域错开，通常"
             "不是主要瓶颈；但当序列长度或 batch 很大导致单次激活张量本身很大、又叠加了跨机网络带宽受限"
             "（而不是 NVLink）时，P2P 传输时间可能追上甚至超过单个 stage 的计算时间，气泡里等待激活到达"
             "的部分就会显著增大。",
             ("点对点", "P2P", "跨机网络带宽受限")),
        ),
        pitfall="第1层经常记错气泡公式或说不出m和p的关系；第2层容易把1F1B和GPipe的区别答成'降低了气泡'，"
                "其实核心收益是显存不随micro-batch数增长。",
        real_world_link="learning/scaling-infra/src/pipeline_parallel_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-04", cat=CAT,
        trigger="你提到用了 ZeRO-3（或 FSDP）训练大模型。",
        chain=(
            ("ZeRO-3 具体在训练的哪几个时间点触发 all-gather、哪几个时间点触发 reduce-scatter？",
             "ZeRO-3 把参数、梯度、优化器状态都按 rank 切分存储，平时每个 rank 只持有自己那一份分片；"
             "前向传播到某一层之前，该层需要先对所有 rank 做一次 all-gather 把完整参数临时重组出来，算完"
             "这一层前向后立刻释放重组出来的完整参数、只留分片；反向传播同理，进入某层反向前先 "
             "all-gather 出完整参数用于计算梯度，算完梯度后对该层梯度做一次 reduce-scatter，把梯度按 "
             "rank 切分累加，每个 rank 只留下属于自己那部分梯度分片用于后续优化器更新。",
             ("all-gather", "reduce-scatter", "临时重组")),

            ("既然每层都要现拼现拆完整参数，这样做通信量比 ZeRO-1/2 大多少？为什么还值得？",
             "ZeRO-1 只切分优化器状态，ZeRO-2 在此基础上切分梯度，通信量与标准 DP 的 all-reduce 同量级"
             "（约 2Ψ，Ψ 为参数量）；ZeRO-3 因为连参数本身都切分了，前向反向各多出一次 all-gather，理论"
             "通信量约为标准 DP 的 1.5 倍（约 3Ψ），换来的收益是显存占用与 rank 数 N 近似成反比而不是像 "
             "ZeRO-1/2 那样只有优化器状态/梯度部分被均分——这是用通信量换取能训练远超单卡显存容量模型的"
             "核心权衡，在显存是硬约束而不是速度是硬约束时，这个交易是值得的。",
             ("1.5 倍", "3Ψ", "显存是硬约束")),

            ("既然每层都要现拼参数，这个开销为什么在实践中没有想象中那么致命？靠什么工程手段掩盖掉？",
             "关键在于 prefetch（预取）和 overlap：实现会在当前层还在计算的时候，提前异步发起下一层参数"
             "的 all-gather，让通信和计算在时间上重叠；同时不需要保留全部层的完整参数，一份显存缓冲区"
             "滚动复用于正在算的这一层和正在预取的下一层，所以额外显存开销是 O(1) 个层的大小而不是 "
             "O(层数)；如果 prefetch depth 设置得当，大部分 all-gather 延迟能被计算时间掩盖，只有第一层"
             "和最后一层的通信难以被完全掩盖。",
             ("prefetch", "滚动复用", "O(1) 个层")),

            ("如果你实测发现 ZeRO-3 训练比理论通信量预估慢很多，你会怀疑是哪几类原因，怎么排查？",
             "首先怀疑 prefetch 没有真正生效——可能是每层计算时间太短，来不及把下一层 all-gather 完全"
             "藏进去，导致通信暴露；其次怀疑通信本身是小消息瓶颈——ZeRO-3 是逐层做 all-gather，如果每层"
             "参数量不大，通信是许多小消息而非少数大消息，NCCL 在小消息场景下有效带宽远低于峰值带宽，"
             "可以用 nsys 抓每次 all-gather 的实际耗时和理论值对比；还可能是 activation checkpointing "
             "与 ZeRO-3 的 prefetch 调度冲突（重计算打乱了正常的层序），需要结合具体框架的调度日志定位，"
             "没有一个通用的单一根因。",
             ("prefetch 没有真正生效", "小消息瓶颈", "activation checkpointing")),
        ),
        pitfall="第2层最容易翻车——很多人张口就说ZeRO-3通信量是'3倍'，说不出1.5Ψ vs 3Ψ这个更准确的比较"
                "基准，也说不清和ZeRO-1/2比到底贵在哪一步；第4层基本没人能给出结构化排查思路。",
        real_world_link="learning/scaling-infra/src/fsdp_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-05", cat=CAT,
        trigger="你说显存不够所以用了梯度累积（gradient accumulation）。",
        chain=(
            ("梯度累积和真正的大 batch 训练在数学上等价吗？哪里可能不等价？",
             "对梯度本身是等价的——把多个 micro-batch 的 loss 梯度累加/平均，数学上等同于对一个大 batch "
             "算一次梯度；但对任何依赖当前 batch 统计量的算子（最典型是 BatchNorm）不等价，BatchNorm 在"
             "每个 micro-batch 上是用这个 micro-batch 自己的均值/方差做归一化，梯度累积并不会把多个 "
             "micro-batch 的统计量合并成一个大 batch 的统计量，相当于模型实际看到的是小 batch 的归一化"
             "配大 batch 的梯度更新，和真正大 batch 训练里两者都是大 batch 是不同的两个训练动态。",
             ("BatchNorm", "小 batch 的归一化", "不等价")),

            ("这个不等价具体会造成什么后果？有没有办法弥补？",
             "后果是训练/验证阶段 BatchNorm 的 running mean/var 统计口径出现偏差，尤其在 micro-batch 本"
             "身很小时，小 batch 统计量噪声很大，归一化后的激活分布不稳定，可能造成收敛变慢或最终精度"
             "下降，这个问题在 CV 模型（ResNet 等）上更明显；弥补方式包括换成不依赖 batch 统计量的归一化"
             "（LayerNorm/GroupNorm/InstanceNorm），或者用 SyncBN 跨 GPU 同步统计量（但这解决的是跨卡不"
             "同步，不解决同卡内跨 micro-batch 不同步），或者尽量保证单个 micro-batch 本身不要太小。",
             ("running mean/var", "SyncBN", "GroupNorm")),

            ("为什么这个问题在今天的 LLM 训练里反而很少被提起？",
             "因为主流 LLM（Transformer 系）几乎不用 BatchNorm，普遍用 LayerNorm 或 RMSNorm，这两种归一"
             "化的统计量是对每个样本自己的特征维度求均值方差，与 batch 维度无关、天然不受切分 "
             "micro-batch 影响，所以梯度累积在 LLM 训练里几乎是免费的技巧；这个坑主要出现在还包含 "
             "BatchNorm 的场景——比如多模态模型里的 CNN 视觉编码器继续用梯度累积去模拟大 batch 时，容易"
             "被忽视，因为做 NLP 出身的人对 BatchNorm 这个坑天然缺乏警觉。",
             ("LayerNorm", "与 batch 维度无关", "CNN 视觉编码器")),
        ),
        pitfall="很多人只知道梯度累积'省显存'，完全没意识到它和BatchNorm有冲突，一问到'哪里不等价'就卡住；"
                "即使知道有冲突，也说不出为什么LLM训练基本不受影响。",
        real_world_link="learning/scaling-infra/src/mixed_precision_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-06", cat=CAT,
        trigger="你算过一次训练大概要多少显存吗？",
        chain=(
            ("训练显存的标准估算公式是什么，每一项怎么来的？",
             "标准公式把显存分四块：参数本身（P*每参数字节数）、梯度（P*bytes，通常和参数同精度或更高）、"
             "优化器状态（Adam 需要额外保存一阶矩 m 和二阶矩 v，通常用 fp32 保存）、以及激活值（activation，"
             "与 batch size、序列长度、层数、是否开 gradient checkpointing 强相关，不是参数量的简单函数）；"
             "混合精度+Adam 的常见估算是每个参数约 16 字节左右（fp16 权重2+fp16梯度2+fp32主权重4+fp32一"
             "阶矩4+fp32二阶矩4），再加上激活值这块通常是估算里最大的变数。",
             ("优化器状态", "一阶矩 m", "激活值")),

            ("实际训练里显存占用为什么经常比这个公式算出来的数字高不少？",
             "几个常见被漏算的部分：1) CUDA context 和 cudnn/nccl 的固定开销，每个进程启动就占用 1-2GB "
             "左右；2) 通信用的临时 buffer，比如 ZeRO/FSDP 做 all-gather 时需要额外分配一块和被 "
             "all-gather 对象同样大小的连续显存作为接收缓冲区；3) PyTorch caching allocator 造成的显存"
             "碎片（fragmentation）——nvidia-smi 看到的已分配显存往往显著小于已预留（reserved）显存；"
             "4) 激活值的估算最容易失真，因为它对 batch size、序列长度是非线性敏感的，很多人只按参数量"
             "的经验倍数估算而不是真的按具体 shape 算。",
             ("CUDA context", "caching allocator 造成的显存碎片", "激活值的估算最容易失真")),

            ("你怎么定位真实训练里显存爆掉，到底是这四块里的哪一块出了问题？",
             "用 torch.cuda.memory_summary() 或 memory snapshot 工具对比实际分配曲线和预期：如果峰值"
             "出现在 forward 刚结束、backward 刚开始的瞬间且和 batch size 强相关，大概率是激活值估算不"
             "准；如果 allocated 和 reserved 差距很大，是碎片问题，可以尝试调大 "
             "PYTORCH_CUDA_ALLOC_CONF 的相关参数或换用 expandable_segments；如果是 ZeRO/FSDP 场景，峰值"
             "往往出现在 all-gather 发生的瞬间，需要检查 prefetch 窗口是不是设得太大。",
             ("memory_summary", "allocated 和 reserved 差距很大", "prefetch 窗口")),

            ("如果理论公式和实测值差距超过 20-30%，你有多大把握找到根因？这类问题有没有普适的解法？",
             "老实说，把握程度取决于问题类型——碎片和固定开销这类是可以用工具精确定位的，通常能在几次"
             "迭代内找到根因；但激活值相关的显存问题，尤其是涉及第三方 kernel 内部分配的临时显存，往往"
             "要靠对着 profiler 的分配调用栈（allocation stack trace）一行行排查，不是一个通用公式能"
             "覆盖的；没有普适解法，估算公式只能给出量级参考而不是精确预测，这也是为什么大规模训练团队"
             "通常会先用小规模配置实测显存曲线再外推，而不是完全依赖纸面公式做容量规划。",
             ("分配调用栈", "量级参考而不是精确预测", "小规模配置实测")),
        ),
        pitfall="绝大多数人只会背四项公式，第2层问到'为什么实测更高'基本说不出CUDA context/碎片/通信"
                "buffer这些具体项，只会含糊说'显存碎片化'；第4层几乎没人诚实承认'不是所有情况都能定位'。",
        real_world_link="learning/scaling-infra/src/capstone_train_estimator.py",
    ),

    DeepPoint(
        id="dp-ai-dist-07", cat=CAT,
        trigger="你提到训练里做了通信和计算的 overlap。",
        chain=(
            ("具体有哪些工程手段能让通信和计算重叠？",
             "常见手段包括：1) 梯度分桶+反向过程中异步发起 all-reduce（DDP 的标准做法），不必等整个 "
             "backward 完成；2) 用独立的 CUDA stream 跑 NCCL 通信，和计算 stream 并行执行，通过 stream "
             "间的事件（event）做必要的同步点而不是全局阻塞；3) ZeRO-3/FSDP 的参数预取（prefetch 下一层 "
             "all-gather）；4) 计算图重排，把不依赖当前通信结果的计算提前调度到通信等待期间执行。",
             ("独立的 CUDA stream", "梯度分桶", "prefetch")),

            ("怎么验证 overlap 真的发生了，而不是代码写了 overlap 但实际还是串行？",
             "用 Nsight Systems 或 PyTorch Profiler 抓 timeline，NCCL 通信 kernel 应该出现在和计算 "
             "kernel 不同但时间区间重叠的 stream 轨道上；如果发现 NCCL kernel 总是紧跟在某个计算 kernel "
             "完成之后才开始、期间 GPU 有明显空闲 gap，说明存在一个隐藏的同步点（常见原因是某处调用了 "
             ".item()、.cpu() 或没必要的 torch.cuda.synchronize()），这类伪 overlap 在代码 review 时很"
             "难发现，必须靠实测 timeline 才能确认。",
             ("Nsight Systems", "GPU 有明显空闲 gap", ".item()")),

            ("overlap 的收益有没有理论上限？什么情况下即使做了完美的 overlap，效果也有限？",
             "理论上限是总 step 时间趋近于 max(计算时间, 通信时间)，而不是两者之和；但当通信时间本身就"
             "大于计算时间时（比如 TP 跨机、或者模型很小而通信频繁），即便完美重叠，step 时间的下限也被"
             "通信时间卡住，这时候单纯做 overlap 工程已经到顶，必须从降低通信量本身入手；另外 overlap 需"
             "要额外的 SM 资源来同时驱动通信 kernel 和计算 kernel，如果 GPU 本身计算已经把 SM 占满，通信 "
             "kernel 抢占 SM 反而可能拖慢计算本身，不是纯粹的免费 overlap。",
             ("max(计算时间, 通信时间)", "通信时间本身就大于计算时间", "抢占 SM")),
        ),
        pitfall="第2层是重灾区——很多人会说'用了async all-reduce'就默认overlap生效，从没用profiler验证"
                "过，问到'怎么确认'就语塞；第3层很少有人提到通信抢占SM资源这个反直觉的点。",
        real_world_link="learning/cluster-networking/src/capstone_cluster_sim.py",
    ),

    DeepPoint(
        id="dp-ai-dist-08", cat=CAT,
        trigger="你提到训练用了混合精度（bf16/fp16）。",
        chain=(
            ("混合精度训练下，模型的哪几份拷贝同时存在，各自是什么精度？",
             "典型的 fp16 混合精度（loss scaling 方案）同时维护：fp16 的权重拷贝（用于前向反向的实际"
             "计算）、fp32 的主权重拷贝（master weights，优化器实际更新的对象，避免小学习率更新量在 "
             "fp16 下下溢为 0）、fp16 的梯度、以及 Adam 优化器的一阶矩 m 和二阶矩 v（通常保存为 fp32 以"
             "保证数值稳定性）；所以每个参数除了 1 份 fp16 权重外，还额外背负 1 份 fp32 主权重+2 份 "
             "fp32 动量项。",
             ("fp32 的主权重拷贝", "一阶矩 m", "二阶矩 v")),

            ("bf16 训练是不是就不需要 fp32 master weights 了？为什么？",
             "这是个有争议、容易搞混的点：bf16 和 fp32 有相同的指数位宽度（动态范围一致），不容易发生 "
             "fp16 那种因为指数位窄导致的溢出/下溢问题，所以很多实现确实省去了 loss scaling；但 bf16 "
             "的尾数位比 fp32 少得多，单次小梯度更新量在只用 bf16 权重直接累加时仍可能被舍入误差吃掉"
             "（尤其是训练后期学习率很小时），所以严谨的实现即使用 bf16 计算，仍然保留 fp32 master "
             "weights 用于实际的参数更新累加，只是不再需要 loss scaling 这一步，这是两个独立的问题"
             "（动态范围 vs 累加精度）容易被混为一谈。",
             ("尾数位比 fp32 少得多", "舍入误差", "两个独立的问题")),

            ("把这些精度细节都考虑进去，每个参数实际占用的总字节数是多少？这对显存估算公式的准确性有什么影响？",
             "一个典型配置（bf16 权重2字节+bf16 梯度2字节+fp32 主权重4字节+fp32 一阶矩4字节+fp32 二阶矩"
             "4字节）合计约 16 字节/参数，如果只按参数量*2字节（bf16）这种粗糙估算，会把优化器状态和 "
             "master weight 这部分显存完全漏掉，实际显存需求可能是粗糙估算的 6-8 倍；单纯说用了 bf16 "
             "所以显存减半是一个常见误区——减半的只是权重和梯度这两项，优化器状态这块占比通常更大且不受"
             "权重精度选择直接影响，除非换用像 8-bit Adam 这类专门压缩优化器状态的技术。",
             ("合计约 16 字节", "常见误区", "8-bit Adam")),
        ),
        pitfall="第2层是典型翻车点——很多人分不清'bf16不需要loss scaling'和'bf16不需要fp32 master "
                "weights'是两码事，把两者混为一谈；第3层容易只记住'混合精度省显存'而说不出优化器状态"
                "占大头这个事实。",
        real_world_link="learning/scaling-infra/src/mixed_precision_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-09", cat=CAT,
        trigger="PyTorch FSDP 和 DeepSpeed ZeRO-3 看起来做的是同一件事，你觉得区别在哪？",
        chain=(
            ("两者在'怎么切分'这个粒度上有什么实现差异？",
             "两者核心思想相同（参数/梯度/优化器状态按 rank 切分，用时现拼），但切分和触发通信的粒度不"
             "同：PyTorch FSDP 以 module（nn.Module 子树）为单位做 wrap，由 auto_wrap_policy 决定哪些"
             "子模块各自成为一个 FSDP unit，每个 unit 在前向/反向各自触发一次 all-gather/"
             "reduce-scatter；DeepSpeed ZeRO-3 则是在更细的 parameter 级别注册 hook，不严格依赖模块"
             "边界，理论上调度更灵活但也更依赖框架自动分析计算图。",
             ("auto_wrap_policy", "FSDP unit", "parameter 级别注册 hook")),

            ("这个粒度差异在实际训练里会带来什么可观察的性能差异？",
             "FSDP 按 module 分片，如果某个 wrap 单元设置得太粗，会导致单次 all-gather 的消息很大、"
             "prefetch 的粒度也很粗，峰值显存和通信延迟都可能变差；调得太细则通信次数增多、每次都是小"
             "消息，同样吃小消息带宽利用率低的亏；DeepSpeed ZeRO-3 因为不严格按模块边界切，理论上能更细"
             "粒度地 control，但代价是配置项更依赖手动调优，两者本质上是在同一个消息大小 vs 通信次数权"
             "衡曲线上，只是暴露的调优旋钮位置不同。",
             ("wrap 单元设置得太粗", "小消息带宽利用率低", "调优旋钮位置不同")),

            ("如果一个团队从 DeepSpeed ZeRO-3 切换到 FSDP（或反过来），收敛结果/最终精度理论上应该完全一致吗？你有多大把握？",
             "理论上数学操作等价，收敛结果应该一致，但实践中不完全有把握——两者在混合精度实现细节（比如 "
             "master weight 怎么存、reduce 的顺序是否引入不同的浮点求和误差）、默认的通信精度、以及各自"
             "默认的初始化/checkpoint 格式转换上可能有细微差别，浮点非结合性（non-associativity）会导致"
             "不同实现的 loss curve 在小数点后几位有差异，大多数情况这个差异在噪声范围内，但不能保证 "
             "100% 数值一致，这也是很多团队迁移框架后要重新跑一次小规模对齐实验的原因。",
             ("浮点非结合性", "不能保证 100% 数值一致", "重新跑一次小规模对齐实验")),
        ),
        pitfall="很多人只会说'FSDP是PyTorch原生的，DeepSpeed是第三方的'，说不出真正的粒度差异；第3层"
                "几乎没人诚实承认'两者收敛结果不能保证完全一致'，会想当然回答'肯定一样'。",
        real_world_link="learning/scaling-infra/src/fsdp_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-10", cat=CAT,
        trigger="你提到用了 gradient checkpointing（activation recomputation）来省显存。",
        chain=(
            ("gradient checkpointing 具体是怎么用计算换显存的？",
             "标准反向传播需要前向时保留每一层的中间激活值以供反向时计算梯度；gradient checkpointing "
             "只保留少数检查点层的激活，反向传播需要某层激活但没保留时，现场从最近的检查点重新做一次局部"
             "前向计算（recompute）把这些激活重新算出来再继续反向；这样显存占用从 O(层数) 降到 O(检查点"
             "数)，代价是多付出一次这部分的前向计算量，理论上总 FLOPs 增加约 33%。",
             ("检查点", "recompute", "增加约 33%")),

            ("既然会增加 33% 左右计算量，为什么大规模训练里几乎默认开着它？这个交易在什么条件下不划算？",
             "因为在显存是硬约束的场景下，多付出算力换取能用更大有效 batch 或更大模型规模几乎总是划算"
             "的，尤其是在算力过剩、显存紧张的当下训练场景中；不划算的情况是模型/batch 本身就能塞进显存、"
             "GPU 计算已经打满（compute-bound 而非 memory-bound），这时候开 checkpoint 纯粹是白白多算"
             "一遍前向、拖慢吞吐而没有任何收益，应该关掉或只对显存占用最大的几层做选择性 checkpoint 而"
             "不是全开。",
             ("显存是硬约束", "compute-bound", "选择性 checkpoint")),

            ("activation recomputation 产生的额外前向计算，会不会挤占本该用来隐藏通信延迟的计算窗口？",
             "会，这是个容易被忽视的耦合：overlap 依赖反向计算的时间窗口去掩盖通信（比如 ZeRO-3 的下一层"
             "参数 prefetch all-gather），而 activation recomputation 本身也要占用同一批计算资源"
             "（SM）——如果 recompute 被安排在原本用来掩盖通信的时间段执行，两者会争抢 GPU 资源，可能导致"
             "要么 recompute 变慢、要么通信没被完全掩盖，重新暴露出来；工程上需要靠 profiler 实测确认"
             "两者没有互相拖累，理论分析很难穷尽所有调度交互情况。",
             ("争抢 GPU 资源", "重新暴露出来", "调度交互情况")),
        ),
        pitfall="第1层容易记错开销是'省了很多显存但计算量翻倍'，其实是约33%而不是100%；第3层几乎没人"
                "主动提到recompute和通信overlap之间会抢资源这个交互效应。",
        real_world_link="learning/scaling-infra/src/pipeline_parallel_demo.py",
    ),

    DeepPoint(
        id="dp-ai-dist-11", cat=CAT,
        trigger="你提到训练集群支持弹性（elastic）训练/容错。",
        chain=(
            ("一个节点在训练中途宕掉，弹性训练框架大致怎么处理？",
             "典型流程是：心跳机制发现某个 worker 失联超时，coordinator 判定该 worker 出局，触发重新 "
             "rendezvous（其余存活 worker 重新协商 world size 和 rank 分配），训练从最近一次成功的 "
             "checkpoint 恢复，以缩小后的 world size（或者等替补节点加入后恢复到原 world size）继续训"
             "练；由于 world size 变化会影响有效 batch size 和学习率的对应关系，很多实现会同步调整学习"
             "率或梯度累积步数来维持训练动态的一致性。",
             ("心跳机制", "rendezvous", "学习率的对应关系")),

            ("频繁 checkpoint 恢复对训练吞吐的实际代价有多大？工程上怎么权衡 checkpoint 频率？",
             "checkpoint 本身的写入（尤其是全量 fp32 优化器状态，通常是模型参数量的好几倍大小）如果同步"
             "阻塞主训练循环，会造成明显的 stall；工程上常用异步 checkpoint（先拷贝到 CPU pinned "
             "memory 或本地 NVMe，再后台异步写入远程存储，不阻塞下一个 training step）来掩盖这部分开销；"
             "checkpoint 频率的权衡是两次 checkpoint 之间丢失的计算量和 checkpoint 本身开销占训练时间"
             "比例之间的期望值最优化，实践中往往用经验值而非精确求解。",
             ("异步 checkpoint", "pinned memory", "期望值最优化")),

            ("如果集群故障率很高、大部分算力都花在'重启+从checkpoint恢复'上，你怎么判断这个训练任务的弹性容错设计是否'划算'？",
             "可以用有效利用率（goodput = 实际推进的有效训练步数 / 总占用 GPU 小时数）这个指标衡量，如果 "
             "goodput 显著低于理论值，说明容错开销已经侵蚀了大量算力；这时候的权衡诚实地说没有一个通用"
             "最优解——需要结合具体的故障分布（是偶发单机故障还是频繁的网络分区）、checkpoint 成本、以及"
             "集群抢占策略做具体分析，业界对'多大规模、多高故障率下该采用哪种弹性策略'并没有一个放之四海"
             "而皆准的公式，更多是工程判断和实测调优。",
             ("goodput", "没有一个通用最优解", "工程判断和实测调优")),
        ),
        pitfall="很多人对弹性训练的理解停留在'能自动重启'，答不出checkpoint恢复对学习率/有效batch size"
                "的影响；第3层基本没人能给出goodput这类量化视角，容易含糊说'看情况'。",
        real_world_link="learning/training-orchestration/src/fault_tolerance.py",
    ),

    DeepPoint(
        id="dp-ai-dist-12", cat=CAT,
        trigger="你说训练用了 3D 并行（DP+TP+PP 组合），这几个维度怎么分配？",
        chain=(
            ("3D 并行里，TP/PP/DP 这三个维度的切分顺序和物理拓扑一般怎么对应？",
             "通常遵循通信量大的维度放在带宽最高的物理层级这一原则：TP 的 all-reduce/all-gather 频率"
             "最高、对延迟最敏感，几乎总是放在单机内的 NVLink 域；PP 的通信是点对点的、频率其次，一般"
             "跨越多个节点组织成流水线，节点间用较低带宽的网络也能接受；DP 的 all-reduce 频率最低（每个"
             "训练 step 一次）、对带宽要求相对最松，通常横跨最外层的节点分组，world size=TP度*PP度*DP度。",
             ("NVLink 域", "每个训练 step 一次", "world size=TP度*PP度*DP度")),

            ("给定一个固定的 GPU 总数和模型大小，怎么决定 TP/PP/DP 各自的度数？",
             "先由显存约束定下 TP 度（模型单层参数+激活能不能塞进单卡显存，不够则增大 TP 切得更细，但 "
             "TP 一般不超过单机 GPU 数）；再由模型总层数和显存/气泡权衡定下 PP 度（TP 切完仍放不下整个"
             "模型时增大 PP）；剩下的 GPU 全部分给 DP 来扩展有效 batch size 和吞吐；这个决策过程本质上是"
             "先满足放得下（TP+PP兜住显存），再用DP扩展吞吐，顺序反过来通常行不通，因为 DP 不能解决单卡"
             "放不下模型的问题。",
             ("先满足放得下", "TP+PP兜住显存", "再用DP扩展吞吐")),

            ("3D 并行组合起来之后，气泡、通信瓶颈这些单独维度的问题会互相放大还是互相抵消？给一个具体的失配例子。",
             "会互相耦合、有时互相放大：比如 PP 的气泡率是 (p-1)/m，如果 TP 也在同一批 GPU 上运行，每个 "
             "micro-batch 在每个 PP stage 内部还要多付出 TP 的 all-reduce 延迟，相当于气泡里空等的绝对"
             "时间被拉长；一个具体失配例子是 TP 度设置过大导致单卡矩阵乘规模变小、GPU 利用率下降，这时候"
             "即使 PP 的 m 设置得很大、气泡率数字看起来很低，由于每个真正的计算步本身效率低，总吞吐仍然"
             "上不去——气泡率是个相对指标，不能脱离绝对计算效率单独看；这类三维耦合的联合调优通常没有解"
             "析解，大规模训练团队普遍靠网格搜索+profiler实测在几组候选配置里挑一个综合最优的组合。",
             ("气泡里空等的绝对时间被拉长", "气泡率是个相对指标", "网格搜索")),
        ),
        pitfall="第2层很多人会说'平均分配'或'随便设置'，说不出'先满足放得下再扩展吞吐'这个决策顺序；"
                "第3层几乎没人能举出具体的三维耦合失配例子，容易泛泛而谈'要综合考虑'。",
        real_world_link="learning/scaling-infra/src/parallelism_demo.py",
    ),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("dp-ai-dist-") for i in ids), "id 前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的点"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的点"
    # 自洽性检验：用每层自己的参考答案作答，采分关键词应能全部命中
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
    print(f"[PASS] dp_distributed_training: {len(BANK)}点 + 追问链自洽性 {ratio:.0%}")


if __name__ == "__main__":
    _self_test()
