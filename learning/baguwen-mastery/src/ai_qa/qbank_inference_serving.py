"""推理部署与服务化 八股问答库（约 12 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "推理部署与服务化"

BANK: list[QA] = [
    QA(
        id="ai-inf-01", cat=CAT,
        q="KV cache 到底占用多大显存？怎么估算？",
        a=(
            "KV cache是把Transformer每一层Attention里算出来的Key和Value缓存下来，避免自回归生成时对"
            "已经算过的token重复计算。每一层都要存一份K和一份V，粗略估算公式是：显存(字节) ≈ 层数 × 2"
            "（K和V两份） × hidden_size × 序列长度(seq_len) × batch_size × 每个数值的字节数（fp16/bf16"
            "是2字节）。如果用了分组查询注意力(GQA)或多查询注意力(MQA)，公式里的hidden_size要换成"
            "\"KV头数×每个头的维度\"而不是全部注意力头对应的维度，因为GQA/MQA下多个query头共享同一组"
            "K/V，能显著把这部分显存压下来。举例：一个32层、hidden size=4096的模型，fp16下batch=1、"
            "seq_len=2048时，KV cache≈32×2×4096×2048×2字节≈1.07GB，而且这个数字是随batch size和序列"
            "长度线性增长的——这也是为什么长上下文、大并发场景下KV cache而不是模型参数本身，常常才是"
            "显存瓶颈，这也是PagedAttention、GQA/MQA、量化KV cache等一系列技术要解决的问题。"
        ),
        keys=("层数", "hidden_size", "seq_len", "batch_size", "字节", "GQA"),
        follow_ups=(
            "为什么说长上下文场景KV cache比参数本身更容易成为显存瓶颈？",
            "GQA相比MHA具体是怎么压缩KV cache的？",
        ),
    ),
    QA(
        id="ai-inf-02", cat=CAT,
        q="PagedAttention 是怎么解决 KV cache 显存管理问题的？",
        a=(
            "传统做法给每个请求的KV cache预先分配一段连续显存，但请求的实际生成长度事先不知道（要按"
            "最大长度预留），而且不同请求长度参差不齐，这导致两类浪费：预留了用不完的部分（内部碎片）"
            "和请求结束后留下大小不一难以复用的空洞（外部碎片），vLLM论文里统计传统系统KV cache有效"
            "利用率往往只有20%-40%，60%-80%的显存被浪费掉。PagedAttention借鉴操作系统虚拟内存分页的"
            "思路：把KV cache切成固定大小的block（页），每个请求的KV cache由若干个block组成，这些"
            "block在物理显存里可以不连续存放，通过一张\"块表\"(block table)记录逻辑序列位置到物理"
            "block的映射；需要更多KV cache时按需分配新block，请求结束立即释放block归还给空闲池供下"
            "个请求复用，不再需要一次性预留最大长度的连续空间。这样几乎消除了外部碎片，内部碎片也被"
            "限制在\"最后一个block未填满\"这一种情况内，显存利用率能提升到接近最优，直接带来的收益"
            "是同样显存下能塞更多并发请求、做更大的batch，vLLM论文里报告在相同硬件下吞吐能比之前最优"
            "系统提升2-4倍。"
        ),
        keys=("虚拟内存", "分页", "block", "块表", "碎片", "2-4倍"),
        follow_ups=(
            "PagedAttention只是省显存，为什么还能带来吞吐提升？",
            "block共享（prefix sharing）是怎么进一步利用这套机制的？",
        ),
    ),
    QA(
        id="ai-inf-03", cat=CAT,
        q="投机采样(speculative decoding)的原理是什么？",
        a=(
            "自回归生成每一步都要跑一次完整的大模型前向，而大模型推理是明显的memory-bound（访存瓶颈，"
            "主要时间花在把权重从显存搬到计算单元，而不是算力本身），导致\"一步只吐一个token\"效率不高。"
            "投机采样引入一个更小、更快的草稿模型(draft model)，让它先连续自回归生成k个候选token"
            "（草稿），然后把这k个草稿token一次性喂给大模型（目标模型）做一次并行前向，大模型对这k个"
            "位置同时计算出\"如果自己来生成，每个位置真正应该选什么token\"的概率分布，再从头到尾逐个"
            "比较：草稿token和大模型认为应该生成的token是否一致，或者用拒绝采样的方式判断草稿token是否"
            "会被大模型接受，从第一个不一致的位置截断，前面被接受的草稿token直接采用，大模型在跑那一次"
            "前向时\"顺带\"验证了这些token，不用逐个token跑，只要草稿模型猜得准（接受率高），就能在"
            "一次大模型前向里吐出多个token，数学上可以证明这样采样出的最终token分布和只用大模型逐token"
            "采样是完全等价的，不损失生成质量。接受率高低取决于任务（代码补全等模式固定的任务接受率能"
            "到70%-90%，开放式创作类接受率较低），常见能带来1.5-3倍左右的延迟降低。"
        ),
        keys=("草稿模型", "目标模型", "并行前向", "接受率", "拒绝采样", "memory-bound"),
        follow_ups=(
            "投机采样为什么不会降低生成质量，数学上是怎么保证分布一致的？",
            "草稿模型选小模型还是用同一个模型的浅层退出(early exit)更好？",
        ),
    ),
    QA(
        id="ai-inf-04", cat=CAT,
        q="连续批处理(continuous batching)是什么？和传统的静态批处理有什么区别？",
        a=(
            "传统静态批处理(static batching)是把固定数量的请求凑成一批一起送进模型，要等这一批里所有"
            "请求都生成完（遇到各自的结束符或达到最大长度）才能把整批换成下一批新请求——但一批内不同"
            "请求的生成长度往往差异很大，短请求早就生成完了却必须陪着长请求一直空等到整批结束，GPU"
            "利用率随之打折扣，新请求也要排队等当前整批结束才能进来，排队延迟高。连续批处理(continuous"
            " batching，又叫iteration-level scheduling)则是在更细粒度——每一次解码迭代（每生成一个"
            "token）之后，就检查有没有请求已经生成结束，一旦某个请求结束立刻把它从batch里移除，腾出的"
            "槽位马上塞入一个新的等待中的请求继续下一步解码，batch的组成是动态、逐迭代变化的，而不是"
            "固定不变直到整批结束。这样GPU始终维持在接近满负荷的batch大小上运行，吞吐显著提升，新请求"
            "的排队等待时间也大幅缩短，是vLLM、TensorRT-LLM等现代推理框架的标配调度策略，通常会和"
            "PagedAttention配合（因为batch动态变化要求KV cache的分配/释放也要能动态、细粒度地进行）。"
        ),
        keys=("静态批处理", "动态", "排队", "GPU利用率", "iteration-level"),
        follow_ups=(
            "continuous batching为什么必须和类似PagedAttention这样的KV cache管理机制配合才好用？",
            "prefill和decode混在同一个batch调度时会有什么额外问题？",
        ),
    ),
    QA(
        id="ai-inf-05", cat=CAT,
        q="推理量化里INT8、AWQ、GPTQ分别是什么，有什么区别？",
        a=(
            "三者都是把模型权重（有些还包括激活值）从fp16/bf16压缩到更低比特，以省显存、加速推理，但"
            "技术路线不同。INT8量化最直接，通常按min-max或absmax对称量化，把权重（有的方案也包括激活"
            "值）映射到int8范围，配合dequant/quant kernel在推理时使用，精度损失相对可控、硬件支持最"
            "广泛，是最基础的baseline方案。GPTQ是一种逐层的训练后量化(PTQ)方法，把权重压到int4/int3"
            "这种更低比特，核心是用近似二阶(Hessian)信息，在量化每个权重时同时补偿由此引入的误差到该"
            "层还未量化的其它权重上，从而在极低比特下也能把整层的量化误差降到较低水平，但计算量较大"
            "（要估计Hessian近似信息）。AWQ（激活感知权重量化）的思路不同：它观察到并不是所有权重通道"
            "都同等重要，少数（约1%）对应大激活值的\"显著通道\"对输出影响更大，AWQ通过统计激活值分布"
            "找出这些显著通道，量化前把这些通道的权重按比例放大、对应的激活值按比例缩小（数学上等价，"
            "不改变最终结果），这样显著通道的相对量化误差变小，从而在同样int4比特下整体精度通常优于"
            "GPTQ，而且AWQ不需要像GPTQ那样做逐层的复杂反向补偿计算，校准速度更快，配套的CUDA kernel在"
            "vLLM/SGLang等框架里的推理速度也通常更快，目前工业界部署int4常优先选AWQ，GPTQ胜在更成熟、"
            "生态支持久。"
        ),
        keys=("对称量化", "GPTQ", "AWQ", "显著通道", "校准", "int4"),
        follow_ups=(
            "AWQ为什么要把显著通道的权重放大、激活值缩小，而不是直接跳过不量化？",
            "量化到int4之后，推理速度一定会变快吗，什么情况下不一定？",
        ),
    ),
    QA(
        id="ai-inf-06", cat=CAT,
        q="vLLM 和 TensorRT-LLM 这类推理框架各自的核心卖点是什么？",
        a=(
            "这两者都是主流的LLM推理服务框架，但设计取舍不同。vLLM的核心卖点是易用性+跨硬件的高吞吐："
            "它以PagedAttention为核心技术管理KV cache，配合continuous batching动态调度，能直接加载"
            "HuggingFace格式的模型开箱即用，不需要离线编译，新模型接入快、迭代成本低，并且支持NVIDIA"
            "之外的AMD ROCm、Intel等多种硬件后端，适合模型/需求经常变化、需要快速上线的场景。"
            "TensorRT-LLM是NVIDIA官方的推理引擎，核心卖点是把模型显式编译成针对特定GPU架构（如"
            "H100/Blackwell）高度优化的静态计算图(engine)，深度利用NVIDIA的FP8/INT4等量化和Tensor"
            "Core特性，并且和NVIDIA生态（Triton Inference Server、NIM等）深度集成，在编译完成后的"
            "稳态吞吐/延迟上通常优于vLLM，代价是每次模型或配置变化都要重新走一次编译流程（可能要几十"
            "分钟）、只支持NVIDIA硬件，灵活性和迁移性较差。简单说：vLLM卖的是\"快速上线+硬件灵活性\""
            "，TensorRT-LLM卖的是\"针对NVIDIA硬件榨到极致的稳态性能\"，实践中不少团队会两者搭配。"
        ),
        keys=("PagedAttention", "编译", "NVIDIA", "开箱即用", "Triton", "FP8"),
        follow_ups=(
            "为什么TensorRT-LLM需要离线编译，这个编译具体在做什么优化？",
            "如果模型经常更新迭代，选哪个框架更合适，为什么？",
        ),
    ),
    QA(
        id="ai-inf-07", cat=CAT,
        q="LLM推理里的prefill和decode两个阶段有什么区别？为什么说它们的性能瓶颈不一样？",
        a=(
            "一次推理请求分两个阶段。Prefill（预填充）阶段：把用户输入的整段prompt一次性喂给模型做前向"
            "计算，一次性算出prompt里所有token对应的KV cache并写入缓存，同时得到基于prompt生成第一个"
            "输出token所需的logits——这个阶段是并行处理整段prompt的所有token，矩阵乘法的规模大，是明显"
            "的compute-bound（计算瓶颈），GPU算力利用率高。Decode（解码）阶段：之后每一步只生成一个新"
            "token，输入只有上一步刚生成的这一个token，需要读取之前全部已缓存的KV再和这一个新token做"
            "attention，矩阵运算规模很小（每步只处理1个token），但仍然需要把全部模型权重从显存搬运到"
            "计算单元参与这一步计算，计算量相对搬运的数据量而言很小，所以decode阶段是明显的memory-bound"
            "（访存瓶颈），GPU算力大量闲置、主要卡在带宽上。这也是为什么continuous batching、投机采样"
            "等技术主要是针对decode阶段\"访存利用率低\"问题去做优化，以及为什么会有专门的chunked"
            " prefill技术尝试把两阶段揉在一起调度以提高整体GPU利用率。"
        ),
        keys=("prefill", "decode", "compute-bound", "memory-bound", "访存"),
        follow_ups=(
            "为什么decode阶段GPU算力会大量闲置，提升batch size能不能缓解？",
            "chunked prefill具体是怎么把两个阶段的调度揉合在一起的？",
        ),
    ),
    QA(
        id="ai-inf-08", cat=CAT,
        q="分组查询注意力(GQA)/多查询注意力(MQA)是怎么减少KV cache占用的？",
        a=(
            "标准多头注意力(MHA)里，Query、Key、Value各自有和头数相同数量的独立投影，每个头都有自己"
            "单独的一份K和V，KV cache的大小和\"注意力头数\"成正比。MQA(多查询注意力)把所有Query头共享"
            "同一组Key和Value（只有一份K、一份V），KV cache直接缩小到\"头数分之一\"，大幅省显存，代价"
            "是模型表达能力/效果打了折扣，毕竟所有头被迫用同一组K/V。GQA(分组查询注意力)是MHA和MQA之间"
            "的折中：把Q的多个头分成若干组，同一组内的Q头共享一份K/V，不同组各自有自己的K/V——比如32个"
            "Q头分成8组，KV头数就是8而不是32，KV cache相应缩小到4分之一（如果MHA是32个KV头的话）。GQA"
            "能在几乎不损失效果的情况下把KV cache压缩到MHA的几分之一，是Llama2/3等现代开源模型的标配"
            "设计，本质是在\"效果（更接近MHA）\"和\"显存/带宽（更接近MQA）\"之间找一个可调节的平衡点。"
        ),
        keys=("MHA", "MQA", "GQA", "共享", "头数", "折中"),
        follow_ups=(
            "GQA的分组数一般怎么选，和模型效果有什么权衡关系？",
            "除了GQA/MQA，还有哪些技术能进一步压缩KV cache（比如量化KV cache）？",
        ),
    ),
    QA(
        id="ai-inf-09", cat=CAT,
        q="什么是前缀缓存(prefix caching)？它能解决什么问题？",
        a=(
            "在很多实际场景里，不同请求的prompt开头部分经常是相同或高度重合的，比如同一个system"
            "prompt被反复复用、多轮对话里后一轮的prompt是前一轮prompt加几句新内容、多个用户共享同一段"
            "长文档做检索问答等。如果每次请求都从头对整个prompt重新做一次prefill计算KV cache，这部分"
            "重复的计算和显存写入是完全浪费的。前缀缓存的做法是：把已经计算过的prompt前缀对应的KV"
            "cache保留（缓存）下来并建立索引（常用哈希/trie结构标记内容），新请求进来时先检查它的开头"
            "有没有命中某个已缓存的前缀，如果命中就直接复用那部分KV cache、只对新增的、之前没算过的"
            "后缀部分做prefill，大幅减少重复计算，尤其在多轮对话、共享system prompt、RAG场景（相同"
            "检索文档反复出现）下能明显降低首token延迟(TTFT)和计算成本。这类技术在vLLM里是自动完成的"
            "常驻能力，由于PagedAttention本身就是按block管理KV cache，内容相同的block可以直接被多个"
            "请求的block table共同指向同一份物理block，不需要额外复制。"
        ),
        keys=("前缀", "缓存", "KV cache", "复用", "TTFT", "block"),
        follow_ups=(
            "前缀缓存命中需要prompt前缀完全一致吗，近似匹配可以吗？",
            "前缀缓存和多轮对话场景结合时，缓存应该保留多久、怎么淘汰？",
        ),
    ),
    QA(
        id="ai-inf-10", cat=CAT,
        q="衡量LLM推理服务性能常用哪些延迟/吞吐指标？它们之间存在什么权衡？",
        a=(
            "常用几个指标：TTFT(time to first token，首token延迟)衡量从请求发出到吐出第一个token的"
            "等待时间，主要由prefill阶段决定，用户体感上是\"要等多久才开始看到回复\"；TPOT(time per"
            " output token，平均每个输出token的生成间隔)衡量decode阶段吐字的流畅度，TPOT越低意味着"
            "\"打字速度\"越快；吞吐量(throughput)一般用每秒生成的token数(tokens/s)衡量，反映服务在"
            "给定硬件下能同时支撑多少请求量、单位成本能摊多少请求。这几个指标之间存在明显权衡：把"
            "batch size开得越大，单位硬件的吞吐量通常越高（GPU利用率更高、边际成本更低），但同一个"
            "batch里每个请求要\"排队\"和其它请求共享计算资源，会拉长该批次内每个请求的TTFT和TPOT；"
            "反过来batch开得小、甚至只服务单个请求，TTFT/TPOT体验最好但吞吐量很低、硬件成本分摊到每"
            "个请求上更贵。实际服务在SLA（比如\"TTFT<1秒\"\"TPOT<50ms\"）约束下去尽量把batch size和"
            "调度策略调到能达到的最大吞吐量，这也是continuous batching存在的意义。"
        ),
        keys=("TTFT", "TPOT", "吞吐量", "batch size", "延迟", "权衡"),
        follow_ups=(
            "如果SLA要求TTFT必须很低，应该怎么调整调度策略？",
            "吞吐量和并发请求数之间是什么关系，是不是并发越高吞吐一定越高？",
        ),
    ),
    QA(
        id="ai-inf-11", cat=CAT,
        q="什么是chunked prefill？它解决了prefill和decode混合调度时的什么问题？",
        a=(
            "在continuous batching里，同一个调度批次(batch)可能同时混有正处于prefill阶段的新请求"
            "（要一次性处理一大段prompt，compute-bound）和处于decode阶段的老请求（每步只处理1个token，"
            "memory-bound）。如果不加处理，一个很长prompt的prefill任务会独占一次迭代的大部分计算时间，"
            "导致同批次里其它请求的decode被迫等待，拖长了它们的TPOT，造成\"长prompt请求插队，影响其它"
            "用户吐字流畅度\"的问题。chunked prefill的做法是把一次prefill的整段prompt切成若干个小"
            "chunk，不再要求一次迭代内把整段prompt算完，而是每次迭代只处理一个chunk的prefill计算量，"
            "和当前batch里其它请求的decode计算混合打包成一个大小适中、耗时可预测的迭代，分散到多次"
            "迭代里完成整个prefill，这样长prompt请求不会在某一次迭代里过度占用计算资源，decode请求的"
            "TPOT更稳定可预期，整体上让prefill和decode能更平滑地共享同一批GPU资源，是vLLM、SGLang等"
            "框架里用来平衡TTFT和TPOT的关键调度技术之一。"
        ),
        keys=("prefill", "decode", "chunk", "调度", "TPOT", "混有"),
        follow_ups=(
            "chunk大小怎么选，切太小或太大分别有什么问题？",
            "chunked prefill和投机采样同时使用会有冲突吗？",
        ),
    ),
    QA(
        id="ai-inf-12", cat=CAT,
        q="大模型推理阶段为什么也需要张量并行？和训练阶段用张量并行的目的有什么区别？",
        a=(
            "当模型本身太大、单张GPU显存装不下完整的模型权重（加上KV cache）时，推理阶段同样需要把"
            "模型切分到多张GPU上，张量并行(TP)是最常用的方式之一——原理和训练阶段一样，把Attention/MLP"
            "的矩阵乘法按行/列切开分布到多卡，每张卡只存并计算参数的一部分，层内需要一次all-reduce把"
            "各卡的部分结果合并。但推理阶段用TP的目的和训练阶段侧重点不同：训练阶段TP主要是为了解决"
            "\"显存装不下\"和加速训练吞吐；推理阶段除了同样要解决\"单卡装不下大模型权重\"的问题外，"
            "更重要的诉求往往是降低单个请求的延迟(latency)——把同一层的计算分摊到多张卡上并行执行，"
            "能直接缩短每一步decode的用时，这对于线上服务的TTFT/TPOT体验很关键；此外TP在推理时还能"
            "变相增加可用的显存带宽总量和KV cache总容量（多卡的显存带宽叠加），对于大batch、长上下文"
            "场景下decode阶段本来就是memory-bound，TP相当于用多张卡的带宽一起分担访存压力。推理阶段TP"
            "并行度的选择通常比训练更保守，因为要在\"更低延迟\"和\"多卡通信开销、以及不必要地浪费GPU"
            "资源做小模型推理\"之间权衡。"
        ),
        keys=("张量并行", "显存带宽", "延迟", "all-reduce", "memory-bound"),
        follow_ups=(
            "推理阶段TP并行度选大了会有什么代价？",
            "除了TP，推理阶段还有哪些方式应对单卡装不下模型的问题（比如流水并行、offload）？",
        ),
    ),
]


def _self_test() -> None:
    assert 10 <= len(BANK) <= 16, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-inf-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_inference_serving: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
