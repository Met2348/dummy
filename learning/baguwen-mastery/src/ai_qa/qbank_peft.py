"""微调与参数高效方法（PEFT）八股问答库（约 12 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "微调与参数高效方法(PEFT)"

BANK: list[QA] = [
    QA(id="ai-peft-01", cat=CAT,
       q="LoRA 的核心思想是什么？为什么可以假设权重更新是低秩的？",
       a="LoRA 冻结预训练权重 W，只学习一个低秩旁路 ΔW=BA（A 是 in×r，B 是 r×out，r 远小于 "
         "in/out），前向变成 h=Wx+BAx。其依据是'内在维度(intrinsic dimension)'假设：大模型针对"
         "下游任务的有效权重更新往往集中在一个很低维的子空间里，用满秩矩阵去拟合是浪费；只训 A、B "
         "就能把可训练参数量从 O(d²) 降到 O(d·r)，同时基座权重完全冻结、不引入推理延迟(可事后合并)。",
       keys=("低秩", "冻结", "ΔW", "内在维度"),
       follow_ups=("秩 r 一般怎么选？调大调小各有什么代价？", "LoRA 一般加在 Transformer 的哪些矩阵上？")),

    QA(id="ai-peft-02", cat=CAT,
       q="LoRA 里的秩 r 怎么选？r 太大或太小分别有什么问题？",
       a="r 是可训练容量和参数量/显存/过拟合风险之间的权衡旋钮。常见取值 4~64：r 太小(如 1~2)在"
         "任务复杂、数据量大时容量不够、欠拟合，收敛后与全参微调差距明显；r 太大则可训练参数逼近"
         "全参微调、失去省显存省算力的意义，还可能在小数据集上过拟合。实践上通常先用小 r(8/16)试，"
         "再按验证集效果和显存预算调整，同时不同层可以用不同 r(如 attention 的 Q/V 常比 FFN 更敏感)。",
       keys=("容量", "过拟合", "参数量", "权衡"),
       follow_ups=("r 和 alpha 应该怎么配比？", "所有层都用同一个 r 合理吗？")),

    QA(id="ai-peft-03", cat=CAT,
       q="LoRA 的缩放系数 alpha(以及 alpha/r)起什么作用？",
       a="LoRA 输出会乘一个缩放因子 alpha/r 再加到基座输出上：h=Wx+(alpha/r)·BAx。alpha 相当于"
         "控制这条低秩旁路对最终输出的'贡献强度'，如果不缩放，改变 r 会连带改变旁路输出的数值量级，"
         "导致换 r 就要重新调学习率；用 alpha/r 归一化后，改变 r 时旁路的有效更新幅度更稳定，超参"
         "搜索更可控。经验上 alpha 常设为 r 的 1~2 倍(如 r=8,alpha=16)。",
       keys=("缩放", "alpha/r", "数值量级", "学习率"),
       follow_ups=("为什么不直接把 alpha 固定为 1？", "QLoRA 里这个缩放系数会受量化影响吗？")),

    QA(id="ai-peft-04", cat=CAT,
       q="为什么 LoRA 里 A 矩阵随机初始化、B 矩阵初始化为零？",
       a="B=0 保证训练刚开始时 BA=0，旁路对输出没有任何扰动，模型初始状态严格等价于原始预训练模型"
         "(不会因为随机初始化的旁路一上来就把模型带偏)；A 需要非零(通常高斯随机)是因为如果 A 和 B "
         "都是 0，B 的梯度里会含有 A 的值(链式法则 dL/dB ∝ Aᵀ·(...))，A 全零会让 B 永远学不到梯度、"
         "旁路死掉。这种'一个矩阵零初始化、另一个随机初始化'的组合，才能同时满足'起点等价原模型'"
         "和'训练能启动'两个条件。",
       keys=("B=0", "等价", "随机", "梯度"),
       follow_ups=("如果反过来 A=0、B随机会怎样？", "这和残差连接的初始化哲学有什么相似之处？")),

    QA(id="ai-peft-05", cat=CAT,
       q="QLoRA 相比普通 LoRA 多做了哪三件事？分别解决什么问题？",
       a="QLoRA=NF4量化+双重量化+分页优化器，三者缺一不可地解决'如何在单卡上微调超大模型'：①用 "
         "NF4(4-bit NormalFloat)把冻结的基座权重压到 4bit 存储，大幅降低静态显存占用；②双重量化"
         "(Double Quantization)把 NF4 量化本身产生的量化常数(scale)再做一次量化，进一步省下约 "
         "0.37 bit/参数(65B 模型约省 3GB)；③分页优化器(Paged Optimizer)借助 NVIDIA 统一内存，在"
         "长序列梯度检查点导致显存瞬时尖峰时把优化器状态换页到 CPU，防止 OOM。三者合力让 65B 模型"
         "能在单张 48GB GPU 上微调、效果接近全精度 LoRA。",
       keys=("NF4", "双重量化", "分页优化器", "显存"),
       follow_ups=("NF4 和普通 int4 量化本质区别是什么？", "QLoRA 训练时前向计算是在 4bit 下做的吗？")),

    QA(id="ai-peft-06", cat=CAT,
       q="NF4(4-bit NormalFloat)这个数据类型和普通 int4/fp4 量化本质区别是什么？",
       a="普通 int4 是等宽量化：把数值范围切成 16 个等距离的格子，格子密度和数据实际分布无关。而"
         "神经网络权重训练后通常近似零均值正态分布，NF4 基于'分位数量化(quantile quantization)'"
         "思想，让每个量化格子(bin)包含大致相等数量的权重，也就是在正态分布密集的零附近安排更多、"
         "更细的格子，在两侧稀疏区域格子更稀。这是信息论意义上对'正态分布数据'最优的 4bit 量化"
         "方式，同等 4bit 位宽下比普通 int4/fp4 保留的信息更多、精度损失更小。",
       keys=("正态分布", "分位数", "零附近", "信息论"),
       follow_ups=("为什么要先假设权重是正态分布，这个假设在实践中稳吗？", "量化误差是怎么在前向时被'补偿'的(LoRA 旁路在做什么)？")),

    QA(id="ai-peft-07", cat=CAT,
       q="LoRA 一般加在 Transformer 的哪些权重矩阵上？为什么通常不把它插在 LayerNorm 之后？",
       a="LoRA 通常插在注意力的 Q/K/V/O 投影和/或 FFN 的上下投影这些大矩阵上(参数量 d×d 或 d×4d "
         "级别)，因为低秩分解在'原矩阵足够大'时才划算——大矩阵里学到的有效更新更可能是低秩的，压缩"
         "收益也大。而 LayerNorm 只有逐通道的 scale/bias 两个向量参数，量级是 O(d)而不是 O(d²)，"
         "参数量本身就很小，几乎没有可压缩的空间；给一个只有几千参数的层再套一层低秩分解(还要多存 "
         "A、B 两个矩阵)得不偿失，直接全量微调这几个参数(或干脆不动)更划算。",
       keys=("Q/K/V", "LayerNorm", "O(d)", "参数量"),
       follow_ups=("那 LoRA 会加在 embedding/lm_head 上吗？", "如果任务对某一层特别敏感，能不能对那一层单独全量微调、其余用 LoRA？")),

    QA(id="ai-peft-08", cat=CAT,
       q="Prompt-tuning 和 Prefix-tuning 是什么？和 LoRA 相比优劣如何？",
       a="两者都不改模型权重，而是在输入前拼接一段可训练的'软提示'：Prompt-tuning 只在输入 "
         "embedding 层前加若干可训练的连续向量(token 级)；Prefix-tuning 更进一步，在每一层 "
         "Transformer 的 K/V 前都插入可训练前缀向量，表达力更强但可训练参数和显存开销也更大。相比 "
         "LoRA：优点是实现极简、可训练参数可以压到更少，也天然适合'多任务共享同一基座、只换前缀'"
         "的场景；缺点是会占用有效上下文长度(prefix 也要过 attention)、优化通常更不稳定难调，且"
         "效果上限一般不如 LoRA 稳健，大规模任务上较少作为首选。",
       keys=("软提示", "前缀", "K/V", "上下文长度"),
       follow_ups=("为什么 prefix-tuning 比 prompt-tuning 表达力更强？", "P-tuning v2 相比 v1 做了什么改进？")),

    QA(id="ai-peft-09", cat=CAT,
       q="Adapter(Houlsby/Pfeiffer)和 LoRA 结构上有什么区别？为什么 LoRA 推理时可以做到零额外延迟？",
       a="Adapter 是串联(serial)结构：在 Transformer 每层的子层之后插入一个小的瓶颈 MLP(降维->激活"
         "->升维)，前向必须先过完原层再过 adapter，推理路径变长、增加延迟，这是 Houlsby(每层两个 "
         "adapter)和 Pfeiffer(每层一个、位置更省)的共同问题。LoRA 是并联(parallel)结构：旁路 BA "
         "和原权重 W 各自独立计算再相加(h=Wx+BAx)，训练时是两条并行路径；一旦训练完成，可以直接把 "
         "BA 合并加回 W 得到 W'=W+(alpha/r)BA，模型结构和推理路径与原始模型完全一样，不需要在线跑"
         "额外的小网络，因此没有推理延迟开销，这是 LoRA 相比 Adapter 最大的部署优势。",
       keys=("串联", "并联", "合并", "延迟"),
       follow_ups=("多个任务各训一个 LoRA，能否共享一个基座在推理时动态切换？", "Adapter 的瓶颈维度怎么选，和 LoRA 的秩 r 是类似的权衡吗？")),

    QA(id="ai-peft-10", cat=CAT,
       q="LoRA 微调完成后部署时，'合并权重'和'保留旁路分开存放'两种方式该怎么选？",
       a="合并权重(W'=W+(alpha/r)BA)：把 A、B 融进原矩阵，推理时和原模型结构完全相同、零额外延迟、"
         "零额外显存，适合'一个基座只服务一个固定任务'的单任务部署。保留旁路分开存放：基座权重不变，"
         "A、B 作为独立的小文件按需加载/切换，适合'一个基座服务很多个任务/客户'的多任务场景——每个"
         "任务只需存几 MB 的 A、B，而不是给每个任务都存一份完整模型；代价是推理时要多做一次旁路矩阵"
         "乘法(通常远小于原矩阵乘法，开销很小)，且要支持按请求路由到不同 adapter 的部署 serving 架构。",
       keys=("合并", "旁路", "多任务", "部署"),
       follow_ups=("如果要同时给一个 batch 里不同请求用不同 LoRA adapter，工程上要怎么处理？", "合并之后还能再叠加训练一个新的 LoRA 吗？")),

    QA(id="ai-peft-11", cat=CAT,
       q="全参数微调和 PEFT(以 LoRA 为代表)该怎么取舍？什么场景该选全参？",
       a="数据量非常大、任务与预训练分布偏移很大(如换语种、换模态、大幅扩充知识)、且有充足算力/"
         "显存时，全参数微调通常能压榨出更高的效果上限，因为它不受低秩假设的容量限制。而数据量有限、"
         "需要同时维护多个任务/客户定制、显存或算力紧张、或者只是想做指令对齐这种'激活已有能力而非"
         "注入大量新知识'的微调时，PEFT 性价比更高——用远少的参数量和显存达到接近全参的效果，且天然"
         "支持多 adapter 切换、更不容易灾难性遗忘基座能力。经验法则：小数据/多任务/资源受限选 "
         "PEFT；大数据/大分布偏移/追求上限选全参(或全参+少量 PEFT 混合)。",
       keys=("数据量", "分布偏移", "显存", "灾难性遗忘"),
       follow_ups=("为什么 PEFT 更不容易灾难性遗忘基座能力？", "有没有'先 PEFT 再全参'或反过来的混合策略？")),

    QA(id="ai-peft-12", cat=CAT,
       q="为什么 LoRA 这种只调极少参数(通常<1%)的方法，效果却能接近全参数微调？",
       a="背后是'内在维度(intrinsic dimension)'假设：Aghajanyan 等的研究发现，预训练大模型针对"
         "下游任务收敛所需的有效参数子空间维度远低于模型全部参数量，也就是说全参微调时真正'有信息量'"
         "的更新方向本来就集中在一个低维子空间里，其余方向的更新对任务效果贡献很小。模型规模越大，"
         "这个内在维度相对全参数量的占比越低(大模型的'冗余度'更高)，所以 LoRA 用一个低秩矩阵去覆盖"
         "这个低维子空间，就能拿到接近全参微调的效果，而且模型越大这个'性价比'优势越明显、跟随规模"
         "扩大而更划算。",
       keys=("内在维度", "低维子空间", "冗余", "规模"),
       follow_ups=("这个假设对所有任务都成立吗，有没有它失效的情况？", "这跟 SAE 讲的'叠加(superposition)'假说有没有相通之处？")),
]


def _self_test() -> None:
    assert 9 <= len(BANK) <= 15, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-peft-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_peft: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
