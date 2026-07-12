"""归一化 八股问答库（约 10 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "归一化"

BANK: list[QA] = [
    QA(id="ai-norm-01", cat=CAT,
       q="BatchNorm 和 LayerNorm 在训练/推理阶段的行为差异分别是什么?BN 的移动平均在 batch size=1 时会出现什么问题?",
       a=(
           "BatchNorm 训练时用当前 mini-batch 的均值/方差归一化,同时维护一份全局的移动平均(running "
           "mean/var);推理时不再用当前样本的统计量,而是直接用训练期间积累的移动平均做归一化,这样"
           "才能保证单个样本推理时结果确定、不依赖'恰好和谁同一个 batch'。LayerNorm 则训练和推理用的"
           "是同一套逻辑——每个样本自己的特征维度统计量,没有训练/推理不一致的问题,也不需要维护移动"
           "平均。BN 的这套机制在 batch size 很小、尤其 batch size=1 时会直接失效:单样本算出的均值"
           "就是它自己、方差恒为 0,统计量完全不可靠,加上 batch size 太小时训练期间的移动平均本身"
           "也估计得不准,这是 BN 在小 batch 场景(如目标检测、逐样本在线推理)表现差、经常要换成 "
           "GroupNorm 等替代方案的直接原因。"
       ),
       keys=("移动平均", "batch size", "训练", "推理", "统计量"),
       follow_ups=("为什么不能推理时也用当前 batch 的统计量,非要维护移动平均?",
                   "移动平均的动量系数(momentum)怎么选,过大过小分别有什么影响?",
                   "GroupNorm 是怎么绕开 batch 维度依赖的?")),

    QA(id="ai-norm-02", cat=CAT,
       q="NLP 场景里直接把 BatchNorm 套用到 Transformer 会遇到哪些具体问题(变长序列/padding/自回归推理)?为什么最终选择了 LayerNorm?",
       a=(
           "第一个问题是变长序列必须靠 padding 对齐成定长 batch,BN 是按特征维在整个 batch(含所有位置)"
           "上算统计量,padding token 的激活值会被一起算进均值/方差里,污染真实 token 的统计量,而且不"
           "同 batch 里 padding 比例还不一样,统计量本身就不稳定。第二个问题是自回归推理(逐 token 生成)"
           "时经常是单样本、甚至逐步只算一个新 token 的激活,batch 维退化到 1,和 BN 在小 batch 下的"
           "问题(见上一题)是同一回事。LayerNorm 对每个样本、每个位置独立按特征维(hidden dim)归一化,"
           "完全不依赖 batch 里其他样本或其他位置的统计量,天然免疫变长、padding、batch size 波动这三个"
           "问题,这正是 Transformer 从一开始就选择 LayerNorm 而不是 BatchNorm 的直接工程原因。"
       ),
       keys=("padding", "变长序列", "自回归", "batch", "hidden dim"),
       follow_ups=("有没有工作专门魔改 BN 让它能用在 Transformer 上(如 PowerNorm)?",
                   "训练时用了 padding mask,为什么统计量还是会被污染?",
                   "encoder(可以整句可见)和 decoder(自回归)在这个问题上有区别吗?")),

    QA(id="ai-norm-03", cat=CAT,
       q="RMSNorm 为什么去掉 LayerNorm 的'去均值'(re-centering)这一步不会明显损失效果?这说明 LayerNorm 的两种不变性谁更关键?",
       a=(
           "LayerNorm 同时提供两种不变性:re-centering(减均值,让输出对输入的整体平移不敏感)和 "
           "re-scaling(除以标准差,让输出对输入的整体缩放不敏感)。RMSNorm 的作者 Zhang & Sennrich 通过"
           "实验发现,LayerNorm 起稳定训练作用的主要是 re-scaling 这一半,re-centering 带来的增益很小、"
           "可以舍弃:RMSNorm 只用均方根(root mean square)统计量做缩放,不减均值,当输入均值恰好为 0 "
           "时 RMSNorm 和 LayerNorm 完全等价。直觉上,Transformer 里到处都是残差连接和线性投影,这些"
           "结构本身就有能力吸收/抵消掉均值偏移,不需要归一化层再显式做一次去均值,所以去掉 "
           "re-centering 几乎不损失效果,却能省掉算均值和减均值这两步计算,这也是 LLaMA 等大模型改用 "
           "RMSNorm 的原因。"
       ),
       keys=("re-centering", "re-scaling", "均方根", "残差连接", "LLaMA"),
       follow_ups=("均方根具体怎么计算,和标准差的区别是什么?", "RMSNorm 省掉的计算量具体体现在哪几步?",
                   "如果输入均值明显不为 0,RMSNorm 会有什么风险?")),

    QA(id="ai-norm-04", cat=CAT,
       q="从梯度随深度 L 的缩放行为看,Post-LN 和 Pre-LN 分别是什么规律?这怎么解释 Post-LN 为什么必须要 warmup?",
       a=(
           "Xiong 等人用均值场理论分析发现:Post-LN(归一化放在残差相加之后)在初始化时,最后一层"
           "参数的梯度期望范数和网络深度 L 无关——也就是说不管模型多深,顶层梯度都同样大,这会让顶层"
           "在训练一开始就承受和浅层网络一样'猛'的更新,如果学习率不小心从一开始就设得大,极容易训练"
           "不稳定甚至发散,这正是 Post-LN Transformer 必须要用学习率 warmup 把前几步的更新压小的理论"
           "根源。Pre-LN(归一化放在残差分支内部、加回主干之前)则相反:注意力/FFN 子层输出在加回残差"
           "前先被本层的 LayerNorm 缩放过,输入到最终归一化层的尺度会随深度 L 线性增长,导致各层参数"
           "的梯度范数会被大约 1/√L 这个因子压小,深度越深梯度自然越小、更新天然温和,因此 Pre-LN 可以"
           "安全地去掉 warmup 阶段而不会训练失稳,代价是深层网络的有效更新量整体偏小、需要更多步数或"
           "更大学习率去弥补。"
       ),
       keys=("Post-LN", "Pre-LN", "梯度范数", "深度", "warmup"),
       follow_ups=("DeepNorm 是怎么让 Post-LN 也能稳定训练到上千层的?",
                   "Pre-LN 梯度变小会不会导致深层实际学不到东西(表示退化)?",
                   "这个理论分析对 Adam 类自适应优化器还适用吗(Adam 会不会自己抹平梯度尺度差异)?")),

    QA(id="ai-norm-05", cat=CAT,
       q="归一化为什么能加速训练?原始的'减少内部协变量偏移'解释和后续的'损失面平滑化'解释分别是什么?",
       a=(
           "BatchNorm 提出时(Ioffe & Szegedy)给出的解释是:训练过程中每层输入的分布会随前面层参数的"
           "更新不断变化,这种'内部协变量偏移(internal covariate shift)'迫使后面的层不断重新适应新的"
           "输入分布,减慢收敛;归一化把每层输入拉回到稳定的均值方差,减少了这种漂移,所以训练更快。但 "
           "Santurkar 等人(2018)用实验挑战了这个解释:他们故意在 BN 之后人为注入噪声、制造更严重的"
           "协变量偏移,发现带 BN 的网络照样训练得很好,协变量偏移的大小和训练效果之间并没有必然联系。"
           "他们提出更站得住脚的解释是:BatchNorm 让损失面变得更平滑(改善了损失和梯度的 Lipschitz "
           "常数,也就是 β-平滑性),梯度的变化更可预测、不会突然剧烈变化,这才让我们能放心用更大的学习"
           "率、更少的迭代次数就收敛,这个'损失面平滑化'的解释目前被认为比'内部协变量偏移'更准确地"
           "刻画了归一化真正起作用的机制。"
       ),
       keys=("内部协变量偏移", "损失面", "平滑", "Lipschitz", "梯度"),
       follow_ups=("Santurkar 的实验具体是怎么设计的(怎么人为制造协变量偏移)?",
                   "'损失面平滑化'这个解释能不能同样解释 LayerNorm 为什么有效?",
                   "既然协变量偏移不是关键,BN 里的移动平均还有存在的必要吗?")),

    QA(id="ai-norm-06", cat=CAT,
       q="GroupNorm 和 InstanceNorm 分别是怎么切分归一化维度的?各自适合什么场景?",
       a=(
           "四种常见归一化的区别都在于'沿哪些维度算统计量':BatchNorm 沿 batch 和空间维、对每个通道"
           "单独算(依赖 batch);LayerNorm 沿通道和空间维、对每个样本单独算(不依赖 batch);GroupNorm "
           "把通道分成若干组,只在同一组的通道(加空间维)内部算统计量、同样对每个样本单独算,是 BN 和 "
           "LN 之间的折中——不依赖 batch,又保留了'按通道分组'这个更贴近卷积网络特征结构的先验,在"
           "batch size 很小(如目标检测、分割这类显存吃紧只能上小 batch 的任务)时明显优于 BN。"
           "InstanceNorm 是 GroupNorm 分组数等于通道数的极端情况,对每个样本的每个通道单独算统计量"
           "(只沿空间维),会抹掉每张图片自己的整体风格/对比度信息,这个特性反而在风格迁移、图像生成"
           "任务里很有用——因为这些任务恰恰希望把'内容'和'风格统计量'解耦开来。"
       ),
       keys=("GroupNorm", "InstanceNorm", "通道", "batch", "风格迁移"),
       follow_ups=("GroupNorm 的组数(group number)一般怎么选?", "为什么检测/分割模型偏爱 GroupNorm 而不是 BN?",
                   "AdaIN(自适应实例归一化)是怎么用 InstanceNorm 做风格迁移的?")),

    QA(id="ai-norm-07", cat=CAT,
       q="归一化层里可学习的 gamma/beta 仿射参数是干什么用的?如果去掉它们会怎样?",
       a=(
           "归一化(减均值除标准差)会把每层输入强行拉到均值 0、方差 1 的固定分布,这虽然稳定了训练,"
           "但也剥夺了网络自己调节'这一层到底需要多大尺度、多大偏移'的能力——如果某一层其实需要更大"
           "的激活幅度才能表达某个特征,强行归一化到方差 1 会限制它的表达能力。gamma(缩放)和 beta"
           "(偏移)是紧跟在归一化后面的一对可学习仿射参数,让网络可以在训练中自己学出'要不要、以及"
           "学多大'的缩放和偏移,在极端情况下甚至可以学出 gamma=标准差、beta=均值,把归一化的效果"
           "完全'学回去',相当于把归一化变成一个可选项而不是硬约束。如果去掉 gamma/beta,网络就被"
           "永久锁死在均值 0、方差 1 的输出分布上,表达能力受限,实践中会观察到收敛变慢、最终精度下降。"
       ),
       keys=("gamma", "beta", "仿射", "表达能力", "均值"),
       follow_ups=("gamma/beta 的初始化一般怎么设(全 1/全 0 还是别的)?",
                   "RMSNorm 里还保留 gamma 吗,为什么不需要 beta?",
                   "这对参数会不会被 weight decay 正则,实践中一般怎么处理?")),

    QA(id="ai-norm-08", cat=CAT,
       q="超深 Transformer(上百甚至上千层)单纯用 Pre-LN 会遇到什么新问题?DeepNorm 是怎么应对的?",
       a=(
           "Pre-LN 虽然让浅层深层梯度都比较稳定、不需要 warmup,但 Xiong 等人和后续 DeepNet 的工作发现"
           "一个新问题:Pre-LN 网络里底层(靠近输入)的梯度范数系统性地大于顶层,深度越深这个差距越"
           "明显,导致底层过度更新、顶层更新不足,实际表现为深度堆得越多、性能提升越不明显,深层网络的"
           "容量没有被充分利用,最终效果反而不如层数少一些但训练更充分的 Post-LN 模型。DeepNorm(Wang "
           "等,DeepNet)的做法是回到 Post-LN 的结构,但在残差分支上乘一个只和层数相关的放大系数 α、"
           "同时配合专门推导的初始化缩放系数 β,从理论上把每次更新的幅度界定在一个和层数无关的稳定"
           "范围内,做到'Post-LN 的最终效果 + Pre-LN 级别的训练稳定性',凭这套方案把 Transformer 成功"
           "堆到了 1000 层。"
       ),
       keys=("Pre-LN", "梯度范数", "DeepNorm", "残差", "初始化"),
       follow_ups=("DeepNorm 的放大系数 α 具体怎么和层数 L 挂钩?",
                   "现在的主流 LLM(GPT/LLaMA 系列)为什么大多还是几十到一百多层的 Pre-LN,而不是 DeepNorm?",
                   "'底层梯度大、顶层梯度小'这个现象和优化器(如 Adam 的逐参数自适应)会不会互相抵消?")),

    QA(id="ai-norm-09", cat=CAT,
       q="LayerNorm 在 Transformer 里具体是对哪个维度做归一化?和 BatchNorm 归一化维度的几何直觉区别是什么?",
       a=(
           "Transformer 的激活张量形状通常是 [batch, seq_len, hidden_dim]。LayerNorm 是对每个 "
           "(batch, seq_len) 位置单独取出它的 hidden_dim 这一条向量,在这一条向量内部算均值和方差、"
           "归一化,几何直觉是'把每个 token 自己的特征向量重新缩放到统一尺度',和其他样本、其他位置"
           "完全无关。BatchNorm 反过来,是固定住某一个特征通道,把 batch 维(以及有空间维的话还有空间"
           "维)上所有位置的值收集起来算均值和方差,几何直觉是'把同一个特征通道在整个数据集/批次里的"
           "分布拉到统一尺度',和同一样本内其他特征无关。简单说:LayerNorm 是'同一个 token 内部各特征"
           "互相看齐',BatchNorm 是'同一个特征跨样本互相看齐',这也是为什么变长序列和小 batch 场景天然"
           "更适合 LayerNorm——它的归一化对象从来没有跨样本。"
       ),
       keys=("hidden_dim", "batch", "seq_len", "特征通道", "跨样本"),
       follow_ups=("如果把 LayerNorm 错误地沿 seq_len 维归一化会发生什么?",
                   "多头注意力里的归一化是在拆分成多头之前还是之后做?",
                   "为什么说 LayerNorm 的归一化对象'从来没有跨样本'是它稳定的关键?")),

    QA(id="ai-norm-10", cat=CAT,
       q="Weight Normalization 和 BatchNorm/LayerNorm 这类方法在本质上有什么不同?",
       a=(
           "BatchNorm/LayerNorm/GroupNorm 这类方法归一化的对象是激活值(activation)——对网络中间层"
           "的输出做统计量归一化。Weight Normalization(Salimans & Kingma)归一化的对象则是权重"
           "(weight)本身:把每个权重向量 w 重参数化为 w = (g/||v||)·v,用一个标量 g 控制权重的模长、"
           "用方向向量 v 控制权重的朝向,把'长度'和'方向'两个自由度解耦开分别优化。因为它不依赖任何"
           "跨样本的统计量,Weight Normalization 天然不受 batch size 影响,可以直接用在 RNN、强化学习、"
           "生成模型这些 BatchNorm 表现不好的场景;但它没有像 BatchNorm/LayerNorm 那样直接稳定住每层"
           "输出的分布,实践中训练加速效果通常不如激活归一化明显,目前在主流架构里已不如 LayerNorm/"
           "RMSNorm 常见。"
       ),
       keys=("权重", "激活值", "重参数化", "batch size", "方向"),
       follow_ups=("Weight Normalization 的 g 和 v 分别怎么初始化?",
                   "Weight Standardization 和 Weight Normalization 是什么关系?",
                   "为什么说 Weight Normalization 可以看作'对白化输入做 BatchNorm'的特例?")),
]


def _self_test() -> None:
    assert 10 <= len(BANK) <= 15, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-norm-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_normalization: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
