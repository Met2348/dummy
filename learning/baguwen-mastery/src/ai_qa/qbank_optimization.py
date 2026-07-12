"""优化与训练动力学 八股问答库（约 12 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "优化与训练动力学"

BANK: list[QA] = [
    QA(id="ai-opt-01", cat=CAT,
       q="SGD -> Momentum -> Adam -> AdamW 这条优化器演进线,每一步分别解决了什么问题?",
       a=(
           "SGD 每步只按当前 mini-batch 的梯度走,方向容易震荡、在峡谷或鞍点附近收敛慢。"
           "动量(Momentum)对历史梯度做指数滑动平均,用惯性抹平震荡、加速穿过平坦区。"
           "Adam 在动量(一阶矩)之上再维护梯度平方的滑动平均(二阶矩),对每个参数按其历史梯度尺度"
           "自适应地缩放步长,对稀疏梯度和不同尺度的特征更鲁棒、也更省调参;但如果像 SGD 那样把 L2 "
           "正则项加进梯度里,会被这个自适应缩放意外地按参数扭曲。AdamW 的解决办法是解耦权重衰减——"
           "把权重衰减从梯度更新里剥离出来,直接以固定比例乘在权重上,不受自适应缩放影响,这才是目前"
           "大模型训练的默认选择。代价是 Adam/AdamW 都要额外维护一阶矩和二阶矩两份状态,相当于多存"
           "两倍参数量的优化器状态,显存开销比 SGD/动量法更大。"
       ),
       keys=("动量", "一阶矩", "二阶矩", "自适应", "解耦"),
       follow_ups=("AdamW 的解耦具体怎么实现?公式层面写一下。", "Adam 为什么需要 bias correction?",
                   "为什么大模型预训练几乎都用 AdamW 而不是 SGD?")),

    QA(id="ai-opt-02", cat=CAT,
       q="RAdam 提出的'方差修正'视角怎么解释 warmup 为什么有效,又是如何让训练可以不用手工 warmup 的?",
       a=(
           "RAdam(Rectified Adam)指出 warmup 起作用的根本原因是:训练早期能用来估计二阶矩(自适应"
           "学习率)的样本太少,导致自适应学习率的方差过大、更新不稳定;warmup 用小学习率把这段"
           "'样本不足、方差大'的窗口期熬过去,等统计量稳定了再放大步长。RAdam 把这个直觉变成解析解——"
           "它显式估计自适应学习率的方差,在训练早期用一个修正项(rectification)按理论方差把步长压小,"
           "方差过大时自动退化到接近 SGD 的行为,方差稳定后逐渐恢复到标准 Adam 更新,因此不再需要"
           "手工设计 warmup 步数这个超参数。"
       ),
       keys=("方差", "二阶矩", "自适应学习率", "修正", "warmup"),
       follow_ups=("warmup 的步数一般怎么设置(线性/常数)?",
                   "工业界为什么大多还是用手工 warmup 而不是 RAdam?",
                   "Post-LN Transformer 为什么也需要 warmup(和优化器方差解释是同一回事吗)?")),

    QA(id="ai-opt-03", cat=CAT,
       q="AdamW 的解耦权重衰减具体怎么实现?为什么最优 weight decay 大小和训练步数有关?",
       a=(
           "标准 Adam+L2 的做法是把 λθ 加进梯度里再走自适应更新,于是权重衰减也被逐参数的二阶矩缩放,"
           "更新频繁的参数反而被正则得更轻,和'该管住谁'的直觉相反。AdamW 的解耦做法是:先用干净梯度"
           "算出 Adam 的自适应更新量 Δ,再单独执行 θ <- θ - lr·Δ - lr·λ·θ,让权重衰减以统一比例直接"
           "作用在权重上,不经过自适应缩放,恢复出等价于 SGD 里 L2 的'全局均匀收缩'效果。论文的实验还"
           "发现,总的训练步数越多,最优的 weight decay 系数应该越小——因为权重衰减是每一步都乘一次的"
           "连乘收缩,步数越多、同样的 λ 累积收缩效果越强,所以要相应调低 λ 才能保持整体正则强度匹配。"
       ),
       keys=("解耦", "自适应", "L2", "全局均匀", "训练步数"),
       follow_ups=("SGD 上 L2 和 weight decay 为什么是等价的?",
                   "AdamW 的 weight decay 一般设多大,和学习率怎么联动调?",
                   "weight decay 对 Adam 的二阶矩估计有没有影响?")),

    QA(id="ai-opt-04", cat=CAT,
       q="梯度裁剪(gradient clipping)里按范数裁剪和按值裁剪有什么区别?各自的副作用是什么?",
       a=(
           "按值裁剪把每个梯度分量独立截断到 [-c, c] 区间,实现最简单,但会破坏梯度向量的方向——"
           "不同分量被裁掉的比例不同,相当于扭曲了下降方向。按范数裁剪先算整个梯度向量(或某一层)的 "
           "L2 范数,若超过阈值就整体等比例缩放回阈值,方向完全保持不变,只压缩长度,是 Pascanu 等人"
           "提出、目前更常用的方案,尤其在 RNN/Transformer 训练中防止梯度爆炸导致的 loss 突刺。按范数"
           "裁剪的副作用是阈值选得太小会人为限制大梯度样本的更新幅度、拖慢收敛;两者都只是治标(压住"
           "已经放大的梯度),治本仍要靠残差连接、归一化、更好的初始化去抑制梯度本身的爆炸倾向。"
       ),
       keys=("按值裁剪", "按范数", "L2 范数", "方向", "梯度爆炸"),
       follow_ups=("为什么 RNN 比 Transformer 更容易梯度爆炸?", "梯度裁剪的阈值一般怎么选?",
                   "梯度裁剪会影响 Adam 的二阶矩估计吗?")),

    QA(id="ai-opt-05", cat=CAT,
       q="linear scaling rule 是什么?大 batch 训练为什么要相应放大学习率,这条经验规则在什么情况下会失效?",
       a=(
           "linear scaling rule(线性缩放法则,来自 Goyal 等 2017 年 ImageNet 大 batch 训练的工作)说:"
           "把 batch size 放大 k 倍时,把学习率也同比放大 k 倍,同时配合几个 epoch 的 warmup,就能在大 "
           "batch 下维持和小 batch 相近的收敛轨迹和最终精度。直觉是:batch 变大后单步梯度估计的方差"
           "降低,每步更'准',但如果学习率不变,相同 epoch 数下总的参数更新步数会因为每个 epoch 步数"
           "变少而变小,所以要把学习率同比放大来补偿,让参数空间里走过的'总路程'大致不变。这条规则在 "
           "batch size 大到一定程度后会失效——超过某个 critical batch size(临界 batch size)后梯度里"
           "的噪声本来就不大,再放大学习率会导致训练不稳定甚至发散,此时只能改用更长的训练时间或 "
           "LARS/LAMB 这类按层自适应学习率的优化器来维持大 batch 下的可训练性。"
       ),
       keys=("linear scaling rule", "batch size", "方差", "warmup", "critical batch size"),
       follow_ups=("LARS/LAMB 是怎么做到按层自适应学习率的?", "batch size 变大对泛化(尖锐极小值)的影响是什么?",
                   "critical batch size 怎么估计?")),

    QA(id="ai-opt-06", cat=CAT,
       q="cosine annealing、step decay、linear warmup+decay 这几种学习率调度分别怎么设计,适合什么场景?",
       a=(
           "step decay 每隔固定 epoch 数把学习率乘一个衰减因子(如每 30 epoch 乘 0.1),简单直观,常见于"
           "传统 CV 分类训练,缺点是衰减点是硬边界、需要人工试探;cosine annealing 让学习率按余弦曲线"
           "从初始值平滑降到接近 0,前期下降慢、后期加速收敛到低点,不需要设衰减点,是当前预训练/微调的"
           "主流选择;linear warmup+decay 是把两段拼起来——先用少量 step 线性从 0 升到峰值学习率"
           "(warmup,稳住训练早期),再线性(或余弦)衰减到 0,大模型预训练(如 BERT/GPT 系列)几乎都用"
           "这套组合。三者的共同目标都是'早期别走太快、后期慢慢收敛到稳定点',区别只在中间衰减的形状。"
       ),
       keys=("step decay", "cosine annealing", "linear warmup", "余弦", "预训练"),
       follow_ups=("为什么 cosine annealing 比 step decay 更常用?", "warmup 的长度一般怎么设置(占总步数比例)?",
                   "one-cycle policy 和这几种调度有什么关系?")),

    QA(id="ai-opt-07", cat=CAT,
       q="Nesterov 动量(NAG)和普通动量(heavy-ball momentum)的区别是什么?",
       a=(
           "普通动量(heavy-ball)在当前参数位置算梯度,再和历史动量方向叠加后一起走一步,本质是'先看"
           "后走'的滞后修正。Nesterov 动量则是'先走后看'——先按历史动量方向往前探一步(look-ahead 位置),"
           "在这个探出去的位置上算梯度,再用这个更靠前的梯度修正动量方向,相当于提前预判了动量会把参数"
           "带到哪里、提前纠偏。这让 Nesterov 在接近极小值时刹车更及时,减少了普通动量常见的振荡和越过"
           "最优点的过冲(overshoot),收敛速度理论上有更好的保证。"
       ),
       keys=("先走后看", "look-ahead", "过冲", "动量", "收敛"),
       follow_ups=("Nesterov 动量在深度学习框架里怎么实现(和普通动量的代码差异)?",
                   "Adam 里有没有用到 Nesterov 的思想(Nadam)?",
                   "为什么大模型训练现在很少见到显式的 Nesterov 动量?")),

    QA(id="ai-opt-08", cat=CAT,
       q="Adam 更新公式里的偏差修正(bias correction)是为了解决什么问题?",
       a=(
           "Adam 的一阶矩 m 和二阶矩 v 都是从全零初始化开始做指数滑动平均,在训练刚开始的几步,滑动"
           "平均还没'攒够'历史数据,估计值会被系统性地拉向 0、比真实的梯度矩偏小,尤其当 β1、β2 接近 1"
           "(如 0.9、0.999)时这个偏置更明显。偏差修正把 m_t、v_t 分别除以 (1-β1^t)、(1-β2^t) 得到 "
           "m_hat、v_hat,t 越小修正力度越大、随着 t 增大修正项趋近于 1、自动退化为不修正,从而消除了"
           "初始化在零点带来的低估偏置,让训练最早几步的步长也是准确的。"
       ),
       keys=("一阶矩", "二阶矩", "滑动平均", "偏置", "偏差修正"),
       follow_ups=("β1、β2 一般怎么取值,调大调小分别有什么影响?", "如果不做偏差修正,训练早期会发生什么现象?",
                   "RAdam 和 bias correction 解决的是同一个问题吗?")),

    QA(id="ai-opt-09", cat=CAT,
       q="Adam 更新公式里的 epsilon 起什么作用?调得过大或过小分别有什么后果?",
       a=(
           "Adam 的参数更新是 θ <- θ - lr·m_hat/(√v_hat + ε),ε 加在分母里纯粹是为了数值稳定——防止"
           "某个参数的二阶矩估计 v_hat 接近 0 时除法爆出极大的更新量甚至除零。ε 调得过小(如 1e-8 以下)"
           "在 v_hat 很小的方向上更新步长会被过度放大,可能导致训练在某些参数上突然震荡或发散;ε 调得"
           "过大(如 1e-4 以上)则会在 v_hat 本身较小的方向上抹平自适应缩放的效果,让 Adam 退化得更接近"
           "普通 SGD、损失自适应带来的收益,这也是有些工作(如混合精度训练)发现需要把 ε 调大以避免数值"
           "下溢时顺带观察到收敛变慢的原因。"
       ),
       keys=("数值稳定", "分母", "二阶矩", "自适应缩放", "除零"),
       follow_ups=("混合精度训练下为什么经常要调大 epsilon?", "epsilon 和学习率是否存在耦合、需要一起调吗?",
                   "AdamW 论文里 epsilon 一般怎么设置?")),

    QA(id="ai-opt-10", cat=CAT,
       q="怎么从训练 loss 曲线判断学习率设得过大还是过小?",
       a=(
           "学习率过大的典型症状:loss 曲线剧烈震荡、忽高忽低甚至出现尖峰(spike)后不回落,严重时直接"
           "发散成 NaN/Inf,或者虽然不发散但在低点附近来回跳、迟迟不收敛到更低的值;这是因为步子迈得"
           "比损失面的曲率半径还大,一步就跨过了最优点甚至跨到更差的区域。学习率过小的典型症状:loss "
           "下降极其平缓、几乎是一条几近水平的直线,或提前进入一个远高于预期的平台期就不再下降,同样"
           "的训练步数内明显落后于正常设置;这是因为每步移动量太小,还没走到损失面的陡峭区域就被训练"
           "步数耗尽。实践中常用 LR range test(学习率从很小指数增大、观察 loss 何时开始上升)来快速"
           "定位合适的学习率区间。"
       ),
       keys=("震荡", "发散", "尖峰", "平台期", "LR range test"),
       follow_ups=("LR range test 具体怎么操作?", "loss 出现 NaN 除了学习率过大还有哪些常见原因?",
                   "学习率过小和欠拟合怎么区分?")),

    QA(id="ai-opt-11", cat=CAT,
       q="二阶优化器(牛顿法、K-FAC、Shampoo)相比一阶方法(SGD/Adam)优势在哪?为什么大模型训练很少直接用?",
       a=(
           "一阶方法只用梯度信息,对损失面的曲率一无所知,在病态曲率(某些方向陡、某些方向平)的区域"
           "收敛慢、需要精心调学习率;二阶方法额外利用 Hessian(或其近似)信息,按曲率对每个方向单独"
           "缩放步长,理论上收敛更快、对学习率也更不敏感——牛顿法用完整 Hessian 求逆,K-FAC 用 "
           "Kronecker 分解近似 Fisher 信息矩阵降低求逆代价,Shampoo 用张量结构的预条件矩阵近似曲率。"
           "但大模型参数量动辄百亿千亿,Hessian 是参数量平方级别的矩阵,即使是 K-FAC/Shampoo 这类近似,"
           "每步的矩阵求逆/分解开销和额外显存仍远高于 Adam 的逐元素运算,工程实现也复杂得多,收益在"
           "超大规模下往往不足以抵消额外的算力和显存成本,所以目前一阶自适应方法(AdamW 及其变种)"
           "仍是大模型训练的主流。"
       ),
       keys=("Hessian", "曲率", "K-FAC", "Shampoo", "显存"),
       follow_ups=("K-FAC 具体怎么用 Kronecker 分解近似 Fisher 信息矩阵?",
                   "Shampoo 在工业界有没有实际大规模应用的例子?",
                   "为什么说 Adam 本质上是一种对角近似的二阶方法?")),

    QA(id="ai-opt-12", cat=CAT,
       q="为什么在某些场景(如 CV 分类)SGD 训练出的模型泛化反而优于 Adam?",
       a=(
           "经验和理论都观察到,Adam 类自适应方法虽然收敛更快、训练 loss 降得更低,但在图像分类等任务"
           "上测试集泛化有时不如调好的 SGD+Momentum,这被称为'自适应方法的边际价值'问题。一种解释是 "
           "Adam 的逐参数自适应步长会让优化轨迹更容易收敛到损失面上尖锐(sharp)的极小值——这类极小值"
           "对参数扰动敏感、在训练/测试分布略有差异时误差上升快;而 SGD 因为噪声结构和各向同性的步长,"
           "更容易被'挤'向宽平(flat)的极小值,这类极小值鲁棒性更好、泛化误差更低。这也是为什么很多 CV "
           "论文在预训练收尾阶段会切换回 SGD 或用 AdamW 但配合更强的权重衰减和数据增强来弥补泛化差距,"
           "而 NLP/Transformer 任务里 Adam 系列几乎是唯一选择(因为其对稀疏梯度和不同尺度特征更鲁棒的"
           "优势更关键)。"
       ),
       keys=("边际价值", "尖锐", "宽平", "泛化", "自适应"),
       follow_ups=("'尖锐极小值 vs 宽平极小值'和泛化的关系有没有争议(sharpness 的定义依赖于参数化)?",
                   "SWA(随机权重平均)是怎么帮助找到更宽平极小值的?",
                   "为什么 Transformer 训练几乎不用纯 SGD?")),
]


def _self_test() -> None:
    assert 10 <= len(BANK) <= 15, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-opt-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_optimization: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
