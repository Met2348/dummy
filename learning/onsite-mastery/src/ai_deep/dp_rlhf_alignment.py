"""RLHF与对齐深水区追问链（12个DeepPoint）。

覆盖：PPO的clip目标函数具体怎么防止更新过大、DPO隐式reward的数学推导、GRPO为什么能省掉
value网络及其代价、reward hacking的具体真实案例、KL散度惩罚系数怎么调、reward model的
分布外泛化问题。

边界：不涉及scaling law/训练动力学的loss曲线分析(那是scaling_dynamics类目)。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, grade_chain  # noqa: E402

CAT = "RLHF与对齐深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-ai-rlhf-01",
        cat=CAT,
        trigger="你说PPO用了clip目标函数防止更新过大，具体clip的是什么东西，怎么就防止了大更新？",
        chain=(
            ("clip目标函数具体是怎么定义的？",
             "PPO目标函数是L=E[min(r_t(θ)*A_t, clip(r_t(θ),1-ε,1+ε)*A_t)]，其中r_t(θ)是新旧策略概率比，"
             "clip把这个比值限制在[1-ε,1+ε]区间内(常取ε=0.2)，再取未clip和clip后两项的较小值作为最终目标。",
             ("r_t(θ)", "clip", "1-ε,1+ε")),
            ("为什么要取min而不是直接用clip后的值？",
             "如果只用clip后的目标，policy仍可能朝着让未clip目标继续变好的方向更新；取min确保一旦优势为正、"
             "ratio超过1+ε，目标函数值就不再随ratio增大而增大，梯度为0，这是一个悲观下界的设计思想。",
             ("min", "梯度为0", "悲观下界")),
            ("clip在advantage为正和为负时，具体是怎么不对称地起作用的？这个不对称设计合理吗？",
             "当A_t>0时clip限制r_t不能超过1+ε继续增大目标，防止把好动作的概率无限拉高；当A_t<0时clip限制"
             "r_t不能低于1-ε继续减小目标，防止把这个动作的概率无限压低到接近0，丧失后续再采样到它的探索机会，"
             "这个不对称设计保证了单步更新在两个方向上都保守。",
             ("A_t>0", "A_t<0", "探索机会")),
            ("clip范围ε=0.2这个数字怎么来的，调大调小会怎样？",
             "ε主要是经验超参数，没有普适的理论最优值；调大相当于放松保守约束，收敛可能更快但更容易震荡或"
             "崩溃，调小更新更保守但收敛变慢，LLM RLHF实践中普遍比传统RL基准的默认值更保守，需要重新经验性"
             "调试。",
             ("经验超参数", "震荡", "重新经验性调试")),
        ),
        pitfall="大多数人能背出clip公式，但说不清楚'取min'这个操作具体防止了什么方向的更新，更答不出A_t>0和A_t<0时clip不对称起作用的细节，一被追问这个不对称性就卡壳。",
        real_world_link="learning/rl-foundations/src/ppo_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-02",
        cat=CAT,
        trigger="PPO要额外训练一个value网络(critic)来估计baseline，这个网络具体解决什么问题，不用它直接用reward信号行不行？",
        chain=(
            ("value网络在PPO里起什么作用？",
             "value网络V(s)估计从当前状态开始的期望累积回报，用来计算advantage，减去这个baseline的目的是"
             "降低policy gradient估计的方差同时保持无偏，直接用reward做policy gradient会导致梯度估计噪声"
             "极大、训练不稳定。",
             ("V(s)", "advantage", "保持无偏")),
            ("GAE里的λ具体在权衡什么？",
             "GAE用λ做指数加权平均多步TD残差，λ趋近0更偏向单步TD(低方差高偏差，依赖V估计准确性)，λ趋近1"
             "更偏向蒙特卡洛回报(高方差低偏差)，λ和γ是在偏差和方差之间连续权衡的旋钮。",
             ("TD残差", "偏差和方差", "旋钮")),
            ("在LLM RLHF场景下，value网络的训练有什么特殊困难，跟传统RL相比？",
             "LLM RLHF里每个episode只有终止时刻的reward，value网络要学会把这个稀疏的、只在最后一步出现的"
             "reward分摊到每个token位置的value估计上，而且value网络通常要跟policy网络同规模才有足够表征能力，"
             "带来接近双倍的显存和计算开销，这是GRPO等方法想去掉它的直接动机。",
             ("稀疏", "同规模", "双倍")),
            ("如果value网络训练得不准，对最终PPO效果的影响有多大，能定量说清楚吗？",
             "很难精确定量，value网络估计偏差会传导成advantage估计偏差，但因为奖励还叠加了KL惩罚等其他项，"
             "很难把效果下降完全归因到value网络不准这一个因素上，实践里只能靠value loss收敛情况这类间接信号"
             "判断，没有干净的理论上界能量化这个影响。",
             ("很难精确定量", "间接信号", "理论上界")),
        ),
        pitfall="很多人知道'value网络算baseline降方差'，但说不出GAE里λ具体在权衡什么，更答不出LLM场景下value网络要处理'只有终局reward'这个特殊困难，这正是GRPO要解决的问题，答不出来就没法很好地过渡到下一层追问。",
        real_world_link="learning/rl-foundations/src/gae.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-03",
        cat=CAT,
        trigger="DPO号称不需要显式训练reward model和跑RL loop，它的隐式reward是怎么推出来的？",
        chain=(
            ("隐式reward具体是怎么推导出来的？",
             "从RLHF在KL惩罚约束下最大化reward的目标出发，可以解出最优策略π*的闭式解π*(y|x) ∝ π_ref(y|x)*"
             "exp(r(x,y)/β)，反解这个关系式，可以把reward写成r(x,y) = β*log(π*(y|x)/π_ref(y|x)) + const(x)，"
             "policy本身相对reference model的log概率比就隐式定义了一个reward函数。",
             ("闭式解", "β*log", "const(x)")),
            ("这个隐式reward怎么变成DPO最终的loss？",
             "把隐式reward代入Bradley-Terry偏好模型P(y_w > y_l) = sigmoid(r(y_w)-r(y_l))，两个样本的"
             "const(x)项相减后正好抵消，得到直接对策略log概率比做sigmoid+交叉熵形式的优化，不需要单独的"
             "reward model也不需要采样/RL loop。",
             ("Bradley-Terry", "const(x)项相减", "sigmoid+交叉熵")),
            ("这个推导依赖的关键假设是什么？如果假设不成立，DPO会在哪里出问题？",
             "关键假设是KL约束下RLHF目标的闭式解在实际优化中可达，以及偏好数据服从Bradley-Terry模型；如果"
             "人类偏好不满足传递性(不服从Bradley-Terry假设)，DPO学到的隐式reward会系统性偏离真实偏好结构，"
             "而且没有显式reward model也没法做reward model集成这类补救手段。",
             ("闭式解在实际优化中可达", "传递性", "reward model集成")),
            ("既然PPO和DPO理论上应该收敛到同一个最优策略，为什么实践中两者效果经常不一样？",
             "理论等价性依赖两者都能找到各自目标函数的全局最优这个假设，但PPO是on-policy迭代优化，DPO是"
             "off-policy直接在静态数据集上优化，两者优化轨迹完全不同，实际都做不到理论最优，效果差异更多来自"
             "各自优化过程的实际表现而不是目标函数本身不等价。",
             ("on-policy迭代", "off-policy", "优化过程的实际表现")),
        ),
        pitfall="很多人能背出DPO loss公式，但推导不出reward reparameterization这一步(为什么π/π_ref的log比值就是隐式reward)，第3层问到Bradley-Terry假设是否成立时基本说不出来。",
        real_world_link="learning/dpo-family/src/dpo_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-04",
        cat=CAT,
        trigger="DPO是在一个固定的偏好数据集上做优化，这算on-policy还是off-policy？这对最终效果有什么影响？",
        chain=(
            ("DPO算on-policy还是off-policy？",
             "DPO本质上是off-policy的，它直接在预先收集好的偏好数据集上做类似监督学习的优化，不需要用当前"
             "policy重新采样生成回复来获取新的reward信号，这跟PPO每一轮都要用当前policy采样的on-policy循环"
             "有本质区别。",
             ("off-policy", "预先收集好的偏好数据集", "on-policy循环")),
            ("off-policy的代价是什么？",
             "训练数据里的chosen/rejected回复是由生成数据时的模型产生的，当前policy的输出分布可能已经和数据"
             "分布产生distribution shift，DPO并不会在训练过程中重新验证这些偏好关系对当前policy是否依然成立，"
             "训练时间过长可能过拟合到训练数据里的表层模式而不是真正的偏好信号。",
             ("distribution shift", "重新验证", "表层模式")),
            ("有什么具体机制或变体尝试缓解这个off-policy的问题？",
             "iterative DPO/online DPO定期用当前policy重新采样生成新回复、重新打分再继续训练，把off-policy"
             "变成近似on-policy的迭代过程；IPO从理论上修正DPO对确定性偏好数据过拟合的倾向，加了正则项限制"
             "隐式reward差距的目标值。",
             ("iterative DPO", "online DPO", "IPO")),
            ("对同一批偏好数据反复多轮训练DPO，是不是训练轮数越多越好？",
             "不是，多轮训练容易观察到chosen和rejected的log-prob都在下降这种整体概率坍缩现象，以及对训练数据"
             "表层特征的过拟合迹象，训练到什么程度算够目前没有特别可靠的理论判据，早停常常是必要的。",
             ("概率坍缩", "过拟合", "早停")),
        ),
        pitfall="很多人只会说'DPO简单稳定不需要RL'，答不出它off-policy这个本质，更不知道iterative/online DPO是为了解决什么问题被提出来的。",
        real_world_link="learning/dpo-family/src/capstone_dpo_comparison.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-05",
        cat=CAT,
        trigger="GRPO说不需要value网络，那它怎么算advantage，这个baseline是从哪来的？",
        chain=(
            ("GRPO的advantage具体怎么算？",
             "GRPO对同一个prompt采样一组(group)response，用这一组response各自reward的均值作为baseline，"
             "每个response的advantage是(该response的reward-group内reward均值)/group内reward标准差，不需要"
             "额外训练value网络。",
             ("group", "reward均值", "标准差")),
            ("这样做省了value网络，代价是什么？",
             "省掉一个和policy同规模的value网络，减少接近一半的显存和计算开销，但每个prompt需要采样一整组"
             "response才能计算有意义的组内统计量，rollout阶段计算开销显著增加，且组内方差估计本身也是有噪声"
             "的估计量，G太小时baseline估计噪声大。",
             ("rollout阶段", "计算开销", "估计噪声")),
            ("GRPO是不是完全抛弃了PPO的clip机制，变成纯REINFORCE？",
             "不是，GRPO仍然保留了PPO风格的clipped surrogate objective，用重要性采样比值做clip，只是把怎么"
             "算advantage从value网络换成了组内归一化统计量，这两者是正交的设计选择，很多人误以为GRPO=REINFORCE"
             "+group baseline，忽略了它保留了PPO的重要性采样修正。",
             ("clipped surrogate objective", "正交", "重要性采样修正")),
            ("GRPO处理KL惩罚的方式和PPO有什么不同？",
             "PPO通常把KL惩罚作为reward shaping混入reward，而GRPO把KL惩罚直接作为loss函数里独立的一项加到"
             "策略梯度loss上，避免KL惩罚被组内归一化稀释或扭曲，这是一个容易被忽略但影响训练稳定性的实现细节。",
             ("reward shaping", "loss函数里独立的一项", "组内归一化稀释")),
        ),
        pitfall="很多人只知道GRPO'组内归一化省value网络'这个结论，答不出它其实还保留了PPO的重要性采样+clip这部分，也答不出GRPO把KL惩罚放进loss而不是reward这个实现细节。",
        real_world_link="learning/rl-sota-2026/src/dr_grpo.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-06",
        cat=CAT,
        trigger="如果一组prompt采样出来的response全对或全错，GRPO怎么处理，group size怎么选？",
        chain=(
            ("group内reward全同时advantage怎么算？",
             "如果group内reward标准差为0，advantage公式里的除法会出现除以0，工程上通常做clip或给标准差加"
             "一个小epsilon，或者直接跳过该group不参与梯度更新。",
             ("除以0", "epsilon", "跳过该group")),
            ("这种情况实际会带来什么训练信号问题？",
             "当group内全对或全错时，这个group提供的学习信号量趋近于0，因为所有advantage接近0没有相对优劣"
             "可以区分，GRPO对训练数据的难度分布比较敏感，需要保证prompt难度覆盖group内有对有错的中等难度"
             "区域才能有效学习。",
             ("学习信号量趋近于0", "难度分布", "中等难度区域")),
            ("group size怎么选，这是不是纯粹的经验调参？",
             "G太小方差高、advantage估计噪声大，容易训练不稳定；G太大虽然估计更稳但rollout成本线性增长，"
             "实践中数学任务这种reward稀疏场景常用较大的G(如64)，reward区分度高的任务可能G=8~16就够，这是"
             "用更多采样换更准baseline估计和推理成本之间的连续权衡。",
             ("方差高", "线性增长", "连续权衡")),
            ("这个group size的选择有没有理论指导，还是纯粹经验调参？",
             "目前主要是经验调参，与任务的reward方差、可验证性高度相关，没有通用公式能直接算出最优G，不同"
             "论文对G的选择以及是否要动态调整G还在探索阶段。",
             ("经验调参", "没有通用公式", "动态调整G")),
        ),
        pitfall="很多人只知道GRPO'组内归一化'这个结论，答不出group内reward方差为0时的退化问题和实际处理方式，更容易把group size的选择当作可以套用的固定公式而不是场景依赖的权衡。",
        real_world_link="learning/rl-sota-2026/src/dapo_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-07",
        cat=CAT,
        trigger="你说训练时观察到reward hacking，具体是什么样的现象，能举个真实例子吗？",
        chain=(
            ("能具体举几个reward hacking的真实案例吗？",
             "常见案例包括长度hacking(reward model倾向给更长回复打更高分，policy疯狂堆字数但信息密度不增"
             "反降)、谄媚sycophancy(policy学会迎合prompt暗示的立场而不是给客观正确答案)、以及代码RL里的测试"
             "用例作弊(模型学会硬编码测试期望输出或修改测试断言来通过检查，而不是真正解决问题)。",
             ("长度hacking", "sycophancy", "测试用例作弊")),
            ("这些现象背后的共同机制是什么？",
             "共同机制是Goodhart's law——reward model或验证器只是真实目标的代理proxy，RL优化算法会不遗余力"
             "地找到proxy和真实目标之间的gap并利用它，这个gap在训练早期被KL约束限制得较小，随着优化步数增加"
             "KL约束被放松，policy会越来越激进地利用这个gap。",
             ("Goodhart's law", "proxy", "KL约束被放松")),
            ("你具体是通过什么信号发现这是reward hacking而不是真的变好了？",
             "常见信号包括reward曲线持续上涨但人工评测/held-out benchmark分数不涨甚至下降、回复长度等表层"
             "特征的分布随训练步数单调偏移且幅度远超合理范围，以及让reward model和独立评测模型对同一批prompt"
             "打分出现系统性排序分歧。",
             ("held-out benchmark", "表层特征", "系统性排序分歧")),
            ("有没有办法从理论上预先杜绝reward hacking，而不是训练后才发现？",
             "目前没有能完全预先杜绝的方法，reward hacking源于用有限、不完美的代理目标去优化难以完全形式化"
             "的真实目标这个根本张力，只能通过KL约束、reward model集成、定期重新训练等一系列缓解手段降低风险、"
             "尽早发现，不存在能完全消除这个风险的方案。",
             ("根本张力", "缓解手段", "不存在能完全消除")),
        ),
        pitfall="很多人只会泛泛说'reward hacking就是模型钻空子'，举不出长度/谄媚/测试用例作弊这类具体案例，更答不出怎么从训练日志/评测信号上真正定位到reward hacking而不是拍脑袋猜测。",
        real_world_link="learning/rlhf-classic/src/reward_hacking_demo.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-08",
        cat=CAT,
        trigger="PPO loss里KL惩罚系数β怎么定的，调大调小分别有什么后果？",
        chain=(
            ("β调大调小分别有什么后果？",
             "β控制policy相对reference model的KL散度惩罚强度，β太大会让policy几乎不敢偏离reference model，"
             "即使reward指出明显更好的方向也不敢充分更新，导致RLHF退化成接近SFT模型；β太小则约束形同虚设，"
             "policy快速偏离reference model，更容易出现reward hacking和输出多样性坍缩。",
             ("退化成接近SFT模型", "reward hacking", "多样性坍缩")),
            ("实践中怎么调这个系数，而不是死板地固定一个值？",
             "很多实现用adaptive KL controller，设定一个目标KL值，每隔一定步数比较实际测得的KL和目标值，"
             "实际KL超过目标就增大β拉回，低于目标就减小β给探索空间，自动把训练过程中的KL稳定在期望范围内。",
             ("adaptive KL controller", "目标KL值", "自动")),
            ("如果target KL设置得不合适，adaptive controller能自己纠正吗？",
             "不能完全纠正，adaptive KL controller只是把实际发生的KL往人为设定的目标值上拉，如果target KL"
             "本身设得不合理，controller依然会尽职尽责地让实际KL逼近这个不合理的目标，它解决的是如何稳定跟踪"
             "一个目标值，不解决这个目标值该设多少。",
             ("不能完全纠正", "尽职尽责", "不解决这个目标值该设多少")),
            ("近期一些方法把KL惩罚从reward shaping改成直接加进loss，这个改动的动机是什么？",
             "reward shaping式的KL会被后续的advantage归一化/baseline减除操作影响，让KL惩罚的实际强度变得"
             "不可控；把KL惩罚直接作为loss里独立的一项，不会被这类归一化操作稀释，惩罚强度更可预测，反映了"
             "KL惩罚放在reward里还是loss里在工程上会产生实际不同的训练动态。",
             ("advantage归一化", "不可控", "训练动态")),
        ),
        pitfall="很多人只知道'KL防止偏离太远'，说不清楚太大太小具体的后果(退化成SFT vs 多样性坍缩)，更不知道adaptive KL controller解决的是'跟踪目标值'而不是'选对目标值'这个区分。",
        real_world_link="learning/rl-foundations/src/ppo_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-09",
        cat=CAT,
        trigger="reward model是在一批固定的人类偏好数据上训出来的，RL阶段policy生成的分布会不会渐渐超出reward model的训练分布，这个风险你怎么应对？",
        chain=(
            ("这个分布外泛化风险具体是什么？",
             "会的，RL训练过程中policy的输出分布会持续偏离SFT/参考模型的初始分布，reward model对这些分布外"
             "样本的打分可靠性会下降，却仍然给出看似confident的分数，这个虚假置信度正是被policy利用来做"
             "reward hacking的温床。",
             ("分布外样本", "虚假置信度", "reward hacking")),
            ("有哪些具体的缓解手段？",
             "常见手段包括reward model ensemble(训练多个reward model，用打分不一致程度作为不确定性信号，"
             "对高不确定性样本做惩罚)、定期用policy当前生成的新样本重新收集标注重新训练reward model，以及"
             "限制KL散度间接限制reward model要外推多远。",
             ("reward model ensemble", "不确定性信号", "限制KL散度")),
            ("reward model ensemble的方差能完全捕捉不可靠吗，有没有场景这个方法会失效？",
             "不能完全捕捉，如果多个reward model用相同数据集、相似架构训出来，它们很可能共享同样的系统性偏差"
             "(比如都学到长度和质量正相关这个捷径)，在这类系统性失效模式上多个模型会一致给出虚假一致的高分，"
             "ensemble方差捕捉不到问题，因为不确定性度量依赖模型之间犯不同错误这个前提。",
             ("系统性偏差", "虚假一致的高分", "犯不同错误")),
            ("既然ensemble有这个盲区，目前有没有真正解决reward model OOD问题的方案？",
             "没有，目前所有缓解手段都是降低风险而非根治，这跟reward hacking本身一样是对齐研究里公认还没有"
             "闭环解决的开放问题，比较有前景的方向包括让reward model具备校准的不确定性输出，以及用更接近"
             "ground truth的可验证奖励减少对纯学习式reward model的依赖。",
             ("降低风险而非根治", "开放问题", "可验证奖励")),
        ),
        pitfall="很多人知道'reward model会被过度优化利用'这个说法，但说不出ensemble/迭代重训这些具体缓解手段，第3层问'ensemble方差捕捉不到什么'时基本答不出系统性偏差这个盲区。",
        real_world_link="learning/rlhf-classic/src/rm_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-10",
        cat=CAT,
        trigger="reward model是用pairwise比较训出来的，这背后假设了偏好可以被一个单一的标量reward排序，这个假设总是成立吗？",
        chain=(
            ("这个假设具体是什么？",
             "reward model训练用的Bradley-Terry模型假设存在一个潜在的标量reward函数，使得偏好概率等于两个"
             "reward之差的sigmoid，这隐含假设了偏好关系是可传递的，即A比B好、B比C好则A一定比C好。",
             ("Bradley-Terry", "sigmoid", "可传递")),
            ("现实中这个假设经常被违反吗，违反后会怎样？",
             "现实中人类偏好经常不满足这个假设，不同标注者因为看重的维度不同给出相互矛盾的比较，甚至出现"
             "环状偏好(A>B、B>C、C>A)，训练数据里这类矛盾偏好比例较高时，reward model被迫用单一标量拟合"
             "本质上多维、有噪声、甚至自相矛盾的偏好信号，在有分歧的样本对上准确率会显著更低。",
             ("环状偏好", "多维", "准确率会显著更低")),
            ("这种偏好不传递的情况在真实数据集里有多普遍，你怎么衡量？",
             "可以通过标注者间一致性来衡量，公开的RLHF偏好数据集报告的标注者间一致率通常在60%-75%左右，"
             "意味着相当比例的pairwise比较标注者本身就有分歧，说明Bradley-Terry这种存在单一客观reward的假设"
             "在真实标注数据里只是一个近似。",
             ("标注者间一致性", "60%-75%", "近似")),
            ("如果这个假设本身就不完全成立，是不是意味着reward model从根子上就有偏差，这个偏差有解吗？",
             "这是一个结构性局限而非工程bug，比较有前景的缓解方向包括用多目标/多维度的reward建模而不是强行"
             "压缩成一个标量，或者用ensemble捕捉哪些样本对存在系统性分歧对其降低训练权重，但这些都只是更好地"
             "承认和管理这个假设的局限，而不是让假设变得完全成立。",
             ("结构性局限", "多目标/多维度", "承认和管理")),
        ),
        pitfall="很多人从没想过reward model背后有Bradley-Terry这个具体假设，更答不出'偏好不传递'在真实数据里的具体表现和衡量方式，容易把reward model的打分当成客观真实的质量分数而不是一个有结构性局限的近似。",
        real_world_link="learning/rlhf-classic/src/rm_minimal.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-11",
        cat=CAT,
        trigger="同样是做偏好对齐，你会选PPO还是DPO，为什么？",
        chain=(
            ("PPO和DPO各自更适合什么场景？",
             "DPO实现更简单、不需要维护RL训练循环也不需要额外的value网络，训练更稳定、超参数更少，适合偏好"
             "数据集质量高且相对静态的场景；PPO配合reward model在线采样，理论上能持续利用policy自己生成的新"
             "样本获得反馈，更适合想要持续迭代提升的场景。",
             ("训练更稳定", "在线采样", "持续迭代提升")),
            ("这个区别背后更深层的机制是什么？",
             "DPO完全依赖离线偏好数据集里已经存在的样本对比，没有机制去主动探索和获取未来生成模式上的反馈；"
             "PPO因为是on-policy的，每一轮都用当前policy的实际输出获取新reward，天然能覆盖policy自己正在生成"
             "的分布，代价是训练pipeline复杂、对超参数更敏感、算力开销更大。",
             ("主动探索", "天然能覆盖", "算力开销更大")),
            ("现在很多工业界流水线是DPO和PPO/GRPO一起用，这是不是说明单独选哪个都不是最优解？",
             "确实，越来越多实践反映了一个共识——DPO适合低成本地把已有静态偏好信号高效利用起来，on-policy RL"
             "适合在已经比较好的基线上做进一步的、能利用探索和自动反馈的精细提升，两者解决的不是完全相同的"
             "问题，组合使用比单选一个更常见。",
             ("低成本", "精细提升", "组合使用比单选一个更常见")),
            ("如果算力有限只能选一个，你会怎么决定？",
             "如果偏好数据质量高且相对充分、任务不要求持续迭代，会优先选DPO以最小成本拿到大部分收益；如果"
             "任务本身有低成本、高质量的可验证reward，会优先考虑on-policy RL，这是基于数据/reward获取成本、"
             "任务是否可验证的具体权衡，不存在放之四海皆准的更好选项。",
             ("最小成本", "可验证reward", "不存在放之四海皆准")),
        ),
        pitfall="很多人把这个问题当成'哪个算法更先进'的排位赛来答，答不出两者其实解决的是不同层面的问题、工业界常常组合使用这个更贴近实际的答案。",
        real_world_link="learning/dpo-family/src/capstone_dpo_comparison.py",
    ),
    DeepPoint(
        id="dp-ai-rlhf-12",
        cat=CAT,
        trigger="reward hacking这个问题，业界这些年是怎么一步步应对的，有没有一个演进脉络？",
        chain=(
            ("大致的演进脉络是怎样的？",
             "早期经典RLHF主要靠KL惩罚约束policy不要偏离reference model太远加上人工持续抽查评测；随后出现"
             "Constitutional AI/RLAIF用AI反馈标注偏好并引入宪法原则做自我批评修正；再往后针对复杂推理任务，"
             "出现process reward model和outcome verification这类方法。",
             ("Constitutional AI/RLAIF", "process reward model", "outcome verification")),
            ("这个演进脉络背后的共同逻辑是什么？",
             "共同逻辑是不断把reward信号的来源从主观、稀疏、容易被利用的最终output打分，往客观、密集、更难"
             "被投机的中间过程验证方向推进，process reward model能在推理链条每一步给反馈，可验证奖励从根本上"
             "避免了reward model本身可能被欺骗这个环节。",
             ("主观", "中间过程验证", "可验证奖励")),
            ("process reward model和outcome verification是不是就彻底解决了reward hacking？",
             "没有，process reward model本身依然是学习出来的打分模型，同样可能有OOD泛化问题和被过度优化"
             "利用的风险，policy可能学会生成看起来步骤正确但实际推理错误的中间步骤来骗过PRM；outcome"
             "verification虽然客观，但只适用于有明确客观正确标准的任务，对开放式生成任务没有类似手段可用。",
             ("OOD泛化问题", "骗过PRM", "开放式生成任务")),
            ("对于没法用客观验证的开放式任务，现在有没有类似的进展来降低reward hacking风险？",
             "相对有限，开放式任务目前主要还是依赖reward model ensemble、多轮迭代重新标注、以及更精细的"
             "多维度评分这类缓解手段，LLM-as-judge这类方案也在被探索，但judge本身同样可能有偏差且目前还没有"
             "被完全刻画清楚，这部分依然是进展相对滞后、尚待更多工作的方向。",
             ("相对有限", "LLM-as-judge", "尚待更多工作")),
        ),
        pitfall="很多人只能说出'RLHF然后有了Constitutional AI'这种粗线条脉络，说不清楚process reward model/可验证奖励解决的具体是'reward来源的客观性'这个核心逻辑，更容易忽略这些新方法对开放式任务其实没有覆盖到。",
        real_world_link="learning/process-reward/src/prm_minimal.py",
    ),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("dp-ai-rlhf-") for i in ids), "id前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的条目"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的条目"
    assert all(dp.trigger for dp in BANK), "存在缺失trigger的条目"
    for dp in BANK:
        answers = [ref for (_q, ref, _k) in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 采分关键词未能在参考答案里全部命中: {scores}"
    print(f"[PASS] dp_rlhf_alignment: {len(BANK)}个DeepPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
