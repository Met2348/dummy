"""RLHF 与对齐八股问答库（约 14 题）。

与 interview-prep/src/mlqa/qbank.py 里 RL/RLHF 类已有的 6 道题（policy gradient 核心思想、
PPO clip 是什么、RLHF 三段流程、KL 惩罚为何必要、DPO 相比 PPO 优势、reward hacking）不重复，
这里写更细/不同角度：GRPO 怎么去掉 critic、reward model 具体训练方式、PPO clip 的具体数值
含义、SFT 数据构造、RLHF vs DPO 工程复杂度对比、拒绝采样、Constitutional AI/RLAIF、奖励过
优化等。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "RLHF与对齐"

BANK: list[QA] = [
    QA(id="ai-rlhf-01", cat=CAT,
       q="GRPO(Group Relative Policy Optimization)的核心思想是什么？它是怎么去掉 critic 网络的？",
       a="GRPO 放弃传统 PPO 里单独训练的价值网络(critic)，转而对同一个 prompt 用当前策略采样一组"
         "(group)输出(比如 G=8 或 16 条)，用奖励模型给这组输出打分，再把'相对这组内其他输出表现"
         "如何'作为优势(advantage)信号，而不是靠 critic 预测的状态价值 V(s) 来算优势。因为不再需要"
         "维护一个和策略同规模的价值网络，训练时同时驻留显存的模型从'policy+critic+reward+"
         "reference'四个降到少一个，显存和计算开销明显下降，这是 DeepSeekMath/DeepSeek-R1 采用它"
         "的直接动机。",
       keys=("critic", "group", "优势", "显存"),
       follow_ups=("组内的优势具体是怎么标准化计算的？", "GRPO 还保留了 PPO 的 clip 机制吗？")),

    QA(id="ai-rlhf-02", cat=CAT,
       q="GRPO 里组内优势(group advantage)具体是怎么算的？",
       a="对同一个 prompt 采样 G 条输出，用奖励模型（或规则奖励）算出对应的标量奖励 r_1...r_G，"
         "然后把每条输出的优势标准化为 A_i=(r_i-mean(r))/std(r)——减去组内均值、除以组内标准差。"
         "均值起到了'动态基线'的作用：奖励高于组内平均的输出优势为正、被推高概率，低于平均的为负、"
         "被压低；除以标准差则让优势的尺度不随奖励绝对数值大小或组内方差变化而剧烈波动，进一步稳定"
         "梯度信号。",
       keys=("标准差", "均值", "基线", "标准化"),
       follow_ups=("如果整组输出的奖励完全一样(std=0)会怎么样？", "组大小 G 怎么选，太小会有什么问题？")),

    QA(id="ai-rlhf-03", cat=CAT,
       q="传统 PPO 里为什么需要 critic 网络？GRPO 去掉它之后，训练时同时要驻留几个模型？",
       a="传统 PPO 用 critic(价值网络)去估计状态价值 V(s)，再结合实际奖励算优势 A=Q-V(常用 "
         "GAE)，目的是降低优势估计的方差、让梯度更稳定；但 critic 通常和 policy 同等规模，训练/"
         "推理都要多算一份前向反向，显存和算力翻倍。标准 RLHF-PPO 训练时通常同时驻留四个模型："
         "policy(待训练)、reference(算 KL 用，冻结)、reward model(打分，冻结)、critic(价值网络，"
         "待训练)；GRPO 去掉 critic 后只剩 policy、reference、reward model 三个，等价于省掉了一整"
         "个和策略同规模的网络的训练开销与显存占用。",
       keys=("价值网络", "GAE", "四个模型", "三个"),
       follow_ups=("reference 模型在训练中起什么作用？", "省掉 critic 会不会让优势估计方差变大？怎么弥补？")),

    QA(id="ai-rlhf-04", cat=CAT,
       q="reward model 具体是怎么训练出来的？",
       a="reward model 通常从 SFT 模型初始化，把最后的语言建模头换成一个输出标量的线性层。训练"
         "数据是人类(或 AI)对同一个 prompt 下多条候选回复的偏好排序/两两比较(chosen vs rejected)。"
         "用 Bradley-Terry 模型把'偏好概率'和'奖励差'联系起来：P(chosen优于rejected)="
         "sigmoid(r(chosen)-r(rejected))，训练目标是最大化这个似然，等价于最小化 loss="
         "-log(sigmoid(r_chosen-r_rejected))，也就是让奖励模型给人类更喜欢的回复打出明显更高的分数。",
       keys=("标量", "Bradley-Terry", "sigmoid", "偏好"),
       follow_ups=("如果标注者给了 K>2 个回复的整体排序，怎么转成训练用的 pairwise 数据？", "reward model 训练完之后，它的分数是绝对可比的吗？")),

    QA(id="ai-rlhf-05", cat=CAT,
       q="PPO 里 clip(ε) 的具体数值一般取多少？它裁剪的是什么量、作用范围是什么？",
       a="clip 作用在新旧策略的概率比 r_t(θ)=π_θ(a|s)/π_θ_old(a|s) 上，把它限制在 [1-ε, 1+ε] "
         "这个区间内；ε 常取 0.1~0.2，InstructGPT 等 RLHF 实现里典型取值就是 0.2。具体做法是目标"
         "函数取 min(r_t·A_t, clip(r_t,1-ε,1+ε)·A_t)：当优势 A_t>0 时，即使 r_t 超过 1+ε 也不会"
         "让目标继续增大(裁掉多余的鼓励)；当 A_t<0 时同理限制 r_t 跌破 1-ε 后的惩罚幅度。本质是给"
         "每一步策略更新画一个'信任域'，防止单步走太远导致新策略和旧策略差异过大而训练崩溃。",
       keys=("概率比", "1-ε", "0.2", "信任域"),
       follow_ups=("clip 和 KL 惩罚是两种独立约束还是同一个东西？", "如果去掉 clip 只留 KL 惩罚会怎样？")),

    QA(id="ai-rlhf-06", cat=CAT,
       q="PPO 的 clip 机制和 KL 惩罚是两种独立约束吗？两者分别管什么？",
       a="是两种互补但不同层面的约束，很多实现(如 InstructGPT)会同时使用。clip 作用在单次梯度更新"
         "内部：限制这一步更新里新旧策略概率比的变化幅度，防止某一步 optimizer 走太猛；KL 惩罚"
         "(KL(π_θ||π_ref))作用在整条训练轨迹的宏观尺度上：把当前策略和(通常是 SFT)参考策略的分布"
         "距离作为惩罚项加进 reward 或 loss 里，防止训练很多步之后策略整体上偏离参考模型太远、钻进"
         "奖励模型的漏洞里(reward hacking)。去掉 clip 只留 KL，单步更新可能仍然过大导致不稳定；"
         "去掉 KL 只留 clip，长期训练仍可能逐步漂移到奖励模型评分很高但语言质量很差的区域。",
       keys=("clip", "KL", "单次梯度更新", "参考策略"),
       follow_ups=("KL 惩罚具体是加在 reward 里还是加在 loss 里？两种实现有什么区别？", "GRPO 论文里 KL 项的计算方式和标准 PPO 一样吗？")),

    QA(id="ai-rlhf-07", cat=CAT,
       q="RLHF 第一阶段 SFT 用的数据一般是怎么构造的？",
       a="SFT(监督微调)数据通常是高质量的'指令-回复'示范对：一部分由人类标注员针对典型指令直接"
         "手写高质量回复，一部分从已上线模型的真实用户请求里挑选、再让标注员编辑打磨，还有一部分用"
         "更强模型(或人工)生成候选后再筛选/改写。构造时特别强调覆盖面(任务类型、难度、语气多样)和"
         "质量(标注员通常需要专门培训、按详细的标注规范打分)，因为 SFT 阶段学到的'回复风格基线'"
         "会直接决定后续奖励模型和 RL 阶段的起点，示范数据的分布偏差会被后续阶段放大。",
       keys=("指令-回复", "标注员", "覆盖面", "起点"),
       follow_ups=("SFT 数据量一般在什么量级？和预训练数据量比差多少？", "SFT 阶段会不会也用模型自己生成的数据(自蒸馏)？")),

    QA(id="ai-rlhf-08", cat=CAT,
       q="RLHF(PPO 路线)和 DPO 在工程复杂度和训练稳定性上具体差在哪？",
       a="PPO 路线要同时训练/服务 policy、reference、reward model(至少三个大模型常驻显存，标准"
         "实现还有 critic 共四个)，需要在线(on-policy)不断从当前策略采样新回复、过 reward model "
         "打分，是一套'采样-打分-更新'循环的 RL 基础设施，超参(学习率、clip ε、KL 系数、batch "
         "size)敏感，训练容易震荡甚至崩溃，需要专门的 RL 调参经验。DPO 只需要 policy 和 reference "
         "两个模型，直接在已经收集好的离线偏好数据集上用一个类似交叉熵的闭式 loss 训练，不需要在线"
         "采样、不需要单独的 reward model、不需要 RL 循环，工程上接近标准的监督微调，超参也少得多、"
         "训练曲线通常更平滑。代价是 DPO 只能在固定的离线偏好数据分布上优化，没有 PPO 那种'策略变了"
         "就重新采样验证'的在线闭环。",
       keys=("四个", "离线", "闭式", "在线采样"),
       follow_ups=("DPO 的 loss 具体长什么样，为什么说它对应一个隐式奖励？", "有没有办法把 DPO 做成'在线'版本弥补这个缺陷？")),

    QA(id="ai-rlhf-09", cat=CAT,
       q="什么是拒绝采样(rejection sampling)微调？在对齐流程里是怎么用的？",
       a="拒绝采样微调(如 Llama 2 的 RFT)是指：对同一个 prompt 用当前策略采样 K 条候选回复(K "
         "常取几十)，用奖励模型给这 K 条打分，只挑出分数最高的一条(或前几条)当作监督数据，再对策略"
         "做标准的 SFT 式训练(而不是走 PPO 那种在线策略梯度更新)。它相当于把'inference 阶段的 "
         "best-of-K 采样'蒸馏进模型参数里：每一轮用当前策略生成、用奖励模型筛选、再监督微调，反复"
         "迭代几轮，让模型逐步学会自己直接生成当初需要采样 K 次才能得到的高分回复。",
       keys=("best-of-K", "奖励模型", "蒸馏", "SFT"),
       follow_ups=("拒绝采样微调和标准 PPO 比，各自的优缺点是什么？", "Llama 2 是怎么把这套方法和 PPO 结合起来分阶段用的？")),

    QA(id="ai-rlhf-10", cat=CAT,
       q="拒绝采样微调相比标准 PPO，优缺点分别是什么？",
       a="优点：不需要在线策略梯度、不用调 PPO 那一堆敏感超参(clip ε、KL 系数)，训练过程更接近"
         "普通 SFT、更简单稳定，工程实现门槛低很多；相当于每轮用'当前策略的 best-of-N'当老师，"
         "逐步把策略往更好的方向拉，也不容易像 PPO 那样一步走太远导致崩溃。缺点：每轮都要对同一批 "
         "prompt 采样多条候选再打分，计算开销很大；而且只学习'组内最优的那一条'，丢弃了组内其他"
         "候选里的相对排序信息(不像 GRPO 那样利用了整组的相对优势)，样本利用效率不如带优势估计的 "
         "RL 方法。",
       keys=("超参", "计算开销", "最优的那一条", "效率"),
       follow_ups=("为什么说拒绝采样'不容易崩溃'，这个稳定性来自哪里？", "能不能把拒绝采样和 GRPO 的组内相对优势结合起来？")),

    QA(id="ai-rlhf-11", cat=CAT,
       q="Constitutional AI 是什么？它的两阶段分别做什么？",
       a="Constitutional AI(CAI，Anthropic 提出)用一份写明白的'宪法'(一组行为原则)取代大量人工"
         "标注来指导对齐，分两个阶段：①监督学习阶段——让一个初始(helpful-only)模型针对可能有害的"
         "提示生成回复，然后让模型自己依据宪法条款批评(critique)并修改(revise)自己的回复，反复"
         "几轮后用修改后的回复做监督微调；②RL 阶段——让模型对同一提示生成成对回复，同样依据宪法"
         "条款由 AI(而不是人类)判断哪个更好，把这些 AI 生成的偏好标注拿去训练奖励模型，再照常跑 "
         "RL。核心特点是'harmless 但不回避'：模型不是简单拒答敏感问题，而是给出符合原则的解释性"
         "回答。",
       keys=("宪法", "批评", "修改", "奖励模型"),
       follow_ups=("CAI 和 RLAIF 是什么关系？", "'harmless 但不回避'具体是什么意思，和简单拒答有什么区别？")),

    QA(id="ai-rlhf-12", cat=CAT,
       q="RLAIF 和 RLHF 的核心区别是什么？",
       a="两者流程框架相同(都是先构造偏好数据训练奖励模型，再用 RL 优化策略)，核心区别在于偏好"
         "标注的来源：RLHF 的偏好对由人类标注员打分/排序；RLAIF(Reinforcement Learning from AI "
         "Feedback)把这一步换成由另一个能力足够强的 AI模型(通常配合一份原则/宪法式的评判标准)来"
         "生成偏好判断。RLAIF 的动机是大规模人工标注成本高、周期长、标注者之间一致性也有限，用 AI "
         "打分可以显著降低成本、提升规模化速度，但也带来新问题——AI 打分本身的偏差(bias)和'奖励"
         "模型的奖励模型'式的误差传递需要额外校验。",
       keys=("偏好标注", "人类标注员", "AI模型", "成本"),
       follow_ups=("RLAIF 打分的一致性/偏差要怎么校验？", "RLAIF 是不是意味着完全不需要人类参与了？")),

    QA(id="ai-rlhf-13", cat=CAT,
       q="什么是奖励过优化(reward overoptimization)？为什么奖励模型分数涨了但真实回复质量反而"
         "下降？",
       a="奖励过优化指策略在 PPO 训练中被优化到奖励模型打分持续上升，但用更可靠的评估(人类评审/"
         "更强模型评审)看回复的真实质量却停滞甚至下降的现象。根源是奖励模型本身只是对'人类偏好'"
         "的一个有噪声、有偏差的近似，策略在强优化压力下会找到奖励模型的系统性弱点/盲区(比如偏爱"
         "啰嗦、堆砌关键词、谄媚)来刷分，这类似 Goodhart 定律('一旦一个指标变成优化目标，它就不再"
         "是好指标')。常见缓解手段包括：加大 KL 惩罚系数把策略锚在参考模型附近、用多个独立训练的"
         "奖励模型做集成投票、提前停止(early stopping)训练、以及扩大奖励模型训练数据的覆盖面减少"
         "其盲区。",
       keys=("奖励模型", "盲区", "Goodhart", "KL 惩罚"),
       follow_ups=("这跟'reward hacking'是不是一回事？", "有没有办法在训练过程中就监测到过优化正在发生？")),

    QA(id="ai-rlhf-14", cat=CAT,
       q="PPO(在线 RL)和 DPO(离线偏好优化)在样本利用/分布覆盖上有什么本质差异？",
       a="PPO 是 on-policy 的：每一轮都用当前策略重新采样新回复、过奖励模型打分再更新，训练过程中"
         "用到的数据分布始终跟着策略的变化实时更新，天然能覆盖当前策略实际会走到的状态分布，但代价"
         "是采样、打分开销大，训练慢。DPO 是 offline 的：直接在一份提前收集好、策略更新前就固定下来"
         "的偏好数据集上训练，效率高、工程简单，但如果策略在训练中偏离得较远，之后遇到的输入/输出"
         "分布已经不在原始偏好数据集覆盖范围内(分布外)，DPO 没有'重新采样验证'的机制去发现和纠正"
         "这种偏移，这也是后续 online/iterative DPO 变体想要弥补的点。",
       keys=("on-policy", "offline", "分布外", "效率"),
       follow_ups=("iterative DPO 是怎么部分解决这个问题的？", "这个差异对应到监督学习里，类似 on-policy imitation learning 和 offline RL 的关系吗？")),
]


def _self_test() -> None:
    assert 9 <= len(BANK) <= 15, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-rlhf-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_rlhf: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
