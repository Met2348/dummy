"""推理模型与Test-Time-Compute深水区追问链（21个DeepPoint）。

覆盖：test-time compute的parallel scaling(多路采样+投票/verifier选优)与sequential
scaling(拉长单条思维链自我修正)的机制区别与可扩展性差异(arXiv 2502.12215对QwQ/R1的
系统评测)、process reward model(PRM)与outcome reward model(ORM)/多数投票在测试时
选优的具体数字差距(《Let's Verify Step by Step》PRM800K)、s1的budget forcing"Wait"
token技巧、underthinking现象(2501.18585)、DeepSeek-R1的"aha moment"及其涌现性争议、
o1/o3/o4-mini/GPT-5系列的推理调度与真实成本数量级(ARC-AGI $1000→$30000/题的估算修订、
GPT-5统一router架构、AIME/SWE-bench具体分数)、reasoning_effort档位的边际收益实证、
DeepConf基于内部置信度的免训练提前终止、以及跨厂商(Claude interleaved thinking/
Gemini thinking budget)的cost-latency-quality工程权衡。

边界：GRPO/PPO等RL训练算法细节、PRM在RLHF训练阶段被reward hacking的问题，已在
RLHF与对齐深水类目覆盖，这里只讨论PRM/置信度作为test-time选优机制的使用；deliberative
alignment只点到"这是test-time compute在安全场景的应用"这一层，具体越狱攻防细节留给
评测与安全深水类目；不涉及KV cache/PagedAttention等服务化显存细节(推理部署与服务化
深水类目的范围)。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, grade_chain  # noqa: E402

CAT = "推理模型与Test-Time-Compute深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-fr-rtc-01",
        cat=CAT,
        trigger="你刚才说'把思维链拉长之后QwQ在几道题上反而错了'，这是个例还是有系统性证据？",
        chain=(
            ("这是个例还是有系统性证据？",
             "2025年论文《Revisiting the Test-Time Scaling of o1-like Models》(arXiv 2502.12215，复旦NLP组/OpenMOSS)"
             "系统评测了QwQ、DeepSeek-R1及其蒸馏版在AIME、MATH、GPQA、Omni-MATH上的表现，发现同一道题里，模型给出的"
             "正确解答普遍比错误解答更短，说明'CoT越长越准'不是普遍规律。",
             ("2502.12215", "正确解答", "更短")),
            ("背后的具体机制是什么，为什么变长反而更容易错？",
             "论文进一步定位到，CoT变长几乎总是伴随更多自我修正标记(比如'Wait'、'Alternatively'这类反思触发词)，"
             "这些反思片段占比越高，最终答案出错的概率越大，尤其是在R1-Distill-1.5b这种小规模蒸馏模型上这个负相关"
             "最明显，说明是自我修正本身在拖累，而不是'思考更多'本身有害。",
             ("自我修正", "反思", "R1-Distill-1.5b")),
            ("那是不是所有'延长推理'的技术都无效，s1的budget forcing不是也靠硬塞'Wait'来延长吗，这不矛盾吗？",
             "不矛盾，关键区别在于'由谁、面向什么状态的模型追加反思'：s1的budget forcing是在一个只用1000条样本SFT过、"
             "还没被RL训练出大量自发反思模式的模型上，用外部注入的'Wait'把模型从过早收敛的答案上拉回来继续想；而"
             "2502.12215观测的是QwQ/R1这类已经被RL训练到会自发大量自我修正的模型，它们自己产生的反思本身就是噪声"
             "甚至负贡献的来源，二者面对的模型状态和反思来源完全不同。",
             ("budget forcing", "自发", "噪声")),
            ("这个'延长CoT有害'的结论你有多大把握，换一个任务类型还成立吗？",
             "论文实验集中在数学/科学推理这类有明确对错标准的任务(AIME/GPQA/Omni-MATH)，对代码生成、开放式写作等"
             "任务是否有同样的负相关没有覆盖，不能直接外推；诚实的说法是至少在竞赛数学这个域里，反思标记密度和"
             "正确率有稳定的负相关，但这不等于'思考多就一定错'的普适结论。",
             ("数学", "没有覆盖", "不能直接外推")),
        ),
        pitfall="很多人只记住'论文说CoT越长越差'这句结论标签，答不出是'自我修正标记密度'这个具体机制在起作用，第3层被问到s1也是硬塞Wait为什么不矛盾时容易语塞。",
        real_world_link="learning/reasoning-eval/lectures/11-reasoning-eval-pitfalls.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-02",
        cat=CAT,
        trigger="你说你们用parallel scaling做多路投票，那和直接把chain-of-thought拉长的sequential scaling比，效果上到底谁更划算？",
        chain=(
            ("先说清楚这两者具体分别是什么？",
             "parallel scaling是对同一个问题独立采样多条推理轨迹(比如self-consistency多数投票、Best-of-N配合"
             "verifier打分、beam/lookahead search)，靠广度覆盖提升pass@k；sequential scaling是让单条推理轨迹本身"
             "变长，靠自我修正/反思在一条链内部把答案改对，靠深度延伸提升单次准确率。",
             ("独立采样", "广度覆盖", "单条推理轨迹")),
            ("2502.12215这篇论文比较这两者，具体谁的可扩展性更好？",
             "论文在QwQ、R1、LIMO上系统对比后发现，parallel scaling(多路采样+投票/verifier选优)的覆盖率随采样数"
             "增加持续提升、可扩展性明显更好；而sequential scaling很快就遇到收益递减甚至转负的情况，说明o1-like"
             "模型自我反思续写的能力比多次独立尝试的能力弱得多。",
             ("覆盖率", "收益递减", "反思续写")),
            ("既然parallel scaling更好，是不是可以完全抛弃sequential scaling、只做多路采样就够了？",
             "不行，因为parallel scaling要真正收敛到好答案依赖一个可靠的选优机制(majority vote或verifier)，如果"
             "任务本身没有清晰的正确性信号(比如开放式代码重构、创意写作)，多路采样之间无法有效投票或打分，"
             "parallel scaling的覆盖率优势就发挥不出来，这时候只能依赖模型自己在单条链内做sequential的自我修正。",
             ("选优机制", "verifier", "开放式")),
            ("如果你的产品既要处理数学题又要处理开放式写作，这个parallel vs sequential的判断在多大程度上能直接套用？",
             "论文的结论建立在AIME/MATH/GPQA这类有标准答案、能用majority vote或规则验证器打分的任务上；一旦离开"
             "这个前提，parallel scaling的优势是否还成立没有被这篇论文验证过，工程上更现实的做法是按任务类型分流："
             "可验证任务上多花预算做parallel+verifier，开放式任务上退化成单条sequential。",
             ("有标准答案", "没有被这篇论文验证过", "分流")),
        ),
        pitfall="很多人能说出parallel/sequential两个词但混淆哪个对应'多路采样'哪个对应'拉长链条'；第3层容易忽略verifier缺失场景下parallel scaling其实用不上。",
        real_world_link="learning/multimodal-agent/lectures/09-test-time-scaling.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-03",
        cat=CAT,
        trigger="你刚才引用Snell那篇论文说sequential revision更好，可是你前面又说2502.12215那篇论文证明parallel scaling更好，这两个结论到底谁对？",
        chain=(
            ("先分别说清楚这两篇论文各自的实验对象和结论。",
             "Snell et al(arXiv 2408.03314，《Scaling LLM Test-Time Compute Optimally》)研究的是专门针对self-"
             "revision做过微调的模型，在compute-matched对比下发现顺序修正配合按难度自适应分配的compute-optimal"
             "策略比朴素best-of-N更省算力；Zeng et al(2502.12215)研究的是QwQ/R1这类已经被RL训练出大量自发长链"
             "反思习惯的o1-like模型，在AIME/GPQA这类竞赛任务上发现继续拉长同一条链收益递减，parallel覆盖率更好。",
             ("2408.03314", "compute-optimal", "已经被RL训练")),
            ("那到底是什么让这两个'sequential vs parallel哪个更好'的结论看起来相反？",
             "核心差异在于模型的起点状态不同：Snell测的模型没有被过度训练出大量自我反思的习惯，额外的sequential"
             "修正步骤能带来净增益；Zeng测的模型本身已经通过RL被推到习惯性大量自我反思的状态，同一条链继续延伸时"
             "新增反思步骤在边际上更容易引入自我否定式的退化，parallel换一条独立轨迹反而更容易跳出这个退化陷阱。",
             ("起点状态", "边际", "退化陷阱")),
            ("所以能不能说'sequential scaling只在没有被RL特化训练反思的模型上有效'？",
             "可以理解成一个更细粒度的假说：sequential self-revision的收益取决于当前这条链是否还处于有效信息增量"
             "为正的区间，模型没有被大量RL训练出反思冗余时更容易处于这个区间；一旦模型已经重度依赖反思，同一条链"
             "的边际反思质量下降，此时parallel的独立重采样更划算——这本质上是同一个compute-optimal分配框架下，"
             "因模型训练阶段不同而落在曲线不同位置的表现，而不是两个互斥的结论。",
             ("有效信息增量", "compute-optimal分配", "曲线不同位置")),
            ("你这套'调和'解释是你自己推理出来的，还是两篇论文里明确写的？",
             "这是诚实的说法——两篇论文没有直接互相引用对比过对方的实验设定，我这里给出的'模型起点状态不同导致"
             "曲线位置不同'是基于两篇论文各自方法论细节做的合理外推解释，不是论文原文的直接结论，更严谨的说法应该"
             "是现有公开文献里还没有一篇论文同时控制模型训练阶段和任务难度来直接解决这个表面矛盾，这是一个空白。",
             ("没有直接互相引用", "合理外推", "空白")),
        ),
        pitfall="大多数人根本没意识到这两篇论文结论'看起来矛盾'需要被调和，直接各自背诵，一旦被面试官同时抛出两个结论对比就露怯；即使意识到了，也很少有人能诚实说清哪部分是论文原文、哪部分是自己的推断。",
        real_world_link="learning/reasoning-eval/lectures/12-capstone-reasoning-bench.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-04",
        cat=CAT,
        trigger="你提到用PRM做验证器选答案比多数投票效果好，好多少？这个PRM是怎么训出来的？",
        chain=(
            ("PRM和ORM具体区别是什么，各自怎么监督？",
             "ORM(outcome-supervised reward model)只看最终答案对不对给一个整体分数，是稀疏监督；PRM(process-"
             "supervised reward model)对推理链里每一步单独打对/错标签，是稠密监督，OpenAI的《Let's Verify Step "
             "by Step》论文配套发布了PRM800K数据集，包含80万条step级别正确性标注。",
             ("outcome", "process", "PRM800K")),
            ("论文里PRM相比ORM和多数投票具体好多少？",
             "在MATH测试集代表性子集上，PRM达到78.2%准确率，ORM是72.4%，纯多数投票(majority voting)是69.6%，PRM"
             "明显领先；而且随着采样答案数增多，PRM相对ORM的优势还会进一步扩大，说明PRM在大规模候选里搜索的能力"
             "更强。",
             ("78.2%", "72.4%", "69.6%")),
            ("为什么候选数越多，PRM的优势越明显，这是什么机制？",
             "ORM和多数投票本质上只利用最终答案是否一致/正确这一个信号，候选数增多带来的信息增量有限；PRM能对"
             "每条候选链的中间步骤逐步打分，候选数越多，PRM越有机会从大量候选里精确定位出中间步骤全对的高质量解，"
             "而不是仅仅依赖最终答案的表面一致性，因此候选池扩大对PRM的边际收益比对ORM/多数投票更大。",
             ("中间步骤", "边际收益", "表面一致性")),
            ("PRM这么强，是不是可以完全替代多数投票和ORM，你在生产里会怎么选？",
             "不会完全替代，PRM需要额外训练一个步骤级验证器，训练成本远高于直接做多数投票，且PRM本身是学出来的"
             "模型，同样存在被过度优化(reward hacking)和在数学以外领域泛化能力未知的问题；工程上更现实的做法是"
             "简单任务先用多数投票兜底，只有当任务价值高时才引入PRM。",
             ("训练成本", "reward hacking", "价值高")),
        ),
        pitfall="很多人只记得'PRM比多数投票好'这句结论，说不出具体的78.2/72.4/69.6这组数字，也说不清'候选越多PRM优势越大'这个具体机制，容易把PRM和普通的reward model混为一谈。",
        real_world_link="learning/process-reward/lectures/01-orm-vs-prm.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-05",
        cat=CAT,
        trigger="你说除了best-of-N，你们还试了用PRM做beam search，为什么不直接用best-of-N配合PRM挑最高分的，两者本质差在哪？",
        chain=(
            ("best-of-N配合PRM打分和PRM引导的beam/lookahead search，流程上有什么不同？",
             "best-of-N是先把N条完整轨迹独立生成完，再用PRM给每条打分选最高分的一条，PRM只在生成完之后介入；"
             "beam/lookahead search是在生成过程中，每走一步就用PRM给当前部分轨迹的候选延续打分，提前剪掉低分分支，"
             "只保留高分分支继续生成，PRM在生成过程中就持续介入指导搜索方向。",
             ("生成完之后", "生成过程中", "剪掉低分分支")),
            ("Snell et al这篇论文里，这种搜索式的方法在什么情况下比best-of-N更省算力？",
             "论文提出的compute-optimal策略指出，用verifier引导的beam/lookahead search能提前放弃明显没希望的"
             "分支，把算力集中花在有潜力的分支上，在同等算力预算下能覆盖到更多有效候选，而best-of-N对所有N条轨迹"
             "平均分配算力、生成完了才知道哪条是浪费的。",
             ("提前放弃", "算力集中", "同等算力预算")),
            ("既然搜索式方法更省算力，为什么工业界很多系统还是直接用最简单的best-of-N或多数投票？",
             "beam/lookahead search需要PRM在每一步都能给出可靠的中间打分，如果PRM本身在中间步骤上的判断噪声较大，"
             "提前剪枝可能会误杀掉后期能自我修正过来的分支；而且这种step-by-step介入的搜索比独立生成N条+事后打分"
             "复杂得多，要和解码流程深度耦合，best-of-N则可以完全解耦、批量并行，鲁棒性更可预测。",
             ("误杀", "深度耦合", "鲁棒性更可预测")),
            ("这和RLHF训练里PRM被reward hacking是不是同一个问题？",
             "这是相关但不完全相同的两个问题——RLHF训练里PRM被reward hacking，是policy在训练阶段针对PRM的打分"
             "做梯度优化专门学会讨好PRM；test-time用PRM做搜索时，policy权重是冻结的，不存在针对PRM做梯度优化这"
             "一步，风险主要是PRM在没见过的中间状态分布上判断不准导致误剪枝，这是泛化误差问题而不是训练时的博弈"
             "问题，两者机制不同但都源于PRM作为学出来的打分器本身不完美这一共同根源。",
             ("policy权重是冻结的", "泛化误差", "共同根源")),
        ),
        pitfall="很多人分不清best-of-N和beam/lookahead search的本质区别(事后打分vs过程中打分)，第4层容易把test-time的PRM误剪枝和训练时的reward hacking混为一谈，说不清两者机制不同。",
        real_world_link="learning/process-reward/notebooks/N1-prm-best-of-n.ipynb",
    ),
    DeepPoint(
        id="dp-fr-rtc-06",
        cat=CAT,
        trigger="你说你们用了类似s1的budget forcing技巧控制推理长度，具体怎么做的，为什么非要用'Wait'这个词？",
        chain=(
            ("budget forcing具体是怎么操作的？",
             "s1论文(arXiv 2501.19393，Muennighoff等，斯坦福)提出budget forcing：如果模型生成token数超过预算，"
             "就强制截断并插入结束思考的标记提前收尾；如果生成token数还不够、模型却想提前结束思考，就抑制结束"
             "思考标记，转而在末尾追加'Wait'，逼模型继续往下推理。",
             ("强制截断", "抑制", "Wait")),
            ("为什么具体是'Wait'这个词，换成别的词效果一样吗？",
             "论文做了消融对比，追加'Wait'比追加中性词'Hmm'效果明显更好(AIME24上Wait达到53.3%，Hmm只有50.0%，"
             "和完全不做延长的50.0%基本持平)，说明具体触发词的语义(暗示重新检查而不是单纯拖延)对能否成功诱导"
             "有效的自我修正是有区别的，不是随便加个填充词都管用。",
             ("Hmm", "53.3%", "语义")),
            ("这套方法配合1000条SFT样本能做到什么效果，是不是意味着大规模RL训练可以被这种便宜技巧替代？",
             "s1只用1000条精选样本(s1K数据集)对Qwen2.5-32B-Instruct做SFT，配合budget forcing就在竞赛数学(MATH、"
             "AIME24)上超过o1-preview最多27个百分点，并且能把外推能力从不干预时的50%推到强制多想后的57%；但这"
             "不代表能完全替代大规模RL，s1证明的是一个已经具备较强基座推理能力的模型靠很小的监督信号加测试时"
             "干预就能撬动可观提升。",
             ("s1K", "27个百分点", "撬动")),
            ("一直追加'Wait'逼模型多想，有没有可能想出反效果，比如陷入死循环或者把原本对的答案改错？",
             "确实存在这个风险，如果不断追加'Wait'超出模型本身还能提供新信息的区间，模型可能陷入死循环、开始重复"
             "已经说过的内容，或者像2502.12215观测到的现象一样过度自我修正反而把原本正确的答案改错；s1论文设置了"
             "预算上限来控制这个风险，但没有系统研究追加多少次Wait开始转为负收益这个边界，这是budget forcing诚实"
             "的局限。",
             ("死循环", "改错", "边界")),
        ),
        pitfall="很多人知道s1用了budget forcing但答不出具体机制(强制截断+抑制结束标记)，更不知道'Wait'是消融对比出来的最优选择而不是随便选的词；第4层很少有人主动提出'一直追加Wait会不会反效果'这个隐患。",
        real_world_link="learning/multimodal-agent/src/s1_budget_forcing.py",
    ),
    DeepPoint(
        id="dp-fr-rtc-07",
        cat=CAT,
        trigger="你提到模型有时候不是想太多，是想得不够深就换方向了，这个现象有专门研究吗？",
        chain=(
            ("这个'想得不够深就换方向'的现象有名字吗，怎么定义的？",
             "这被称为underthinking，来自论文《Thoughts Are All Over the Place: On the Underthinking of o1-Like "
             "LLMs》(arXiv 2501.18585)，指o1-like模型在推理时频繁地在不同思路(thought)之间切换，还没有把一条有"
             "潜力的思路探索到底就提前放弃转向下一个思路，导致推理深度不够。",
             ("underthinking", "切换", "提前放弃")),
            ("这个现象是怎么被量化验证的，跟准确率有什么关系？",
             "论文在多个高难度测试集和两个开源o1-like模型上做实验，发现频繁的思路切换和答错高度相关；他们提出了"
             "一个专门的underthinking指标，衡量错误回答里有多少token比例其实是花在了原本能通向正确思路的片段上"
             "却没有走完，把这个指标和准确率结合起来能更全面评估模型。",
             ("思路切换", "错误回答", "token比例")),
            ("这和之前提到的overthinking(想太多反而变差)是不是矛盾的，到底该让模型想多还是想少？",
             "不矛盾，两者是同一套长链推理里的思路管理机制在不同方向上的失败模式——overthinking是在同一个思路上"
             "过度反复修正、纠缠不放；underthinking是在思路之间切换太频繁、每个思路都没探索到底；论文针对"
             "underthinking提出了TIP(thought switching penalty)解码策略，通过惩罚过早的思路切换来缓解，不需要"
             "重新训练模型就能提升准确率。",
             ("失败模式", "TIP", "不需要重新训练")),
            ("TIP这种解码时加惩罚的方法，会不会矫枉过正，把本该切换的思路强行摁住不让换？",
             "确实存在矫枉过正的风险，惩罚力度需要人工调节，如果惩罚过强，模型该放弃一个明显错误的思路时也可能"
             "因为惩罚而被迫继续深挖下去，浪费更多token；论文里这属于一个超参数选择问题，目前没有一个自适应的、"
             "该切换时就切换该深挖时就深挖的判据，仍然是启发式设定。",
             ("矫枉过正", "超参数", "启发式")),
        ),
        pitfall="很多人只知道'overthinking想太多'，没听过underthinking这个对称的失败模式；被问到'两者矛盾吗'时容易答混，答不出TIP解码策略这个具体缓解手段。",
        real_world_link="learning/reasoning-eval/lectures/11-reasoning-eval-pitfalls.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-08",
        cat=CAT,
        trigger="你说DeepSeek-R1的aha moment特别有名，具体是指什么，给我说说R1-Zero训练的具体数字。",
        chain=(
            ("aha moment具体指的是训练过程中的什么现象？",
             "DeepSeek-R1论文(arXiv 2501.12948)里，研究者用GRPO对DeepSeek-V3-Base直接做大规模强化学习(即R1-"
             "Zero，不经过SFT)，观察到中间某个checkpoint的模型开始用拟人化的语气重新审视自己之前的解题步骤，"
             "论文作者称这是他们训练过程中的aha moment，认为这体现了RL让模型涌现出自我反思、验证、动态调整"
             "策略的能力。",
             ("GRPO", "R1-Zero", "自我反思")),
            ("这套RL训练具体带来了多大的数字提升？",
             "在AIME 2024上，DeepSeek-R1-Zero的pass@1从15.6%提升到71.0%，如果再配合多数投票(majority voting)"
             "进一步提升到86.7%，达到和OpenAI o1-0912相当的水平，这组数字被广泛引用来说明纯RL(不依赖人工标注CoT)"
             "也能激发出强推理能力。",
             ("15.6%", "71.0%", "86.7%")),
            ("这个aha moment是不是证明了'反思能力是RL训练涌现出来的新能力'？",
             "这一点存在争议，Sea AI Lab等后续研究复现时发现，如果直接在DeepSeek-V3-Base这个base model上不做"
             "任何RL、只用R1的模板测试，就已经能观察到'aha'、'wait'这类反思关键词，说明反思行为可能本来就存在，"
             "RL训练更多是把这类行为放大、强化，而不是从无到有地涌现出全新能力，这类反思频率提升也不必然意味着"
             "准确率同步提升，可能只是Superficial Self-Reflection。",
             ("base model", "放大", "Superficial Self-Reflection")),
            ("那你觉得aha moment这个说法算不算被过度营销了？",
             "诚实地说，这是技术叙事和严格科学结论之间的落差——论文原文对aha moment的表述本身是描述性、带点故事"
             "色彩的，而不是经过严格消融的科学论断；后续研究提出的质疑目前也没有完全定论，比较负责任的说法是"
             "反思行为很可能不是纯粹从零涌现，但RL训练确实系统性地放大和稳定了这种行为，这中间的因果链条还需要"
             "更细致的研究。",
             ("技术叙事", "落差", "因果链条")),
        ),
        pitfall="很多人只会复述'DeepSeek-R1有个很神奇的aha moment'当故事讲，说不出15.6%/71.0%/86.7%这组具体数字，更不知道后续有研究质疑这个'涌现'说法本身，容易被面试官的'过度营销'追问问住。",
        real_world_link="learning/reasoning-r1/lectures/03-r1-zero.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-09",
        cat=CAT,
        trigger="你说R1训练过程中response长度一直在涨，这就是test-time scaling吧？",
        chain=(
            ("先说清楚，R1论文里观测到的'response长度在RL训练过程中持续增长'具体是什么现象？",
             "DeepSeek-R1论文报告，随着GRPO训练步数增加，模型在训练集上生成的平均response长度持续变长，这是一条"
             "训练过程中的动态曲线(x轴是训练步数)，论文把这个曲线的增长和模型逐渐学会更复杂推理行为关联在一起"
             "呈现。",
             ("训练步数", "response长度", "GRPO训练")),
            ("这和test-time scaling是同一件事吗？",
             "不是同一件事——test-time scaling指的是在模型训练完成、权重冻结之后，在推理/部署阶段主动多花计算"
             "来换取更高准确率，是一个部署时刻的、可由使用者主动控制的旋钮；而R1论文里那条曲线是训练时刻的现象，"
             "说的是随着RL训练本身进行，模型学会的默认行为倾向变成了输出更长的链，这是训练动力学，不是推理阶段"
             "的计算分配决策，两者时间维度和控制主体都不同。",
             ("训练动力学", "部署时刻", "控制主体")),
            ("那这两者有没有关系，是不是训练时变长就必然带来更好的test-time scaling能力？",
             "有关系但不等价：R1训练时学会默认输出更长的链，客观上是模型具备了sequential test-time scaling的"
             "行为基础(它现在会自发地想更久)；但根据2502.12215的发现，这种被RL塑造出来的默认行为，在真正部署"
             "阶段继续人为拉长时反而容易收益递减，说明训练时学会想更久和推理时想更久就更准并不是一回事，需要"
             "单独验证。",
             ("行为基础", "收益递减", "单独验证")),
            ("业界很多分享/面经把'R1训练时输出变长'直接等同于'test-time scaling生效'，这个说法你觉得有多大问题？",
             "这个说法混淆了两个不同层次的现象，属于常见的表述简化，问题在于它容易让人误以为只要训练让模型输出"
             "变长部署时效果就自动变好，但实际上training-time行为改变和deployment-time效用提升中间还隔着这个"
             "更长的输出是否真的承载了更多正确信息这一步，比较严谨的说法应该把这两件事分开讨论。",
             ("混淆", "常见的表述简化", "分开讨论")),
        ),
        pitfall="几乎所有人都会把'R1训练时response变长'和'test-time scaling'当成同一件事直接划等号，答不出'训练动力学vs部署时刻计算分配'这个时间维度和控制主体上的根本区别。",
        real_world_link="learning/reasoning-r1/lectures/12-spurious-rewards.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-10",
        cat=CAT,
        trigger="你说o3在ARC-AGI上刷到快90%，但代价很大，'代价很大'具体是多大，能给出数量级吗？",
        chain=(
            ("o3在ARC-AGI上的具体表现和对应的成本级别是什么？",
             "OpenAI在2024年12月公布o3时，高算力(o3 high-compute)配置在ARC-AGI半私有测试集上拿到87.5%，相比o1"
             "模型的32%是巨大跨越；ARC Prize团队最初估算这个高分配置每道题消耗超过1000美元的计算成本，而低算力"
             "版本只用大约20美元/题就能拿到76%。",
             ("87.5%", "1000美元", "76%")),
            ("这个1000美元的估算后来有变化吗？",
             "有，ARC Prize基金会在2025年4月发布的修订估算里，把高算力o3配置的成本从最初约1000-3000美元大幅上调"
             "到接近30000美元/题，这个数量级的上修说明外部对o3真实推理成本的最初估计严重偏低。",
             ("2025年4月", "30000美元", "上调")),
            ("Chollet自己怎么评价这个成本，这对'test-time compute能替代人类专家'这个叙事有什么冲击？",
             "Chollet指出用人力解ARC-AGI题目大约每题5美元、耗电成本几乎可以忽略，而o3高算力配置的成本比人力贵"
             "几个数量级换来的准确率提升，说明当前阶段test-time compute带来的性能提升是用极不成比例的经济代价"
             "换来的，只有成本随硬件/算法进步大幅下降后，test-time compute的经济性才可能反超。",
             ("5美元", "不成比例", "经济性")),
            ("这组成本数字你有多大把握是准的，会不会已经过时了？",
             "这些数字本身就经历过一次从千美元级到万美元级的大幅修订，说明外部对闭源模型test-time推理成本的估算"
             "本身高度不确定，依赖的是第三方基于API定价和token消耗的反推，不是OpenAI官方逐题公布的真实成本；诚实"
             "的态度是这组数字只能说明量级在千到万美元这个区间，具体数值会持续变化，不应该当作固定不变的事实。",
             ("反推", "官方", "量级")),
        ),
        pitfall="很多人只记得'o3很贵'这种模糊说法，答不出$1000到$30000这个具体数量级变化，更不会主动提到这组数字本身经历过大幅上修、可信度需要打折扣。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-fr-rtc-11",
        cat=CAT,
        trigger="你说o4-mini用更便宜的方式做到了差不多的效果，背后的原理是什么，是模型变聪明了还是别的什么在起作用？",
        chain=(
            ("o4-mini相比o3系列，在成本-能力权衡上的定位是什么？",
             "o4-mini是OpenAI在o3之后推出的更小、更便宜的推理模型，采用OpenAI所称的simulated reasoning(模拟推理)"
             "方式，用大约o3量级十分之一左右的成本，在多项推理基准上达到有竞争力的结果，是test-time compute花钱"
             "买效果这条曲线上性价比更高的一个点。",
             ("o4-mini", "simulated reasoning", "十分之一")),
            ("为什么更小的模型配合test-time compute能追上更大模型的效果，这背后的一般性原理是什么？",
             "这体现的是训练时算力(模型参数规模)和测试时算力(推理阶段思考时长/采样数)之间存在一定的可替代性"
             "(compute-equivalence)：一个更小但经过强化学习训练擅长利用推理时间的模型，通过测试时多花计算，可以"
             "在某些任务上逼近参数规模大得多的模型，本质上是把花在训练阶段的算力部分转移到了推理阶段。",
             ("compute-equivalence", "可替代性", "转移")),
            ("这种'小模型+更多测试时算力'替代'大模型'的策略，是不是在所有任务上都成立？",
             "不是，这种替代性更多体现在需要多步逻辑推导、有清晰验证信号的任务上(数学、代码、逻辑谜题)，因为这类"
             "任务的正确性可以在测试时通过反复采样/验证搜索出来；但对依赖大模型内隐知识广度的任务(冷门事实检索)，"
             "小模型的知识容量是硬约束，无论测试时怎么多想都补不回来。",
             ("知识容量", "硬约束", "补不回来")),
            ("'约1/10成本'这个说法你验证过具体来源吗，有多确定？",
             "这个约1/10成本、有竞争力的结果更接近业界和媒体报道对o4-mini定位的概括性描述，而不是OpenAI公布的"
             "逐项可复现的成本核算，具体的成本倍数会随定价策略、任务类型剧烈波动；诚实的说法是方向上确实存在小"
             "模型+更多test-time compute逼近大模型效果这个现象，但恰好10倍这个具体数字需要看到官方逐任务的成本"
             "数据才能给出更严谨的结论。",
             ("概括性描述", "波动", "官方逐任务")),
        ),
        pitfall="很多人只会说'o4-mini更便宜效果差不多'，说不出背后train-time compute和test-time compute可替代这个一般性原理，也说不出这种替代性在知识密集型任务上会失效；第4层几乎没人会主动质疑'1/10'这个数字的确定性。",
        real_world_link="",
    ),
    DeepPoint(
        id="dp-fr-rtc-12",
        cat=CAT,
        trigger="你说GPT-5用了一个router来决定要不要深度推理，这个router具体是怎么工作的？",
        chain=(
            ("GPT-5的router具体解决的是什么问题？",
             "在GPT-5之前，用户需要在ChatGPT里手动挑选用GPT-4o(快、但推理浅)还是o3/o4-mini(慢、但会深度推理)，"
             "选择本身就是体验负担；GPT-5把这些能力统一成一个系统，内部包含一个较快的基础模型和一个更深的推理"
             "模型(GPT-5 Thinking)，由一个实时router根据query的复杂度、是否需要工具调用、上下文情况自动判断该"
             "调用哪一个。",
             ("手动挑选", "GPT-5 Thinking", "实时router")),
            ("这次统一具体替换掉了哪些原来的模型，说明了什么？",
             "GPT-5发布后在ChatGPT里替换掉了GPT-4o、o3、o4-mini、GPT-4.1、GPT-4.5这几个此前并存的模型，把要不要"
             "深度推理这个决策权从用户手动选择转移到了系统自动路由，说明OpenAI认为test-time compute的分配应该是"
             "一个对用户透明的系统级调度问题。",
             ("GPT-4o", "o3", "系统级调度")),
            ("这个自动路由本质上和test-time compute有什么关系？把router做进产品里，是不是把研究结论工程化了？",
             "是的，本质上router就是把什么时候该多花test-time compute这个决策自动化了——如果每次请求都无脑上"
             "最贵的深度推理模型，成本和延迟都不可接受；router通过预测query复杂度来做这个花多少test-time compute"
             "的实时决策，是test-time compute研究成果从固定策略走向生产系统里动态调度的工程化体现。",
             ("成本和延迟", "动态调度", "工程化体现")),
            ("router自己判断错了怎么办，比如一个看起来简单实际很难的问题被分到了快速模型？",
             "这是这套系统的诚实局限——router本身是一个分类决策，存在误判风险，看起来简单实际很难的问题如果被"
             "误判成用快速路径处理，用户体验到的就是一个本可以答对却答错的结果，OpenAI没有公开router误判率的"
             "具体数字，这类系统级路由决策的可靠性目前更多依赖大规模在线数据迭代。",
             ("误判", "没有公开", "在线数据")),
        ),
        pitfall="很多人知道GPT-5能自动判断该不该深度思考，但说不清router具体替换掉了哪些历史模型，也想不到这本质上是把test-time compute分配问题产品化/系统化了；几乎没人会主动追问router误判的风险和不透明性。",
        real_world_link="learning/serving-graduation/src/multi_model_router.py",
    ),
    DeepPoint(
        id="dp-fr-rtc-13",
        cat=CAT,
        trigger="你说GPT-5在数学和代码上的分数比o3高不少，具体高多少，是靠更强的模型还是靠更多test-time compute堆出来的？",
        chain=(
            ("能具体说说GPT-5相比o3在几个关键benchmark上的数字吗？",
             "GPT-5在AIME 2025(不用工具)上达到94.6%，在SWE-bench Verified上是74.9%，相比o3的69.1%有明显提升；"
             "同时OpenAI报告，在生产流量的匿名prompt上开启网页搜索时，GPT-5的回答包含事实错误的概率比之前的模型"
             "低了大约45%。",
             ("94.6%", "74.9%", "69.1%")),
            ("GPT-5在SWE-bench上比o3高的同时，消耗的资源反而更少，这是怎么做到的？",
             "OpenAI报告GPT-5在SWE-bench Verified上达到74.9%的同时，用了比o3少22%的输出token和少45%的工具调用"
             "次数，说明这不是单纯靠更多test-time compute堆算力实现的分数提升，而是路由/训练让模型在需要深度"
             "推理时思考得更有效率，用更少的中间步骤达到更高正确率。",
             ("22%", "45%", "更有效率")),
            ("'幻觉降低45%'和另一个'GPQA上幻觉少6倍'是同一个数字吗？",
             "不是同一个数字，这是两个不同评测口径下的结果：约45%更少事实错误是在开启网页搜索、模拟真实生产流量"
             "匿名prompt上测出来的整体幻觉率下降；6倍更少幻觉则特指GPT-5 Thinking这个深度推理变体在GPQA Diamond"
             "这个学术科学问答基准上相对此前模型的降幅，两者的测试条件都不同，不能简单地把两个数字混为一谈。",
             ("网页搜索", "GPQA Diamond", "测试条件")),
            ("这些benchmark数字能直接说明GPT-5在你实际业务场景里也会有同等提升吗？",
             "不能直接下这个结论，AIME/SWE-bench/GPQA都是公开的、有一定刷分历史的标准化评测集，模型厂商在训练"
             "数据构建和后训练阶段有动机针对性优化这些基准表现；实际业务场景的任务分布、prompt风格和这些学术"
             "基准有差异，具体收益需要在自己的业务评测集上单独验证。",
             ("刷分", "有差异", "单独验证")),
        ),
        pitfall="很多人只会笼统说'GPT-5比o3强'，说不出94.6%/74.9%/69.1%这组具体数字，更容易把'45%更少事实错误'和'6倍更少幻觉'这两个不同测试条件下的数字混着引用。",
        real_world_link="learning/reasoning-eval/lectures/01-reasoning-bench-overview.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-14",
        cat=CAT,
        trigger="你说把reasoning_effort调到high，结果和medium几乎没差别，这是个例还是有系统性发现？",
        chain=(
            ("reasoning_effort这个参数具体控制的是什么？",
             "GPT-5 API暴露了一个reasoning_effort参数，提供minimal/low/medium/high四档，用来显式控制模型愿意花"
             "多少思考token再给出最终答案，本质上是把test-time compute的预算旋钮直接暴露给开发者，让应用按需在"
             "延迟/成本和准确率之间做取舍。",
             ("reasoning_effort", "四档", "延迟/成本")),
            ("有没有具体研究证明'调到high不一定比medium好'这个现象？",
             "2025年一篇关于LLM在基础数学推理上是否过度思考的评测论文发现，GPT-5在medium和high两档reasoning "
             "effort下都达到97%左右的准确率，几乎没有差别，说明继续加大effort在这类任务上边际收益为零；同一篇"
             "论文还记录了更极端的情况，某些题目上模型反复验证同一个结果多达3次、导致token消耗放大31倍的病态"
             "过度思考。",
             ("97%", "边际收益为零", "31倍")),
            ("如果边际收益经常是零甚至更差，为什么不干脆默认都用minimal/low档？",
             "不能一刀切，因为medium和high没差别是在特定任务(相对容易饱和的基础数学题)上观测到的，另一项针对o1"
             "的研究发现，把reasoning effort从high调到low时，模型的过度思考(overthinking)得分反而升高了35%，"
             "说明低预算下模型可能因为被迫仓促收敛而效率反而更差，effort档位和任务难度之间的关系不是单调的，需要"
             "按任务类型分别测。",
             ("35%", "仓促收敛", "按任务类型分别测")),
            ("那你在生产里会怎么定这个reasoning_effort的档位策略，有没有一个通用公式？",
             "没有通用公式，诚实的做法是把reasoning_effort当成一个需要按任务类型离线评测后再定的超参数：先在"
             "自己的任务分布上跑minimal到high几档，画出准确率-token成本曲线，找到边际收益开始趋平的那个档位作为"
             "默认值，并且要持续监控，因为随着底层模型版本升级这条曲线本身也会变化，需要纳入模型升级的回归测试"
             "流程。",
             ("准确率-token成本曲线", "趋平", "回归测试")),
        ),
        pitfall="很多人以为reasoning effort越高越好，说不出'medium和high在特定任务上没有边际收益'这个具体反直觉发现，更答不出'调低反而overthinking score更高'这个更细腻的非单调关系。",
        real_world_link="learning/serving-graduation/src/thinking_budget.py",
    ),
    DeepPoint(
        id="dp-fr-rtc-15",
        cat=CAT,
        trigger="你说用置信度提前终止推理链省了不少token，这和用verifier打分选答案有什么不同？",
        chain=(
            ("这种基于置信度提前终止的方法具体是怎么工作的？",
             "这类似Meta AI和UCSD在2025年提出的DeepConf(《Deep Think with Confidence》)方法，它利用模型内部"
             "生成时的token级置信度信号(不需要额外训练一个verifier)，在offline模式下用置信度对多条候选推理轨迹"
             "做加权投票或过滤低置信度轨迹；在online模式下，一旦某条轨迹在滑动窗口内的置信度掉到动态阈值以下，"
             "就直接提前停止这条轨迹的生成。",
             ("DeepConf", "不需要额外训练", "提前停止")),
            ("这个方法具体能带来多大的效率提升，有没有具体数字？",
             "论文报告DeepConf@512在AIME 2025上配合开源模型GPT-OSS-120B能达到99.9%的准确率，同时相比全量"
             "parallel thinking减少最多约84.7%的生成token；在DeepSeek-8B上AIME24这个题集里，标准多数投票准确率"
             "是86.7%，用DeepConf过滤后能提升到93.3%，说明低置信度轨迹里确实混杂了更多噪声。",
             ("84.7%", "86.7%", "93.3%")),
            ("这和PRM引导搜索有什么本质区别，为什么DeepConf能做到'不需要训练'？",
             "PRM是额外训练出来的一个独立打分模型，需要专门的step级标注数据；DeepConf利用的是生成模型自己在"
             "解码时天然产生的token级logprob/置信度信号，通过滑动窗口统计这些内部信号判断当前轨迹是否走不下去了，"
             "不引入任何外部模型，因此可以零训练成本地集成进现有的推理服务框架，比如在vLLM里只需要扩展logprob"
             "处理器、加一个提前停止检查。",
             ("logprob", "零训练成本", "vLLM")),
            ("完全依赖模型自己的置信度信号，会不会有模型'自信地错'这种情况，这个方法有没有失效的时候？",
             "会，这正是论文自己指出的局限——如果模型在某条错误的推理路径上恰好表现出很高的内部置信度(自信地错)，"
             "DeepConf的置信度过滤机制就没法识别出这条轨迹该被淘汰，论文把这个高置信度但错误的校准问题列为未来"
             "工作方向，说明基于内部信号的方法目前还不能完全替代有外部标注支撑的verifier在可靠性上的保证。",
             ("自信地错", "校准问题", "未来工作方向")),
        ),
        pitfall="很多人只知道'置信度提前终止能省token'这个笼统说法，说不出online/offline两种模式的区别，也说不出具体的84.7%/93.3%这类数字；第4层很少有人主动指出'模型自信地错'这个校准失效风险。",
        real_world_link="learning/serving-graduation/lectures/03-reasoning-cache.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-16",
        cat=CAT,
        trigger="你说o1系列在安全对齐上用了deliberative alignment，这和普通的RLHF安全训练有什么本质区别？",
        chain=(
            ("deliberative alignment具体是怎么训练的？",
             "OpenAI的Deliberative Alignment(arXiv 2412.16339)让模型直接学习人类写的安全规范文本本身，训练模型"
             "在回答之前显式回忆并推理这些规范条款，分两阶段：先用SFT教模型显式引用规范做推理，再用RL基于是否"
             "正确遵循规范来强化，应用在o1-preview、o1、o3-mini上。",
             ("2412.16339", "SFT", "o1-preview")),
            ("这套方法效果如何，为什么不直接把安全规范全文放进prompt里让模型现场对照？",
             "论文报告这套方法让o1在拒绝恶意越狱请求(StrongREJECT)和不过度拒绝良性请求(XSTest)这两个此前互相"
             "制约的指标上同时取得帕累托改进；直接把规范全文放进context现场对照虽然可行，但会带来明显的推理"
             "延迟成本，把规范内化训练进模型权重里可以在不增加额外延迟的情况下达到类似效果。",
             ("帕累托改进", "延迟成本", "内化")),
            ("这和test-time compute是什么关系？",
             "这是test-time compute思路在安全场景的一个具体应用——核心逻辑是给模型回答前先想一想的推理时间，让"
             "它在推理链里显式对照安全规范再作答，而不是要求模型对borderline请求做出瞬时反应；这说明test-time"
             "compute带来的能力提升不止体现在数学/代码这类可验证任务上，也可以被迁移到提升安全决策的审慎程度上。",
             ("回答前先想一想", "审慎程度", "迁移")),
            ("这套方法能防住所有新型越狱吗？",
             "不能，论文本身也承认这套方法能提升对已知越狱模式和分布外场景的泛化能力，但这是一个持续的攻防过程，"
             "新型的编码式/多轮式越狱手法仍可能绕过训练时见过的规范推理模式，具体绕过案例、红队ASR数字更适合"
             "放在评测与安全深水这个类目里深入讨论，这里只需要理解它是test-time compute在安全场景的一种应用。",
             ("持续的攻防", "泛化能力", "评测与安全深水")),
        ),
        pitfall="很多人知道o1'安全性更好'，但说不清deliberative alignment具体两阶段训练流程，也想不到要把它和'给模型推理时间'这个test-time compute核心逻辑联系起来。",
        real_world_link="learning/cot-faithfulness-oversight/lectures/L1-reasoning-model-interp.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-17",
        cat=CAT,
        trigger="你提到Claude的extended thinking有个budget_tokens参数，这个和它的interleaved thinking是什么关系？",
        chain=(
            ("budget_tokens具体控制什么？",
             "Claude的extended thinking用budget_tokens参数设定模型在给出最终回答前，最多能用多少token做内部"
             "推理，在Claude 4系列模型上这个上限针对的是完整的思考token(而不是展示给用户的摘要)，budget_tokens"
             "通常要求小于max_tokens总输出上限。",
             ("budget_tokens", "内部推理", "max_tokens")),
            ("interleaved thinking和普通的extended thinking有什么不同？",
             "interleaved thinking是Claude 4系列(Sonnet 4/4.5、Haiku 4.5、Opus 4/4.1/4.5)支持的特性，允许模型"
             "在多次工具调用之间穿插思考，先调用工具拿到结果后再针对这个结果做一轮新的推理，再决定下一步调用什么"
             "工具，把思考和行动交替串联起来，而不是像普通extended thinking那样只在最开始一次性想完再输出。",
             ("interleaved thinking", "工具调用之间", "交替串联")),
            ("既然是多轮思考累加，token预算怎么控制，是每轮单独设上限吗？",
             "不是每轮单独设上限，interleaved thinking下budget_tokens代表的是一次assistant轮次里所有思考块加总"
             "的预算，且这个预算可以超过max_tokens的单次限制，实际能用到的上限接近整个上下文窗口(200K token量级)，"
             "这是为了支持多步工具调用+推理这种更复杂的agentic工作流。",
             ("加总的预算", "超过max_tokens", "agentic工作流")),
            ("这种'预算可以很大'的设计，在实际账单和延迟上是什么代价？",
             "思考token本身是要计费的，生产实践里一次深度思考调用可能消耗普通completion 3到10倍的token量，这"
             "意味着虽然interleaved thinking在能力上支持很大的预算，但要不要真的用到这么大的预算需要按答错的"
             "代价是否明显高于多花几倍token的成本这个经验法则来决定，这是一个需要业务方自己权衡的成本决策。",
             ("3到10倍", "经验法则", "成本决策")),
        ),
        pitfall="很多人只知道'Claude有个思考预算参数'，说不清interleaved thinking和普通extended thinking的具体区别(是否允许工具调用间穿插)，更不知道interleaved模式下预算可以突破max_tokens限制这个细节。",
        real_world_link="learning/serving-graduation/lectures/02-thinking-budget.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-18",
        cat=CAT,
        trigger="你说对比过几家的推理预算设计，思路不太一样，具体说说Gemini和GPT-5在'要不要深度推理'这件事上的产品哲学有什么不同？",
        chain=(
            ("先说说两家具体的设计差异？",
             "Gemini 2.5系列把thinking budget作为一个显式的token数量参数暴露给开发者自己设定，是用户主动拨盘的"
             "设计思路，另外还有面向高难度数学/代码的Deep Think增强推理模式；GPT-5则是把这个决策收进一个自动"
             "router内部，由系统根据query复杂度自动判断该不该启用更深的推理模型，是系统代为决策的设计思路。",
             ("thinking budget", "Deep Think", "自动router")),
            ("有没有实验数据支持'预算加大后收益递减'这个现象，用户会不会因为不知道该设多少预算而吃亏？",
             "有研究针对Gemini-2.5-Flash做过消融，把thinking token预算设为1024/2048/4096/8192等几档，发现预算"
             "增加确实能提升准确率，但增长到一定程度后开始变平，1000/2000档的平均表现已经很接近5000/10000档，"
             "说明显式拨盘设计下用户如果不清楚这条收益曲线，很容易把预算设得过高浪费成本却拿不到相应收益。",
             ("1024", "变平", "浪费成本")),
            ("那是不是说明GPT-5的自动路由设计天然更优，因为它替用户避免了'不知道该设多少'的问题？",
             "不能这么简单下结论，自动路由虽然免去了用户手动调参的负担，但代价是决策过程对用户不透明，用户没法"
             "针对自己特定的任务类型微调这个决策边界；两种设计本质上是在用户可控性和使用简便性之间做了不同取舍，"
             "不存在绝对意义上更优的一方，取决于产品面向的用户群体。",
             ("不透明", "用户可控性", "取决于产品面向")),
            ("如果非要你判断，未来哪种设计会成为主流？",
             "诚实的说法是目前没有定论，2026年的行业趋势里两种范式都还在并存甚至融合(比如Claude最新引入的"
             "adaptive thinking，让模型自己决定要不要深度思考，介于两者之间)，更可能的走向是显式预算参数和自动"
             "路由会长期共存，任何断言某一种范式已经赢了目前都缺乏足够证据支撑。",
             ("并存甚至融合", "adaptive thinking", "缺乏足够证据")),
        ),
        pitfall="很多人会简单地说'自动路由更智能、更先进'，答不出两种设计在可控性和不透明性上的具体取舍，更容易在最后一层给出'某种设计已经胜出'这种缺乏依据的武断结论。",
        real_world_link="learning/agent-foundations/lectures/07-router-orchestration.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-19",
        cat=CAT,
        trigger="你说test-time compute在数学代码上很好用，但你们业务是客服对话，感觉不太适用，为什么不适用，具体卡在哪一步？",
        chain=(
            ("test-time compute方法(多路采样投票、PRM搜索、置信度过滤)起作用的共同前提是什么？",
             "无论是majority vote、best-of-N配合verifier还是DeepConf这类置信度过滤，本质上都需要一个能区分哪条"
             "候选更好的选优机制——majority vote依赖正确答案会重复出现的假设，verifier/PRM依赖有明确对错标准的"
             "训练信号，这些前提在数学、代码这类有可验证答案的任务上天然成立。",
             ("选优机制", "重复出现", "可验证答案")),
            ("客服对话这类开放式任务，这个前提为什么不成立？",
             "客服回复没有唯一正确答案，多条候选回复可能都合理但风格/详略不同，majority vote在这种情况下无法"
             "通过找重复来选出更好的答案，也没有现成的verifier能像判断数学题对错一样判断哪条客服回复更好，除非"
             "额外训练一个针对客服场景的偏好打分模型，但那已经不是test-time compute能独立解决的问题了。",
             ("没有唯一", "无法通过找重复", "偏好打分模型")),
            ("如果一定要在客服场景引入某种test-time compute，有没有退而求其次的做法？",
             "可以退化成更弱的形式，比如让模型生成多条候选、再用一个训练过的偏好reward model挑一条得分最高的，"
             "这本质上是把best-of-N的选优环节从规则验证换成学出来的偏好判断；或者只做sequential层面的改进，比如"
             "让模型回复前先做一步检查是否遗漏用户诉求的自我审查，这类做法不依赖强验证信号，但收益也天然比数学"
             "代码场景弱很多。",
             ("偏好reward model", "自我审查", "收益也天然比")),
            ("这些退而求其次的做法，有没有公开研究验证过在开放式任务上的实际收益有多大？",
             "这方面的公开研究远不如数学/代码领域充分，目前主流的test-time scaling论文几乎都在数学、代码、科学"
             "问答这类可验证任务上做实验，开放式生成任务上test-time compute能带来多大提升缺乏系统性的公开基准"
             "数据，这是诚实的知识空白，不能直接套用数学场景验证过的收益幅度到开放式任务上。",
             ("远不如数学", "知识空白", "不能直接套用")),
        ),
        pitfall="很多人只会笼统说'开放式任务不好用test-time compute'，说不出'缺少选优机制'这个具体卡点，第3层想不出'换成学出来的偏好打分器'这个退而求其次的方案。",
        real_world_link="learning/reasoning-eval/lectures/05-math-shepherd-verifier.md",
    ),
    DeepPoint(
        id="dp-fr-rtc-20",
        cat=CAT,
        trigger="你说上线时要决定每个请求给多少推理预算，这个策略是你自己设计的，说说具体怎么设计的？",
        chain=(
            ("你会用哪些信号来决定一个请求该分配多少test-time compute？",
             "常见信号包括任务类型的历史难度分布(比如数学/代码类默认给更高预算)、请求本身的复杂度特征(prompt"
             "长度、是否涉及多步骤指令或工具调用)、以及产品侧对该请求的SLA要求(付费用户/高价值场景可以容忍更高"
             "延迟换取准确率)；这基本对应GPT-5 router做的事，只是很多团队在router能力还不够强之前需要自己做一层"
             "轻量分类器。",
             ("历史难度分布", "SLA要求", "轻量分类器")),
            ("如果分类器判断错了，把一个难题分到了低预算档，你怎么在系统里兜底？",
             "常见兜底策略是分层升级(escalation)：先用低预算档跑一次，如果输出触发了某些不确定信号(比如置信度"
             "低、多次采样结果不一致、下游校验失败)，再自动升级到更高预算档重新跑一次，这样大多数简单请求只花"
             "一次低成本调用，少数难题才会额外付出二次调用的成本，是用偶尔多算一次的代价换取整体成本优化。",
             ("分层升级", "不确定信号", "整体成本优化")),
            ("这套分层升级策略在延迟上有什么代价，比如用户体验会不会因为'二次调用'变差？",
             "会有代价，二次调用意味着触发升级的那部分请求的端到端延迟是两次调用时间之和而不是并行，如果对延迟"
             "极度敏感的场景，这个先失败后重试的串行升级模式可能不可接受，这时候需要改成并行双轨策略——同时以"
             "低预算和高预算各跑一次，用低预算结果先返回，具体选哪种要看产品对延迟和算力成本谁更敏感。",
             ("串行升级", "并行双轨", "谁更敏感")),
            ("你这套分层/双轨的框架，业界有没有公认的最佳实践，还是每家都是各自摸索？",
             "目前没有公认的统一最佳实践，GPT-5的router、Claude的adaptive thinking、各家自建的分类器本质上都在"
             "解决同一个实时预测该给多少test-time compute的问题，但具体信号、升级策略、延迟-成本权衡点都不一样，"
             "也缺乏标准化评测来比较这些路由策略的优劣，诚实的态度是承认这套策略需要针对自己的业务用线上数据"
             "持续迭代。",
             ("没有公认的统一最佳实践", "缺乏标准化评测", "持续迭代")),
        ),
        pitfall="很多人只会说'难的问题多给点算力'这种模糊策略，说不出具体的信号来源和分层升级/并行双轨这类具体机制，更容易在最后一层把自己的方案包装成'业界标准做法'而不是诚实承认这是持续迭代的工程摸索。",
        real_world_link="learning/serving-graduation/src/multi_model_router.py",
    ),
    DeepPoint(
        id="dp-fr-rtc-21",
        cat=CAT,
        trigger="你说整个行业好像都在从'训练更大模型'转向'推理时多花算力'，这个范式转变你怎么理解，它能一直这样scale下去吗？",
        chain=(
            ("这个范式转变具体指的是什么？",
             "指的是从预训练规模(参数量、数据量、训练算力)是提升能力主要杠杆的传统scaling law叙事，转向在推理"
             "阶段花更多计算也能大幅提升能力这一新维度，o1/o3/DeepSeek-R1这批推理模型是这个转变的标志性例子，"
             "本质上是把花算力换能力这件事从只能在训练阶段做，扩展到了训练和推理两个阶段都能做。",
             ("传统scaling law", "训练和推理两个阶段", "标志性例子")),
            ("从产业角度看，这个转变对算力需求结构有什么具体影响？",
             "这个转变正在改变GPU采购和数据中心建设的方向，此前算力需求以训练为主，现在推理侧尤其是reasoning"
             "模型的推理需求因为每次请求要生成大量思考token而大幅增长，行业对推理优化型硬件的重视程度明显提升。",
             ("GPU采购", "推理侧", "推理优化型硬件")),
            ("那这个'多花test-time compute就能提升能力'的曲线，是不是可以无限延伸下去？",
             "不是，sequential scaling很快遇到收益递减甚至转负，parallel scaling虽然覆盖率更好但依赖verifier/"
             "投票机制，且随着采样数增加边际收益也会衰减；同时o3的例子说明把这条曲线推到极致的经济成本在当前"
             "阶段完全不具备大规模商用的可行性，所以test-time compute能一直scale更准确的说法是在特定预算区间"
             "内、特定任务类型上能带来实在收益。",
             ("收益递减", "边际收益也会衰减", "不具备大规模商用")),
            ("如果几年后回看，你觉得这波'test-time compute范式'会被证明是根本性的范式转变，还是只是阶段性的技术潮流？",
             "诚实的答案是现在还无法判断，这既可能是长期有效的根本性范式(如果算法效率持续提升、成本持续下降)，"
             "也可能只是在预训练数据/算力接近瓶颈期的一个阶段性补充手段，比较负责任的态度是承认这是一个正在快速"
             "演化、缺乏几年后验证的开放问题，不应该在面试或论文里把它包装成已经盖棺定论的历史判断。",
             ("现在还无法判断", "阶段性补充手段", "开放问题")),
        ),
        pitfall="很多人会把'test-time compute范式转变'讲成一个已经确定无疑的历史结论，说不出这条曲线本身在sequential/parallel两个维度上都有收益递减的证据，更不会在最后诚实承认这仍是一个开放判断而不是定论。",
        real_world_link="",
    ),
]


def _self_test() -> None:
    assert 19 <= len(BANK) <= 23, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("dp-fr-rtc-") for i in ids), "id前缀不一致"
    assert all(len(dp.chain) >= 3 for dp in BANK), "存在追问链层数不足3层的条目"
    assert all(dp.pitfall for dp in BANK), "存在缺失pitfall的条目"
    assert all(dp.trigger for dp in BANK), "存在缺失trigger的条目"
    for dp in BANK:
        answers = [ref for (_q, ref, _k) in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 采分关键词未能在参考答案里全部命中: {scores}"
    print(f"[PASS] dp_reasoning_testtime: {len(BANK)}个DeepPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
