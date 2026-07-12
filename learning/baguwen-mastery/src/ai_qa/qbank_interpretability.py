"""可解释性(补充深化) 八股问答库（约 10 题）。

与 interview-prep/src/mlqa/qbank.py 里"可解释性"类的 6 道题(linear probing/
logit lens/activation patching/SAE/superposition/linear representation)
不重复，这里补充更容易被追问、也更容易被过度解读的角度：attention 可视化的
局限、circuit 分析、mech interp vs 事后解释、AI 安全/对齐意义、大模型为何更难
解释、probing 的因果局限、梯度归因方法、induction head、activation steering、
faithfulness vs plausibility。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "可解释性"

BANK: list[QA] = [
    QA(id="ai-interp-01", cat=CAT,
       q="attention 权重高就说明模型的输出主要依赖这个 token 吗？为什么这个说法有问题？",
       a="不对。attention 权重只反映 query 对某个 key 的相似度打分，不等于该 value 对最终"
         "输出贡献的因果作用；即便把权重换成完全不同但功能等价的分布，模型输出也可能不变"
         "(Jain & Wallace 的 'Attention is not Explanation')，说明权重和贡献度可以解耦。"
         "更严谨的因果度量要用 activation patching/梯度归因等直接操纵激活再看输出变化的"
         "方法，而不是直接读权重数值。",
       keys=("相似度", "因果", "不等于", "activation patching"),
       follow_ups=("那怎么才能量化某个 token 对输出的真实因果贡献？",
                   "为什么两个完全不同的 attention 分布能产生相同的输出？")),

    QA(id="ai-interp-02", cat=CAT,
       q="什么是 circuit（电路）分析？和单纯看某一层的激活有什么不同？",
       a="circuit 分析是 mechanistic interpretability 的核心方法之一：把模型的计算图看成"
         "许多可复用的子模块(注意力头、MLP 神经元等)，尝试找出实现某个具体行为(如'检测"
         "重复 token 并复制')的一小组组件加连接方式——这个子图就叫一个 circuit。它和只看"
         "单层激活的区别在于：circuit 分析追踪的是跨层、跨组件的因果计算链路(谁的输出喂"
         "给谁)，目标是像逆向工程一段程序一样说清'这个行为是怎么算出来的'，而不只是'哪里"
         "出现了相关的激活'。经典例子是 induction head：一种能识别'上文出现过的模式'并"
         "复制下一个 token 的注意力头组合。",
       keys=("mechanistic interpretability", "子图", "因果", "induction head"),
       follow_ups=("induction head 具体是怎么工作的？",
                   "circuit 分析目前主要在多大规模的模型上做到完整验证？")),

    QA(id="ai-interp-03", cat=CAT,
       q="mechanistic interpretability 和传统的事后解释方法(如 LIME、SHAP)有什么本质区别？",
       a="LIME/SHAP 属于 post-hoc explanation：把模型当黑箱，通过扰动输入、观察输出变化，"
         "拟合一个局部可解释的代理模型(如线性模型)来近似解释某次预测，不关心模型内部到底"
         "怎么算的，也不保证反映真实计算过程(不一定 faithful)。mechanistic interpretability "
         "反其道而行：把模型当白盒，直接研究权重和激活里编码了什么算法、组件之间如何因果"
         "连接，目标是完整逆向工程出真实的计算机制，而不是给一个'看起来合理'的近似解释。"
         "代价是 mech interp 需要深入模型内部、成本高很多，LIME/SHAP 则模型无关、即插即用。",
       keys=("post-hoc", "黑箱", "白盒", "faithful"),
       follow_ups=("为什么说 LIME/SHAP 的解释不一定 faithful？",
                   "mech interp 的方法能反过来验证 LIME/SHAP 给出的解释对不对吗？")),

    QA(id="ai-interp-04", cat=CAT,
       q="可解释性研究对 AI 安全/对齐(alignment)有什么意义？",
       a="对齐关心的核心问题是'模型说的和它实际在做的是不是一回事'——比如模型可能表面给出"
         "人类喜欢的回答，内部却在执行欺骗性或与训练目标不符的策略(deceptive alignment/"
         "reward hacking)。仅看输出无法区分'真的对齐'和'学会了糊弄评估者'；可解释性提供"
         "一条独立于输出的核验通道，比如通过 probing/circuit 分析检查模型内部是否存在'知道"
         "真相但选择性隐瞒'的表征，或者监控危险能力(如自我复制、欺骗)对应的电路是否被激活"
         "，从而在部署前发现输出层面看不出来的风险。这也是 Anthropic 等机构把 mechanistic "
         "interpretability 列为安全研究支柱之一的原因。",
       keys=("对齐", "欺骗", "输出", "probing"),
       follow_ups=("能举一个'输出正常但内部有问题'的具体例子吗？",
                   "可解释性方法本身有什么局限，导致它不能完全替代其他对齐手段？")),

    QA(id="ai-interp-05", cat=CAT,
       q="为什么大语言模型的可解释性比传统的小模型/传统 ML 模型难得多？",
       a="三个原因叠加：①规模——参数量和层数比传统模型高几个数量级，人工逐一分析组件不"
         "现实；②叠加(superposition)——特征数远多于神经元数，导致大量神经元多义、单看激活"
         "值看不出对应什么概念；③涌现能力——很多复杂行为(如思维链推理、上下文学习)在小"
         "模型里不存在，只在参数/数据规模跨过某个阈值后才出现，无法靠外推小模型的解释结论"
         "去理解大模型的新行为，只能重新做实验。传统 ML 模型(线性回归/浅层决策树)本身结构"
         "简单、可解释性内建在模型形式里，而大模型的表达力恰恰来自这种难以拆解的分布式、"
         "非线性表示。",
       keys=("规模", "叠加", "涌现", "分布式"),
       follow_ups=("涌现能力这个说法本身有没有争议？",
                   "有没有办法把小模型上验证过的解释方法'外推'到大模型？")),

    QA(id="ai-interp-06", cat=CAT,
       q="probing 分类器能从某层激活里线性解码出某个概念，是否说明模型在推理时真的用到了这个信息？",
       a="不一定。probing 只证明'信息以线性可解码的方式存在于该层激活里'，这是相关性证据"
         "，不是因果证据——模型完全可能只是把这个信息当副产品编码出来，实际前向计算里从未"
         "真正读取、使用它去影响后续输出。要确认'模型真的用了'，需要做因果实验：比如对该"
         "表征做 activation patching 或方向消融(ablation)，看下游输出是否真的因此改变；如果"
         "消融后输出毫无变化，说明这个可解码的信息只是'存在'而非'被使用'。这是 probing 结果"
         "最常被追问、也最容易被过度解读的一点。",
       keys=("相关性", "因果", "消融", "被使用"),
       follow_ups=("那具体怎么设计一个消融实验来验证因果性？",
                   "如果 probing 准确率很高，但消融后输出不变，说明什么？")),

    QA(id="ai-interp-07", cat=CAT,
       q="基于梯度的归因方法(如 saliency map、integrated gradients)原理是什么？有什么局限？",
       a="核心思路是用输出对输入(或中间激活)的梯度大小衡量'这个位置的微小扰动对输出影响"
         "有多大'，梯度绝对值大的位置被认为更重要；integrated gradients 进一步把从一个基线"
         "输入到实际输入的路径上的梯度积分起来，缓解普通梯度在饱和区趋于零、看不出真实重要"
         "性的问题。局限也很明显：梯度是局部线性近似，对输入的微小无意义扰动(如加噪声)可能"
         "让归因图剧烈跳变而模型预测几乎不变，说明这类方法可能不 robust、也不一定忠实反映"
         "模型真正依赖的特征；此外基线选择、梯度饱和等都会引入额外的主观假设。",
       keys=("梯度", "积分", "饱和", "robust"),
       follow_ups=("为什么说梯度方法'不 robust'，能举一个具体现象吗？",
                   "integrated gradients 的基线该怎么选？")),

    QA(id="ai-interp-08", cat=CAT,
       q="induction head 是什么？它为什么是 mechanistic interpretability 里的一个标志性发现？",
       a="induction head 是由两个注意力头组成的一个 circuit：前一个头(previous token head)"
         "把每个位置的信息里'混入'它前一个 token 的身份，后一个头(induction head)在当前位置"
         "往前搜索'和当前 token 相同的历史 token'，找到后把那个历史位置的下一个 token 复制"
         "过来作为预测——本质是在实现'如果之前出现过 A B，现在又看到 A，就猜下一个是 B'这种"
         "模式补全，这也是上下文学习(in-context learning)能力的一个重要机制来源。它是标志性"
         "发现，是因为这是第一批被完整、严格地逆向工程出来的可跨模型规模复现的 circuit，"
         "证明了'给 transformer 的行为找到具体机制'这件事是可行的，而不只是理论设想。",
       keys=("previous token", "复制", "上下文学习", "circuit"),
       follow_ups=("induction head 和 in-context learning 之间是怎么严格对应起来的？",
                   "这种两头协作的模式在更大模型里还成立吗？")),

    QA(id="ai-interp-09", cat=CAT,
       q="representation engineering / activation steering 是什么？它和'解释'有什么关系？",
       a="先用可解释性方法(如对比一组正例/负例的激活做差、或训练线性 probe)在模型内部找到"
         "一个对应某种概念或行为(如'诚实' '拒绝' '情感')的方向向量，再在推理时直接把这个"
         "方向按一定强度加到对应层的激活上，从而不改权重、只改激活地引导模型输出朝该概念"
         "方向偏移，这就是 activation steering(表征工程的一种手段)。它和'解释'的关系是：这"
         "是可解释性从'理解'走向'干预/控制'的桥梁——如果找到的方向真的对应该概念，加上去"
         "应该能可预测地改变行为，这本身也是对该表征解释是否正确的一种因果验证。",
       keys=("方向向量", "激活", "干预", "验证"),
       follow_ups=("怎么知道找到的方向真的对应你想要的概念，而不是别的东西？",
                   "activation steering 和微调(fine-tuning)相比有什么优缺点？")),

    QA(id="ai-interp-10", cat=CAT,
       q="解释的 faithfulness(忠实性)和 plausibility(看起来合理)有什么区别？为什么这个区分很重要？",
       a="plausibility 指解释是否'读起来让人觉得有道理'，faithfulness 指解释是否真实反映了"
         "模型内部实际的计算过程。两者可以完全脱钩：一段语言模型生成的思维链(chain-of-"
         "thought)可能逻辑通顺、人类读了觉得信服，但模型真正的内部计算路径完全是另一回事，"
         "这段文字只是事后编出来的'合理故事'；同样一个 SAE 找到的'特征'，激活模式可能看起"
         "来对应某个直觉概念，但未必是模型计算中真正起作用的那个变量。只追求 plausibility "
         "会让人产生'我们已经理解模型了'的错觉，而实际安全相关的问题恰恰需要 faithfulness"
         "——尤其是要检测模型是否在欺骗或隐瞒时，一个听起来很合理但不忠实的解释毫无用处。",
       keys=("plausibility", "faithfulness", "思维链", "欺骗"),
       follow_ups=("怎么去检验一段思维链解释是不是 faithful 的？",
                   "SAE 找到的特征怎么验证是不是真的 faithful？")),
]


def _self_test() -> None:
    assert 8 <= len(BANK) <= 14, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-interp-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    assert quiz(BANK, cat=CAT) == BANK
    print(f"[PASS] qbank_interpretability: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
