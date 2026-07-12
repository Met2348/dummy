"""评估指标八股问答库（约 11 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "评估指标"

BANK: list[QA] = [
    QA(id="ai-met-01", cat=CAT,
       q="混淆矩阵(confusion matrix)由哪些元素组成，为什么它是理解分类指标的基础？",
       a="混淆矩阵是一个表格，记录预测类别与真实类别的交叉计数：TP(真正例，预测为正且实际为正)、"
         "FP(假正例，预测为正但实际为负，即误报)、FN(假负例，预测为负但实际为正，即漏报)、"
         "TN(真负例，预测为负且实际为负)。precision、recall、F1、accuracy 全部是这四个数的不同组合，"
         "脱离混淆矩阵单独硬记指标公式容易混乱，面试时建议先画出这张表再推导每个指标。",
       keys=("TP", "FP", "FN", "TN"),
       follow_ups=("precision和recall分别怎么用这四个数算？", "多分类问题的混淆矩阵怎么画、还是2x2吗？")),

    QA(id="ai-met-02", cat=CAT,
       q="precision 和 recall 的区别是什么？各自更适合什么场景？",
       a="precision(查准率)=TP/(TP+FP)，衡量预测为正的样本里有多少是真正例，代表'预测的可信度'；"
         "recall(查全率)=TP/(TP+FN)，衡量真正例里有多少被找出来，代表'覆盖程度'。二者通常此消彼长——"
         "放宽判正阈值能提高recall但拉低precision。场景选择：垃圾邮件识别宁可漏判也不误伤正常邮件"
         "(重precision)；癌症筛查、风控宁可多查一些疑似也不漏诊(重recall)。",
       keys=("查准率", "查全率", "TP", "阈值"),
       follow_ups=("如果只能上报一个指标，什么情况该只看precision或只看recall？", "precision和recall的trade-off背后的数学原因是什么？")),

    QA(id="ai-met-03", cat=CAT,
       q="什么时候 accuracy 具有误导性？",
       a="类别高度不平衡时accuracy会严重失真：比如欺诈样本只占0.1%，模型无脑把全部样本预测为'非欺诈'"
         "就能拿到99.9%的accuracy，却没有任何实际检测能力。这种情况下应改看F1、PR-AUC、召回率或"
         "每类单独的precision/recall，而不是被accuracy的高数字误导。",
       keys=("不平衡", "欺诈", "F1", "PR-AUC"),
       follow_ups=("除了F1，还有哪些指标能应对类别不平衡？", "什么叫baseline accuracy(多数类占比)，为什么要先算它？")),

    QA(id="ai-met-04", cat=CAT,
       q="F1 score 是什么？为什么用调和平均而不是算术平均？",
       a="F1 = 2·precision·recall/(precision+recall)，是precision和recall的调和平均而非算术平均。"
         "用调和平均是因为它对较小值更敏感——只要precision和recall有一个很低，F1就会被明显拉低，"
         "从而惩罚'一高一低'的失衡策略(比如全预测为正来刷recall)，逼模型两者兼顾。",
       keys=("调和平均", "precision", "recall", "惩罚"),
       follow_ups=("F-beta score里的beta参数是干什么的？", "F1只反映一个阈值下的表现，这算它的局限吗？")),

    QA(id="ai-met-05", cat=CAT,
       q="ROC-AUC 和 PR-AUC 该在什么情况下分别选用？",
       a="ROC-AUC衡量模型在所有阈值下真正例率(TPR/recall)-假正例率(FPR)曲线下面积，适合正负样本"
         "大致均衡的场景；PR-AUC衡量precision-recall曲线下面积，在正负不平衡、正例稀少(如欺诈检测、"
         "信息检索)的场景更敏感，因为正例极少时FPR天然很低，ROC曲线会显得'虚高'、掩盖模型在少数正类"
         "上的真实表现，这时PR-AUC不被大量真负例稀释、更能反映实际检索/召回质量。",
       keys=("TPR", "FPR", "不平衡", "稀释"),
       follow_ups=("ROC-AUC=0.5代表什么？", "PR-AUC的随机基线水平取决于什么？")),

    QA(id="ai-met-06", cat=CAT,
       q="什么是模型校准(calibration)？怎么衡量和修正？",
       a="模型校准(calibration)指预测概率与真实频率的一致程度——如果模型说某组样本是正类的概率是0.8，"
         "这组样本里真实为正的比例应该也接近80%。常用可靠性图(reliability diagram)和ECE(期望校准误差，"
         "把预测按置信度分桶后算每桶预测概率与实际频率之差的加权平均)衡量；现代深度网络往往过度自信"
         "(over-confident)，校准较差。可以用temperature scaling(用一个标量温度缩放logits后再softmax)"
         "做后校准，它只改变概率的绝对值、不改变排序(不影响accuracy/AUC)。",
       keys=("真实频率", "ECE", "温度", "过度自信"),
       follow_ups=("temperature scaling为什么不会改变accuracy？", "除了ECE还有哪些校准误差的衡量方式？")),

    QA(id="ai-met-07", cat=CAT,
       q="困惑度(perplexity)是什么？和交叉熵是什么关系？",
       a="困惑度(perplexity)= exp(平均交叉熵损失)，衡量语言模型对测试集每个token的平均'意外程度'或"
         "不确定性——可以直觉理解为模型平均在多少个候选词之间犹豫不决。困惑度越低说明模型给真实next "
         "token分配的概率越高、语言建模能力越强；它和交叉熵是单调的指数关系，只是把log空间的loss换算成"
         "更直观的'有效候选词数'。",
       keys=("交叉熵", "exp", "token", "不确定性"),
       follow_ups=("perplexity能不能跨不同tokenizer的模型直接比较？", "perplexity低是否等价于生成质量好？")),

    QA(id="ai-met-08", cat=CAT,
       q="BLEU 是什么、怎么算？",
       a="BLEU(常用于机器翻译)通过统计候选译文与一个或多个参考译文之间n-gram(通常1-gram到4-gram)的"
         "重合精度(precision)，再乘以一个brevity penalty(简短惩罚)防止模型靠输出很短的句子刷高精度——"
         "因为n-gram precision天然偏向短输出。多个n-gram精度通常取几何平均后再乘惩罚项。BLEU对参考"
         "译文的表面词形匹配敏感，同义改写、语序调整会被低估。",
       keys=("n-gram", "precision", "brevity penalty", "机器翻译"),
       follow_ups=("brevity penalty具体怎么计算？", "BLEU对同义改写不敏感的问题，后续有什么改进指标？")),

    QA(id="ai-met-09", cat=CAT,
       q="ROUGE 是什么？和 BLEU 的核心区别在哪？",
       a="ROUGE(常用于摘要生成)和BLEU相反地偏重召回(recall)：ROUGE-N统计候选摘要覆盖了参考摘要里"
         "多少n-gram，ROUGE-L用最长公共子序列(LCS)衡量顺序保持的重合程度。因为摘要任务更关心'该说的"
         "要点有没有被覆盖'而不是'是否一字不差'，所以用recall导向的ROUGE而不是precision导向的BLEU；"
         "实践中常常两个方向都报(ROUGE的precision/recall/F值三件套)。",
       keys=("recall", "LCS", "摘要", "n-gram"),
       follow_ups=("ROUGE-1/ROUGE-2/ROUGE-L分别衡量什么？", "为什么摘要评测常用ROUGE而翻译常用BLEU？")),

    QA(id="ai-met-10", cat=CAT,
       q="多分类场景下 macro-F1 和 micro-F1 有什么区别？",
       a="macro-F1先对每个类别单独算F1再取算术平均，每个类别权重相同，对小类别更敏感，能暴露模型在"
         "少数类上的短板；micro-F1把所有类别的TP/FP/FN汇总后统一算一个全局F1，等价于按样本数加权，"
         "大类别主导结果，会掩盖小类别上的糟糕表现。类别不平衡时应同时报两者，避免micro-F1的高分掩盖"
         "macro-F1暴露的问题。",
       keys=("macro", "micro", "加权", "不平衡"),
       follow_ups=("weighted-F1和macro/micro-F1有什么区别？", "类别数很多且极不平衡时你会怎么汇报整体指标？")),

    QA(id="ai-met-11", cat=CAT,
       q="分类决策阈值(threshold)应该怎么选，为什么不是固定 0.5？",
       a="分类模型输出的是概率，决策阈值(threshold)不是固定0.5——应结合precision-recall的业务代价来选："
         "漏检代价高(如疾病筛查、风控)就调低阈值提高recall，误报代价高(如自动封号)就调高阈值提高"
         "precision。实践中常画precision-recall曲线或代价曲线，选在满足业务约束(如precision>=90%)下"
         "recall最大的那个阈值点，而不是拍脑袋用0.5。",
       keys=("阈值", "precision", "recall", "业务"),
       follow_ups=("怎么用precision-recall曲线选一个满足业务约束的阈值？", "阈值选择和模型训练过程本身有关系吗？")),
]


def _self_test() -> None:
    assert 9 <= len(BANK) <= 16, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-met-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    assert quiz(BANK, cat=CAT) == BANK
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_metrics: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
