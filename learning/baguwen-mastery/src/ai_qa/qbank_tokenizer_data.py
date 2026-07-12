"""Tokenizer 与数据八股问答库（约 11 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "Tokenizer与数据"

BANK: list[QA] = [
    QA(id="ai-tok-01", cat=CAT,
       q="为什么现代 LLM 用 subword(子词)分词，而不是 word-level 或 char-level？",
       a="早期NLP用word-level分词，词表必须覆盖几乎所有出现过的单词，遇到训练时没见过的词(OOV, "
         "out-of-vocabulary)只能用统一的<UNK>代替，丢失信息；char-level分词虽然没有OOV问题，但序列"
         "长度暴涨、单个字符几乎不携带语义，模型要学习的组合层级太深。subword(子词)分词是二者的折中："
         "把词切成更细的、复用率高的片段(如词根、词缀、常见字节组合)，常见词整体作为一个token、生僻"
         "词/新词拆成几个子词拼接表示，既避免了OOV(任何字符串都能用已有子词拼出来)，又把序列长度控制"
         "在合理范围，这正是BPE/WordPiece/SentencePiece等现代tokenizer的共同出发点。",
       keys=("OOV", "word-level", "char-level", "子词"),
       follow_ups=("如果一个词被切成好几个子词，模型怎么知道它们该拼回一个词？", "中文这种没有天然空格边界的语言，用BPE会有什么问题？")),

    QA(id="ai-tok-02", cat=CAT,
       q="BPE(Byte Pair Encoding)训练阶段具体是怎么构造词表的？",
       a="BPE训练时先把语料切成最基本的符号单元(字符或字节)，词表初始就是这些基本符号；然后统计语料"
         "里所有相邻符号对(pair)的出现频率，把频率最高的那一对合并成一个新的符号、加入词表，重复"
         "'统计频率->合并最高频pair'这个过程若干轮，每轮词表增加一个新token，直到达到预设的词表大小。"
         "因为合并顺序是按频率从高到低进行的，最终词表里天然会保留高频完整词、把低频词拆成更细的子词"
         "组合。",
       keys=("字节", "相邻", "频率", "合并"),
       follow_ups=("BPE的词表大小是提前定好的吗，合并轮数怎么对应最终词表大小？", "BPE对生僻词/多语言的处理为什么不如WordPiece细致？")),

    QA(id="ai-tok-03", cat=CAT,
       q="训练好 BPE 之后，编码(推理时切词)是怎么执行的？",
       a="训练阶段得到的不只是最终词表，还有一份按顺序记录下来的合并规则列表(merge rules，即'哪两个"
         "符号在第几轮被合并')。编码一个新句子时，先把它切成最基本的字符/字节序列，然后严格按训练时"
         "的合并顺序，只要句子里出现了某条规则里的相邻符号对，就执行同样的合并，直到没有规则能再应用"
         "为止。这保证了推理时的切词方式和训练时词表的构造方式完全一致，同一个词永远被切成同样的子词"
         "序列。",
       keys=("合并规则", "顺序", "字节", "一致"),
       follow_ups=("如果编码时遇到训练阶段完全没见过的字符组合怎么办？", "BPE的合并规则数量和词表大小是什么关系？")),

    QA(id="ai-tok-04", cat=CAT,
       q="WordPiece 和 BPE 的核心区别是什么？",
       a="BPE每一轮选择的是当前语料里出现频率最高的相邻符号对，是纯粹的计数问题；WordPiece的合并"
         "标准不是频率而是'合并后能让训练语料的似然提升最多'，具体用一个类似互信息的打分公式——"
         "score(pair) ≈ freq(pair)/(freq(first)·freq(second))，也就是这对符号共同出现的频率相对于"
         "二者各自独立出现频率的'意外程度'。这意味着两个都非常高频、但共现纯属巧合的符号不会被优先"
         "合并，而像'play'和'##ing'这种共现频率明显高于各自独立频率之积的组合会被优先合并，因此"
         "WordPiece切出来的子词往往更贴近有意义的词根/词缀边界，对生僻词的切分也更保守，BERT系模型"
         "采用的就是WordPiece。",
       keys=("似然", "互信息", "频率", "BERT"),
       follow_ups=("WordPiece的打分公式具体是怎么体现'似然增益'的？", "为什么BERT选WordPiece而不是原始BPE？")),

    QA(id="ai-tok-05", cat=CAT,
       q="SentencePiece 相比 BPE/WordPiece 的传统实现，最大的特点是什么？",
       a="SentencePiece最大的特点是不依赖任何语言相关的预分词(pre-tokenization)步骤——BPE和"
         "WordPiece传统上都要求先用空格/标点把文本切成'词'，再在词内部做子词合并，这对中文、日文这类"
         "没有天然空格分词边界的语言很不友好。SentencePiece把原始文本(包括空格)当作一个统一的字符流"
         "直接喂给BPE或Unigram算法，用一个特殊符号▁(U+2581)显式表示空格并把它当成普通字符参与合并/"
         "建模，这样解码时只需把所有子词拼接、再把▁还原成空格即可，整个流程语言无关、可逆，不需要为"
         "不同语言写不同的预分词器，因此被T5、LLaMA等多语言/跨语言模型广泛采用。",
       keys=("预分词", "▁", "字符流", "语言无关"),
       follow_ups=("▁符号具体解决了什么问题，不用它会怎样？", "SentencePiece训练时可以选BPE也可以选Unigram，实践中怎么选？")),

    QA(id="ai-tok-06", cat=CAT,
       q="Unigram 语言模型分词法和 BPE/WordPiece 的构造方向有什么不同？",
       a="Unigram语言模型分词法(SentencePiece里可选的另一种算法)和BPE/WordPiece的构造方向相反："
         "BPE/WordPiece是自底向上、从最小单元开始逐步合并扩大词表；Unigram是自顶向下，先建一个远大于"
         "目标大小的候选子词词表，为每个子词估计一个概率，然后迭代地裁剪掉那些去掉后对整个语料似然"
         "损失最小的子词，直到收缩到目标词表大小。它的额外好处是分词本身是概率性的，同一个词可能有"
         "多种切分方式，训练时可以按概率采样不同切分做数据增强(subword regularization)，让模型对"
         "切分边界更鲁棒。",
       keys=("自顶向下", "裁剪", "概率", "subword regularization"),
       follow_ups=("subword regularization具体怎么在训练时做概率采样？", "Unigram和BPE最终切出来的词表在效果上有什么实测差异？")),

    QA(id="ai-tok-07", cat=CAT,
       q="byte-level BPE(GPT 系 tokenizer)相比字符级 BPE 有什么好处？",
       a="byte-level BPE(GPT-2/GPT-4系tokenizer采用)不是在Unicode字符上做BPE，而是先把文本按UTF-8"
         "编码成字节序列(0-255共256个基础符号)，再在字节上做标准BPE合并。好处是词表的基础层永远只有"
         "256个字节、天生覆盖任意语言、任意符号(emoji、特殊符号、代码里的运算符)，从根本上杜绝了UNK"
         "——因为任何输入串都能被分解成已知的字节，不存在'词表外字符'这回事；代价是常见非ASCII字符"
         "(比如中文汉字)会先被拆成多个字节再重新学习合并，早期训练不充分时可能切得比较碎。",
       keys=("UTF-8", "字节", "UNK", "256"),
       follow_ups=("byte-level BPE怎么保证常见ASCII字符不会被过度拆分？", "为什么GPT系列选byte-level而不是Unicode字符级BPE？")),

    QA(id="ai-tok-08", cat=CAT,
       q="词表大小(vocab size)怎么影响模型的参数量和序列长度？",
       a="词表大小是一个直接的工程权衡：词表越大，同样的文本可以用更少的token表示、序列变短、自回归"
         "解码步数变少、长距离依赖更容易建模；但输入embedding矩阵和输出softmax层的参数量都正比于词表"
         "大小，词表越大这两层的参数和计算量(尤其输出层的矩阵乘+softmax)越大，对小模型来说这部分开销"
         "占比会显得不成比例地高。词表太小则相反——序列变长、生成变慢，但两头的embedding/输出层更省"
         "参数；本质上是在词表参数量和序列长度之间做权衡。",
       keys=("embedding", "softmax", "序列长度", "参数量"),
       follow_ups=("词表大小翻倍，模型的embedding参数量大概涨多少？", "有没有办法让大词表不显著增加输出层的计算量？")),

    QA(id="ai-tok-09", cat=CAT,
       q="预训练数据配比(data mixture)为什么重要？",
       a="预训练数据配比(data mixture)指的是网页文本、代码、书籍、学术论文、对话等不同来源在训练"
         "语料里各占多少比例。不同来源对模型能力的贡献是不对称的——代码数据被反复观察到能提升模型的"
         "逻辑推理能力(不仅是写代码的能力)，高质量书籍/论文提升长程连贯性和知识密度，网页文本提供规模"
         "和多样性但质量参差不齐；配比选得不好，要么某些能力(如代码、多语言)学不够，要么低质量来源"
         "占比过高拖累整体质量，因此现代大模型预训练通常会做配比消融实验、甚至在训练后期动态调整"
         "配比(如后期提高高质量数据比例)。",
       keys=("代码", "配比", "来源", "消融"),
       follow_ups=("代码数据为什么能提升非代码任务的推理能力？", "怎么用消融实验验证一个配比方案更好？")),

    QA(id="ai-tok-10", cat=CAT,
       q="预训练数据去重(deduplication)为什么重要？不去重会有什么后果？",
       a="预训练语料里常有大量重复或近似重复的文档(转载、镜像站、模板化网页)，如果不做去重，模型会"
         "在同一段文本上被训练很多遍。已有研究(如Anthropic的repeated data scaling law工作)发现，"
         "哪怕只有很小一部分数据(如0.1%)被重复上百次，也会显著伤害模型效果，甚至出现类似double "
         "descent的非单调性能下降，因为模型会把大量容量用来记忆(memorize)这些重复片段而不是学习"
         "可泛化的模式，这种记忆还会挤占induction head等有利于泛化的内部结构。因此数据去重(exact/"
         "fuzzy dedup)被现代预训练pipeline(如FineWeb-Edu、Dolma、RedPajama)当作标配步骤，能同时"
         "降低记忆风险、提升有效数据利用率。",
       keys=("重复", "记忆", "double descent", "去重"),
       follow_ups=("exact dedup和fuzzy dedup(近似去重)具体方法上有什么不同？", "重复数据的伤害和重复次数/占比之间是什么关系？")),

    QA(id="ai-tok-11", cat=CAT,
       q="数据质量对模型能力有什么影响？常见的质量过滤手段有哪些？",
       a="数据质量过滤(quality filtering)对模型能力的影响遵循'垃圾进垃圾出'：低质量(乱码、SEO堆砌"
         "关键词、机器生成的低质文本、有害内容)如果不过滤，模型会把这些模式也学进语言分布里，拖累"
         "下游任务表现、甚至污染指令遵循和事实性。常见的质量过滤手段包括：基于规则的启发式过滤(去除"
         "过短/过长文档、异常符号比例、语言检测)、用小分类器对文档质量打分(如是否类似维基百科/教科书"
         "的高质量文本)、近似去重、以及针对已知有害/低价值域名的黑名单过滤；这些步骤组合起来往往比"
         "单纯堆数据规模对模型能力的提升更明显，这也是FineWeb-Edu等强调'教育价值过滤'数据集出现的"
         "原因。",
       keys=("质量过滤", "启发式", "打分", "FineWeb"),
       follow_ups=("常见的启发式过滤规则具体有哪些例子？", "'教育价值过滤'的分类器一般是怎么训练出来的？")),
]


def _self_test() -> None:
    assert 9 <= len(BANK) <= 16, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("ai-tok-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    assert quiz(BANK, cat=CAT) == BANK
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    print(f"[PASS] qbank_tokenizer_data: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
