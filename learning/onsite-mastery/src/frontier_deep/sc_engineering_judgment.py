"""无标准答案的工程判断力场景题：ScenarioPoint 版本。

区别于 DeepPoint（每层追问都有唯一参考答案+采分关键词，逼你对照"接没接住"），
这里的场景故意不给"标准答案"——"线上效果变差怎么定位""供应商限流怎么降级"这类问题
考的是工程判断力和思路完整度，不存在唯一正解。ScenarioPoint 用 rubric 列出一个扎实回答
应该覆盖的要点（3-5条），用 trap 说明大多数人只说了什么浅层内容就停住了；grade_scenario
只判断"覆盖面够不够"，不判断"对不对"。

场景类型来源：2025-2026 面经调研 + WebSearch 交叉验证，覆盖生产事故定位、供应商容灾、
Agent 安全、评测方法论、训练基础设施故障、对齐与红队、A/B统计陷阱、研究可复现性、
跨团队协调、伦理/PR危机等多种工程判断场景，不集中在单一主题。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import ScenarioPoint, categories, grade_scenario  # noqa: E402

CAT = "工程判断力场景题(无标准答案)"

BANK: list[ScenarioPoint] = [
    ScenarioPoint(
        id="sc-fr-eng-01",
        cat=CAT,
        trigger="线上模型效果突然变差，你怎么定位？",
        rubric=(
            "对比线上线下数据分布差异",
            "区分数据/模型/服务三层问题",
            "灰度回滚止损",
            "事后建立监控告警防复发",
        ),
        trap="大多数人只会说'查日志'或'先回滚'，说不清先看哪一层、怎么用数据证明是分布问题"
        "还是服务问题，也没提事后怎么建立监控防止同类问题复发。",
        real_world_link="production ML monitoring / online-offline skew 排查套路",
    ),
    ScenarioPoint(
        id="sc-fr-eng-02",
        cat=CAT,
        trigger="你依赖的模型API供应商突然限流或降级，你怎么办？",
        rubric=(
            "降级到备用模型或本地小模型",
            "请求排队与背压控制",
            "同步业务方降级后的效果预期",
            "事前做好多供应商容灾设计",
        ),
        trap="大多数人只会说'重试'或'换个供应商'，没考虑背压不当会导致请求雪崩，"
        "没想过要提前跟业务方说清楚降级会牺牲哪些效果，更没有事前多供应商容灾的意识。",
        real_world_link="多模型网关 fallback 路由设计模式",
    ),
    ScenarioPoint(
        id="sc-fr-eng-03",
        cat=CAT,
        trigger="Agent 在生产环境里调用了错误的工具，或者传错了参数，你怎么办？",
        rubric=(
            "排查是否schema设计导致模型误解",
            "工具调用前增加校验层",
            "补偿已产生的错误动作",
            "高风险操作引入人工确认关卡",
        ),
        trap="大多数人只会说'加日志监控'，不会往前追问是不是工具schema本身让模型产生歧义，"
        "也没想清楚已经执行的错误动作该怎么补偿或回滚。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-04",
        cat=CAT,
        trigger="如何证明你的新Prompt或新模型版本真的比旧的好，而不是你自己感觉更好？",
        rubric=(
            "离线评测集加在线A/B双重验证",
            "看指标分布而不只看均值",
            "明确定义好坏的具体指标",
            "警惕评测集被过拟合或污染",
        ),
        trap="大多数人只会说'我跑了几个case感觉更好'，或者只报一个均值分数，"
        "没有做对照实验、没定义清楚'好'到底是什么指标，也没意识到自己可能在无意识地对评测集调参。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-05",
        cat=CAT,
        trigger="预训练一个千亿参数模型，训练中途loss突然剧烈spike，你怎么诊断和处理？",
        rubric=(
            "检查数据batch里的异常样本",
            "检查数值不稳定如fp16溢出",
            "检查优化器与学习率调度",
            "回滚到spike前checkpoint并跳过可疑数据",
        ),
        trap="大多数人只会说'降低学习率重跑'，不会系统性排查到底是数据脏了、数值溢出了、"
        "还是调度器本身的问题，处理上也想不到要精确跳过具体哪一段可疑数据。",
        real_world_link="Meta OPT-175B logbook 记录的loss spike排查",
    ),
    ScenarioPoint(
        id="sc-fr-eng-06",
        cat=CAT,
        trigger="一个AI系统通过了你所有的安全评测，但部署后表现出令人担忧的行为，你怎么应对？",
        rubric=(
            "承认评测集和真实分布存在gap",
            "建立部署后持续监控机制",
            "准备快速下线或限流的应急预案",
            "诚实评估是否是评测设计的系统性盲区",
        ),
        trap="大多数人只会说'赶紧修复模型'，不会往回反思评测体系本身是不是有系统性盲区，"
        "也没有提前准备好能快速止损的应急预案。",
        real_world_link="Anthropic Responsible Scaling Policy 部署后监控理念",
    ),
    ScenarioPoint(
        id="sc-fr-eng-07",
        cat=CAT,
        trigger="A/B实验结果显示统计显著，但业务方就是不信、坚持不愿意上线，你怎么办？",
        rubric=(
            "区分统计显著与效果量大小",
            "检查是否存在分群体的负向效应",
            "排查peeking等实验设计漏洞",
            "用业务语言而非p值解释结论",
        ),
        trap="大多数人只会说'p值小于0.05就该上'，既不会检查有没有对某个高价值分群是负向的，"
        "也不知道用业务能听懂的语言重新解释结论，只会反复强调统计显著这一件事。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-08",
        cat=CAT,
        trigger="你想复现一篇论文的核心结果，但怎么都跑不出论文报的数字，你怎么排查？",
        rubric=(
            "核对超参数与数据预处理细节",
            "检查随机种子与评测协议一致性",
            "联系作者或查看官方开源代码",
            "评估是否论文本身存在选择性汇报",
        ),
        trap="大多数人只会说'再多调调参数试试'，不会系统性对照超参和评测协议的每个细节，"
        "也没想过论文本身可能存在cherry-pick或者没公开的trick。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-09",
        cat=CAT,
        trigger="两个团队用各自的评测方法对同一个模型给出了相反的结论，你怎么协调？",
        rubric=(
            "对齐评测集和评测协议是否一致",
            "拆解到具体case级别的分歧点",
            "明确谁的评测更贴近真实业务目标",
            "推动建立统一的评测基线",
        ),
        trap="大多数人只会说'开会讨论一下'，不会真的去对比两边评测集和协议的具体差异，"
        "也没想着建立一个双方都认可的统一基线来避免以后反复扯皮。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-10",
        cat=CAT,
        trigger="模型在训练集和离线评测集上表现完美，但线上效果很差，你怎么判断是过拟合还是分布漂移？",
        rubric=(
            "对比训练分布与线上真实分布",
            "检查评测集是否泄漏或过于简单",
            "分析线上badcase的具体类型",
            "用新鲜未见数据重新评估",
        ),
        trap="大多数人只会说'可能过拟合了加正则化'，不会先去对比线上真实数据分布和训练分布"
        "差在哪里，也不会怀疑评测集本身是不是已经泄漏或者太简单。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-11",
        cat=CAT,
        trigger="大规模分布式训练任务跑到一半，某个节点突然挂了，你怎么处理？",
        rubric=(
            "从最近checkpoint自动恢复",
            "隔离故障节点重新调度",
            "排查是硬件故障还是软件bug",
            "评估是否需要调整容错策略",
        ),
        trap="大多数人只会说'重启整个任务'，不知道成熟的系统应该能自动从checkpoint恢复"
        "并隔离坏节点，也不会去追问故障是偶发硬件问题还是会重复出现的系统性问题。",
        real_world_link="Minder faulty-machine detection / L4 log-based diagnosis",
    ),
    ScenarioPoint(
        id="sc-fr-eng-12",
        cat=CAT,
        trigger="你的模型被发现有偏见或歧视性输出，并且已经被媒体曝光了，你怎么应对？",
        rubric=(
            "第一时间评估影响范围并限流下线",
            "复现问题定位偏见来源",
            "对外沟通承认问题避免二次伤害",
            "建立长期的偏见检测与治理机制",
        ),
        trap="大多数人只会说'赶紧道歉发公关声明'，不会先去评估影响范围和限流止损，"
        "也不会真正花精力去定位偏见到底来自训练数据还是模型本身，更没想过要建立长期机制而不是一次性补丁。",
        real_world_link="Amazon 招聘AI性别偏见事件",
    ),
    ScenarioPoint(
        id="sc-fr-eng-13",
        cat=CAT,
        trigger="你的模型在某个benchmark上分数很高，但你怀疑这只是数据污染，怎么验证？",
        rubric=(
            "构造与benchmark同分布的全新题目对比",
            "检测模型是否能背出原题内容",
            "查看训练语料是否包含benchmark数据",
            "用动态更新的评测集交叉验证",
        ),
        trap="大多数人只会说'再多测几个benchmark'，不会想到构造一份全新的同分布题目做对照，"
        "也不知道可以直接测模型能不能背出原题内容来判断是记忆还是真实能力。",
        real_world_link="GSM8K vs GSM1K contamination研究",
    ),
    ScenarioPoint(
        id="sc-fr-eng-14",
        cat=CAT,
        trigger="你的RAG系统上线后幻觉反而变多了，你怎么排查？",
        rubric=(
            "检查检索召回的文档是否相关",
            "区分是检索问题还是生成阶段问题",
            "检查知识库是否存在过时或矛盾内容",
            "评估是否该用fine-tune而非RAG解决",
        ),
        trap="大多数人只会说'换个更大的模型'，不会先去分离到底是检索召回错了还是生成阶段编造的，"
        "也不会检查知识库本身是不是有过时或互相矛盾的文档。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-15",
        cat=CAT,
        trigger="用户投诉你的模型输出的内容给他造成了实际伤害（比如错误的医疗或法律建议），你怎么处理？",
        rubric=(
            "第一时间复现并评估伤害范围",
            "判断是否触发合规或法律责任流程",
            "针对该类高风险场景加限制或免责提示",
            "回溯该输出是否评测阶段本该被拦截",
        ),
        trap="大多数人只会说'给用户道歉退款'，不会去判断这是不是应该走合规或法律流程的严重事件，"
        "也不会回头检查这类高风险内容为什么没有在评测阶段被拦下来。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-16",
        cat=CAT,
        trigger="A/B实验整体指标是正向的，但你发现对某个高价值用户分群其实是负向的，怎么决策要不要上线？",
        rubric=(
            "先确认分群是提前定义还是事后现挖的",
            "评估该分群的业务权重和长期价值",
            "考虑分群定向的差异化策略",
            "警惕多重比较导致的伪发现",
        ),
        trap="大多数人只会说'整体正向就该上'，不会区分这个分群结果是提前假设好的还是事后从"
        "一堆分群里挑出来的，也不知道可以用差异化策略而不是非此即彼地做决策。",
        real_world_link="Spotify式分群反转案例",
    ),
    ScenarioPoint(
        id="sc-fr-eng-17",
        cat=CAT,
        trigger="老板要求你把一个还没充分测试的新功能立刻上线赶一个发布节点，你怎么办？",
        rubric=(
            "量化清楚现有测试覆盖的风险敞口",
            "提出灰度或功能开关的折中方案",
            "把风险决策权明确交还给业务负责人",
            "为快速回滚预留应急预案",
        ),
        trap="大多数人只会说'听老板的按时上线'或者'坚决不上线'两个极端，不会用具体的风险量化"
        "去争取折中方案，也不会想到用灰度和功能开关来降低这次赌注。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-18",
        cat=CAT,
        trigger="线上推理成本突然暴涨，你怎么排查和应对？",
        rubric=(
            "拆解是流量异常还是单次调用变贵",
            "排查是否有异常重试或死循环调用",
            "检查是否被恶意刷量或滥用攻击",
            "短期限流长期做成本监控告警",
        ),
        trap="大多数人只会说'换个便宜的模型'，不会先拆解到底是量涨了还是单价涨了，"
        "也不会去查是不是代码bug导致的重试风暴或者被恶意刷量。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-19",
        cat=CAT,
        trigger="模型升级后，下游好几个业务方都来投诉，但各自说的问题都不一样，你怎么判断是不是同一个根因？",
        rubric=(
            "收集各方badcase做统一归类",
            "对比升级前后模型行为的具体差异",
            "排查是否共享了同一个上游改动",
            "评估是否需要先回滚止损再定位",
        ),
        trap="大多数人只会说'一个个业务方单独排查'，不会先把所有badcase汇总归类找共性，"
        "也不会先考虑回滚止损再慢慢定位根因，而是在没回滚的情况下让问题持续影响。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-20",
        cat=CAT,
        trigger="另一个团队说复现不出你论文里报告的结果，你怎么处理？",
        rubric=(
            "主动提供完整代码和超参配置",
            "确认双方评测协议完全一致",
            "自己重新独立复现一遍排除偶然性",
            "诚实排查是否存在选择性汇报的最好结果",
        ),
        trap="大多数人只会说'可能是他们环境配置不对'，把责任先推给对方，不会主动去核对"
        "自己实验里有没有选择性汇报最好的一次结果，也不会自己重新独立跑一遍验证。",
        real_world_link="ML研究可复现性危机",
    ),
    ScenarioPoint(
        id="sc-fr-eng-21",
        cat=CAT,
        trigger="同样的输入，你的模型有时候给出不同的输出，用户抱怨不稳定，你怎么排查？",
        rubric=(
            "检查采样温度等解码参数设置",
            "排查是否服务端有负载均衡到不同版本",
            "评估该场景是否需要贴近确定性输出",
            "向用户说明生成式模型的固有随机性边界",
        ),
        trap="大多数人只会说'把temperature调成0就行'，不会去查是不是服务端负载均衡把请求"
        "路由到了不同的模型版本或者机器，也不会去想清楚这个场景到底需不需要完全确定性。",
        real_world_link="",
    ),
    ScenarioPoint(
        id="sc-fr-eng-22",
        cat=CAT,
        trigger="红队测试发现你的模型在某类场景下可以被越狱，绕过安全限制，你怎么应对？",
        rubric=(
            "评估越狱手法的传播风险和影响面",
            "短期打补丁长期做鲁棒性加固",
            "复盘该类越狱是否评测阶段本该覆盖",
            "建立持续红队机制而非一次性修复",
        ),
        trap="大多数人只会说'针对这个case加个过滤规则'，只做一次性打补丁，不会去评估这个"
        "越狱手法会不会被公开传播扩散，也不会建立持续的红队机制去覆盖同类变种。",
        real_world_link="",
    ),
]


def _self_test() -> None:
    assert 18 <= len(BANK) <= 24, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [sp.id for sp in BANK]
    assert len(ids) == len(set(ids)), "存在重复id"
    assert all(i.startswith("sc-fr-eng-") for i in ids), "id前缀不一致"
    assert all(3 <= len(sp.rubric) <= 5 for sp in BANK), "存在rubric要点数不在3-5之间的条目"
    assert all(sp.trap and sp.trigger for sp in BANK), "存在缺失trap/trigger的条目"
    for sp in BANK:
        full_answer = "。".join(sp.rubric)  # 拼接所有rubric要点作为"覆盖全部要点"的示例答案
        score = grade_scenario(sp, full_answer)
        assert score == 1.0, f"{sp.id} rubric关键词未能在拼接示例答案里全部命中: {score}"
    print(f"[PASS] sc_engineering_judgment: {len(BANK)}个ScenarioPoint 自洽性检查通过")


if __name__ == "__main__":
    _self_test()
