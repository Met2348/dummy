# papers/ — open-science-and-communication 参考源

本专题教**跨学科合作方法论、科学传播与公众沟通、开放科学实践(预注册/代码发布)、竞赛的组织与参与、学术社交媒体边界**这五件事(L1-L5), papers/ 收官方指南和公开工具类的真实参考, 不确定真实性的宁可留空不编造。

## 跨学科合作方法论 (L1)
- **National Academies of Sciences, Engineering, and Medicine, 《Facilitating Interdisciplinary Research》**(National Academies Press, 2004/2005, DOI: 10.17226/11153)——系统总结跨学科研究(IDR)的定义、驱动因素以及研究者/机构/资助方各自能采取的具体促进措施, 是理解"跨学科合作为什么天然更难"这一现象的权威参考, 本讲"三类落差"(默认假设/术语/评审标准)的拆解正是在这份报告揭示的一般性困境基础上做的具体细化。

## 科学传播与公众沟通 (L2)
- **Alan Alda, 《If I Understood You, Would I Have This Look on My Face?: My Adventures in the Art and Science of Relating and Communicating》**(Random House, 2017)——作者创立了Stony Brook大学Alan Alda科学传播中心(Alan Alda Center for Communicating Science), 书中系统讨论用即兴表演训练里的"共情"与"心智理论(theory of mind)"技巧帮助科学家真正站在听众角度组织语言, 是本讲科普写作技巧的方法论来源。
- 也见本仓库`research-presentation`(9.7)专题L1「知识的诅咒」——聚焦学术场合下的理解落差, 本讲聚焦完全没有专业背景的公众和媒体, 两讲互补不重复。

## 开放科学实践 (L3)
- **Center for Open Science(简称COS, cos.io)**运营的**OSF(Open Science Framework)**平台——目前学界最广泛使用的预注册公开时间戳存档平台, 官网提供预注册模板、已采纳Registered Report的期刊列表和相关资源说明。
- **ACM Artifact Review and Badging政策**(见acm.org/publications/policies/artifact-review-and-badging-current)——定义Artifacts Available / Artifacts Evaluated / Results Validated三种独立徽章, 是代码/数据公开规范如何被计算机领域会议/期刊制度化的具体样本。
- **Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T., "The Preregistration Revolution", *PNAS*, 2018**——系统论述预注册解决HARKing和p-hacking问题的机制, 是本讲预注册一节的核心方法论出处。
- Registered Report这一出版形式由Chris Chambers于2013年在期刊《Cortex》正式引入, 此后被多个期刊/会议采纳。

## 竞赛的组织与参与策略 (L4)
- **Roelofs, R., Fridovich-Keil, S., Miller, J., Shankar, V., Hardt, M., Recht, B., & Schmidt, L., "A Meta-Analysis of Overfitting in Machine Learning", *NeurIPS*, 2019**——系统分析大量真实Kaggle竞赛公开/私有测试集的名次差距, 发现"排行榜过拟合"的实际严重程度比民间直觉更微妙(多数shake-up更多来自方差而非系统性过拟合), 是本讲"公开/私有测试集划分"一节的核心方法论出处。
- **Narayanan, A. & Shmatikov, V., "Robust De-anonymization of Large Sparse Datasets", *IEEE Symposium on Security and Privacy*, 2008**——证明Netflix Prize"匿名化"竞赛数据集可被交叉比对重新识别真实用户身份, 是本讲数据隐私一节的具体出处, 也是数据发布领域被反复引用的经典教训。
- **NeurIPS官方Competition Track**的历年Call for Competitions说明——公开的竞赛提案评审流程和时间线要求, 是本讲组织流程一节的具体参照样本, 具体要求逐年可能调整, 请以当年官网发布版本为准。

## 学术社交媒体的边界与风险管理 (L5)
- **美国大学教授协会(American Association of University Professors, AAUP)**发布的关于学术自由与电子通讯/社交媒体的相关报告与声明——系统讨论教师/研究人员社交媒体言论如何与学术自由原则、机构利益产生张力, 见aaup.org, 具体条款请以官网当前发布版本为准。
- **2014年伊利诺伊大学厄巴纳-香槟分校(University of Illinois at Urbana-Champaign)Steven Salaita事件**——学者已接受教职邀约后, 因社交媒体上的政治言论被校方撤回聘用, 引发广泛的学术自由与社交媒体边界公开讨论, 是本讲反复引用的真实案例, 具体经过可查阅当年公开报道与后续学术自由组织声明。
- 也见本仓库`research-visibility-negotiation`(9.11)专题L1「个人科研品牌建设」——教的是主动正向的可见度经营内容策略, 本讲教的是防御性的边界与风险管理, 两者互补, 不重复彼此内容。

## 关于跨讲交叉引用的说明
L1引用`research-team-operations`L5(功能角色协作 vs 学科间协作)、L2引用`research-presentation`L1(学术场合 vs 公众场合的理解落差)、L3引用`experiment-ops-repro`(可复现性工程 vs 公开发布承诺)与`research-life`L3(AI工具披露)、L4引用`eval-foundations`(评测方法论技术原理 vs 竞赛组织形式)、L5引用`research-visibility-negotiation`L1与`team-leadership-for-researchers`, 都是同一原则: 这些专题已经讲过对应的**基础/相邻**内容, 本专题只聚焦"开放与传播"这个维度上此前完全空白的**公开、核验、边界管理**动作, 不重复它们已经讲过的内容。

## 为什么 papers/ 这么轻
本专题知识在**可跑工具**(`open_science_audit`)、L1-L5课件、以及上面的官方指南和公开资源里。真练习: 真的给自己现在(或未来)的一个研究项目/竞赛/社交媒体身份写一版五块骨架并自检, 而不是空想。
