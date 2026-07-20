# papers/ — research-integrity-and-compliance 参考源

本专题教**学术不端调查/authorship仲裁、IRB/伦理审查、知识产权与成果转化、国际合作合规、安全与负责任披露**这五件事(L1-L5), papers/ 收官方指南和公开工具类的真实参考, 不确定真实性的宁可留空不编造。

## 学术诚信深水: 不端调查流程/authorship仲裁/造假识别 (L1)
- **美国研究诚信办公室(Office of Research Integrity, ORI)**——隶属美国卫生与公众服务部, 公开发布对公共卫生服务基金资助研究项目的不端案例调查结论摘要(case summaries), 是理解"举报→初步排查→正式调查→裁决"这套正式程序如何在真实案例中运作的第一手公开材料, 见ori.hhs.gov。
- **42 CFR Part 93**——美国联邦法规中定义"研究不端行为"(伪造/篡改/剽窃)及其调查处理程序的具体条款, 是ORI处理流程的法律依据。
- **COPE(Committee on Publication Ethics, 出版伦理委员会)**——面向期刊编辑/作者公开发布一系列处理署名纠纷、疑似造假等场景的流程图(flowcharts), 是理解期刊层面如何仲裁authorship纠纷的权威参考, 见publicationethics.org。
- **Bik EM, Casadevall A, Fang FC, "The Prevalence of Inappropriate Image Duplication in Biomedical Research Publications", *mBio*, 2016**——大规模系统排查生物医学论文图像重复/拼接问题的代表性研究, 是本讲"图像取证"一节的具体方法论出处。
- 也见本仓库`research-life`专题`L3-authorship-ethics.md`——署名规则与伦理红线(怎么不踩红线)已在那一讲详细讲过, 本讲聚焦红线被踩之后走什么正式调查/仲裁程序, 不重复其内容。

## IRB/伦理审查全流程 (L2)
- **《贝尔蒙报告》(The Belmont Report, 美国生物医学与行为研究人类受试者保护国家委员会, 1979)**——确立尊重人/有利/公正三条原则, 是几乎所有IRB审查依据的理论基础。
- **美国联邦《通用规则》(Common Rule, 45 CFR 46)**——多数美国联邦资助机构共同遵守的人类受试者保护规定, 对脆弱群体设有专门附则。
- **世界医学协会《赫尔辛基宣言》(Declaration of Helsinki)**——国际通行的医学研究伦理原则声明, 知情同意与弱势群体保护精神被广泛援引到其他涉及人类被试的研究领域。
- 各校IRB官网通常公开本机构的审查类别判定流程、申请表格和审查周期说明——具体流程因机构而异, 请以本人所在机构IRB官网当年发布的版本为准, 本专题不代为总结某一具体学校的流程以避免过时或误导。

## 知识产权与成果转化 (L3)
- **《拜杜法案》(Bayh-Dole Act, 1980, 美国)**——首次明确允许大学保留联邦资助研究所产生的专利所有权, 是现代大学技术转移办公室(TTO)建制的立法基础。
- **AUTM(Association of University Technology Managers, 大学技术经理人协会)**——发布年度许可调查报告(licensing survey), 是了解大学技术转移行业整体规模和惯例的公开参考来源。
- 也见本仓库`research-funding-and-resources`(9.14)专题——该专题L1讲经费申请全流程、L5讲供应商/API合规, 关注的是"研究投入端"的资源规划; 本讲关注的是"研究产出端"的知识产权保护与商业化, 两者互补, 不重复彼此内容。

## 国际合作合规 (L4)
- **《出口管理条例》(Export Administration Regulations, EAR)**——美国商务部工业与安全局(Bureau of Industry and Security, BIS)执行, 管辖民用及军民两用技术的出口管制。
- **《国际武器交易条例》(International Traffic in Arms Regulations, ITAR)**——美国国务院国防贸易管制局(Directorate of Defense Trade Controls, DDTC)执行, 管辖国防相关物项和技术。
- **欧盟《通用数据保护条例》(GDPR)第五章(第44-50条)**——个人数据向欧盟以外地区传输的法定机制(充分性认定/标准合同条款等)。
- **中国《个人信息保护法》(PIPL, 2021年施行)**——对跨境提供重要数据/大规模个人信息设有安全评估要求。
- 也见本仓库`research-funding-and-resources`专题`L5-vendor-and-api-compliance.md`——该讲聚焦你与具体供应商之间的合同性合规(数据处理协议DPA), 本讲聚焦国家层面强制施加的合规义务(出口管制/跨境数据传输法律), 两者常同时适用于同一国际合作项目但解决不同问题, 不重复摘抄。

## 安全与负责任披露 (L5)
- **ISO/IEC 29147(信息技术安全技术——漏洞披露)**——定义漏洞披露从接收报告到协调发布的角色和流程规范的国际标准。
- **CERT/CC(卡内基梅隆大学软件工程研究所计算机应急响应中心)**——长期运营被广泛参照的漏洞披露政策实践。
- **Google Project Zero公开的90天默认披露窗口政策**——业界广为人知的协调披露时间线具体范例。
- 也见本仓库`red-team-jailbreak`专题——该专题教攻击技术本身(GCG/PAIR/AutoDAN/Crescendo等具体越狱手法的机制), 并明确声明所有代码为教学mock、不输出真实有效的越狱prompt; 本讲聚焦"发现真实漏洞之后该走什么披露流程", 不重复其攻击技术内容, 而是解释那份"教学mock而非真实攻击代码"的克制选择背后遵循的正是本讲讲的负责任披露原则。

## 关于跨讲交叉引用的说明
L1引用`research-life/L3-authorship-ethics.md`、L3引用`research-funding-and-resources`、L4引用`research-funding-and-resources/L5-vendor-and-api-compliance.md`、L5引用`red-team-jailbreak`, 都是同一原则: 这些专题已经讲过对应的**基础/预防性**内容, 本专题只聚焦"诚信与合规"这个维度上此前完全空白的**处理/申请/核查**动作, 不重复它们已经讲过的内容。

## 为什么 papers/ 这么轻
本专题知识在**可跑工具**(`compliance_checklist`)、L1-L5课件、以及上面的官方指南和公开资源里。真练习: 真的给自己现在(或未来)的一个研究项目写一版五块骨架并自检, 而不是空想。
