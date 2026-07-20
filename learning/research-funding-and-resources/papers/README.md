# papers/ — research-funding-and-resources 参考源

本专题教**经费申请全流程、算力资源规划、数据管理规划(DMP)、多机构合作管理、供应商/API合规评估**这五件事(L1-L5), papers/ 收官方指南和公开工具类的真实参考, 不确定真实性的宁可留空不编造。

## 经费申请全流程 (L1)
- **NSF, *Proposal & Award Policies & Procedures Guide (PAPPG)***(NSF官方发布、定期更新的proposal撰写与执行全流程规范, 见 nsf.gov/policies/pappg): 涵盖budget、budget justification、current & pending support、rebudgeting阈值等本讲提到的行政环节, 是理解"钱怎么申请和管理"最权威的第一手材料。
- **NIH, SF424 (R&R) Application Guide**(NIH官方申请材料撰写指南): 详细规定budget page和budget justification的具体格式要求, 和PAPPG分属不同资助机构体系, 核心逻辑相通。

## 算力资源规划与申请 (L2)
- **ACCESS(access-ci.org)官方文档**——NSF资助的国家级先进计算资源协调生态(Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support), 2022年接替此前运行11年的XSEDE项目, 公开说明Explore/Discover/Accelerate/Maximize分级资源申请制度, 是"申请材料详细程度应匹配资源规模"这一原则的直接官方案例。
- 也见本仓库`training-orchestration`专题`01-slurm.md`——集群任务调度的技术操作(排队策略/backfill/预留窗口), 是本讲"资源怎么被申请到"之后"怎么被使用"的技术基础, L2不重复其内容。

## 数据管理规划 DMP (L3)
- **Wilkinson et al., "The FAIR Guiding Principles for scientific data management and stewardship"**(*Scientific Data* 3, 160018, 2016): FAIR(可发现/可访问/可互操作/可复用)四原则的原始出处, 本讲第1节直接引用。
- **DMPTool**(dmptool.org): 由加州大学Curation中心(University of California Curation Center)运营、多所大学共同参与的公开在线DMP撰写辅助工具, 整合NSF/NIH等主要资助机构对DMP的具体格式要求, 提供对应模板辅助生成DMP草稿。
- **NSF, "Preparing Your Data Management Plan"**(NSF官网关于DMP的官方说明页面, 2011年起强制要求)与**NIH Data Management and Sharing Policy**(要求提交DMSP, 2023年起生效): 两大资助机构名称不同、生效年份不同——NSF的是"Data Management Plan (DMP)", NIH的才是"Data Management and Sharing Plan (DMSP)", 不能混用。

## 大型多机构合作项目管理 (L4)
- **2 CFR Part 200("Uniform Guidance")**——美国联邦政府(由白宫管理与预算办公室OMB发布)关于统一行政要求、成本原则和联邦资助审计要求的官方规定, 其中对主承担机构监督子机构(subrecipient)经费使用的责任有明确条款, 是理解多机构经费监管制度背景的权威来源。
- 也见本仓库`research-life`专题`L3-authorship-ethics.md`——署名的基本判断原则(谁该署名/顺序惯例/贡献冲突预防)已在那一讲详细讲过, 本讲聚焦多机构场景下如何把这些原则落成书面协议, 不重复摘抄。

## 供应商/API/第三方合规评估 (L5)
- **GDPR(General Data Protection Regulation)第28条**——欧盟关于数据处理者(processor)必须签署书面数据处理协议(DPA)、明确处理范围与责任的官方法律条款, 是理解"为什么需要DPA"这一制度背景的权威来源。
- 主要模型/API提供商(如OpenAI、Anthropic等)公开发布的usage policy/terms of service——具体条款逐年更新, 请以厂商官网当年发布的最新版本为准, 本专题不代为总结具体条款内容以避免过时或误导。
- 也见本仓库`data-curation`专题`01-data-overview.md`——数据集本身的技术特性和常见许可类型, 是本讲"数据集许可证覆盖范围"判断的技术背景, L5聚焦"许可证是否覆盖实际用途"这一合规判断, 不重复数据技术细节。

## 关于跨讲交叉引用的说明
L2引用`training-orchestration/01-slurm.md`、L4引用`research-life/L3-authorship-ethics.md`、L5引用`data-curation/01-data-overview.md`, 都是同一原则: 这些专题已经讲过对应的**技术/伦理基础**, 本专题只聚焦"经费与资源"这个维度上此前完全空白的申请/规划/合规动作, 不重复它们已经讲过的内容。

## 为什么 papers/ 这么轻
本专题知识在**可跑工具**(`funding_plan_audit`)、L1-L5课件、以及上面的官方指南和公开工具里。真练习: 真的给自己现在(或未来)要申请的一份经费/资源写一版五块骨架并自检, 而不是空想。
