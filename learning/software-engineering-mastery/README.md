# Software Engineering Mastery —— 软件工程专题(浅→深→社招级别)

> 定位:老手指出用户 EE 转 NLP/LLM 背景,缺少科班 CS 学生默认已掌握的基础知识——软件工程、设计模式、数据库、网络、操作系统——这些也会被问到,建议每个专题各做100+知识点系统化学习。这是五个 CS 基础专题队列的第一个。

## 与 FM/FL/Diffusion 三个专题的关键差异:新增 `explain` 字段

`foundation-model-mastery`/`federated-learning-mastery`/`diffusion-mastery` 三个专题用户本身已有研究生级基础,只需要把知识整理成"面试追问链"。这五个 CS 基础专题用户是**完全没有系统学过**,所以特别要求"讲仔细"。为此 `DeepPoint`/`ScenarioPoint` 在原有字段基础上新增了 `explain: str` 字段——每个知识点先给一段≥100字的系统性讲解(是什么/为什么/怎么用/常见误区),再接原有的三层追问链(`DeepPoint.chain`)或场景判断要点(`ScenarioPoint.rubric`),兼顾"系统学会"和"面试接得住"两个目标。

## 组织轴:难度分层(与 FM/FL/Diffusion 一致)

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | SDLC与敏捷/Scrum/Kanban基础认知框架、版本控制(git分支模型)与CI/CD协作流程基础 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 测试方法论深水、架构与模块化深水、代码质量与重构深水 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 技术债务与工程文化判断、交付与发布策略判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**,练法建议按 Tier 1→2→3 顺序推进——先读 `explain` 建立认知,再练 `chain`/`rubric` 检验是否真的能被追问住。

## 与仓库已有同名/近名目录的边界

仓库里已有 `agent-design-patterns/`(LLM agent 编排模式:prompt chaining/routing/orchestrator-workers)和 `cluster-networking/`(GPU集群NCCL/allreduce/fat-tree),两者分别是"agent工程模式"和"分布式训练网络",和本队列后续要做的"GoF经典设计模式"、"通用计算机网络基础"是完全不同的细分领域,没有内容重叠。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_se_sdlc_agile_requirements.py` | 瀑布模型、敏捷宣言、Scrum三角色与仪式、Kanban、用户故事与INVEST、故事点估算、DoD/DoR、需求获取方法 | 18 |
| `tier1_shallow/dp_se_version_control_cicd_basics.py` | git commit快照模型、分支本质、merge vs rebase、code review目的、CI/CD核心思想、Git Flow vs Trunk-Based、Feature Flag | 17 |
| `tier2_deep/dp_se_testing_methodology.py` | 测试金字塔、TDD/BDD、mock-stub-fake-spy区别、覆盖率陷阱、契约测试、变异测试 | 15 |
| `tier2_deep/dp_se_architecture_modularity.py` | 分层架构、六边形/Clean Architecture、SOLID深水、耦合内聚、DI容器、微服务vs单体 | 15 |
| `tier2_deep/dp_se_code_quality_refactoring.py` | code smell、重构手法、DRY真正含义、linter能力边界、技术债务利息、评审有效性研究、圈复杂度 | 15 |
| `tier3_social_hire/sc_se_tech_debt_engineering_culture_judgment.py` | 重写vs渐进重构、技术债务优先级、团队规模流程调整、测试文化推动、遗留系统接手 | 14 |
| `tier3_social_hire/sc_se_delivery_release_judgment.py` | 灰度vs蓝绿vs滚动更新、DORA指标误用、Blameless Postmortem、跨团队依赖协调、数据库迁移顺序 | 14 |

## 新增:本科课件式完整教材

老手进一步反馈:即便加了 `explain` 字段,108个知识点的追问链结构本质上仍然是"直接上面试",对完全没系统学过的人不够。因此新增 `lectures/01-textbook.md`——一份约24万字符、七章的完整教材式讲义,按 SDLC与需求工程→版本控制与CI/CD→测试方法论→架构与模块化→代码质量与重构→技术债务与工程文化判断→交付与发布策略判断的顺序,用连贯的教科书叙述(引言/核心概念展开/场景走查/常见误区/小结)系统教会每个主题,不是知识点罗列或追问链。**建议学习顺序:先读完 `01-textbook.md` 建立系统认知,再回到下面的 DeepPoint/ScenarioPoint 用追问链和场景判断检验"是否真的能被追问住"。**

## 数据结构:DeepPoint + ScenarioPoint(新增 explain 字段)

```python
import sys
sys.path.insert(0, "learning/software-engineering-mastery/src")
from software_engineering_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/software-engineering-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典CS教材/一线工程共识(GoF设计模式的应用场景、《人月神话》/敏捷宣言原文精神、Martin Fowler重构手法、Google工程实践研究关于代码评审规模效应的结论、DORA指标定义等),不需要像FM/FL/Diffusion那样追赶前沿论文,但仍要求核实具体术语准确性。`real_world_link` 字段全部留空——这批内容没有可验证的本地文件或需要引用的具体外部产品案例,不编造路径或案例。

至此,老手要求的五个CS基础专题队列(软件工程/设计模式/数据库/网络/OS)第一个完成,后续四个陆续推进。

## 补充:直接问答自测

`qa-practice.md`——一份轻量级补充练习(24题,不含追问链评分逻辑),每题一个问题配一个完整参考答案,和上面108点的3层追问链形式不同,提问角度也刻意避开了源码里的trigger原文,适合刷完追问链之后做最后一遍考前快速过关。
