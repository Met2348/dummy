# 科研生涯与共同体扩展(Module 9.10-9.17 + 现有专题深化) Design

## 背景

老师在"科研技能"(Module 9,现10个专题、20环节地图)之外,进一步指出用户需要"科研全生命周期"更全面的知识和技能,要求补充50个新专题。用户确认扩展方向是"广度+深度都要",且5个候选分组(求职生涯/团队管理/经费合规/学术共同体/深化现有环节)优先级均衡、不做取舍。

Module 9现有10个专题(`research-knowledge-mgmt`/`literature-mapping`/`critical-reading-gap`/`experiment-design`/`experiment-ops-repro`/`research-figures`/`paper-writing-submission`/`research-presentation`/`research-life`/`research-direction-proposal`)全部服务于**单个研究项目**的生命周期(从选方向到发表答辩,20环节地图见`research-direction-proposal/lectures/L0-research-lifecycle-map.md`)。新增的50个专题分两类:

- **40个 = 跨项目、贯穿整个科研生涯反复出现的技能**(求职/带团队/经费/学术共同体参与)——这些不属于"单个项目"周期,属于更大尺度的"科研生涯"周期,现有20环节地图容不下,需要新的专题群。
- **10个 = 对现有20环节地图里3个专题的加深**(文献综述方法论/实验统计与ablation深水/论文发表策略深水)——直接以追加讲次的形式挂在对应现有专题下,不新建专题。

## 命名与编号(避让已有Module 10-13)

仓库里"Module 10/11/12/13"编号已被前沿扩展模块占用(`modules10-13-frontier`:多模态/VLA/可解释性/扩散,纯ML技术内容,和这里的科研软技能内容完全不同领域)。新内容不使用"Module 10"这个编号,而是延伸"Module 9 科研技能"家族本身的编号——正如`research-direction-proposal`(9.0)已经展示过M9编号可以非线性插入,这里进一步延伸为 **9.10 - 9.17**(8个新专题),沿用同一个"Module 9"家族标识,不新开一个Module号。

## 新增内容清单

### 8个全新专题(9.10-9.17,每个5讲=40讲)

| 编号 | 专题目录 | 一句话 | 5讲构成 |
|---|---|---|---|
| 9.10 | `career-pathways` | 学术界/工业界/博后路径选择与转换 | L1学术界求职全流程 / L2工业界研究岗求职全流程 / L3学界⇄业界转换策略 / L4博后阶段选择与规划 / L5教职早期生存 |
| 9.11 | `research-visibility-negotiation` | 个人科研品牌、推荐信、谈判 | L1科研品牌建设 / L2推荐信策略 / L3面试research/job talk设计 / L4谈判技巧 / L5 CV版本化管理 |
| 9.12 | `team-leadership-for-researchers` | 带教与向下管理 | L1指导低年级学生/实习生 / L2向下管理与建设性反馈 / L3高效组会/brainstorm主持 / L4团队协作冲突处理 / L5团队健康度诊断 |
| 9.13 | `research-team-operations` | 团队运营实务 | L1多项目并行时间管理 / L2招募筛选合作者 / L3远程/异步跨时区协作 / L4团队知识传承与onboarding / L5跨专业背景团队协作 |
| 9.14 | `research-funding-and-resources` | 经费与资源规划 | L1经费申请全流程 / L2算力资源规划与申请 / L3数据管理规划(DMP) / L4大型多机构合作项目管理 / L5供应商/API/第三方合规评估 |
| 9.15 | `research-integrity-and-compliance` | 科研诚信与合规深水 | L1学术诚信深水(不端调查/authorship仲裁) / L2 IRB/伦理审查全流程 / L3知识产权与成果转化 / L4国际合作合规 / L5安全与负责任披露 |
| 9.16 | `academic-community-engagement` | 学术共同体参与 | L1审稿之外的学术服务(PC/AC) / L2组织workshop/研讨会 / L3学术会议社交与人脉建设 / L4建立长期合作关系 / L5跨机构评审网络建设 |
| 9.17 | `open-science-and-communication` | 开放科学与科学传播 | L1跨学科合作方法论 / L2科学传播与公众沟通 / L3开放科学实践(预注册/开源发布规范) / L4竞赛/challenge组织参与策略 / L5学术社交媒体边界与风险 |

### 3个现有专题追加讲次(共10讲)

| 专题 | 追加讲次 | 内容 |
|---|---|---|
| `literature-mapping` | +L5 | 系统性文献综述/meta-analysis方法论 |
| `experiment-design` | +L6, +L7, +L8 | 贝叶斯vs频率派统计方法选择 / 大规模实验算力预算规划 / Ablation设计系统化方法论深水 |
| `paper-writing-submission` | +L5...+L10(6讲) | 负结果处理与发表策略 / 多论文组合发表策略 / 会议vs期刊选择+dual submission规则 / Camera-ready与artifact evaluation / 论文长期影响力经营 / 非英语母语者学术写作策略 |

## 房屋风格(严格复用`research-direction-proposal`已验证的约定,不自创新格式)

每个新专题(9.10-9.17)目录结构:

```
learning/<topic>/
  README.md                  — 沿用9.0的README结构:一句话+"这个专题要解决的真问题"对比图+学习路径表+notebook表+模板列表+工具列表+环境+checklist+"在Module 9中的位置"
  lectures/L1-L5.md           — 每讲: `# Ln · 中文标题 (English)` + `> XX-min lecture · 目标: ...` + `## 0.`起编号章节 + ASCII图表 + 具体例子(优先引用用户仓库里已有的真实专题,如`interview-prep`/`harness-engineering`等) + "本讲小结+通往下一讲" + "**动手**:"
  templates/*.md              — 每讲至少1个可复用模板(打分表/checklist/清单类)
  src/*.py                    — 1-2个纯stdlib工具,遵循`blank_*()`/`audit()`/`render()`模式,每个工具at least一个"好例子vs敷衍例子"对比demo
  notebooks/_gen_notebooks.py + N1/N2.ipynb — 用nbformat生成,jupyter nbconvert --execute验证跑通
  papers/README.md            — 真实引用(经典科研生涯/管理类著作+官方指南,不编造)
  environment/requirements.txt + verify_env.py
```

现有专题追加讲次:只新增`lectures/L{n}.md`文件 + 更新该专题`README.md`的学习路径表和(若有)notebook/模板引用,不改变该专题已有的L1-L4/L5内容。

## 内容标准

- 讲义文风:严格复刻`critical-reading-gap` L4 / `research-knowledge-mgmt` L3 / `research-direction-proposal` L1-L4已验证的文风(不是CS-mastery的教科书体),每讲实质内容(不含代码块)不少于4000字符。
- 真实引用:每个新专题的`papers/README.md`必须是真实存在的书籍/官方指南(如经费申请引用NSF/NIH官网流程说明、学术诚信引用COPE指南、职业规划引用《The Chicago Guide to Your Career in Science》《A PhD Is Not Enough》等已在仓库出现过的经典),不确定真实性的宁可留空不编造。
- 工具:纯stdlib,可独立`python src/xxx.py`跑demo,输出好例子vs敷衍例子的对比。
- 去重:所有新讲次标题/`src`文件名与仓库现有全部专题(含`_shared`)不冲突,执行前用一次全局`grep`确认。

## 执行顺序

1. `career-pathways`(9.10)→ `research-visibility-negotiation`(9.11)→ `team-leadership-for-researchers`(9.12)→ `research-team-operations`(9.13)→ `research-funding-and-resources`(9.14)→ `research-integrity-and-compliance`(9.15)→ `academic-community-engagement`(9.16)→ `open-science-and-communication`(9.17)
2. 现有专题追加十讲:`literature-mapping` → `experiment-design` → `paper-writing-submission`
3. 全部完成后:更新`research-direction-proposal`的L0地图文档,增补一节"科研生涯周期(9.10-9.17)与项目周期(20环节)的关系"说明两套体系并列而非从属;更新顶层specs索引;写memory。

## 验证

- 每个新专题:`python environment/verify_env.py` 全部通过;`jupyter nbconvert --to notebook --execute --inplace` 对2个notebook成功;`src/`工具独立运行demo成功。
- 讲义字符数抽查(每讲≥4000字符实质内容)。
- 全局标题/文件名去重grep,零冲突。
- 现有3个专题追加讲次后,原L1-L4/L5内容未被修改(git diff只包含新增,不含删除/修改已有行,除README表格追加行外)。

## 不做的事(YAGNI)

- 不强行把9.10-9.17塞进20环节地图的编号序列——它们是并列的"生涯周期",不是"项目周期"的第21-60环节。
- 不为8个新专题重新设计房屋风格或引入新的数据结构(如DeepPoint)——完全复用已验证的lecture-style。
- 不修改`docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`原文档,只新增本文档 + 未来在L0地图里加一节说明。
