# 9.17 open-science-and-communication — 开放科学与科学传播 (跨学科合作 → 公众沟通 → 预注册/开源发布 → 竞赛组织与参与 → 社交媒体边界)

> **Module 9「科研技能」第18个专题 · 阶段: 科研生涯周期(继`academic-community-engagement` 9.16之后, 第8块拼图, 也是9.10-9.17扩展的收官专题)**
> `research-life`(9.9)的L1-L2讲的是审稿和导师沟通这类日常软技能, `academic-community-engagement`(9.16)讲的是要不要接PC/AC、怎么组织workshop、怎么建立长期合作这类共同体参与动作。但还有一整块围绕"公开"这件事本身的能力此前完全空白: 跟另一个学科的合作者怎么对齐语言、跟完全没有专业背景的公众和媒体怎么讲清楚研究结论而不失真、怎么把"开放科学"从一句口号变成预注册和代码发布的具体承诺、一场有排行榜的竞赛该怎么组织和参与才经得起核验、你在社交媒体上的发言到底代表个人还是代表机构。本专题补上这块"公开"维度上的完整拼图。

---

## 这个专题要解决的真问题

大多数博士生/年轻研究者对"开放"和"传播"的认知停留在"论文发出来、代码放上GitHub"这句听起来正确但远远不够用的口号上——从没有人系统教过"跟另一个学科的合作者怎么发现彼此其实在说不同的话""跟记者/公众讲研究结论怎么避免被过度简化成一句耸动标题""预注册到底解决了什么问题、和'代码以后会整理'有什么本质区别""办一场竞赛/参加一场竞赛怎么避免被排行榜数字带偏""社交媒体上一句'观点仅代表个人'到底能保护你到什么程度"。

```
                没有本专题                              有本专题
跨学科合作      以为"大家都是搞科研的, 术语应该都懂"       意识到默认假设/术语/评审标准三类落差, 用双向
                                                        术语对照表持续对齐
公众沟通        随便讲讲, 讲不清是听众理解力的问题         用倒金字塔+术语首解释+"所以呢"自测, 主动为
                                                        媒体断章取义做准备
开放科学        "代码以后会整理好再公开"                   知道预注册把假设时间戳钉在实验前, 发表时同步
                                                        公开才是承诺, 而非"以后"
竞赛组织/参与    只看公开排行榜名次决定模型好不好           懂得公开/私有测试集划分、代码复现验证、内部
                                                        交叉验证优先于死盯公开榜
社交媒体        以为写了"观点仅代表个人"就万事大吉          知道免责声明降低风险不能消除风险, 分清专业
                                                        判断与个人立场, 假设一切都可能被永久截图
      ↓                                                    ↓
"公开"全靠临场随便讲、随手一发的直觉                     每个"公开"场景都有可核查、可复用的具体方法,
                                                        不再事后才追悔莫及
```

---

## 学习路径 (5 讲)

| 讲 | 文件 | 一句话 | 产出物 |
|---|---|---|---|
| L1 | `lectures/L1-interdisciplinary-collaboration.md` | 跨学科合作方法论: 默认假设/术语/评审标准三类落差如何识别, 用双向术语对照表持续对齐, 区别于`research-team-operations`L5聚焦"同一团队内不同职能角色"协作 | 跨学科术语对照表草稿 |
| L2 | `lectures/L2-science-communication.md` | 科学传播与公众沟通: 过度简化导致的失真/媒体采访陷阱/类比的双刃剑, 尤重"经得起被单独截取而不失真"的措辞习惯 | 3句核心信息点草稿 |
| L3 | `lectures/L3-open-science-practices.md` | 开放科学实践: 预注册与HARKing、Registered Report如何从制度上消除发表偏见、代码/数据发布规范(ACM Artifact Badging), 区别于`experiment-ops-repro`聚焦"可复现性工程"而非"公开承诺" | 预注册文本+代码发布时间表草稿 |
| L4 | `lectures/L4-organizing-competitions.md` | 竞赛/Challenge的组织与参与策略: 公开/私有测试集划分、Netflix Prize数据隐私教训、NeurIPS Competition Track流程、参赛者的内部交叉验证策略 | 竞赛规则/参赛自查清单草稿 |
| L5 | `lectures/L5-academic-social-media-boundaries.md` | 学术社交媒体的边界与风险管理: 免责声明的实际效力边界、专业判断与个人立场的区分、匿名账号的薄弱保护, 区别于`research-visibility-negotiation`L1聚焦"正向可见度经营" | 社交媒体发言自查草稿 |

> 读法: L1→L5 顺序; L1-L2讲和"人"(合作者/公众)对齐语言, L3-L4讲和"共同体"(读者/竞赛)之间可核验的公开承诺, L5讲个人社交媒体这个最日常也最容易被忽视边界的公开场域。每讲读完立刻去 `templates/open-science-checklist.md` 对应小节上手一次。

## 动手 (1 个 notebook — 真实科研动作)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-audit-open-science.ipynb` | 用 `src/open_science_audit.py` 给你现在(或未来可能)真实要处理的一个研究项目/竞赛/社交媒体身份写一版五块骨架, 自动检查漏项 |

## 可复用模板 (`templates/`)

- `open-science-checklist.md` — 开放科学实践五块骨架模板(配L1-L5, 对应 `SECTIONS` 的五个key)

## 工具 (`src/`)

- `open_science_audit.py` — 开放科学实践骨架自检(五块: 跨学科术语对照表/公众沟通材料/预注册计划/代码数据发布规范/学术社交媒体边界, 纯stdlib)

---

## 环境

```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / Windows native 即可, 无需 WSL2。

## 完成本专题后你应该能 (产出 checklist)

- [ ] 识别一次跨学科合作里的默认假设/术语/评审标准三类落差, 用双向术语对照表持续对齐, 而不是假设对方"应该听得懂"
- [ ] 面向公众/媒体讲清楚一个研究结论, 提炼出经得起被单独截取而不失真的核心信息点, 避免拟人化语言造成的过度简化
- [ ] 说清预注册和Registered Report分别怎么防止HARKing和发表偏见, 以及代码/数据发布规范为什么"发表时同步公开"而非"以后整理"
- [ ] 判断一场竞赛该怎么设计公开/私有测试集划分和防作弊规则, 参赛时优先信任内部交叉验证而非死盯公开排行榜
- [ ] 说清"观点仅代表个人"这句免责声明的实际效力边界, 分清专业判断与个人立场, 理解匿名账号的薄弱保护
- [ ] 用 `open_science_audit.py` 给自己的项目/竞赛/社交媒体身份做一次完整性自检

---

## 在 Module 9 中的位置

```
Module 9 科研技能 (现18个专题, 两套并列体系)

体系① 20环节项目周期 (单个研究项目从选方向到发表答辩的生命周期, 见 research-direction-proposal/L0)
  起点   9.0 research-direction-proposal L1-L2 (方向选择+可行性)
  地基   9.1 research-knowledge-mgmt
  输入   9.2 literature-mapping / 9.3 critical-reading-gap
  立项   9.0 research-direction-proposal L3-L4 (开题写作+答辩)
  执行   9.4 experiment-design / 9.5 experiment-ops-repro
  输出   9.6 research-figures / 9.7 paper-writing-submission / 9.8 research-presentation

体系② 科研生涯周期 (跨越多个项目、以年为尺度的个人发展轨迹, 20环节地图容不下, 新分支)
  科研生活 9.9 research-life (审稿/导师沟通/署名伦理/可持续 —— 贯穿全程的通用软技能)
  职业路径 9.10 career-pathways (求职/转换/博后/教职早期 —— 关键节点的路径选择)
  品牌谈判 9.11 research-visibility-negotiation (可见度/推荐信/job talk/谈判/CV管理)
  带教管理 9.12 team-leadership-for-researchers (指导新手/向下反馈/组会主持/冲突处理/团队健康诊断)
  运营实务 9.13 research-team-operations (时间管理/招募筛选/异步协作/onboarding/跨专业协作)
  经费资源 9.14 research-funding-and-resources (经费申请/算力规划/数据管理/多机构合作/供应商合规)
  诚信合规 9.15 research-integrity-and-compliance (不端调查/authorship仲裁/IRB伦理/IP成果转化/国际合作合规/负责任披露)
  共同体参与 9.16 academic-community-engagement (审稿之外的学术服务/组织workshop/会议社交/长期合作/跨机构评审网络)
  开放传播 9.17 open-science-and-communication (跨学科合作/公众沟通/预注册与开源发布/竞赛组织参与/社交媒体边界)  ◄── 你在这里(新增, 9.10-9.17扩展收官)
```
> 9.17和9.16的区别: 9.16教的是"怎么主动参与、经营共同体内部的角色和关系"(该不该接PC/AC、怎么组织活动、怎么维持合作), 关注对象是共同体内部的人际网络; 9.17教的是"怎么把研究本身、研究过程、以及你自己, 以经得起核验的方式向外公开"(跨学科合作者/公众媒体/预注册核验/竞赛排行榜/社交媒体陌生读者), 关注对象是不同层次的"外部"公开场域, 两者互补但不重复。至此9.10-9.17科研生涯与共同体扩展的8个专题全部建成, 与9.9-9.15共同构成体系②完整的"科研生涯周期"覆盖。
>
> 设计文档: `docs/superpowers/specs/2026-06-17-research-skills-module9-design.md`(原9模块设计文档)、`docs/superpowers/specs/2026-07-14-research-career-lifecycle-design.md`(9.10-9.17科研生涯与共同体扩展设计文档, 本专题为该扩展的第8个、也是最后一个专题)。
